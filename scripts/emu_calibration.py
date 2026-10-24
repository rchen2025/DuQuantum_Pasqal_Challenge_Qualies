#!/usr/bin/env python
"""Time the emulators on AnalogDevice registers so we know what we can afford.

    python scripts/emu_calibration.py                 # default grid, writes docs/EMU_CALIBRATION.md
    python scripts/emu_calibration.py --atoms 12 20 --bond-dims 16 32 --shots 200 --duration 2000

For each atom count we build a generic register (points on a triangular-ish
grid at 7 µm spacing, centred, radius <= 38 µm) and a baseline adiabatic pulse
(three-point Omega ramp to 0.8*Omega_max, linear delta sweep from -2*Omega_max to
+Omega_max over `duration` ns), then time:

  * pulser-simulation (QutipBackendV2)   only for the smallest sizes (<= 12 atoms)
  * emu-sv                              up to ~20 atoms
  * emu-mps                             every size, for each bond dimension

and record wall time, peak max bond dimension reached (emu-mps statistics),
and the total-variation distance between the emu-mps bitstring histograms at
successive bond dimensions (a cheap convergence indicator; also emu-sv vs
emu-mps when both ran).  Results go to docs/EMU_CALIBRATION.md as a table.
The same register+pulse are used for every backend so the numbers compare.

Nothing in here is a result about the CVRP model: it is timing only.
"""
from __future__ import annotations

import argparse
import math
import platform
import time
from pathlib import Path

import numpy as np


def make_register(n: int, spacing: float = 7.0):
    """Roughly square grid of n atoms at `spacing` µm, centred at the origin."""
    from pulser import Register
    side = math.ceil(math.sqrt(n))
    pts = []
    for r in range(side):
        for c in range(side):
            if len(pts) < n:
                # offset alternate rows (triangular-ish) to keep neighbours at 'spacing'
                pts.append((c * spacing + (0.5 * spacing if r % 2 else 0.0), r * spacing * math.sqrt(3) / 2))
    pts = np.array(pts) - np.mean(pts, axis=0)
    rad = np.sqrt((pts ** 2).sum(1)).max()
    return Register.from_coordinates(pts, prefix="q", center=True), float(rad)


def make_sequence(reg, device, duration: int, omega_scale: float = 0.8):
    from pulser import Pulse, Sequence
    from pulser.waveforms import InterpolatedWaveform
    ch = device.channels["rydberg_global"]
    om = omega_scale * ch.max_amp
    seq = Sequence(reg, device)
    seq.declare_channel("ryd", "rydberg_global")
    amp = InterpolatedWaveform(duration, [0.0, om, om, 0.0], times=[0.0, 0.15, 0.85, 1.0])
    det = InterpolatedWaveform(duration, [-2 * om, -2 * om, om, om], times=[0.0, 0.15, 0.85, 1.0])
    seq.add(Pulse(amp, det, 0.0), "ryd")
    return seq


def tv_distance(a: dict, b: dict) -> float:
    na, nb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def run_qutip(seq, shots: int):
    from pulser.backend import BitStrings
    from pulser_simulation import QutipBackendV2, QutipConfig
    cfg = QutipConfig(sampling_rate=0.1, observables=[BitStrings(evaluation_times=[1.0], num_shots=shots)])
    res = QutipBackendV2(seq, config=cfg).run()
    return dict(res.get_result("bitstrings", 1.0))


def run_sv(seq, shots: int, dt: float = 10.0):
    from emu_sv import BitStrings, SVBackend, SVConfig
    cfg = SVConfig(dt=dt, log_level=40, observables=[BitStrings(evaluation_times=[1.0], num_shots=shots)])
    res = SVBackend(seq, config=cfg).run()
    return dict(res.get_result("bitstrings", 1.0))


def run_mps(seq, shots: int, bond_dim: int, dt: float = 10.0, precision: float = 1e-5):
    from emu_mps import BitStrings, MPSBackend, MPSConfig
    cfg = MPSConfig(dt=dt, precision=precision, max_bond_dim=bond_dim, log_level=40,
                    observables=[BitStrings(evaluation_times=[1.0], num_shots=shots)])
    res = MPSBackend(seq, config=cfg).run()
    counts = dict(res.get_result("bitstrings", 1.0))
    stats = getattr(res, "statistics", None)
    peak = None
    try:
        if stats:
            last = stats[-1] if isinstance(stats, (list, tuple)) else stats
            peak = last.get("max_bond_dimension") if hasattr(last, "get") else getattr(last, "max_bond_dimension", None)
    except Exception:  # noqa: BLE001
        peak = None
    return counts, peak


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, nargs="+", default=[12, 20, 30, 40])
    ap.add_argument("--bond-dims", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--duration", type=int, default=2000)
    ap.add_argument("--qutip-max", type=int, default=12)
    ap.add_argument("--sv-max", type=int, default=20)
    ap.add_argument("--out", default="docs/EMU_CALIBRATION.md")
    a = ap.parse_args(argv)

    import pulser
    from pulser.devices import AnalogDevice
    try:
        import torch
        torch_v = torch.__version__
        threads = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        torch_v, threads = "n/a", "n/a"

    rows = []
    for n in a.atoms:
        reg, rad = make_register(n)
        seq = make_sequence(reg, AnalogDevice, a.duration)
        print(f"\n== {n} atoms, radius {rad:.1f} µm, duration {a.duration} ns, shots {a.shots}")
        hist = {}
        if n <= a.qutip_max:
            t0 = time.perf_counter()
            try:
                hist["qutip"] = run_qutip(seq, a.shots)
                rows.append((n, "pulser-simulation (qutip)", "-", time.perf_counter() - t0, "-", "-"))
                print(f"   qutip   {rows[-1][3]:.1f} s")
            except Exception as e:  # noqa: BLE001
                rows.append((n, "pulser-simulation (qutip)", "-", float("nan"), "-", f"ERROR {type(e).__name__}: {e}"[:80]))
                print("   qutip   ERROR", e)
        if n <= a.sv_max:
            t0 = time.perf_counter()
            try:
                hist["sv"] = run_sv(seq, a.shots)
                rows.append((n, "emu-sv", "-", time.perf_counter() - t0, "-", "-"))
                print(f"   emu-sv  {rows[-1][3]:.1f} s")
            except Exception as e:  # noqa: BLE001
                rows.append((n, "emu-sv", "-", float("nan"), "-", f"ERROR {type(e).__name__}: {e}"[:80]))
                print("   emu-sv  ERROR", e)
        prev = None
        for chi in a.bond_dims:
            t0 = time.perf_counter()
            try:
                counts, peak = run_mps(seq, a.shots, chi)
                dt_s = time.perf_counter() - t0
                note = []
                if prev is not None:
                    note.append(f"TV vs chi={prev[0]}: {tv_distance(prev[1], counts):.3f}")
                if "sv" in hist:
                    note.append(f"TV vs emu-sv: {tv_distance(hist['sv'], counts):.3f}")
                rows.append((n, "emu-mps", chi, dt_s, peak if peak is not None else "?", "; ".join(note) or "-"))
                print(f"   emu-mps chi={chi:<3} {dt_s:.1f} s  peak bond dim {peak}  {'; '.join(note)}")
                prev = (chi, counts)
            except Exception as e:  # noqa: BLE001
                rows.append((n, "emu-mps", chi, float("nan"), "-", f"ERROR {type(e).__name__}: {e}"[:80]))
                print(f"   emu-mps chi={chi} ERROR", e)

    lines = [
        "# Emulator timing calibration",
        "",
        f"Machine: {platform.platform()}, {platform.processor() or 'cpu'}; python {platform.python_version()}, "
        f"pulser {pulser.__version__}, torch {torch_v} ({threads} threads).",
        f"Register: generic grid at 7 µm; pulse: three-point Ω ramp to 0.8·Ω_max + linear δ sweep, {a.duration} ns; "
        f"{a.shots} shots; emu dt = 10 ns, emu-mps precision 1e-5.",
        "",
        "Timing only — says nothing about solution quality. TV = total-variation distance between bitstring histograms "
        "(sampling noise alone gives roughly sqrt(#distinct/shots)/2).",
        "",
        "| atoms | backend | bond dim | wall time (s) | peak bond dim | note |",
        "|---|---|---|---|---|---|",
    ]
    for n, be, chi, t, peak, note in rows:
        lines.append(f"| {n} | {be} | {chi} | {t:.1f} | {peak} | {note} |")
    lines += ["", "## How to read this", "",
              "* Tier S (≤ 16 atoms with any sensible encoding) → pulser-simulation or emu-sv, exact.",
              "* Tier M/L → emu-mps; a result is only trusted when TV between successive bond dimensions is small "
              "(< ~0.05 beyond sampling noise) and the peak bond dimension stayed below the cap. "
              "Report the bond dimension with every emu-mps row in results.csv.",
              ""]
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
