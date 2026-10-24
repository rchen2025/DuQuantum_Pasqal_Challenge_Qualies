"""Adiabatic pulse for the AnalogDevice global Rydberg channel, validated by Pulser.

Shape (fractions of the total duration T; see docs/DERIVATIONS.md §6):

    Omega(t):  0 --rise_frac--> Omega  ---- plateau ----  --fall_frac--> 0        (three-point ramp)
    delta(t):  delta_init (hold) ---- linear sweep from sweep_start_frac to sweep_end_frac ---- delta_final (hold)

Both are ``InterpolatedWaveform`` objects (PCHIP through the knots, like Pulser
itself), so the sampled waveforms are monotone between knots: Omega never
overshoots Omega_max and delta never leaves [delta_init, delta_final].

Why this shape.  At t = 0 the detuning is large and negative, so the ground
state of H is "every atom in |g>" - the state the machine prepares anyway.
Omega is switched on smoothly (rise), the detuning is swept through zero to
delta_final = -alpha*h while Omega is on, and Omega is switched off (fall).
If the sweep is slow compared with the minimum gap, the adiabatic theorem
keeps the system in the instantaneous ground state, which at the end is the
minimiser of E_dev(x) = -delta_final sum x_i + sum U_ij x_i x_j = alpha*E_QUBO(x).

Optional ``slow_region = (centre_frac, width_frac, slowdown_factor)``: the
detuning sweep gets extra knots so that, over the window of the detuning path
centred at ``centre_frac`` (0 = delta_init, 1 = delta_final) and covering
``width_frac`` of the path, the slope is ``slowdown_factor`` times smaller than
outside the window.  Total sweep time is unchanged, so the rest of the sweep
is correspondingly faster.  The smallest gap sits near where the first atoms
flip (delta ~ 0 ... delta_final), which is where the knob is meant to be used.

Validation: ``pulser.Sequence`` on the device + ``seq.add`` raise on any
amplitude/detuning/duration violation; we also call ``to_abstract_repr()``
(QPU-submission serialisation) and re-check the sampled waveforms against the
device object in ``validate``.  Limits are read from the device, never typed.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_DURATION_NS = 5800      # M2 pulse scan (DERIVATIONS §6.6): device max 6000 ns minus the 200 ns margin
DURATION_MARGIN_NS = 200       # keep this far from device.max_sequence_duration


def _default_device():
    from pulser.devices import AnalogDevice
    return AnalogDevice


def _device_by_name(name: str):
    import pulser.devices as pd
    return getattr(pd, name)


def _ryd(device):
    return device.channels["rydberg_global"]


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class PulseParams:
    """Everything that defines the pulse.  Frequencies in rad/µs, durations in ns.

    omega = omega_scale * omega_device_max is the plateau amplitude.
    delta_init: start of the detuning sweep (negative: all atoms in |g> is the
        ground state).  Default (see ``default_for``):
        -min(max(|delta_final|, 2*Omega_max), max_abs_detuning).
    slow_region: (centre_frac, width_frac, slowdown_factor) or None, see module docstring.
    """
    delta_final: float
    delta_init: float
    omega_device_max: float
    duration_ns: int = DEFAULT_DURATION_NS
    omega_scale: float = 0.75      # M2 pulse scan (DERIVATIONS §6.6): 0.75*Omega_max beat full power
    rise_frac: float = 0.15
    fall_frac: float = 0.15
    sweep_start_frac: float = 0.15
    sweep_end_frac: float = 0.85
    slow_region: Optional[Tuple[float, float, float]] = None
    name: str = "three-point-ramp+linear-sweep"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def omega(self) -> float:
        return float(self.omega_scale * self.omega_device_max)

    @property
    def duration_us(self) -> float:
        return self.duration_ns * 1e-3

    def with_params(self, **kw: Any) -> "PulseParams":
        return replace(self, **kw)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["omega"] = self.omega
        d["slow_region"] = list(self.slow_region) if self.slow_region is not None else None
        return d

    def check(self) -> None:
        if not (0 < self.rise_frac < 1 and 0 < self.fall_frac < 1 and self.rise_frac + self.fall_frac < 1):
            raise ValueError("rise_frac/fall_frac must be in (0,1) with rise+fall < 1")
        if not (0 <= self.sweep_start_frac < self.sweep_end_frac <= 1):
            raise ValueError("need 0 <= sweep_start_frac < sweep_end_frac <= 1")
        if self.slow_region is not None:
            c, w, k = self.slow_region
            if not (0 <= c <= 1 and 0 < w <= 1 and k >= 1):
                raise ValueError("slow_region = (centre_frac in [0,1], width_frac in (0,1], slowdown_factor >= 1)")
        if self.duration_ns <= 0:
            raise ValueError("duration_ns must be positive")


def snap_duration(duration_ns: float, device: Any, margin_ns: int = DURATION_MARGIN_NS) -> int:
    """Round to the channel clock period and clip to [min_duration, max_sequence_duration - margin]."""
    ch = _ryd(device)
    cp = int(ch.clock_period)
    T = int(round(duration_ns / cp)) * cp
    if device.max_sequence_duration is not None:
        T = min(T, (int(device.max_sequence_duration) - margin_ns) // cp * cp)
    T = max(T, int(math.ceil(ch.min_duration / cp)) * cp)
    return T


def default_delta_init(delta_final: float, device: Any) -> float:
    """-min(max(|delta_final|, 2*Omega_max), max_abs_detuning).

    * at least 2*Omega_max below resonance so the initial gap (~sqrt(delta^2+Omega^2) per
      atom) is comfortably larger than Omega and the all-|g> state is unambiguous;
    * at least |delta_final| so the sweep is symmetric about delta = 0 (the first
      flips happen around delta ~ 0 ... U) and the slope is the same on both sides;
    * clipped to the channel limit.
    """
    ch = _ryd(device)
    mag = max(abs(float(delta_final)), 2.0 * float(ch.max_amp))
    if ch.max_abs_detuning is not None:
        mag = min(mag, float(ch.max_abs_detuning))
    return -mag


def default_for(emb_or_delta_final: Any, device: Any = None, **overrides: Any) -> PulseParams:
    """Baseline pulse for an ``cvrp.embed.Embedding`` (or a bare delta_final).

    duration defaults to min(5800 ns, max_sequence_duration - 200 ns), snapped to
    the clock period; omega_scale 0.75 (both from the M2 pulse scan, DERIVATIONS §6.6;
    the pre-scan baseline was 4000 ns at full power); delta_init from ``default_delta_init``.
    Any field of PulseParams can be overridden by keyword.
    """
    if device is None:
        name = getattr(emb_or_delta_final, "device", None)
        device = _device_by_name(name) if isinstance(name, str) else _default_device()
    delta_final = float(getattr(emb_or_delta_final, "delta_final", emb_or_delta_final))
    ch = _ryd(device)
    p = PulseParams(
        delta_final=delta_final,
        delta_init=default_delta_init(delta_final, device),
        omega_device_max=float(ch.max_amp),
        duration_ns=snap_duration(DEFAULT_DURATION_NS, device),
    )
    if overrides:
        p = replace(p, **overrides)
        if "duration_ns" in overrides:
            p = replace(p, duration_ns=snap_duration(p.duration_ns, device))
    p.check()
    return p


# --------------------------------------------------------------------------- #
# Knots and samples
# --------------------------------------------------------------------------- #
def _sweep_knots(p: PulseParams) -> Tuple[np.ndarray, np.ndarray]:
    """Detuning knots as (time fractions, values) including the holds."""
    s0, s1 = p.sweep_start_frac, p.sweep_end_frac
    di, df = p.delta_init, p.delta_final
    if p.slow_region is None or di == df:
        return np.array([0.0, s0, s1, 1.0]), np.array([di, di, df, df])
    c, w, k = p.slow_region
    lo, hi = max(0.0, c - w / 2), min(1.0, c + w / 2)
    # detuning-path segments (fractions of delta_final - delta_init) and their time weights
    segs: List[Tuple[float, float]] = []   # (path length, time weight per unit path)
    if lo > 0:
        segs.append((lo, 1.0))
    if hi > lo:
        segs.append((hi - lo, float(k)))
    if hi < 1:
        segs.append((1.0 - hi, 1.0))
    tw = np.array([L * wgt for L, wgt in segs])
    tw = tw / tw.sum() * (s1 - s0)                 # same total sweep time as the linear baseline
    path = np.concatenate([[0.0], np.cumsum([L for L, _ in segs])])
    path[-1] = 1.0
    times = np.concatenate([[s0], s0 + np.cumsum(tw)])
    times[-1] = s1
    vals = di + (df - di) * path
    t = np.concatenate([[0.0], times, [1.0]])
    v = np.concatenate([[di], vals, [df]])
    # remove duplicates created when lo == 0 or hi == 1
    keep = np.concatenate([[True], np.diff(t) > 1e-12])
    return t[keep], v[keep]


def knots(p: PulseParams) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """{'amp': (time fractions, rad/µs), 'det': (time fractions, rad/µs)}.
    Time fraction 1.0 corresponds to t = duration - 1 ns (Pulser's convention)."""
    p.check()
    t_amp = np.array([0.0, p.rise_frac, 1.0 - p.fall_frac, 1.0])
    v_amp = np.array([0.0, p.omega, p.omega, 0.0])
    t_det, v_det = _sweep_knots(p)
    return {"amp": (t_amp, v_amp), "det": (t_det, v_det)}


def waveforms(p: PulseParams, dt: float = 1.0) -> Dict[str, np.ndarray]:
    """Numpy samples of Omega(t), delta(t) exactly as Pulser's InterpolatedWaveform
    builds them (PCHIP through knots placed at frac*(T-1), sampled every ``dt`` ns)."""
    from scipy.interpolate import PchipInterpolator
    T = p.duration_ns
    t = np.arange(0.0, T, dt)
    out: Dict[str, np.ndarray] = {"t": t}
    for key, (tk, vk) in knots(p).items():
        f = PchipInterpolator(tk * (T - 1), vk)
        out[key] = np.asarray(f(np.minimum(t, T - 1)), float)
    out["amp"] = np.maximum(out["amp"], 0.0)
    return out


def sweep_slope(p: PulseParams, dt: float = 1.0) -> np.ndarray:
    """d delta / dt in rad/µs per µs (finite differences of the samples)."""
    w = waveforms(p, dt)
    return np.gradient(w["det"], dt * 1e-3)


# --------------------------------------------------------------------------- #
# Pulser sequence
# --------------------------------------------------------------------------- #
def build_sequence(emb_or_register: Any, p: PulseParams, device: Any = None, auto_layout: bool = True):
    """Embedding (or pulser Register) + PulseParams -> validated ``pulser.Sequence``.

    Pulser validates while building: ``Sequence(reg, device)`` checks the register
    (atom count, min distance, radius) and ``seq.add`` checks the pulse against the
    channel (max_amp, max_abs_detuning, duration, clock).  Any violation raises
    ValueError here.  We then serialise with ``to_abstract_repr()`` (what the cloud
    receives) and assert the duration.
    """
    from pulser import Pulse, Sequence
    from pulser.waveforms import InterpolatedWaveform

    if device is None:
        name = getattr(emb_or_register, "device", None)
        device = _device_by_name(name) if isinstance(name, str) else _default_device()
    p.check()
    if hasattr(emb_or_register, "register") and callable(emb_or_register.register):
        reg = emb_or_register.register(auto_layout=auto_layout, device=device)
    else:
        reg = emb_or_register
        if auto_layout and device.requires_layout and getattr(reg, "layout", None) is None:
            reg = reg.with_automatic_layout(device)
    T = int(p.duration_ns)
    K = knots(p)
    amp = InterpolatedWaveform(T, K["amp"][1].tolist(), times=K["amp"][0].tolist())
    det = InterpolatedWaveform(T, K["det"][1].tolist(), times=K["det"][0].tolist())
    seq = Sequence(reg, device)
    seq.declare_channel("ryd", "rydberg_global")
    seq.add(Pulse(amp, det, 0.0), "ryd")
    seq.to_abstract_repr()
    if seq.get_duration() != T:
        raise RuntimeError(f"sequence duration {seq.get_duration()} != requested {T}")
    return seq


def _global_samples(seq) -> Dict[str, np.ndarray]:
    from pulser.sampler import sample
    s = sample(seq).to_nested_dict()
    g = s["Global"]["ground-rydberg"]
    return {k: np.asarray(g[k], float) for k in ("amp", "det", "phase")}


def validate(seq, device: Any = None) -> List[str]:
    """Independent re-check of a built sequence against the device object.
    Returns a list of violations (empty = OK)."""
    device = device or seq.device
    ch = _ryd(device)
    v: List[str] = []
    T = seq.get_duration()
    if device.max_sequence_duration is not None and T > device.max_sequence_duration:
        v.append(f"duration {T} ns > max_sequence_duration {device.max_sequence_duration}")
    if T % int(ch.clock_period):
        v.append(f"duration {T} ns not a multiple of clock_period {ch.clock_period}")
    if T < ch.min_duration:
        v.append(f"duration {T} ns < min_duration {ch.min_duration}")
    g = _global_samples(seq)
    if ch.max_amp is not None and g["amp"].max() > ch.max_amp * (1 + 1e-9):
        v.append(f"max amplitude {g['amp'].max():.4g} > max_amp {ch.max_amp:.4g}")
    if g["amp"].min() < -1e-9:
        v.append(f"negative amplitude {g['amp'].min():.3g}")
    if ch.max_abs_detuning is not None and np.abs(g["det"]).max() > ch.max_abs_detuning * (1 + 1e-9):
        v.append(f"max |detuning| {np.abs(g['det']).max():.4g} > max_abs_detuning {ch.max_abs_detuning:.4g}")
    if ch.min_avg_amp and g["amp"].mean() < ch.min_avg_amp:
        v.append(f"average amplitude {g['amp'].mean():.3g} < min_avg_amp {ch.min_avg_amp}")
    reg = seq.register
    coords = np.array([np.asarray(reg.qubits[q], float) for q in reg.qubit_ids])
    if len(coords) > 1:
        d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        if d.min() < device.min_atom_distance - 1e-6:
            v.append(f"min atom distance {d.min():.4f} < {device.min_atom_distance}")
    if device.max_radial_distance is not None and np.sqrt((coords ** 2).sum(1)).max() > device.max_radial_distance + 1e-6:
        v.append("register exceeds max_radial_distance")
    if device.max_atom_num is not None and len(coords) > device.max_atom_num:
        v.append(f"{len(coords)} atoms > max_atom_num {device.max_atom_num}")
    return v


def final_detuning_from_sequence(seq) -> float:
    """delta(T) read back from the sampled sequence (rad/µs) - must equal PulseParams.delta_final."""
    return float(_global_samples(seq)["det"][-1])


def final_hamiltonian_from_sequence(seq) -> Tuple[float, np.ndarray]:
    """(delta_final, U matrix in rad/µs) read back from an actual Sequence."""
    from qopt.embed import coupling_matrix
    reg = seq.register
    coords = np.array([np.asarray(reg.qubits[q], float) for q in reg.qubit_ids])
    return final_detuning_from_sequence(seq), coupling_matrix(coords, float(seq.device.interaction_coeff))


# --------------------------------------------------------------------------- #
# Plot / CLI
# --------------------------------------------------------------------------- #
def plot_waveforms(p: PulseParams, path: str, device: Any = None, title: Optional[str] = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = device or _default_device()
    ch = _ryd(device)
    w = waveforms(p)
    t_us = w["t"] * 1e-3
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.tick_params(colors="#52514e")
    a1.plot(t_us, w["amp"], color="#2a78d6", linewidth=2)
    a1.axhline(ch.max_amp, color="#52514e", linewidth=0.8, linestyle=":")
    a1.text(t_us[-1], ch.max_amp, f" Ω_max = {ch.max_amp:.2f}", va="bottom", ha="left", fontsize=8, color="#52514e")
    a1.set_ylabel("Ω (rad/µs)", color="#52514e")
    a1.set_title(title or f"{p.name}: Ω = {p.omega:.2f} rad/µs, T = {p.duration_us:.2f} µs", loc="left", fontsize=10, color="#0b0b0b")
    a2.plot(t_us, w["det"], color="#eb6834", linewidth=2)
    a2.axhline(0.0, color="#52514e", linewidth=0.8)
    a2.text(t_us[-1], p.delta_final, f" δ_final = {p.delta_final:.2f}", va="center", ha="left", fontsize=8, color="#52514e")
    a2.text(0, p.delta_init, f"δ_init = {p.delta_init:.2f}", va="bottom", ha="left", fontsize=8, color="#52514e")
    if p.slow_region is not None:
        tk, vk = knots(p)["det"]
        a2.plot(tk * (p.duration_ns - 1) * 1e-3, vk, "o", color="#eb6834", markersize=4)
    a2.set_ylabel("δ (rad/µs)", color="#52514e")
    a2.set_xlabel("t (µs)", color="#52514e")
    a2.set_xlim(0, t_us[-1] * 1.18)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Baseline adiabatic pulse on AnalogDevice, validated by Pulser")
    ap.add_argument("--demo", action="store_true", help="toy embedding -> default pulse -> Sequence -> validation")
    ap.add_argument("--plot", type=str, default=None, help="save the waveforms to this PNG")
    ap.add_argument("--duration", type=int, default=None, help="duration in ns (snapped)")
    ap.add_argument("--omega-scale", type=float, default=None)
    ap.add_argument("--slow", type=float, nargs=3, metavar=("CENTRE", "WIDTH", "FACTOR"), default=None)
    a = ap.parse_args(argv)
    if not a.demo:
        ap.print_help()
        return 0
    from cvrp.embed import demo_embedding
    emb = demo_embedding()
    print(emb.fidelity.summary())
    over: Dict[str, Any] = {}
    if a.duration is not None:
        over["duration_ns"] = a.duration
    if a.omega_scale is not None:
        over["omega_scale"] = a.omega_scale
    if a.slow is not None:
        over["slow_region"] = tuple(a.slow)
    p = default_for(emb, **over)
    print("\npulse params:")
    for k, v in p.to_dict().items():
        print(f"  {k:18s} {v}")
    seq = build_sequence(emb, p)
    print(f"\nSequence built on {seq.device.name}: duration {seq.get_duration()} ns, "
          f"{len(seq.register.qubit_ids)} atoms, layout={'yes' if seq.register.layout is not None else 'no'}")
    v = validate(seq)
    print("validation:", "OK" if not v else v)
    dT = final_detuning_from_sequence(seq)
    print(f"delta(T) read back from sampled sequence: {dT:.6f}  (delta_final = {p.delta_final:.6f}, "
          f"diff {abs(dT - p.delta_final):.2e})")
    print(f"abstract repr: {len(seq.to_abstract_repr())} chars")
    if a.plot:
        import os
        os.makedirs(os.path.dirname(a.plot) or ".", exist_ok=True)
        print("wrote", plot_waveforms(p, a.plot))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
