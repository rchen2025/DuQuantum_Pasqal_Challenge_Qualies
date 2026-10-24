#!/usr/bin/env python
"""Parameter scans for M2, scored the way the brief asks (best gap, P(reach reference), feasibility).

    python scripts/scan.py weights --instances S_seed00 S_seed01 S_seed04 --A 0.5 0.75 1 1.5 --lam2 0.5 1 2 --out out/m2_weights
    python scripts/scan.py pulse   --instances ... --duration 2000 4000 5800 --omega 0.5 0.75 1.0 --delta-final-scale 0.8 1 1.25 --out out/m2_pulse
    python scripts/scan.py slow    --instances ... --slow 0.5,0.3,3 0.6,0.3,3 0.7,0.4,4 --out out/m2_slow

Every point is a full `cvrp.run.run_one` call (register, Pulser-validated sequence, backend, §9 metrics) and
writes a normal run record, so `python -m cvrp.results <out>` turns a scan into results.csv rows.  The
score table printed/saved (<out>/scan.md) lists, per point: feasibility rate, best model gap, best CVRP gap,
P(reach model optimum), P(reach CVRP reference), P(model gap <= 5 %), averaged over the instances, plus the
per-instance P(reach).  Never a mean cost.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvrp.common import Instance  # noqa: E402
from cvrp.model import get_formulation  # noqa: E402
from cvrp.run import run_one, save_record  # noqa: E402


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(v):
    return "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def run_points(points, insts, formulation, backend, shots, seed, embedder, out, label_fn, quiet=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pt in points:
        per = []
        t0 = time.time()
        for inst in insts:
            kw = pt["kw"](inst)
            rec = run_one(inst, formulation, backend, shots, seed, embed_kw={"embedder": embedder, "cache_dir": str(out / "cache")},
                          verbose=not quiet, **kw)
            rec["scan_point"] = pt["label"]
            save_record(rec, out)
            m = rec["metrics_raw"]
            per.append(m)
        row = {
            "point": pt["label"],
            "feasibility": mean([m["feasible_rate"] for m in per]),
            "best_model_gap": mean([m["best_model_gap"] for m in per]),
            "best_cvrp_gap": mean([m["best_cvrp_gap"] for m in per]),
            "P_reach_model": mean([m["p_reach_model_optimum"] for m in per]),
            "P_reach_ref": mean([m["p_reach_cvrp_reference"] for m in per]),
            "P_gap_le_5": mean([m["p_gap_le"]["5"] for m in per]),
            "per_instance_P_reach_model": " / ".join(f"{m['p_reach_model_optimum']:.2f}" for m in per),
            "wall_s": time.time() - t0,
        }
        rows.append(row)
        print(f"{row['point']:<40} feas={fmt(row['feasibility'])} gap={fmt(row['best_model_gap'])} "
              f"P(reach model)={fmt(row['P_reach_model'])} [{row['per_instance_P_reach_model']}]  ({row['wall_s']:.0f}s)", flush=True)
    cols = ["point", "feasibility", "best_model_gap", "best_cvrp_gap", "P_reach_model", "P_reach_ref", "P_gap_le_5", "per_instance_P_reach_model"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    (out / "scan.md").write_text("\n".join(md) + "\n")
    (out / "scan.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(md))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["weights", "pulse", "slow"])
    ap.add_argument("--instances", nargs="+", default=["S_seed00", "S_seed01", "S_seed04"])
    ap.add_argument("--formulation", default="onehot-ub")
    ap.add_argument("--backend", default="numpy-sv")
    ap.add_argument("--embedder", default="polygons")
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--A", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.5], help="multipliers of the A rule")
    ap.add_argument("--lam2", nargs="+", type=float, default=[0.5, 1.0, 2.0], help="multipliers of the lam2 rule")
    ap.add_argument("--duration", nargs="+", type=int, default=[2000, 4000, 5800])
    ap.add_argument("--omega", nargs="+", type=float, default=[0.5, 0.75, 1.0], help="omega_scale (fraction of Omega_max)")
    ap.add_argument("--delta-final-scale", nargs="+", type=float, default=[1.0], help="multiplier on the embedding's delta_final")
    ap.add_argument("--slow", nargs="+", default=["0.5,0.3,3", "0.6,0.3,3", "0.7,0.4,4"], help="centre,width,factor")
    ap.add_argument("--base-pulse", default="", help="pulse overrides applied to every point, e.g. duration_ns=5800,omega_scale=0.75")
    a = ap.parse_args(argv)

    insts = [Instance.load(ROOT / "instances" / f"{n}.json") for n in a.instances]
    base = {}
    for part in a.base_pulse.split(","):
        if part.strip():
            k, v = part.split("="); base[k] = int(float(v)) if k == "duration_ns" else float(v)

    points = []
    if a.what == "weights":
        for mA, mL in itertools.product(a.A, a.lam2):
            def kw(inst, mA=mA, mL=mL):
                f0 = get_formulation(a.formulation, inst)   # rule values
                return {"weights": {"A": f0.weights["A"] * mA, "lam2": f0.weights["lam2"] * mL}, "pulse_overrides": dict(base)}
            points.append({"label": f"A x{mA:g}, lam2 x{mL:g}", "kw": kw})
    elif a.what == "pulse":
        for T, om, ds in itertools.product(a.duration, a.omega, a.delta_final_scale):
            def kw(inst, T=T, om=om, ds=ds):
                po = dict(base); po.update(duration_ns=T, omega_scale=om)
                if ds != 1.0:
                    po["delta_final_scale"] = ds
                return {"pulse_overrides": po}
            points.append({"label": f"T={T} ns, Omega x{om:g}, delta_f x{ds:g}", "kw": kw})
    else:
        for s in a.slow:
            c, w, k = (float(v) for v in s.split(","))
            def kw(inst, c=c, w=w, k=k):
                po = dict(base); po["slow_region"] = (c, w, k)
                return {"pulse_overrides": po}
            points.append({"label": f"slow centre={c:g} width={w:g} factor={k:g}", "kw": kw})
        def kw0(inst):
            return {"pulse_overrides": dict(base)}
        points.insert(0, {"label": "no slow region (baseline)", "kw": kw0})

    run_points(points, insts, a.formulation, a.backend, a.shots, a.seed, a.embedder, a.out, None)


if __name__ == "__main__":
    main()
