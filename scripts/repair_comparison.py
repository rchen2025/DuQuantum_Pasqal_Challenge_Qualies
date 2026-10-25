#!/usr/bin/env python
"""Rescore frozen headline counts and matched random-onehot controls, without emulation.

    python scripts/repair_comparison.py --out out/two_hour/repair

Historical files and results.csv are read-only. The source manifest records SHA256
hashes; absent quantum runs remain missing, while their classical controls are run.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvrp.common import Instance
from cvrp.metrics import model_optimum, random_onehot_counts, score_counts
from cvrp.model import get_formulation
from cvrp.repair import capacity_local_repair
from cvrp.results import rows_from_record, write_csv
from cvrp.run import save_record


def source_path(name):
    folder, backend = ("m3", "qutip") if name.startswith("S") else ("m_tier", "emu-mps")
    return ROOT / "out" / folder / f"{name}__onehot-ub[emb=polygons]_{backend}__seed0.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/two_hour/repair")
    parser.add_argument("--instances", nargs="*", help="optional subset for focused reruns")
    args = parser.parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    names = args.instances or ([f"S_seed{i:02d}" for i in range(5)] +
                               [f"{tier}_seed{i:02d}" for tier in "ML" for i in range(3)])
    rows, manifest, summary = [], [], []
    for name in names:
        started = time.perf_counter()
        inst = Instance.load(ROOT / "instances" / f"{name}.json")
        source = source_path(name)
        original = json.loads(source.read_text()) if source.exists() else None
        if original and not original.get("counts"):
            original = None
        weights = ({k[2:]: v for k, v in original["formulation"].items() if k.startswith("w_")}
                   if original else {})
        f = get_formulation("onehot-ub", inst, **weights)
        optimum = model_optimum(f)
        shots = sum(original["counts"].values()) if original else 500
        assert shots == 500, "Expected the frozen headline budget of 500 shots"
        manifest.append({"instance": name, "source": str(source.relative_to(ROOT)),
                         "status": "available" if original else "missing_quantum_counts",
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest() if original else None,
                         "shots": shots, "random_seeds": [0, 1, 2]})
        records = []
        if original:
            rec = copy.deepcopy(original)
            rec["source_record"] = manifest[-1]["source"]
            rec["source_sha256"] = manifest[-1]["sha256"]
            rec["source_wall_s"] = rec.get("wall_s")
            records.append(rec)
        for seed in range(3):
            records.append({"instance": name, "tier": inst.tier, "instance_seed": inst.seed,
                            "formulation": f.describe(), "n_vars": f.n_vars, "atoms": None,
                            "method": "onehot-ub+random-onehot", "backend": "random-onehot",
                            "seed": seed, "shots": shots, "counts": random_onehot_counts(f, shots, seed),
                            "dmm": False, "device_faithful": None,
                            "backend_params": {"backend": "random-onehot", "shots": shots,
                                               "seed": seed, "note": "uniform one-hot assignments; no device run"}})
        for rec in records:
            t0 = time.perf_counter()
            raw, _ = score_counts(rec["counts"], f, inst, mopt=optimum)
            repair = capacity_local_repair(f)
            repaired, _ = score_counts(rec["counts"], f, inst, mopt=optimum, transform=repair)
            rec.update(metrics_raw=raw.to_dict(), metrics_repaired=repaired.to_dict(),
                       repair="capacity-local", repair_stats=repair.stats(),
                       analysis_wall_s=time.perf_counter() - t0,
                       analysis_note="Raw counts unchanged; capacity-local and best-route selection are classical processing.")
            rec["method"] = rec["method"].split("+repair:")[0] + "+repair:capacity-local"
            if rec["backend"] == "random-onehot":
                rec["wall_s"] = rec["analysis_wall_s"]
            path = save_record(rec, out / "records")
            rows.extend(rows_from_record(rec, str(path.relative_to(ROOT))))
            summary.append({"instance": name, "backend": rec["backend"], "seed": rec["seed"],
                            "shots": shots, "raw_feasible": raw.feasible_rate,
                            "repaired_feasible": repaired.feasible_rate,
                            "raw_model_gap": raw.best_model_gap,
                            "repaired_model_gap": repaired.best_model_gap,
                            "raw_route_gap_model_selection": raw.best_cvrp_gap,
                            "raw_route_gap_route_selection": raw.cvrp_gap_best_routed,
                            "repaired_route_gap_model_selection": repaired.best_cvrp_gap,
                            "repaired_route_gap_route_selection": repaired.cvrp_gap_best_routed,
                            "raw_p_reference": raw.p_reach_cvrp_reference,
                            "repaired_p_reference": repaired.p_reach_cvrp_reference,
                            "repair_wall_s": repaired.repair_wall_s})
        print(f"{name}: {len(records)} samplers scored in {time.perf_counter()-started:.2f}s", flush=True)
        # Checkpoint after each instance, so a bounded run always leaves usable records.
        write_csv(rows, out / "comparison.csv")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Raw and classical capacity-local comparisons", "",
             "500 shots per sampler; random controls use seeds 0, 1, 2. Missing quantum records are not fabricated.",
             "Route-selection columns optimize route length over observed feasible samples; this is classical post-selection.",
             "Wilson intervals quantify finite-shot uncertainty conditional on the fixed sampler only.", "",
             "| Instance | Sampler | Seed | Raw feasible | Repaired feasible | Raw route gap (model selection) | Raw route gap (route selection) | Repaired route gap (route selection) | Repair seconds |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "undefined" if value is None else f"{value:.3f}"
    for s in summary:
        lines.append("| " + " | ".join(str(s[k]) if k in ("instance", "backend", "seed") else fmt(s[k])
                     for k in ("instance", "backend", "seed", "raw_feasible", "repaired_feasible",
                               "raw_route_gap_model_selection", "raw_route_gap_route_selection",
                               "repaired_route_gap_route_selection", "repair_wall_s")) + " |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} rows saved to {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
