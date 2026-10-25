#!/usr/bin/env python
"""Development-only capacity weight selection and held-out quantum validation.

Reproduce with the project Python environment:
  python scripts/capacity_validation.py prepare
  python scripts/capacity_validation.py quantum

Original instances, embeddings and records are never overwritten. A selection.json
is written before any holdout weight diagnostics/sampling; only development data
enter selection. This tests existing onehot-ub weights, not a new formulation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

# Set before numpy/scipy/torch imports. Child processes inherit these caps.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from cvrp.common import Instance, all_bitstrings, bits_to_str, clustering_of, gap_percent
from cvrp.exact import solve_cvrp
from cvrp.instances import generate
from cvrp.metrics import model_optimum
from cvrp.model import get_formulation

KAPPAS = (0.0, 0.05, 0.15, 0.3, 0.6, 1.0)
TOL = 1e-9
DEFAULT_OUT = ROOT / "out" / "two_hour" / "capacity"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def capacity_binding_audit(inst):
    """All unrestricted pairwise minimizers must overload (ties included)."""
    f = get_formulation("onehot-ub", inst)
    assignments = list(itertools.product(range(inst.K), repeat=inst.N))
    objectives = np.array([f.objective(a) for a in assignments])
    best = float(objectives.min())
    optima = [a for a, value in zip(assignments, objectives) if abs(value - best) <= TOL]
    feasible = [a for a in optima if inst.capacity_ok(a)]
    return {
        "seed": inst.seed, "instance": inst.name,
        "unconstrained_objective": best,
        "n_optimal_assignments": len(optima),
        "n_optimal_clusterings": len({clustering_of(a) for a in optima}),
        "n_capacity_feasible_optimal_assignments": len(feasible),
        "all_unconstrained_optima_infeasible": not feasible,
        "unconstrained_optimal_assignments": [list(a) for a in optima],
    }


def weight_for(inst, kappa):
    R = float(inst.customer_dist().sum(axis=1).max())
    return {"lam2": float(kappa) * R / (2 * inst.Q)}


def literal_energy(f, x):
    rows = np.asarray(x).reshape(f.N, f.K)
    e = f.weights["A"] * sum((1 - int(row.sum())) ** 2 for row in rows)
    for i in range(f.N):
        for j in range(i + 1, f.N):
            coefficient = f.D[i, j] + 2 * f.weights["lam2"] * f.inst.demands[i] * f.inst.demands[j]
            e += coefficient * int(rows[i] @ rows[j])
    return float(e)


def ground_state_audit(inst, kappa, include_states=False):
    f = get_formulation("onehot-ub", inst, **weight_for(inst, kappa))
    X = all_bitstrings(f.n_vars)
    energies = f.energies(X)
    literal = np.array([literal_energy(f, x) for x in X])
    err = float(np.max(np.abs(literal - energies)))
    assert err < 1e-8, (inst.name, kappa, err)
    assignments = [f.decode(x) for x in X]
    feasible = np.array([f.feasible(a) for a in assignments], dtype=bool)
    ground = np.abs(energies - energies.min()) <= TOL
    optimum = model_optimum(f)
    feasible_ground_objectives = [float(f.objective(a)) for a, ok in zip(assignments, ground & feasible) if ok]
    gaps = [gap_percent(v, optimum.value) for v in feasible_ground_objectives]
    result = {
        "instance": inst.name, "kappa": float(kappa), **f.weights,
        "n_states": len(X), "literal_max_abs_error": err,
        "ground_energy": float(energies.min()), "n_ground_states": int(ground.sum()),
        "n_feasible_ground_states": int((ground & feasible).sum()),
        "all_ground_states_onehot": all(a is not None for a, ok in zip(assignments, ground) if ok),
        "all_ground_states_feasible": bool(np.all(feasible[ground])),
        "constrained_model_optimum": optimum.value,
        "worst_feasible_ground_model_gap": max(gaps) if gaps else None,
        "ground_bits": [bits_to_str(x) for x in X[ground]],
        "ground_assignments": [list(a) if a is not None else None for a, ok in zip(assignments, ground) if ok],
    }
    if include_states:
        result["states"] = [{"bits": bits_to_str(x), "energy": float(e), "assignment": list(a) if a is not None else None,
                             "feasible": bool(ok), "ground_state": bool(g)}
                            for x, e, a, ok, g in zip(X, energies, assignments, feasible, ground)]
    return result


def select_kappa(development_rows):
    """Lexicographic selector; any instance without feasible ground state => inf gap."""
    if not development_rows or any(r.get("split") != "development" for r in development_rows):
        raise ValueError("selection accepts development rows only")
    summaries = []
    names = {r["instance"] for r in development_rows}
    for kappa in KAPPAS:
        rows = [r for r in development_rows if r["kappa"] == kappa]
        if len(rows) != len(names) or {r["instance"] for r in rows} != names:
            raise ValueError("each kappa must have exactly one row per development instance")
        all_feasible_count = sum(r["all_ground_states_feasible"] for r in rows)
        missing = any(r["worst_feasible_ground_model_gap"] is None for r in rows)
        worst_gap = math.inf if missing else max(r["worst_feasible_ground_model_gap"] for r in rows)
        summaries.append({"kappa": kappa, "all_ground_feasible_instances": all_feasible_count,
                          "worst_feasible_ground_model_gap": None if missing else worst_gap,
                          "missing_feasible_ground_state": missing})
    winner = min(summaries, key=lambda r: (-r["all_ground_feasible_instances"],
                 math.inf if r["missing_feasible_ground_state"] else r["worst_feasible_ground_model_gap"], r["kappa"]))
    return winner["kappa"], summaries


def prepare(out):
    out = Path(out)
    if (out / "selection.json").exists():
        raise FileExistsError("selection already frozen; reuse it or choose a fresh --out directory")
    (out / "instances").mkdir(parents=True, exist_ok=True)
    audit, selected = [], []
    for seed in range(1000, 1200):
        inst = generate("S", seed)
        row = capacity_binding_audit(inst)
        audit.append(row)
        if row["all_unconstrained_optima_infeasible"]:
            selected.append(inst)
        if len(selected) == 8:
            break
    write_json(out / "selection_audit.json", audit)
    if len(selected) != 8:
        raise RuntimeError(f"only {len(selected)} eligible instances in seeds 1000..1199")
    manifest = {"created_utc": utc_now(), "candidate_seeds": [1000, 1199],
        "selection_rule": "first eight seeds whose ALL unconstrained pairwise-optimal clusterings overload; tolerance 1e-9",
        "development": [i.name for i in selected[:4]], "validation": [i.name for i in selected[4:]],
        "kappas": list(KAPPAS), "sampling_seed": 0, "shots": 500,
        "pulse": "existing defaults: 5800 ns, omega_scale .75, no slow region",
        "reference_policy": "exact set-partition DP and exact Held-Karp, independent assignment enumeration cross-check on tier S"}
    for inst in selected:
        ref = solve_cvrp(inst)
        from cvrp.common import route_exactly
        enum_cost = min(route_exactly(inst, a)[0] for a in itertools.product(range(inst.K), repeat=inst.N)
                        if inst.capacity_ok(a))
        assert abs(enum_cost - ref["cost"]) < 1e-8
        inst.reference = {"cost": ref["cost"], "best_assignment": list(ref["assignment"]),
            "reference_status": "proven_optimum", "method": "exact set-partition DP and exact Held-Karp; assignment enumeration agrees",
            "n_distinct_optima": ref["n_optimal"], "exact_details": ref, "enumeration_cost": enum_cost}
        inst.extra["capacity_validation_split"] = "development" if inst in selected[:4] else "validation"
        inst.save(out / "instances" / f"{inst.name}.json")
    write_json(out / "manifest.json", manifest)
    dev_rows = []
    for inst in selected[:4]:
        for kappa in KAPPAS:
            row = ground_state_audit(inst, kappa, include_states=True)
            row["split"] = "development"
            write_json(out / "ground_states" / f"{inst.name}__kappa{kappa:g}.json", row)
            dev_rows.append({k: v for k, v in row.items() if k != "states"})
    chosen, summaries = select_kappa(dev_rows)
    frozen = {"frozen_utc": utc_now(), "selected_kappa": chosen,
              "selection_data": manifest["development"], "used_validation_data": False,
              "tie_break": "max all-ground-feasible count; min worst feasible-ground model gap; min kappa",
              "no_feasible_ground_state_policy": "infinite worst gap, serialized as null plus missing_feasible_ground_state=true",
              "candidates": summaries}
    write_json(out / "selection.json", frozen)  # MUST precede validation audits.
    validation_rows = []
    for inst in selected[4:]:
        for kappa in sorted({0.15, chosen}):
            row = ground_state_audit(inst, kappa, include_states=True)
            row["split"] = "validation"
            write_json(out / "ground_states" / f"{inst.name}__kappa{kappa:g}.json", row)
            validation_rows.append({k: v for k, v in row.items() if k != "states"})
    write_json(out / "ground_summary.json", {"development": dev_rows, "validation": validation_rows})
    write_csv(out / "ground_summary.csv", dev_rows + validation_rows)
    print(json.dumps(frozen, indent=2), flush=True)


def wilson(successes, trials):
    z = 1.959963984540054
    p = successes / trials
    den = 1 + z*z/trials
    centre = (p + z*z/(2*trials))/den
    half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/den
    return [max(0.0, centre-half), min(1.0, centre+half)]


def worker(args):
    from cvrp.run import run_one, save_record
    inst = Instance.load(args.instance)
    np.random.seed(args.seed)
    cache_dir = Path(args.out) / "embeddings" / f"kappa{args.kappa:g}"
    record = run_one(inst, "onehot-ub", "qutip", 500, args.seed,
        weights=weight_for(inst, args.kappa),
        embed_kw={"embedder": "polygons", "cache_dir": str(cache_dir)},
        dry_run=args.dry_run, verbose=False)
    # Existing describe() exposes class default kappa: retain it but explicitly
    # identify the effective experimental value and lambda override.
    record["capacity_validation"] = {"effective_kappa": args.kappa,
        "split": inst.extra.get("capacity_validation_split", "original_regression"),
        "sampling_rng": "numpy.random.seed(seed); Pulser multinomial uses numpy.random.rand",
        "sampling_seed": args.seed}
    record["method"] += f"|capacity-kappa{args.kappa:g}"
    if not args.dry_run:
        m = record["metrics_raw"]
        record["confidence_intervals_95"] = {
            k: wilson(round(m[k]*m["shots"]), m["shots"])
            for k in ("feasible_rate", "p_reach_model_optimum", "p_reach_cvrp_reference")}
    destination = Path(args.out) / ("dry_runs" if args.dry_run else "records")
    print(save_record(record, destination), flush=True)


def run_stage(command, timeout, log_path):
    start = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
        status = "complete" if proc.returncode == 0 else "failed"
        output = proc.stdout + "\n" + proc.stderr
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        status, code = "timeout", None
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        output = stdout + "\n" + stderr
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(output, encoding="utf-8")
    return {"status": status, "return_code": code, "wall_s": time.perf_counter()-start,
            "timeout_s": timeout, "command": command, "log": str(log_path)}


def quantum(out, regression=False):
    out = Path(out)
    selection = json.loads((out / "selection.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    chosen = selection["selected_kappa"]
    jobs_path = out / ("regression_jobs.json" if regression else "quantum_jobs.json")
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    names = [f"S_seed{i:02}" for i in range(5)] if regression else manifest["validation"]
    for name in names:
        for kappa in sorted({0.15, chosen}):
            if any(j["instance"] == name and j["kappa"] == kappa for j in jobs):
                continue
            instance_path = (ROOT / "instances" if regression else out / "instances") / f"{name}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--out", str(out),
                       "--instance", str(instance_path), "--kappa", str(kappa), "--seed", "0"]
            stem = f"{name}__kappa{kappa:g}"
            job = {"instance": name, "kappa": kappa, "split": "original_regression" if regression else "validation",
                   "started_utc": utc_now()}
            job["embedding"] = run_stage(command + ["--dry-run"], 120, out / "logs" / f"{stem}__embedding.log")
            if job["embedding"]["status"] == "complete":
                job["sampling"] = run_stage(command, 180, out / "logs" / f"{stem}__sampling.log")
            else:
                job["sampling"] = {"status": "not_attempted", "reason": "embedding failed or exceeded 120 s"}
            job["finished_utc"] = utc_now()
            jobs.append(job)
            write_json(jobs_path, jobs)
            summarize(out)
            print(f"{stem}: embedding={job['embedding']['status']} sampling={job['sampling']['status']}", flush=True)
    summarize(out)


def summarize(out):
    out = Path(out)
    rows = []
    for path in sorted((out / "records").glob("*.json")):
        r = json.loads(path.read_text())
        m = r["metrics_raw"]
        cv = r["capacity_validation"]
        row = {"instance": r["instance"], "split": cv["split"], "kappa": cv["effective_kappa"],
               "shots": m["shots"], "n_feasible": round(m["feasible_rate"]*m["shots"]),
               "feasible_rate": m["feasible_rate"], "feasible_ci95": r["confidence_intervals_95"]["feasible_rate"],
               "p_model_optimum": m["p_reach_model_optimum"],
               "p_model_ci95": r["confidence_intervals_95"]["p_reach_model_optimum"],
               "p_cvrp_reference": m["p_reach_cvrp_reference"], "best_model_gap": m["best_model_gap"],
               "best_cvrp_gap": m["best_cvrp_gap"], "cvrp_gap_best_routed": m["cvrp_gap_best_routed"],
               "device_argmin_feasible": r["realized_register"]["argmin_feasible"],
               "device_argmin_is_model_optimum": r["realized_register"]["argmin_is_model_optimum"],
               "wall_s": r["wall_s"], "record": str(path.relative_to(ROOT))}
        rows.append(row)
    write_json(out / "quantum_summary.json", rows)
    write_csv(out / "quantum_summary.csv", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "quantum", "regression", "worker", "summarize"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--instance")
    ap.add_argument("--kappa", type=float, default=.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args.out)
    elif args.mode in ("quantum", "regression"):
        quantum(args.out, regression=args.mode == "regression")
    elif args.mode == "worker":
        worker(args)
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()
