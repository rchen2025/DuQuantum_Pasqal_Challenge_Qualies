"""Reproducible classical objective comparison, with a hard timeout per solve.

Run from the repository root:
    python scripts/compare_clustering_models.py --out out/two_hour/models
Every model-instance is isolated in a child process; timed-out computations do
not emit a provisional optimum. Original instance references are never modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ("pairwise", "medoid", "depot-mst")
CSV_FIELDS = (
    "instance", "tier", "model", "status", "model_reference_status",
    "model_optimum", "n_model_optimal_partitions", "routed_cost",
    "reference_cost", "reference_status", "cvrp_gap_percent", "n_vehicles_used",
    "n_feasible_clusters", "time_cluster_costs_s", "time_partition_s",
    "time_routing_s", "time_total_s", "wall_time_s", "timeout_s", "error",
)


def run_isolated(instance_path: Path, model: str, timeout: float, threads: int) -> dict:
    """Keep missing results null and distinguish timeout from other failure."""
    raw = json.loads(instance_path.read_text())
    reference = raw.get("reference", {})
    base = {
        "instance": raw["name"], "tier": raw["tier"], "model": model,
        "instance_path": str(instance_path.resolve()),
        "instance_sha256": hashlib.sha256(instance_path.read_bytes()).hexdigest(),
        "reference_cost": reference.get("cost"),
        "reference_status": reference.get("reference_status", "unavailable"),
        "timeout_s": timeout, "numerical_threads": threads,
        "experiment_kind": "classical_model_selection",
    }
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               str(instance_path.resolve()), model]
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = str(threads)
    start = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                   text=True, timeout=timeout, check=True)
        base.update(json.loads(completed.stdout))
    except subprocess.TimeoutExpired:
        base.update(status="timeout", model_reference_status="not_computed",
                    model_optimum=None, routed_cost=None, cvrp_gap_percent=None,
                    error=f"Exceeded {timeout:g}s wall-clock limit; no optimum reported")
    except subprocess.CalledProcessError as exc:
        base.update(status="error", model_reference_status="not_computed",
                    model_optimum=None, routed_cost=None, cvrp_gap_percent=None,
                    error=(exc.stderr or str(exc))[-3000:])
    base["wall_time_s"] = time.perf_counter() - start
    return base


def write_summary(out: Path, records: list[dict]) -> None:
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Classical model comparison", "",
             "All models enforce exact capacity and at most K groups. Selected partitions",
             "are routed exactly. These are classical results, not quantum improvements.", "",
             "| Instance | Model | Status | Optima | Routed cost | CVRP gap (%) | Reference | Time (s) |",
             "|---|---|---|---:|---:|---:|---|---:|"]
    for record in records:
        def fmt(key, digits=3):
            value = record.get(key)
            return "—" if value is None else f"{value:.{digits}f}"
        lines.append(f"| {record['instance']} | {record['model']} | {record['status']} | "
                     f"{record.get('n_model_optimal_partitions', '—')} | {fmt('routed_cost')} | "
                     f"{fmt('cvrp_gap_percent')} | {record['reference_status']} | {fmt('wall_time_s')} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--worker":
        from cvrp.clustering_comparison import compare_model
        from cvrp.common import Instance
        print(json.dumps(compare_model(Instance.load(argv[1]), argv[2]), allow_nan=False))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Instance files; default is the 11 original instances")
    parser.add_argument("--out", type=Path, default=ROOT / "out/two_hour/models")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    paths = [Path(p).resolve() for p in args.paths] if args.paths else [
        ROOT / "instances" / f"{tier}_seed{seed:02d}.json"
        for tier, count in (("S", 5), ("M", 3), ("L", 3)) for seed in range(count)
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *argv],
        "models": list(args.models), "instances": [str(p) for p in paths],
        "timeout_s": args.timeout, "numerical_threads": args.threads,
        "routing_selection": "route-cost-independent deterministic model-optimum tie break",
        "reference_policy": "unchanged existing instance reference cost and status",
        "experiment_kind": "classical_model_selection", "python": sys.version,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    records = []
    for path in paths:
        for model in args.models:
            record = run_isolated(path, model, args.timeout, args.threads)
            records.append(record)
            (args.out / f"{record['instance']}__{model}.json").write_text(
                json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            write_summary(args.out, records)
            gap = record.get("cvrp_gap_percent")
            print(f"{record['instance']} {model}: {record['status']} gap={gap} "
                  f"wall={record['wall_time_s']:.2f}s", flush=True)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["n_complete"] = sum(r["status"] == "complete" for r in records)
    manifest["n_incomplete"] = len(records) - manifest["n_complete"]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return int(any(r["status"] in ("timeout", "error") for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
