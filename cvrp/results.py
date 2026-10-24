"""Deliverable D4: results.csv from the run records written by cvrp.run / cvrp.baseline.

    python -m cvrp.results out/ [out/m1 ...] -o results.csv [--md results.md]

One row per (instance, method, seed) — a record that carries a labelled repair
pass contributes a SECOND row with the same method name plus the repair label
in the `repair` column, so raw and repaired numbers never share a row.
Every Table-3 metric (brief §9.4) and every applicable §9.2 gap (Track A: model
gap, CVRP gap; plus the informative decomposition gap) is a column, with the
backend / bond-dimension / DMM / repair labels the brief asks for.  Rows are
sorted by tier, instance, method, seed.  Nothing here is a mean cost.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

COLUMNS = [
    # identity
    "instance", "tier", "instance_seed", "method", "formulation", "seed", "shots",
    # labels the brief requires
    "backend", "backend_version", "bond_dim", "peak_bond_dim", "dmm", "repair", "embedder", "device", "device_faithful",
    # size / atom efficiency
    "n_vars", "atoms",
    # Table 3 metrics
    "feasibility_rate", "decodable_rate",
    "best_model_gap_pct", "best_cvrp_gap_pct", "decomposition_gap_pct",
    "cvrp_gap_best_routed_pct",
    "p_reach_reference", "p_reach_model_optimum",
    "p_model_gap_le_0pct", "p_model_gap_le_1pct", "p_model_gap_le_5pct", "p_model_gap_le_10pct",
    "p_cvrp_gap_le_5pct",
    "embedding_fidelity_rel_frobenius", "embedding_worst_pair_rel_err", "embedding_subspace_rel_frobenius",
    "realized_argmin_is_model_optimum", "realized_rank_of_model_optimum", "embedding_alpha", "delta_final",
    "reference_status", "cvrp_reference", "model_optimum", "model_n_optima",
    # best shot details
    "best_model_objective", "best_routed_cost", "best_assignment",
    "n_distinct_feasible_clusterings", "argmin_energy_feasible",
    # weights & pulse
    "weights", "pulse_duration_ns", "omega_scale", "delta_init", "slow_region",
    "wall_s", "record",
]


def _g(d: Optional[Dict[str, Any]], *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return json.dumps(v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return v


def rows_from_record(rec: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
    """Flatten a run record into one row per (raw, repaired) metric block."""
    if not rec.get("counts"):
        return []  # dry runs carry no results
    out: List[Dict[str, Any]] = []
    blocks = [("none", rec.get("metrics_raw"))]
    if rec.get("metrics_repaired"):
        blocks.append((rec.get("repair", "repair"), rec["metrics_repaired"]))
    bp = rec.get("backend_params") or {}
    emb = rec.get("embedding") or {}
    fid = emb.get("fidelity") or emb.get("report") or {}
    pulse = rec.get("pulse") or {}
    form = rec.get("formulation") or {}
    weights = {k[2:]: v for k, v in form.items() if k.startswith("w_")}
    for repair, m in blocks:
        if not m:
            continue
        method = rec["method"]
        # strip any "+repair:..." suffix from the base label; the repair column carries it
        base = method.split("+repair:")[0]
        row = {
            "instance": rec.get("instance"), "tier": rec.get("tier"), "instance_seed": rec.get("instance_seed"),
            "method": base, "formulation": form.get("formulation"), "seed": rec.get("seed"), "shots": m.get("shots"),
            "backend": rec.get("backend"), "backend_version": bp.get("version"),
            "bond_dim": bp.get("bond_dim"), "peak_bond_dim": bp.get("peak_bond_dim"),
            "dmm": rec.get("dmm", False), "repair": repair, "embedder": (rec.get("embedder") or emb.get("embedder") or fid.get("embedder") or "generic") if rec.get("backend") != "random" else "n/a (no register sampled)",
            "device": rec.get("device"),
            "device_faithful": rec.get("device_faithful"),
            "n_vars": rec.get("n_vars"), "atoms": rec.get("atoms"),
            "feasibility_rate": m.get("feasible_rate"), "decodable_rate": m.get("decodable_rate"),
            "best_model_gap_pct": m.get("best_model_gap"), "best_cvrp_gap_pct": m.get("best_cvrp_gap"),
            "decomposition_gap_pct": m.get("decomposition_gap"),
            "cvrp_gap_best_routed_pct": m.get("cvrp_gap_best_routed"),
            "p_reach_reference": m.get("p_reach_cvrp_reference"), "p_reach_model_optimum": m.get("p_reach_model_optimum"),
            "p_model_gap_le_0pct": _g(m, "p_gap_le", "0"), "p_model_gap_le_1pct": _g(m, "p_gap_le", "1"),
            "p_model_gap_le_5pct": _g(m, "p_gap_le", "5"), "p_model_gap_le_10pct": _g(m, "p_gap_le", "10"),
            "p_cvrp_gap_le_5pct": _g(m, "p_cvrp_gap_le", "5"),
            "embedding_fidelity_rel_frobenius": fid.get("rel_frobenius_offdiag"),
            "embedding_worst_pair_rel_err": fid.get("worst_pair_rel_error"),
            "embedding_subspace_rel_frobenius": fid.get("subspace_rel_frobenius"),
            "realized_argmin_is_model_optimum": _g(rec, "realized_register", "argmin_is_model_optimum"),
            "realized_rank_of_model_optimum": _g(rec, "realized_register", "rank_of_model_optimum"),
            "embedding_alpha": emb.get("alpha"), "delta_final": pulse.get("delta_final", emb.get("delta_final")),
            "reference_status": m.get("reference_status"), "cvrp_reference": m.get("cvrp_reference"),
            "model_optimum": m.get("model_optimum"), "model_n_optima": m.get("model_n_optima"),
            "best_model_objective": m.get("best_model_objective"), "best_routed_cost": m.get("best_routed_cost"),
            "best_assignment": m.get("best_assignment"),
            "n_distinct_feasible_clusterings": m.get("n_distinct_feasible_clusterings"),
            "argmin_energy_feasible": m.get("argmin_energy_feasible"),
            "weights": weights, "pulse_duration_ns": pulse.get("duration_ns"), "omega_scale": pulse.get("omega_scale"),
            "delta_init": pulse.get("delta_init"), "slow_region": pulse.get("slow_region"),
            "wall_s": rec.get("wall_s"), "record": path,
        }
        out.append(row)
    return out


def collect(dirs: Iterable[str | Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*.json")):
            try:
                rec = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(rec, dict) or "method" not in rec or "instance" not in rec:
                continue
            rows.extend(rows_from_record(rec, str(p)))
    tier_order = {"S": 0, "M": 1, "L": 2}
    rows.sort(key=lambda r: (tier_order.get(r["tier"], 9), r["instance"], r["method"], r["repair"], r["seed"]))
    return rows


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in COLUMNS})


SUMMARY_COLS = ["instance", "method", "repair", "seed", "shots", "atoms", "feasibility_rate",
                "best_model_gap_pct", "best_cvrp_gap_pct", "p_reach_reference", "p_model_gap_le_5pct",
                "embedding_fidelity_rel_frobenius", "realized_rank_of_model_optimum", "reference_status"]


def write_md(rows: List[Dict[str, Any]], path: str | Path) -> str:
    lines = ["| " + " | ".join(SUMMARY_COLS) + " |", "|" + "---|" * len(SUMMARY_COLS)]
    for r in rows:
        lines.append("| " + " | ".join(str(_fmt(r.get(c))) for c in SUMMARY_COLS) + " |")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="directories with run-record JSONs (searched recursively)")
    ap.add_argument("-o", "--out", default="results.csv")
    ap.add_argument("--md", default=None, help="also write a Markdown summary table")
    a = ap.parse_args(argv)
    rows = collect(a.dirs)
    write_csv(rows, a.out)
    print(f"{len(rows)} rows -> {a.out}")
    if a.md:
        print(write_md(rows, a.md))


if __name__ == "__main__":
    main()
