#!/usr/bin/env python
"""Generate the result tables of docs/REPORT.md from results.csv (never typed).

    python scripts/report_tables.py [--results results.csv]

Fills the block between <!-- RESULTS_TABLE --> and <!-- /RESULTS_TABLE --> with one row per
(instance, method) for the headline methods, and the block between <!-- PER_CLUSTER --> markers
with the per-instance decomposition information.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (method key, label, group) — one row per (instance, group): the first key found wins
HEADLINE_METHODS = [
    ("onehot-ub[emb=polygons]+qutip", "ours", "ours"),
    ("onehot-ub[emb=polygons]+emu-sv", "ours", "ours"),
    ("onehot-ub[emb=polygons]+emu-mps", "ours (emu-mps)", "ours"),
    ("onehot-ub[emb=polygons]+numpy-sv", "ours (in-house SV)", "ours"),
    ("onehot-ub-tight[emb=polygons]+qutip", "ours, tight A (ablation)", "tight"),
    ("onehot-ub-tight[emb=polygons]+numpy-sv", "ours, tight A (ablation, SV)", "tight"),
    ("onehot-ub[emb=polygons]+random-onehot", "random assignments (one-hot)", "rndoh"),
    ("onehot-ub+qubosolver-quantum", "qubo-solver quantum", "qsq"),
    ("onehot-ub+qubosolver-tabu", "qubo-solver tabu", "qst"),
    ("onehot-ub[emb=polygons]+random", "random sampler", "rnd"),
    ("onehot-ub[emb=generic]+random", "random sampler", "rnd"),
]


def f(v, pct=False, nd=3):
    if v in ("", None):
        return "undef."
    try:
        x = float(v)
    except ValueError:
        return str(v)
    return (f"{x:.1f} %" if pct else f"{x:.{nd}f}")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--doc", default="docs/REPORT.md")
    ap.add_argument("--tiers", nargs="+", default=["S", "M", "L"])
    ap.add_argument("--compact-tiers", nargs="*", default=["M", "L"],
                    help="tiers that only get the 'ours' and qubo-solver rows (page budget); controls/ablations stay in results.csv")
    ap.add_argument("--controls-only-for", nargs="*", default=None,
                    help="headline mode: full control block only for these instances; every other S instance shows 'ours', "
                         "every other M/L instance 'ours' + qubo-solver quantum (everything else stays in results.csv)")
    a = ap.parse_args(argv)
    compact_groups = {"ours", "qsq", "qst"}
    rows = list(csv.DictReader((ROOT / a.results).open(encoding="utf-8")))
    rows = [r for r in rows if r["repair"] == "none" and r["tier"] in a.tiers]
    # the dash counts set pandoc's relative column widths: wide 'method' / 'backend', narrow numbers
    out = ["| instance | method | feas. | model gap | CVRP gap | P(reach C*) | P(gap≤5%) | fidelity | backend |",
           "|------|------------------------|----|------|------|------|------|-----|-----------|"]
    for inst in sorted({r["instance"] for r in rows}, key=lambda n: ({"S": 0, "M": 1, "L": 2}.get(n[0], 9), n)):
        done_groups = set()
        tier = inst.split("_")[0]
        if a.controls_only_for is not None:
            allowed = None if inst in a.controls_only_for else ({"ours"} if tier == "S" else {"ours", "qsq"})
        else:
            allowed = compact_groups if tier in a.compact_tiers else None
        for key, label, group in HEADLINE_METHODS:
            if group in done_groups or (allowed is not None and group not in allowed):
                continue
            for r in rows:
                if r["instance"] != inst or r["method"] != key:
                    continue
                done_groups.add(group)
                be = (r["backend"].replace("qubosolver-", "qs-") + (f" χ={r['bond_dim']}" if r.get("bond_dim") else ""))
                out.append(f"| {inst} | {label} | {f(r['feasibility_rate'], nd=2)} | {f(r['best_model_gap_pct'], True)} | "
                           f"{f(r['best_cvrp_gap_pct'], True)} | {f(r['p_reach_reference'], nd=2)} | {f(r['p_model_gap_le_5pct'], nd=2)} | "
                           f"{f(r['embedding_fidelity_rel_frobenius'], nd=2)} | {be} |")
                break
    note = (f"\n\n(Source: `scripts/report_tables.py`, generated from `{a.results}`. Values are raw samples; "
            "repair and additional control rows remain in `results.csv`")
    if a.controls_only_for is not None:
        note += f"; the full control block is shown only for {', '.join(a.controls_only_for)}"
    elif a.compact_tiers:
        note += f"; tiers {' '.join(a.compact_tiers)} show only the principal quantum comparisons"
    block = "\n".join(out) + note + ")\n"
    # decomposition info
    dec = ["| instance | ρ | C* (best known) | model optimum routed exactly | decomposition gap |", "|---|---|---|---|---|"]
    done = set()
    for r in sorted(rows, key=lambda r: ({"S": 0, "M": 1, "L": 2}.get(r["instance"][0], 9), r["instance"])):
        if r["instance"] in done or not r.get("decomposition_gap_pct"):
            continue
        done.add(r["instance"])
        import json
        inst = json.loads((ROOT / "instances" / f"{r['instance']}.json").read_text())
        dec.append(f"| {r['instance']} | {inst['rho']:.3f} | {float(r['cvrp_reference']):.2f} | "
                   f"{float(r['cvrp_reference']) * (1 + float(r['decomposition_gap_pct']) / 100):.2f} | {f(r['decomposition_gap_pct'], True)} |")
    dblock = "\n".join(dec) + "\n"
    doc = ROOT / a.doc
    text = doc.read_text(encoding="utf-8")
    text = re.sub(r"<!-- RESULTS_TABLE -->.*?<!-- /RESULTS_TABLE -->", "<!-- RESULTS_TABLE -->\n" + block + "<!-- /RESULTS_TABLE -->", text, flags=re.S)
    text = re.sub(r"<!-- DECOMPOSITION_TABLE -->.*?<!-- /DECOMPOSITION_TABLE -->", "<!-- DECOMPOSITION_TABLE -->\n" + dblock + "<!-- /DECOMPOSITION_TABLE -->", text, flags=re.S)

    # Two-hour follow-up analyses are kept outside results.csv: model selection
    # is classical, while the capacity table is a frozen development/validation
    # ablation. Missing files leave an explicit marker rather than fabricated rows.
    model_path = ROOT / "out/two_hour/models/comparison.csv"
    if model_path.exists():
        mrows = list(csv.DictReader(model_path.open(encoding="utf-8")))
        mb = ["| tier | pairwise routed gaps | medoid routed gaps | depot-MST routed gaps |",
              "|---|---|---|---|"]
        for tier in ("S", "M", "L"):
            cells = []
            for model in ("pairwise", "medoid", "depot-mst"):
                vals = [float(r["cvrp_gap_percent"]) for r in mrows
                        if r["tier"] == tier and r["model"] == model and r["status"] == "complete"]
                cells.append(" / ".join(f"{v:.1f}%" for v in vals) if vals else "not completed")
            mb.append(f"| {tier} | {cells[0]} | {cells[1]} | {cells[2]} |")
        mblock = "\n".join(mb) + "\n"
    else:
        mblock = "Model comparison not completed.\n"
    text = re.sub(r"<!-- MODEL_COMPARISON -->.*?<!-- /MODEL_COMPARISON -->",
                  "<!-- MODEL_COMPARISON -->\n" + mblock + "<!-- /MODEL_COMPARISON -->", text, flags=re.S)

    capacity_path = ROOT / "out/two_hour/capacity/quantum_summary.csv"
    if capacity_path.exists():
        crows = [r for r in csv.DictReader(capacity_path.open(encoding="utf-8")) if r["split"] == "validation"]
        cb = ["| validation instance | feasibility κ=.15 | feasibility κ=.6 | P(model optimum) .15 / .6 | P(C*) .15 / .6 |",
              "|---|---|---|---|---|"]
        for inst in sorted({r["instance"] for r in crows}):
            by_k = {float(r["kappa"]): r for r in crows if r["instance"] == inst}
            if .15 not in by_k or .6 not in by_k:
                continue
            lo = by_k[.15]; hi = by_k[.6]
            def feas_cell(row):
                ci = __import__("json").loads(row["feasible_ci95"])
                return f"{float(row['feasible_rate']):.2f} [{ci[0]:.2f},{ci[1]:.2f}]"
            cb.append(f"| {inst} | {feas_cell(lo)} | {feas_cell(hi)} | "
                      f"{float(lo['p_model_optimum']):.2f} / {float(hi['p_model_optimum']):.2f} | "
                      f"{float(lo['p_cvrp_reference']):.2f} / {float(hi['p_cvrp_reference']):.2f} |")
        cblock = "\n".join(cb) + "\n"
    else:
        cblock = "Held-out quantum comparison not completed.\n"
    text = re.sub(r"<!-- CAPACITY_VALIDATION -->.*?<!-- /CAPACITY_VALIDATION -->",
                  "<!-- CAPACITY_VALIDATION -->\n" + cblock + "<!-- /CAPACITY_VALIDATION -->", text, flags=re.S)
    doc.write_text(text, encoding="utf-8")
    print(block)
    print(dblock)


if __name__ == "__main__":
    main()
