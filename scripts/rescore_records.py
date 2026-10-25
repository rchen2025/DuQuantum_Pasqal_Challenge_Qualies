#!/usr/bin/env python
"""Re-score every run record against the *current* instance reference blocks.

    python scripts/rescore_records.py out/m1 out/m2 out/m3 out/m4 out/m_tier [--dry-run]

Used after the §9.3 reference values change (e.g. the switch to qubo-solver best-known references):
the raw shot counts in each record are decoded and scored again with cvrp.metrics.score_counts, so
every gap / P(reach) column of results.csv follows the reference in instances/*.json instead of the
value frozen at run time.  Labelled repair blocks are re-applied with the same repair.  Records
without counts (dry runs) are skipped.  Prints one line per record whose metrics changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvrp.common import Instance  # noqa: E402
from cvrp.metrics import model_optimum, score_counts  # noqa: E402
from cvrp.model import get_formulation  # noqa: E402
from cvrp.run import _json_default, energy_descent_repair  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--instances", default="instances")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    insts, mopts, forms = {}, {}, {}
    n_seen = n_changed = 0
    for d in a.dirs:
        for p in sorted(Path(d).rglob("*.json")):
            try:
                rec = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(rec, dict) or not rec.get("counts") or "metrics_raw" not in rec:
                continue
            name = rec["instance"]
            if name not in insts:
                insts[name] = Instance.load(ROOT / a.instances / f"{name}.json")
            inst = insts[name]
            form = rec["formulation"]
            fname = form["formulation"]
            weights = {k[2:]: v for k, v in form.items() if k.startswith("w_")}
            key = (name, fname, json.dumps(weights, sort_keys=True))
            if key not in forms:
                forms[key] = get_formulation(fname, inst, **weights)
                mopts[key] = model_optimum(forms[key])
            f, mopt = forms[key], mopts[key]
            old = rec["metrics_raw"]
            m_raw, _ = score_counts(rec["counts"], f, inst, mopt=mopt)
            new = m_raw.to_dict()
            rec["metrics_raw"] = new
            if rec.get("metrics_repaired"):
                if rec.get("repair") == "energy-descent":
                    tr = energy_descent_repair(f.Q(), float(f.offset))
                    m_rep, _ = score_counts(rec["counts"], f, inst, mopt=mopt, transform=tr)
                    rec["metrics_repaired"] = m_rep.to_dict()
                elif rec.get("repair") == "capacity-local":
                    from cvrp.repair import capacity_local_repair
                    tr = capacity_local_repair(f)
                    m_rep, _ = score_counts(rec["counts"], f, inst, mopt=mopt, transform=tr)
                    rec["metrics_repaired"] = m_rep.to_dict()
                    rec["repair_stats"] = tr.stats()
                else:
                    print(f"WARN {p}: unknown repair {rec.get('repair')!r}, repaired block left as is")
            n_seen += 1
            watch = ("cvrp_reference", "reference_status", "best_cvrp_gap", "p_reach_cvrp_reference",
                     "decomposition_gap", "model_optimum", "best_model_gap")
            diff = {k: (old.get(k), new.get(k)) for k in watch if _ne(old.get(k), new.get(k))}
            if diff:
                n_changed += 1
                print(f"{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}: " +
                      ", ".join(f"{k} {o!r}->{n!r}" for k, (o, n) in diff.items()))
            if not a.dry_run:
                p.write_text(json.dumps(rec, indent=1, default=_json_default))
    print(f"re-scored {n_seen} records, {n_changed} with changed reference-dependent metrics"
          + (" (dry run, nothing written)" if a.dry_run else ""))
    return 0


def _ne(x, y) -> bool:
    try:
        if x is None or y is None:
            return (x is None) != (y is None)
        return not np.isclose(float(x), float(y), rtol=0, atol=1e-9)
    except (TypeError, ValueError):
        return x != y


if __name__ == "__main__":
    raise SystemExit(main())
