#!/usr/bin/env python
"""Append a tag to the method label of existing run records (and rename the files accordingly).

    python scripts/relabel_record.py out/m_tier/M_seed0*__onehot-ub[emb=polygons]_emu-mps__seed0.json --tag pre-6.7

Used once, when a physics bound (DERIVATIONS §6.7) changed the default pulse end point for tier M/L: the rows
measured before the bound keep their data untouched but carry the tag `|pre-6.7`, so the corrected rows can take
the default label and results.csv keeps both (before/after).  Records whose label already carries the tag are
skipped; a note is appended to the record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvrp.run import record_path  # noqa: E402  (run from the repo root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    for f in a.files:
        p = Path(f)
        rec = json.loads(p.read_text())
        if rec["method"].endswith("|" + a.tag):
            print(f"{p.name}: already tagged")
            continue
        rec["method"] += "|" + a.tag
        rec.setdefault("notes", [])
        rec["notes"].append(a.note or f"relabelled with tag {a.tag!r} by scripts/relabel_record.py; data untouched")
        new = record_path(p.parent, rec)
        print(f"{p.name} -> {new.name}")
        if not a.dry_run:
            new.write_text(json.dumps(rec, indent=1))
            if new != p:
                p.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
