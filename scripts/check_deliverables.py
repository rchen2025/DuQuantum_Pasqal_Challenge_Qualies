#!/usr/bin/env python
"""Pass/fail checklist for Deliverable 10.1 (D1–D5).  Run by `make deliverables`.

Exit code 1 if any required item fails.  Items that legitimately cannot pass
before Gate 1 (D2 table "ours" row, D5 report) are reported as FAIL, not hidden.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CSV_COLUMNS = [
    "instance", "tier", "method", "seed", "shots", "backend", "bond_dim", "dmm", "repair",
    "atoms", "feasibility_rate", "best_model_gap_pct", "best_cvrp_gap_pct", "decomposition_gap_pct",
    "p_reach_reference", "p_model_gap_le_5pct", "embedding_fidelity_rel_frobenius", "reference_status",
]


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    ok_all = True
    print("D1 README.md + requirements.txt")
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    ok_all &= check("README exists and names the one reproduction command", "make demo" in readme or "python -m cvrp" in readme)
    req = (ROOT / "requirements.txt").read_text().splitlines() if (ROOT / "requirements.txt").exists() else []
    pkgs = [l for l in req if l.strip() and not l.startswith("#")]
    ok_all &= check("requirements.txt pins every package with ==", bool(pkgs) and all("==" in l for l in pkgs),
                    f"{len(pkgs)} packages")

    print("D2 docs/MODEL_WRITEUP.md")
    mw = (ROOT / "docs/MODEL_WRITEUP.md").read_text(encoding="utf-8") if (ROOT / "docs/MODEL_WRITEUP.md").exists() else ""
    m = re.search(r"<!-- VARIABLE_COUNT_TABLE -->(.*?)<!-- /VARIABLE_COUNT_TABLE -->", mw, re.S)
    table = m.group(1) if m else ""
    rows = [l for l in table.splitlines() if l.startswith("|") and not l.startswith("|---") and "Encoding" not in l]
    ok_all &= check("variable-count table present (script-generated)", bool(rows), f"{len(rows)} rows")
    ours = [r for r in rows if "ours" in r.lower()]
    ok_all &= check("table has the 'ours' row with numbers (Gate 1)", bool(ours) and "…" not in "".join(ours))
    ok_all &= check("write-up references DERIVATIONS.md", "DERIVATIONS.md" in mw)

    print("D3 instances/")
    r = subprocess.run([sys.executable, "-m", "cvrp.validator", "instances/"], cwd=ROOT, capture_output=True, text=True)
    ok_all &= check("python -m cvrp.validator instances/ passes", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:])
    ok_all &= check("generator + seeds committed", (ROOT / "cvrp/instances.py").exists() and (ROOT / "instances/SEEDS.md").exists())

    print("D4 results.csv")
    p = ROOT / "results.csv"
    if p.exists():
        with p.open() as fh:
            rd = csv.DictReader(fh)
            cols = rd.fieldnames or []
            n = sum(1 for _ in rd)
        missing = [c for c in REQUIRED_CSV_COLUMNS if c not in cols]
        ok_all &= check("results.csv has every required column", not missing, f"missing: {missing}" if missing else f"{len(cols)} columns")
        ok_all &= check("results.csv has rows", n > 0, f"{n} rows")
    else:
        ok_all &= check("results.csv exists", False)

    print("D5 report/report.pdf (<= 4 pages)")
    pdf = ROOT / "report/report.pdf"
    if pdf.exists():
        pages = None
        try:
            from pypdf import PdfReader  # type: ignore
            pages = len(PdfReader(str(pdf)).pages)
        except Exception:  # noqa: BLE001
            try:
                out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
                mm = re.search(r"Pages:\s+(\d+)", out)
                pages = int(mm.group(1)) if mm else None
            except Exception:  # noqa: BLE001
                pages = None
        if pages is None:
            data = pdf.read_bytes()
            pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
        ok_all &= check("report.pdf exists and has <= 4 pages", 0 < pages <= 4, f"{pages} pages")
        rep = (ROOT / "docs/REPORT.md").read_text(encoding="utf-8").lower() if (ROOT / "docs/REPORT.md").exists() else ""
        for sec in ("model", "variable", "embedding", "pulse", "results", "did not work", "ai"):
            ok_all &= check(f"REPORT.md has a '{sec}' section", sec in rep)
    else:
        ok_all &= check("report/report.pdf exists", False)

    print("Standing docs")
    for f in ("docs/DERIVATIONS.md", "docs/DID_NOT_WORK.md", "docs/AI_USE.md", "STATUS.md"):
        ok_all &= check(f"{f} exists", (ROOT / f).exists())
    print("\nALL PASS" if ok_all else "\nSOME ITEMS FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
