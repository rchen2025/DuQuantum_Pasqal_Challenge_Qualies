#!/usr/bin/env python
"""Cross-platform stand-in for the Makefile targets (Windows has no `make`).

    python scripts/make.py demo         [INSTANCE=instances/S_seed00.json BACKEND=qutip SHOTS=500 SEED=0]
    python scripts/make.py deliverables
    python scripts/make.py report
    python scripts/make.py test

Same commands as the Makefile, run with the current interpreter.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args: str, check: bool = True) -> int:
    print("+", " ".join(args), flush=True)
    r = subprocess.run(list(args), cwd=ROOT)
    if check and r.returncode:
        sys.exit(r.returncode)
    return r.returncode


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.exit(__doc__)
    target = argv[0]
    kv = dict(a.split("=", 1) for a in argv[1:] if "=" in a)
    inst = kv.get("INSTANCE", "instances/S_seed00.json")
    form = kv.get("FORMULATION", "onehot-ub")
    backend = kv.get("BACKEND", "qutip")
    shots = kv.get("SHOTS", "500")
    seed = kv.get("SEED", "0")
    embedder = kv.get("EMBEDDER", "polygons")
    if target == "test":
        rc = run(PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", check=False)
        if rc:
            run(PY, "scripts/run_tests.py")
    elif target == "validate":
        run(PY, "-m", "cvrp.validator", "instances/")
    elif target == "demo":
        run(PY, "-m", "cvrp.validator", "instances/")
        run(PY, "scripts/device_limits.py", "--assert-brief")
        run(PY, "-m", "cvrp.run", "--instance", inst, "--formulation", form, "--embedder", embedder,
            "--backend", "random", "--shots", "2000", "--seed", seed, "--out", "out/demo/")
        run(PY, "-m", "cvrp.run", "--instance", inst, "--formulation", form, "--embedder", embedder,
            "--backend", backend, "--shots", shots, "--seed", seed, "--out", "out/demo/")
        run(PY, "-m", "cvrp.results", "out/demo", "-o", "out/demo/results_demo.csv", "--md", "out/demo/summary.md")
    elif target == "report":
        (ROOT / "report/figures").mkdir(parents=True, exist_ok=True)
        run(PY, "-m", "cvrp.report", "out/m1", "out/m2", "out/m3", "--figures", "report/figures", "--results", "results.csv",
            "--pulse-scan", "out/m2_pulse/scan.md", check=False)
        run(PY, "scripts/report_tables.py", "--tiers", "S", "M", "L", "--controls-only-for", "S_seed00")
        run(PY, "scripts/build_report.py", "--max-pages", "4")
    elif target == "deliverables":
        run(PY, "-m", "cvrp.validator", "instances/")
        run(PY, "scripts/variable_count.py", "--check", "instances/", "--write-md")
        run(PY, "scripts/make.py", "report", check=False)
        run(PY, "scripts/check_deliverables.py")
    elif target == "results":
        run(PY, "-m", "cvrp.results", "out/m1", "out/m2", "out/m3", "-o", "results.csv")
    else:
        sys.exit(f"unknown target {target!r}; see --help")


if __name__ == "__main__":
    main()
