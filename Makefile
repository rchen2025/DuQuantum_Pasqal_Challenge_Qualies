# qhack — Pasqal challenge, Track A.  See CLAUDE.md.
# PY: the interpreter to use.  On Windows: make PY=..\pasqal\Scripts\python.exe   (or run the same
# commands from scripts/win_runner.bat jobs).  In a sandbox with only pulser-core: PYTHONPATH as below.
PY ?= python
INSTANCE ?= instances/S_seed00.json
FORMULATION ?= textbook
BACKEND ?= qutip
SHOTS ?= 500
SEED ?= 0

.PHONY: test demo deliverables report instances references validate variable-table clean

test:
	$(PY) -m pytest -q || $(PY) scripts/run_tests.py

instances:
	$(PY) -m cvrp.instances --out instances/

references:
	$(PY) -m cvrp.reference instances/

validate:
	$(PY) -m cvrp.validator instances/

variable-table:
	$(PY) scripts/variable_count.py --check instances/ --write-md

# One tier-S seed end to end: formulation -> embedding -> Pulser-validated pulse -> backend -> metrics
# -> run record in out/ -> headline row in results.csv (cvrp.results, after Gate 1).
demo: validate
	$(PY) scripts/device_limits.py --assert-brief
	$(PY) -m cvrp.run --instance $(INSTANCE) --formulation $(FORMULATION) --backend random --shots $(SHOTS) --seed $(SEED) --out out/
	$(PY) -m cvrp.run --instance $(INSTANCE) --formulation $(FORMULATION) --backend $(BACKEND) --shots $(SHOTS) --seed $(SEED) --out out/
	-$(PY) -m cvrp.results out/ -o results.csv

# D1–D5 pass/fail checklist (D2 'ours' row and D5 report fail until Gate 1 / the report is built).
deliverables: validate variable-table
	-$(MAKE) report
	$(PY) scripts/check_deliverables.py

# D5: docs/REPORT.md -> report/report.pdf via pandoc, then page count.
report:
	mkdir -p report/figures
	-$(PY) -m cvrp.report out/m1 out/m2 out/m3 --figures report/figures --results results.csv
	$(PY) scripts/report_tables.py
	$(PY) scripts/build_report.py --max-pages 4

clean:
	rm -rf out/*.json .pytest_cache
