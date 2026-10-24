# STATUS — Sat 2026-10-24, 12:45 ET (hackathon day; freeze Sun 11:00 ET)

**What changed and why.** The repo is being re-aimed from yesterday's generic `qopt/` toolkit to the actual
challenge: Track A, capacity-feasible clustering for CVRP on Pulser's `AnalogDevice` (80 atoms, 5/38 µm,
global detuning only). The shared contract (`cvrp/common.py`: instance JSON, `Formulation` interface, exact
TSP, clustering-as-set helpers) is committed and three parallel builds are running against it: instances +
validator + exact references; the model layer (brute-force checker, throwaway textbook formulation,
variable-count script, DERIVATIONS.md §0–3); embedding + pulse (positive couplings only, Pulser-validated
sequences). One environment fact shaped everything: neither the cloud sandbox nor the laptop's Linux shell can
reach PyPI/GitHub, so emulators (qutip/emu-sv/emu-mps/qubo-solver) run only in the Windows venv, through a
tiny job runner (`scripts/win_runner.bat`, needs one double-click). Pulser-core itself imports fine here, so
registers, sequences and all validation happen before anything is queued.

## Build order (user's list) — state
| # | item | state |
|---|---|---|
| 0 | env: runner bat ✔, `device_limits.py --assert-brief` ✔ (brief table == pulser), `requirements.txt` pinned ✔ (pytest/pulp/dotenv pins to be confirmed from Windows `pip freeze`), emu-mps calibration script ✔ written, **waiting for the runner** to execute |
| 1 | instances.py + validator.py | in progress (subagent) |
| 2 | reference.py (exact set-partition DP, proven optimum on all tiers + SA cross-check) | in progress (subagent) |
| 3 | model.py + textbook formulation + variable_count.py + DERIVATIONS §0–3 | in progress (subagent) |
| 4 | embed.py | in progress (subagent) |
| 5 | pulse.py | in progress (subagent) |
| 6 | run.py (metrics layer `cvrp/metrics.py` drafted) | next, then STOP for Gate 1 |

## Needs you
- Double-click `qhack\scripts\win_runner.bat` (leave it open). Queued: pip install pytest/pulp, `pip freeze`,
  brief assertion, emu-mps timing calibration (12/20/30/40 atoms × χ 16/32/64).

## Decisions taken so far (all in CLAUDE.md / DERIVATIONS.md)
- Depot at the square centre (50, 50); feasibility by certificate-first construction.
- References: exact for all tiers (enumeration of all capacity-feasible clusterings via subset DP × exact TSP),
  SA as the independent cross-check; CPLEX not available → noted in DID_NOT_WORK.md.
- Gaps only on feasible decodes; "best shot" = lowest model objective; routing-post-selected best reported
  separately and labeled; never mean cost.

## Open judge questions (running list, answers in DERIVATIONS.md §7)
- Why cluster-first; why pairwise-distance closeness; why every coupling is positive on this device; what the
  textbook formulation's non-uniform linear terms do to a global-detuning device; how references are proven.
