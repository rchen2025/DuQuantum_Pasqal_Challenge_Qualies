# STATUS — Sat 2026-10-24, 13:40 ET (hackathon day; freeze Sun 11:00 ET)

**What changed and why.** Items 0–6 of the build are done and committed: the whole Track-A pipeline runs end
to end on a real tier-S instance with a throwaway textbook formulation (`python -m cvrp.run ... --backend
random` works everywhere; `--backend qutip` builds and validates the Pulser sequence here and will sample
once the Windows job runner is up). References are *proven optima* for all 11 instances (exhaustive
set-partition DP × exact TSP, cross-checked by SA and, on S, by naive enumeration). The derivations document
(`docs/DERIVATIONS.md`, 780 lines) has §0–3 (CVRP, Eq. 1 → classical cost, textbook formulation with
brute-force checker numbers), §5–6 (embedding, pulse) and 13 judge Q&As; §4 waits for your formulation.
**Stopped at Gate 1 as instructed.** A fresh-context reviewer is checking physics signs, textbook algebra,
metrics and the exact solver in the background; findings will land here.

## Build order — state
| # | item | state |
|---|---|---|
| 0 | env | runner bat shipped (**not yet launched — needs your double-click**); `device_limits.py --assert-brief` ✔; `requirements.txt` pinned (pytest/pulp/dotenv pins to be confirmed from Windows `pip freeze`); emu-mps calibration script written, queued |
| 1 | instances + validator | ✔ 11 instances (S×5, M×3, L×3), certificate-first, ρ 0.60–0.95, `validator` 11/11 |
| 2 | reference | ✔ proven optimum all tiers (L: 1.3 s), SA 10 restarts agrees 11/11, naive brute force agrees on S; CPLEX not available |
| 3 | model + textbook + variable_count | ✔ checker: feasible fraction, argmin→feasible/optimal, energy gap, weight half/double/zero scan, landscape correlation; D2 table 8/30/60 and 14/45/80 (L = cap) |
| 4 | embed | ✔ refuses J<0, 5/38/80 enforced, α capped by \|δ\|≤125.66, fidelity = rel. Frobenius off-diag + worst pair, uniform-h check |
| 5 | pulse | ✔ 3-point Ω ramp + linear δ sweep + slow-region knob; every sequence built on AnalogDevice, `to_abstract_repr()`, δ(T) read-back exact |
| 6 | run | ✔ CLI → run-record JSON; metrics on feasible decodes only; labelled `repair=energy-descent`; backends qutip/emu-sv/emu-mps/random/counts |
| — | Makefile `demo`/`deliverables`/`report`, README (D1), `check_deliverables.py` | ✔ (checklist currently FAILs D2-ours, D4, D5 as expected pre-gate) |
| 7–10 | results.py (D4), baseline.py (qubo-solver), report.py figures, report.pdf | waiting for Gate 1 "go" |

Tests: 80 pass (`scripts/run_tests.py`, pytest stand-in; real pytest once Windows runner installs it).

## Current best rows (textbook = throwaway, random sampler only — NOT a result)
S_seed00, textbook (14 atoms), random 2000 shots: feasible 4.5 % raw, best model gap 0 %, CVRP gap 0 %,
P(reach ref) 0.006 raw / 0.066 with energy-descent repair. Textbook linear terms are non-uniform → flagged
`device_faithful=False`; rel. Frobenius embedding error 0.55. This is the bar any quantum run must beat.

## Needs you
1. **Double-click `qhack\scripts\win_runner.bat`** (queued: pip install pytest/pulp, pip freeze, brief assert, emu-mps calibration).
2. **Gate 1: your formulation in `docs/MODEL.md`.** Two facts from the instances to design against:
   - optima on all L instances use only 3 of 4 vehicles → the model must allow empty vehicles (brief §2 assumption 6);
   - with the pairwise-distance objective, the capacity constraint is non-binding on most tier-S seeds (34/40 toy
     seeds) — pick M1 seeds where it binds (S_seed01 ρ=0.86, S_seed00 ρ=0.71) and check the checker's `B` margin.

## Decisions taken (all in CLAUDE.md / DERIVATIONS.md)
Depot (50,50); certificate-first instances; references proven by enumeration; gaps only on feasible decodes;
"best shot" = lowest model objective, routing-post-selected best reported separately; never mean cost;
clusterings compared as sets.

## Open judge questions — see DERIVATIONS.md §7 (13 answered)
Still open for §4: why linear terms are uniform in *our* formulation; how each weight was set and what breaks when halved/doubled.
