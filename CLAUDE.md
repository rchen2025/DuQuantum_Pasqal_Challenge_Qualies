# qhack — Pasqal challenge: CVRP clustering on an analog neutral-atom device (Track A)

Hackathon repo. The challenge brief is `docs/CHALLENGE.pdf` (brief v3.0, 2026-10-24).
**Read `docs/DERIVATIONS.md` before touching any model, weight, embedding or pulse code: it is the
single source of truth; code follows it, never the reverse.** `STATUS.md` says where we are.

## Commands
```bash
# Windows hackathon venv (..\pasqal, Python 3.14) has the full stack; this is the only place emulators run.
python -m pip install -r requirements.txt                  # exact pins
pytest -q                                                   # all tests (emulator tests skip if torch/qutip missing)
python scripts/run_tests.py [pattern]                       # pytest stand-in for sandboxes without pytest
python scripts/device_limits.py --assert-brief              # pulser AnalogDevice == brief §3.1 table (gate)
python -m cvrp.instances --out instances/                   # D3: generate seeded instances (certificate first)
python -m cvrp.reference instances/                         # D3: exact references + SA cross-check -> JSON
python -m cvrp.validator instances/                         # D3 gate: every §2 assumption + certificate + reference
python scripts/variable_count.py --write-md                 # D2: §4 table into docs/MODEL_WRITEUP.md (never typed)
python -m cvrp.run --instance instances/S_seed00.json --formulation <name> --backend qutip --shots 500 --seed 0 --out out/
python -m cvrp.results out/ -o results.csv                  # D4
python -m cvrp.baseline ...                                 # qubo-solver quantum + classical, same scoring
make demo | make deliverables | make report                 # end-to-end on one tier-S seed | D1–D5 checklist | report.pdf (<= 4 pages)
scripts/win_runner.bat                                      # Windows job runner: drops .cmd into .runner/queue, logs in .runner/out
```

## Device (Pulser 1.9.1 `AnalogDevice`, asserted by `scripts/device_limits.py --assert-brief`)
| max atoms | min separation | max register radius | Ω_max | \|δ\| max (global, no DMM) | max sequence | C6/ħ |
|---|---|---|---|---|---|---|
| 80 | 5 µm | 38 µm | 12.566 rad/µs (2π·2 MHz) | 125.66 rad/µs (2π·20 MHz) | 6000 ns | 8.657e5 rad/µs·µm⁶ |

U at 5 µm = 55.4 rad/µs. Clock 4 ns, min pulse 16 ns. Layout required (`Register.with_automatic_layout`).
Units everywhere: rad/µs, µm, int ns. δ<0 favours |g⟩, δ>0 favours |r⟩; interactions are repulsive (+C6/r⁶).

## Package map (`cvrp/`)
| module | what |
|---|---|
| `common.py` | `Instance` (JSON io), `Formulation` ABC (Q, decode, feasible, objective, cluster_cost), exact TSP (Held–Karp), `route_exactly`, `clustering_of`, `gap_percent` |
| `instances.py` / `validator.py` | tiers S/M/L, seeded, certificate-first; validator = D3 gate |
| `exact.py` / `reference.py` | exact set-partition DP (all tiers, proven optimum, #optima mod relabeling) + SA cross-check; writes `reference` into instance JSON |
| `model.py` + `formulations/` | registry, `brute_force_check` (feasible fraction, argmin decodes feasible?, gap landscape, weight half/double) ; `textbook` = throwaway one-hot+slack for plumbing |
| `embed.py` | QUBO → register on AnalogDevice: refuses J≤0, enforces 5/38/80, fidelity = rel. Frobenius off-diag + worst pair, uniform-linear check |
| `pulse.py` | three-point Ω ramp + linear δ sweep (+ slow region), validated by Pulser's `Sequence` |
| `run.py` | formulation+embedding+pulse+backend → bitstrings → decode → feasible/infeasible → optional *labeled* repair → §9 metrics → run-record JSON |
| `results.py` / `baseline.py` / `report.py` | results.csv (D4) / qubo-solver baselines / figures |
| `qopt/` | yesterday's generic toolkit (reused where it fits: position optimiser, emulator adapters). Not the deliverable. |

## Rules (non-negotiable)
- **Every coupling positive.** `cvrp.embed.embed` raises on any J_ij < 0. Zero couplings are only approximated (far atoms) and reported.
- **Every sequence passes Pulser validation** (`Sequence` on `AnalogDevice` + `to_abstract_repr()`); limits are read from the device object, never hard-coded.
- **Every gap is computed on a feasible decode.** Infeasible shots never contribute a cost. Repair, when used, is a separate labeled method (`repair=greedy`), never silently mixed in.
- **Never mean cost.** Score shots by best gap, feasibility rate, P(gap ≤ ε) (brief §8.2, §9.4).
- **Vehicles identical → compare clusterings as sets** of customer groups (`clustering_of`), for #optima and P(reach reference).
- **DERIVATIONS.md before code.** A weight, encoding or pulse change updates `docs/DERIVATIONS.md` in the same commit.
- **Numbers in docs come from scripts** (`variable_count.py`, `results.py`), never typed.
- `STATUS.md` opens with one plain-language paragraph (what changed, why). `docs/DID_NOT_WORK.md`: one honest sentence per failure. `docs/AI_USE.md`: what AI did, per deliverable.
- Never commit credentials; `.env` is ignored. `.runner/`, `out/` are ignored.

## Gates
- **Gate 1**: the real formulation arrives in `docs/MODEL.md` → write its DERIVATIONS.md §4 first, then `cvrp/formulations/<name>.py`, regenerate the D2 table, run M1 (3 tier-S seeds: ours + qubo-solver + random sampler), reviewer pass, then M2 (embedding fidelity → weights via brute-force checker → pulse scan). Before/after rows for every change.
- **Gate 2**: Track B decision, overnight run (`make deliverables` at the end).
