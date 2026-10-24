# qhack — analog neutral-atom QUBO optimisation on the Pasqal stack

One-day hackathon toolkit. The problem arrives on the day; this repo makes the
day about **modeling → embedding → tuning**, not plumbing. Read this file, then
`docs/STACK.md`; module-level docstrings have the conventions.

## Commands
```bash
pip install -e ".[all]"            # or: pip install -r requirements.txt   (venv already has the stack)
cp .env.example .env               # fill PASQAL_USERNAME / PASQAL_PASSWORD / PASQAL_PROJECT_ID  — NEVER commit .env
pytest -q                          # pure-numpy + pulser-core tests; emulator/qubo-solver tests skip if not installed
pytest -q -m slow                  # the real emulator + qubo-solver tests (minutes)
python -m qopt.demo                # toy knapsack: verify QUBO -> embed -> classical -> qubo-solver -> ours -> table + out/report.png
python -m qopt.demo --backend qutip|emu-sv|emu-mps|mock --problem knapsack|setpacking|mis --device WeightedAnalogDevice --shots 500
python scripts/device_limits.py [DeviceName] [--json dev.json] [--cloud FRESNEL_CAN1]   # print every device constraint
QOPT_DEVICE=AnalogDevice python -m qopt.demo                                           # device via env var
```

## Device limits (pulser 1.9.1, read programmatically — re-run device_limits.py on the organizers' device)
| | WeightedAnalogDevice (DMM, default) | AnalogDevice | DigitalAnalogDevice |
|---|---|---|---|
| C6 (rad/µs·µm⁶) | 12 241 414.5 | 865 723 | 5 420 158.5 |
| min dist / max radial / max atoms | 5 µm / 80 µm / 256 | 5 / 38 / 80 | 4 / 50 / 100 |
| Ω_max, min avg Ω | 2π·2 = 12.57 rad/µs, 2π·0.3 | 2π·2, 0 | 2π·2.5, 0 |
| \|δ_global\|_max | 2π·10 = 62.8 | 2π·20 | 2π·20 |
| DMM (local δ ≤ 0) | per-atom ≥ −62.8, total ≥ −6283, min_nonzero_w·avg\|δ\| ≥ 0.628 | **none** | ≥ −125.7 / −12566 |
| max sequence duration / runs | 6000 ns / 500 | 6000 / 2000 | – |
| clock / min pulse | 4 ns / 16 ns | 4 / 16 | 4 / 16 |
| layout | required, 150–512 traps, filling 0.35–0.5 above 150 traps | required (triangular 61) | required |
| U at min distance | 783 rad/µs | 55 | 1324 |

Units everywhere: rad/µs (2π·MHz), µm, int ns. δ<0 favours ground, δ>0 favours Rydberg, interactions repulsive (+C6/r⁶).

## Module map (`qopt/`)
| module | what | key API |
|---|---|---|
| `model.py` | Problem → QUBO: penalty selection, slack expansion for ≤/≥, brute-force checker | `Problem`, `to_qubo()`, `auto_penalty()`, `brute_force_verify()`, `knapsack/set_packing/max_independent_set` |
| `classical.py` | ground truth + bar to beat | `solve_exact` (brute force ≤20, MILP via scipy/HiGHS), `simulated_annealing`, `tabu_search`, `solve_best`, `greedy_descent` |
| `embed.py` | QUBO → register + DMM weights; error report | `DeviceLimits.from_pulser_device`, `embed_qubo()` → `Embedding(.coords, .weights, .decode, .report, .register())`, `realized_qubo()` |
| `schedule.py` | adiabatic Ω ramp + δ sweep + DMM within limits; scan/optimise | `Schedule.default_for(emb)`, `build_sequence(emb, sched)`, `final_hamiltonian_from_sequence`, `scan`, `optimize` |
| `run.py` | sequence → emulator/cloud counts → decode → repair → score | `run_pipeline(problem, qubo, emb, sched, backend=…)`, `score_counts`, `repair`, `cloud_submit/cloud_fetch`, `mps_convergence_check`, `make_evaluator` |
| `baseline.py` | same QUBO through Pasqal's qubo-solver, same scorer | `run_qubo_solver(problem, qubo, mode="functional"|"solver")` |
| `report.py` | register plot (excited atoms), embedding-error heatmap, waveforms, score vs shots, table | `make_report`, `comparison_table` |
| `demo.py` | end-to-end on a toy problem | `python -m qopt.demo` |

Conventions: QUBO energy `E = xᵀQx + offset`, Q symmetric; `h_i = Q_ii`, `J_ij = 2Q_ij`. Embedding realises `U_ij = α J_ij` (only J>0 — negative couplings are dropped and reported), `δ_i = −α h_i` via global δ + DMM (`w_i·δ_dmm ≤ 0`). Bitstring index i = atom `q{i}` = QUBO variable i (slack atoms included; `Embedding.decode` drops them). `'1'` = Rydberg = x_i = 1.

## Workflow on the day
1. **Model** in `model.Problem`; `brute_force_verify` on a shrunken instance (≤ 20 QUBO vars) before anything else. Keep penalties minimal (`auto_penalty`); dense penalty terms (equality/knapsack sums) create rank-1 all-to-all couplings that 1/r⁶ in 2D cannot reproduce (rel. error ~0.5) — prefer native graph-structured formulations (MIS/MWIS, conflict graphs), native inequalities via sparse slacks, or decompose.
2. **Embed** with `embed_qubo(qubo, DeviceLimits.from_pulser_device(dev))`; look at `report.summary()`: `rel_error_offdiag`, `dropped_negative_mass`, `ground_state_preserved`, `limit_violations`. Compare with qubo-solver's `embedding_rel_error` from `baseline.run_qubo_solver`.
3. **Schedule**: `Schedule.default_for(emb)` then `schedule.scan/optimize` over `duration`, `omega_max`, `delta_init`, `sweep_shape`, `dmm_shape` using `run.make_evaluator(metric="p_optimal_repaired")` on qutip/emu-sv.
4. **Run & score** with `run_pipeline`; `score.table()`; repair is on by default (constraint-check + greedy fix + feasible descent).
5. **Baseline** with `baseline.run_qubo_solver` (functional = raw counts through our scorer; solver = their post-processed solution).
6. **Cloud/QPU**: `run_pipeline(..., backend="cloud-mps"|"cloud-sv"|"cloud-free"|"qpu")` submits async and writes `jobs/*.json`; `run.cloud_fetch(batch_id)` later; `mimic_qpu=True` validates like the QPU; QPU needs `reg.with_automatic_layout(device)` (done in `Embedding.register`).

## Emulator ceilings
qutip (`QutipBackendV2`, `sampling_rate=0.1`): ≤ 12–14 atoms. emu-sv: 15–22 on CPU, ~25 GPU (exact). emu-mps: 25–80+ but approximate — trust only for 1D/quasi-1D, low entanglement; always `run.mps_convergence_check` (TV distance < 0.05 between precision 1e-5/χ and 1e-6/2χ) and read `results.statistics[-1]["max_bond_dimension"]`. Cloud: EMU_FREE ≤12, EMU_SV ≤25, EMU_MPS ≤80; FRESNEL_CAN1 ≤1000 runs/job, 6000 ns. `auto` backend picks by atom count.

## Rules
- **Never commit credentials.** Cloud creds come from env vars only (`.env` is git-ignored). Job ids go to `jobs/` (ignored).
- Every modelling change must pass `brute_force_verify` on a tiny instance (tests enforce this on the toy problems).
- Device numbers are read from Pulser objects at runtime; the tables above are a cached snapshot (`tests/test_embed.py::test_known_limits_match_pulser_if_installed` checks they match).
- Scoring is shared: anything compared in the table goes through `run.score_counts` (same decode, same repair, same classical reference).
- `--backend mock` samples a Boltzmann distribution of the realised final Hamiltonian — for exercising the pipeline only, never for results.
- Research notes with links: `docs/STACK.md`, `docs/api/*.md` (from installed source), `docs/research/*.md` (web).

## Agents (`.claude/agents/`)
`stack-docs` (read-only researcher), `modeler`, `embedder`, `tuner`, `baseline`, `reviewer` (fresh context: diff + requirements, correctness/scoring gaps only).
