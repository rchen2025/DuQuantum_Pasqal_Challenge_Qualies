# STATUS — 2026-10-24 (evening before the hackathon)

## Environment caveat (read first)
This toolkit was built in a sandbox that could import **pulser-core 1.9.1 and pasqal-cloud 0.24.0 from the venv's source**
but had **no torch / qutip**, so nothing below marked "untested" has executed. The hackathon venv
(`Pasqal Challenge DuQuantum/pasqal`, Python 3.14, pulser 1.9.1, qoolqit 1.5.0, qubo-solver 0.9.1, emu-sv/emu-mps 2.10.0,
pasqal-cloud 0.24.0, torch 2.14, qutip 5.3) has everything. **First thing tonight, in that venv:**
```bash
pip install -e . pytest python-dotenv      # pytest is NOT in the venv yet
pytest -q                                   # expect all green (4 slow tests now execute instead of skipping)
pytest -q -m slow -v                        # qutip / emu-sv / emu-mps / qubo-solver end to end (minutes)
python -m qopt.demo                         # full comparison table + out/report.png
```
If `python -m qopt.demo` fails, the failure is in one of the "untested" adapters below — all thin, all in one place.

## Works (executed, tests pass: 31 passed, 4 skipped)
- `qopt/model.py`: Problem → QUBO with penalty selection, slack expansion (≤, ≥, =, one-hot), brute-force verification, `auto_penalty`; knapsack / set-packing / MIS builders. Verified on all three toy problems.
- `qopt/classical.py`: brute force (≤ 26), MILP via scipy HiGHS (checked against brute force), SA, tabu (both hit the optimum on 12-var random QUBOs), `solve_best`.
- `qopt/embed.py`: `DeviceLimits.from_pulser_device` (matches cached table for all 4 devices), position optimiser (analytic gradient verified numerically; recovers realisable geometries to 1e-5), dimension-annealing starts, detuning plan with DMM floors (min-avg rule → tiny weights zeroed), rescaling to detuning limits, embedding report, brute-force ground-state check, `realized_qubo`. MIS instances embed exactly and preserve the ground state.
- `qopt/schedule.py`: parameterised schedules snapped to clock/duration/amplitude/detuning limits; **`build_sequence` executed on WeightedAnalogDevice / DigitalAnalogDevice / MockDevice with DMM** (parallel DMM + global pulse, duration = T), `to_abstract_repr()` works, `final_hamiltonian_from_sequence` reads back per-atom detunings that match the plan to 1e-3 and U to 1e-5; scan/optimize loops run.
- `qopt/run.py`: decode, repair (constraint-check + greedy fix + feasible descent), `score_counts` (feasible rates, P(opt) raw/repaired, approx ratio, rank), `run_pipeline` with the `mock` backend, `make_evaluator`.
- `qopt/report.py` + `qopt/demo.py`: end-to-end with `--backend mock --no-baseline` produces the table and `out/report.png` (register, error heatmap, waveforms, score vs shots).
- `scripts/device_limits.py`: prints every constraint from the device object (all 4 devices, `--json`, `--cloud`).
- Docs: `docs/STACK.md`, `docs/api/*.md` (digested from the installed source), `docs/research/*.md` (web, with links).

## Untested (written against the installed source / current docs, never executed here)
- `run.emulate_qutip` (`QutipBackendV2` + `QutipConfig(sampling_rate, observables=[BitStrings(evaluation_times=[1.0], num_shots)])`).
- `run.emulate_emu_sv` / `run.emulate_emu_mps` / `run.mps_convergence_check` (`SVConfig`, `MPSConfig`, `results.get_result("bitstrings", 1.0)`; `results.statistics` for bond dimension).
- `run.cloud_submit` / `cloud_fetch` (`PasqalCloudConnection`, `RemoteMPSBackend/RemoteSVBackend/RemoteEmuFreeBackend(seq, conn, config=…)`, `pulser.QPUBackend`, `RemoteResults`) — signatures checked against the installed pasqal_cloud, but no credentials here. `QPUBackend` needs the register from a layout (`Embedding.register()` does `with_automatic_layout`).
- `baseline.run_qubo_solver` (both modes): `Instance(matrix.tensor(Q))`, `Solver/SolverConfig/QuantumSolvingConfig/EmbeddingConfig/DriveShapingConfig`, functional `embedding.blade.embed_for_device` → `drive_shaping.<alg>.build_drive` → `solving.analog_quantum_sampling.compile` → `LocalEmulator.run(program).results()`, `extract_qubo`. Private attrs `solver._solver._cached_register/_cached_drive` are used to recover their register. `qoolqit.Device(pulser_device=WeightedAnalogDevice)` wrapping for `device_name=` is untested (qubo-solver's default is `AnalogDeviceWithDMM`).
- `tests/test_backends.py` (the 4 skipped tests) and `--backend qutip|emu-sv|emu-mps` in the demo.
- Physics tuning: no schedule has been validated against an actual quantum emulation — expect to tune `duration`, `omega_max`, `delta_init` tomorrow; `adiabaticity_hints` are heuristics.

## Known limitations / things to decide tomorrow
- Dense penalty QUBOs (knapsack capacity, equality sums) embed with ~0.5 relative off-diagonal error and the embedded ground state can differ from the QUBO's (reported by `ground_state_preserved=False`). Sparse/graph-structured models (MIS-like) embed exactly. Same physics limit applies to qubo-solver's BLaDE. Decide the formulation with this in mind.
- `WeightedAnalogDevice` is the assumed target (only realistic device with a DMM). If the organizers name `AnalogDevice`, there is no local detuning: diagonal terms collapse to their mean (reported). If they give a JSON device, `DeviceLimits.from_pulser_device(Device.from_abstract_repr(...))`.
- The WeightedAnalogDevice layout rule (≥150 traps, filling 0.35–0.5 above 150 traps) is handled by `with_automatic_layout`; cloud/QPU submission may still reject exotic registers — try `EMU_FRESNEL`/`mimic_qpu=True` first.
- `emu-mps` on 2D registers must be convergence-checked; `qutip` above ~14 atoms is slow.
- Repair is greedy 1-flip; for problems with structured constraints write a problem-specific repair in the model file.
- Research reports in `docs/research/` were written from web pages dated ≤ 2026-10-24; `docs/api/` is from the installed source and wins on conflict.
