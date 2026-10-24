# The Pasqal stack — what we use, verified 2026-10-24

Sources of truth, in order: (1) the **installed source** in the hackathon venv
(`pasqal/Lib/site-packages`, versions below) — code-level digests in
[`docs/api/`](api/); (2) current docs/PyPI/GitHub — notes with links in
[`docs/research/`](research/). Anything older than pulser 1.9 / qubo-solver 0.9
on the web is stale (both had breaking renames in Aug–Oct 2026).

| package | version in venv | python | import | role |
|---|---|---|---|---|
| pulser (= pulser-core + pulser-simulation + pasqal-cloud) | **1.9.1** (2026-09-08) | ≥3.10 | `pulser`, `pulser_simulation` | registers, sequences, devices, QuTiP emulator |
| qoolqit | **1.5.0** | ≥3.10 | `qoolqit` | dimensionless programs → compile to device; embedders; one run() API |
| qubo-solver | **0.9.1** (2026-10-15) | 3.10–3.14 | `qubosolver` | Pasqal's end-to-end QUBO pipeline (our baseline) |
| emu-sv | **2.10.0** | 3.11–3.14 | `emu_sv` | exact state-vector emulator (torch) |
| emu-mps | **2.10.0** | 3.11–3.14 | `emu_mps` | MPS/TDVP emulator (torch) |
| pasqal-cloud | **0.24.0** | ≥3.10 | `pasqal_cloud` | cloud emulators + QPU (async batches) |

Install (what the venv already has): `pip install pulser qoolqit qubo-solver emu-sv emu-mps pasqal-cloud`
(`emu-*` pull torch ~2.7; installing an emulator may bump pulser-core — keep all at the versions above, see `requirements.txt`).

## 1. Pulser — the 5 calls we use

```python
from pulser import Register, Sequence, Pulse, InterpolatedWaveform, ConstantWaveform
from pulser.devices import WeightedAnalogDevice            # or AnalogDevice / DigitalAnalogDevice / MockDevice

reg = Register.from_coordinates(coords_um, prefix="q", center=True)        # centred!  (qubit order = bitstring order)
reg = reg.with_automatic_layout(device)                                    # needed for QPU (requires_layout=True)
seq = Sequence(reg, device); seq.declare_channel("ryd", "rydberg_global")
seq.add(Pulse(InterpolatedWaveform(T, [0, Om, Om, 0], times=[0, .15, .85, 1]),
              InterpolatedWaveform(T, [d0, d0, d1, d1], times=[0, .15, .85, 1]), 0.0), "ryd")
dmap = reg.define_detuning_map({"q0": 1.0, "q1": 0.3, ...})               # weights in [0,1]
seq.config_detuning_map(dmap, "dmm_0"); seq.add_dmm_detuning(ConstantWaveform(T, -delta_dmm), "dmm_0", protocol="no-delay")
samples = pulser.sampler.sample(seq).to_nested_dict()                      # per-atom detuning read-back (qopt.schedule.final_hamiltonian_from_sequence)
json_str = seq.to_abstract_repr()                                          # what the cloud receives
```
* Units: Ω, δ in **rad/µs** (2π·MHz), lengths µm, durations **int ns** (multiples of `clock_period`=4, ≥ `min_duration`=16).
* `H = Σ Ω/2 σx − Σ δ_i n_i + Σ C6/r⁶ n_i n_j`: **δ<0 favours ground, δ>0 favours Rydberg**; interactions are always repulsive (+).
* DMM: local detuning on atom k = `w_k·δ_DMM(t)` with δ_DMM ≤ 0 and `max(w)·min(δ) ≥ bottom_detuning`, `Σw·min(δ) ≥ total_bottom_detuning`, `min_nonzero(w)·avg|δ| ≥ min_avg_abs_detuning` (WeightedAnalogDevice: 2π·0.1 → tiny weights must be zeroed, qopt does this).
* `seq.add` defaults to protocol `"min-delay"` but `add_dmm_detuning` to `"no-delay"` — add the DMM **after** the global pulse with `"no-delay"` or the two land sequentially (qopt does this; duration check in tests).
* Amplitude samples must be ≥ 0; Pchip interpolation never overshoots the knots.
* Emulator (1.9): `pulser_simulation.QutipBackendV2(seq, config=QutipConfig(sampling_rate=0.1, observables=[BitStrings(evaluation_times=[1.0], num_shots=N)])).run().final_bitstrings` — `QutipEmulator.run()` is deprecated.

## 2. Devices — read programmatically (`python scripts/device_limits.py [name]`)

| | AnalogDevice | **WeightedAnalogDevice** (new in 1.9) | DigitalAnalogDevice | MockDevice |
|---|---|---|---|---|
| C6 (rad/µs·µm⁶) / Rydberg level | 865 723 / 60 | **12 241 414.5 / 75** | 5 420 158.5 / 70 | 5 420 158.5 / 70 |
| min atom distance / max radial | 5 / 38 µm | **5 / 80 µm** | 4 / 50 µm | 0 / ∞ |
| max atoms | 80 | **256** | 100 | ∞ |
| Ω_max | 2π·2 | **2π·2** (min avg amp 2π·0.3!) | 2π·2.5 | ∞ |
| \|δ\|_max (global) | 2π·20 | **2π·10** | 2π·20 | ∞ |
| max sequence duration / runs | 6000 ns / 2000 | **6000 ns / 500** | – / – | – |
| DMM (local detuning) | **none** | `dmm_0`: bottom −2π·10 per atom, total −2π·1000, min avg 2π·0.1 | `dmm_0`: −2π·20 / −2π·2000 | `dmm_0` unlimited |
| layout | required; 1 pre-calibrated triangular(61, 5µm) | required; **150–512 traps, filling 0.35–0.5** (not enforced at 150 traps) | required | no |
| mod bandwidth | 8 MHz (+EOM) | 50 MHz (DMM 22) | – | – |
| U at min distance | 55 rad/µs | **783 rad/µs** | 1324 rad/µs | – |
| blockade radius at Ω_max | 7.2 µm | 9.96 µm | 9.8 µm | – |

The organizers' device is read with `DeviceLimits.from_pulser_device(dev)` (`qopt/embed.py`) or `scripts/device_limits.py --json dev.json` / `--cloud FRESNEL_CAN1`. **If it has no DMM, diagonal QUBO terms can only be realised on average** (qopt reports `uses_dmm=False` and the diagonal error).

## 3. qubo-solver 0.9.1 — what it does end to end ([full digest](api/qubosolver.md))

`Instance(matrix.tensor(Q))` (torch, symmetric, cost = xᵀQx) → `Solver(instance, SolverConfig(solving=QuantumSolvingConfig(embedding=EmbeddingConfig(algorithm="blade"|"greedy_layout"), drive_shaping=DriveShapingConfig(algorithm="local_energy_scale"|"proportional_diagonal"|"bayesian_search", local_energy_scale_kappa=0.1, proportional_diagonal_kappa=1.0, bayesian_search_n_calls=20), backend=LocalEmulator(num_shots=N), device=qoolqit.AnalogDeviceWithDMM()), preprocessing=True, postprocessing=True)).solve()` → `Solution` (torch `bitstrings`, `costs`, `counts`, `probabilities`; `solution[0].string/.cost`).

1. **Preprocess**: Hansen variable fixing, then a GLPK bit-flip ILP to remove negative off-diagonals; whatever negative couplings remain are **zeroed** (approximation, logged) — 1/r⁶ is strictly positive. Hard cap **80 variables**.
2. **Embed**: **BLaDE** (force-directed fit of 1/r⁶ ∝ Q_ij, MDS warm start, dims (6,5,4,3,2,2,2)×200 steps, bounded by device max_radial/min_distance) or greedy placement on a triangular/square lattice. Returns a dimensionless `qoolqit.Register` (atom i = variable i).
3. **Drive**: target detunings d_i = −Q_ii/2 (dimensionless); Ω(t) = Interp[1e-9, Ω_max, Ω_max, 1e-9]; δ_g(t) = Interp[−max|d|, −max|d|, δ_g(T), δ_g(T)]; **DMM constant** = −(d_max−d_min) with weights w_i = (d_max−d_i)/(d_max−d_min), δ_g(T)=d_max (without DMM: δ_g(T)=mean d). `local_energy_scale`: Ω_max = κ·mean(|δ_i|+Σ|V_ij|), κ=0.1; `proportional_diagonal`: Ω_max = κ·max|d|. Ω clamped to the device amplitude, |d| rescaled to ≤ 10·Ω_max (κ must be ≥ 0.1 or compile fails).
4. **Compile** with qoolqit profile `MAX_ENERGY` (scale so Ω_max hits device max amplitude or min distance hits device min, whichever binds), duration pinned to ~6000 ns; **run** on auto-selected backend (qutip <15 atoms, emu-sv 15–25, emu-mps ≥26; `backend_type=BackendType.SVBackend` to force).
5. **Post-process**: greedy-sweep 1-flip local search on every sample (rewrites bitstrings, merges counts), lift transforms, sort by cost.

Functional pieces (used by `qopt/baseline.py` to get RAW counts + the register/drive): `embedding.blade.embed_for_device(inst, dev)`, `drive_shaping.local_energy_scale.build_drive(inst, reg, device=dev, dmm=True, kappa=…)`, `solving.analog_quantum_sampling.compile(reg, drive, dev, default_sequence_duration=6000)` → `program.compiled_sequence`, `backend.run(program).results().final_bitstrings`, `extract_qubo(reg, drive)` (the QUBO actually encoded → embedding error). Classical solvers: `solving.tabu_search/simulated_annealing/brute_force/cplex.solve(instance, ...)`.
Knobs: embedding algorithm, drive algorithm + κ, Bayesian n_calls, shots, backend, device, pre/post-processing flags, `torch.manual_seed`. Seeds for BLaDE go through numpy's global RNG.
Gotchas: default device is `AnalogDeviceWithDMM` (virtual AnalogDevice + DMM bottom −2π·20) — **not** WeightedAnalogDevice; wrap ours with `qoolqit.Device(pulser_device=WeightedAnalogDevice)`; emulator seed not exposed; dense penalty QUBOs embed badly (same physics limit as ours); the Bayesian drive search optimises *sampled minimum cost*, not P(optimum).

## 4. QoolQit 1.5 — dimensionless design → device ([digest](api/qoolqit.md))
`Register.from_coordinates`/`from_graph`, `Drive(amplitude=, detuning=, dmm=DetuningMapModulator(waveform≤0, weights))`, waveforms (`Constant/Ramp/Interpolated/PiecewiseLinear/Blackman`, `>>` to concatenate), `QuantumProgram(register, drive).compile_to(device, profile="default"|"max_energy")` → `program.compiled_sequence` (pulser). Units: distance unit = device min distance (r̃=1), energy unit J_max = C6/r_min⁶ (AnalogDevice: Ω̃_max 0.227, δ̃_max 2.27, T̃_max 332), `UnitConverter` invariants `time*energy = 1000`, `distance⁶*energy = C6`. Embedders: `InteractionEmbedder` (Nelder-Mead Σ(Q_ij − 1/r⁶)²), `Blade`, `SpringLayoutEmbedder` (unit-disk graphs). Execution: `LocalEmulator(backend_type=…, num_shots=…).run(program).results().final_bitstrings`, `RemoteEmulator`, `QPU`.
**Worth it when**: you want to think in dimensionless units and let `max_energy` saturate the device, or reuse BLaDE / qubo-solver pieces. **Not needed** for us: our `qopt.embed`/`qopt.schedule` work directly in physical units read from the device, which is simpler to reason about and to tune. Limits: one global channel + one DMM, no parametrized sequences, no optimiser.

## 5. Emulator ceilings ([digest](api/emulators.md))
| backend | exact? | atoms (laptop CPU) | notes |
|---|---|---|---|
| `QutipBackendV2` (pulser-simulation) | yes | ≤ 12–14 in seconds–minutes (`sampling_rate=0.1` helps ×10) | noise via `NoiseModel`; 16+ needs GBs |
| `emu_sv.SVBackend` | yes | 15–22 CPU (minutes), ~25 GPU | `SVConfig(dt=10, gpu=None)`; Lindblad noise halves the count |
| `emu_mps.MPSBackend` | approximate (TDVP) | 25–80+ (100 in 1D) | `MPSConfig(dt=10, precision=1e-5, max_bond_dim=…)`; 2D is mapped to a chain |
| cloud EMU_FREE / EMU_SV / EMU_MPS | – | ≤12 / ≤25 / ≤80 | async batches |

**When MPS is trustworthy**: 1D/quasi-1D registers, low-entanglement adiabatic sweeps, short quenches. **Not**: dense 2D, long highly-entangled evolutions — χ hits `max_bond_dim` and truncation error is uncontrolled. There is no "cap reached" exception: read `results.statistics[-1]["max_bond_dimension"]` (or the step log), and **converge**: rerun with `precision/10` and `2×max_bond_dim`, compare (qopt: `run.mps_convergence_check` → TV distance < 0.05), cross-check vs emu-sv at ≤ 20 atoms. Runtime per step ~ N²χ³; memory (CPU) ~ χ²N².
Common interface: `XBackend(seq, config=XConfig(observables=[BitStrings(evaluation_times=[1.0], num_shots=N)])).run()` → `Results.final_bitstrings` (Counter; bit i = `results.atom_order[i]` = register order; '1' = Rydberg). `evaluation_times` are fractions of the duration and must be multiples of `dt` for emu-*.

## 6. Cloud SDK job flow ([digest](api/pasqal_cloud.md))
```python
from pasqal_cloud import PasqalCloudConnection, RemoteMPSBackend   # RemoteSVBackend, RemoteEmuFreeBackend
conn = PasqalCloudConnection(username=os.environ["PASQAL_USERNAME"], password=os.environ["PASQAL_PASSWORD"], project_id=os.environ["PASQAL_PROJECT_ID"])
bk   = RemoteMPSBackend(seq, conn, config=RemoteMPSBackend.default_config.with_changes(observables=[BitStrings(num_shots=500)]), mimic_qpu=True)
rr   = bk.run(job_params=[{"runs": 500}], wait=False)        # async -> rr.batch_id, rr.job_ids   (qopt saves them in jobs/*.json)
rr.get_batch_status(); counts = rr.results[0].final_bitstrings   # when DONE   (or pulser.backend.RemoteResults(batch_id, conn))
qpu  = pulser.QPUBackend(seq, conn).run(job_params=[{"runs": 500}])   # needs reg.with_automatic_layout(device)
devs = conn.fetch_available_devices()                         # {"FRESNEL_CAN1": Device(...)} -> feed to DeviceLimits.from_pulser_device
```
Raw client (`pasqal_cloud.pasqal_cloud_client.PasqalCloudClient`): `create_batch(serialized_sequence=seq.to_abstract_repr(), jobs=[{"runs": N, "variables": {...}}], device_type=DeviceTypeName.EMU_MPS|EMU_SV|EMU_FREE|EMU_FRESNEL|FRESNEL_CAN1, backend_configuration=EmulationConfig.to_abstract_repr(), open=False, wait=False, tags=[...])` → `batch.id`, `batch.ordered_jobs[i].status ∈ PENDING|RUNNING|DONE|ERROR|CANCELED`, `job.result` = `{bitstring: count}`; `client.get_batch(id)` / `get_job(id)` to poll later; open batches (`open=True`, `batch.add_jobs`, `batch.close()`). Batch `DONE` ≠ every job DONE — check each job. Limits: ≤1000 runs/job and 6000 ns on FRESNEL_CAN1; 429s retried with Retry-After; `EmulatorType`, `SDK`, `emulator=`, `configuration=` are deprecated. **Credentials only from env** (`.env` is git-ignored; `.env.example` lists the names).

## 7. Breaking changes that bite old snippets (last 12 months)
`QutipEmulator.run()` → `QutipBackendV2` (1.9); `Chadoq2`/`IroiseMVP` gone; `pulser_pasqal.PasqalCloud` → `pasqal_cloud.PasqalCloudConnection`, `EmuMPSBackend/EmuSVBackend/EmuFreeBackendV2` → `RemoteMPSBackend/RemoteSVBackend/RemoteEmuFreeBackend`, `EmuTNBackend` removed; `Sequence.switch_device/switch_register` → `with_new_device/with_new_register`; `Waveform.change_duration` → `with_new_duration`; `SimConfig` → `NoiseModel`; `Device.default_noise_model` → `noise_model`; modulation transfer function corrected in 1.6.5 (√2 slower); qubo-solver 0.9: `QUBOInstance/QuboSolver/QuboSolution` → `Instance/Solver/Solution`, `DriveShapingConfig` has no `dmm=` field (DMM auto if the device has one); qoolqit 1.4 default compile profile became `"default"` (qubo-solver forces `max_energy`).
