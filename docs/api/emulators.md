# Emulator backends — code-level API digest

Source: `/home/claude/vendor/` — pulser-core 1.9.1, pulser_simulation 1.9.1, emu_sv 2.10.0, emu_mps 2.10.0, emu_base 2.10.0.
Everything below is quoted from the installed source; "not present" means it is not in this code.

## 0. The common pattern (all three emulators)

```python
from pulser.backend import BitStrings, EmulationConfig   # generic
backend = XBackend(sequence, config=XConfig(observables=[BitStrings(evaluation_times=[1.0], num_shots=1000)]))
results = backend.run()                                   # -> pulser.backend.Results
counts: dict[str, int] = results.final_bitstrings         # Counter[str]; same as results.get_result("bitstrings", 1.0)
```

* `XBackend(sequence, *, config=None, mimic_qpu=False)` — `pulser.backend.abc.EmulatorBackend.__init__(self, sequence: pulser.Sequence, *, config: EmulationConfig | None = None, mimic_qpu: bool = False)`. The config is merged over the backend's `default_config` (`validate_config` does `cls.config_type(**{**cls.default_config._backend_options, **config._backend_options})`), so any field you omit falls back to that backend's default.
* `EmulatorBackend.config_type` (classproperty) = `type(cls.default_config)`.
* The sequence must be built (not parametrized / register-mappable) and non-empty.

### `pulser.backend.config.EmulationConfig` (base of all configs)

```python
EmulationConfig(
    *,
    callbacks: Sequence[Callback] = (),
    observables: Sequence[Observable] = (),
    default_evaluation_times: Sequence[SupportsFloat] | Literal["Full"] = (1.0,),
    initial_state: StateType | None = None,      # default: all qudits in ground state
    with_modulation: bool = False,
    interaction_matrix: ArrayLike | None = None, # (N,N), (1,N,N) or (2,N,N) for XY
    prefer_device_noise_model: bool = False,
    noise_model: NoiseModel | None = None,       # None -> NoiseModel()
    n_trajectories: int | None = None,           # None -> 40 if prefer_device_noise_model else (noise_model.runs or 1)
    default_num_shots: int = 1000,               # used by BitStrings when num_shots is None
    **backend_options,
)
```

* Evaluation times are **relative** (0.0 = start, 1.0 = end), ascending, unique. `"Full"` means every emulation step.
* Warns if `observables` and `callbacks` are both empty ("results will be empty").
* Raises if two observables share a tag (use `tag_suffix`).
* Read-only: use `config.with_changes(**changes)` to get a modified copy.
* `config.is_evaluation_time(t, tol=1e-6)`; `to_abstract_repr()` / `from_abstract_repr()` for the cloud.
* `prefer_device_noise_model=True` + device noise model: device's `NoiseModel.runs` is ignored in favour of `n_trajectories` (warning).

Legacy `EmulatorConfig` (frozen dataclass) is still present: `backend_options: dict = {}`, `sampling_rate: float = 1.0`, `evaluation_times: float | Sequence[float] | Literal["Full","Minimal","Final"] = "Full"`, `initial_state: "all-ground" | array = "all-ground"`, `with_modulation: bool = False`, `prefer_device_noise_model: bool = False`, `noise_model: NoiseModel = NoiseModel()`. Only used by the deprecated `QutipBackend` (V1).

### Observables (`pulser.backend.default_observables`)

All take keyword-only `evaluation_times: Sequence[float] | None = None` (None -> config's `default_evaluation_times`), `tag_suffix: str | None = None`, `default_aggregation_method`.

| class | tag (`_base_tag`) | extra ctor args | value stored |
|---|---|---|---|
| `BitStrings` | `"bitstrings"` | `num_shots: int \| None = None`, `one_state: Eigenstate \| None = None` | `Counter[str]` |
| `StateResult` | `"state"` | — | backend State (deepcopy) |
| `Fidelity(state, *, ...)` | `"fidelity"` | positional `state: State` | scalar |
| `Expectation(operator, *, ...)` | `"expectation"` | positional `operator: Operator` | scalar |
| `CorrelationMatrix` | `"correlation_matrix"` | `one_state` | `list[list]` of <n_i n_j> |
| `Occupation` | `"occupation"` | `one_state` | `list` of <n_i> |
| `Energy` | `"energy"` | — | <H(t)> |
| `EnergyVariance` | `"energy_variance"` | — | scalar |
| `EnergySecondMoment` | `"energy_second_moment"` | — | scalar |

`BitStrings.apply` calls `state.sample(num_shots=self._num_shots or config.default_num_shots, one_state=self.one_state, p_false_pos=config.noise_model.p_false_pos, p_false_neg=config.noise_model.p_false_neg)` — SPAM readout errors come from the config's noise model. The tag with a suffix is `f"bitstrings_{tag_suffix}"` (e.g. `BitStrings(tag_suffix="x").tag == "bitstrings_x"`). Accessing `BitStrings.num_shots` when it is `None` emits a RuntimeWarning (the 1.7 default change).

### `pulser.backend.results.Results`

```python
@dataclass
class Results:
    atom_order: tuple[str, ...]   # order of qubits in bitstrings/states
    total_duration: int           # ns
```

* `results.get_result(observable_or_tag: Observable | str, time: float) -> Any` — exact relative time match (`list.index`), raises `ValueError` if absent.
* `results.get_result_times(observable_or_tag) -> list[float]`
* `results.get_result_tags() -> list[str]`
* `results.get_tagged_results() -> dict[str, list[Any]]`
* `results.<tag>` via `__getattr__` -> `list` of values, one per evaluation time (e.g. `results.bitstrings[-1]`, `results.occupation`).
* `results.final_bitstrings -> dict[str, int]` (= `get_result("bitstrings", time=1.0)`; RuntimeError if missing).
* `results.final_state -> State` (= `get_result("state", 1.0)`).
* `results.bitstring_counts` — deprecated alias to `final_bitstrings` (FutureWarning).
* `Results.from_final_bitstrings(atom_order, total_duration, final_bitstrings: Mapping[str,int])` — builds a Results holding a counter at t=1.0.
* `Results.aggregate(results_seq, **aggregation_functions)` — averages observables across trajectories; `BitStrings` default aggregation is `BAG_UNION` (counters are summed), `StateResult`/`EnergyVariance` default to `SKIP_WARN`.
* `to_abstract_repr()` / `from_abstract_repr(json_str)`.
* `ResultsSequence` is a `typing.Sequence[Results]`; `RemoteResults` (remote.py) subclasses it.

### Bitstring ordering convention

* Bitstring position i = `results.atom_order[i]` = `sequence.register.qubit_ids[i]` (register insertion order). All three emulators set `atom_order=tuple(register.qubit_ids)` / `data.qubit_ids`.
* `'1'` = the `one_state`; inferred via `State.infer_one_state()`: `{"r","g"} -> "r"`, `{"0","1"} -> "1"`, `{"g","h"} -> "h"`, `{"u","d"} -> "d"`. So in the Rydberg (ground-rydberg) basis `'1'` = Rydberg, `'0'` = ground.
* emu-sv: `index_to_bitstring(n, idx) = format(idx, f"0{n}b")` with eigenstates `("r","g")` i.e. index 0 of each qubit = `r`, so the most significant bit is qubit 0 → leftmost char = first qubit. emu-mps: `"".join("1" if x == 1 else "0" for x in outcome)` over sites 0..N-1 (left = site 0); `one_state` must be `None | "r" | "1"`. If `optimize_qubit_ordering=True` emu-mps internally permutes qubits and un-permutes bitstrings (`permute_bitstrings`) before returning, so the output order is still register order.
* qutip: `QutipState.bitstring_probabilities(one_state=...)` replaces `one_state` with `"1"` and the other eigenstates with `"0"` in the basis label string (qubit 0 leftmost).

### `with_modulation`

Config field `with_modulation: bool = False`. qutip: `QutipEmulator.from_sequence(seq, with_modulation=...)` samples with `modulation=True` and `extended_duration=sequence.get_duration(include_fall_time=True)` (NotImplementedError when combined with an SLM mask). emu-sv/emu-mps: `emu_base.pulser_adapter` uses `sequence.get_duration(include_fall_time=config.with_modulation)` and `HamiltonianData.from_sequence(sequence, with_modulation=config.with_modulation, ...)`.

---

## 1. pulser_simulation (QuTiP)

Exports (`pulser_simulation/__init__.py`): `EmulatorConfig, NoiseModel, QutipState, QutipOperator, QutipConfig, QutipBackend, QutipBackendV2, QutipEmulator, SimConfig, Solver, density_matrix_aggregator`.

### `QutipBackendV2(EmulatorBackend)` — current API

```python
QutipBackendV2(sequence: pulser.Sequence, *, config: EmulationConfig | None = None, mimic_qpu: bool = False)
default_config = QutipConfig(observables=[BitStrings(evaluation_times=[1.0]), StateResult()])
run(self) -> Results
QutipBackendV2.run_from_sequence_samples(sequence_samples, register, device, *, config=None) -> Results   # staticmethod
```

### `QutipConfig(EmulationConfig[QutipState])`

```python
QutipConfig(
    *,
    sampling_rate: float = 1.0,                 # (0, 1]
    solver: Solver | Literal["default", "MasterEquation", "MonteCarlo"] = Solver.DEFAULT,
    print_progress: bool = False,
    progress_bar: bool = False,
    **backend_options,                          # all EmulationConfig fields
)
```
* `_enforce_expected_kwargs = True` → unknown kwargs raise `ValueError`.
* `interaction_matrix` → `NotImplementedError` ("does not handle custom interaction matrices").
* `initial_state` must be a `QutipState`.
* Warns if `noise_model.samples_per_run not in [None, 1]` (ignored).
* `state_type = QutipState`, `operator_type = QutipOperator`.
* `Solver` (str Enum): `DEFAULT = "default"`, `MESOLVER = "MasterEquation"`, `MCSOLVER = "MonteCarlo"`. No collapse operators → `qutip.sesolve` regardless.

Noise: with stochastic noise (shot-to-shot params or SPAM state_prep_error != 0) the backend runs `n_trajectories` noisy Hamiltonians and returns `Results.aggregate(...)` (bitstring counters joined, states combined by `density_matrix_aggregator`).

### Legacy `QutipEmulator` (deprecated as of 1.9, still present)

```python
QutipEmulator.from_sequence(
    sequence: Sequence,
    sampling_rate: float = 1.0,
    config: SimConfig | None = None,            # deprecated, use noise_model
    evaluation_times: float | str | ArrayLike = "Full",   # "Full" | "Minimal" | list of µs | float sampling rate
    with_modulation: bool = False,
    noise_model: NoiseModel | None = None,
    solver: Solver = Solver.DEFAULT,
    n_trajectories: int | None = None,
) -> QutipEmulator
QutipEmulator.run(self, progress_bar: bool = False, print_progress: bool = False, **qutip_options) -> SimulationResults  # emits DeprecationWarning
```
`SimulationResults` (`CoherentResults` | `NoisyResults`, a `ResultsSequence`):
* `sample_final_state(self, N_samples: int = 1000) -> Counter`
* `sample_state(self, t: float, n_samples: int = 1000, t_tol: float = 1e-3) -> Counter` (t in µs)
* `get_final_state()`, `get_state(t)`, `states`, `expect(obs_list)`, `plot(op)`.
Also `set_evaluation_times`, `set_initial_state`, `get_hamiltonian(time)`, `build_operator(...)`, `draw(...)`.

### Legacy `QutipBackend(Backend)` — deprecated in 1.6
`QutipBackend(sequence, config: EmulatorConfig = EmulatorConfig(), mimic_qpu=False)`; `run(progress_bar=False, **qutip_options) -> SimulationResults`.

### Minimal snippet (qutip)

```python
from pulser.backend import BitStrings
from pulser_simulation import QutipBackendV2, QutipConfig

cfg = QutipConfig(observables=[BitStrings(evaluation_times=[1.0], num_shots=1000)])
results = QutipBackendV2(seq, config=cfg).run()
counts: dict[str, int] = dict(results.final_bitstrings)       # Counter -> dict; '1' = Rydberg, index = register order

# legacy
from pulser_simulation import QutipEmulator
counts = dict(QutipEmulator.from_sequence(seq).run().sample_final_state(1000))
```

---

## 2. emu_sv (state vector / density matrix, torch)

Exports (`emu_sv/__init__.py`): `BitStrings, CorrelationMatrix, DenseOperator, Energy, EnergySecondMoment, EnergyVariance, Expectation, Fidelity, Occupation, Results, SVBackend, SVConfig, StateResult, StateVector, inner, DensityMatrix, SparseOperator, Solver`.

### `SVBackend(EmulatorBackend)`

```python
SVBackend(sequence, *, config: EmulationConfig | None = None, mimic_qpu: bool = False)
default_config = SVConfig(observables=[BitStrings(evaluation_times=[1.0])])
run(self) -> Results | list[Results]     # in practice always `Results.aggregate(results)` -> Results
```
* `assert self._sequence._in_ising, "Only Rydberg interaction is currently supported"` (no XY).
* Builds `PulserData(sequence=..., config=..., dt=config.dt)` and iterates `get_sequences()` (one `SequenceData` per noise trajectory), running `SVBackendImpl(config, sequence_data)._run()` for each, then aggregates.

### `SVConfig(EmulationConfig)`

```python
SVConfig(
    *,
    dt: float = 10.0,                 # ns timestep; observables only computed when evaluation_times fall on the dt grid (plus explicit obs times are added to the grid)
    max_krylov_dim: int = 100,
    krylov_tolerance: float = 1e-10,
    gpu: bool | None = None,          # None -> backend chooses 1 GPU (impl: requested_gpu=True if None)
    interaction_cutoff: float = 0.0,  # |U_ij| below this set to 0
    log_level: int = logging.INFO,    # logging.WARN silences per-step info
    log_file: pathlib.Path | None = None,
    solver: Solver = Solver.DEFAULT,
    **kwargs,                         # EmulationConfig fields
)
```
* `_enforce_expected_kwargs = True`. `state_type = DenseState`, `operator_type = MatrixOperator`.
* Warns (via logger) that `noise_model.runs` / `samples_per_run` are ignored.
* `emu_sv.Solver` (str Enum): `DEFAULT = "default"`, `MONTECARLO = "montecarlo"`, `LINDBLAD = "lindblad"`.

### Density-matrix / noise behaviour (`sv_backend_impl.py`)
```python
if data.lindblad_ops:                                   # any Lindblad noise (dephasing/relaxation/depolarizing/eff_noise/leakage)
    if (config.solver == Solver.DEFAULT and data.has_stochastic_noise) or config.solver == Solver.MONTECARLO:
        stepper = EvolveMonteCarlo();  state_type = StateVector         # quantum-jump trajectories
    else:
        stepper = EvolveDensityMatrix(); state_type = DensityMatrix     # full Lindblad master equation
else:
    stepper = EvolveStateVector();  state_type = StateVector
```
* `has_stochastic_noise` = shot-to-shot noise other than SPAM, or `"SPAM" in noise_types and state_prep_error != 0`.
* `DensityMatrix.sample` uses `abs(diagonal)`; `StateVector.sample` uses `|ψ|²`; both via `torch.multinomial`, then `apply_measurement_errors(p_false_pos, p_false_neg)`.
* Raises `NotImplementedError` if both `initial_state` and `state_prep_error > 0`.
* Diagnostics: an internal `Statistics` observable (tag `"statistics"`, aggregation SKIP) logs `step = i/n, RSS = … MB, Δt = … s` at INFO per step and stores `{"RSS": …, "duration": …}`.
* No "max bond dim" concept in emu-sv (not present).

### Minimal snippet (emu-sv)

```python
import logging
from emu_sv import SVBackend, SVConfig, BitStrings

cfg = SVConfig(observables=[BitStrings(evaluation_times=[1.0], num_shots=1000)],
               dt=10.0, gpu=False, log_level=logging.WARN)
results = SVBackend(seq, config=cfg).run()
counts: dict[str, int] = dict(results.get_result("bitstrings", 1.0))   # or results.final_bitstrings / results.bitstrings[-1]
```

---

## 3. emu_mps (matrix product states, torch)

Exports (`emu_mps/__init__.py`): `MPO, MPS, inner, MPSConfig, Solver, MPSBackend, StateResult, BitStrings, Occupation, CorrelationMatrix, Expectation, Fidelity, Energy, EnergyVariance, EnergySecondMoment, EntanglementEntropy`.

### `MPSBackend(EmulatorBackend)`

```python
MPSBackend(sequence, *, config: EmulationConfig | None = None, mimic_qpu: bool = False)
default_config = MPSConfig(observables=[BitStrings(evaluation_times=[1.0])])
run(self) -> Results                          # Results.aggregate over noise trajectories; results un-permuted via impl.permute_results
MPSBackend.resume(autosave_file: str | pathlib.Path) -> Results   # staticmethod; unpickles impl (trust the file)
```

### `MPSConfig(EmulationConfig)`

```python
MPSConfig(
    *,
    dt: float = 10.0,
    precision: float = DEFAULT_PRECISION,          # 1e-5 (truncation error threshold)
    max_bond_dim: int = DEFAULT_MAX_BOND_DIM,      # 1024
    max_krylov_dim: int = 100,
    extra_krylov_tolerance: float = 1e-3,          # Lanczos tol = precision * extra_krylov_tolerance (clamped so product >= 1e-12, with a logged warning)
    gpu: bool | None = None,                       # None -> use GPU iff DEVICE_COUNT > 0; True w/o GPU -> warning + CPU
    num_gpus_to_use: int | None = None,            # DEPRECATED -> gpu = bool(num_gpus_to_use); error if both given
    optimize_qubit_ordering: bool = True,          # reorders qubits internally; auto-disabled (with warning) if observables other than
                                                   #   {bitstrings, occupation, correlation_matrix, statistics, energy, energy_variance, energy_second_moment}
    interaction_cutoff: float = 0.0,
    log_level: int = logging.INFO,
    log_file: pathlib.Path | None = None,
    autosave_prefix: str = "emu_mps_save_",
    autosave_dt: float = float("inf"),             # seconds; must be > 10
    solver: Solver = Solver.TDVP,
    **kwargs,                                      # EmulationConfig fields
)
```
* `_enforce_expected_kwargs = True`. `state_type = MPS`, `operator_type = MPO`.
* `emu_mps.Solver` (str Enum): `TDVP = "tdvp"`, `DMRG = "dmrg"` (DMRG adiabatically follows the ground state).
* Warns (logger) that `noise_model.runs` / `samples_per_run` are ignored.

### Bond-dimension / truncation diagnostics
* Truncation: `truncate_impl(factors, precision, max_bond_dim)` → `split_matrix(..., max_error=precision, max_rank=max_bond_dim)`; silently caps the rank. There is **no explicit "max bond dim reached" warning** in the code (not present).
* Diagnostics come from the internal `Statistics` observable (tag `"statistics"`, aggregation SKIP), evaluated every step and logged at INFO: `step = i/n, χ = <state.get_max_bond_dim()>, |ψ| = … MB, RSS = … MB, Δt = … s`. The stored dict per step is `{"max_bond_dimension": int, "memory_footprint": float, "RSS": float, "duration": float}` (+ `"RSS_cpu"` on GPU). Retrieve after a run with `results.statistics` (list) or `results.get_result("statistics", t)`; the final χ is `results.statistics[-1]["max_bond_dimension"]`. If it equals `max_bond_dim`, truncation was rank-limited.
* `MPS.get_max_bond_dim()`, `MPS.get_memory_footprint()` are public on the state (`StateResult` is not permutable, so it forces `optimize_qubit_ordering=False`).
* Other logged warnings: "The state is not normalized, normalizing it for you." (from `MPS.from_state_amplitudes`), "Requested to run on GPU but none is available, running on CPU".

### Minimal snippet (emu-mps)

```python
import logging
from emu_mps import MPSBackend, MPSConfig, BitStrings

cfg = MPSConfig(observables=[BitStrings(evaluation_times=[1.0], num_shots=1000)],
                dt=10.0, precision=1e-6, max_bond_dim=400, gpu=False, log_level=logging.WARN)
results = MPSBackend(seq, config=cfg).run()
counts: dict[str, int] = dict(results.final_bitstrings)
chi_final = results.statistics[-1]["max_bond_dimension"]   # truncation diagnostic
```

---

## 4. emu_base.pulser_adapter (shared by emu_sv / emu_mps)

* `PulserData(*, sequence: pulser.Sequence, config: EmulationConfig, dt: float)` — attributes: `qubit_ids`, `qubit_count`, `target_times: list[float]` (absolute ns; uniform `dt` grid ∪ observable times, always ending at the full duration), `noise_model` (device default if `prefer_device_noise_model`), `has_stochastic_noise`, `hamiltonian_type: HamiltonianType.{Rydberg, XY}`, `lindblad_ops`, `full_interaction_matrix` (from `config.interaction_matrix` if set), `interaction_cutoff`, `slm_end_time`, `eigenstates`, `dim`.
* `PulserData.get_sequences() -> Iterator[SequenceData]` — yields one `SequenceData` per noise trajectory (`samples.reps` repeats); each carries `omega, delta, phi` tensors sampled at `target_times`, an `interaction_matrix(t)` callable (with SLM-masked qubits zeroed before `slm_end_time` and entries below `interaction_cutoff` zeroed), `bad_atoms`, `lindblad_ops`, `state_prep_error`, `target_times`, `eigenstates`, `hamiltonian_type`, `has_stochastic_noise`.
* `is_evaluation_time(config, observable, t, tolerance=1e-6)`; `_get_target_times` uses `sequence.get_duration(include_fall_time=config.with_modulation)`.

## 5. Remote pieces in pulser-core (`pulser/backend/remote.py`, `qpu.py`)

* `JobParams(TypedDict, total=False)`: `runs: int`, `variables: dict[str, Any]`.
* `BatchStatus` Enum: `PENDING, RUNNING, DONE, CANCELED, TIMED_OUT, ERROR, PAUSED`. `JobStatus` Enum: `PENDING, RUNNING, DONE, CANCELED, ERROR, PAUSED`.
* `RemoteResults(batch_id: str, connection: RemoteConnection, job_ids: list[str] | None = None)` — `ResultsSequence`; `.results -> tuple[Results,...]` (lazy fetch, raises `RemoteResultsError` if any job lacks results), `.batch_id`, `.job_ids`, `.get_batch_status() -> BatchStatus`, `.get_available_results() -> dict[job_id, Results]`. Index `remote_results[0].final_bitstrings`.
* `RemoteConnection(ABC)`: abstract `submit(sequence, wait=False, open=False, batch_id=None, **kwargs) -> RemoteResults`, `_fetch_result`, `_query_job_progress`, `_get_batch_status`, `supports_open_batch()`; concrete `fetch_available_devices()` (NotImplementedError by default), `update_sequence_device(sequence)`.
* `RemoteBackend(sequence, connection, mimic_qpu=False, *, config: BackendConfig | None = None)`; `run(job_params: list[JobParams] | None = None, wait: bool = False) -> RemoteResults`; `open_batch()` context manager.
* `QPUBackend(sequence, connection, *, config: BackendConfig | None = None)` = `RemoteBackend` with `mimic_qpu=True`; `run(job_params=None, wait=False)` fills `runs` from `config.default_num_shots` when missing; requires `runs <= device.max_runs`.
* `BackendConfig(*, default_num_shots: int | None = None, **backend_options)` — base of `EmulationConfig`.
* **pulser-pasqal is not installed** (no `pulser_pasqal` package in `/home/claude/vendor`). The `PasqalCloud` connection class lives in `pasqal_cloud` 0.24.0 as `PasqalCloudConnection` (see `pasqal_cloud.md`).
