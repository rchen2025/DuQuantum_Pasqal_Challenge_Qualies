# qoolqit 1.5.0 — code-level API digest

Source: `/home/claude/vendor/qoolqit/` (`__version__ = "1.5.0"`). Depends on pulser-core 1.9.1, pulser_simulation, pasqal_cloud (for `qoolqit.execution`), networkx, scipy, matplotlib, optional torch.
Everything below is quoted from the installed source; "not present" means not in this code.

Top-level exports (`qoolqit/__init__.py`): `BlackmanWaveform, ConstantWaveform, DelayWaveform, InterpolatedWaveform, PiecewiseLinearWaveform, RampWaveform, Drive, Register, QuantumProgram, SequenceCompiler, available_default_devices, AnalogDevice, AnalogDeviceWithDMM, DigitalAnalogDevice, MockDevice, Device`. Short aliases `Blackman, Constant, Delay, Interpolated, PiecewiseLinear, Ramp` and `qoolqit.DataGraph` still work but emit `DeprecationWarning` (removed in v2). `qoolqit.execution` and `qoolqit.embedding`/`qoolqit.graphs` are separate subpackages.

## 1. The dimensionless model

Hamiltonian (Drive docstring): `H(t) = Σᵢ [Ω(t)/2 (cos φ σˣᵢ − sin φ σʸᵢ)] − Σᵢ [δ(t) + εᵢ Δ(t)] nᵢ + Σ_{i<j} 1/rᵢⱼ⁶ nᵢ nⱼ` with the interaction coefficient fixed to 1 (`Register.interactions()` returns `1/r**6`). Everything is unitless: time·energy = 1, energy·distance⁶ = 1. Compilation picks the physical scale (§4).

### `Register` (`qoolqit/register.py`)

```python
Register(qubits: Mapping[str | int, Sequence[float] | np.ndarray | torch.Tensor])   # {qid: (x, y)}; non-empty; coords shape (2,)
Register.from_coordinates(coords)        # qubit ids = 0..N-1 ints; coords list[(x,y)] | ndarray (N,2) | tensor (N,2); a Mapping raises TypeError
Register.from_graph(graph: DataGraph)    # requires graph.has_coords
Register.triangular(rows, atoms_per_row, spacing=1.0); .rectangular(rows, cols, row_spacing=1.0, col_spacing=1.0)
Register.square(n, spacing=1.0); .line(n, spacing=1.0); .circle(n, spacing=1.0)
```
Properties/methods: `qubits -> dict[qid, coord]`, `qubits_ids -> tuple`, `n_qubits`, `len(reg)`, `distances() -> dict[(i,j), r]`, `min_distance()`, `radial_distances()`, `max_radial_distance()`, `interactions() -> dict[(i,j), 1/r^6]`, `interaction_matrix() -> ndarray|Tensor` (0 on diagonal), `draw(...)`. Qubit ids are kept in insertion order; compilation stringifies them (`str(q)`) for the pulser register, so bitstring index i corresponds to `reg.qubits_ids[i]`. Torch coordinates are supported (differentiable `interaction_matrix`).

### Waveforms (`qoolqit/waveforms/`)

Base `Waveform(duration: float, **params)` — abstract `function(t)`, `max()`, `min()`, `__mul__(scalar)`, `_to_pulser(duration: int)`. Concrete:
```python
DelayWaveform(duration)                         # zero
ConstantWaveform(duration, value)
RampWaveform(duration, initial_value, final_value)
BlackmanWaveform(duration, area)                # max() = area/(0.42*duration)
PiecewiseLinearWaveform(durations: list[N], values: list[N+1])   # CompositeWaveform of N RampWaveforms, N >= 2, no zero-length segment
InterpolatedWaveform(duration, values: ArrayLike, times: ArrayLike | None = None)   # PCHIP; times are fractions in [0,1]
CompositeWaveform(*waveforms)                   # sequential concatenation; .waveforms, .durations, .times, .n_waveforms
```
Arithmetic that exists: `wf * c` / `c * wf` (scalar rescale, returns same class), `wf1 >> wf2` (→ `CompositeWaveform`), `wf(t)` for float or array/list evaluation (0 outside `[0, duration]`), `.duration`, `.params`, `.draw(...)`. **Not present**: `+`, `-`, `/`, waveform×waveform.

### `Drive` (`qoolqit/drive.py`)

```python
Drive(*, amplitude: Waveform, detuning: Waveform | None = None, dmm: DetuningMapModulator | None = None, phase: float = 0.0)
```
* `amplitude.min() >= 0` required; `detuning=None` → `DelayWaveform(amplitude.duration)`; mismatched durations are padded with a `DelayWaveform` (internally `_amplitude`/`_detuning`; `.amplitude`/`.detuning` return the originals).
* `phase` is reduced mod 2π and stored as `ConstantWaveform(duration, phase)` → `.phase.value`.
* `.duration`, `.amplitude`, `.detuning`, `.dmm`, `.phase`, `.draw(fig=None)`.
* `drive1 >> drive2` concatenates (NotImplementedError if either has a DMM or phases differ).

```python
@dataclass(frozen=True)
class DetuningMapModulator:
    waveform: Waveform          # must be <= 0 everywhere (waveform.max() <= 0)
    weights: dict[Any, float]   # qid -> weight in [0, 1]
```

### `QuantumProgram` (`qoolqit/program.py`)

```python
QuantumProgram(register: Register, drive: Drive)
.register, .drive, .is_compiled, .compiled_sequence -> pulser.Sequence   # ValueError if not compiled
.compile_to(device: Device,
            profile: Literal["default", "max_energy"] | CompilerProfile = CompilerProfile.DEFAULT,
            device_max_duration_ratio: float | None = None) -> None
.draw(compiled: bool = False, fig=None)
```
* Constructor checks every DMM weight key is a register qubit id.
* `compile_to` raises `TypeError` (not a qoolqit Device), `ValueError` (bad ratio / device has no max duration), `qoolqit.exceptions.CompilationError` ("The device does not support DMM…", or any device-constraint violation).
* **Not present**: `program.run(...)`, `program.compile(...)`, `program.run(runs=...)`. Execution is done by backends in `qoolqit.execution` (§5).

## 2. Devices (`qoolqit/devices/device.py`)

`Device(pulser_device: pulser BaseDevice, default_converter: UnitConverter | None = None)` — wraps a pulser device, reads `interaction_coeff` (C6), the `"rydberg_global"` channel (`clock_period`, `max_amp`, `max_abs_detuning`), `max_sequence_duration`, `min_atom_distance`, `max_radial_distance`, `requires_layout`.
* `_upper_amp = max_amp or 4π`; `_lower_distance = min_atom_distance or 5.0`; `_energy_ratio = upper_amp * lower_distance**6 / C6`.
* `.converter: UnitConverter`, `.reset_converter()`, `.set_energy_unit(energy)`, `.set_distance_unit(distance)`, `.specs -> dict` (`max_duration, max_amplitude, max_abs_detuning, min_distance, max_radial_distance` in **dimensionless** units for the current converter), `.name`, `.info()`, `Device.from_connection(connection: RemoteConnection, name: str)`.
* `._device` → the underlying pulser device.

Built-ins:
| class | wraps |
|---|---|
| `MockDevice()` | `pulser.MockDevice` |
| `AnalogDevice()` | `pulser.AnalogDevice` |
| `AnalogDeviceWithDMM()` | `pulser.AnalogDevice.to_virtual()` with one `DMM(clock_period=4, min_duration=16, max_duration=6000, mod_bandwidth=8, bottom_detuning=-2π·20, total_bottom_detuning=-2π·20)`, name `"AnalogDeviceWithDMM"` |
| `DigitalAnalogDevice()` | `pulser.DigitalAnalogDevice` |
`available_default_devices()` prints the first three. **`WeightedAnalogDevice` is not present.**

### `UnitConverter` (`qoolqit/devices/unit_converter.py`)

```python
@dataclass
class UnitConverter: C6: float; time: float; energy: float; distance: float
# invariants: time * energy == 1000  (pulser ns vs rad/µs)  and  distance**6 * energy == C6
UnitConverter.from_time(C6, time) | .from_energy(C6, energy) | .from_distance(C6, distance)
.factors -> (time, energy, distance)  (settable); .factors_from_time/energy/distance(x) -> tuple
```
Default converter for a device: `UnitConverter.from_distance(C6, lower_distance)` — i.e. dimensionless distance 1 = the device's minimum atom distance, energy unit = C6/r_min⁶, time unit = 1000/energy ns.

## 3. Compilation (`qoolqit/execution/compilation_functions.py`, `sequence_compiler.py`)

```python
class CompilerProfile(Enum):
    DEFAULT = "default"
    MAX_ENERGY = "max_energy"
    WORKING_POINT = DEFAULT      # deprecated alias
```
`basic_compilation(register, drive, device, profile, device_max_duration_ratio=None) -> pulser.Sequence`:
* **DEFAULT**: `TIME, ENERGY, DISTANCE = device.converter.factors`; then `_validate_program_default_profile` checks `drive.amplitude.max() <= specs["max_amplitude"]`, `max|detuning| <= specs["max_abs_detuning"]`, `drive.duration <= specs["max_duration"]` (skipped if `device_max_duration_ratio` set), `register.min_distance() >= specs["min_distance"]`, `register.max_radial_distance() <= specs["max_radial_distance"]` → `CompilationError` otherwise.
* **MAX_ENERGY**: compares `program_energy_ratio = drive.amplitude.max() * register.min_distance()**6` to `device._energy_ratio`. If larger (or 1 qubit) → `ENERGY = device._target_amp / drive.amplitude.max()` (amplitude hits device max); else → `DISTANCE = device._target_dist / register.min_distance()` (min pair distance hits device min). Other factors follow from the invariants. Then validates detuning, duration, radial distance after rescaling.
* `device_max_duration_ratio` (0,1] overrides `TIME = ratio * device._max_duration / drive.duration`.
* Waveform conversion `_WaveformConverter`: pulser duration = `round(wf.duration * TIME)` rounded **up** to a multiple of the channel clock period; pulser waveform = `wf._to_pulser(duration) * ENERGY`.
* Register: `pulser.Register({str(q): (DISTANCE*x, DISTANCE*y)})`; if the pulser device is a real `Device` with `requires_layout`, `with_automatic_layout(device=...)` is applied.
* Sequence: `Sequence(reg, device._device)`, `declare_channel("rydberg", "rydberg_global")`, `add(Pulse(amp_wf, det_wf, drive.phase.value), "rydberg")`.
* **DMM / local detuning: supported.** If `drive.dmm is not None`, `_DMMAdder` does `reg.define_detuning_map(detuning_weights={str(q): w})`, `seq.config_detuning_map(detuning_map, dmm_id=<first key of device._device.dmm_channels>)`, `seq.add_dmm_detuning(converted_waveform, dmm_id)`. Requires a device with a DMM channel (`AnalogDeviceWithDMM`, `MockDevice`, `DigitalAnalogDevice`); `AnalogDevice` has none → `CompilationError`. Only one DMM channel is used.

`SequenceCompiler(register, drive, device, profile, device_max_duration_ratio=None).compile_sequence() -> pulser.Sequence` wraps the above, re-raising non-`CompilationError` exceptions as `CompilationError("Failed to compile the sequence due to:\n\n...")`.

## 4. Execution (`qoolqit/execution/`)

`qoolqit.execution.__all__`: `LocalEmulator, RemoteEmulator, QPU, EmulationConfig, BitStrings, CorrelationMatrix, Energy, EnergySecondMoment, EnergyVariance, Expectation, Fidelity, Occupation, StateResult, Results, BackendType, Job, JobStatus, get_batch_id, retrieve_remote_job`. `BackendType` is just `pulser.backends` (so `BackendType.QutipBackendV2`, and — if installed — `BackendType.SVBackend`, `BackendType.MPSBackend`, `BackendType.RemoteEmuFreeBackend`, …).

```python
LocalEmulator(*, backend_type: type[EmulatorBackend] = QutipBackendV2, emulation_config: EmulationConfig | None = None, num_shots: int | None = None)
    .run(program: QuantumProgram) -> Job[Results]          # synchronous; instantiates backend_type(program.compiled_sequence, config=...)
RemoteEmulator(*, backend_type: type[RemoteEmulatorBackend] = RemoteEmuFreeBackend, connection: RemoteConnection, emulation_config=None, num_shots=None)
    .run(program) -> Job[Results]                           # async; one job in a new batch; backend_type(seq, connection=..., config=...).run(wait=False)
QPU(*, connection: RemoteConnection, num_shots: int | None = None)   # num_shots REQUIRED (ValueError); uses QPUBackend + BackendConfig(default_num_shots=num_shots)
    .run(program) -> Job[Results]
```
* `PulserEmulatorBackend.validate_emulation_config`: if `emulation_config is None` → `EmulationConfig(observables=(BitStrings(num_shots=num_shots),))`; otherwise appends `BitStrings(num_shots=num_shots)` if no `BitStrings` present; if `num_shots` given, sets `default_num_shots=num_shots`. So bitstrings are **always** sampled.
* `Job` (abstract, `qoolqit/execution/job.py`): `get_status() -> JobStatus` (= `pulser.backend.remote.JobStatus`), `job_id() -> str`, `results(timeout: float = math.inf) -> Results` (blocks; raises `TimeoutError`, `JobFailedError`, `JobCancelledError`), `cancel()`, `has_ended() -> bool` (DONE/ERROR/CANCELED). `_LocalJob` is already DONE; `_RemoteJob(connection, job_id, *, batch_id="")` polls `connection._query_job_progress(batch_id)`.
* `retrieve_remote_job(connection, job_id, *, batch_id="") -> Job[Results]`; `get_batch_id(job) -> str`.
* Return type of a run is a pulser `Results`: counts = `job.results().final_bitstrings` (`Counter[str]`) or `results.get_result("bitstrings", 1.0)`; `results.atom_order` = `str(q)` for each register qubit in order; `'1'` = Rydberg.
* **Not present**: `program.run(runs=...)`; `QutipBackend`/`emu-mps` choice inside `QuantumProgram`.

## 5. Graphs (`qoolqit/graphs/`)

`BaseGraph(nx.Graph)` and `DataGraph(BaseGraph)`. Constructors: `from_nodes(nodes)`, `from_coordinates(coords: list | dict)` (list → ids 0..N-1 with `"pos"` attr), `from_nx(g)`, `from_matrix(M)` (symmetric; diagonal → `node_weights` (None if all ~0), off-diagonal |M_ij| ≥ 1e-7 → edge with `edge_weights`), plus `DataGraph.line/circle/random_er/triangular/hexagonal/heavy_hexagonal/square/random_ud/from_pyg`.
Key API: `has_coords`, `coords` (get/set dict or list), `node_weights`, `edge_weights`, `has_node_weights`, `has_edge_weights`, `to_matrix()`, `distances(edge_list=None)`, `interactions()`, `interaction_matrix()`, `min_distance(connected=None)`, `max_distance(...)`, `ud_radius_range()`, `is_ud_graph()`, `ud_edges(radius, tol=1e-7)`, `set_ud_edges(radius)`, `rescale_coords(*, scaling=None, spacing=None)`, `draw(ax=None, **kw)`. Utilities: `all_node_pairs, distances, random_coords, random_edge_list, scale_coords, space_coords`.

## 6. Embedding (`qoolqit/embedding/`)

`__all__`: `BaseEmbedder, EmbedderConfig, GraphToGraphEmbedder, MatrixToGraphEmbedder, InteractionEmbedderConfig, InteractionEmbedder, SpringLayoutConfig, SpringLayoutEmbedder, BladeConfig, Blade`.

* `BaseEmbedder(algorithm: Callable, config: EmbedderConfig)` — `.embed(data)` = `validate_input` → `algorithm(data, **config.dict())` → `validate_output`; `.config`, `.algorithm`, `.info`. Config dataclass fields must match the algorithm's keyword parameters (checked with `inspect.signature`).
* `MatrixToGraphEmbedder[Config]` — input `np.ndarray`, 2-D, symmetric (`atol=1e-7`); output `DataGraph`.
* `GraphToGraphEmbedder[Config]` — input/output `DataGraph`.

### `InteractionEmbedder(config: InteractionEmbedderConfig | None = None)` — matrix → DataGraph
```python
@dataclass
class InteractionEmbedderConfig(EmbedderConfig):
    method: str = "Nelder-Mead"     # scipy.optimize.minimize method
    maxiter: int = 200000
    tol: float = 1e-8
    x0: np.ndarray | None = None    # (N, 2) initial positions; None -> uniform [0,1) seeded by `seed`
    seed: int | None = None
```
`interaction_embedding(matrix, method, x0, maxiter, tol, seed=None) -> DataGraph`: minimizes `||squareform(1/pdist(coords)**6) - matrix||_F` over 2-D positions; the diagonal of `matrix` is part of the norm (pass 0 diagonal). Output coords centered on their median; graph has nodes 0..N-1 with `"pos"`, **no edges**.

### `Blade(config: BladeConfig | None = None)` — matrix → DataGraph (BLaDE = Balanced Latently Dimensional Embedder)
```python
@dataclass
class BladeConfig(EmbedderConfig):
    max_min_dist_ratio: float | None = None          # bound on max_radial_distance / min_pairwise_distance
    dimensions: tuple[int, ...] = (5, 4, 3, 2, 2, 2)  # sequence of latent dimensions explored; last must be 2
    starting_positions: np.ndarray | None = None
    pca: bool = True
    steps_per_round: int = 200
    compute_weight_relative_threshold: Callable[[float], float] = default_...
    compute_max_distance_to_walk: Callable[[float, float], float | tuple[float,float,float]] = default_...
    compute_regulation_cursor: Callable[[float], float] = default_...
    compute_ratio_step_factors: Callable[[float], float] = default_...
    ratio_rerun: int = 2
    device: InitVar[Device | None] = None            # if given: max_min_dist_ratio = device._max_radial_distance / device._min_distance
```
`Blade.embed(matrix)` → `_blade(matrix, **config) -> np.ndarray (N,2)` → `DataGraph.from_coordinates(positions.tolist())`. `_blade` accepts a symmetric **or upper-triangular** QUBO/interaction matrix; interaction coefficient assumed 1; Fruchterman–Reingold-style force layout; extra `_blade` kwargs `draw_steps`, `draw_weighted_graph`, `draw_differences` exist but are not in `BladeConfig`. No seed field in `BladeConfig` (randomness from numpy global RNG unless `starting_positions` is given) — control it with `np.random.seed(...)`.

### `SpringLayoutEmbedder(config: SpringLayoutConfig | None = None)` — DataGraph → DataGraph
```python
@dataclass
class SpringLayoutConfig(EmbedderConfig): iterations: int = 100; threshold: float = 1e-4; seed: int | None = None
```
`spring_layout_embedding(graph, iterations, threshold, seed)`: copies the graph, sets edge weight → `sqrt(w)`, runs `nx.spring_layout(k=1, scale=None, ...)` so that `w_ij ≈ 1/r_ij^6`; returns a copy of the input graph (same nodes/edges) with `coords` set.

Typical QUBO path: `graph = Blade().embed(Q)` (or `InteractionEmbedder().embed(Q_offdiag)`) → `Register.from_graph(graph)`; the diagonal of Q (local fields) is not embedded by either — it has to go into the detuning, globally (`Drive(detuning=...)`) or per-qubit via a DMM (`DetuningMapModulator(waveform<=0, weights)`), on a DMM-capable device.

## 7. Minimal snippet: QUBO matrix → embedding → program → compile → counts

```python
import numpy as np
from qoolqit import Register, Drive, QuantumProgram, AnalogDevice, AnalogDeviceWithDMM
from qoolqit.drive import DetuningMapModulator
from qoolqit.waveforms import PiecewiseLinearWaveform, ConstantWaveform
from qoolqit.embedding import Blade, BladeConfig            # or InteractionEmbedder, InteractionEmbedderConfig
from qoolqit.execution import LocalEmulator, BackendType, EmulationConfig, BitStrings

Q = np.array([[-1.0, 0.8, 0.3],
              [ 0.8, -1.0, 0.5],
              [ 0.3, 0.5, -1.0]])                           # symmetric QUBO

device = AnalogDevice()                                      # AnalogDeviceWithDMM() if you want per-qubit detuning
graph = Blade(BladeConfig(device=device)).embed(np.triu(Q, k=1) + np.triu(Q, k=1).T)   # couplings -> positions (1/r^6 ~ Q_ij)
register = Register.from_graph(graph)                        # dimensionless coords; ids 0..N-1

T = 4.0                                                      # dimensionless duration
amp = PiecewiseLinearWaveform([T/3, T/3, T/3], [0.0, 1.0, 1.0, 0.0])
det = PiecewiseLinearWaveform([T/3, T/3, T/3], [-2.0, -2.0, 2.0, 2.0])    # adiabatic sweep; local fields Q_ii not encoded here
drive = Drive(amplitude=amp, detuning=det)
# optional per-qubit detuning (needs AnalogDeviceWithDMM):
# dmm = DetuningMapModulator(waveform=ConstantWaveform(T, -1.0), weights={i: w for i, w in enumerate(...)})  # weights in [0,1], waveform <= 0
# drive = Drive(amplitude=amp, detuning=det, dmm=dmm)

program = QuantumProgram(register, drive)
program.compile_to(device, profile="max_energy")             # or "default" (must fit device.specs), optional device_max_duration_ratio=0.5
seq = program.compiled_sequence                               # pulser.Sequence (also usable with emu_sv/emu_mps/pasqal_cloud directly)

backend = LocalEmulator(backend_type=BackendType.QutipBackendV2, num_shots=1000)   # or emu_sv.SVBackend / emu_mps.MPSBackend
job = backend.run(program)                                    # synchronous for LocalEmulator
results = job.results()                                       # pulser.backend.Results
counts: dict[str, int] = dict(results.final_bitstrings)       # bit i <-> register.qubits_ids[i]; '1' = Rydberg (x_i = 1)
```

Remote variant: `from pasqal_cloud import PasqalCloudConnection, RemoteMPSBackend`; `RemoteEmulator(backend_type=RemoteMPSBackend, connection=PasqalCloudConnection(username, password, project_id), num_shots=100).run(program).results(timeout=600)`.
