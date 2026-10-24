# `qubo-solver` 0.9.1 (import name `qubosolver`) — code-level API digest

Source read: `/home/claude/vendor/qubosolver` (pure Python, requires `torch`, `qoolqit>=1.4.0`,
`pulser[torch]>=1.9.0`, `emu-sv`, `emu-mps`, `pasqal-cloud`, `swiglpk`, `scikit-optimize`,
`scikit-learn`, `scipy`, `pandas`, `seaborn`, `shapely`). Optional extra `cplex`.
Python `>=3.10,<3.15`.

> **Naming caveat.** The public names are `Instance`, `Solver`, `Solution`, `SolverConfig`
> — **not** `QUBOInstance` / `QuboSolver` / `QuboSolution`. There is no top-level factory other
> than the `Solver` class itself. `_QuboSolverQuantum`, `_QuboSolverClassical`,
> `_DecomposeQuboSolver` are private classes in `qubosolver/solver/solver.py`.

---

## 0. Public surface (`qubosolver/__init__.py`)

```python
from qubosolver import (
    Instance, Solution, Candidate, Dataset,              # data types
    Solver, SolverConfig,                                # entry point + top-level config
    QuantumSolvingConfig, ClassicalSolvingConfig,        # config.solving alternatives
    EmbeddingConfig, DriveShapingConfig,                 # nested quantum configs
    LocalEmulator, RemoteEmulator,                       # backends (auto-select wrappers)
    AutoLocalEmulatorBackend, AutoRemoteEmulatorBackend, # backend_type factories
    torch_rng, extract_qubo, analysis,                   # helpers
    bitstring, bitstrings, matrix, vector, vectori, tensor, linalg, protocols,  # tensor submodules
    Bitstring, Bitstrings, Matrix, Vector, Vectori, Tensor,                      # type aliases
    drive_shaping, embedding, solving, transforms,                               # submodules
)
```

`import qubosolver` also calls `pulser.sequence.store_package_version_metadata("qubosolver", __version__)`.
Setting `QUBO_SOLVER_RUNTIME_CHECKS=1` enables beartype runtime checks (requires Python ≥ 3.14).

---

## 1. Constructors and signatures

### 1.1 `Instance` — `qubosolver/types/instance.py`

```python
from qubosolver import Instance, matrix

class Instance:
    def __init__(self, matrix: Matrix | None = None) -> None
    # self._matrix = matrix if matrix is not None else matrix.zeros(0)   (0x0 matrix by default)

    size: int                 # property: matrix.shape[0]
    __len__ -> int            # same as size  (so `if not instance:` means size == 0)
    matrix: Matrix            # property; asserts ndim == 2 and square
    _max_off_diag: float      # max of off-diagonal entries (RuntimeError if size < 2)
    def cost(self, solution: Bitstring) -> float        # x^T Q x via utils._costs.quadratic_cost
    def save(self, file_like) -> None
    @classmethod load(cls, file_like) -> Self
    # narrowing views for transform subclasses:
    variable_fixing -> transforms.variable_fixing.Instance
    zeroing         -> transforms.zeroing.Instance
    negative_bitflip-> transforms.negative_bitflip.Instance
```

`Matrix` is just `torch.Tensor` at runtime (2-D, global float dtype). **The constructor does not
convert, copy, validate symmetry or validate dtype** — it stores whatever you give it. Build
the tensor yourself with the `qubosolver.matrix` helpers so it lands on the globally
configured dtype/device:

```python
import numpy as np, torch
from qubosolver import Instance, matrix

Q_np = np.array([[...]], dtype=float)
inst = Instance(matrix.as_tensor(Q_np))          # torch.as_tensor(data, dtype=matrix.dtype(), device=matrix.device())
inst = Instance(matrix.tensor([[0, 1], [1, 0]]))  # from nested list
inst = Instance(torch.tensor(Q_np, dtype=torch.float32))  # raw torch also works
```

Global dtype/device (`qubosolver/types/linalg.py`): default `float32` on `cpu`. Env vars read
at import: `QUBO_SOLVER_FLOAT_DTYPE` (`float16|bfloat16|float32|float64`), `USE_DOUBLE_PRECISION`
(`1|true`), `QUBO_SOLVER_DEVICE` (`cpu|cuda|cuda:1|mps`), `USE_GPU` (`1|true`). Programmatic:
`qubosolver.types.linalg._GlobalConfig.use_double_precision()`, `.set_float_precision(dtype)`,
`.use_gpu()`, `.set_device(device)`. **Note:** `Instance.cost` does `torch.linalg.multi_dot([x, Q, x])`
with `x` cast to `Q.dtype`, so float32 costs can have rounding error.

Objective (docstring + `utils/_costs.py`): **minimise** `cost(x) = xᵀ Q x`, `x ∈ {0,1}ⁿ`. Q is
documented as **symmetric**; off-diagonal pairs `Q_ij` and `Q_ji` are *both* counted (so an
upper-triangular matrix is interpreted as half the coupling you may intend). Diagonal `Q_ii`
is the linear term (since `x_i² = x_i`).

### 1.2 `Solution` and `Candidate` — `qubosolver/types/solution.py`

```python
@dataclass
class Candidate:
    bitstring: Bitstring          # int8 tensor (n,)
    cost: float = float("inf")
    count: int = 0
    probability: float = 0.0
    string: str                   # property -> "0101..."

@dataclass
class Solution:
    bitstrings: Bitstrings  = zeros(0, 0)   # int8 (num_solutions, n)
    costs: Vector           = zeros(0)      # float (num_solutions,)
    counts: Vectori         = zeros(0)      # int64 (num_solutions,)
    probabilities: Vector   = zeros(0)      # float (num_solutions,)

    __getitem__(idx) -> Candidate ; __len__ -> num_solutions ; __iter__ -> Candidate
    num_variables: int                      # bitstrings.shape[1]
    truncate(k) -> Self                      # keep first k, renormalise probabilities
    deduplicate(update=True) -> Self         # merge identical rows (sum counts, min cost)
    @staticmethod concat(solutions, *, unit_counts=False) -> Solution
    @staticmethod from_results(results: pulser.backend.results.Results, instance) -> Solution
    @staticmethod zeros(length, *, count=1) -> Solution
    save(file_like) / @staticmethod load(file_like)
    check_consistency(*, instance=None, throw=False, full=True, rtol=1e-5, atol=1e-8) -> bool
    # private but used everywhere:
    _compute_costs(matrix) ; _compute_probabilities() ; _sort_by_cost() ; _update(instance)
```

**There is no `solution_status`, `metadata`, embedding/pulse artefacts or `best_*` attribute on
`Solution`.** Solutions returned by `Solver.solve()` are sorted ascending by cost, so:

```python
best = solution[0]                       # Candidate
best_bits  = best.string                 # "0110..."
best_cost  = best.cost                   # == solution.costs[0].item()
counts = {c.string: c.count for c in solution}          # dict bitstring -> count
# or, vectorised:
from qubosolver import bitstrings
counts = dict(zip(bitstrings.to_strings(solution.bitstrings), solution.counts.tolist()))
```

Embedding/drive artefacts live on the solver, not on the solution:
`solver._solver._cached_register: qoolqit.Register`, `solver._solver._cached_drive: qoolqit.Drive`
(only for the quantum solver, populated after `solve()`), plus `solver._embedding()` /
`solver._drive(register)` can be called directly. `solver._draw_sequence(drive, register)` draws
the compiled pulser sequence.

### 1.3 `Solver` — `qubosolver/solver/solver.py`

```python
from qubosolver import Solver, SolverConfig

class Solver(BaseSolver):
    def __init__(self, instance: Instance, config: SolverConfig | None = None) -> None
    def solve(self) -> Solution
    def _embedding(self) -> qoolqit.Register
    def _drive(self, embedding: qoolqit.Register) -> tuple[qoolqit.Drive, Solution]
```

`config.solving_mode == "quantum"` → wraps `_QuboSolverQuantum(instance, config)`;
otherwise `_QuboSolverClassical(instance, config)`. `_QuboSolverQuantum.__init__` raises
`ValueError` if `instance.size > 80` and logs a warning if the matrix has negative off-diagonals
and `config.preprocessing` is `False`.

### 1.4 `SolverConfig` — `qubosolver/solver/config/config.py`

```python
@dataclass
class SolverConfig:
    config_name: str = ""
    solving: QuantumSolvingConfig | ClassicalSolvingConfig = field(default_factory=QuantumSolvingConfig)
    postprocessing: bool = True
    preprocessing: bool = True

    solving_mode -> Literal["quantum", "classical"]   # property, from type of `solving`
    quantum   -> QuantumSolvingConfig                   # property; ValueError if classical
    classical -> ClassicalSolvingConfig                 # property; ValueError if quantum
```
`__repr__` returns `config_name` (so a default config prints as an empty string).

### 1.5 `QuantumSolvingConfig` (`solving.QuantumConfig`) — `qubosolver/solver/config/solving.py`

```python
@dataclass
class QuantumConfig:            # exported as QuantumSolvingConfig
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    drive_shaping: DriveShapingConfig = field(default_factory=DriveShapingConfig)
    backend: LocalEmulator | RemoteEmulator | qoolqit.execution.QPU = field(default_factory=LocalEmulator)
    device: qoolqit.Device = field(default_factory=qoolqit.AnalogDeviceWithDMM)

    max_min_dist_ratio -> float   # property: device.specs["max_radial_distance"]/device.specs["min_distance"] or inf
```

### 1.6 `ClassicalSolvingConfig` (`solving.ClassicalConfig`)

```python
@dataclass
class ClassicalConfig:          # exported as ClassicalSolvingConfig
    algorithm: Literal["tabu_search", "simulated_annealing", "cplex", "random_sampling"] = "tabu_search"
    time_limit: float = float("inf")
    max_iter: int = 100
    max_bitstrings: int = 1
    # __post_init__ raises ValueError on an unknown algorithm
```

### 1.7 `EmbeddingConfig` — `qubosolver/solver/config/embedding.py`

```python
@dataclass
class Config:                   # exported as EmbeddingConfig
    algorithm: Literal["blade", "greedy_layout"] = "blade"
    greedy_layout_lattice: Literal["square", "triangular"] = "triangular"
    blade_steps_per_round: int | None = 200
    # __post_init__ validates both literals
    def _to_greedy_layout_config(self, device: qoolqit.Device) -> embedding.greedy_layout.Config
```

### 1.8 `DriveShapingConfig` — `qubosolver/solver/config/drive_shaping.py`

```python
@dataclass
class Config:                   # exported as DriveShapingConfig
    algorithm: Literal["local_energy_scale", "proportional_diagonal", "bayesian_search"] = "local_energy_scale"
    bayesian_search_n_calls: int = 20        # skopt gp_minimize n_calls (docstring: minimum 12)
    proportional_diagonal_kappa: float = 1.0 # Ω_max = kappa * max|d_i|
    local_energy_scale_kappa: float = 0.1    # Ω_max = kappa * mean_i(E_i)
    def _to_bayesian_search_config(self) -> solving.hybrid.drive_bayesian_search.Config
```

### 1.9 Backend config (there is no separate "backend config" dataclass; the backend object *is* the config)

`qubosolver/types/backends.py`:

```python
class LocalEmulator(qoolqit.execution.LocalEmulator):
    def __init__(self, backend_type: type[EmulatorBackend] = AutoLocalEmulatorBackend, **kwargs) -> None
    # kwargs forwarded to qoolqit: emulation_config: EmulationConfig | None = None, num_shots: int | None = None

class RemoteEmulator(qoolqit.execution.RemoteEmulator):
    def __init__(self, backend_type: type[RemoteEmulatorBackend] = RemoteEmuFreeBackend, **kwargs) -> None
    # kwargs forwarded to qoolqit: connection: RemoteConnection (REQUIRED, keyword-only),
    #                              emulation_config=None, num_shots: int | None = None

# qoolqit.execution.QPU(*, connection: RemoteConnection, num_shots: int | None = None)  -> num_shots REQUIRED
```

Auto-selection thresholds (`_SV_THRESHOLD = 15`, `_MPS_THRESHOLD = 26`, on `len(sequence.register.qubit_ids)`):

| n_qubits | local (`AutoLocalEmulatorBackend`) | remote (`AutoRemoteEmulatorBackend`) |
|---|---|---|
| `< 15` | `pulser_simulation.QutipBackendV2` | `pasqal_cloud.backends.RemoteEmuFreeBackend` |
| `15..25` | `emu_sv.SVBackend` | `pasqal_cloud.backends.RemoteSVBackend` |
| `>= 26` | `emu_mps.MPSBackend` | `pasqal_cloud.backends.RemoteMPSBackend` |

If you pass an explicit `backend_type` that differs from the recommended one, `run()` emits a
`UserWarning` via `_warn_suboptimal_backend` (and mentions fees for remote).

Shots: `num_shots=None` → pulser `EmulationConfig.default_num_shots`, which defaults to **1000**
(`pulser/backend/config.py`). The emulation config always carries a `BitStrings(num_shots=...)`
observable, so `job.results().final_bitstrings` is a `dict[str, int]`.

### 1.10 Enum members, verbatim

- `qubosolver.embedding.enums.Lattice`: `SQUARE = "square"`, `TRIANGULAR = "triangular"`.
- `qubosolver.types._enums._DensityType`: `SPARSE = "sparse"`, `MEDIUM = "medium"`, `HIGH = "high"`.
- `qubosolver.types._enums._QUBOType`: `MAX_CUT = "max_cut"`, `ISING_MODEL = "ising_model"`, `GENERAL_QUBO = "general_qubo"`.
- `qoolqit.execution.compilation_functions.CompilerProfile`: `DEFAULT = "default"`, `MAX_ENERGY = "max_energy"`, `WORKING_POINT = DEFAULT` (deprecated alias).
- All other "enums" in configs are `typing.Literal` strings (listed above).

### 1.11 Other config dataclasses reachable from the pipeline

```python
# qubosolver/embedding/blade.py  (extends qoolqit.embedding.BladeConfig)
@dataclass
class Config(BladeConfig):
    # inherited from qoolqit BladeConfig:
    max_min_dist_ratio: float | None = None
    dimensions: tuple[int, ...] = (6, 5, 4, 3, 2, 2, 2)        # overridden default (qoolqit: (5,4,3,2,2,2))
    starting_positions: np.ndarray | None = None
    pca: bool = True
    steps_per_round: int = 200
    compute_weight_relative_threshold: Callable[[float], float] = lambda _: 0.1
    compute_max_distance_to_walk: Callable[[float, float], float | tuple] = qoolqit default
    compute_regulation_cursor: Callable[[float], float] = lambda _: 0.5   # overridden (qoolqit: 0.1)
    compute_ratio_step_factors: Callable[[float], float] = qoolqit default (interp [2, 0.94, 0.98])
    ratio_rerun: int = 2
    device: InitVar[Device | None] = None     # sets max_min_dist_ratio = max_radial/min_distance
    # added by qubosolver:
    initialize_with_mds: bool = True

# qubosolver/embedding/greedy_layout.py
@dataclass
class Config:
    traps: int = 200
    max_min_dist_ratio: float = float("inf")
    max_possible_term: tuple[Literal["quantile", "factor"], float] | float = ("quantile", 0.95)
    lattice: Lattice = Lattice.TRIANGULAR
    p: Literal[1, 2] = 1
    @staticmethod from_device(device) -> Config    # traps from device, ratio from device

# qubosolver/solving/hybrid/drive_bayesian_search.py
@dataclass
class Config:
    initial_amplitude_knots: list[float] = [0.5, 0.9, 0.5]
    initial_detuning_knots: list[float] = [-0.8, 0.0, 0.8]
    n_evaluations: int = 20
    seed: int | None = None
    objective_fn: Callable[[Solution], float] = _default_objective   # min cost, inf if empty
    default_sequence_duration: int = 50000

# qubosolver/solver/config/config.py (private; only used by _DecomposeQuboSolver)
@dataclass
class _DecompositionConfig:
    decompose_threshold: float = 250.0
    decompose_stop_number: int = 15
    decompose_break_placement: int = 3
    neglecting_inter_distance: float = 1.5
    neglecting_max_coefficient: float = 1.0
    classical_max_min_dist_ratio: float = float("inf")
```

---

## 2. End-to-end flow of `Solver(instance, config).solve()` (quantum mode)

All in `qubosolver/solver/solver.py::_QuboSolverQuantum.solve` and
`qubosolver/solver/_basesolver.py::BaseSolver`.

### Step 0 — construction (`_QuboSolverQuantum.__init__`)
1. Warn if `_has_negative_offdiagonal(instance.matrix)` and `not config.preprocessing`.
2. `_check_size_limit()` → `ValueError` if `instance.size > 80`.
3. `self.backend = config.quantum.backend`
4. `self.embedder = _get_embedder(instance, config.quantum, backend)` → `BLaDEmbedder` or `GreedyEmbedder` (`solver/_embedder.py`).
5. `self.drive_shaper = _get_drive_shaper(instance, config.quantum, backend)` → `ProportionalDiagonalDriveShaper` | `LocalEnergyScaleDriveShaper` | `BayesianSearchDriveShaper` (`solver/_drive_shaper.py`).

### Step 1 — trivial check (`BaseSolver._trivial_solution` → `solving.classical.trivial_solution_search.solve`)
Returns a 1-bitstring `Solution` immediately if: all coefficients ≥ 0 (→ all-zeros), all ≤ 0
(→ all-ones), or Q is diagonal (→ `x_i = 1 iff Q_ii < 0`). Otherwise an empty `Solution`
(falsy), and the pipeline continues. **This runs before any quantum step and even before
preprocessing**, so a non-negative Q never hits the emulator.

### Step 2 — preprocessing (`BaseSolver._preprocess`, only if `config.preprocessing`)
```python
instance = transforms.variable_fixing.apply_recursively(self.instance)   # Hansen fixing, repeat until no change
instance = transforms.negative_bitflip.apply(instance)                   # GLPK ILP, time_limit_s=60.0 default
self._update_instance(instance.negative_bitflip)
```
- **Variable fixing** (`transforms/variable_fixing.py`): `hansen_fixing` computes for each `i`
  `c_i + 2·Σ_j min(0,Q_ij) ≥ -1e-8 → fix 0`, `c_i + 2·Σ_j max(0,Q_ij) ≤ 1e-8 → fix 1`.
  `_reduce_qubo` folds `2·Q_ij` of each var fixed to 1 into `Q_jj`, deletes rows/cols.
  Returns `variable_fixing.Instance` (subclass) with `_fixed_indices: list[dict[int,int]]`
  and `_parent_instance`. `lift()` reinserts bits and recomputes costs on the parent matrix.
- **Negative bit-flip** (`transforms/negative_bitflip.py`): solves an ILP with `swiglpk`
  (variables `f_i ∈ {0,1}`, `y_k = f_i XOR f_j`, minimise remaining negative off-diagonal
  weight). Applies `x_i = 1 - y_i` where `f_i = 1`: `Q' = Q ∘ (s sᵀ) + diag(2 s ∘ (Q f))`,
  `s = 1 - 2f`, `offset = fᵀ Q f` (so `xᵀQx = yᵀQ'y + offset`). Returns
  `negative_bitflip.Instance` with `flips`, `metrics`, `status` (`"NONE"`, `"OPTIMAL"`,
  `"REJECTED_WORSE_THAN_NOOP"`, GLPK status names), `offset`. If the ILP leaves the problem
  worse, flips are reset to a no-op. `lift()` XORs bitstrings with `flips` and recomputes costs
  on the parent.

### Step 3 — zeroing fallback (always, independent of `preprocessing`)
```python
if _has_negative_offdiagonal(self.instance.matrix):
    self.instance = transforms.zeroing.apply(self.instance)   # sets Q_ij<0 (i≠j) to 0, records them in negative_matrix
```
`zeroing.Instance.zeroed_edges` → `(N,2)` index pairs. `zeroing.lift()` keeps bitstrings and
recomputes costs on the un-zeroed parent. **Negative couplings are therefore approximated
away** (Rydberg blockade is purely repulsive/positive).

If `instance.size == 0` after preprocessing: `solution = Solution.zeros(0)` and the quantum
pipeline is skipped.

### Step 4 — embedding (`_QuboSolverQuantum._embedding` → `self.embedder.embed()`)

**Target device**: `config.quantum.device` (default `qoolqit.AnalogDeviceWithDMM()`).
Device constraints are read through `device.specs` (adimensional units; the default
`UnitConverter.from_distance(C6, min_atom_distance)` makes `min_distance == 1.0`):
`{"max_duration", "max_amplitude", "max_abs_detuning", "min_distance", "max_radial_distance"}`.
Only `max_min_dist_ratio = max_radial_distance / min_distance` is consumed by the embedders
(AnalogDevice: 38/5 = 7.6).

**`"blade"` (default)** — `BLaDEmbedder.embed` → `embedding.blade.embed(instance, config=blade.Config(steps_per_round=..., max_min_dist_ratio=...))`:
- Raises `ValueError` for empty instance or negative off-diagonal.
- `size == 1` → `Register.from_coordinates(zeros(1,2))`.
- If `initialize_with_mds` and `(triu(Q,1) > 0).any()` → `starting_positions = embed_mds(Q)`
  (`embedding/_algorithms/mds.py`: zero the diagonal, `d_ij = Q_ij^(-1/6)` via
  `normalized_best_dist`, replace disconnected/∞ distances with graph shortest paths (then
  `2×max` for still-infinite), `sklearn.manifold.smacof(n_components=2, n_init=4, max_iter=3000, eps=1e-9)`).
- `qoolqit.embedding.Blade(blade_config).embed(Q)` runs `_blade(...)` (qoolqit
  `embedding/algorithms/blade/blade.py`): a Fruchterman–Reingold-style force simulation whose
  target is `U_ij = 1/‖r_i−r_j‖⁶ ≈ Q_ij` (interaction coefficient normalised to 1). It starts
  in `dimensions[0]`=6 dims (random or MDS-augmented), runs `steps_per_round` (200) force steps
  per dimension transition `6→5→4→3→2→2→2` (6 transitions → 1200 steps), shrinking dimension
  (PCA-prioritised) between rounds. Forces: interaction forces toward target weights
  (only weight differences above `compute_weight_relative_threshold`·max are "significant"),
  min-distance and max-radius constraint forces when `max_min_dist_ratio` is set (target
  ratio ramps via `compute_ratio_step_factors`), capped per-step displacement
  (`compute_max_distance_to_walk`), and a `regulation_cursor` balancing force priorities. If the
  final `max_radial/min_pairwise > max_min_dist_ratio`, it reruns up to `ratio_rerun=2` times
  in 2-D with ratio priority, then raises `DistanceRatioException`.
- Returns `qoolqit.Register.from_graph(graph)` → qubit ids `0..n-1`, 2-D coordinates in
  adimensional units (min pairwise distance ≈ 1 ⇒ interaction ≈ 1).

**`"greedy_layout"`** — `GreedyEmbedder.embed` → `greedy_layout.embed(instance, config=EmbeddingConfig._to_greedy_layout_config(device))`
where `Config.from_device(device)` gives `traps = device.max_layout_traps or ceil(max_atom_num/max_layout_filling) or 200`
and `max_min_dist_ratio` from the device; `lattice` from `greedy_layout_lattice`.
Algorithm (`embedding/_algorithms/greedy/greedy.py::Greedy.launch_greedy`):
- `spacing = max_possible_term ** (-1/6)` where `max_possible_term` = 0.95-quantile of strictly
  positive upper-triangular Q entries (default) — so the largest "typical" coupling maps to
  nearest-neighbour distance.
- Trap coordinates: `spacing * get_layout(lattice, traps)` (`pulser.register.special_layouts.TriangularLatticeLayout(n, spacing=1)`
  or the `n_traps` points closest to origin of a `SquareLatticeLayout`).
- `U[p,q] = 1/‖r_p−r_q‖⁶` between traps. For every start node `v`: place `v` on the trap closest
  to origin; then repeatedly pick the unplaced node with the largest summed coupling to placed
  nodes (`argmax Σ_j∈placed Q[i,j]`), and the free trap minimising
  `s(t) = Σ_{j placed} |Q[u,j] − U[t,trap(j)]|^p`; traps with `‖r‖ ≥ max_min_dist_ratio·spacing`
  are discarded (ValueError if none left). Keep the start node whose final `‖U−Q‖_p` over all
  pairs is smallest. Returns `Register.from_coordinates(coords)`.

The register is cached in `self._cached_register`.

### Step 5 — drive shaping (`_QuboSolverQuantum._drive(register)` → `self.drive_shaper.generate(register)`)

`dmm = drive_shaping._device_specs.support_dmm(device)` = `pulser_specs(device)["dmm_bottom_detuning"] is not None`,
i.e. **True only if the pulser device has at least one `dmm_channels` entry** (true for the
default `AnalogDeviceWithDMM`, `DigitalAnalogDevice`, `MockDevice`; **false for `AnalogDevice`**).
Without DMM every shaper falls back to a global detuning equal to the mean target and logs a
warning.

Common definitions (both heuristic shapers, `drive_shaping/local_energy_scale.py` and
`drive_shaping/proportional_diagonal.py`):

- `T = device.specs["max_duration"] or 1000.0` (adimensional; for AnalogDevice-based devices
  `6000 ns / TIME`). Both Ω and δ waveforms use this full duration.
- Target local final detunings from the **diagonal**: `d_i = -0.5 * Q_ii` (so a negative linear
  term → positive detuning → favours `x_i = 1`; matches `H = … − Σ δ_i n_i` and
  `extract_qubo` which rebuilds `Q_ii = -2 δ_i`).
- Amplitude: `Ω(t) = qoolqit.InterpolatedWaveform(T, [1e-9, Ω_max, Ω_max, 1e-9])` (PCHIP through
  4 evenly spaced knots: ramp up, plateau, ramp down).
- Global detuning: `δ_g(t) = InterpolatedWaveform(T, [δ_0, δ_0, δ_g(T), δ_g(T)])` with
  `δ_0 = -max_i |d_i|` (start strongly negative = all atoms in ground state).
- DMM (if `use_dmm` and spread `d_max − d_min > 1e-15`):
  `δ_g(T) = d_max`, `δ_dmm(T) = -(d_max − d_min) ≤ 0`,
  `w_i = (d_max − d_i)/(d_max − d_min) ∈ [0,1]` → `δ_i(T) = δ_g(T) + δ_dmm(T)·w_i = d_i`.
  Built by `_waveforms.constant_weighted_dmm(weights, T, final_detuning=δ_dmm(T), device, energy_scale)`
  = `qoolqit.drive.DetuningMapModulator(weights={i: w_i}, waveform=ConstantWaveform(T, δ_dmm(T)))`,
  returns `None` if all weights are 0. **Clipping:** `_clip_final_detuning_to_dmm_budget` caps
  `|δ_dmm(T)| ≤ 0.999·|total_bottom_detuning| / (Σ w_i · energy_scale)` where
  `energy_scale = device._target_amp / Ω_max` is the predicted MAX_ENERGY rescale factor
  (AnalogDeviceWithDMM: `total_bottom_detuning = -2π·20 rad/µs`). Logs when clipped.
  Without DMM: `δ_g(T) = mean(d)`, no DMM.
- Amplitude clamp: `Ω_max ≤ max_virtual_amplitude(device, register)` =
  `specs["max_amplitude"] · (max_radial_device / max_radial_register)⁶ · (1 − 1e-3)` (if the
  register is wider than the device, the realisable amplitude shrinks).
- Detuning clamp: `max_detuning = (max_abs_detuning/max_amplitude) · Ω_max · (1 − 1e-3)`; if
  `max|d_i| > max_detuning`, `d ← d · max_detuning/max|d_i|` (all detunings rescaled, DMM
  weights recomputed). For AnalogDevice `max_abs_detuning/max_amp = 20/2 = 10`, so you need
  `kappa ≥ 0.1` or the shaper warns "you're likely to get a qoolqit CompilationError".

**`"local_energy_scale"` (default)**: `E_i = |δ_i(T)| + Σ_{j≠i} |V_ij|` with
`V = register.interaction_matrix()` (`1/r⁶`), `Ω_max = kappa · mean_i(E_i)`, `kappa = local_energy_scale_kappa = 0.1`.
Returns `(Drive, Solution())` (empty solution).

**`"proportional_diagonal"`**: `Ω_max = kappa · max_i |d_i|`, `kappa = proportional_diagonal_kappa = 1.0`.
Returns `(Drive, Solution())`.

**`"bayesian_search"`** (`solving/hybrid/drive_bayesian_search.py::solve`): 6 parameters —
3 interior amplitude knots in `[1e-4, 1−1e-4]` and 3 detuning knots with bounds
`[(-1,-ε), (-1,1), (ε,1)]` (start negative, end positive). `_build_drive`:
`Ω = InterpolatedWaveform(T, [1e-9, a1, a2, a3, 1e-9] · max_virtual_amplitude)`,
`δ = InterpolatedWaveform(T, [d1, d2, d3] · det_ratio · Ω.max() · (1−1e-3))`; DMM only if
`dmm and final δ > 0`, with weights `w_i = 1 − |Q_ii| / max_j |Q_jj|` (**note: uses `|Q_ii|`,
different convention from the heuristic shapers**) and `final_detuning = -δ(T)`. Each
evaluation compiles with `default_sequence_duration=config.default_sequence_duration` (50000)
and runs `backend.run(program)`; objective = `config.objective_fn(solution)` (min cost; 1e4 on
failure). `skopt.gp_minimize(objective, bounds, x0=initial_params, n_calls=n_evaluations, random_state=seed)`.
Returns `(solution_at_best_params, drive)` — this solution is **already a sampled result**, so
`_QuboSolverQuantum.solve` skips `_execute` when `solution` is non-empty and algorithm is
`"bayesian_search"`.

Drive cached in `self._cached_drive`.

### Step 6 — compile + execute (`BaseSolver._execute` → `_submit`)
```python
program = solving.analog_quantum_sampling.compile(embedding, drive, config.quantum.device, default_sequence_duration=6000)
job = config.quantum.backend.run(program)
solution = Solution.from_results(job.results(), self.instance)
```
`solving/quantum/analog_quantum_sampling.py::compile(register, drive, device, *, compiler_profile=CompilerProfile.MAX_ENERGY, default_sequence_duration=None)`:
- Raises `ValueError` if `register.max_radial_distance()/register.min_distance() > device max ratio`.
- If `device.specs["max_duration"] is None`, patches the pulser device with
  `max_sequence_duration=default_sequence_duration` (6000 ns here).
- `qoolqit.QuantumProgram(register=..., drive=...).compile_to(device, profile=MAX_ENERGY, device_max_duration_ratio=0.99)`.
  `MAX_ENERGY` (qoolqit `execution/compilation_functions.py::basic_compilation`): if
  `Ω_max · r_min⁶ > device._energy_ratio` map `Ω_max → device._target_amp` (2π·2 rad/µs for
  AnalogDevice, floor-rounded), else map `r_min → min_atom_distance` (5 µm); distances scale as
  `(C6/ENERGY)^(1/6)` with `C6 = 865723.02` (Rydberg level 60); duration is **always** set to
  `0.99 · max_sequence_duration` = 5940 ns when the device has a max duration; a `rydberg_global`
  channel is declared and the pulse added; DMM is added via `define_detuning_map` +
  `config_detuning_map` + `add_dmm_detuning` on the first `dmm_channels` id. Real pulser
  `Device`s with `requires_layout` get `with_automatic_layout(device)`.
  `_validate_program_max_energy_profile` raises `qoolqit.CompilationError` if the rescaled
  radial distance / detuning exceed the device.
- `LocalEmulator.run`: instantiates `backend_type(program.compiled_sequence, config=emulation_config)`
  and runs synchronously, returning `job._LocalJob(results)`. `RemoteEmulator.run` /
  `QPU.run`: `backend_type(sequence, connection=..., config=...).run(wait=False)` → `_RemoteJob`;
  `job.results()` blocks (optionally `timeout=`).
- `Solution.from_results` reads `results.final_bitstrings: dict[str,int]`, builds int8
  bitstrings + int64 counts, then `_update(instance)` = compute costs on the **current (possibly
  transformed) instance**, sort ascending, probabilities = counts / Σcounts.

### Step 7 — post-processing
1. If the instance is a `zeroing.Instance`: `solution = zeroing.lift(solution, instance)`
   (recost on un-zeroed matrix, re-sort), instance ← parent.
2. `_post_process_fixation` (only if `config.preprocessing`): `negative_bitflip.lift` then
   `variable_fixing.lift` — restores original variables, recomputes costs on the original Q.
   **Counts/probabilities are carried over unchanged, and the lifted solution is not re-sorted
   in `negative_bitflip.lift`/`variable_fixing.lift`** (costs are recomputed, order kept).
3. `_post_process` (only if `config.postprocessing`):
   `solving.iterative_bitflip_local_search.solve(instance, starts=solution, strategy="greedy_sweep", max_iterations=1, time_limit=600.0)`
   — one greedy sweep of single-bit flips over every sampled bitstring, then `deduplicate()`
   (merged bitstrings sum their counts) and re-sort by cost. This means **the returned bitstrings
   are generally not the raw samples** when `postprocessing=True`.

Return: `Solution` sorted ascending by cost with `bitstrings`, `costs`, `counts`, `probabilities`.

### Classical mode (`_QuboSolverClassical.solve`)
trivial check → `_preprocess` → `get_classical_solver(instance, config.classical).solve()` →
`_post_process_fixation` → `_post_process`. Dispatch (`solver/_classical_solver.py`):
- `"cplex"` → `solving.classical.cplex.solve(instance, time_limit=...)` (lazy import of `cplex`).
- `"simulated_annealing"` → `solving.simulated_annealing.solve(instance, top_k=max_bitstrings, max_iter=max_iter, starts=bitstrings.rand(1, n, rng), time_limit, stats="per_run", rng=torch_rng("global"))`.
- `"tabu_search"` → `solving.tabu_search.solve(instance, starts=bitstrings.rand(1, n, rng), max_iter=max_iter, time_limit)` (defaults `tabu_tenure=7`, `max_no_improve=20`).
- `"random_sampling"` → `solving.classical.random_sampling.solve(instance, rng, max_bitstrings)`.
Also available but not wired into configs: `solving.brute_force.solve(instance, *, max_bitstrings=1, time_limit=inf)` (exact, 2ⁿ).

### `_DecomposeQuboSolver` (private, `solver/solver.py`)
`__init__(instance, *, solver_config=None, decompose_config=None)`; rejects negative off-diagonals.
If `size <= decompose_stop_number` (15) → classical tabu. Otherwise builds
`transforms._decompositions.Instance`, repeatedly `extract_subqubo` (geometric search seeded from
a random vertex, placing vertices within `max_min_dist_ratio` of the device) → `Solver(subqubo, config).solve()`
→ `_decompositions.update` (writes best bitstring into the global solution, propagates edge
values, fixes positive-field vertices); the tail is solved classically. Returns exactly one bitstring.

---

## 3. The result object — `Solution` (see §1.2)

| Attribute | Type / shape | Meaning |
|---|---|---|
| `bitstrings` | `int8 (k, n)` | candidate bitstrings, sorted by ascending cost |
| `costs` | `float (k,)` | `xᵀQx` on the **original** Q (after lifts) |
| `counts` | `int64 (k,)` | number of shots per bitstring (after dedup/post-processing merges) |
| `probabilities` | `float (k,)` | `counts / counts.sum()` |
| `len(solution)` | int | k |
| `solution.num_variables` | int | n |
| `solution[i]` | `Candidate(bitstring, cost, count, probability)`; `.string` → `"0101"` |

Not present: `solution_status`, `metadata`, `best_bitstring`, register/drive. For those use the
solver's `_cached_register` / `_cached_drive`, and `transforms.*.Instance` attributes
(`flips`, `status`, `metrics`, `offset`, `zeroed_edges`, `_fixed_indices`) if you run the transforms
yourself.

Helpers: `qubosolver.analysis.to_dataframe([sol, ...], labels=[...]) -> pandas.DataFrame`
(columns `labels, bitstrings, costs, counts, probs`; it calls `check_consistency(throw=True)` so the
solution must be fully consistent). `Solution.save/load` (torch.save with `weights_only=True`).
`qubosolver.extract_qubo(register, drive) -> Instance` rebuilds the physically encoded Q
(`Q_ij = 1/r⁶`, `Q_ii = -2 δ_i(T)`).

---

## 4. Backend and device selection

```python
from qubosolver import Solver, SolverConfig, QuantumSolvingConfig, LocalEmulator, RemoteEmulator
import qoolqit
```

| Goal | Config |
|---|---|
| Local, auto-select qutip / emu-sv / emu-mps by size (default) | `QuantumSolvingConfig(backend=LocalEmulator(num_shots=1000))` |
| Force local qutip | `from pulser_simulation import QutipBackendV2; LocalEmulator(backend_type=QutipBackendV2, num_shots=...)` |
| Force local emu-sv | `from emu_sv import SVBackend; LocalEmulator(backend_type=SVBackend)` |
| Force local emu-mps | `from emu_mps import MPSBackend; LocalEmulator(backend_type=MPSBackend)` |
| Pasqal cloud emulator (free tier, <15 q) | `from pasqal_cloud import PasqalCloudConnection; conn = PasqalCloudConnection(username=..., password=..., project_id=..., region="fr"); RemoteEmulator(connection=conn, num_shots=...)` (default `backend_type=RemoteEmuFreeBackend`) |
| Cloud auto-select | `RemoteEmulator(backend_type=AutoRemoteEmulatorBackend, connection=conn)` |
| Cloud emu-sv / emu-mps | `backend_type=pasqal_cloud.backends.RemoteSVBackend` / `RemoteMPSBackend` |
| Real QPU | `qoolqit.execution.QPU(connection=conn, num_shots=100)` (num_shots **required**) |
| Credential-less "remote" for tutorials | `from qoolqit.execution.example_connection import ExampleConnection; RemoteEmulator(connection=ExampleConnection())` |

Credentials: **qubosolver itself reads no cloud env vars**. `PasqalCloudConnection(username="", password="", project_id="", region="fr", **kwargs)`
takes credentials as arguments (passed to `PasqalCloudClient`). The pasqal-cloud package reads only
`PASQAL_SKIP_SSL_VERIFY`, `PASQAL_SKIP_GZIP_REQUEST_BODY` (testing) and, for the OVH path,
`PASQAL_PULSER_ACCESS_TOKEN`. Env vars that qubosolver *does* read: `QUBO_SOLVER_FLOAT_DTYPE`,
`USE_DOUBLE_PRECISION`, `QUBO_SOLVER_DEVICE`, `USE_GPU`, `QUBO_SOLVER_RUNTIME_CHECKS`.

Device (`QuantumSolvingConfig.device`, a `qoolqit.Device`):
- **Default `qoolqit.AnalogDeviceWithDMM()`**: `pulser.AnalogDevice.to_virtual()` + one
  `DMM(clock_period=4, min_duration=16, max_duration=6000, mod_bandwidth=8, bottom_detuning=-2π·20, total_bottom_detuning=-2π·20)`,
  renamed `"AnalogDeviceWithDMM"`. Inherits AnalogDevice limits: `max_atom_num=80`,
  `max_radial_distance=38 µm`, `min_atom_distance=5 µm`, `max_sequence_duration=6000 ns`,
  `max_amp=2π·2`, `max_abs_detuning=2π·20` (rad/µs), Rydberg level 60 (`C6 = 865723.02`).
  Because it is a *virtual* device, `requires_layout` is dropped and no automatic trap layout is applied.
- `qoolqit.AnalogDevice()` (real pulser `AnalogDevice`): **no DMM channel** → `support_dmm` is False →
  all shapers fall back to a single global detuning `δ_g(T) = mean(d_i)`; the diagonal is then
  only encoded on average. It also has `requires_layout=True`, so compilation calls
  `with_automatic_layout`.
- `qoolqit.DigitalAnalogDevice()`, `qoolqit.MockDevice()` have DMM channels (pulser defaults);
  `MockDevice` has no `max_sequence_duration`, so `compile` injects 6000 ns.
- Any `qoolqit.Device(pulser_device=..., default_converter=None)` works, incl.
  `qoolqit.Device.from_connection(connection, name="FRESNEL")`. Pulser's `WeightedAnalogDevice`
  is not referenced anywhere in qubosolver.
- The device is used for: `max_min_dist_ratio` (embedding), `specs` + DMM budget (drive shaping),
  and `compile_to` (execution). It is **not** cross-checked against the backend.

---

## 5. Minimal copy-paste example (strictly from the source)

```python
import numpy as np
import torch

from qubosolver import (
    Instance, Solver, SolverConfig, QuantumSolvingConfig,
    EmbeddingConfig, DriveShapingConfig, LocalEmulator, bitstrings, matrix,
)

# QUBO: minimise x^T Q x, Q symmetric, off-diagonals >= 0 for a native quantum embedding.
Q_np = np.array(
    [[-1.0,  0.8,  0.0,  0.3],
     [ 0.8, -1.2,  0.5,  0.0],
     [ 0.0,  0.5, -0.9,  0.7],
     [ 0.3,  0.0,  0.7, -1.1]]
)
instance = Instance(matrix.as_tensor(Q_np))   # global dtype (float32) on global device (cpu)

torch.manual_seed(0)                          # feeds torch_rng("global") used by classical pieces

config = SolverConfig(
    config_name="local-emulator",
    solving=QuantumSolvingConfig(
        embedding=EmbeddingConfig(algorithm="blade", blade_steps_per_round=200),
        drive_shaping=DriveShapingConfig(algorithm="local_energy_scale", local_energy_scale_kappa=0.1),
        backend=LocalEmulator(num_shots=500),  # auto: QutipBackendV2 (<15 q) / SVBackend / MPSBackend
        # device defaults to qoolqit.AnalogDeviceWithDMM()
    ),
    preprocessing=True,     # variable fixing + GLPK bit-flip
    postprocessing=True,    # one greedy bit-flip sweep on every sampled bitstring
)

solver = Solver(instance, config)
solution = solver.solve()                     # Solution sorted by ascending cost

best = solution[0]
print("best bitstring:", best.string)         # e.g. "1010"
print("best cost     :", best.cost)           # == solution.costs[0].item()
print("counts        :", dict(zip(bitstrings.to_strings(solution.bitstrings), solution.counts.tolist())))
print("probabilities :", solution.probabilities.tolist())

# artefacts from the quantum pipeline (populated after solve()):
register = solver._solver._cached_register    # qoolqit.Register, coordinates via register.qubits
drive    = solver._solver._cached_drive       # qoolqit.Drive (amplitude, detuning, dmm)

# classical cross-check
from qubosolver import ClassicalSolvingConfig
ref = Solver(instance, SolverConfig(solving=ClassicalSolvingConfig(algorithm="tabu_search", max_iter=200))).solve()
print("tabu best:", ref[0].string, ref[0].cost)
```

Note: the 4×4 example above is not trivial (mixed signs, non-diagonal), so it really reaches the
emulator; a Q with all entries ≥ 0 or ≤ 0 is answered by `trivial_solution_search` without embedding.

---

## 6. Knobs to tune

| Knob | Where | Default | Effect |
|---|---|---|---|
| `SolverConfig.solving` | top | `QuantumSolvingConfig()` | quantum vs classical pipeline |
| `SolverConfig.preprocessing` | top | `True` | Hansen variable fixing + GLPK bit-flip (60 s ILP limit, hard-coded) |
| `SolverConfig.postprocessing` | top | `True` | greedy-sweep bit-flip local search (`max_iterations=1`, `time_limit=600`) on all samples |
| `QuantumSolvingConfig.backend` | quantum | `LocalEmulator()` (auto, 1000 shots) | emulator / cloud / QPU; `num_shots=` |
| `QuantumSolvingConfig.device` | quantum | `AnalogDeviceWithDMM()` | limits, DMM availability, duration (6000 ns) |
| `EmbeddingConfig.algorithm` | embedding | `"blade"` | `"blade"` (force-directed, MDS init) vs `"greedy_layout"` (lattice) |
| `EmbeddingConfig.blade_steps_per_round` | embedding | `200` | more steps = better fit, slower (×6 transitions) |
| `EmbeddingConfig.greedy_layout_lattice` | embedding | `"triangular"` | `"square"` alternative |
| `DriveShapingConfig.algorithm` | drive | `"local_energy_scale"` | `"proportional_diagonal"`, `"bayesian_search"` |
| `DriveShapingConfig.local_energy_scale_kappa` | drive | `0.1` | `Ω_max = κ·mean(E_i)`; must be ≥ `max_amp/max_abs_det` (0.1 on AnalogDevice) |
| `DriveShapingConfig.proportional_diagonal_kappa` | drive | `1.0` | `Ω_max = κ·max|Q_ii|/2` |
| `DriveShapingConfig.bayesian_search_n_calls` | drive | `20` | `gp_minimize` evaluations (each = one emulation); min 12 |
| `ClassicalSolvingConfig.algorithm` | classical | `"tabu_search"` | `"simulated_annealing"`, `"cplex"`, `"random_sampling"` |
| `ClassicalSolvingConfig.max_iter / time_limit / max_bitstrings` | classical | `100 / inf / 1` | search budget / returned candidates |
| `torch.manual_seed(s)` | global | — | seeds `torch_rng("global")` (classical starts, decomposition) |
| `QUBO_SOLVER_FLOAT_DTYPE=float64` | env | float32 | exact-ish costs |
| `QUBO_SOLVER_DEVICE=cuda` | env | cpu | torch device for tensors (not the emulator) |
| Direct API: `embedding.blade.embed(instance, config=blade.Config(device=..., dimensions=..., steps_per_round=...))`, `drive_shaping.local_energy_scale.build_drive(instance, register, device=..., dmm=..., kappa=...)`, `solving.analog_quantum_sampling.compile(...)`, `backend.run(program)`, `Solution.from_results(...)` | modules | — | hand-rolled pipeline with full control |

---

## 7. Gotchas

1. **Names**: `Instance`/`Solver`/`Solution`/`SolverConfig`, not `QUBOInstance`/`QuboSolver`.
   `QuantumSolvingConfig`/`ClassicalSolvingConfig` are aliases of `solving.QuantumConfig`/`ClassicalConfig`.
2. **Objective**: minimises `xᵀQx` with Q treated as **symmetric** (both triangles summed). Give a
   full symmetric matrix; an upper-triangular matrix halves the couplings, and the BLaDE/qoolqit
   `Qubo.from_matrix` accepts triangular input and takes `np.triu` of a symmetric one, so
   asymmetric non-triangular matrices are rejected by qoolqit (`ValueError`).
3. **Sign convention**: off-diagonal `Q_ij > 0` = repulsive = natively embeddable (Rydberg
   blockade `1/r⁶`). `Q_ij < 0` is handled by bit-flip ILP (`preprocessing=True`) and whatever
   remains is **zeroed** (approximation, silently apart from an `info` log). Negative diagonal
   → positive final detuning (`d_i = -Q_ii/2`) → favours `x_i = 1`.
4. **Size limit**: quantum path raises `ValueError` for `n > 80` (device atom cap). Practical
   emulation limits: qutip < 15 qubits, emu-sv ≤ ~25, emu-mps beyond (approximate, slower).
   Local emulators run on CPU unless emu-sv/emu-mps are configured otherwise.
5. **Trivial short-circuit**: all-non-negative or all-non-positive or diagonal Q returns one
   bitstring with `count=1` without touching the quantum stack. Don't use such Q to "test the
   emulator".
6. **Zero-variable instance**: if variable fixing fixes everything, `Solution.zeros(0)` is
   lifted back — fine, but no emulation happens.
7. **DMM only on devices with `dmm_channels`**: default `AnalogDeviceWithDMM` has one;
   `qoolqit.AnalogDevice()` does **not** → global detuning `mean(d_i)` only, diagonal information
   mostly lost. `DMM` waveform must be ≤ 0; the weighted detuning is clipped to
   `total_bottom_detuning = -2π·20` after energy rescaling — large diagonal spreads get squashed.
8. **Durations are fixed**: waveforms span `max_duration` (adimensional) and the compiler pins the
   sequence to `0.99 × 6000 ns = 5940 ns` for AnalogDevice-based devices (`device_max_duration_ratio=0.99`).
   There is no config knob for pulse duration short of a custom `qoolqit.Device`.
9. **Energy rescaling**: `CompilerProfile.MAX_ENERGY` rescales Ω_max to the device max amplitude
   (2π·2 rad/µs) *or* the min register distance to 5 µm, whichever is binding; detuning scale
   follows. `max|δ|` is capped to `10·Ω_max·(1−1e-3)` by the shapers — small `kappa` ⇒ detunings
   scaled down ⇒ diagonal under-weighted.
10. **Post-processing rewrites samples**: with `postprocessing=True` the returned bitstrings are
    local minima reached by a greedy sweep from each sample; `counts` are merged. Set
    `postprocessing=False` to see raw shot statistics.
11. **Costs after lift**: `negative_bitflip.lift` and `variable_fixing.lift` recompute costs on
    the original matrix but do **not** re-sort; `zeroing.lift` does re-sort. The final
    `_post_process` re-sorts (when enabled). With `preprocessing=True, postprocessing=False`
    the order could in principle be non-monotone in cost if bit-flips changed the ranking —
    check `solution.costs` rather than assuming `[0]` is best, or call `solution._sort_by_cost()`.
12. **Randomness**: BLaDE uses numpy's global RNG (random initial positions when MDS is skipped);
    MDS `smacof` uses sklearn defaults (no `random_state`); greedy is deterministic; classical
    solvers draw from `torch_rng("global")` (seed via `torch.manual_seed`); Bayesian search takes
    `Config.seed` but `DriveShapingConfig` does **not** expose it (always `None`); emulator sampling
    randomness is the backend's. `qubosolver.utils._random.manual_seed(seed)` seeds numpy, torch and
    `random` together; `seed_context(seed)` is a context manager.
13. **Torch everywhere**: `Instance.matrix` must be a `torch.Tensor`; numpy arrays passed raw to
    `Instance(...)` will break at the first `.shape`/`.device`-dependent op (e.g. `_max_off_diag`
    uses `self.matrix.device`). Use `qubosolver.matrix.as_tensor`.
14. **GLPK dependency**: `negative_bitflip.apply` imports `swiglpk` lazily; with negative
    off-diagonals and `preprocessing=True` it must be installed (it is a hard dependency in the
    wheel metadata). `cplex` is optional and only needed for `algorithm="cplex"`.
15. **`check_consistency`** is `O(k·n²)`; `analysis.to_dataframe` calls it with `throw=True`.
16. **Suboptimal-backend warning**: passing an explicit `backend_type` to `LocalEmulator` that
    does not match the size-based recommendation prints a `UserWarning` on every `run`.
17. **Register coordinates** are adimensional (min distance ≈ 1); physical µm positions only
    exist in `program.compiled_sequence.register` after `compile`.
18. **Bayesian search exceptions are swallowed**: `_run_simulation` catches *all* exceptions,
    prints `"Simulation failed: ..."` and returns an empty solution (objective 1e4), so a
    misconfigured device silently degrades to a 20-evaluation no-op; if the final re-run is empty,
    `_QuboSolverQuantum.solve` falls back to `_execute(drive, embedding)` with the best drive.
