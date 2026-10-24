"""Baseline: the SAME QUBO through Pasqal's ``qubo-solver`` (0.9.1), scored by OUR scorer.

    python -m cvrp.baseline --instance instances/S_seed00.json --formulation onehot-ub \
        --mode both --shots 500 --seed 0 --out out/m1/
    python -m cvrp.baseline ... --dry-run        # build instance/config/embedding/drive/compiled
                                                # sequence, print what would run, no emulation

Two modes, one record each (``cvrp.run.save_record`` schema, see ``cvrp.run.run_one``):

(a) quantum   method ``<formulation>+qubosolver-quantum``
    qubo-solver's own embedder (BLaDE), its own drive shaper (local-energy-scale) and its
    own emulator selection, run on Pulser's ``AnalogDevice`` (the brief's envelope) unless
    ``--device dmm`` asks for qubo-solver's default ``AnalogDeviceWithDMM`` (then the record
    says ``dmm: true`` and ``device_faithful: false``; the brief wants DMM results labelled
    and compared separately).  RAW emulator counts are scored with ``cvrp.metrics.score_counts``
    exactly like our pipeline.  qubo-solver's built-in post-processing (a greedy single-flip
    sweep over every sample, ``BaseSolver._post_process``) is a *repair* in our vocabulary,
    so it is OFF by default and available as the labelled ``--repair qs-postprocess``.

(b) classical method ``<formulation>+qubosolver-cplex`` or ``...+qubosolver-sa``
    qubo-solver's classical solver (CPLEX when ``import cplex`` works, else simulated
    annealing), one bitstring, scored as the single-shot histogram ``{bits: 1}``.

How qubo-solver's quantum pipeline differs from ours (verified in the installed source)
------------------------------------------------------------------------------------
* Objective and convention.  ``qubosolver.Instance(matrix)`` minimises ``cost(x) = x^T M x``
  with ``M`` a full symmetric matrix (``types/instance.py`` docstring; ``utils/_costs.py:34``
  ``torch.linalg.multi_dot([x, Q, x])``), i.e. both triangles are summed, same as ours.  Our
  ``E = x^T Q x + offset`` therefore maps to ``M = Q`` with the constant ``offset`` dropped
  (``to_qubosolver_matrix``).  Internally they read the physics as ``U_ij <-> M_ij`` and
  ``delta_i <-> -M_ii/2`` (``utils/quantum.py::extract_qubo``: ``Q = 1/r^6; Q_ii = -2 delta_i``;
  ``drive_shaping/local_energy_scale.py:133`` ``d = -0.5 * diag(M)``), while we use
  ``J_ij = 2 Q_ij <-> U_ij`` and ``h_i = Q_ii <-> -delta_i`` (``cvrp.common``): the two are the
  same Hamiltonian up to the overall energy scale, which the compiler fixes anyway.
* Embedding.  Default ``EmbeddingConfig(algorithm="blade")`` -> ``embedding/blade.py::embed``:
  MDS initialisation (``d_ij = M_ij^(-1/6)``, sklearn smacof) then qoolqit's BLaDE force
  simulation in 6->5->4->3->2->2->2 dimensions, 200 steps per round, targeting
  ``1/r_ij^6 ~ M_ij`` with the device's ``max_radial/min_distance`` ratio (38/5 = 7.6 for
  AnalogDevice) as the only device constraint.  Output coordinates are adimensional
  (min pairwise distance ~ 1); qubit ids are ``0..n-1`` in variable order
  (qoolqit ``blade.py:528-531`` builds the adjacency with ``nodelist=range(n)`` and
  ``matrix_embedder.py:89`` ``DataGraph.from_coordinates(positions)``).  Ours
  (``cvrp.embed``) optimises positions directly in µm on the device, refuses J <= 0, and
  reports a fidelity (relative Frobenius error of the off-diagonal couplings).  No
  embedding-error number is reported by qubo-solver; we compute one from
  ``qubosolver.extract_qubo(register, drive)`` (their own reconstruction, adimensional) and
  another from the compiled physical register in our convention (µm, C6/r^6).
* Drive shaping.  Default ``DriveShapingConfig(algorithm="local_energy_scale", kappa=0.1)``
  -> ``drive_shaping/local_energy_scale.py::build_drive``: ``Omega(t)`` is a PCHIP
  ``InterpolatedWaveform`` through ``(~0, Omega_max, Omega_max, ~0)`` and the global detuning a
  PCHIP through ``(delta_0, delta_0, delta_T, delta_T)`` with ``delta_0 = -max|d_i|``,
  ``delta_T = mean(d_i)`` (no DMM) and ``Omega_max = kappa * mean_i(|d_i| + sum_j V_ij)``.
  The target detunings are then CLAMPED to ``|d| <= 10 * Omega_max * (1 - 1e-3)`` by rescaling
  ``d`` (``local_energy_scale.py:179-190``): with kappa = 0.1 this usually shrinks the
  diagonal relative to the couplings, which the record exposes as
  ``embedding.linear_scale_ratio`` (realised h / target h).  Both waveforms span the
  device's maximum duration and the compiler pins the sequence to
  ``0.99 * 6000 ns = 5940 ns`` (``analog_quantum_sampling.py:79``).  Ours (``cvrp.pulse``)
  is a three-point Omega ramp plus a linear delta sweep of chosen duration, with
  ``delta_final = alpha * A`` exactly (no clamp), validated by Pulser.
* Compilation.  ``solving/quantum/analog_quantum_sampling.py::compile`` ->
  ``qoolqit.QuantumProgram.compile_to(device, profile=MAX_ENERGY, device_max_duration_ratio=0.99)``:
  either ``Omega_max`` is mapped to the device's max amplitude (2pi*2 rad/µs) or the minimum
  pairwise distance to 5 µm, whichever binds (``qoolqit/execution/compilation_functions.py``),
  then a Pulser ``Sequence`` on the real ``AnalogDevice`` with ``with_automatic_layout``.
  Bit ``i`` of a returned bitstring is qubit ``qubit_ids[i]`` of that Pulser register, whose
  keys are ``str(q)`` in qoolqit order (``compilation_functions.py::_build_register``); we
  verify the ids are a permutation of ``0..n-1`` and reorder if needed, so bit i = x_i.
* Emulation.  ``qubosolver.LocalEmulator(num_shots)`` -> ``AutoLocalEmulatorBackend``
  (``types/backends.py``): QutipBackendV2 (< 15 atoms), emu-sv (15-25), emu-mps (>= 26), each
  configured with only the ``BitStrings`` observable; ``job.results().final_bitstrings`` is
  the ``{bitstring: count}`` histogram we score (``--backend`` forces one of them).
* Pre/post-processing.  ``SolverConfig`` defaults ``preprocessing=True`` (classical variable
  fixing + bit-flip ILP for negative couplings) and ``postprocessing=True`` (greedy sweep).
  For a fair raw-vs-raw comparison both are OFF in mode (a) (``--pipeline raw``, default).
  ``--pipeline default`` runs ``Solver(...).solve()`` untouched (pre + post on): the returned
  bitstrings are then post-processed local minima, so the record carries them under
  ``metrics_repaired`` with ``repair="qubosolver-builtin"`` and ``metrics_raw: null``.
* Randomness.  ``qubosolver.utils._random.manual_seed(seed)`` seeds numpy (BLaDE, MDS), torch
  (classical solvers, ``torch_rng("global")``) and ``random``; the emulators' own sampling RNG
  is not controlled by qubo-solver.

Everything torch/qubo-solver related is imported lazily, so this module imports and its
pure-python helpers (matrix conversion, counts conversion, labels) are tested without torch.
Errors that would hide a wrong comparison (asymmetric Q, convention mismatch on random
bitstrings, bitstring length / ordering, cost mismatch of the classical solution, trivial
short-circuit) RAISE; nice-to-have diagnostics are wrapped and recorded as notes.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import Formulation, Instance, bits_to_str, str_to_bits
from .metrics import ModelOptimum, model_optimum, score_counts
from .model import get_formulation
from .run import energy_descent_repair, parse_kv, save_record

QUBOSOLVER_EXPECTED_VERSION = "0.9.1"
QS_DEFAULT_SEQUENCE_DURATION_NS = 6000     # what BaseSolver._submit passes to compile()
REPAIRS = ("none", "qs-postprocess", "energy-descent")
PIPELINES = ("raw", "default")
DEVICES = ("analog", "dmm")
EMBEDDERS = ("blade", "greedy_layout")
DRIVES = ("local_energy_scale", "proportional_diagonal", "bayesian_search")
CLASSICAL = ("auto", "cplex", "simulated_annealing", "tabu_search")


class ConventionError(RuntimeError):
    """Raised when qubo-solver's numbers disagree with ours (wrong Q convention, bit order,
    bitstring length, ...).  Never swallowed: a baseline scored on the wrong QUBO is worse
    than no baseline."""


# --------------------------------------------------------------------------- #
# Pure-python helpers (tested without torch)
# --------------------------------------------------------------------------- #
def to_qubosolver_matrix(Q: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """Our ``E(x) = x^T Q x + offset`` (Q symmetric) -> the matrix for ``qubosolver.Instance``.

    qubo-solver minimises ``x^T M x`` with ``M`` taken as a full symmetric matrix (both
    triangles summed), so ``M = Q``; the constant offset is dropped and must be added back
    to any cost qubo-solver reports (``energy_from_qubosolver_cost``).  Raises on a
    non-square or non-symmetric input instead of silently symmetrising: an upper-triangular
    Q would halve every coupling under this convention."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ConventionError(f"Q must be square, got shape {Q.shape}")
    if not np.allclose(Q, Q.T, atol=atol, rtol=0):
        raise ConventionError("Q must be symmetric (cvrp.common convention); refusing to guess qubo-solver's reading of it")
    if not np.all(np.isfinite(Q)):
        raise ConventionError("Q has non-finite entries")
    return Q.copy()


def qubosolver_cost(M: np.ndarray, x: Sequence[int]) -> float:
    """qubo-solver's objective, re-implemented: ``x^T M x`` (``qubosolver/utils/_costs.py:34``,
    ``torch.linalg.multi_dot([x, Q, x])``; docstring example ``Q=[[1,-1],[-1,2]], x=[1,0] -> 1``).
    Used by the tests; at run time the real ``Instance.cost`` is cross-checked against ours."""
    x = np.asarray(x, dtype=float)
    M = np.asarray(M, dtype=float)
    return float(x @ M @ x)


def energy_from_qubosolver_cost(cost: float, offset: float) -> float:
    return float(cost) + float(offset)


def method_label(formulation: str, mode: str, classical_algorithm: Optional[str] = None,
                 repair: str = "none", pipeline: str = "raw") -> str:
    """``onehot-ub+qubosolver-quantum``, ``onehot-ub+qubosolver-cplex``, ``onehot-ub+qubosolver-sa``,
    with ``+repair:<name>`` appended like ``cvrp.run.method_label`` when a labelled repair is on."""
    if mode == "quantum":
        base = f"{formulation}+qubosolver-quantum" + ("-default" if pipeline == "default" else "")
    elif mode == "classical":
        short = {"cplex": "cplex", "simulated_annealing": "sa", "tabu_search": "tabu",
                 "random_sampling": "random"}.get(classical_algorithm or "", classical_algorithm or "classical")
        base = f"{formulation}+qubosolver-{short}"
    else:
        raise ValueError(f"mode must be quantum or classical, got {mode!r}")
    return base if repair == "none" else f"{base}+repair:{repair}"


def cplex_available() -> bool:
    try:
        import cplex  # noqa: F401
        return True
    except Exception:  # noqa: BLE001  (ImportError or a licence/ABI error: either way no cplex)
        return False


def choose_classical_algorithm(prefer: str = "auto") -> str:
    """``cplex`` if importable else ``simulated_annealing`` (the brief's rule); an explicit
    name is returned as is (validated against qubo-solver's ``ClassicalConfig`` choices)."""
    if prefer == "auto":
        return "cplex" if cplex_available() else "simulated_annealing"
    if prefer not in ("cplex", "simulated_annealing", "tabu_search", "random_sampling"):
        raise ValueError(f"unknown qubo-solver classical algorithm {prefer!r}")
    return prefer


def _ids_to_positions(qubit_ids: Sequence[Any], n: int) -> List[int]:
    """Map qubit ids (``"0".."n-1"`` or ints, any order) to variable indices.  Raises when the
    ids are not a permutation of ``range(n)``: then bit i cannot be identified with x_i."""
    try:
        pos = [int(str(q)) for q in qubit_ids]
    except ValueError:
        raise ConventionError(f"cannot map qubit ids {list(qubit_ids)!r} to variable indices") from None
    if sorted(pos) != list(range(n)):
        raise ConventionError(f"qubit ids {list(qubit_ids)!r} are not a permutation of 0..{n - 1}")
    return pos


def counts_from_final_bitstrings(final_bitstrings: Dict[str, int], n_vars: int,
                                 qubit_ids: Optional[Sequence[Any]] = None) -> Dict[str, int]:
    """Pulser ``Results.final_bitstrings`` (``{bitstring: count}``, bit i = ``qubit_ids[i]``,
    ``'1'`` = Rydberg) -> our counts dict with bit i = variable i.  Validates length and
    alphabet, reorders when the register's qubit ids are a permutation of ``0..n-1``."""
    if not final_bitstrings:
        raise ConventionError("emulator returned no bitstrings")
    perm = _ids_to_positions(qubit_ids, n_vars) if qubit_ids is not None else list(range(n_vars))
    identity = perm == list(range(n_vars))
    out: Dict[str, int] = {}
    for bits, cnt in final_bitstrings.items():
        bits = str(bits)
        if len(bits) != n_vars:
            raise ConventionError(f"bitstring {bits!r} has length {len(bits)}, expected n_vars={n_vars}")
        if set(bits) - {"0", "1"}:
            raise ConventionError(f"bitstring {bits!r} is not over {{0,1}}")
        if not identity:
            x = ["0"] * n_vars
            for i, c in enumerate(bits):
                x[perm[i]] = c
            bits = "".join(x)
        out[bits] = out.get(bits, 0) + int(cnt)
    return out


def counts_from_solution(solution: Any, n_vars: int) -> Dict[str, int]:
    """``qubosolver.Solution`` (iterates ``Candidate`` with ``.string``/``.count``) -> counts.
    Candidates with ``count == 0`` (e.g. a solver that reports a bitstring without sampling it)
    are counted once so the scorer sees them."""
    out: Dict[str, int] = {}
    for cand in solution:
        bits = str(cand.string)
        if len(bits) != n_vars or set(bits) - {"0", "1"}:
            raise ConventionError(f"solution bitstring {bits!r} is not a {n_vars}-bit string")
        cnt = int(cand.count) if int(cand.count) > 0 else 1
        out[bits] = out.get(bits, 0) + cnt
    if not out:
        raise ConventionError("qubo-solver returned an empty Solution")
    return out


def fit_alpha(U: np.ndarray, J: np.ndarray) -> float:
    """Least-squares scale alpha with ``U ~ alpha * J`` over the pairs with ``J > 0``."""
    mask = np.triu(J > 0, k=1)
    denom = float((J[mask] ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((U[mask] * J[mask]).sum() / denom)


def coupling_fidelity(U: np.ndarray, J: np.ndarray) -> Dict[str, Any]:
    """Our fidelity numbers (``cvrp.embed.FidelityReport`` definitions) for realised couplings
    ``U`` (any units) against targets ``J``: alpha, relative Frobenius error of the off-diagonal,
    worst relative pair error (J > 0), largest spurious coupling (J == 0) in J units."""
    U = np.asarray(U, float).copy()
    J = np.asarray(J, float).copy()
    np.fill_diagonal(U, 0.0)
    np.fill_diagonal(J, 0.0)
    alpha = fit_alpha(U, J)
    if not np.isfinite(alpha) or alpha <= 0:
        return {"alpha": alpha, "rel_frobenius_offdiag": None, "worst_pair_rel_error": None,
                "worst_pair": None, "max_spurious_coupling": None, "spurious_pair": None}
    R = U / alpha
    rel_frob = float(np.linalg.norm(R - J) / max(np.linalg.norm(J), 1e-300))
    iu = np.triu_indices(len(J), k=1)
    pos = J[iu] > 0
    worst = worst_pair = None
    if pos.any():
        rel = np.abs(R[iu][pos] - J[iu][pos]) / J[iu][pos]
        k = int(np.argmax(rel))
        worst = float(rel[k])
        worst_pair = [int(iu[0][pos][k]), int(iu[1][pos][k])]
    spur = spur_pair = None
    if (~pos).any():
        k = int(np.argmax(R[iu][~pos]))
        spur = float(R[iu][~pos][k])
        spur_pair = [int(iu[0][~pos][k]), int(iu[1][~pos][k])]
    return {"alpha": alpha, "rel_frobenius_offdiag": rel_frob, "worst_pair_rel_error": worst,
            "worst_pair": worst_pair, "max_spurious_coupling": spur, "spurious_pair": spur_pair}


def _versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {"python": platform.python_version()}
    for mod in ("pulser", "qubosolver", "qoolqit", "torch", "pulser_simulation", "emu_sv", "emu_mps"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def _dataclass_to_jsonable(obj: Any) -> Any:
    """Record a qubo-solver config dataclass without dragging torch objects into JSON."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _dataclass_to_jsonable(v) for k, v in obj.items()}
    return type(obj).__name__


# --------------------------------------------------------------------------- #
# qubo-solver glue (torch needed from here on)
# --------------------------------------------------------------------------- #
def _import_qubosolver():
    try:
        import torch  # noqa: F401
        import qubosolver
    except ImportError as e:
        raise ImportError("cvrp.baseline needs torch + qubo-solver (the Windows venv): " + str(e)) from e
    v = getattr(qubosolver, "__version__", "?")
    if v != QUBOSOLVER_EXPECTED_VERSION:
        warnings.warn(f"qubo-solver {v} installed, module written against {QUBOSOLVER_EXPECTED_VERSION}; "
                      "API drift possible", stacklevel=2)
    return qubosolver


def _seed_everything(seed: int) -> Dict[str, Any]:
    """Seed numpy (BLaDE/MDS), torch (classical solvers) and random, the way qubo-solver's own
    helper does; fall back to doing it by hand if the helper moved."""
    try:
        from qubosolver.utils._random import manual_seed
        manual_seed(int(seed))
        return {"seeder": "qubosolver.utils._random.manual_seed", "seed": int(seed)}
    except Exception as e:  # noqa: BLE001
        import random
        import torch
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        random.seed(int(seed))
        return {"seeder": "np/torch/random manual (fallback)", "seed": int(seed), "note": repr(e)}


def _qs_instance(Q: np.ndarray):
    from qubosolver import Instance as QSInstance
    from qubosolver import matrix as qs_matrix
    M = to_qubosolver_matrix(Q)
    return QSInstance(qs_matrix.as_tensor(M))


def check_convention(qs_inst: Any, f: Formulation, seed: int = 0, n_random: int = 32) -> Dict[str, Any]:
    """Cross-check qubo-solver's ``Instance.cost`` against our ``Formulation.energy`` on random
    bitstrings (plus 0^n, 1^n, one-hot rows if any).  Raises ``ConventionError`` on mismatch
    beyond float32 rounding."""
    from qubosolver import bitstring as qs_bitstring
    n = f.n_vars
    Q = f.Q()
    offset = float(f.offset)
    scale = max(1.0, float(np.abs(Q).sum()))          # worst-case |x^T Q x|
    tol = 1e-5 * scale                                 # float32 relative precision x safety
    rng = np.random.default_rng(seed)
    X = [np.zeros(n, int), np.ones(n, int)] + [rng.integers(0, 2, n) for _ in range(n_random)]
    worst = 0.0
    for x in X:
        ours = f.energy(x) - offset
        theirs = float(qs_inst.cost(qs_bitstring.tensor([int(b) for b in x])))
        worst = max(worst, abs(ours - theirs))
        if abs(ours - theirs) > tol:
            raise ConventionError(
                f"qubo-solver cost {theirs!r} != our energy-offset {ours!r} on x={bits_to_str(x)} "
                f"(tol {tol:.3g}); Q convention or dtype mismatch")
    return {"n_checked": len(X), "max_abs_diff": worst, "tol": tol,
            "qs_float_dtype": str(qs_inst.matrix.dtype)}


def _check_not_trivial(qs_inst: Any) -> None:
    """qubo-solver short-circuits all-non-negative / all-non-positive / diagonal matrices
    (``solving/classical/trivial_solution_search.py``) and never touches the quantum stack.
    That would be a comparison of nothing, so raise."""
    from qubosolver import solving
    trivial = solving.trivial_solution_search.solve(qs_inst)
    if len(trivial) > 0:
        raise ConventionError("qubo-solver classifies this QUBO as trivial (all coefficients of one sign or "
                              "diagonal) and would return one bitstring without any quantum sampling")


def _negative_offdiag(Q: np.ndarray) -> bool:
    off = Q[~np.eye(len(Q), dtype=bool)]
    return bool((off < 0).any())


def _device(kind: str):
    import qoolqit
    if kind == "analog":
        return qoolqit.AnalogDevice(), False
    if kind == "dmm":
        return qoolqit.AnalogDeviceWithDMM(), True
    raise ValueError(f"--device must be one of {DEVICES}, got {kind!r}")


def _backend(shots: int, name: str):
    from qubosolver import LocalEmulator
    if name == "auto":
        return LocalEmulator(num_shots=int(shots))
    if name == "qutip":
        from pulser_simulation import QutipBackendV2 as BT
    elif name == "emu-sv":
        from emu_sv import SVBackend as BT
    elif name == "emu-mps":
        from emu_mps import MPSBackend as BT
    else:
        raise ValueError(f"--backend must be auto|qutip|emu-sv|emu-mps, got {name!r}")
    return LocalEmulator(backend_type=BT, num_shots=int(shots))


def _quantum_config(backend, device, embedder: str, drive: str, kappa: Optional[float],
                    preprocessing: bool, postprocessing: bool, bayes_calls: int):
    from qubosolver import DriveShapingConfig, EmbeddingConfig, QuantumSolvingConfig, SolverConfig
    if embedder not in EMBEDDERS:
        raise ValueError(f"--embedder must be one of {EMBEDDERS}")
    if drive not in DRIVES:
        raise ValueError(f"--drive must be one of {DRIVES}")
    ds: Dict[str, Any] = {"algorithm": drive, "bayesian_search_n_calls": int(bayes_calls)}
    if kappa is not None:
        ds["local_energy_scale_kappa" if drive == "local_energy_scale" else "proportional_diagonal_kappa"] = float(kappa)
    return SolverConfig(
        config_name="cvrp-baseline",
        solving=QuantumSolvingConfig(embedding=EmbeddingConfig(algorithm=embedder),
                                     drive_shaping=DriveShapingConfig(**ds),
                                     backend=backend, device=device),
        preprocessing=bool(preprocessing), postprocessing=bool(postprocessing))


def _pulser_register_coords(seq) -> Tuple[List[str], np.ndarray]:
    reg = seq.register
    ids = [str(q) for q in reg.qubit_ids]
    coords = np.array([np.asarray(reg.qubits[q], dtype=float).ravel()[:2] for q in reg.qubit_ids])
    return ids, coords


def _describe_compiled(program, Q: np.ndarray, register, drive, device, notes: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Embedding + pulse dicts for the record, from the compiled Pulser sequence (µm, rad/µs,
    ns) and the adimensional qoolqit objects.  Everything here is diagnostic: failures are
    recorded as notes, never raised (the counts are what matters)."""
    emb: Dict[str, Any] = {"embedder": None, "notes": notes}
    pulse: Dict[str, Any] = {}
    seq = program.compiled_sequence
    n = Q.shape[0]
    J = 2.0 * Q.copy()
    np.fill_diagonal(J, 0.0)
    h = np.diag(Q).copy()
    # --- physical register -------------------------------------------------- #
    try:
        ids, coords_um = _pulser_register_coords(seq)
        order = _ids_to_positions(ids, n)                # variable index of each atom
        coords_var = np.zeros_like(coords_um)
        coords_var[order] = coords_um                    # row i = variable i
        emb["qubit_ids"] = ids
        emb["coords_um"] = coords_var.tolist()
        C6 = float(seq.device.interaction_coeff)
        emb["C6"] = C6
        d = np.linalg.norm(coords_var[:, None, :] - coords_var[None, :, :], axis=-1)
        with np.errstate(divide="ignore"):
            U = np.where(d > 0, C6 / np.maximum(d, 1e-12) ** 6, 0.0)
        np.fill_diagonal(U, 0.0)
        fid = coupling_fidelity(U, J)
        emb["alpha"] = fid["alpha"]
        emb["U_realized"] = U.tolist()
        emb["fidelity"] = fid
        emb["min_distance_um"] = float(d[np.triu_indices(n, 1)].min()) if n > 1 else None
        emb["max_radial_um"] = float(np.linalg.norm(coords_var - coords_var.mean(0), axis=1).max())
        emb["max_radial_um_from_origin"] = float(np.linalg.norm(coords_var, axis=1).max())
    except Exception as e:  # noqa: BLE001
        notes.append(f"physical register diagnostics failed: {e!r}")
    # --- adimensional register / their own reconstruction ------------------- #
    try:
        ids_adim = list(register.qubits_ids)
        order = _ids_to_positions(ids_adim, n)
        ca = np.array([np.asarray(register.qubits[q], dtype=float).ravel()[:2] for q in ids_adim])
        coords_adim = np.zeros_like(ca)
        coords_adim[order] = ca
        emb["coords_adim"] = coords_adim.tolist()
        from qubosolver import extract_qubo
        Qr = extract_qubo(register, drive).matrix.detach().cpu().numpy().astype(float)
        Qr_var = np.zeros_like(Qr)
        Qr_var[np.ix_(order, order)] = Qr
        Mo = Q.copy(); np.fill_diagonal(Mo, 0.0)
        fid_qs = coupling_fidelity(Qr_var, Mo)          # their convention: 1/r^6 ~ M_ij
        emb["qubosolver_extract_qubo"] = {
            "scale_adim_per_qubo_unit": fid_qs["alpha"],
            "rel_frobenius_offdiag": fid_qs["rel_frobenius_offdiag"],
            "worst_pair_rel_error": fid_qs["worst_pair_rel_error"],
            "diag_realized_adim": np.diag(Qr_var).tolist(),
            "note": "qubosolver.extract_qubo(register, drive): M_ij = 1/r_ij^6, M_ii = -2 delta_i(T), adimensional",
        }
    except Exception as e:  # noqa: BLE001
        notes.append(f"adimensional register / extract_qubo diagnostics failed: {e!r}")
    # --- pulse ------------------------------------------------------------- #
    try:
        pulse["duration_ns"] = int(seq.get_duration())
        from pulser.sampler import sample
        ch = sample(seq).channel_samples["rydberg"]
        amp = np.asarray(ch.amp, dtype=float)
        det = np.asarray(ch.det, dtype=float)
        pulse.update(omega_max=float(amp.max()), omega_mean=float(amp.mean()),
                     delta_init=float(det[0]), delta_final=float(det[-1]),
                     delta_min=float(det.min()), delta_max=float(det.max()))
        if "alpha" in emb and emb["alpha"] and np.isfinite(emb["alpha"]):
            h_real = -float(det[-1]) / emb["alpha"]
            emb["delta_final"] = float(det[-1])
            emb["h"] = h.tolist()
            emb["h_realized_uniform"] = h_real
            emb["linear_terms_uniform"] = bool(np.allclose(h, h[0], rtol=1e-9, atol=1e-9))
            hm = float(h.mean())
            emb["linear_scale_ratio"] = (h_real / hm) if abs(hm) > 1e-12 else None
    except Exception as e:  # noqa: BLE001
        notes.append(f"pulse sampling diagnostics failed: {e!r}")
    try:
        pulse["adim"] = {"duration": float(drive.duration), "omega_max": float(drive.amplitude.max()),
                         "delta_0": float(drive.detuning(0.0)), "delta_T": float(drive.detuning(drive.duration)),
                         "dmm": drive.dmm is not None}
        if drive.dmm is not None:
            pulse["adim"]["dmm_weights"] = {str(k): float(v) for k, v in drive.dmm.weights.items()}
            pulse["adim"]["dmm_final_detuning"] = float(drive.dmm.waveform(drive.duration))
    except Exception as e:  # noqa: BLE001
        notes.append(f"adimensional drive diagnostics failed: {e!r}")
    pulse["waveform"] = ("qoolqit InterpolatedWaveform (PCHIP): Omega through (~0, Omega_max, Omega_max, ~0), "
                         "global delta through (delta_0, delta_0, delta_T, delta_T); compiled with MAX_ENERGY, "
                         "device_max_duration_ratio=0.99")
    try:
        pulse["pulser_validated"] = bool(seq.to_abstract_repr())
    except Exception as e:  # noqa: BLE001
        pulse["pulser_validated"] = False
        notes.append(f"to_abstract_repr failed: {e!r}")
    try:
        emb["device_specs_adim"] = {k: (float(v) if v is not None else None) for k, v in device.specs.items()}
    except Exception as e:  # noqa: BLE001
        notes.append(f"device.specs failed: {e!r}")
    return emb, pulse


def _base_record(inst: Instance, f: Formulation, method: str, backend: str, seed: int, shots: int,
                 repair: str, dmm: bool) -> Dict[str, Any]:
    return {
        "instance": inst.name, "tier": inst.tier, "instance_seed": inst.seed,
        "formulation": f.describe(), "n_vars": f.n_vars, "atoms": None,
        "method": method, "backend": backend, "seed": int(seed), "shots": int(shots),
        "repair": repair, "dmm": bool(dmm), "device": None, "device_faithful": False,
        "embedding": None, "pulse": None, "sequence_duration_ns": None,
        "counts": None, "metrics_raw": None, "backend_params": {}, "wall_s": None,
        "versions": _versions(), "baseline": "qubo-solver",
    }


# --------------------------------------------------------------------------- #
# (a) quantum
# --------------------------------------------------------------------------- #
def run_quantum(inst: Instance, formulation: str, shots: int, seed: int,
                weights: Optional[Dict[str, float]] = None, backend: str = "auto", device: str = "analog",
                embedder: str = "blade", drive: str = "local_energy_scale", kappa: Optional[float] = None,
                pipeline: str = "raw", repair: str = "none", bayes_calls: int = 20,
                allow_negative_couplings: bool = False, dry_run: bool = False,
                mopt: Optional[ModelOptimum] = None, verbose: bool = True) -> Dict[str, Any]:
    """qubo-solver's quantum pipeline on our QUBO, raw counts scored by cvrp.metrics."""
    t0 = time.perf_counter()
    if pipeline not in PIPELINES:
        raise ValueError(f"pipeline must be one of {PIPELINES}")
    if repair not in REPAIRS:
        raise ValueError(f"repair must be one of {REPAIRS}")
    f = get_formulation(formulation, inst, **(weights or {}))
    Q = f.Q()
    n = f.n_vars
    if n > 80:
        raise ValueError(f"{n} variables > qubo-solver's 80-atom cap")
    if _negative_offdiag(Q):
        msg = ("QUBO has negative off-diagonal couplings: qubo-solver would bit-flip (preprocessing) or ZERO "
               "them before embedding, i.e. solve a different QUBO")
        if not allow_negative_couplings:
            raise ConventionError(msg + "; pass allow_negative_couplings=True to run anyway (recorded)")
        warnings.warn(msg, stacklevel=2)

    qubosolver = _import_qubosolver()
    import qoolqit
    from qubosolver import Solution, Solver, solving
    notes: List[str] = []
    seed_info = _seed_everything(seed)
    qs_inst = _qs_instance(Q)
    conv = check_convention(qs_inst, f, seed=seed)
    _check_not_trivial(qs_inst)

    dev, dmm = _device(device)
    be = _backend(shots, backend)
    use_default = pipeline == "default"
    cfg = _quantum_config(be, dev, embedder, drive, kappa, preprocessing=use_default,
                          postprocessing=use_default, bayes_calls=bayes_calls)
    if dmm:
        notes.append("device=AnalogDeviceWithDMM (qubo-solver's default, virtual device with a DMM channel): "
                     "not the brief's AnalogDevice envelope; compare separately")
    if _negative_offdiag(Q):
        notes.append("negative couplings present: qubo-solver zeroed/bit-flipped them (different QUBO)")

    rec = _base_record(inst, f, method_label(formulation, "quantum", repair=repair, pipeline=pipeline),
                       "qubosolver-quantum", seed, shots, repair, dmm)
    rec["backend_params"] = {
        "backend": "qubosolver-quantum", "shots": int(shots), "requested_emulator": backend,
        "pipeline": pipeline, "preprocessing": use_default, "postprocessing": use_default,
        "qubosolver_config": _dataclass_to_jsonable(cfg), "device_name": dev.name,
        "compiler_profile": "MAX_ENERGY", "default_sequence_duration_ns": QS_DEFAULT_SEQUENCE_DURATION_NS,
        "seed_info": seed_info, "convention_check": conv,
        "qubosolver_version": getattr(qubosolver, "__version__", None),
        "qoolqit_version": getattr(qoolqit, "__version__", None),
    }
    rec["device"] = dev.name

    solver = Solver(qs_inst, cfg)
    if verbose:
        print(f"[{inst.name}] {formulation}: {n} vars -> qubo-solver quantum ({embedder} + {drive}"
              f"{'' if kappa is None else f', kappa={kappa}'}) on {dev.name}, pipeline={pipeline}, "
              f"shots={shots}, emulator={backend}")

    if use_default:
        # qubo-solver untouched: trivial check -> variable fixing -> embed -> drive -> sample -> lift -> greedy sweep
        if dry_run:
            rec["counts"] = None
            rec["embedding"] = {"embedder": embedder, "notes": notes + ["dry run: Solver.solve() not called"]}
            rec["wall_s"] = time.perf_counter() - t0
            return rec
        t1 = time.perf_counter()
        solution = solver.solve()
        rec["backend_params"]["wall_s"] = time.perf_counter() - t1
        counts = counts_from_solution(solution, n)
        register = getattr(solver._solver, "_cached_register", None)
        drv = getattr(solver._solver, "_cached_drive", None)
        rec["atoms"] = int(register.n_qubits) if register is not None else n
        if register is not None and drv is not None:
            try:
                program = solving.analog_quantum_sampling.compile(register, drv, dev,
                                                                  default_sequence_duration=QS_DEFAULT_SEQUENCE_DURATION_NS)
                emb, pulse = _describe_compiled(program, Q, register, drv, dev, notes)
                emb["embedder"] = embedder
                rec["embedding"], rec["pulse"] = emb, pulse
                rec["sequence_duration_ns"] = pulse.get("duration_ns")
            except Exception as e:  # noqa: BLE001
                notes.append(f"re-compiling cached register/drive for the record failed: {e!r}")
                rec["embedding"] = {"embedder": embedder, "notes": notes}
        else:
            notes.append("solver exposed no cached register/drive (variable fixing may have emptied the instance)")
            rec["embedding"] = {"embedder": embedder, "notes": notes}
        rec["repair"] = "qubosolver-builtin"
        rec["method"] = method_label(formulation, "quantum", repair="qubosolver-builtin", pipeline=pipeline)
        rec["counts"] = counts
        if mopt is None:
            mopt = model_optimum(f)
        m, _ = score_counts(counts, f, inst, mopt=mopt)
        rec["metrics_raw"] = None
        rec["metrics_repaired"] = m.to_dict()
        notes.append("pipeline=default: bitstrings are qubo-solver's post-processed (greedy single-flip sweep, "
                     "deduplicated) candidates, NOT raw shots -> scored under metrics_repaired only")
        rec["wall_s"] = time.perf_counter() - t0
        if verbose:
            print("   builtin-postprocessed: " + m.summary().replace("\n", "\n             "))
        return rec

    # ---- raw pipeline: the same calls BaseSolver._execute makes, with the raw histogram kept --- #
    t1 = time.perf_counter()
    register = solver._embedding()                       # BLaDE / greedy  (solver/solver.py:157)
    t_emb = time.perf_counter() - t1
    t1 = time.perf_counter()
    drv, pre_solution = solver._drive(register)          # drive shaper      (solver/solver.py:171)
    t_drv = time.perf_counter() - t1
    program = solving.analog_quantum_sampling.compile(   # == BaseSolver._submit (solver/_basesolver.py:101)
        register, drv, dev, default_sequence_duration=QS_DEFAULT_SEQUENCE_DURATION_NS)
    seq = program.compiled_sequence
    ids = [str(q) for q in seq.register.qubit_ids]
    _ids_to_positions(ids, n)                            # raises if bit i cannot be x_i
    rec["atoms"] = int(len(ids))
    rec["device"] = str(seq.device.name)
    emb, pulse = _describe_compiled(program, Q, register, drv, dev, notes)
    emb["embedder"] = embedder
    emb["wall_s_embed"] = t_emb
    emb["wall_s_drive"] = t_drv
    rec["embedding"], rec["pulse"] = emb, pulse
    rec["sequence_duration_ns"] = pulse.get("duration_ns")
    from pulser.devices import AnalogDevice as PulserAnalog
    from pulser.devices._device_datacls import Device as PulserRealDevice
    real_device = isinstance(seq.device, PulserRealDevice) and seq.device.name == PulserAnalog.name
    rec["device_faithful"] = bool(real_device and not dmm and pulse.get("pulser_validated", False)
                                  and emb.get("linear_terms_uniform", False))
    identity_order = ids == [str(i) for i in range(n)]
    if not identity_order:
        notes.append(f"compiled register qubit ids {ids} are a permutation of 0..n-1: counts were reordered to bit i = x_i")
        if repair == "qs-postprocess":
            raise ConventionError("qs-postprocess repair needs qubo-solver's Solution in variable order, but the "
                                  "compiled register permuted the qubits; rerun with --repair none or energy-descent")
    if verbose:
        fid = emb.get("fidelity") or {}
        print(f"   register: {rec['atoms']} atoms, min dist {emb.get('min_distance_um')} um, "
              f"alpha={emb.get('alpha')} rad/us per QUBO unit, rel_frob={fid.get('rel_frobenius_offdiag')} "
              f"worst pair={fid.get('worst_pair_rel_error')}")
        print(f"   pulse: {pulse.get('duration_ns')} ns, Omega_max={pulse.get('omega_max')}, "
              f"delta {pulse.get('delta_init')} -> {pulse.get('delta_final')} rad/us; "
              f"realised h / target h = {emb.get('linear_scale_ratio')}; validated={pulse.get('pulser_validated')}")
    if dry_run:
        rec["counts"] = None
        rec["wall_s"] = time.perf_counter() - t0
        return rec

    t1 = time.perf_counter()
    if drive == "bayesian_search" and len(pre_solution) > 0:
        # the Bayesian shaper already sampled with its best drive; qubo-solver keeps that solution
        notes.append("bayesian_search: counts are the shaper's final sampled solution (BaseSolver skips _execute)")
        results = None
        counts = counts_from_solution(pre_solution, n)
        solution = pre_solution
    else:
        job = be.run(program)
        results = job.results()
        fb = dict(results.final_bitstrings)              # {bitstring: count}, bit i = qubit_ids[i]
        counts = counts_from_final_bitstrings(fb, n, qubit_ids=ids)
        solution = Solution.from_results(results, qs_inst)
    rec["backend_params"]["wall_s"] = time.perf_counter() - t1
    try:
        rec["backend_params"]["emulator"] = type(be._backend).__name__
    except Exception:  # noqa: BLE001
        rec["backend_params"]["emulator"] = backend
    if sum(counts.values()) != shots:
        notes.append(f"emulator returned {sum(counts.values())} shots, {shots} requested")
    rec["counts"] = counts
    # cross-check qubo-solver's own costs for these samples against our energies (same bit order
    # only when the register kept 0..n-1; otherwise the reordered counts are the reference)
    if identity_order:
        try:
            offset = float(f.offset)
            worst = 0.0
            for cand in solution:
                e_ours = f.energy(str_to_bits(cand.string)) - offset
                worst = max(worst, abs(e_ours - float(cand.cost)))
            tol = conv["tol"]
            if worst > tol:
                raise ConventionError(f"qubo-solver cost of a sampled bitstring differs from ours by {worst} (tol {tol})")
            rec["backend_params"]["sample_cost_check_max_abs_diff"] = worst
        except ConventionError:
            raise
        except Exception as e:  # noqa: BLE001
            notes.append(f"sample cost cross-check skipped: {e!r}")

    if mopt is None:
        mopt = model_optimum(f)
    m_raw, _ = score_counts(counts, f, inst, mopt=mopt)
    rec["metrics_raw"] = m_raw.to_dict()
    if repair == "qs-postprocess":
        # exactly BaseSolver._post_process (solver/_basesolver.py:244): one greedy sweep per sample, deduplicated
        rep = solving.iterative_bitflip_local_search.solve(qs_inst, starts=solution, strategy="greedy_sweep",
                                                           max_iterations=1, time_limit=600.0)
        counts_rep = counts_from_solution(rep, n)
        m_rep, _ = score_counts(counts_rep, f, inst, mopt=mopt)
        rec["metrics_repaired"] = m_rep.to_dict()
        rec["counts_repaired"] = counts_rep
        rec["repair_detail"] = "qubosolver.solving.iterative_bitflip_local_search(strategy=greedy_sweep, max_iterations=1)"
    elif repair == "energy-descent":
        tr = energy_descent_repair(Q, float(f.offset))
        m_rep, _ = score_counts(counts, f, inst, mopt=mopt, transform=tr)
        rec["metrics_repaired"] = m_rep.to_dict()
        rec["repair_detail"] = "cvrp.run.energy_descent_repair (same as our pipeline's labelled repair)"
    rec["wall_s"] = time.perf_counter() - t0
    if verbose:
        print("   raw:      " + m_raw.summary().replace("\n", "\n             "))
        if repair != "none":
            print("   repaired: " + m_rep.summary().replace("\n", "\n             "))
    return rec


# --------------------------------------------------------------------------- #
# (b) classical
# --------------------------------------------------------------------------- #
def run_classical(inst: Instance, formulation: str, seed: int, weights: Optional[Dict[str, float]] = None,
                  algorithm: str = "auto", sa_iters: int = 20000, time_limit: float = float("inf"),
                  max_bitstrings: int = 1, preprocessing: bool = True, postprocessing: bool = True,
                  dry_run: bool = False, mopt: Optional[ModelOptimum] = None, verbose: bool = True) -> Dict[str, Any]:
    """qubo-solver's classical solver on our QUBO; its best bitstring scored as ``{bits: 1}``."""
    t0 = time.perf_counter()
    f = get_formulation(formulation, inst, **(weights or {}))
    Q = f.Q()
    n = f.n_vars
    algo = choose_classical_algorithm(algorithm)
    qubosolver = _import_qubosolver()
    from qubosolver import ClassicalSolvingConfig, Solver, SolverConfig
    seed_info = _seed_everything(seed)
    qs_inst = _qs_instance(Q)
    conv = check_convention(qs_inst, f, seed=seed)
    cfg = SolverConfig(config_name="cvrp-baseline-classical",
                       solving=ClassicalSolvingConfig(algorithm=algo, time_limit=float(time_limit),
                                                      max_iter=int(sa_iters), max_bitstrings=int(max_bitstrings)),
                       preprocessing=bool(preprocessing), postprocessing=bool(postprocessing))
    rec = _base_record(inst, f, method_label(formulation, "classical", classical_algorithm=algo),
                       "qubosolver-classical", seed, 1, "none", False)
    rec["backend_params"] = {
        "backend": "qubosolver-classical", "algorithm": algo, "cplex_available": cplex_available(),
        "qubosolver_config": _dataclass_to_jsonable(cfg), "preprocessing": bool(preprocessing),
        "postprocessing": bool(postprocessing), "seed_info": seed_info, "convention_check": conv,
        "qubosolver_version": getattr(qubosolver, "__version__", None), "shots": 1,
        "note": ("single start drawn from torch_rng('global'); SA/tabu: max_iter bit-flip proposals; "
                 "cplex: exact MIP (time_limit)"),
    }
    rec["device"] = "none (classical)"
    if verbose:
        print(f"[{inst.name}] {formulation}: {n} vars -> qubo-solver classical {algo} "
              f"(max_iter={sa_iters}, pre={preprocessing}, post={postprocessing})")
    if dry_run:
        rec["counts"] = None
        rec["wall_s"] = time.perf_counter() - t0
        return rec
    t1 = time.perf_counter()
    solution = Solver(qs_inst, cfg).solve()
    rec["backend_params"]["wall_s"] = time.perf_counter() - t1
    if len(solution) == 0:
        raise ConventionError("qubo-solver classical solver returned an empty Solution")
    cands = list(solution)
    best = min(cands, key=lambda c: float(c.cost))        # do not trust [0] (lifts do not re-sort)
    bits = str(best.string)
    if len(bits) != n:
        raise ConventionError(f"classical solution has {len(bits)} bits, expected {n}")
    offset = float(f.offset)
    e_ours = f.energy(str_to_bits(bits)) - offset
    if abs(e_ours - float(best.cost)) > conv["tol"]:
        raise ConventionError(f"classical best cost {best.cost} != our energy-offset {e_ours} for {bits}")
    counts = {bits: 1}
    rec["counts"] = counts
    rec["backend_params"]["n_candidates"] = len(cands)
    rec["backend_params"]["candidates"] = [{"bits": str(c.string), "cost": float(c.cost), "count": int(c.count)}
                                           for c in sorted(cands, key=lambda c: float(c.cost))[:10]]
    rec["backend_params"]["best_energy"] = energy_from_qubosolver_cost(best.cost, offset)
    if mopt is None:
        mopt = model_optimum(f)
    m, _ = score_counts(counts, f, inst, mopt=mopt)
    rec["metrics_raw"] = m.to_dict()
    rec["wall_s"] = time.perf_counter() - t0
    if verbose:
        print("   " + m.summary().replace("\n", "\n   "))
    return rec


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--formulation", required=True)
    ap.add_argument("--mode", default="both", choices=("quantum", "classical", "both"))
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="", help="penalty weights, e.g. A=12,lam2=3 (default: formulation's rule)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--dry-run", action="store_true", help="build instance/config (+ embedding/drive/compiled "
                    "sequence in quantum mode), print, write record without counts")
    # quantum knobs
    ap.add_argument("--backend", default="auto", help="auto | qutip | emu-sv | emu-mps (qubo-solver picks by size when auto)")
    ap.add_argument("--device", default="analog", choices=DEVICES, help="analog = Pulser AnalogDevice (brief); dmm = "
                    "qubo-solver's AnalogDeviceWithDMM (labelled dmm=true)")
    ap.add_argument("--embedder", default="blade", choices=EMBEDDERS)
    ap.add_argument("--drive", default="local_energy_scale", choices=DRIVES)
    ap.add_argument("--kappa", type=float, default=None, help="drive shaper kappa (default: qubo-solver's 0.1 / 1.0)")
    ap.add_argument("--pipeline", default="raw", choices=PIPELINES, help="raw = embed+drive+sample, no pre/post; "
                    "default = Solver.solve() untouched (post-processed candidates, labelled)")
    ap.add_argument("--repair", default="none", choices=REPAIRS, help="labelled repair scored separately")
    ap.add_argument("--bayes-calls", type=int, default=20)
    ap.add_argument("--allow-negative-couplings", action="store_true")
    # classical knobs
    ap.add_argument("--classical", default="auto", choices=CLASSICAL, help="auto = cplex if importable else simulated_annealing")
    ap.add_argument("--sa-iters", type=int, default=20000, help="max_iter for simulated_annealing / tabu_search")
    ap.add_argument("--time-limit", type=float, default=float("inf"))
    ap.add_argument("--no-classical-preprocessing", action="store_true")
    ap.add_argument("--no-classical-postprocessing", action="store_true")
    a = ap.parse_args(argv)

    inst = Instance.load(a.instance)
    weights = parse_kv(a.weights)
    f = get_formulation(a.formulation, inst, **weights)
    mopt = model_optimum(f)
    paths = []
    if a.mode in ("quantum", "both"):
        rec = run_quantum(inst, a.formulation, a.shots, a.seed, weights=weights, backend=a.backend, device=a.device,
                          embedder=a.embedder, drive=a.drive, kappa=a.kappa, pipeline=a.pipeline, repair=a.repair,
                          bayes_calls=a.bayes_calls, allow_negative_couplings=a.allow_negative_couplings,
                          dry_run=a.dry_run, mopt=mopt)
        if a.dry_run:
            print(json.dumps({k: rec[k] for k in ("method", "device", "atoms", "sequence_duration_ns", "pulse", "backend_params")},
                             indent=1, default=str))
        paths.append(save_record(rec, a.out))
    if a.mode in ("classical", "both"):
        rec = run_classical(inst, a.formulation, a.seed, weights=weights, algorithm=a.classical, sa_iters=a.sa_iters,
                            time_limit=a.time_limit, preprocessing=not a.no_classical_preprocessing,
                            postprocessing=not a.no_classical_postprocessing, dry_run=a.dry_run, mopt=mopt)
        if a.dry_run:
            print(json.dumps({k: rec[k] for k in ("method", "backend_params")}, indent=1, default=str))
        paths.append(save_record(rec, a.out))
    for p in paths:
        print(f"record -> {p}")


if __name__ == "__main__":
    main()
