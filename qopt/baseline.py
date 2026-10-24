"""Baseline: run the same QUBO through Pasqal's `qubo-solver` and score it
with exactly the same scorer as our pipeline (qopt.run.score_counts).

Two modes
---------
"solver"      : Solver(instance, config).solve()  -> Solution (pre-processed:
                variable fixing + bit-flip, post-processed: 1-flip local search).
                We score its candidates as counts (bitstring, count).
"functional"  : embed_for_device -> build_drive -> compile -> backend.run,
                i.e. qubo-solver's embedding + drive but RAW emulator counts,
                scored through our decode/repair.  Also returns the qoolqit
                register / drive / compiled pulser Sequence so we can plot and
                compare against ours, plus qubo-solver's own embedding error
                (via qubosolver.extract_qubo).

qubo-solver conventions (0.9.1): cost = x^T Q x, Q symmetric, off-diagonals
must be >= 0 (negatives are bit-flipped away or zeroed); physical H_f = Q/2 in
dimensionless units (Q_ij <-> U_ij, Q_ii <-> -2 delta_i); 80-variable cap;
emulator auto-selected by size (qutip <15, emu-sv 15-25, emu-mps >= 26).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .classical import ClassicalResult
from .model import Problem, QUBO
from .run import Score, score_counts


@dataclass
class BaselineResult:
    mode: str
    counts: Dict[str, int]
    score: Score
    elapsed_s: float
    coords: Optional[np.ndarray] = None          # qoolqit register (dimensionless) coords
    coords_um: Optional[np.ndarray] = None       # physical coords from the compiled sequence (µm)
    embedding_rel_error: Optional[float] = None  # ||U/alpha - J|| / ||J|| in our convention
    sequence: Any = None                         # compiled pulser Sequence
    register: Any = None
    drive: Any = None
    solution: Any = None                         # qubosolver.Solution (solver mode)
    info: Dict[str, Any] = field(default_factory=dict)


def _device(device_name: Optional[str]):
    """qoolqit device: default AnalogDeviceWithDMM (what qubo-solver uses) or a
    wrapped pulser device by name (e.g. 'WeightedAnalogDevice')."""
    import qoolqit
    if device_name is None or device_name == "AnalogDeviceWithDMM":
        return qoolqit.AnalogDeviceWithDMM()
    import pulser.devices as pd
    return qoolqit.Device(pulser_device=getattr(pd, device_name))


def _backend(shots: int, backend_type: Optional[str]):
    from qubosolver import LocalEmulator
    if backend_type is None or backend_type == "auto":
        return LocalEmulator(num_shots=shots)
    from qoolqit.execution import BackendType
    bt = {"qutip": BackendType.QutipBackendV2, "emu-sv": BackendType.SVBackend, "emu-mps": BackendType.MPSBackend}[backend_type]
    return LocalEmulator(backend_type=bt, num_shots=shots)


def _instance(qubo: QUBO):
    from qubosolver import Instance, matrix
    return Instance(matrix.tensor(np.asarray(qubo.Q, dtype=float)))


def _solution_counts(solution) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cand in solution:
        counts[cand.string] = counts.get(cand.string, 0) + int(cand.count)
    return counts


def run_qubo_solver(
    problem: Problem,
    qubo: QUBO,
    shots: int = 500,
    mode: str = "functional",
    backend_type: Optional[str] = None,
    device_name: Optional[str] = None,
    embedding_algorithm: str = "blade",
    drive_algorithm: str = "local_energy_scale",
    kappa: Optional[float] = None,
    preprocessing: bool = True,
    postprocessing: bool = True,
    classical: Optional[ClassicalResult] = None,
    do_repair: bool = True,
) -> BaselineResult:
    t0 = time.time()
    inst = _instance(qubo)
    dev = _device(device_name)
    backend = _backend(shots, backend_type)

    if mode == "solver":
        from qubosolver import DriveShapingConfig, EmbeddingConfig, QuantumSolvingConfig, Solver, SolverConfig
        ds_kw: Dict[str, Any] = {"algorithm": drive_algorithm}
        if kappa is not None:
            ds_kw["local_energy_scale_kappa" if drive_algorithm == "local_energy_scale" else "proportional_diagonal_kappa"] = kappa
        cfg = SolverConfig(
            solving=QuantumSolvingConfig(
                embedding=EmbeddingConfig(algorithm=embedding_algorithm),
                drive_shaping=DriveShapingConfig(**ds_kw),
                backend=backend, device=dev),
            preprocessing=preprocessing, postprocessing=postprocessing)
        solver = Solver(inst, cfg)
        solution = solver.solve()
        counts = _solution_counts(solution)
        sc = score_counts(counts, problem, qubo, _decoder(qubo), classical=classical, do_repair=do_repair)
        reg = getattr(solver._solver, "_cached_register", None)
        drv = getattr(solver._solver, "_cached_drive", None)
        res = BaselineResult("solver", counts, sc, time.time() - t0, register=reg, drive=drv, solution=solution)
        _attach_geometry(res, reg, drv, qubo, dev)
        return res

    # functional: qubo-solver's embedding + drive, raw emulator counts
    from qubosolver import drive_shaping, embedding, solving
    if embedding_algorithm == "blade":
        reg = embedding.blade.embed_for_device(inst, dev)
    else:
        reg = embedding.greedy_layout.embed_for_device(inst, dev)
    shaper = getattr(drive_shaping, drive_algorithm)
    kw = {"kappa": kappa} if kappa is not None else {}
    drv = shaper.build_drive(inst, reg, device=dev, dmm=True, **kw)
    program = solving.analog_quantum_sampling.compile(reg, drv, dev, default_sequence_duration=6000)
    job = backend.run(program)
    results = job.results()
    counts = dict(results.final_bitstrings)
    sc = score_counts(counts, problem, qubo, _decoder(qubo), classical=classical, do_repair=do_repair)
    res = BaselineResult("functional", counts, sc, time.time() - t0, register=reg, drive=drv,
                         sequence=program.compiled_sequence)
    _attach_geometry(res, reg, drv, qubo, dev)
    return res


def _decoder(qubo: QUBO):
    def decode(bs: str) -> np.ndarray:
        return qubo.project(np.array([int(c) for c in bs], dtype=int))
    return decode


def _attach_geometry(res: BaselineResult, reg, drv, qubo: QUBO, dev) -> None:
    try:
        ids = list(reg.qubits_ids)
        res.coords = np.array([np.asarray(reg.qubits[q], float) for q in ids])
        from qubosolver import extract_qubo
        Qr = extract_qubo(reg, drv).matrix.detach().cpu().numpy()
        # their convention: Q_ij = U_ij (dimensionless), ours: J_ij = 2 Q_ij; alpha free -> closed form
        J = qubo.couplings
        Ur = Qr.copy(); np.fill_diagonal(Ur, 0.0)
        Jpos = np.maximum(J, 0.0)
        a = float((Ur * Jpos).sum() / max((Jpos * Jpos).sum(), 1e-30))
        res.embedding_rel_error = float(np.linalg.norm(Ur / a - J) / max(np.linalg.norm(J), 1e-30))
        res.info["alpha_dimensionless"] = a
        if res.sequence is not None:
            r = res.sequence.register
            res.coords_um = np.array([np.asarray(r.qubits[q], float) for q in r.qubit_ids])
    except Exception as e:  # geometry is a nice-to-have; never break the baseline
        res.info["geometry_error"] = repr(e)
