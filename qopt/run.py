"""One function: (QUBO, Embedding, Schedule) -> bitstrings -> decode -> repair -> score.

Backends ("backend=" argument):
    "qutip"          pulser_simulation.QutipBackendV2        (<= ~14 atoms, exact)
    "emu-sv"         emu_sv.SVBackend                        (<= ~25 atoms, exact, torch)
    "emu-mps"        emu_mps.MPSBackend                      (25-80+ atoms, approximate; check bond dim)
    "cloud-mps" / "cloud-sv" / "cloud-free"   pasqal_cloud Remote*Backend (async: returns job ids)
    "qpu"            pulser.QPUBackend over PasqalCloudConnection (async)
    "samples"        a dict bitstring->count you already have (e.g. fetched cloud results)

Cloud credentials come from env vars only: PASQAL_USERNAME, PASQAL_PASSWORD,
PASQAL_PROJECT_ID (see .env.example; load with python-dotenv if you like).
Job ids are saved under jobs/<timestamp>_<name>.json.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .classical import ClassicalResult, greedy_descent, solve_best
from .embed import Embedding
from .model import Problem, QUBO
from .schedule import Schedule, build_sequence

JOBS_DIR = Path(os.environ.get("QOPT_JOBS_DIR", "jobs"))


# --------------------------------------------------------------------------- #
# Backends: Sequence -> counts
# --------------------------------------------------------------------------- #
def emulate_qutip(seq, shots: int = 1000, sampling_rate: float = 0.1, with_modulation: bool = False,
                  noise_model: Any = None) -> Dict[str, int]:
    from pulser.backend import BitStrings
    from pulser_simulation import QutipBackendV2, QutipConfig
    kw: Dict[str, Any] = {}
    if noise_model is not None:   # QutipConfig dereferences noise_model before its base class turns None into NoiseModel()
        kw["noise_model"] = noise_model
    cfg = QutipConfig(sampling_rate=sampling_rate, with_modulation=with_modulation,
                      observables=[BitStrings(evaluation_times=[1.0], num_shots=shots)], **kw)
    res = QutipBackendV2(seq, config=cfg).run()
    return dict(res.get_result("bitstrings", 1.0))


def emulate_emu_sv(seq, shots: int = 1000, dt: float = 10.0, gpu: Optional[bool] = None,
                   with_modulation: bool = False, noise_model: Any = None, log_level: int = 40) -> Dict[str, int]:
    from emu_sv import BitStrings, SVBackend, SVConfig
    cfg = SVConfig(dt=dt, gpu=gpu, log_level=log_level, with_modulation=with_modulation, noise_model=noise_model,
                   observables=[BitStrings(evaluation_times=[1.0], num_shots=shots)])
    res = SVBackend(seq, config=cfg).run()
    return dict(res.get_result("bitstrings", 1.0))


def emulate_emu_mps(seq, shots: int = 1000, dt: float = 10.0, precision: float = 1e-5, max_bond_dim: int = 256,
                    gpu: Optional[bool] = None, with_modulation: bool = False, noise_model: Any = None,
                    log_level: int = 40, return_stats: bool = False):
    from emu_mps import BitStrings, MPSBackend, MPSConfig
    cfg = MPSConfig(dt=dt, precision=precision, max_bond_dim=max_bond_dim, gpu=gpu, log_level=log_level,
                    with_modulation=with_modulation, noise_model=noise_model,
                    observables=[BitStrings(evaluation_times=[1.0], num_shots=shots)])
    res = MPSBackend(seq, config=cfg).run()
    counts = dict(res.get_result("bitstrings", 1.0))
    if return_stats:
        return counts, getattr(res, "statistics", None)
    return counts


def mps_convergence_check(seq, shots: int = 1000, **kw) -> Dict[str, Any]:
    """Run emu-mps twice (base and tighter precision / 2x bond dim) and report
    the total-variation distance of the sampled distributions.  < ~0.05 is
    usually fine; larger means the MPS is not converged (2D / high
    entanglement) — distrust it or move to emu-sv / cloud."""
    base = emulate_emu_mps(seq, shots=shots, **kw)
    kw2 = dict(kw)
    kw2["precision"] = kw.get("precision", 1e-5) / 10
    kw2["max_bond_dim"] = 2 * kw.get("max_bond_dim", 256)
    tight = emulate_emu_mps(seq, shots=shots, **kw2)
    keys = set(base) | set(tight)
    tv = 0.5 * sum(abs(base.get(k, 0) / shots - tight.get(k, 0) / shots) for k in keys)
    return {"tv_distance": tv, "base": base, "tight": tight, "converged": tv < 0.05}


def _connection():
    from pasqal_cloud import PasqalCloudConnection
    u, p, pid = (os.environ.get("PASQAL_USERNAME"), os.environ.get("PASQAL_PASSWORD"), os.environ.get("PASQAL_PROJECT_ID"))
    if not (u and p and pid):
        raise RuntimeError("set PASQAL_USERNAME, PASQAL_PASSWORD, PASQAL_PROJECT_ID in the environment (.env, never committed)")
    return PasqalCloudConnection(username=u, password=p, project_id=pid)


def cloud_submit(seq, backend: str = "cloud-mps", shots: int = 500, name: str = "run", wait: bool = False,
                 config_changes: Optional[Dict[str, Any]] = None, mimic_qpu: bool = True) -> Dict[str, Any]:
    """Submit asynchronously. Returns {"batch_id", "job_ids", "backend", "file"}; results via cloud_fetch."""
    from pulser.backend import BitStrings
    conn = _connection()
    if backend == "qpu":
        from pulser import QPUBackend
        from pulser.backend import BackendConfig
        bk = QPUBackend(seq, conn, config=BackendConfig(default_num_shots=shots))
    else:
        import pasqal_cloud as pc
        cls = {"cloud-mps": pc.RemoteMPSBackend, "cloud-sv": pc.RemoteSVBackend, "cloud-free": pc.RemoteEmuFreeBackend}[backend]
        cfg = cls.default_config.with_changes(observables=[BitStrings(num_shots=shots)], **(config_changes or {}))
        bk = cls(seq, conn, config=cfg, mimic_qpu=mimic_qpu)
    rr = bk.run(job_params=[{"runs": shots}], wait=wait)
    rec = {"batch_id": rr.batch_id, "job_ids": list(rr.job_ids or []), "backend": backend, "shots": shots,
           "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": name}
    JOBS_DIR.mkdir(exist_ok=True)
    rec["file"] = str(JOBS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.json")
    Path(rec["file"]).write_text(json.dumps(rec, indent=2))
    if wait:
        rec["counts"] = dict(rr[0].final_bitstrings)
    return rec


def cloud_fetch(batch_id: str, job_ids: Optional[List[str]] = None) -> Optional[Dict[str, int]]:
    """Fetch counts for a finished batch, or None if still running."""
    from pulser.backend import RemoteResults
    rr = RemoteResults(batch_id, _connection(), job_ids=job_ids)
    status = str(rr.get_batch_status())
    if "DONE" not in status.upper():
        return None
    results = rr.results
    total: Counter = Counter()
    for r in results:
        total.update(r.final_bitstrings)
    return dict(total)


def mock_boltzmann(embedding: Embedding, shots: int = 1000, temperature: float = 1.0, seed: int = 0) -> Dict[str, int]:
    """NOT a quantum simulation. Samples bitstrings from exp(-H_f/T) of the
    *realized* final Hamiltonian (n <= 20) so the decode/repair/score/report
    code can be exercised without an emulator. Clearly labelled 'mock'."""
    from .model import _all_bitstrings
    n = embedding.qubo.n
    if n > 20:
        raise ValueError("mock backend only for n <= 20")
    X = _all_bitstrings(n)
    E = np.array([embedding.final_hamiltonian_energy(x) for x in X])
    scale = max(float(np.ptp(E)), 1e-9)
    p = np.exp(-(E - E.min()) / (temperature * scale / 10.0))
    p /= p.sum()
    rng = np.random.default_rng(seed)
    draw = rng.choice(len(X), size=shots, p=p)
    counts: Counter = Counter("".join(str(b) for b in X[i]) for i in draw)
    return dict(counts)


def run_backend(seq, backend: str, shots: int, **kw) -> Dict[str, int]:
    if backend == "qutip":
        return emulate_qutip(seq, shots, **kw)
    if backend in ("emu-sv", "emu_sv", "sv"):
        return emulate_emu_sv(seq, shots, **kw)
    if backend in ("emu-mps", "emu_mps", "mps"):
        return emulate_emu_mps(seq, shots, **kw)
    if backend in ("cloud-mps", "cloud-sv", "cloud-free", "qpu"):
        rec = cloud_submit(seq, backend, shots, wait=True, **kw)
        return rec["counts"]
    raise ValueError(f"unknown backend {backend}")


def auto_backend(n_atoms: int) -> str:
    if n_atoms <= 13:
        return "qutip"
    if n_atoms <= 24:
        return "emu-sv"
    return "emu-mps"


# --------------------------------------------------------------------------- #
# Decode / repair / score
# --------------------------------------------------------------------------- #
def repair(x: np.ndarray, problem: Problem, qubo: Optional[QUBO] = None, max_flips: int = 1000) -> Tuple[np.ndarray, bool]:
    """Constraint-check + greedy fix.  Flips the single bit that reduces total
    violation the most (ties: better objective) until feasible; then a greedy
    descent on the QUBO (if given) that only accepts feasible moves."""
    x = np.asarray(x, dtype=int).copy()
    a = problem.assignment(x)
    sense = 1.0 if problem.sense == "min" else -1.0
    for _ in range(max_flips):
        if problem.is_feasible(a):
            break
        v0 = problem.total_violation(a)
        best = None
        for i, name in enumerate(problem.variables):
            a2 = dict(a); a2[name] = 1 - a2[name]
            v = problem.total_violation(a2)
            obj = sense * problem.objective(a2)
            key = (v, obj)
            if best is None or key < best[0]:
                best = (key, i)
        if best is None or best[0][0] >= v0 - 1e-12:
            return x, False  # stuck
        i = best[1]
        x[i] = 1 - x[i]
        a = problem.assignment(x)
    feasible = problem.is_feasible(a)
    if feasible:
        # feasible-only greedy improvement on the objective
        improved = True
        while improved:
            improved = False
            cur = sense * problem.objective(a)
            for i, name in enumerate(problem.variables):
                a2 = dict(a); a2[name] = 1 - a2[name]
                if problem.is_feasible(a2) and sense * problem.objective(a2) < cur - 1e-12:
                    a, x[i] = a2, 1 - x[i]
                    improved = True
                    break
    return x, feasible


@dataclass
class Score:
    n_shots: int
    feasible_rate_raw: float
    feasible_rate_repaired: float
    best_objective: Optional[float]          # best feasible objective found (after repair), in problem units
    best_x: Optional[np.ndarray]
    classical_objective: Optional[float]     # ground truth / bar
    classical_x: Optional[np.ndarray]
    classical_method: str
    approx_ratio: Optional[float]            # best / classical (max problems) or classical / best (min)
    p_optimal_raw: float                     # fraction of raw shots that already hit the optimum
    p_optimal_repaired: float
    mean_objective_repaired: Optional[float]
    optimum_rank: Optional[int]              # rank of the optimum among sampled distinct bitstrings (1 = most sampled)
    top_counts: List[Tuple[str, int, float, bool]] = field(default_factory=list)  # (bitstring, count, objective, feasible)

    def table(self) -> str:
        rows = [
            ("shots", self.n_shots),
            ("feasible rate raw / repaired", f"{self.feasible_rate_raw:.3f} / {self.feasible_rate_repaired:.3f}"),
            ("best objective (repaired)", self.best_objective),
            ("classical", f"{self.classical_objective} ({self.classical_method})"),
            ("approx ratio", None if self.approx_ratio is None else f"{self.approx_ratio:.4f}"),
            ("P(optimum) raw / repaired", f"{self.p_optimal_raw:.3f} / {self.p_optimal_repaired:.3f}"),
            ("optimum rank among samples", self.optimum_rank),
        ]
        w = max(len(r[0]) for r in rows)
        return "\n".join(f"  {k:<{w}}  {v}" for k, v in rows)


def score_counts(counts: Dict[str, int], problem: Problem, qubo: QUBO, decode: Callable[[str], np.ndarray],
                 classical: Optional[ClassicalResult] = None, do_repair: bool = True, top: int = 10) -> Score:
    shots = sum(counts.values())
    sense = 1.0 if problem.sense == "min" else -1.0
    if classical is None:
        classical = solve_best(qubo.Q)
    cx = qubo.project(classical.x)
    c_obj = problem.objective(problem.assignment(cx))
    c_feas = problem.is_feasible(problem.assignment(cx))
    if not c_feas:
        # classical QUBO optimum infeasible -> penalty too low; fall back to enumeration if small
        if len(problem.variables) <= 20:
            cx, c_obj = problem.constrained_optimum()
        else:
            cx, _ = repair(cx, problem)
            c_obj = problem.objective(problem.assignment(cx))
    feas_raw = 0; feas_rep = 0; p_opt_raw = 0; p_opt_rep = 0
    best_obj, best_x = None, None
    objs: List[Tuple[float, int]] = []
    rows = []
    for bs, cnt in counts.items():
        x = decode(bs)
        a = problem.assignment(x)
        f = problem.is_feasible(a)
        obj = problem.objective(a)
        if f:
            feas_raw += cnt
            if abs(obj - c_obj) < 1e-9:
                p_opt_raw += cnt
        rows.append((bs, cnt, obj, f))
        if do_repair and not f:
            x, f = repair(x, problem, qubo)
            a = problem.assignment(x); obj = problem.objective(a)
        if f:
            feas_rep += cnt
            objs.append((obj, cnt))
            if abs(obj - c_obj) < 1e-9:
                p_opt_rep += cnt
            if best_obj is None or sense * obj < sense * best_obj:
                best_obj, best_x = obj, x
    mean_obj = (sum(o * c for o, c in objs) / sum(c for _, c in objs)) if objs else None
    ratio = None
    if best_obj is not None and c_obj is not None and c_obj != 0:
        ratio = (best_obj / c_obj) if problem.sense == "max" else (c_obj / best_obj if best_obj != 0 else None)
    rows.sort(key=lambda r: -r[1])
    rank = next((i + 1 for i, r in enumerate(rows) if r[3] and abs(r[2] - c_obj) < 1e-9), None)
    return Score(shots, feas_raw / shots, feas_rep / shots, best_obj, best_x, c_obj, cx, classical.method, ratio,
                 p_opt_raw / shots, p_opt_rep / shots, mean_obj, rank, rows[:top])


# --------------------------------------------------------------------------- #
# The one function
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    counts: Dict[str, int]
    score: Score
    backend: str
    schedule: Schedule
    embedding: Embedding
    sequence: Any = None
    job: Optional[Dict[str, Any]] = None  # cloud record when async
    elapsed_s: float = 0.0


def run_pipeline(
    problem: Problem,
    qubo: QUBO,
    embedding: Embedding,
    schedule: Schedule,
    backend: str = "auto",
    shots: int = 500,
    classical: Optional[ClassicalResult] = None,
    counts: Optional[Dict[str, int]] = None,
    device: Any = None,
    do_repair: bool = True,
    **backend_kw: Any,
) -> RunResult:
    """Build the Pulser sequence, run it, decode, repair, score."""
    t0 = time.time()
    seq = build_sequence(embedding, schedule, device=device)
    if backend == "auto":
        backend = auto_backend(qubo.n)
    job = None
    if counts is None:
        if backend == "mock":
            counts = mock_boltzmann(embedding, shots, **backend_kw)
        elif backend in ("cloud-mps", "cloud-sv", "cloud-free", "qpu") and not backend_kw.get("wait", False):
            job = cloud_submit(seq, backend, shots, **backend_kw)
            counts = {}
        else:
            counts = run_backend(seq, backend, shots, **backend_kw)
    if counts:
        sc = score_counts(counts, problem, qubo, embedding.decode, classical=classical, do_repair=do_repair)
    else:
        sc = Score(0, 0, 0, None, None, None, None, "pending", None, 0, 0, None, None)
    return RunResult(counts, sc, backend, schedule, embedding, seq, job, time.time() - t0)


def make_evaluator(problem: Problem, qubo: QUBO, embedding: Embedding, backend: str = "auto", shots: int = 300,
                   classical: Optional[ClassicalResult] = None, metric: str = "p_optimal_repaired", **kw):
    """For schedule.scan/optimize: Schedule -> (score, info)."""
    if classical is None:
        classical = solve_best(qubo.Q)

    def evaluate(s: Schedule):
        r = run_pipeline(problem, qubo, embedding, s, backend=backend, shots=shots, classical=classical, **kw)
        sc = r.score
        val = {"p_optimal_repaired": sc.p_optimal_repaired, "p_optimal_raw": sc.p_optimal_raw,
               "approx_ratio": sc.approx_ratio or 0.0, "feasible_rate_raw": sc.feasible_rate_raw,
               "neg_mean_objective": -(sc.mean_objective_repaired or 0.0)}[metric]
        return float(val), {"score": sc, "schedule": s}
    return evaluate
