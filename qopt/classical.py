"""Classical QUBO solvers: ground truth for small n, strong heuristics for large n.

All functions take a symmetric Q (energy = x^T Q x) and return
``(x_best, energy_best)``; energies EXCLUDE the QUBO offset (add
``qubo.offset`` yourself or use the QUBO-aware wrappers).

- solve_exact        : brute force (n <= 24) or MILP linearisation via scipy (n <= ~40)
- simulated_annealing: fast incremental-delta SA with restarts (any n)
- tabu_search        : one-flip tabu with aspiration (any n)
- solve_best         : exact if small, else best of SA + tabu
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class ClassicalResult:
    x: np.ndarray
    energy: float
    method: str
    exact: bool

    def __iter__(self):  # allow tuple unpacking
        yield self.x
        yield self.energy


def _sym(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    return 0.5 * (Q + Q.T)


def energy(Q: np.ndarray, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ _sym(Q) @ x)


# --------------------------------------------------------------------------- #
# exact
# --------------------------------------------------------------------------- #
def brute_force(Q: np.ndarray, chunk: int = 1 << 18) -> ClassicalResult:
    Q = _sym(Q)
    n = Q.shape[0]
    if n > 26:
        raise ValueError("brute force only for n <= 26")
    best_e, best_x = math.inf, None
    N = 1 << n
    shifts = np.arange(n)
    for start in range(0, N, chunk):
        ar = np.arange(start, min(N, start + chunk), dtype=np.int64)
        X = ((ar[:, None] >> shifts[None, :]) & 1).astype(float)
        E = np.einsum("mi,ij,mj->m", X, Q, X)
        k = int(np.argmin(E))
        if E[k] < best_e:
            best_e, best_x = float(E[k]), X[k].astype(int)
    return ClassicalResult(best_x, best_e, "brute_force", True)


def milp(Q: np.ndarray, time_limit: Optional[float] = None) -> ClassicalResult:
    """Exact QUBO via standard linearisation y_ij = x_i x_j, solved with
    scipy.optimize.milp (HiGHS).  Practical up to ~40-60 variables depending
    on density."""
    from scipy.optimize import Bounds, LinearConstraint, milp as _milp

    Q = _sym(Q)
    n = Q.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(Q[i, j]) > 1e-12]
    m = len(pairs)
    c = np.concatenate([np.diag(Q), np.array([2.0 * Q[i, j] for i, j in pairs])])
    # constraints: y <= x_i, y <= x_j, y >= x_i + x_j - 1
    rows, lb, ub = [], [], []
    for k, (i, j) in enumerate(pairs):
        r = np.zeros(n + m); r[n + k] = 1; r[i] = -1; rows.append(r); lb.append(-np.inf); ub.append(0)
        r = np.zeros(n + m); r[n + k] = 1; r[j] = -1; rows.append(r); lb.append(-np.inf); ub.append(0)
        r = np.zeros(n + m); r[n + k] = 1; r[i] = -1; r[j] = -1; rows.append(r); lb.append(-1); ub.append(np.inf)
    cons = [LinearConstraint(np.array(rows), lb, ub)] if rows else []
    opts = {"time_limit": time_limit} if time_limit else {}
    res = _milp(c, constraints=cons, integrality=np.ones(n + m), bounds=Bounds(0, 1), options=opts)
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    x = np.round(res.x[:n]).astype(int)
    return ClassicalResult(x, energy(Q, x), "milp", bool(res.success))


def solve_exact(Q: np.ndarray, time_limit: Optional[float] = None) -> ClassicalResult:
    n = np.asarray(Q).shape[0]
    if n <= 20:
        return brute_force(Q)
    return milp(Q, time_limit=time_limit)


# --------------------------------------------------------------------------- #
# heuristics
# --------------------------------------------------------------------------- #
def _flip_deltas(Q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """dE_i for flipping bit i: (1-2x_i) * (Q_ii + 2 * sum_{j!=i} Q_ij x_j)."""
    s = Q @ x
    return (1 - 2 * x) * (np.diag(Q) + 2 * (s - np.diag(Q) * x))


def simulated_annealing(
    Q: np.ndarray,
    n_sweeps: int = 2000,
    n_restarts: int = 8,
    T0: Optional[float] = None,
    T1: Optional[float] = None,
    seed: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
) -> ClassicalResult:
    Q = _sym(Q)
    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    scale = float(np.abs(Q).sum(axis=1).max()) or 1.0
    T0 = T0 if T0 is not None else 1.0 * scale
    T1 = T1 if T1 is not None else 1e-3 * scale
    betas = 1.0 / np.geomspace(T0, T1, n_sweeps)
    best_x, best_e = None, math.inf
    for r in range(n_restarts):
        x = (x0.copy() if (x0 is not None and r == 0) else rng.integers(0, 2, n)).astype(float)
        e = energy(Q, x)
        if e < best_e:
            best_e, best_x = e, x.copy()
        for beta in betas:
            order = rng.permutation(n)
            u = rng.random(n)
            for k, i in enumerate(order):
                si = Q[i] @ x - Q[i, i] * x[i]
                d = (1 - 2 * x[i]) * (Q[i, i] + 2 * si)
                if d <= 0 or u[k] < math.exp(-beta * d):
                    x[i] = 1 - x[i]
                    e += d
                    if e < best_e - 1e-12:
                        best_e, best_x = e, x.copy()
    return ClassicalResult(best_x.astype(int), float(best_e), "simulated_annealing", False)


def tabu_search(
    Q: np.ndarray,
    n_iter: int = 5000,
    tenure: Optional[int] = None,
    seed: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
    n_restarts: int = 4,
) -> ClassicalResult:
    Q = _sym(Q)
    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    tenure = tenure if tenure is not None else max(5, n // 10)
    best_x, best_e = None, math.inf
    for r in range(n_restarts):
        x = (x0.copy() if (x0 is not None and r == 0) else rng.integers(0, 2, n)).astype(float)
        e = energy(Q, x)
        tabu_until = np.zeros(n, dtype=int)
        for it in range(n_iter):
            d = _flip_deltas(Q, x)
            allowed = tabu_until <= it
            # aspiration: allow tabu moves that beat the best
            allowed |= (e + d) < best_e - 1e-12
            if not allowed.any():
                allowed[:] = True
            cand = np.where(allowed, d, np.inf)
            i = int(np.argmin(cand))
            x[i] = 1 - x[i]
            e += d[i]
            tabu_until[i] = it + tenure + rng.integers(0, 3)
            if e < best_e - 1e-12:
                best_e, best_x = e, x.copy()
    return ClassicalResult(best_x.astype(int), float(best_e), "tabu_search", False)


def greedy_descent(Q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """1-flip local search to a local minimum (used by repair)."""
    Q = _sym(Q)
    x = np.asarray(x, dtype=float).copy()
    while True:
        d = _flip_deltas(Q, x)
        i = int(np.argmin(d))
        if d[i] >= -1e-12:
            return x.astype(int)
        x[i] = 1 - x[i]


def solve_best(Q: np.ndarray, exact_max_n: int = 30, seed: Optional[int] = None) -> ClassicalResult:
    """Exact when affordable, else the best of SA and tabu."""
    n = np.asarray(Q).shape[0]
    if n <= exact_max_n:
        try:
            return solve_exact(Q, time_limit=60)
        except Exception:
            pass
    a = simulated_annealing(Q, seed=seed)
    b = tabu_search(Q, seed=seed)
    r = a if a.energy <= b.energy else b
    return ClassicalResult(r.x, r.energy, f"best_of(sa,tabu)->{r.method}", False)
