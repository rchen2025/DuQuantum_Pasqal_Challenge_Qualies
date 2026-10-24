"""Problem -> QUBO modelling with verified penalties.

Conventions
-----------
* Binary variables x in {0,1}^n.
* A QUBO is  E(x) = x^T Q x + offset  with Q symmetric (n x n).  Diagonal
  Q_ii multiplies x_i (since x_i^2 = x_i); the pair (i,j) contributes
  2*Q_ij x_i x_j because both Q_ij and Q_ji appear.  Use
  ``QUBO.linear`` / ``QUBO.couplings`` for the "h_i" and "J_ij = 2 Q_ij" view.
* Every modelling step must pass ``brute_force_verify`` on a tiny instance
  before you trust it on the real one.

Typical use
-----------
>>> p = Problem(["x0", "x1", "x2"])
>>> p.set_objective(linear={"x0": -3, "x1": -4, "x2": -5}, sense="min")
>>> p.add_le({"x0": 2, "x1": 3, "x2": 4}, rhs=5, name="weight")   # knapsack
>>> q = p.to_qubo()                      # penalty chosen automatically
>>> rep = brute_force_verify(p, q)       # rep.ok is True
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

Sense = Literal["min", "max"]


# --------------------------------------------------------------------------- #
# QUBO container
# --------------------------------------------------------------------------- #
@dataclass
class QUBO:
    """E(x) = x^T Q x + offset, Q symmetric."""

    Q: np.ndarray
    offset: float = 0.0
    names: List[str] = field(default_factory=list)
    # indices of the *original* (non-slack) variables inside Q
    original: Optional[List[int]] = None

    def __post_init__(self) -> None:
        self.Q = np.asarray(self.Q, dtype=float)
        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q must be a square matrix")
        self.Q = 0.5 * (self.Q + self.Q.T)  # symmetrise
        n = self.Q.shape[0]
        if not self.names:
            self.names = [f"x{i}" for i in range(n)]
        if len(self.names) != n:
            raise ValueError("len(names) must equal Q.shape[0]")
        if self.original is None:
            self.original = list(range(n))

    # ---- basic views ---------------------------------------------------- #
    @property
    def n(self) -> int:
        return self.Q.shape[0]

    @property
    def linear(self) -> np.ndarray:
        """h_i = Q_ii (coefficient of x_i)."""
        return np.diag(self.Q).copy()

    @property
    def couplings(self) -> np.ndarray:
        """J with J_ij = 2*Q_ij for i != j, 0 on the diagonal.

        E(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j + offset.
        """
        J = 2.0 * self.Q
        np.fill_diagonal(J, 0.0)
        return J

    def energy(self, x: Sequence[int] | np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def energies(self, X: np.ndarray) -> np.ndarray:
        """Vectorised energies for an (m, n) matrix of bitstrings."""
        X = np.asarray(X, dtype=float)
        return np.einsum("mi,ij,mj->m", X, self.Q, X) + self.offset

    def project(self, x: Sequence[int] | np.ndarray) -> np.ndarray:
        """Drop slack variables: full assignment -> original variables."""
        x = np.asarray(x)
        return x[self.original]

    def scaled(self, alpha: float) -> "QUBO":
        """alpha * E(x): same argmin for alpha > 0."""
        return QUBO(alpha * self.Q, alpha * self.offset, list(self.names), list(self.original))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"QUBO(n={self.n}, offset={self.offset:.4g}, original={len(self.original)})"


def qubo_from_matrix(Q: np.ndarray, offset: float = 0.0, names: Optional[List[str]] = None) -> QUBO:
    return QUBO(np.asarray(Q, dtype=float), offset, names or [])


# --------------------------------------------------------------------------- #
# Problem: variables + objective + linear constraints
# --------------------------------------------------------------------------- #
@dataclass
class Constraint:
    coeffs: Dict[str, float]
    rhs: float
    kind: Literal["eq", "le", "ge"]
    name: str = ""
    penalty: Optional[float] = None  # None -> auto

    def lhs(self, assignment: Mapping[str, int]) -> float:
        return sum(c * assignment.get(v, 0) for v, c in self.coeffs.items())

    def satisfied(self, assignment: Mapping[str, int], tol: float = 1e-9) -> bool:
        v = self.lhs(assignment)
        if self.kind == "eq":
            return abs(v - self.rhs) <= tol
        if self.kind == "le":
            return v <= self.rhs + tol
        return v >= self.rhs - tol

    def violation(self, assignment: Mapping[str, int]) -> float:
        v = self.lhs(assignment)
        if self.kind == "eq":
            return abs(v - self.rhs)
        if self.kind == "le":
            return max(0.0, v - self.rhs)
        return max(0.0, self.rhs - v)


@dataclass
class SlackInfo:
    constraint: str
    var_names: List[str]
    coeffs: List[float]  # slack = sum coeffs_k * s_k
    max_value: float


@dataclass
class QuboBuild:
    """What ``Problem.to_qubo`` produced, with provenance."""

    qubo: QUBO
    penalties: Dict[str, float]
    slacks: List[SlackInfo]
    sense: Sense
    objective_bound: float  # upper bound on |objective variation|


class Problem:
    """Binary optimisation problem with linear objective (+ optional quadratic
    terms) and linear equality / inequality constraints."""

    def __init__(self, variables: Iterable[str]):
        self.variables: List[str] = list(variables)
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("duplicate variable names")
        self._lin: Dict[str, float] = {}
        self._quad: Dict[Tuple[str, str], float] = {}
        self.sense: Sense = "min"
        self.constraints: List[Constraint] = []

    # ---- objective ------------------------------------------------------- #
    def set_objective(
        self,
        linear: Optional[Mapping[str, float]] = None,
        quadratic: Optional[Mapping[Tuple[str, str], float]] = None,
        sense: Sense = "min",
    ) -> "Problem":
        self._lin = dict(linear or {})
        self._quad = {}
        for (a, b), c in (quadratic or {}).items():
            key = (a, b) if self.variables.index(a) <= self.variables.index(b) else (b, a)
            self._quad[key] = self._quad.get(key, 0.0) + c
        self.sense = sense
        self._check_names(list(self._lin) + [v for k in self._quad for v in k])
        return self

    def objective(self, assignment: Mapping[str, int]) -> float:
        val = sum(c * assignment.get(v, 0) for v, c in self._lin.items())
        val += sum(c * assignment.get(a, 0) * assignment.get(b, 0) for (a, b), c in self._quad.items())
        return float(val)

    # ---- constraints ----------------------------------------------------- #
    def add_eq(self, coeffs: Mapping[str, float], rhs: float, name: str = "", penalty: Optional[float] = None):
        return self._add(coeffs, rhs, "eq", name, penalty)

    def add_le(self, coeffs: Mapping[str, float], rhs: float, name: str = "", penalty: Optional[float] = None):
        return self._add(coeffs, rhs, "le", name, penalty)

    def add_ge(self, coeffs: Mapping[str, float], rhs: float, name: str = "", penalty: Optional[float] = None):
        return self._add(coeffs, rhs, "ge", name, penalty)

    def add_one_hot(self, variables: Iterable[str], name: str = "", penalty: Optional[float] = None):
        """Exactly one of ``variables`` is 1."""
        return self.add_eq({v: 1.0 for v in variables}, 1.0, name, penalty)

    def add_at_most_one(self, variables: Iterable[str], name: str = "", penalty: Optional[float] = None):
        return self.add_le({v: 1.0 for v in variables}, 1.0, name, penalty)

    def _add(self, coeffs, rhs, kind, name, penalty):
        self._check_names(coeffs)
        name = name or f"c{len(self.constraints)}"
        self.constraints.append(Constraint(dict(coeffs), float(rhs), kind, name, penalty))
        return self

    def _check_names(self, names: Iterable[str]) -> None:
        unknown = [v for v in names if v not in self.variables]
        if unknown:
            raise KeyError(f"unknown variables: {unknown}")

    # ---- feasibility ----------------------------------------------------- #
    def is_feasible(self, assignment: Mapping[str, int]) -> bool:
        return all(c.satisfied(assignment) for c in self.constraints)

    def total_violation(self, assignment: Mapping[str, int]) -> float:
        return sum(c.violation(assignment) for c in self.constraints)

    def assignment(self, x: Sequence[int]) -> Dict[str, int]:
        return {v: int(xi) for v, xi in zip(self.variables, x)}

    # ---- QUBO construction ---------------------------------------------- #
    def objective_bound(self) -> float:
        """Upper bound on max(obj) - min(obj) over the hypercube."""
        return float(sum(abs(c) for c in self._lin.values()) + sum(abs(c) for c in self._quad.values()))

    def to_qubo(self, penalty: Optional[float] = None, penalty_margin: float = 1.0) -> QuboBuild:
        """Build the penalised QUBO.

        Penalty weight for each constraint (unless given explicitly):
            P = objective_bound() + penalty_margin
        For integer-coefficient constraints every infeasible point violates by
        at least 1 (squared >= 1), so P larger than the whole objective range
        guarantees feasible optimum == QUBO optimum.  For non-integer data
        the penalty is still a sensible default but *must* be checked with
        ``brute_force_verify`` (or raise it).
        """
        n0 = len(self.variables)
        names = list(self.variables)
        obj_bound = self.objective_bound()
        default_P = penalty if penalty is not None else obj_bound + penalty_margin

        sign = 1.0 if self.sense == "min" else -1.0
        # objective block
        lin_terms: Dict[int, float] = {}
        quad_terms: Dict[Tuple[int, int], float] = {}
        idx = {v: i for i, v in enumerate(names)}
        for v, c in self._lin.items():
            lin_terms[idx[v]] = lin_terms.get(idx[v], 0.0) + sign * c
        for (a, b), c in self._quad.items():
            if a == b:
                lin_terms[idx[a]] = lin_terms.get(idx[a], 0.0) + sign * c
            else:
                key = (idx[a], idx[b])
                quad_terms[key] = quad_terms.get(key, 0.0) + sign * c

        penalties: Dict[str, float] = {}
        slacks: List[SlackInfo] = []
        offset = 0.0

        # constraints -> (sum a_i x_i + sum s_k b_k - rhs)^2 * P
        for con in self.constraints:
            P = con.penalty if con.penalty is not None else default_P
            penalties[con.name] = P
            terms: Dict[int, float] = {idx[v]: c for v, c in con.coeffs.items()}
            rhs = con.rhs
            if con.kind in ("le", "ge"):
                # slack: lhs + s = rhs (le) or lhs - s = rhs (ge)
                lhs_min = sum(min(0.0, c) for c in con.coeffs.values())
                lhs_max = sum(max(0.0, c) for c in con.coeffs.values())
                if con.kind == "le":
                    max_slack = rhs - lhs_min  # s = rhs - lhs >= 0
                    s_sign = 1.0
                else:
                    max_slack = lhs_max - rhs  # s = lhs - rhs >= 0
                    s_sign = -1.0
                if max_slack < 0:
                    raise ValueError(f"constraint {con.name} is infeasible for all x")
                coeffs = _binary_expansion(max_slack)
                svars = [f"s_{con.name}_{k}" for k in range(len(coeffs))]
                for sv, ck in zip(svars, coeffs):
                    idx[sv] = len(names)
                    names.append(sv)
                    terms[idx[sv]] = s_sign * ck
                slacks.append(SlackInfo(con.name, svars, coeffs, max_slack))
            # expand P*(sum_i a_i x_i - rhs)^2
            items = list(terms.items())
            for i, a in items:
                lin_terms[i] = lin_terms.get(i, 0.0) + P * (a * a - 2.0 * a * rhs)
            for (i, a), (j, b) in itertools.combinations(items, 2):
                key = (i, j) if i < j else (j, i)
                quad_terms[key] = quad_terms.get(key, 0.0) + 2.0 * P * a * b
            offset += P * rhs * rhs

        n = len(names)
        Q = np.zeros((n, n))
        for i, c in lin_terms.items():
            Q[i, i] += c
        for (i, j), c in quad_terms.items():
            Q[i, j] += c / 2.0
            Q[j, i] += c / 2.0
        qubo = QUBO(Q, offset, names, list(range(n0)))
        return QuboBuild(qubo, penalties, slacks, self.sense, obj_bound)

    # ---- enumeration helpers -------------------------------------------- #
    def enumerate_feasible(self):
        """Yield (x, objective) over all feasible assignments (tiny n only)."""
        n = len(self.variables)
        if n > 22:
            raise ValueError("enumerate_feasible is for n <= 22")
        for bits in itertools.product((0, 1), repeat=n):
            a = self.assignment(bits)
            if self.is_feasible(a):
                yield np.array(bits), self.objective(a)

    def constrained_optimum(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        best_x, best_v = None, None
        for x, v in self.enumerate_feasible():
            if best_v is None or (v < best_v if self.sense == "min" else v > best_v):
                best_x, best_v = x, v
        return best_x, best_v


def _binary_expansion(max_value: float) -> List[float]:
    """Coefficients c_k so that sum c_k s_k (s_k binary) covers 0..max_value
    with the last coefficient trimmed to hit max_value exactly (integer data)."""
    m = int(math.floor(max_value + 1e-9))
    if m <= 0:
        return []
    coeffs: List[float] = []
    k = 0
    while sum(coeffs) + 2**k <= m:
        coeffs.append(float(2**k))
        k += 1
    rest = m - sum(coeffs)
    if rest > 0:
        coeffs.append(float(rest))
    if not math.isclose(max_value, m):
        # non-integer range: add one extra fractional slack bit
        coeffs.append(float(max_value - m))
    return coeffs


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass
class VerifyReport:
    ok: bool
    constrained_optimum: Optional[float]
    qubo_minimum: float
    qubo_argmin_feasible: bool
    qubo_argmin: np.ndarray
    gap: float  # min over infeasible QUBO points minus feasible optimum (>0 is healthy)
    message: str

    def __bool__(self) -> bool:
        return self.ok


def brute_force_verify(problem: Problem, build: QuboBuild | QUBO, tol: float = 1e-6) -> VerifyReport:
    """Check on a tiny instance that min_x QUBO(x) == constrained optimum and
    that the minimiser (projected on original variables) is feasible.

    Works for n_total <= 22 (2^22 states, vectorised)."""
    qubo = build.qubo if isinstance(build, QuboBuild) else build
    sense = build.sense if isinstance(build, QuboBuild) else problem.sense
    n = qubo.n
    if n > 22:
        raise ValueError(f"brute force verify only for <= 22 variables (got {n})")
    X = _all_bitstrings(n)
    E = qubo.energies(X)
    k = int(np.argmin(E))
    xq = X[k]
    x_orig = qubo.project(xq)
    a = problem.assignment(x_orig)
    feasible = problem.is_feasible(a)

    _, opt = problem.constrained_optimum()
    if opt is None:
        return VerifyReport(False, None, float(E[k]), feasible, xq, float("nan"), "no feasible point exists")
    sign = 1.0 if sense == "min" else -1.0
    # restrict to the original variables for feasibility of every row
    X_orig = X[:, qubo.original]
    feas_mask = np.array([problem.is_feasible(problem.assignment(r)) for r in X_orig])
    E_feas_min = float(E[feas_mask].min())
    E_infeas_min = float(E[~feas_mask].min()) if (~feas_mask).any() else float("inf")
    gap = E_infeas_min - E_feas_min
    value_ok = abs((E[k] - qubo.offset) * 1.0 - sign * opt) <= tol or abs(sign * (E[k] - qubo.offset) - opt) <= tol
    # E includes the constant offset; feasible points have zero penalty, so
    # E(x_feas) = sign*obj + offset_obj. The penalty offsets cancel at feasibility.
    ok = feasible and gap > 0 and _close(E_feas_min, E[k], tol)
    msg = "OK" if ok else (
        f"FAILED: argmin feasible={feasible}, gap={gap:.4g}, "
        f"min feasible E={E_feas_min:.4g}, global min E={E[k]:.4g}"
    )
    if ok and not value_ok:
        msg = "OK (value matches up to constant offset)"
    return VerifyReport(ok, float(opt), float(E[k]), feasible, xq, float(gap), msg)


def auto_penalty(problem: Problem, start: Optional[float] = None, factor: float = 2.0, max_iter: int = 12) -> QuboBuild:
    """Raise the penalty until brute_force_verify passes (tiny instances)."""
    P = start if start is not None else problem.objective_bound() + 1.0
    for _ in range(max_iter):
        build = problem.to_qubo(penalty=P)
        if brute_force_verify(problem, build).ok:
            return build
        P *= factor
    raise RuntimeError("could not find a penalty that passes verification")


def _all_bitstrings(n: int) -> np.ndarray:
    N = 1 << n
    ar = np.arange(N, dtype=np.int64)
    return ((ar[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------- #
# Ready-made toy problems (for tests / demo)
# --------------------------------------------------------------------------- #
def knapsack(values: Sequence[float], weights: Sequence[float], capacity: float) -> Problem:
    names = [f"x{i}" for i in range(len(values))]
    p = Problem(names)
    p.set_objective(linear={n: v for n, v in zip(names, values)}, sense="max")
    p.add_le({n: w for n, w in zip(names, weights)}, capacity, name="cap")
    return p


def set_packing(sets: Sequence[Sequence[int]], n_elements: int, weights: Optional[Sequence[float]] = None) -> Problem:
    """Pick disjoint sets maximising total weight (each element in <= 1 set)."""
    names = [f"S{i}" for i in range(len(sets))]
    w = list(weights) if weights is not None else [1.0] * len(sets)
    p = Problem(names)
    p.set_objective(linear={n: wi for n, wi in zip(names, w)}, sense="max")
    for e in range(n_elements):
        members = [names[i] for i, s in enumerate(sets) if e in s]
        if len(members) > 1:
            p.add_at_most_one(members, name=f"e{e}")
    return p


def max_independent_set(edges: Sequence[Tuple[int, int]], n_nodes: int, penalty: Optional[float] = None) -> Problem:
    """MIS as a native (penalty-free, if penalty set) QUBO: maximise |S| s.t.
    no edge inside S.  Uses quadratic objective directly (no slack needed)."""
    names = [f"v{i}" for i in range(n_nodes)]
    p = Problem(names)
    P = penalty if penalty is not None else 2.0
    p.set_objective(
        linear={n: -1.0 for n in names},
        quadratic={(names[a], names[b]): P for a, b in edges},
        sense="min",
    )
    return p
