"""Shared types for the CVRP challenge package (Track A: capacity-feasible clustering).

Everything else in ``cvrp/`` imports from here and nowhere else inside the
package, so modules can be written independently.

Conventions (also in CLAUDE.md):
* Customers are indexed 0..N-1; the depot is a separate point ``inst.depot``.
* An *assignment* is a tuple of length N with the vehicle index (0..K-1) of
  each customer.  Vehicles are identical, so two assignments that differ only
  by relabeling are the same *clustering*; compare with ``clustering_of()``.
* Cost = Euclidean distance, symmetric, triangle inequality holds.
* A QUBO is  E(x) = x^T Q x + offset  with Q symmetric; h_i = Q_ii,
  J_ij = 2 Q_ij (i != j).  The device realises  H_f(x) = -delta sum_i x_i
  + sum_{i<j} U_ij x_i x_j  with U_ij = C6 / r_ij^6 > 0.  Hence every
  coupling must be positive and all linear terms equal (global detuning).
"""
from __future__ import annotations

import itertools
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Assignment = Tuple[int, ...]            # vehicle index per customer
Clustering = FrozenSet[FrozenSet[int]]  # set of customer groups (non-empty)

# --------------------------------------------------------------------------- #
# Tiers (brief, Table 1).  N, K, Q and demand range are fixed by the brief.
# --------------------------------------------------------------------------- #
TIERS: Dict[str, Dict[str, Any]] = {
    "S": {"N": 4, "K": 2, "Q": 7, "d_min": 1, "d_max": 4, "intended": "pulser-simulation, emu-sv"},
    "M": {"N": 10, "K": 3, "Q": 20, "d_min": 1, "d_max": 8, "intended": "emu-mps"},
    "L": {"N": 15, "K": 4, "Q": 30, "d_min": 1, "d_max": 10, "intended": "emu-mps or QPU"},
}
SQUARE_SIDE = 100.0


# --------------------------------------------------------------------------- #
# Instance
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    name: str
    tier: str
    seed: int
    N: int
    K: int
    Q: int
    depot: Tuple[float, float]
    locations: List[Tuple[float, float]]       # N customer locations in [0, 100]^2
    demands: List[int]                         # N integers in [d_min, d_max]
    certificate: Assignment                    # one valid customer->vehicle assignment
    square_side: float = SQUARE_SIDE
    depot_rule: str = "centre of the square (50, 50)"
    rho: float = 0.0                           # sum d / (K Q)
    dmax_over_Q: float = 0.0                   # max d / Q
    feasibility_method: str = ""               # how feasibility was guaranteed
    generator: Dict[str, Any] = field(default_factory=dict)
    reference: Dict[str, Any] = field(default_factory=dict)   # filled by cvrp.reference
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- geometry --------------------------------------------------------- #
    def points(self) -> np.ndarray:
        """(N+1, 2) array: row 0 = depot, rows 1..N = customers."""
        return np.array([self.depot] + [tuple(p) for p in self.locations], dtype=float)

    def dist_matrix(self) -> np.ndarray:
        """(N+1, N+1) Euclidean distances, index 0 = depot."""
        P = self.points()
        diff = P[:, None, :] - P[None, :, :]
        return np.sqrt((diff ** 2).sum(-1))

    def customer_dist(self) -> np.ndarray:
        """(N, N) customer-customer distances."""
        return self.dist_matrix()[1:, 1:]

    def depot_dist(self) -> np.ndarray:
        """(N,) customer-depot distances."""
        return self.dist_matrix()[0, 1:]

    # ---- constraints ------------------------------------------------------ #
    def loads(self, a: Assignment) -> List[int]:
        L = [0] * self.K
        for i, k in enumerate(a):
            L[k] += self.demands[i]
        return L

    def capacity_ok(self, a: Assignment) -> bool:
        return all(l <= self.Q for l in self.loads(a))

    def is_valid_assignment(self, a: Sequence[int]) -> bool:
        return (len(a) == self.N and all(isinstance(k, (int, np.integer)) and 0 <= k < self.K for k in a)
                and self.capacity_ok(tuple(int(k) for k in a)))

    # ---- io --------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["certificate"] = list(self.certificate)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Instance":
        d = dict(d)
        d["depot"] = tuple(d["depot"])
        d["locations"] = [tuple(p) for p in d["locations"]]
        d["certificate"] = tuple(int(k) for k in d["certificate"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "Instance":
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_instances(folder: str | Path = "instances") -> List[Instance]:
    return [Instance.load(p) for p in sorted(Path(folder).glob("*.json"))]


# --------------------------------------------------------------------------- #
# Clusterings (vehicles identical -> compare as sets of customer groups)
# --------------------------------------------------------------------------- #
def clustering_of(a: Assignment) -> Clustering:
    groups: Dict[int, set] = {}
    for i, k in enumerate(a):
        groups.setdefault(int(k), set()).add(i)
    return frozenset(frozenset(g) for g in groups.values())


def assignment_of(clusters: Iterable[Iterable[int]], N: int) -> Assignment:
    a = [-1] * N
    for k, g in enumerate(clusters):
        for i in g:
            a[i] = k
    if any(k < 0 for k in a):
        raise ValueError("clusters do not cover every customer")
    return tuple(a)


def same_clustering(a: Assignment, b: Assignment) -> bool:
    return clustering_of(a) == clustering_of(b)


# --------------------------------------------------------------------------- #
# Exact TSP per cluster (Held-Karp), depot fixed at index 0 of the dist matrix
# --------------------------------------------------------------------------- #
def tsp_exact(D: np.ndarray, customers: Sequence[int]) -> Tuple[float, List[int]]:
    """Shortest depot -> customers -> depot tour.  ``customers`` are indices
    into D *including the depot offset* (i.e. customer i is row i+1).
    Returns (length, order of customers).  n <= ~15 is fine (2^n n^2)."""
    cs = list(customers)
    n = len(cs)
    if n == 0:
        return 0.0, []
    if n == 1:
        return float(2 * D[0, cs[0]]), cs
    INF = float("inf")
    # dp[mask][j] = shortest path from depot visiting mask, ending at cs[j]
    size = 1 << n
    dp = np.full((size, n), INF)
    parent = np.full((size, n), -1, dtype=int)
    for j in range(n):
        dp[1 << j, j] = D[0, cs[j]]
    for mask in range(size):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        for j in range(n):
            if not (mask >> j) & 1 or not np.isfinite(row[j]):
                continue
            base = row[j]
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                v = base + D[cs[j], cs[k]]
                if v < dp[nm, k]:
                    dp[nm, k] = v
                    parent[nm, k] = j
    full = size - 1
    best_j = int(np.argmin(dp[full] + D[[cs[j] for j in range(n)], 0]))
    length = float(dp[full, best_j] + D[cs[best_j], 0])
    order = []
    mask, j = full, best_j
    while j >= 0:
        order.append(cs[j])
        pj = parent[mask, j]
        mask ^= 1 << j
        j = pj
    order.reverse()
    return length, order


def route_exactly(inst: Instance, a: Assignment) -> Tuple[float, Dict[int, Tuple[float, List[int]]]]:
    """CVRP cost of a clustering when every cluster is routed by exact TSP.
    Returns (total, {vehicle: (length, order of customer indices 0..N-1)})."""
    D = inst.dist_matrix()
    total = 0.0
    routes: Dict[int, Tuple[float, List[int]]] = {}
    for k in range(inst.K):
        members = [i + 1 for i in range(inst.N) if a[i] == k]
        if not members:
            continue
        L, order = tsp_exact(D, members)
        routes[k] = (L, [c - 1 for c in order])
        total += L
    return total, routes


def gap_percent(C: float, C_ref: float) -> float:
    """Brief Eq. (2).  May be negative if C beats a best-known reference."""
    return 100.0 * (C - C_ref) / C_ref


# --------------------------------------------------------------------------- #
# Formulation interface (Gate 1 plugs the real one in here)
# --------------------------------------------------------------------------- #
class Formulation(ABC):
    """A clustering model as a QUBO over binary variables.

    Subclasses set ``name`` and implement the abstract methods.  ``weights``
    holds every penalty weight so DERIVATIONS.md and the code agree.
    Objective + feasibility are *model level*: they define the model gap
    (brief §9.2, "Model gap"), independent of how the QUBO encodes them.
    """

    name: str = "abstract"

    def __init__(self, inst: Instance, **weights: float) -> None:
        self.inst = inst
        self.weights: Dict[str, float] = dict(weights)

    # ---- size ------------------------------------------------------------- #
    @property
    @abstractmethod
    def n_vars(self) -> int: ...

    @abstractmethod
    def var_names(self) -> List[str]: ...

    @classmethod
    @abstractmethod
    def count_vars(cls, N: int, K: int, Q: int) -> int:
        """Variable count from the tier parameters alone (for the §4 table)."""

    # ---- QUBO ------------------------------------------------------------- #
    @abstractmethod
    def Q(self) -> np.ndarray:
        """Symmetric (n, n) matrix: E(x) = x^T Q x + offset."""

    offset: float = 0.0

    def linear(self) -> np.ndarray:
        return np.diag(self.Q()).copy()

    def couplings(self) -> np.ndarray:
        J = 2.0 * self.Q()
        np.fill_diagonal(J, 0.0)
        return J

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, float)
        return float(x @ self.Q() @ x + self.offset)

    def energies(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return np.einsum("mi,ij,mj->m", X, self.Q(), X) + self.offset

    # ---- semantics ---------------------------------------------------------- #
    @abstractmethod
    def decode(self, x: np.ndarray) -> Optional[Assignment]:
        """Bitstring -> assignment, or None when the bits do not describe a
        valid customer->vehicle map (e.g. a one-hot row with 0 or 2 ones)."""

    def feasible(self, a: Optional[Assignment]) -> bool:
        """Model-level feasibility: decodable + every cluster within capacity."""
        return a is not None and self.inst.capacity_ok(a)

    @abstractmethod
    def objective(self, a: Assignment) -> float:
        """Clustering objective of the model (defines the model gap)."""

    def cluster_cost(self, members: FrozenSet[int]) -> float:
        """If the objective is a sum over clusters, return the cost of one
        cluster here (enables the exact partition DP in cvrp.reference).
        Default: not additive."""
        raise NotImplementedError

    @property
    def additive(self) -> bool:
        try:
            self.cluster_cost(frozenset([0]))
            return True
        except NotImplementedError:
            return False

    # ---- description (goes into results.csv / reports) --------------------- #
    def describe(self) -> Dict[str, Any]:
        return {"formulation": self.name, "n_vars": self.n_vars, **{f"w_{k}": v for k, v in self.weights.items()}}


def all_bitstrings(n: int) -> np.ndarray:
    """(2^n, n) array of all bitstrings, bit i = column i (n <= 24)."""
    if n > 24:
        raise ValueError("too many variables to enumerate")
    idx = np.arange(1 << n, dtype=np.int64)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)


def bits_to_str(x: Iterable[int]) -> str:
    return "".join(str(int(b)) for b in x)


def str_to_bits(s: str) -> np.ndarray:
    return np.array([int(c) for c in s], dtype=np.int8)
