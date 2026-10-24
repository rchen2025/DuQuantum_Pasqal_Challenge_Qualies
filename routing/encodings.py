"""Two TSP encodings with uniform linear terms and non-negative couplings.

Conventions (shared with ``cvrp.common``):

* ``D`` is the (n+1, n+1) symmetric non-negative distance matrix of one cluster,
  **row/column 0 is the depot**, rows 1..n are the customers in *local* index
  order.  Original customer ids are the caller's business (``routing.fixtures``
  keeps them separately); nothing in here ever sees them.
* A QUBO is  E(x) = sum_u h_u x_u + sum_{u<v} J_uv x_u x_v + offset, with
  ``J`` symmetric and zero on the diagonal.  The symmetric matrix form used by
  the rest of the repo is  E(x) = x^T Q x + offset  with  Q_uu = h_u and
  Q_uv = J_uv / 2.
* A *tour* is the tuple of local customer indices (1..n) in visiting order; the
  depot is implicit at both ends.  ``tour_cost`` includes both depot legs.

Encoding 1 — depot-fixed position one-hot (handoff spec, "Encoding 1")
---------------------------------------------------------------------
Variables y_{i,t}, i = customer 1..n, t = position 1..n  (n^2 bits; the depot
has no variables).  With

    w(i,j,t) = c_ij + [t = 1] c_0i + [t = n-1] c_j0        (i != j, t = 1..n-1)

the pair sum  F_pair = sum_t sum_{i!=j} w(i,j,t) y_{i,t} y_{j,t+1}  equals the
depot-to-depot route cost on every permutation (the first transition carries the
outgoing depot leg, the last the returning one; for n = 2 the single transition
carries both).  Energy

    E = F_pair + A sum_i (sum_t y_it - 1)^2 + A sum_t (sum_i y_it - 1)^2 .

Expanding with y^2 = y:  offset = 2nA, every h = -2A, J += 2A for two
positions of the same customer and for two customers in the same position,
J += w(i,j,t) for (i,t),(j,t+1) with i != j.

Encoding 2 — undirected edge variables (handoff spec, "Encoding 2")
-------------------------------------------------------------------
One bit x_e per undirected edge of the complete graph on the m = n+1 vertices
(n(n+1)/2 bits).  d_v = sum_{e incident to v} x_e.

    F_pair = sum_v sum_{unordered distinct e,f incident to v} (c_e + c_f)/2 x_e x_f
    E      = F_pair + A sum_v (d_v - 2)^2

When every degree is 2 each selected edge contributes half its length at each
endpoint, so F_pair = tour length.  Expanding:  offset = 4Am, every h = -6A,
J_ef = 2A + (c_e + c_f)/2 when e, f share a vertex, 0 otherwise.
Degree-2 subgraphs of K_5 are Hamiltonian cycles (two cycles need >= 6
vertices), so for n = 4 this is exact TSP; for n >= 5 the ground state may be
a cycle cover (two triangles) — ``build_edge`` refuses n > 4 unless the caller
asks for the relaxation explicitly (``allow_subtours=True``, used only by
``routing.check`` to *demonstrate* the failure).

Penalty weight
--------------
A = C_ub * (1 + A_MARGIN) with C_ub the length of the nearest-neighbour tour.
F_pair >= 0 on every bitstring, every infeasible string pays at least one
integer-squared penalty A, and a feasible string of energy C_ub exists, so no
infeasible string can be the ground state.  No optimum is needed to set A.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

A_MARGIN = 0.01          # A = C_ub * (1 + A_MARGIN)
TIE_TOL = 1e-9           # energies within this are "equal"

Tour = Tuple[int, ...]   # local customer indices 1..n in visiting order (depot implicit)


# --------------------------------------------------------------------------- #
# Distances / tours
# --------------------------------------------------------------------------- #
def check_dist_matrix(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 2:
        raise ValueError(f"D must be a square (n+1)x(n+1) matrix with n >= 1, got {D.shape}")
    if not np.allclose(D, D.T, atol=1e-9):
        raise ValueError("D must be symmetric")
    if (D < -1e-12).any() or not np.isfinite(D).all():
        raise ValueError("D must be finite and non-negative")
    return D


def tour_cost(D: np.ndarray, tour: Sequence[int]) -> float:
    """Depot -> tour[0] -> ... -> tour[-1] -> depot."""
    if len(tour) == 0:
        return 0.0
    c = D[0, tour[0]] + D[tour[-1], 0]
    for a, b in zip(tour[:-1], tour[1:]):
        c += D[a, b]
    return float(c)


def canonical_tour(tour: Sequence[int]) -> Tour:
    """Undirected canonical form: the direction whose first customer is the smaller one."""
    t = tuple(int(v) for v in tour)
    if len(t) >= 2 and t[-1] < t[0]:
        t = t[::-1]
    return t


def nearest_neighbour_tour(D: np.ndarray) -> Tuple[float, Tour]:
    """Cheap valid tour: greedy nearest unvisited customer from the depot.  Its cost C_ub is
    the only thing the penalty weight A depends on."""
    n = D.shape[0] - 1
    left = set(range(1, n + 1))
    cur, order = 0, []
    while left:
        nxt = min(left, key=lambda j: (D[cur, j], j))
        order.append(nxt)
        left.remove(nxt)
        cur = nxt
    return tour_cost(D, order), tuple(order)


def penalty_weight(D: np.ndarray, margin: float = A_MARGIN) -> Tuple[float, float, Tour]:
    """(A, C_ub, nn_tour) with A = C_ub (1 + margin)."""
    C_ub, t = nearest_neighbour_tour(D)
    return float(C_ub * (1.0 + margin)), float(C_ub), t


# --------------------------------------------------------------------------- #
# Encoding object
# --------------------------------------------------------------------------- #
@dataclass
class Encoding:
    """A TSP QUBO plus its strict decoder.

    h, J, offset:  E(x) = sum h_u x_u + sum_{u<v} J_uv x_u x_v + offset  (J symmetric, zero diag)
    Q:             E(x) = x^T Q x + offset  (Q_uu = h_u, Q_uv = J_uv/2)
    """
    name: str
    n_customers: int
    D: np.ndarray
    var_names: List[str]
    h: np.ndarray
    J: np.ndarray
    offset: float
    A: float
    C_ub: float
    nn_tour: Tour
    meta: Dict[str, object] = field(default_factory=dict)

    # ---- sizes ---------------------------------------------------------- #
    @property
    def n_vars(self) -> int:
        return int(self.h.shape[0])

    # ---- energies ------------------------------------------------------- #
    def Q(self) -> np.ndarray:
        Q = 0.5 * self.J.copy()
        np.fill_diagonal(Q, self.h)
        return Q

    def energy(self, x: Sequence[int]) -> float:
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_vars:
            raise ValueError(f"bitstring has {x.shape[0]} bits, encoding has {self.n_vars}")
        return float(x @ self.h + 0.5 * x @ self.J @ x + self.offset)

    def energies(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.h + 0.5 * np.einsum("mi,ij,mj->m", X, self.J, X) + self.offset

    def energy_of_str(self, s: str) -> float:
        return self.energy([int(c) for c in s])

    # ---- semantics (overridden per encoding) ---------------------------- #
    def decode(self, x: Sequence[int]) -> Optional[Tour]:        # pragma: no cover - abstract
        raise NotImplementedError

    def encode(self, tour: Sequence[int]) -> np.ndarray:         # pragma: no cover - abstract
        raise NotImplementedError

    def tour_cost(self, tour: Sequence[int]) -> float:
        return tour_cost(self.D, tour)

    def describe(self) -> Dict[str, object]:
        return {
            "encoding": self.name, "n_customers": self.n_customers, "n_vars": self.n_vars,
            "A": self.A, "C_ub": self.C_ub, "nn_tour": list(self.nn_tour), "offset": self.offset,
            "h_uniform_value": float(self.h[0]), "J_min_offdiag": float(self.J[~np.eye(self.n_vars, dtype=bool)].min()),
            "J_max": float(self.J.max()), "n_zero_couplings": int(np.sum(np.triu(self.J == 0, 1))),
            **{k: (v if not isinstance(v, (list, tuple)) else [list(e) if isinstance(e, tuple) else e for e in v])
               for k, v in self.meta.items() if k != "edge_index"},
        }


# --------------------------------------------------------------------------- #
# Encoding 1: depot-fixed position one-hot
# --------------------------------------------------------------------------- #
class PositionEncoding(Encoding):
    """y_{i,t}: customer i (1..n) at position t (1..n); variable index (i-1)*n + (t-1)."""

    def var(self, i: int, t: int) -> int:
        n = self.n_customers
        return (i - 1) * n + (t - 1)

    def decode(self, x: Sequence[int]) -> Optional[Tour]:
        n = self.n_customers
        y = np.asarray(x, dtype=int).reshape(n, n)       # y[i-1, t-1]
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("bits must be 0/1")
        if not (y.sum(1) == 1).all() or not (y.sum(0) == 1).all():
            return None                                   # not a permutation matrix
        order = [int(np.argmax(y[:, t]) + 1) for t in range(n)]
        return tuple(order)

    def encode(self, tour: Sequence[int]) -> np.ndarray:
        n = self.n_customers
        if sorted(tour) != list(range(1, n + 1)):
            raise ValueError("tour must be a permutation of 1..n")
        x = np.zeros(n * n, dtype=np.int8)
        for t, i in enumerate(tour, start=1):
            x[self.var(i, t)] = 1
        return x


def transition_weight(D: np.ndarray, i: int, j: int, t: int) -> float:
    """w(i,j,t) = c_ij + [t=1] c_0i + [t=n-1] c_j0  for i != j, t = 1..n-1 (1-indexed positions)."""
    n = D.shape[0] - 1
    if i == j or not (1 <= t <= n - 1):
        raise ValueError("w(i,j,t) needs i != j and 1 <= t <= n-1")
    w = D[i, j]
    if t == 1:
        w += D[0, i]
    if t == n - 1:
        w += D[j, 0]
    return float(w)


def build_position(D: np.ndarray, A: Optional[float] = None, margin: float = A_MARGIN) -> PositionEncoding:
    """Encoding 1.  n >= 2 (n <= 1 is handled classically by the caller)."""
    D = check_dist_matrix(D)
    n = D.shape[0] - 1
    if n < 2:
        raise ValueError("position encoding needs n >= 2 customers (route n <= 1 classically)")
    A_auto, C_ub, nn = penalty_weight(D, margin)
    if A is None:
        A = A_auto
    N = n * n
    h = np.full(N, -2.0 * A)
    J = np.zeros((N, N))
    offset = 2.0 * n * A

    def var(i: int, t: int) -> int:
        return (i - 1) * n + (t - 1)

    def add(u: int, v: int, val: float) -> None:        # canonical unordered pair, symmetric store
        if u == v:
            raise RuntimeError("pair term on a single variable")
        J[u, v] += val
        J[v, u] += val

    # one-hot rows (each customer in exactly one position)
    for i in range(1, n + 1):
        for t in range(1, n + 1):
            for t2 in range(t + 1, n + 1):
                add(var(i, t), var(i, t2), 2.0 * A)
    # one-hot columns (each position holds exactly one customer)
    for t in range(1, n + 1):
        for i in range(1, n + 1):
            for i2 in range(i + 1, n + 1):
                add(var(i, t), var(i2, t), 2.0 * A)
    # transitions with the depot legs folded in
    for t in range(1, n):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    add(var(i, t), var(j, t + 1), transition_weight(D, i, j, t))
    names = [f"y[c{i},p{t}]" for i in range(1, n + 1) for t in range(1, n + 1)]
    return PositionEncoding(name="position", n_customers=n, D=D, var_names=names, h=h, J=J, offset=offset,
                            A=float(A), C_ub=C_ub, nn_tour=nn, meta={"A_margin": margin, "A_auto": A_auto})


# --------------------------------------------------------------------------- #
# Encoding 2: undirected edge variables
# --------------------------------------------------------------------------- #
def edge_list(m: int) -> List[Tuple[int, int]]:
    """All undirected edges (u, v), u < v, of K_m in lexicographic order."""
    return [(u, v) for u in range(m) for v in range(u + 1, m)]


class EdgeEncoding(Encoding):
    """x_e for every undirected edge e = (u, v), u < v, of K_{n+1} (vertex 0 = depot)."""

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return list(self.meta["edges"])  # type: ignore[arg-type]

    def edge_index(self, u: int, v: int) -> int:
        return self.meta["edge_index"][(min(u, v), max(u, v))]  # type: ignore[index]

    def selected_edges(self, x: Sequence[int]) -> List[Tuple[int, int]]:
        x = np.asarray(x, dtype=int).ravel()
        return [e for e, b in zip(self.edges, x) if b]

    def degrees(self, x: Sequence[int]) -> np.ndarray:
        m = self.n_customers + 1
        d = np.zeros(m, dtype=int)
        for u, v in self.selected_edges(x):
            d[u] += 1
            d[v] += 1
        return d

    def components(self, x: Sequence[int]) -> List[List[int]]:
        """Connected components (vertex lists) of the selected-edge graph, isolated vertices included."""
        m = self.n_customers + 1
        adj: Dict[int, List[int]] = {v: [] for v in range(m)}
        for u, v in self.selected_edges(x):
            adj[u].append(v)
            adj[v].append(u)
        seen, comps = set(), []
        for s in range(m):
            if s in seen:
                continue
            stack, comp = [s], []
            seen.add(s)
            while stack:
                v = stack.pop()
                comp.append(v)
                for w in adj[v]:
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
            comps.append(sorted(comp))
        return comps

    def is_degree_two(self, x: Sequence[int]) -> bool:
        return bool((self.degrees(x) == 2).all())

    def decode(self, x: Sequence[int]) -> Optional[Tour]:
        """Strict: every degree 2 AND one connected cycle through all vertices; returned
        in canonical direction (second vertex smaller than last)."""
        x = np.asarray(x, dtype=int).ravel()
        if x.shape[0] != self.n_vars or not set(np.unique(x)).issubset({0, 1}):
            raise ValueError("bits must be 0/1 of the right length")
        if not self.is_degree_two(x):
            return None
        m = self.n_customers + 1
        adj: Dict[int, List[int]] = {v: [] for v in range(m)}
        for u, v in self.selected_edges(x):
            adj[u].append(v)
            adj[v].append(u)
        # walk from the depot
        order, prev, cur = [0], -1, 0
        while True:
            a, b = adj[cur]
            nxt = a if a != prev else b
            if nxt == 0:
                break
            order.append(nxt)
            prev, cur = cur, nxt
            if len(order) > m:
                return None   # cannot happen with degrees 2, defensive
        if len(order) != m:
            return None       # degree-2 but disconnected: a cycle cover, not a tour
        tour = tuple(order[1:])
        return canonical_tour(tour)

    def encode(self, tour: Sequence[int]) -> np.ndarray:
        n = self.n_customers
        if sorted(tour) != list(range(1, n + 1)):
            raise ValueError("tour must be a permutation of 1..n")
        x = np.zeros(self.n_vars, dtype=np.int8)
        cyc = [0, *tour, 0]
        for a, b in zip(cyc[:-1], cyc[1:]):
            x[self.edge_index(a, b)] = 1
        return x


EDGE_MAX_CUSTOMERS = 4   # K_5 has no two disjoint cycles; K_6 does (two triangles)


def build_edge(D: np.ndarray, A: Optional[float] = None, margin: float = A_MARGIN,
               allow_subtours: bool = False) -> EdgeEncoding:
    """Encoding 2.  Exact TSP for 2 <= n <= 4.  For n > 4 the degree-only model is a
    cycle-cover relaxation: refused unless ``allow_subtours=True`` (research/check use only)."""
    D = check_dist_matrix(D)
    n = D.shape[0] - 1
    if n < 2:
        raise ValueError("edge encoding needs n >= 2 customers (route n <= 1 classically)")
    if n > EDGE_MAX_CUSTOMERS and not allow_subtours:
        raise ValueError(
            f"edge encoding with n = {n} > {EDGE_MAX_CUSTOMERS}: on {n + 1} >= 6 vertices the degree-2 "
            "constraint admits disjoint cycles (two triangles), so the ground state need not be a tour. "
            "Use the position encoding, or pass allow_subtours=True to study the relaxation.")
    A_auto, C_ub, nn = penalty_weight(D, margin)
    if A is None:
        A = A_auto
    m = n + 1
    E = edge_list(m)
    idx = {e: k for k, e in enumerate(E)}
    N = len(E)
    h = np.full(N, -6.0 * A)
    J = np.zeros((N, N))
    offset = 4.0 * A * m
    for v in range(m):
        inc = [k for k, (a, b) in enumerate(E) if v in (a, b)]
        for p in range(len(inc)):
            for q in range(p + 1, len(inc)):
                e, f = inc[p], inc[q]
                val = 2.0 * A + 0.5 * (D[E[e]] + D[E[f]])
                J[e, f] += val
                J[f, e] += val
    names = [f"x[{u}-{v}]" for u, v in E]
    return EdgeEncoding(name="edge", n_customers=n, D=D, var_names=names, h=h, J=J, offset=offset,
                        A=float(A), C_ub=C_ub, nn_tour=nn,
                        meta={"A_margin": margin, "A_auto": A_auto, "edges": E, "edge_index": idx,
                              "allow_subtours": allow_subtours})


BUILDERS = {"position": build_position, "edge": build_edge}


def build(name: str, D: np.ndarray, **kw) -> Encoding:
    return BUILDERS[name](D, **kw)
