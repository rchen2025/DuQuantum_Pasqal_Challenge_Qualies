"""Exact CVRP (and generic clustering) solver by dynamic programming over subsets.

Why this is exact and still fast enough for N = 15
---------------------------------------------------
A CVRP solution for *identical* vehicles is a partition of the customers into
at most K non-empty groups ("clusters"), each within capacity, each routed by
its own shortest depot -> group -> depot tour.  The cost is a *sum over
clusters*, so the best partition can be found with two classic dynamic
programmes over bitmasks (bit i of a mask = customer i is in the set):

(a) **Held-Karp over all customers at once.**  ``dp[mask, j]`` = shortest path
    that starts at the depot, visits exactly the customers in ``mask`` and
    ends at customer ``j``.  One pass over the 2^N masks gives the exact tour
    length ``tour[mask] = min_j dp[mask, j] + D[j, depot]`` for *every* subset
    simultaneously.  Cost: O(2^N N^2) ~ 7 million operations at N = 15.

(b) **Set-partition DP.**  ``f[k][S]`` = cheapest way to cover the set ``S``
    with at most ``k`` clusters.  To avoid counting the same partition several
    times (once per ordering of its clusters) the recursion always peels off
    the cluster ``R`` that contains the *lowest-numbered* customer of ``S``::

        f[k][S] = min_{R ⊆ S, lowest(S) ∈ R}  cost[R] + f[k-1][S \\ R]

    with ``f[0][∅] = 0`` and ``f[0][S≠∅] = +inf``.  Every partition of ``S``
    corresponds to exactly one path through this recursion, so counting the
    number of ways to attain the minimum counts the optimal partitions exactly
    once each, i.e. **modulo vehicle relabeling** (brief §9.3).  The number of
    (S, R) pairs is 3^N / 2 ≈ 7 million at N = 15; the inner loop is numpy.

The partition DP (b) does not care where the cluster costs come from: it is
exposed as ``exact_partition(N, K, cluster_cost)`` so the clustering *model*
(``cvrp/model.py``) can plug in its own additive per-cluster cost and get the
model optimum with the same machinery.  The CVRP solver is just (b) with
``cost[R] = tour[R] if load(R) <= Q else +inf``.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

from cvrp.common import Assignment, Instance, assignment_of, tsp_exact

INF = float("inf")
TIE_TOL = 1e-9   # two partition costs within this are counted as the same optimum


# --------------------------------------------------------------------------- #
# Bit helpers
# --------------------------------------------------------------------------- #
def popcount(masks: np.ndarray, n_bits: int) -> np.ndarray:
    """Number of set bits of every entry (works on any numpy version)."""
    masks = np.asarray(masks)
    out = np.zeros(masks.shape, dtype=np.int64)
    for b in range(n_bits):
        out += (masks >> b) & 1
    return out


def mask_to_set(mask: int) -> frozenset:
    return frozenset(i for i in range(mask.bit_length()) if (mask >> i) & 1)


def set_to_mask(s) -> int:
    m = 0
    for i in s:
        m |= 1 << int(i)
    return m


def subset_sums(values: Sequence[int], N: int) -> np.ndarray:
    """``out[mask] = sum of values[i] over bits i of mask`` for all 2^N masks."""
    # Build the table bit by bit: the masks 2^i .. 2^(i+1)-1 are the masks
    # 0 .. 2^i-1 with bit i added, so their sums are the old table plus v_i.
    out = np.zeros(1, dtype=np.int64)
    for v in values[:N]:
        out = np.concatenate([out, out + int(v)])
    return out


# --------------------------------------------------------------------------- #
# (a) Held-Karp over all subsets at once
# --------------------------------------------------------------------------- #
def tsp_tour_table(D: np.ndarray) -> np.ndarray:
    """Exact depot->subset->depot tour length for every subset of customers.

    ``D`` is the (N+1, N+1) distance matrix with the depot at index 0 (as
    returned by ``Instance.dist_matrix``).  Returns an array ``tour`` of length
    2^N with ``tour[0] = 0`` and ``tour[mask]`` the optimal tour through the
    customers in ``mask`` (bit i <-> customer i <-> row i+1 of D).

    The DP runs layer by layer in the number of visited customers so that each
    layer is one vectorised numpy step: for all masks with p customers we form
    ``cand[mask, k] = min_j dp[mask, j] + D[j, k]`` and write it into
    ``dp[mask | bit_k, k]`` for the customers k not yet in ``mask``.  Different
    (mask, k) pairs never collide on the same target because
    ``mask = target ^ bit_k`` is unique.
    """
    N = D.shape[0] - 1
    if N == 0:
        return np.zeros(1)
    Dc = np.asarray(D[1:, 1:], dtype=float)          # customer-customer
    d0 = np.asarray(D[0, 1:], dtype=float)           # depot-customer
    size = 1 << N
    dp = np.full((size, N), INF)
    dp[1 << np.arange(N), np.arange(N)] = d0         # start: depot -> single customer
    all_masks = np.arange(size, dtype=np.int64)
    pc = popcount(all_masks, N)
    for p in range(1, N):                            # extend paths with p customers to p+1
        masks = all_masks[pc == p]
        # cand[m, k] = min over j of dp[masks[m], j] + Dc[j, k]   (inf where j not in mask)
        cand = (dp[masks][:, :, None] + Dc[None, :, :]).min(axis=1)
        for k in range(N):
            bit = 1 << k
            free = (masks & bit) == 0
            tgt = masks[free] | bit
            dp[tgt, k] = np.minimum(dp[tgt, k], cand[free, k])
    tour = (dp + d0[None, :]).min(axis=1)            # close the tour back to the depot
    tour[0] = 0.0
    return tour


# --------------------------------------------------------------------------- #
# (b) set-partition DP
# --------------------------------------------------------------------------- #
_SUBMASK_CACHE: Dict[int, np.ndarray] = {}


def _bit_table(p: int) -> np.ndarray:
    """(2^p, p) 0/1 array: row r = binary digits of r.  Cached."""
    if p not in _SUBMASK_CACHE:
        idx = np.arange(1 << p, dtype=np.int64)
        _SUBMASK_CACHE[p] = ((idx[:, None] >> np.arange(p)) & 1).astype(np.int64)
    return _SUBMASK_CACHE[p]


def _submasks(T: int, N: int) -> np.ndarray:
    """All 2^|T| submasks of ``T`` as an int64 array (vectorised)."""
    bits = [1 << i for i in range(N) if (T >> i) & 1]
    if not bits:
        return np.zeros(1, dtype=np.int64)
    return _bit_table(len(bits)) @ np.array(bits, dtype=np.int64)


def partition_dp(cost: np.ndarray, K: int, tie_tol: float = TIE_TOL
                 ) -> Tuple[float, List[int], int, np.ndarray]:
    """Cheapest partition of {0..N-1} into at most K clusters, from a cost table.

    Parameters
    ----------
    cost : array of length 2^N; ``cost[mask]`` = cost of using the customers
           in ``mask`` as one cluster, ``+inf`` if that cluster is not allowed.
           ``cost[0]`` is ignored (empty clusters do not exist).
    K    : maximum number of clusters.
    tie_tol : partitions whose cost is within this of the minimum are counted
           as optimal too.

    Returns ``(value, clusters_as_masks, n_optimal, f)`` where ``f[k][S]`` is
    the full DP table (``f[K][full]`` is the answer), ``n_optimal`` is the
    number of optimal partitions (each counted once, i.e. modulo relabeling of
    the clusters) and ``clusters_as_masks`` is one optimal partition.  If no
    partition is feasible the value is ``inf``, the list empty and the count 0.
    """
    cost = np.asarray(cost, dtype=float)
    size = cost.shape[0]
    N = size.bit_length() - 1
    assert size == 1 << N, "cost table length must be a power of two"
    full = size - 1
    K = int(K)
    if K < 1:
        raise ValueError("K must be >= 1")

    # f[k][S]: best cost covering S with <= k clusters; n[k][S]: how many ways
    # attain it; choice[k][S]: the cluster R (mask) containing lowest(S) used.
    f = np.full((K + 1, size), INF)
    n = np.zeros((K + 1, size), dtype=np.int64)
    choice = np.zeros((K + 1, size), dtype=np.int64)
    f[0, 0] = 0.0
    n[0, 0] = 1
    # k = 1: the only option is R = S itself.
    f[1] = cost
    f[1, 0] = 0.0
    n[1] = np.isfinite(cost).astype(np.int64)
    n[1, 0] = 1
    choice[1] = np.arange(size)
    choice[1, 0] = 0

    for k in range(2, K + 1):
        fp, npv = f[k - 1], n[k - 1]
        f[k, 0], n[k, 0] = 0.0, 1
        for l in range(N):                     # l = lowest customer of S
            lbit = 1 << l
            # T = S without l; T ranges over all masks using only bits > l.
            # Enumerate them as (bits above l) shifted: T = t << (l + 1).
            for t in range(1 << (N - l - 1)):
                T = t << (l + 1)
                S = T | lbit
                Rp = _submasks(T, N)           # R' = R \ {l} ⊆ T
                vals = cost[Rp | lbit] + fp[T ^ Rp]
                j = int(np.argmin(vals))
                best = vals[j]
                if not np.isfinite(best):
                    continue
                f[k, S] = best
                ties = vals <= best + tie_tol
                n[k, S] = int(npv[(T ^ Rp)[ties]].sum())
                choice[k, S] = int(Rp[j] | lbit)

    value = float(f[K, full])
    if not np.isfinite(value):
        return INF, [], 0, f
    # Reconstruct one optimum by following the stored choices.
    clusters: List[int] = []
    S, k = full, K
    while S:
        R = int(choice[k, S])
        clusters.append(R)
        S ^= R
        k -= 1
    return value, clusters, int(n[K, full]), f


CostSpec = Union[np.ndarray, Callable[[frozenset], float]]


def exact_partition(N: int, K: int, cluster_cost: CostSpec, tie_tol: float = TIE_TOL
                    ) -> Tuple[float, List[frozenset], int]:
    """Generic exact optimum of an *additive* clustering objective.

    ``cluster_cost(frozenset_of_customers) -> float`` gives the cost of one
    cluster (``inf`` = that cluster is not allowed, e.g. over capacity).  It is
    evaluated once for each of the 2^N - 1 non-empty subsets, then the
    set-partition DP finds the cheapest partition into at most K clusters.
    A precomputed length-2^N cost array is accepted in place of the callable.

    Returns ``(value, clusters, n_optimal)``: the optimal value, one optimal
    partition as a list of frozensets (ordered by their smallest member), and
    the number of distinct optimal partitions (modulo relabeling).
    """
    if callable(cluster_cost):
        cost = np.empty(1 << N)
        cost[0] = INF
        for mask in range(1, 1 << N):
            cost[mask] = float(cluster_cost(mask_to_set(mask)))
    else:
        cost = np.asarray(cluster_cost, dtype=float)
        assert cost.shape == (1 << N,)
    value, masks, n_opt, _ = partition_dp(cost, K, tie_tol)
    clusters = [mask_to_set(m) for m in masks]
    return value, clusters, n_opt


# --------------------------------------------------------------------------- #
# CVRP = partition DP with capacity-feasible exact tours as cluster costs
# --------------------------------------------------------------------------- #
def cvrp_cluster_costs(inst: Instance, tour: np.ndarray | None = None) -> np.ndarray:
    """``cost[mask] = exact tour length of the cluster, or inf if its load > Q``."""
    if tour is None:
        tour = tsp_tour_table(inst.dist_matrix())
    loads = subset_sums(inst.demands, inst.N)
    cost = np.where(loads <= inst.Q, tour, INF)
    cost[0] = INF
    return cost


def solve_cvrp(inst: Instance, tie_tol: float = TIE_TOL) -> Dict:
    """Proven-optimal CVRP solution of ``inst`` (all tiers).

    Returns a dict with ``cost`` (C*), ``assignment`` (tuple, vehicle per
    customer, clusters numbered by smallest member), ``clusters`` (list of
    sorted lists), ``routes`` (list of K visiting orders, empty for unused
    vehicles), ``route_lengths``, ``n_optimal`` (distinct optimal clusterings
    modulo relabeling) and timing information.
    """
    t0 = time.perf_counter()
    D = inst.dist_matrix()
    tour = tsp_tour_table(D)
    t1 = time.perf_counter()
    cost = cvrp_cluster_costs(inst, tour)
    value, masks, n_opt, _ = partition_dp(cost, inst.K, tie_tol)
    t2 = time.perf_counter()
    if not np.isfinite(value):
        raise ValueError(f"{inst.name}: no capacity-feasible clustering with K={inst.K} vehicles")
    clusters = [sorted(mask_to_set(m)) for m in masks]
    assignment: Assignment = assignment_of(clusters, inst.N)
    routes: List[List[int]] = [[] for _ in range(inst.K)]
    lengths = [0.0] * inst.K
    for k, g in enumerate(clusters):
        L, order = tsp_exact(D, [i + 1 for i in g])      # order is in depot-offset indices
        routes[k] = [c - 1 for c in order]
        lengths[k] = float(L)
    return {
        "cost": value,
        "assignment": assignment,
        "clusters": clusters,
        "routes": routes,
        "route_lengths": lengths,
        "n_optimal": n_opt,
        "n_feasible_clusters": int(np.isfinite(cost).sum()),
        "time_tsp_table_s": t1 - t0,
        "time_partition_dp_s": t2 - t1,
        "time_total_s": time.perf_counter() - t0,
    }


if __name__ == "__main__":   # quick manual check: python -m cvrp.exact instances/L_seed00.json
    import sys
    for path in sys.argv[1:]:
        inst = Instance.load(path)
        r = solve_cvrp(inst)
        print(f"{inst.name}: C*={r['cost']:.4f} n_opt={r['n_optimal']} clusters={r['clusters']} "
              f"tsp={r['time_tsp_table_s']:.2f}s dp={r['time_partition_dp_s']:.2f}s")
