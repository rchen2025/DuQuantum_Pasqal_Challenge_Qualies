"""Tests for cvrp.exact (set-partition DP, Held-Karp table) and cvrp.reference.

Plain ``def test_*():`` functions with no fixtures so they run both under real
pytest and under ``scripts/run_tests.py``.  Everything is seeded.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from cvrp.common import Instance, clustering_of, route_exactly, tsp_exact
from cvrp.exact import (cvrp_cluster_costs, exact_partition, mask_to_set, partition_dp,
                        popcount, solve_cvrp, subset_sums, tsp_tour_table)
from cvrp.instances import generate
from cvrp.reference import brute_force_naive, compute_reference, simulated_annealing

TOL = 1e-9


def _random_instance(rng: np.random.Generator, N: int, K: int, Q: int, d_max: int) -> Instance:
    """A small ad-hoc instance (not tied to a tier) whose certificate is just
    *some* feasible assignment found by trial; Q is raised one unit at a time
    until such an assignment exists, so the exact solver always has something
    to find (and the loop always terminates)."""
    demands = [int(x) for x in rng.integers(1, d_max + 1, size=N)]
    while True:
        for a in itertools.product(range(K), repeat=N):
            loads = [0] * K
            for i, k in enumerate(a):
                loads[k] += demands[i]
            if max(loads) <= Q:
                break
        else:
            Q += 1
            continue
        locs = [(round(float(x), 2), round(float(y), 2)) for x, y in rng.uniform(0, 100, size=(N, 2))]
        return Instance(name="T_seed00", tier="T", seed=0, N=N, K=K, Q=Q, depot=(50.0, 50.0),
                        locations=locs, demands=demands, certificate=tuple(a))


def _naive_optimum(inst: Instance):
    """All K^N assignments, exact routing, distinct optimal clusterings."""
    best, opt = math.inf, set()
    for a in itertools.product(range(inst.K), repeat=inst.N):
        if not inst.capacity_ok(a):
            continue
        c, _ = route_exactly(inst, a)
        if c < best - 1e-9:
            best, opt = c, {clustering_of(a)}
        elif abs(c - best) <= 1e-9:
            opt.add(clustering_of(a))
    return best, len(opt)


# --------------------------------------------------------------------------- #
# bit helpers
# --------------------------------------------------------------------------- #
def test_popcount_and_subset_sums():
    masks = np.arange(1 << 6)
    assert popcount(masks, 6).tolist() == [bin(m).count("1") for m in range(1 << 6)]
    vals = [3, 1, 4, 1, 5, 9]
    sums = subset_sums(vals, 6)
    for m in range(1 << 6):
        assert sums[m] == sum(vals[i] for i in range(6) if (m >> i) & 1)
    assert mask_to_set(0b1011) == frozenset({0, 1, 3})


# --------------------------------------------------------------------------- #
# Held-Karp table vs common.tsp_exact
# --------------------------------------------------------------------------- #
def test_tsp_table_matches_tsp_exact_on_random_subsets():
    rng = np.random.default_rng(1)
    for N in (1, 2, 5, 9):
        pts = rng.uniform(0, 100, size=(N + 1, 2))
        D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        table = tsp_tour_table(D)
        assert table.shape == (1 << N,) and table[0] == 0.0
        for _ in range(40):
            mask = int(rng.integers(1, 1 << N))
            members = [i + 1 for i in range(N) if (mask >> i) & 1]
            L, order = tsp_exact(D, members)
            assert abs(table[mask] - L) < 1e-9, (N, mask, table[mask], L)
            assert sorted(order) == members


def test_tsp_table_full_instance_L():
    inst = generate("L", 0)
    D = inst.dist_matrix()
    table = tsp_tour_table(D)
    L_full, _ = tsp_exact(D, list(range(1, inst.N + 1)))
    assert abs(table[-1] - L_full) < 1e-9
    # a single customer: out and back
    for i in range(inst.N):
        assert abs(table[1 << i] - 2 * D[0, i + 1]) < 1e-12


# --------------------------------------------------------------------------- #
# set-partition DP
# --------------------------------------------------------------------------- #
def test_partition_dp_simple_hand_case():
    """3 items, cost = size^2 (so singletons are cheapest), K limits the count."""
    N = 3
    cost = np.array([math.inf] + [bin(m).count("1") ** 2 for m in range(1, 1 << N)], dtype=float)
    v1, cl1, n1 = exact_partition(N, 1, cost)
    assert v1 == 9 and cl1 == [frozenset({0, 1, 2})] and n1 == 1
    v2, cl2, n2 = exact_partition(N, 2, cost)
    assert v2 == 5 and n2 == 3                    # {a},{b,c} for each of the 3 choices of a
    v3, cl3, n3 = exact_partition(N, 3, cost)
    assert v3 == 3 and n3 == 1 and len(cl3) == 3
    # callable interface, with an infeasible cluster
    v, cl, n = exact_partition(N, 3, lambda s: math.inf if s == frozenset({0}) else len(s) ** 2)
    assert v == 4 + 1 and n == 2                  # {0,1},{2} or {0,2},{1}
    # all infeasible -> inf, [], 0
    v, cl, n = exact_partition(N, 3, lambda s: math.inf)
    assert v == math.inf and cl == [] and n == 0


def test_partition_dp_counts_all_set_partitions_when_costs_tie():
    """cost = 0 for every cluster -> every partition is optimal; the count must
    be the number of partitions of N items into <= K blocks."""
    def stirling2(n, k):
        if n == k == 0:
            return 1
        if n == 0 or k == 0:
            return 0
        return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)
    for N in (1, 2, 4, 6):
        cost = np.zeros(1 << N)
        cost[0] = math.inf
        for K in range(1, N + 1):
            v, cl, n = exact_partition(N, K, cost)
            assert v == 0.0
            assert n == sum(stirling2(N, k) for k in range(1, K + 1)), (N, K, n)
            assert len(cl) <= K and frozenset().union(*cl) == frozenset(range(N))


def test_exact_cvrp_agrees_with_naive_brute_force_on_random_small_instances():
    rng = np.random.default_rng(42)
    for trial in range(25):
        N = int(rng.integers(2, 7))
        K = int(rng.integers(1, 4))
        d_max = int(rng.integers(1, 6))
        Q = int(rng.integers(d_max, 3 * d_max + 2))
        inst = _random_instance(rng, N, K, Q, d_max)
        naive_cost, naive_n = _naive_optimum(inst)
        r = solve_cvrp(inst)
        assert abs(r["cost"] - naive_cost) < 1e-9, (trial, r["cost"], naive_cost)
        assert r["n_optimal"] == naive_n, (trial, r["n_optimal"], naive_n)
        # the returned assignment really costs that much and is feasible
        assert inst.is_valid_assignment(r["assignment"])
        c, _ = route_exactly(inst, r["assignment"])
        assert abs(c - r["cost"]) < 1e-9
        assert abs(sum(r["route_lengths"]) - r["cost"]) < 1e-9
        for k, order in enumerate(r["routes"]):
            assert sorted(order) == [i for i in range(N) if r["assignment"][i] == k]


def test_exact_cvrp_detects_symmetric_ties():
    """Four customers at the corners of a square centred on the depot, unit
    demands, K=2, Q=2 (so clusters are pairs).  The two pairings into adjacent
    corners, {01,23} and {03,12}, have the same cost by symmetry and are both
    optimal; the diagonal pairing {02,13} is longer.  So n_optimal must be 2."""
    locs = [(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)]
    inst = Instance(name="X_seed00", tier="X", seed=0, N=4, K=2, Q=2, depot=(50.0, 50.0),
                    locations=locs, demands=[1, 1, 1, 1], certificate=(0, 0, 1, 1))
    r = solve_cvrp(inst)
    assert r["n_optimal"] == 2
    naive_cost, naive_n = _naive_optimum(inst)
    assert naive_n == 2 and abs(naive_cost - r["cost"]) < 1e-9


def _set_partitions(items, max_blocks):
    """Yield every partition of ``items`` into at most ``max_blocks`` blocks."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest, max_blocks):
        # put `first` into an existing block ...
        for i in range(len(part)):
            yield part[:i] + [part[i] + [first]] + part[i + 1:]
        # ... or open a new one
        if len(part) < max_blocks:
            yield part + [[first]]


def test_partition_dp_matches_direct_partition_enumeration_on_tier_M():
    """Independent of the DP: list all set partitions of the 10 customers into
    <= 3 blocks (9842 of them), price each from the tour table, compare the
    optimum and the number of optimal partitions."""
    for seed in range(3):
        inst = generate("M", seed)
        cost = cvrp_cluster_costs(inst)
        best, n_best = math.inf, 0
        for part in _set_partitions(list(range(inst.N)), inst.K):
            c = sum(cost[sum(1 << i for i in g)] for g in part)
            if c < best - 1e-9:
                best, n_best = c, 1
            elif abs(c - best) <= 1e-9:
                n_best += 1
        r = solve_cvrp(inst)
        assert abs(r["cost"] - best) < 1e-9 and r["n_optimal"] == n_best, (seed, r["cost"], best, r["n_optimal"], n_best)


def test_cluster_cost_table_marks_overloaded_clusters_infeasible():
    inst = generate("M", 1)
    cost = cvrp_cluster_costs(inst)
    loads = subset_sums(inst.demands, inst.N)
    assert cost[0] == math.inf
    assert np.all(np.isinf(cost[1:][loads[1:] > inst.Q]))
    assert np.all(np.isfinite(cost[1:][loads[1:] <= inst.Q]))


def test_exact_solver_runs_all_tiers_and_reproduces_cost():
    for tier, seed in (("S", 0), ("M", 0), ("L", 0)):
        inst = generate(tier, seed)
        r = solve_cvrp(inst)
        c, _ = route_exactly(inst, r["assignment"])
        assert abs(c - r["cost"]) < 1e-9
        assert inst.is_valid_assignment(r["assignment"])
        cert_cost, _ = route_exactly(inst, inst.certificate)
        assert r["cost"] <= cert_cost + 1e-9          # the certificate is feasible, so C* can't beat it


# --------------------------------------------------------------------------- #
# reference block: SA, naive brute force, consistency
# --------------------------------------------------------------------------- #
def test_sa_finds_optimum_on_tier_S():
    for seed in range(5):
        inst = generate("S", seed)
        r = solve_cvrp(inst)
        best = math.inf
        for s in range(3):
            E, a, _ = simulated_annealing(inst, s)
            assert inst.is_valid_assignment(a)
            c, _ = route_exactly(inst, a)
            assert abs(c - E) < 1e-9
            best = min(best, E)
        assert abs(best - r["cost"]) < 1e-6, (seed, best, r["cost"])


def test_sa_is_deterministic_given_seed():
    inst = generate("M", 2)
    E1, a1, s1 = simulated_annealing(inst, 7)
    E2, a2, s2 = simulated_annealing(inst, 7)
    assert E1 == E2 and a1 == a2 and s1["iterations"] == s2["iterations"]


def test_naive_brute_force_matches_dp_on_tier_S():
    for seed in range(5):
        inst = generate("S", seed)
        bf = brute_force_naive(inst)
        r = solve_cvrp(inst)
        assert abs(bf["cost"] - r["cost"]) < 1e-9
        assert bf["n_distinct_optima"] == r["n_optimal"]
        assert bf["n_assignments"] == inst.K ** inst.N


def test_compute_reference_block_is_consistent_and_validates():
    from cvrp.validator import validate

    inst = generate("S", 1)
    ref = compute_reference(inst, sa_restarts=3)
    assert ref["reference_status"] == "proven_optimum"
    assert ref["agree"] is True and ref["sa_matches_optimum"] is True
    assert ref["solver_settings"]["cplex"] == "not available"
    assert ref["solver_settings"]["sa"]["restarts"] == 3 and ref["solver_settings"]["sa"]["seeds"] == [0, 1, 2]
    assert len(ref["routes"]) == inst.K
    inst.reference = ref
    assert validate(inst) == [], validate(inst)
    # the block survives a JSON round trip
    import json
    again = Instance.from_dict(json.loads(json.dumps(inst.to_dict())))
    assert validate(again) == []
