"""routing.solve_cluster: validation, classical paths, fallback flags, cost recomputation, decomposition identity.

No emulation here (the n = 4 quantum path is exercised by scripts/routing_integrate.py).
Run:  PYTHONPATH=/home/claude/_pydeps:. python3 scripts/run_tests.py routing_solve
"""
import itertools
from pathlib import Path

import numpy as np

from cvrp.common import Instance, clustering_of, route_exactly, tsp_exact
from routing.encodings import tour_cost
from routing.solve_cluster import (FALLBACK_REASON_LARGE, RoutingConfig, cluster_name, local_dist_matrix, local_fixture,
                                   route_cost_from_instance, solve_cluster, validate_cluster, validate_clustering)

ROOT = Path(__file__).resolve().parents[1]
INST = Instance.load(ROOT / "instances" / "M_seed01.json")          # N = 10, K = 3, Q = 20


def _raises(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError as e:
        return str(e)
    raise AssertionError("expected ValueError")


def test_validation_errors_are_explicit():
    assert "empty" in _raises(validate_cluster, INST, [])
    assert "duplicate" in _raises(validate_cluster, INST, [1, 1, 2])
    assert "not customers" in _raises(validate_cluster, INST, [0, INST.N])
    assert "not customers" in _raises(validate_cluster, INST, [-1])
    # demands of M_seed01: [2,4,5,5,8,3,3,8,2,4]; {3,4,7} = 21 > 20
    assert "> Q" in _raises(validate_cluster, INST, [3, 4, 7])
    assert validate_cluster(INST, (7, 2, 9)) == [7, 2, 9]            # order preserved (it is the local index order)
    # clustering level
    good = validate_clustering(INST, [[0, 1, 6], [2, 7, 8, 9], [3, 4, 5]])
    assert good == [[0, 1, 6], [2, 7, 8, 9], [3, 4, 5]]
    assert "not covering" in _raises(validate_clustering, INST, [[0, 1, 6], [2, 7, 8, 9]])
    assert "not disjoint" in _raises(validate_clustering, INST, [[0, 1, 6], [2, 7, 8, 9], [3, 4, 5, 6]])   # 6 twice (19 <= Q)
    assert "> K" in _raises(validate_clustering, INST, [[0, 1], [6], [2, 7, 8, 9], [3, 4, 5]])
    # a disjointness failure that is not masked by capacity: customer 8 (demand 2) twice
    assert "not disjoint" in _raises(validate_clustering, INST, [[0, 1, 6, 8], [2, 7, 9, 8], [3, 4, 5]])


def test_local_matrix_has_depot_at_zero_and_matches_instance():
    ids = [7, 2, 9]
    Dl = local_dist_matrix(INST, ids)
    D = INST.dist_matrix()
    assert Dl.shape == (4, 4) and abs(Dl).trace() == 0
    for k, c in enumerate(ids, start=1):
        assert abs(Dl[0, k] - D[0, c + 1]) < 1e-12               # depot leg: instance row 0 is the depot
        for l, d in enumerate(ids, start=1):
            assert abs(Dl[k, l] - D[c + 1, d + 1]) < 1e-12
    fx = local_fixture(INST, ids, "x")
    assert np.allclose(fx.dist_matrix(), Dl) and fx.customer_ids == (7, 2, 9) and fx.local_to_id(1) == 7
    assert cluster_name(INST, ids) == "M_seed01_c7-2-9" and cluster_name(INST, ids, "t") == "t__M_seed01_c7-2-9"


def test_route_cost_is_recomputed_from_the_instance_matrix():
    ids = [2, 7, 8, 9]
    Dl = local_dist_matrix(INST, ids)
    for perm in itertools.permutations(range(1, 5)):
        route = [0, *perm, 0]
        assert abs(route_cost_from_instance(INST, ids, route) - tour_cost(Dl, perm)) < 1e-9
    assert "depot" in _raises(route_cost_from_instance, INST, ids, [1, 2, 3, 4, 0])
    assert "exactly once" in _raises(route_cost_from_instance, INST, ids, [0, 1, 2, 3, 0])
    assert "exactly once" in _raises(route_cost_from_instance, INST, ids, [0, 1, 2, 3, 3, 0])


def test_small_clusters_are_exact_classical_and_not_flagged():
    for ids in ([4], [1, 8], [0, 1, 6]):
        r = solve_cluster(INST, ids)
        assert r.method == "classical-exact" and not r.fallback and r.fallback_reason is None
        assert not r.quantum_attempted and r.encoding is None and r.atoms == 0 and r.shots == 0
        assert r.route[0] == 0 and r.route[-1] == 0 and sorted(r.route[1:-1]) == list(range(1, len(ids) + 1))
        assert r.route_ids[0] == "depot" and r.route_ids[-1] == "depot" and sorted(r.route_ids[1:-1]) == sorted(ids)
        L, _ = tsp_exact(local_dist_matrix(INST, ids), list(range(1, len(ids) + 1)))
        assert abs(r.cost - L) < 1e-9 and abs(r.exact_cost - L) < 1e-9 and abs(r.gap_percent_vs_exact) < 1e-12
        assert r.n == len(ids) and r.customer_ids == list(ids) and r.instance == "M_seed01"
        d = r.to_dict()
        assert d["label"].startswith("emulation only") and "versions" in d and d["seed"] == 0
    r1 = solve_cluster(INST, [4])
    assert abs(r1.cost - 2 * INST.dist_matrix()[0, 5]) < 1e-12 and r1.route == [0, 1, 0]


def test_large_cluster_is_exact_fallback_and_flagged():
    ids = [0, 2, 5, 6, 8]                                   # demand 2+5+3+3+2 = 15 <= 20
    r = solve_cluster(INST, ids)
    assert r.method == "classical-exact-fallback" and r.fallback and r.fallback_reason == FALLBACK_REASON_LARGE
    assert not r.quantum_attempted and r.encoding is None and r.raw_best_cost is None
    L, _ = tsp_exact(local_dist_matrix(INST, ids), list(range(1, 6)))
    assert abs(r.cost - L) < 1e-9 and abs(r.cost - route_cost_from_instance(INST, ids, r.route)) < 1e-12


def test_decomposition_identity_and_sum_of_cluster_costs():
    """C_end - C_ref = (C_dec - C_ref) + (C_clust - C_dec) + (C_end - C_clust) holds by algebra;
    here C_end (sum of returned routes, all exact for n <= 3 / fallback) equals C_clust = route_exactly."""
    a = (1, 1, 0, 2, 2, 2, 1, 0, 1, 0)                     # the M_seed01 best feasible shot
    clusters = validate_clustering(INST, clustering_of(a))
    results = [solve_cluster(INST, g) for g in clusters if len(g) != 4]      # skip the quantum path (emulation)
    quad = [g for g in clusters if len(g) == 4]
    C_end = sum(r.cost for r in results) + sum(tsp_exact(local_dist_matrix(INST, g), [1, 2, 3, 4])[0] for g in quad)
    C_clust, _ = route_exactly(INST, a)
    assert abs(C_end - C_clust) < 1e-9
    C_ref = INST.reference["cost"]
    a_opt = tuple(INST.reference["best_assignment"])
    C_dec, _ = route_exactly(INST, a_opt)
    lhs = C_end - C_ref
    rhs = (C_dec - C_ref) + (C_clust - C_dec) + (C_end - C_clust)
    assert abs(lhs - rhs) < 1e-9


def test_config_roundtrip():
    cfg = RoutingConfig(shots=7, seed=3, embed_seeds=2, position_emulation="no", out_dir="x", tag="t")
    p = cfg.pilot_config("name")
    assert p.shots == 7 and p.seed == 3 and p.embed_seeds == 2 and p.position_emulation == "no" and p.out_dir == "x"
    assert p.fixture == "name"
