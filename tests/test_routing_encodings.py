"""Exhaustive checks of the Track B routing encodings (standalone fixtures, classical only).

Run:  PYTHONPATH=/home/claude/_pydeps:. python3 scripts/run_tests.py routing_encodings
"""
import numpy as np

from cvrp.common import all_bitstrings, tsp_exact
from routing.check import (check_seeds, dist_matrix, edge_relaxation_failure, exhaustive_check, optimal_tours,
                           random_points, two_triangles_points)
from routing.encodings import (build_edge, build_position, canonical_tour, nearest_neighbour_tour, penalty_weight,
                               tour_cost, transition_weight)
from routing.fixtures import FIXTURES, fixture


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


# --------------------------------------------------------------------------- #
# coefficient structure (spec equations)
# --------------------------------------------------------------------------- #
def test_position_coefficients_match_spec():
    D = dist_matrix(random_points(4, 0))
    enc = build_position(D)
    n, A = 4, enc.A
    assert enc.n_vars == 16 and abs(enc.offset - 2 * n * A) < 1e-9
    assert np.allclose(enc.h, -2 * A)
    v = enc.var
    # same customer, two positions / same position, two customers: 2A
    assert abs(enc.J[v(1, 1), v(1, 3)] - 2 * A) < 1e-9
    assert abs(enc.J[v(2, 2), v(4, 2)] - 2 * A) < 1e-9
    # adjacent positions, different customers: w(i,j,t) with depot legs folded in
    assert abs(enc.J[v(2, 1), v(3, 2)] - (D[2, 3] + D[0, 2])) < 1e-9          # t = 1: outgoing depot leg
    assert abs(enc.J[v(2, 2), v(3, 3)] - D[2, 3]) < 1e-9                       # middle transition
    assert abs(enc.J[v(2, 3), v(3, 4)] - (D[2, 3] + D[3, 0])) < 1e-9          # t = n-1: return leg
    # non-adjacent positions, different customers: no coupling
    assert enc.J[v(1, 1), v(2, 3)] == 0.0 and enc.J[v(1, 1), v(2, 4)] == 0.0
    # Q convention: Q_uu = h, Q_uv = J/2
    Q = enc.Q()
    assert np.allclose(np.diag(Q), enc.h) and np.allclose(2 * (Q - np.diag(np.diag(Q))), enc.J)
    # n = 2: the single transition carries both depot legs
    D2 = dist_matrix(random_points(2, 1))
    assert abs(transition_weight(D2, 1, 2, 1) - (D2[1, 2] + D2[0, 1] + D2[2, 0])) < 1e-12


def test_edge_coefficients_match_spec():
    D = dist_matrix(random_points(4, 0))
    enc = build_edge(D)
    m, A = 5, enc.A
    assert enc.n_vars == 10 and abs(enc.offset - 4 * A * m) < 1e-9
    assert np.allclose(enc.h, -6 * A)
    e01, e02, e34, e12 = enc.edge_index(0, 1), enc.edge_index(0, 2), enc.edge_index(3, 4), enc.edge_index(1, 2)
    assert abs(enc.J[e01, e02] - (2 * A + 0.5 * (D[0, 1] + D[0, 2]))) < 1e-9   # share vertex 0
    assert abs(enc.J[e01, e12] - (2 * A + 0.5 * (D[0, 1] + D[1, 2]))) < 1e-9   # share vertex 1
    assert enc.J[e01, e34] == 0.0                                               # disjoint edges
    assert (enc.J[~np.eye(10, dtype=bool)] >= 0).all()


def test_penalty_weight_is_nn_tour_times_margin():
    D = dist_matrix(random_points(4, 2))
    A, C_ub, t = penalty_weight(D)
    C_nn, t_nn = nearest_neighbour_tour(D)
    assert t == t_nn and abs(C_ub - C_nn) < 1e-12 and abs(A - 1.01 * C_ub) < 1e-9
    assert abs(tour_cost(D, t) - C_ub) < 1e-12
    assert C_ub >= tsp_exact(D, [1, 2, 3, 4])[0] - 1e-9


# --------------------------------------------------------------------------- #
# exhaustive: n = 2, 3, 4 on three coordinate seeds, both encodings
# --------------------------------------------------------------------------- #
def test_exhaustive_n2_n3_n4_three_seeds():
    reports = check_seeds(n_values=(2, 3, 4), seeds=(0, 1, 2))
    assert len(reports) == 18
    bad = [(r["encoding"], r["n_customers"], r["seed"], r["failures"]) for r in reports if not r["ok"]]
    assert not bad, bad
    for r in reports:
        assert r["h_uniform"] and r["J_min_offdiag"] >= 0
        assert abs(r["ground_energy"] - r["C_opt"]) < 1e-7
        assert r["max_valid_tour_energy_error"] < 1e-7
        assert r["min_infeasible_energy"] > r["C_opt"]
        # position strings are directed: 2 ground states per undirected optimal tour; edge strings are undirected
        factor = 2 if r["encoding"] == "position" else 1
        assert r["n_ground_states"] == factor * r["n_optimal_undirected_tours"]


def test_decoders_are_strict():
    D = dist_matrix(random_points(4, 0))
    pos, edg = build_position(D), build_edge(D)
    # position: a row with two ones / a column with none is rejected
    x = pos.encode((1, 2, 3, 4)).copy()
    x[pos.var(1, 2)] = 1
    assert pos.decode(x) is None
    assert pos.decode(np.zeros(16, dtype=int)) is None
    assert pos.decode(pos.encode((3, 1, 4, 2))) == (3, 1, 4, 2)
    # edge: degree-2 is necessary; reversal is canonicalised
    assert edg.decode(np.zeros(10, dtype=int)) is None
    x = edg.encode((1, 2, 3, 4))
    assert edg.decode(x) == canonical_tour((1, 2, 3, 4)) == (1, 2, 3, 4)
    assert edg.decode(edg.encode((4, 3, 2, 1))) == (1, 2, 3, 4)
    x[edg.edge_index(1, 3)] = 1                      # extra edge -> degrees 3
    assert edg.decode(x) is None
    assert edg.energy(x) > tour_cost(D, (1, 2, 3, 4)) + edg.A - 1e-9   # at least one penalty A


def test_tied_optima_are_all_ground_states():
    f = fixture("hex6_ties")
    D = f.dist_matrix()
    C_opt, opt = optimal_tours(D)
    assert len(opt) == 6, opt                        # remove any one hexagon edge
    enc = build_position(D)                          # 36 bits: too many to enumerate; check the tours directly
    for t in opt:
        assert abs(enc.energy(enc.encode(t)) - C_opt) < 1e-7
    # Held-Karp agrees
    assert abs(tsp_exact(D, list(range(1, 7)))[0] - C_opt) < 1e-7


# --------------------------------------------------------------------------- #
# the n = 5 cycle-cover failure of the degree-only edge model
# --------------------------------------------------------------------------- #
def test_edge_model_fails_on_two_triangles_and_is_refused_for_n5():
    r = edge_relaxation_failure()
    assert r["is_failure_demonstrated"]
    assert r["ground_state_degree_two"] and not r["ground_state_decodes_to_tour"]
    assert r["ground_state_components"] == [[0, 1, 2], [3, 4, 5]]
    assert abs(r["ground_energy"] - 6.0) < 1e-6                   # two unit triangles
    assert r["ground_energy"] < r["best_tour_energy"] - 100       # any real tour crosses the gap twice
    assert r["production_builder_refuses_n5"]
    D = dist_matrix(two_triangles_points())
    _raises(ValueError, build_edge, D)
    # the position encoding on the same instance has no such failure: its optimum IS the tour
    pos = build_position(D)
    C_opt, opt = optimal_tours(D)
    assert abs(pos.energy(pos.encode(opt[0])) - C_opt) < 1e-7
    f5 = fixture("tri5_subtour")
    assert abs(f5.dist_matrix() - D).max() < 1e-12


def test_fixtures_keep_ids_separate_and_depot_at_zero():
    for name in FIXTURES:
        f = fixture(name)
        D = f.dist_matrix()
        assert D.shape == (f.n + 1, f.n + 1) and len(f.customer_ids) == f.n
        assert len(set(f.customer_ids)) == f.n and all(i >= 100 for i in f.customer_ids)   # not 0..n-1
        assert np.allclose(D[0], np.r_[0.0, np.linalg.norm(np.asarray(f.customers) - np.asarray(f.depot), axis=1)])
        assert f.local_to_id(1) == f.customer_ids[0] and f.id_to_local(f.customer_ids[-1]) == f.n
    f4 = fixture("quad4")
    d = f4.dist_matrix()[0, 1:]
    assert d.max() - d.min() > 10            # unequal depot distances
    assert f4.n == 4 and len(optimal_tours(f4.dist_matrix())[1]) == 1


def test_small_n_rejected():
    D = dist_matrix(random_points(1, 0))
    _raises(ValueError, build_position, D)
    _raises(ValueError, build_edge, D)
