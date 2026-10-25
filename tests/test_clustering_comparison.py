"""Independent, small-instance checks of the classical objective comparison."""
import itertools
import math
from pathlib import Path

import numpy as np

from cvrp.clustering_comparison import MODELS, cluster_cost_table, compare_model, mst_cost
from cvrp.common import Instance, clustering_of, route_exactly
from cvrp.exact import exact_partition
from cvrp.instances import generate


def _triangle():
    return Instance(name="triangle", tier="test", seed=0, N=3, K=2, Q=3,
                    depot=(0., 0.), locations=[(1., 0.), (4., 0.), (1., 4.)],
                    demands=[1, 1, 1], certificate=(0, 0, 1))


def test_known_cluster_costs_and_singletons():
    inst = _triangle()
    pairwise = cluster_cost_table(inst, "pairwise")
    medoid = cluster_cost_table(inst, "medoid")
    depot = cluster_cost_table(inst, "depot-mst")
    assert pairwise[7] == 12.0  # edges 3, 4, 5
    assert medoid[7] == 7.0
    assert depot[7] == 9.0       # MST 7 + two depot legs of length 1
    for i in range(inst.N):
        assert pairwise[1 << i] == medoid[1 << i] == 0.0
        assert depot[1 << i] == 2.0 * inst.depot_dist()[i]
    assert math.isinf(pairwise[0])


def test_mst_handles_zero_edges_and_one_vertex():
    assert mst_cost(np.zeros((1, 1))) == 0.0
    D = np.array([[0, 0, 2], [0, 0, 2], [2, 2, 0]], dtype=float)
    assert mst_cost(D) == 2.0


def test_capacity_is_enforced_for_all_objectives():
    inst = _triangle()
    inst.Q = 2
    for model in MODELS:
        costs = cluster_cost_table(inst, model)
        assert math.isinf(costs[7])
        result = compare_model(inst, model)
        assert result["status"] == "complete"
        assert inst.is_valid_assignment(result["assignment"])
        assert result["n_vehicles_used"] <= inst.K
        cost, _ = route_exactly(inst, result["assignment"])
        assert abs(cost - result["routed_cost"]) < 1e-9
        assert result["cvrp_gap_percent"] is None  # no reference supplied


def test_tiny_exact_solutions_and_optimum_counts_match_enumeration():
    for seed in (0, 3, 4):
        inst = generate("S", seed)
        for model in MODELS:
            costs = cluster_cost_table(inst, model)
            seen = {}
            for assignment in itertools.product(range(inst.K), repeat=inst.N):
                if not inst.is_valid_assignment(assignment):
                    continue
                clusters = clustering_of(assignment)
                seen[clusters] = sum(costs[sum(1 << i for i in group)] for group in clusters)
            best = min(seen.values())
            n_best = sum(abs(value - best) <= 1e-9 for value in seen.values())
            result = compare_model(inst, model)
            assert abs(result["model_optimum"] - best) < 1e-9
            assert result["n_model_optimal_partitions"] == n_best


def test_tie_break_deterministic_and_reference_independent():
    inst = Instance(name="square", tier="test", seed=0, N=4, K=2, Q=2,
                    depot=(0., 0.), locations=[(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)],
                    demands=[1, 1, 1, 1], certificate=(0, 0, 1, 1))
    for model in MODELS:
        first = compare_model(inst, model)
        assert first["n_model_optimal_partitions"] == 2
        inst.reference = {"cost": 123., "assignment": [0, 1, 1, 0], "reference_status": "best_known"}
        second = compare_model(inst, model)
        assert first["clusters"] == second["clusters"]
        assert first["routed_cost"] == second["routed_cost"]
        assert second["reference_status"] == "best_known"


def test_at_most_k_can_use_fewer_groups_and_infeasible_is_not_scored():
    inst = _triangle()
    inst.locations = [(1., 0.), (2., 0.), (3., 0.)]
    result = compare_model(inst, "depot-mst")
    assert result["n_vehicles_used"] == 1
    inst.Q = 1
    result = compare_model(inst, "pairwise")
    assert result["status"] == "infeasible"
    assert result["model_optimum"] is None
    assert result["routed_cost"] is None
    assert result["cvrp_gap_percent"] is None


def test_subprocess_timeout_never_reports_optimum():
    from scripts.compare_clustering_models import run_isolated
    path = Path(__file__).resolve().parents[1] / "instances/S_seed00.json"
    result = run_isolated(path, "pairwise", timeout=0.00001, threads=1)
    assert result["status"] == "timeout"
    assert result["model_reference_status"] == "not_computed"
    assert result["model_optimum"] is None
    assert result["routed_cost"] is None
    assert result["cvrp_gap_percent"] is None
