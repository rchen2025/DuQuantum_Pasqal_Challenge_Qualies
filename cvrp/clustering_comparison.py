"""Classical, capacity-exact comparison of three additive clustering objectives.

These objectives are diagnostic alternatives, not implemented Rydberg QUBOs.
All use the same at-most-K set-partition solver, independent of route cost.
"""
from __future__ import annotations

import math
import time

import numpy as np

from cvrp.common import Instance, assignment_of, gap_percent, route_exactly
from cvrp.exact import TIE_TOL, exact_partition, subset_sums

MODELS = ("pairwise", "medoid", "depot-mst")
MODEL_DEFINITIONS = {
    "pairwise": "sum_{i<j in S} d_ij",
    "medoid": "min_{m in S} sum_{i in S} d_im",
    "depot-mst": "MST(S) + 2 min_{i in S} d_0i",
}
TIE_RULE = (
    "existing exact_partition: ascending submasks, first minimum by numpy.argmin; "
    "clusters peeled in increasing smallest-customer order; route cost never breaks ties"
)


def mst_cost(distances: np.ndarray) -> float:
    """Deterministic Prim MST on a complete symmetric distance matrix."""
    n = len(distances)
    if n <= 1:
        return 0.0
    selected = np.zeros(n, dtype=bool)
    nearest = np.full(n, math.inf)
    nearest[0] = 0.0
    total = 0.0
    for _ in range(n):
        node = int(np.argmin(nearest))
        total += float(nearest[node])
        selected[node] = True
        nearest = np.minimum(nearest, distances[node])
        nearest[selected] = math.inf
    return total


def cluster_cost_table(inst: Instance, model: str) -> np.ndarray:
    """Costs for every subset; empty/overloaded groups are explicitly forbidden."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
    D = inst.dist_matrix()
    Dc, d0 = D[1:, 1:], D[0, 1:]
    loads = subset_sums(inst.demands, inst.N)
    costs = np.full(1 << inst.N, math.inf)
    for mask in range(1, 1 << inst.N):
        if loads[mask] > inst.Q:
            continue
        members = [i for i in range(inst.N) if mask & (1 << i)]
        within = Dc[np.ix_(members, members)]
        if model == "pairwise":
            value = float(np.triu(within, 1).sum())
        elif model == "medoid":
            value = float(within.sum(axis=0).min())
        else:
            value = mst_cost(within) + 2.0 * float(d0[members].min())
        costs[mask] = value
    return costs


def compare_model(inst: Instance, model: str) -> dict:
    """Solve one alternative exactly, then route the chosen partition exactly.

    The stored reference is used only to calculate the reported CVRP gap;
    neither reference assignments nor feasibility certificates guide the solve.
    """
    start = time.perf_counter()
    costs = cluster_cost_table(inst, model)
    costs_done = time.perf_counter()
    optimum, clusters, n_optimal = exact_partition(inst.N, inst.K, costs)
    solve_done = time.perf_counter()
    reference = inst.reference
    reference_cost = reference.get("cost")
    record = {
        "instance": inst.name, "tier": inst.tier, "seed": inst.seed,
        "N": inst.N, "K": inst.K, "Q": inst.Q,
        "model": model, "model_definition": MODEL_DEFINITIONS[model],
        "experiment_kind": "classical_model_selection",
        "status": "complete" if math.isfinite(optimum) else "infeasible",
        "model_reference_status": "proven_optimum" if math.isfinite(optimum) else "proven_infeasible",
        "model_optimum": optimum if math.isfinite(optimum) else None,
        "n_model_optimal_partitions": n_optimal,
        "tie_tolerance": TIE_TOL, "tie_rule": TIE_RULE,
        "reference_cost": reference_cost,
        "reference_status": reference.get("reference_status", "unavailable"),
        "reference_method": reference.get("method"),
        "n_feasible_clusters": int(np.isfinite(costs).sum()),
        "capacity_enforcement": "exclude all subsets with load > Q",
        "fleet_constraint": "at most K nonempty groups",
        "clusters": [sorted(group) for group in clusters],
        "assignment": None, "loads": None, "n_vehicles_used": None,
        "routed_cost": None, "cvrp_gap_percent": None, "routes": None,
        "time_cluster_costs_s": costs_done - start,
        "time_partition_s": solve_done - costs_done,
    }
    if math.isfinite(optimum):
        assignment = assignment_of(clusters, inst.N)
        if not inst.is_valid_assignment(assignment) or len(clusters) > inst.K:
            raise AssertionError("Exact partition result violates capacity or fleet constraints")
        routed_cost, routes = route_exactly(inst, assignment)
        record.update({
            "assignment": list(assignment), "loads": inst.loads(assignment),
            "n_vehicles_used": len(clusters), "routed_cost": routed_cost,
            "cvrp_gap_percent": (
                gap_percent(routed_cost, float(reference_cost))
                if reference_cost is not None and float(reference_cost) > 0 else None
            ),
            "routes": {str(k): {"cost": value[0], "customers": value[1]}
                       for k, value in routes.items()},
        })
    record["time_routing_s"] = time.perf_counter() - solve_done
    record["time_total_s"] = time.perf_counter() - start
    return record
