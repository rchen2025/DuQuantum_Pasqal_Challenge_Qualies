"""Reference values C* for every instance (brief §9.3) -- deliverable D3.

For each instance JSON this module fills ``inst.reference`` and re-saves the
file.  The brief asks for the first applicable method out of

1. brute force  -> proven optimum (+ the number of distinct optimal solutions),
2. CPLEX,
3. simulated annealing with stated parameters.

All three tiers are small enough for method 1: ``cvrp.exact.solve_cvrp``
enumerates *every* capacity-feasible clustering through a set-partition DP
(with exact Held-Karp tours per cluster), which is an exhaustive search in the
space of clusterings, hence a proven optimum.  Method 2 is recorded as
"not available" (no CPLEX licence in the hackathon environment).  Method 3,
simulated annealing over assignments, is run anyway on every instance as an
independent cross-check (``sa_cost``, ``sa_matches_optimum``).  On tier S the
brief asks for brute force *and* a second method that agree: in addition to SA
we run a completely naive enumeration of all K^N assignments
(``brute_force_naive_cost``, ``agree``).

Vehicles are identical, so "distinct optimal solutions" means distinct
*clusterings* (sets of customer groups): two assignments that differ only by
relabeling the vehicles are the same solution (``common.clustering_of``).  The
DP counts each clustering exactly once (see ``cvrp.exact``).

Usage::

    python -m cvrp.reference instances/            # fills reference blocks, prints a table
    python -m cvrp.reference instances/S_seed00.json --sa-restarts 20
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cvrp.common import Assignment, Instance, clustering_of, route_exactly
from cvrp.exact import cvrp_cluster_costs, set_to_mask, solve_cvrp, tsp_tour_table

METHOD = ("brute_force: exhaustive set-partition DP over all capacity-feasible "
          "clusterings x exact Held-Karp TSP per cluster")
COST_TOL = 1e-6   # two CVRP costs within this are "the same value"

# Simulated-annealing defaults (all recorded in the JSON under solver_settings.sa).
SA_DEFAULTS: Dict[str, Any] = {
    "sa_initial_temp": 20.0,      # distance units; a typical bad move costs O(10) on a side-100 square
    "sa_final_temp": 0.01,
    "sa_alpha": 0.995,            # geometric cooling T <- alpha * T after every sweep
    "moves_per_temp": "N",        # one sweep = N proposed moves
    "max_iter": 50_000,           # cap on proposed moves per restart
    "sa_time_limit": 10.0,        # seconds per restart
    "restarts": 10,
    "seeds": list(range(10)),
    "moves": "reassign one customer (p=0.5) or swap two customers of different vehicles (p=0.5); "
             "moves that overload a vehicle are rejected",
    "start": "the feasibility certificate, perturbed by N random feasible moves",
}


# --------------------------------------------------------------------------- #
# Simulated annealing over assignments
# --------------------------------------------------------------------------- #
def _vehicle_masks(a: Sequence[int], K: int) -> List[int]:
    masks = [0] * K
    for i, k in enumerate(a):
        masks[k] |= 1 << i
    return masks


def simulated_annealing(inst: Instance, seed: int, cost_table: Optional[np.ndarray] = None,
                        sa_initial_temp: float = SA_DEFAULTS["sa_initial_temp"],
                        sa_final_temp: float = SA_DEFAULTS["sa_final_temp"],
                        sa_alpha: float = SA_DEFAULTS["sa_alpha"],
                        max_iter: int = SA_DEFAULTS["max_iter"],
                        sa_time_limit: float = SA_DEFAULTS["sa_time_limit"],
                        start: Optional[Sequence[int]] = None) -> Tuple[float, Assignment, Dict[str, Any]]:
    """One SA run; returns (best cost, best assignment, run statistics).

    State: a capacity-feasible assignment.  Energy: the exact CVRP cost of the
    clustering, read from ``cost_table[mask]`` (the exact tour length of each
    vehicle's customer set, ``inf`` if it exceeds Q -- precomputed once per
    instance by ``cvrp.exact``).  Moves: reassign one customer to another
    vehicle, or swap two customers on different vehicles; a move that makes a
    vehicle overloaded is rejected outright, so the walk stays feasible.
    Metropolis acceptance ``exp(-dE / T)``, geometric cooling after each sweep
    of N moves, stop at ``sa_final_temp``, ``max_iter`` moves or the time limit.
    """
    rng = np.random.default_rng(seed)
    N, K = inst.N, inst.K
    if cost_table is None:
        cost_table = cvrp_cluster_costs(inst)

    def mask_cost(m: int) -> float:
        return 0.0 if m == 0 else float(cost_table[m])

    a = list(start if start is not None else inst.certificate)
    masks = _vehicle_masks(a, K)
    assert all(math.isfinite(mask_cost(m)) for m in masks), "SA must start from a feasible assignment"
    if start is None:
        # Scramble the start so different seeds explore different regions.
        for _ in range(N):
            i, k = int(rng.integers(N)), int(rng.integers(K))
            if k != a[i] and math.isfinite(mask_cost(masks[k] | (1 << i))):
                masks[a[i]] ^= 1 << i
                masks[k] |= 1 << i
                a[i] = k
    E = sum(mask_cost(m) for m in masks)
    best_E, best_a = E, tuple(a)
    T = sa_initial_temp
    it = accepted = 0
    t0 = time.perf_counter()
    while T >= sa_final_temp and it < max_iter and time.perf_counter() - t0 < sa_time_limit:
        for _ in range(N):
            it += 1
            if rng.random() < 0.5:
                # --- reassign customer i from vehicle k1 to vehicle k2 -------- #
                i = int(rng.integers(N))
                k1, k2 = a[i], int(rng.integers(K))
                if k1 == k2:
                    continue
                m1, m2 = masks[k1] ^ (1 << i), masks[k2] | (1 << i)
                new_pair = mask_cost(m1) + mask_cost(m2)
                if not math.isfinite(new_pair):
                    continue
                dE = new_pair - mask_cost(masks[k1]) - mask_cost(masks[k2])
                if dE <= 0 or rng.random() < math.exp(-dE / T):
                    masks[k1], masks[k2], a[i] = m1, m2, k2
                    E += dE
                    accepted += 1
            else:
                # --- swap customers i and j that sit on different vehicles ---- #
                i, j = int(rng.integers(N)), int(rng.integers(N))
                k1, k2 = a[i], a[j]
                if k1 == k2:
                    continue
                m1 = (masks[k1] ^ (1 << i)) | (1 << j)
                m2 = (masks[k2] ^ (1 << j)) | (1 << i)
                new_pair = mask_cost(m1) + mask_cost(m2)
                if not math.isfinite(new_pair):
                    continue
                dE = new_pair - mask_cost(masks[k1]) - mask_cost(masks[k2])
                if dE <= 0 or rng.random() < math.exp(-dE / T):
                    masks[k1], masks[k2], a[i], a[j] = m1, m2, k2, k1
                    E += dE
                    accepted += 1
            if E < best_E - 1e-12:
                best_E, best_a = E, tuple(a)
        T *= sa_alpha
    stats = {"seed": int(seed), "iterations": it, "accepted": accepted, "final_temp": round(T, 6),
             "time_s": round(time.perf_counter() - t0, 4)}
    return best_E, best_a, stats


def sa_reference(inst: Instance, cost_table: np.ndarray, restarts: int, seeds: Sequence[int],
                 **params: Any) -> Dict[str, Any]:
    """Run SA ``restarts`` times (one seed each) and keep the best."""
    best_E, best_a, runs = math.inf, None, []
    for s in list(seeds)[:restarts]:
        E, a, st = simulated_annealing(inst, s, cost_table, **params)
        runs.append({**st, "cost": E})
        if E < best_E:
            best_E, best_a = E, a
    return {"cost": best_E, "assignment": list(best_a), "runs": runs,
            "n_restarts_hitting_best": sum(1 for r in runs if abs(r["cost"] - best_E) <= COST_TOL)}


# --------------------------------------------------------------------------- #
# Naive brute force (tier S cross-check): all K^N assignments
# --------------------------------------------------------------------------- #
def brute_force_naive(inst: Instance, max_assignments: int = 2_000_000) -> Dict[str, Any]:
    """Enumerate every one of the K^N assignments and route each exactly.

    Deliberately independent of ``cvrp.exact``: it only uses
    ``common.route_exactly`` (Held-Karp per vehicle) and ``clustering_of`` to
    count distinct optimal clusterings.  Only sensible for tier S (2^4 = 16
    assignments); M would be 3^10 = 59049 (still fine), L is 4^15 ~ 10^9 (no).
    """
    n_total = inst.K ** inst.N
    if n_total > max_assignments:
        raise ValueError(f"K^N = {n_total} assignments is too many for the naive brute force")
    best, optimal_clusterings, n_feasible = math.inf, set(), 0
    best_a = None
    for a in itertools.product(range(inst.K), repeat=inst.N):
        if not inst.capacity_ok(a):
            continue
        n_feasible += 1
        c, _ = route_exactly(inst, a)
        if c < best - COST_TOL:
            best, best_a, optimal_clusterings = c, a, {clustering_of(a)}
        elif abs(c - best) <= COST_TOL:
            optimal_clusterings.add(clustering_of(a))
    return {"cost": best, "assignment": list(best_a) if best_a else None,
            "n_assignments": n_total, "n_feasible_assignments": n_feasible,
            "n_distinct_optima": len(optimal_clusterings)}


# --------------------------------------------------------------------------- #
# Reference block
# --------------------------------------------------------------------------- #
def compute_reference(inst: Instance, sa_restarts: int = SA_DEFAULTS["restarts"],
                      sa_seed: int = 0, naive_brute_force: Optional[bool] = None,
                      **sa_params: Any) -> Dict[str, Any]:
    """Compute the full reference block for ``inst`` (does not modify it).

    ``sa_seed`` is the first of ``sa_restarts`` consecutive SA seeds.
    ``naive_brute_force`` defaults to True on tier S only.
    """
    t0 = time.perf_counter()
    exact = solve_cvrp(inst)
    t_exact = time.perf_counter() - t0

    cost_table = cvrp_cluster_costs(inst)
    seeds = list(range(sa_seed, sa_seed + sa_restarts))
    params = {k: sa_params.get(k, SA_DEFAULTS[k]) for k in
              ("sa_initial_temp", "sa_final_temp", "sa_alpha", "max_iter", "sa_time_limit")}
    t1 = time.perf_counter()
    sa = sa_reference(inst, cost_table, sa_restarts, seeds, **params)
    t_sa = time.perf_counter() - t1
    sa_matches = abs(sa["cost"] - exact["cost"]) <= COST_TOL

    ref: Dict[str, Any] = {
        "cost": exact["cost"],
        "best_assignment": list(exact["assignment"]),
        "best_clusters": exact["clusters"],
        "routes": exact["routes"],
        "route_lengths": exact["route_lengths"],
        "n_distinct_optima": exact["n_optimal"],
        "n_vehicles_used": sum(1 for r in exact["routes"] if r),
        "n_feasible_clusters": exact["n_feasible_clusters"],
        "reference_status": "proven_optimum",
        "method": METHOD,
        "solver_settings": {
            "brute_force": {
                "algorithm": "set-partition DP f[k][S] = min_R cost[R] + f[k-1][S\\R], R containing "
                             "lowest(S); cost[R] = Held-Karp tour if load(R) <= Q else inf",
                "tie_tolerance": 1e-9,
                "counts": "distinct optimal clusterings (vehicle relabelings identified)",
            },
            "cplex": "not available",
            "sa": {**params, "moves_per_temp": inst.N, "restarts": sa_restarts, "seeds": seeds,
                   "moves": SA_DEFAULTS["moves"], "start": SA_DEFAULTS["start"],
                   "rng": "numpy.random.default_rng(seed)"},
        },
        "sa_cost": sa["cost"],
        "sa_best_assignment": sa["assignment"],
        "sa_matches_optimum": bool(sa_matches),
        "sa_restarts_hitting_best": sa["n_restarts_hitting_best"],
        "sa_runs": sa["runs"],
        "timing_s": {"exact": round(t_exact, 4), "tsp_table": round(exact["time_tsp_table_s"], 4),
                     "partition_dp": round(exact["time_partition_dp_s"], 4), "sa_total": round(t_sa, 4)},
    }
    if naive_brute_force is None:
        naive_brute_force = inst.tier == "S"
    if naive_brute_force:
        t2 = time.perf_counter()
        bf = brute_force_naive(inst)
        ref["brute_force_naive_cost"] = bf["cost"]
        ref["brute_force_naive_n_distinct_optima"] = bf["n_distinct_optima"]
        ref["brute_force_naive_n_feasible_assignments"] = bf["n_feasible_assignments"]
        ref["agree"] = bool(abs(bf["cost"] - exact["cost"]) <= COST_TOL
                            and bf["n_distinct_optima"] == exact["n_optimal"])
        ref["timing_s"]["naive_brute_force"] = round(time.perf_counter() - t2, 4)
    return ref


def summary_row(inst: Instance) -> str:
    r = inst.reference
    sa = "yes" if r.get("sa_matches_optimum") else "NO"
    bf = "" if "agree" not in r else ("  naive=agree" if r["agree"] else "  naive=DISAGREE")
    t = r["timing_s"]
    return (f"{inst.tier:>4} {inst.seed:>4} {inst.rho:6.3f} {inst.dmax_over_Q:8.3f} {r['cost']:10.4f} "
            f"{r['n_distinct_optima']:>8} {sa:>8} {t['exact']:8.2f} {t['sa_total']:7.2f}{bf}")


SUMMARY_HEADER = (f"{'tier':>4} {'seed':>4} {'rho':>6} {'dmax/Q':>8} {'C*':>10} {'n_optima':>8} "
                  f"{'SA=opt':>8} {'t_exact':>8} {'t_SA':>7}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fill the reference block of CVRP instance files.")
    p.add_argument("paths", nargs="+", help="JSON files or folders containing them")
    p.add_argument("--sa-restarts", type=int, default=SA_DEFAULTS["restarts"])
    p.add_argument("--sa-seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="compute but do not write the JSON files")
    args = p.parse_args(argv)
    files: List[Path] = []
    for s in args.paths:
        q = Path(s)
        files += sorted(q.glob("*.json")) if q.is_dir() else [q]
    if not files:
        print("no instance files found", file=sys.stderr)
        return 1
    print(SUMMARY_HEADER)
    bad = 0
    for f in files:
        inst = Instance.load(f)
        inst.reference = compute_reference(inst, sa_restarts=args.sa_restarts, sa_seed=args.sa_seed)
        if not args.dry_run:
            inst.save(f)
        print(summary_row(inst))
        if not inst.reference["sa_matches_optimum"] or inst.reference.get("agree") is False:
            bad += 1
    if bad:
        print(f"\nWARNING: {bad} instance(s) where the cross-check did not reproduce the optimum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
