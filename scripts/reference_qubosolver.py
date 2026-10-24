#!/usr/bin/env python
"""§9.3 reference values the way the brief describes them: qubo-solver + a stated max solve time.

    python scripts/reference_qubosolver.py instances/ --time-limit 60 --restarts 10 --seed 0

For every instance, C* is the **best known** CVRP cost found by, each under the same wall-clock budget:

  (2) qubo-solver's classical solver on our penalized QUBO (`onehot-ub`): CPLEX when the `cplex`
      package is importable (brief level 2), otherwise qubo-solver's simulated annealing called with
      *stated* parameters (T0 = max penalty weight, Tf = 1e-3 T0, geometric cooling set by qubo-solver
      from (T0, Tf, max_iter), `--restarts` seeded starts, `--time-limit` seconds for the batch).
      Every returned bitstring is decoded; only feasible decodes count (brief: "its optimum is a valid
      reference only if it decodes to a feasible solution"); the clustering is routed exactly.
  (3) our constrained simulated annealing (cvrp.reference, stated T0/Tf/alpha/max_iter, `--restarts`
      seeds, time limit `--time-limit / --restarts` per restart) on the CVRP itself.

C* = min of the two, recorded with `reference_status = "best_known"` and every solver setting the brief
asks for.  On tier S the brief also demands brute force *and* the solver and a check that they agree,
so the exact set-partition DP is run there and `agree` is recorded; when it agrees the S reference is a
`proven_optimum` (brief level 1 applies); on M/L no exhaustive search is run.
Needs the Windows venv (torch + qubo-solver).  Writes the `reference` block into the instance JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvrp.common import Instance, clustering_of, route_exactly, str_to_bits  # noqa: E402
from cvrp.exact import cvrp_cluster_costs, solve_cvrp  # noqa: E402
from cvrp.model import get_formulation  # noqa: E402
from cvrp.reference import COST_TOL, SA_DEFAULTS, sa_reference  # noqa: E402

FORMULATION = "onehot-ub"


def qubosolver_reference(inst: Instance, time_limit: float, restarts: int, seed: int,
                         max_iter: int, top_k: int) -> Dict[str, Any]:
    """Best feasible CVRP solution found by qubo-solver's classical solver on the penalized QUBO."""
    from cvrp.baseline import (_import_qubosolver, _qs_instance, _seed_everything, check_convention,
                               cplex_available)
    qubosolver = _import_qubosolver()
    f = get_formulation(FORMULATION, inst)
    Q = f.Q()
    qs_inst = _qs_instance(Q)
    seed_info = _seed_everything(seed)
    conv = check_convention(qs_inst, f, seed=seed)
    t0 = time.perf_counter()
    settings: Dict[str, Any] = {"package": "qubo-solver", "version": getattr(qubosolver, "__version__", None),
                                "qubo": f"{FORMULATION} penalized QUBO, {f.n_vars} variables, weights "
                                        + json.dumps({k: float(v) for k, v in f.weights.items()}),
                                "time_limit_s": float(time_limit), "seed_info": seed_info,
                                "convention_check": conv}
    if cplex_available():
        from qubosolver import ClassicalSolvingConfig, Solver, SolverConfig
        cfg = SolverConfig(config_name="cvrp-reference-cplex",
                           solving=ClassicalSolvingConfig(algorithm="cplex", time_limit=float(time_limit),
                                                          max_bitstrings=int(top_k)),
                           preprocessing=False, postprocessing=False)
        solution = Solver(qs_inst, cfg).solve()
        settings.update({"algorithm": "cplex", "cplex_maxtime_s": float(time_limit)})
    else:
        import torch
        from qubosolver.solving import simulated_annealing as qs_sa
        A_scale = float(max(abs(v) for v in f.weights.values()))
        T0, Tf = A_scale, max(1e-3, 1e-3 * A_scale)
        gen = torch.Generator().manual_seed(int(seed))
        solution = qs_sa.solve(qs_inst, starts=int(restarts), merge=True, top_k=int(top_k), max_iter=int(max_iter),
                               initial_temp=T0, final_temp=Tf, time_limit=float(time_limit), rng=gen)
        settings.update({"algorithm": "simulated_annealing (cplex not importable)",
                         "sa_initial_temp": T0, "sa_final_temp": Tf,
                         "sa_alpha": "geometric, (Tf/T0)^(1/max_iter), set by qubo-solver",
                         "max_iter": int(max_iter), "sa_time_limit_s": float(time_limit),
                         "restarts": int(restarts), "seeds": [int(seed)],
                         "note": "qubosolver.solving.simulated_annealing.solve, vectorized batch of "
                                 f"{restarts} starts from torch.Generator({seed}); one time budget for the batch"})
    wall = time.perf_counter() - t0
    cands = []
    best = None
    n_feasible = 0
    for c in solution:
        bits = str(c.string)
        a = f.decode(str_to_bits(bits))
        feas = f.feasible(a)
        if not feas:
            cands.append({"bits": bits, "qubo_cost": float(c.cost), "feasible": False})
            continue
        n_feasible += 1
        total, routes = route_exactly(inst, a)
        row = {"bits": bits, "qubo_cost": float(c.cost), "feasible": True, "model_objective": float(f.objective(a)),
               "cvrp_cost": float(total), "assignment": list(int(v) for v in a)}
        cands.append(row)
        if best is None or total < best["cvrp_cost"] - 1e-12:
            best = {**row, "routes": routes}
    return {"best": best, "candidates": sorted(cands, key=lambda r: r["qubo_cost"])[:top_k],
            "n_candidates": len(cands), "n_feasible_candidates": n_feasible, "wall_s": round(wall, 3),
            "settings": settings}


def build_reference(inst: Instance, time_limit: float, restarts: int, seed: int, max_iter: int, top_k: int,
                    brute_force_check: bool) -> Dict[str, Any]:
    qs = qubosolver_reference(inst, time_limit, restarts, seed, max_iter, top_k)
    # (3) constrained SA with the same total budget
    seeds = list(range(seed, seed + restarts))
    params = {k: SA_DEFAULTS[k] for k in ("sa_initial_temp", "sa_final_temp", "sa_alpha", "max_iter")}
    params["sa_time_limit"] = float(time_limit) / max(1, restarts)
    t1 = time.perf_counter()
    cost_table = cvrp_cluster_costs(inst)
    sa = sa_reference(inst, cost_table, restarts, seeds, **params)
    t_sa = time.perf_counter() - t1

    sources = []
    if qs["best"] is not None:
        sources.append(("qubo-solver", qs["best"]["cvrp_cost"], tuple(qs["best"]["assignment"])))
    sources.append(("constrained-sa", sa["cost"], tuple(sa["assignment"])))
    winner = min(sources, key=lambda s: s[1])
    C, a = float(winner[1]), winner[2]
    total, routes_d = route_exactly(inst, a)
    assert abs(total - C) <= COST_TOL
    routes = [routes_d[k][1] if k in routes_d else [] for k in range(inst.K)]
    lengths = [routes_d[k][0] if k in routes_d else 0.0 for k in range(inst.K)]
    # distinct best clusterings seen by either method (lower bound on the number of optima)
    seen = set()
    for r in qs["candidates"]:
        if r.get("feasible") and abs(r["cvrp_cost"] - C) <= COST_TOL:
            seen.add(clustering_of(tuple(r["assignment"])))
    if abs(sa["cost"] - C) <= COST_TOL:
        seen.add(clustering_of(tuple(sa["assignment"])))
    ref: Dict[str, Any] = {
        "cost": C,
        "best_assignment": list(int(v) for v in a),
        "best_clusters": [sorted(i for i in range(inst.N) if a[i] == k) for k in range(inst.K) if any(a[i] == k for i in range(inst.N))],
        "routes": routes,
        "route_lengths": lengths,
        "n_distinct_optima": max(1, len(seen)),
        "n_distinct_optima_note": "distinct best-known clusterings found by the two solvers (a lower bound)",
        "n_vehicles_used": sum(1 for r in routes if r),
        "reference_status": "best_known",
        "found_by": winner[0],
        "method": (f"brief §9.3 levels 2–3: qubo-solver {qs['settings']['algorithm'].split(' ')[0]} on the "
                   f"{FORMULATION} QUBO (max solve time {time_limit:g} s) and constrained simulated annealing "
                   f"({restarts} restarts, {time_limit:g} s total); C* = best known of the two"),
        "solver_settings": {
            "qubo_solver": qs["settings"],
            "sa": {**params, "moves_per_temp": inst.N, "restarts": restarts, "seeds": seeds,
                   "moves": SA_DEFAULTS["moves"], "start": SA_DEFAULTS["start"], "rng": "numpy.random.default_rng(seed)"},
            "cplex": qs["settings"].get("cplex_maxtime_s", "not available (package `cplex` not importable)"),
        },
        "qubosolver_cost": qs["best"]["cvrp_cost"] if qs["best"] else None,
        "qubosolver_model_objective": qs["best"]["model_objective"] if qs["best"] else None,
        "qubosolver_best_assignment": qs["best"]["assignment"] if qs["best"] else None,
        "qubosolver_n_candidates": qs["n_candidates"],
        "qubosolver_n_feasible_candidates": qs["n_feasible_candidates"],
        "qubosolver_candidates": qs["candidates"][:10],
        "sa_cost": sa["cost"],
        "sa_best_assignment": sa["assignment"],
        "sa_matches_optimum": bool(abs(sa["cost"] - C) <= COST_TOL),
        "sa_restarts_hitting_best": sa["n_restarts_hitting_best"],
        "sa_runs": sa["runs"],
        "timing_s": {"qubo_solver": qs["wall_s"], "sa_total": round(t_sa, 4)},
    }
    if brute_force_check:
        t2 = time.perf_counter()
        ex = solve_cvrp(inst)
        ref["brute_force_check"] = {
            "cost": ex["cost"], "n_distinct_optima": ex["n_optimal"],
            "method": "exhaustive set-partition DP x exact TSP (tier S only, brief §9.3: 'run both ... and check that they agree')",
            "time_s": round(time.perf_counter() - t2, 4),
        }
        ref["agree"] = bool(abs(ex["cost"] - C) <= COST_TOL)
        ref["brute_force_naive_cost"] = ex["cost"]
        if ref["agree"]:
            # brief §9.3 level 1 applies: the exhaustive search certifies the value the solvers found
            ref["reference_status"] = "proven_optimum"
            ref["n_distinct_optima"] = int(ex["n_optimal"])
            ref["n_distinct_optima_note"] = "exhaustive set-partition DP (vehicle relabellings identified)"
            ref["method"] += "; tier S: exhaustive brute force agrees -> proven optimum"
    return ref


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("paths", nargs="+")
    p.add_argument("--time-limit", type=float, default=60.0, help="max solve time per instance per method [s]")
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=2_000_000, help="qubo-solver SA proposals (time limit usually binds)")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--brute-force-check", choices=["S", "all", "none"], default="S",
                   help="which tiers also get the exhaustive DP agreement check (brief: tier S)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    files: List[Path] = []
    for s in a.paths:
        q = Path(s)
        files += sorted(q.glob("*.json")) if q.is_dir() else [q]
    print(f"{'instance':10s} {'C* (best known)':>16s} {'qubo-solver':>12s} {'constr. SA':>11s} {'found by':>14s} "
          f"{'agree(S)':>8s} {'t_qs [s]':>8s}")
    bad = 0
    for fpath in files:
        inst = Instance.load(fpath)
        bf = a.brute_force_check == "all" or (a.brute_force_check == "S" and inst.tier == "S")
        ref = build_reference(inst, a.time_limit, a.restarts, a.seed, a.max_iter, a.top_k, bf)
        inst.reference = ref
        if not a.dry_run:
            inst.save(fpath)
        qs_c = ref["qubosolver_cost"]
        print(f"{inst.name:10s} {ref['cost']:16.4f} {(f'{qs_c:.4f}' if qs_c is not None else 'infeasible'):>12s} "
              f"{ref['sa_cost']:11.4f} {ref['found_by']:>14s} {str(ref.get('agree', '-')):>8s} "
              f"{ref['timing_s']['qubo_solver']:8.1f}")
        if ref.get("agree") is False:
            bad += 1
    if bad:
        print(f"\nWARNING: {bad} tier-S instance(s) where the solvers did not reach the brute-force optimum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
