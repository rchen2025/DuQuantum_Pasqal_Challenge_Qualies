"""End-to-end demo:  python -m qopt.demo [--backend auto|qutip|emu-sv|emu-mps|mock]
                                       [--device WeightedAnalogDevice] [--shots 500]
                                       [--problem knapsack|setpacking|mis] [--no-baseline]

1. builds a toy constrained problem, converts to QUBO, brute-force verifies it
2. embeds it (register + DMM weights), prints the embedding report
3. solves classically (ground truth)
4. runs Pasqal's qubo-solver on the same QUBO (baseline)
5. runs our adiabatic pipeline on the chosen emulator
6. prints the comparison table and saves out/report.png
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import classical, model
from .embed import DeviceLimits, embed_qubo
from .report import comparison_table, make_report, row_from_score, score_vs_shots
from .run import run_pipeline
from .schedule import Schedule


def toy_problem(name: str) -> model.Problem:
    if name == "knapsack":
        # 4 items, capacity 7 -> optimum value 9 (items 1,2: weights 3+4)
        return model.knapsack(values=[3, 4, 5, 6], weights=[2, 3, 4, 5], capacity=7)
    if name == "setpacking":
        return model.set_packing([[0, 1], [1, 2], [2, 3], [0, 3], [4], [3, 4]], n_elements=5, weights=[2, 2, 2, 2, 1, 3])
    if name == "mis":
        return model.max_independent_set([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)], 6)
    raise ValueError(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--device", default=None, help="pulser device name (default: $QOPT_DEVICE or WeightedAnalogDevice)")
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--problem", default="knapsack")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--duration", type=int, default=None)
    ap.add_argument("--out", default="out/report.png")
    args = ap.parse_args(argv)

    t_all = time.time()
    limits = DeviceLimits.from_device_name(args.device) if args.device else DeviceLimits.default()
    print(f"== device {limits.name}: C6={limits.c6:.4g}, d_min={limits.min_atom_distance}, R_max={limits.max_radial_distance}, "
          f"Omega_max={limits.max_amp:.3g}, |delta|_max={limits.max_abs_detuning:.3g}, DMM={limits.has_dmm}, T_max={limits.max_sequence_duration} ns")

    # 1. model + verify
    prob = toy_problem(args.problem)
    build = model.auto_penalty(prob)
    qubo = build.qubo
    ver = model.brute_force_verify(prob, build)
    print(f"== problem '{args.problem}': {len(prob.variables)} vars, {len(prob.constraints)} constraints -> QUBO n={qubo.n} "
          f"(slack vars {qubo.n - len(prob.variables)}), penalties {build.penalties}")
    print(f"   brute-force verification: {ver.message}; constrained optimum = {ver.constrained_optimum}")
    assert ver.ok, "QUBO does not reproduce the constrained optimum -- fix the model before going further"

    # 2. embed
    t = time.time()
    emb = embed_qubo(qubo, limits)
    print(f"== embedding ({time.time() - t:.1f}s)\n" + "\n".join("   " + l for l in emb.report.summary().splitlines()))

    # 3. classical
    t = time.time()
    cl = classical.solve_best(qubo.Q)
    t_cl = time.time() - t
    cx = qubo.project(cl.x)
    c_obj = prob.objective(prob.assignment(cx))
    print(f"== classical ({cl.method}, {t_cl:.2f}s): QUBO energy {cl.energy + qubo.offset:.4g}, objective {c_obj}, x={cx}")

    rows = {"classical": {"objective": c_obj, "feasible": prob.is_feasible(prob.assignment(cx)), "approx_ratio": 1.0,
                          "p_opt": "-", "time_s": t_cl, "note": cl.method}}
    curves = {}

    # 4. baseline
    if not args.no_baseline:
        try:
            from .baseline import run_qubo_solver
            bt = None if args.backend in ("auto", "mock") else args.backend
            base = run_qubo_solver(prob, qubo, shots=args.shots, mode="functional", backend_type=bt, classical=cl)
            print(f"== qubo-solver (functional, {base.elapsed_s:.1f}s) embedding rel err={base.embedding_rel_error}\n{base.score.table()}")
            rows["qubo-solver"] = row_from_score(base.score, base.elapsed_s, f"blade+local_energy_scale, emb err {base.embedding_rel_error:.2f}" if base.embedding_rel_error is not None else "")
            curves["qubo-solver"] = score_vs_shots(base.counts, prob, qubo, base.score and (lambda bs: qubo.project(np.array([int(c) for c in bs]))), classical=cl)
        except ImportError as e:
            print(f"== qubo-solver baseline skipped (not installed: {e})")
            rows["qubo-solver"] = {"note": "not installed"}
        except Exception as e:  # keep the demo alive; the comparison table shows the failure
            print(f"== qubo-solver baseline FAILED: {type(e).__name__}: {e}")
            rows["qubo-solver"] = {"note": f"failed: {type(e).__name__}"}

    # 5. ours
    sched = Schedule.default_for(emb, duration=args.duration)
    print(f"== schedule: T={sched.duration} ns, Omega_max={sched.omega_max:.3g}, delta {sched.delta_init:.3g} -> {sched.delta_final:.3g}, "
          f"dmm {sched.delta_dmm:.3g}; hints {sched.adiabaticity_hints(emb)}")
    try:
        res = run_pipeline(prob, qubo, emb, sched, backend=args.backend, shots=args.shots, classical=cl)
        print(f"== ours ({res.backend}, {res.elapsed_s:.1f}s)\n{res.score.table()}")
        rows["ours"] = row_from_score(res.score, res.elapsed_s, f"{res.backend}, emb err {emb.report.rel_error_offdiag:.2f}")
        curves["ours"] = score_vs_shots(res.counts, prob, qubo, emb.decode, classical=cl)
        counts, score = res.counts, res.score
    except ImportError as e:
        print(f"== our pipeline could not run on backend '{args.backend}': {e}\n   (try --backend mock to exercise the pipeline without an emulator)")
        rows["ours"] = {"note": f"backend missing: {e}"}
        counts, score = None, None

    # 6. table + plot
    table = comparison_table(rows)
    print("\n" + table)
    path = make_report(emb, sched, counts, score, curves or None, path=args.out, table=table)
    print(f"\nreport saved to {path}   (total {time.time() - t_all:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
