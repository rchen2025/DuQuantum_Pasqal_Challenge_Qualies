"""Tests for cvrp.model (registry, brute-force checker, model optimum) and the
throwaway textbook formulation (cvrp/formulations/textbook.py).

Plain ``def test_x()`` functions, no fixtures, so ``scripts/run_tests.py model``
runs them without pytest.  The legacy qopt tests from the generic toolkit are
kept at the bottom behind a tiny pytest shim.

Tiny instances are built directly from ``cvrp.common.Instance`` (no dependency
on cvrp.instances / cvrp.exact, which are written in parallel).
"""
from __future__ import annotations

import itertools
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from cvrp import common
from cvrp.common import Instance, TIERS, all_bitstrings, clustering_of
from cvrp import model as cm
from cvrp.model import (brute_force_check, get_formulation, make_toy_instance, model_optimum,
                        registered_formulations)
from cvrp.formulations.textbook import AssignmentOnly, TextbookOneHotSlack

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def hand_instance() -> Instance:
    """N=4, K=2, Q=7 with demands [3, 3, 3, 4]: only the clusterings {3,4}|{3,3}
    are feasible (three of them); the all-but-one on one vehicle is infeasible."""
    return Instance(
        name="hand-S", tier="S", seed=-1, N=4, K=2, Q=7, depot=(50.0, 50.0),
        locations=[(10.0, 10.0), (20.0, 10.0), (90.0, 90.0), (80.0, 95.0)],
        demands=[3, 3, 3, 4], certificate=(0, 0, 1, 1),
    )


def literal_textbook_energy(f: TextbookOneHotSlack, x: np.ndarray) -> float:
    """E(x) written exactly as in DERIVATIONS.md §3.2, no algebra."""
    inst, N, K = f.inst, f.N, f.K
    A, B = f.weights["A"], f.weights["B"]
    D = inst.customer_dist()
    xs = lambda i, k: int(x[f.xidx(i, k)])
    obj = sum(D[i, j] * xs(i, k) * xs(j, k) for k in range(K) for i in range(N) for j in range(i + 1, N))
    one_hot = sum((sum(xs(i, k) for k in range(K)) - 1) ** 2 for i in range(N))
    cap = sum((sum(inst.demands[i] * xs(i, k) for i in range(N)) + f.slack_value(x, k) - inst.Q) ** 2
              for k in range(K))
    return obj + A * one_hot + B * cap


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_has_both_textbook_formulations():
    reg = registered_formulations()
    assert reg["textbook"] is TextbookOneHotSlack
    assert reg["assignment-only"] is AssignmentOnly
    inst = hand_instance()
    f = get_formulation("textbook", inst, A=10.0)
    assert f.weights["A"] == 10.0 and f.weights["B"] == f.objective_upper_bound() + 1
    try:
        get_formulation("does-not-exist", inst)
        assert False, "expected KeyError"
    except KeyError:
        pass


# --------------------------------------------------------------------------- #
# textbook Q expansion == literal penalty formula
# --------------------------------------------------------------------------- #
def test_textbook_Q_matches_literal_formula_on_random_bitstrings():
    rng = np.random.default_rng(1)
    for inst in (hand_instance(), make_toy_instance("S", 3)):
        f = TextbookOneHotSlack(inst, A=7.5, B=2.25)
        Q = f.Q()
        assert np.allclose(Q, Q.T), "Q must be symmetric"
        X = rng.integers(0, 2, size=(200, f.n_vars))
        E = f.energies(X)
        for x, e in zip(X, E):
            assert math.isclose(e, literal_textbook_energy(f, x), rel_tol=1e-12, abs_tol=1e-9)
            assert math.isclose(f.energy(x), e, rel_tol=1e-12, abs_tol=1e-9)
    # also exhaustively on the 2^14 strings of the hand instance
    f = TextbookOneHotSlack(hand_instance())
    X = all_bitstrings(f.n_vars)
    E = f.energies(X)
    lit = np.array([literal_textbook_energy(f, x) for x in X])
    assert np.allclose(E, lit)


def test_textbook_linear_terms_and_couplings_match_derivation():
    inst = hand_instance()
    A, B = 5.0, 3.0
    f = TextbookOneHotSlack(inst, A=A, B=B)
    h, J = f.linear(), f.couplings()
    Qc, d = inst.Q, inst.demands
    for i in range(inst.N):
        for k in range(inst.K):
            assert math.isclose(h[f.xidx(i, k)], -A + B * (d[i] ** 2 - 2 * Qc * d[i]))
    for k in range(inst.K):
        for b in range(f.n_slack_bits):
            assert math.isclose(h[f.sidx(k, b)], B * (4 ** b - 2 * Qc * 2 ** b))
    D = inst.customer_dist()
    # same vehicle, two customers: objective d_ij + capacity 2B d_i d_j
    assert math.isclose(J[f.xidx(0, 1), f.xidx(2, 1)], D[0, 2] + 2 * B * d[0] * d[2])
    # same customer, two vehicles: one-hot 2A
    assert math.isclose(J[f.xidx(1, 0), f.xidx(1, 1)], 2 * A)
    # customer to slack bit of its vehicle: 2B d_i 2^b ; slack-slack: 2B 2^b 2^b'
    assert math.isclose(J[f.xidx(3, 0), f.sidx(0, 2)], 2 * B * d[3] * 4)
    assert math.isclose(J[f.sidx(1, 0), f.sidx(1, 2)], 2 * B * 1 * 4)
    # different vehicles, different customers: uncoupled
    assert J[f.xidx(0, 0), f.xidx(2, 1)] == 0.0 and J[f.sidx(0, 0), f.sidx(1, 0)] == 0.0
    # every coupling non-negative; linear terms NOT uniform (the device problem)
    assert (J >= 0).all()
    assert len(set(np.round(h, 9))) > 1
    # offset = A N + B K Q^2
    assert math.isclose(f.offset, A * inst.N + B * inst.K * Qc ** 2)


# --------------------------------------------------------------------------- #
# decode round trip
# --------------------------------------------------------------------------- #
def test_decode_round_trip_and_invalid_rows():
    inst = hand_instance()
    f = TextbookOneHotSlack(inst)
    for a in itertools.product(range(inst.K), repeat=inst.N):
        x = f.encode(a)
        assert f.decode(x) == tuple(a)
        # with the right slack a feasible assignment has zero penalty: E == objective
        if inst.capacity_ok(a):
            assert math.isclose(f.energy(x), f.objective(a), rel_tol=1e-12, abs_tol=1e-9)
        else:
            assert f.energy(x) > f.objective(a)
    x = f.encode((0, 0, 1, 1))
    x[f.xidx(2, 0)] = 1                      # customer 2 on two vehicles
    assert f.decode(x) is None
    x = f.encode((0, 0, 1, 1))
    x[f.xidx(1, 0)] = 0                      # customer 1 on no vehicle
    assert f.decode(x) is None
    X = all_bitstrings(f.n_vars)
    A = f.decode_many(X)
    for m in range(0, X.shape[0], 97):
        a = f.decode(X[m])
        assert (a is None and (A[m] == -1).all()) or (a is not None and tuple(A[m]) == a)
    # AssignmentOnly decodes the same bits
    g = AssignmentOnly(inst)
    assert g.decode(f.encode((1, 0, 1, 0))[: g.n_vars]) == (1, 0, 1, 0)
    assert g.n_vars == 8 and f.n_vars == 14


# --------------------------------------------------------------------------- #
# objective / cluster_cost / model optimum
# --------------------------------------------------------------------------- #
def test_objective_is_additive_and_model_optimum_is_exact():
    inst = hand_instance()
    f = TextbookOneHotSlack(inst)
    assert f.additive
    D = inst.customer_dist()
    assert math.isclose(f.cluster_cost(frozenset([0, 1, 2])), D[0, 1] + D[0, 2] + D[1, 2])
    assert f.cluster_cost(frozenset([2])) == 0.0
    assert math.isclose(f.objective((0, 0, 1, 1)), D[0, 1] + D[2, 3])
    assert math.isclose(f.objective_upper_bound(), f.objective((0, 0, 0, 0)))
    v, a = model_optimum(f)
    # by hand: feasible clusterings pair the demand-4 customer (3) with one other
    best = min(f.objective(b) for b in itertools.product(range(2), repeat=4) if inst.capacity_ok(b))
    assert math.isclose(v, best)
    assert clustering_of(a) == clustering_of((0, 0, 1, 1))   # geometry: {0,1} close, {2,3} close
    assert inst.capacity_ok(a)


def test_model_optimum_refuses_huge_enumerations():
    t = TIERS["L"]
    inst = Instance(name="big", tier="L", seed=0, N=t["N"], K=t["K"], Q=t["Q"], depot=(50.0, 50.0),
                    locations=[(float(i), 0.0) for i in range(t["N"])], demands=[1] * t["N"],
                    certificate=tuple([0] * t["N"]))
    f = AssignmentOnly(inst)
    try:
        model_optimum(f)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "cvrp.exact" in str(e)


# --------------------------------------------------------------------------- #
# brute-force checker
# --------------------------------------------------------------------------- #
def test_brute_force_check_passes_on_tier_S_with_default_weights():
    for inst in (hand_instance(), make_toy_instance("S", 0)):
        f = get_formulation("textbook", inst)
        r = brute_force_check(f)
        assert r.n_vars == 14 and r.n_bitstrings == 2 ** 14
        assert r.sound and r.argmin_decodes_feasible and r.argmin_is_model_optimum
        assert r.energy_gap > 0
        assert r.min_feasible_energy == r.min_energy
        assert math.isclose(r.min_feasible_energy, r.model_optimum)
        # decodable fraction: exactly one 1 per one-hot row -> (K/2^K)^N = (2/4)^4 = 1/16
        assert math.isclose(r.decodable_fraction, (inst.K / 2 ** inst.K) ** inst.N)
        assert 0 < r.feasible_fraction <= r.decodable_fraction
        # per-assignment min energy equals the objective -> perfect correlation
        assert math.isclose(r.gap_landscape["pearson"], 1.0, abs_tol=1e-9)
        assert math.isclose(r.gap_landscape["spearman"], 1.0, abs_tol=1e-9)
        assert r.gap_landscape["table"][0]["gap_pct"] == 0.0
        names = {m.weight for m in r.penalty_margin}
        assert names == {"A", "B"}
        assert "SOUND" in r.summary()


def test_halving_A_enough_times_breaks_the_penalty():
    inst = make_toy_instance("S", 0)
    r = brute_force_check(get_formulation("textbook", inst))
    mA = next(m for m in r.penalty_margin if m.weight == "A")
    # Default rule A = sum_{i<j} d_ij + 1 is deliberately loose, so a single
    # halving still passes on this seed (ok_at_half True); the scan must find
    # the factor at which it breaks, and an independent re-check must agree.
    assert mA.ok_at_double
    assert mA.halvings_until_failure is not None, "A never bound within 10 halvings -- rule far too loose?"
    A0 = r.weights["A"]
    bad = brute_force_check(get_formulation("textbook", inst, A=A0 * 0.5 ** mA.halvings_until_failure), margins=False)
    assert not bad.sound and bad.energy_gap <= 0
    good = brute_force_check(get_formulation("textbook", inst, A=A0 * 0.5 ** (mA.halvings_until_failure - 1)), margins=False)
    assert good.sound
    assert good.sound == mA.ok_at_half or mA.halvings_until_failure != 1
    # A = 0 is never sound: the all-zero string then wins (nobody served)
    assert not brute_force_check(get_formulation("textbook", inst, A=0.0), margins=False).sound


def test_capacity_weight_binds_on_some_seed_and_not_others():
    # Honest finding (DERIVATIONS §3.4): the pairwise-distance objective already
    # prefers small clusters, so on most tier-S seeds B is not binding.
    r0 = brute_force_check(get_formulation("textbook", make_toy_instance("S", 0)))
    r14 = brute_force_check(get_formulation("textbook", make_toy_instance("S", 14)))
    mB0 = next(m for m in r0.penalty_margin if m.weight == "B")
    mB14 = next(m for m in r14.penalty_margin if m.weight == "B")
    assert mB0.ok_at_zero and mB0.halvings_until_failure is None
    assert not mB14.ok_at_zero and mB14.halvings_until_failure is not None
    assert r14.sound


def test_checker_detects_the_unconstrained_formulation():
    r = brute_force_check(AssignmentOnly(hand_instance()))
    assert not r.sound and not r.argmin_decodes_feasible
    assert r.min_energy == 0.0 and r.energy_gap < 0
    assert r.penalty_margin == []           # no weights to scan


def test_checker_refuses_more_than_20_vars():
    t = TIERS["M"]
    inst = Instance(name="m", tier="M", seed=0, N=t["N"], K=t["K"], Q=t["Q"], depot=(50.0, 50.0),
                    locations=[(float(5 * i), 0.0) for i in range(t["N"])], demands=[1] * t["N"],
                    certificate=tuple(i % t["K"] for i in range(t["N"])))
    try:
        brute_force_check(AssignmentOnly(inst))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ranks_and_pearson_helpers():
    assert cm._ranks(np.array([3.0, 1.0, 2.0])).tolist() == [3.0, 1.0, 2.0]
    assert cm._ranks(np.array([1.0, 1.0, 2.0])).tolist() == [1.5, 1.5, 3.0]
    assert math.isclose(cm._pearson(np.array([1, 2, 3.0]), np.array([2, 4, 6.0])), 1.0)
    assert math.isnan(cm._pearson(np.array([1.0, 1.0]), np.array([1.0, 2.0])))


# --------------------------------------------------------------------------- #
# variable counts
# --------------------------------------------------------------------------- #
def test_count_vars_matches_n_vars_and_brief_numbers():
    expect = {"S": (8, 14), "M": (30, 45), "L": (60, 80)}
    for tier, (a, b) in expect.items():
        t = TIERS[tier]
        assert AssignmentOnly.count_vars(t["N"], t["K"], t["Q"]) == a
        assert TextbookOneHotSlack.count_vars(t["N"], t["K"], t["Q"]) == b
        assert 2 ** TextbookOneHotSlack.slack_bits(t["Q"]) - 1 >= t["Q"]
        assert 2 ** (TextbookOneHotSlack.slack_bits(t["Q"]) - 1) - 1 < t["Q"]
    for seed in (0, 5):
        inst = make_toy_instance("S", seed)
        for cls in registered_formulations().values():
            f = cls(inst)
            assert f.n_vars == cls.count_vars(inst.N, inst.K, inst.Q)
            assert len(f.var_names()) == f.n_vars
            assert f.Q().shape == (f.n_vars, f.n_vars)


def test_variable_count_script_runs_and_write_md_is_idempotent():
    env = {"PYTHONPATH": ":".join(p for p in sys.path if p)}
    import os
    env = {**os.environ, **env}
    out = subprocess.run(PY + [str(ROOT / "scripts" / "variable_count.py")], capture_output=True, text=True,
                         cwd=ROOT, env=env)
    assert out.returncode == 0, out.stderr
    assert "| 14 | 45 | 80 |" in out.stdout and "| 8 | 30 | 60 |" in out.stdout
    md = ROOT / "docs" / "MODEL_WRITEUP.md"
    before = md.read_text()
    out = subprocess.run(PY + [str(ROOT / "scripts" / "variable_count.py"), "--write-md"], capture_output=True,
                         text=True, cwd=ROOT, env=env)
    assert out.returncode == 0, out.stderr
    after = md.read_text()
    assert "<!-- VARIABLE_COUNT_TABLE -->" in after and "| 14 | 45 | 80 |" in after
    # running twice yields the same file (block is replaced, not appended)
    subprocess.run(PY + [str(ROOT / "scripts" / "variable_count.py"), "--write-md"], capture_output=True,
                   text=True, cwd=ROOT, env=env)
    assert md.read_text() == after
    assert after.count("<!-- VARIABLE_COUNT_TABLE -->") == 1
    if before != after:
        print("note: MODEL_WRITEUP.md table was stale and has been regenerated")


def test_toy_instance_is_deterministic_and_certified():
    a, b = make_toy_instance("S", 7), make_toy_instance("S", 7)
    assert a.to_dict() == b.to_dict()
    assert a.capacity_ok(a.certificate) and a.N == 4 and a.K == 2 and a.Q == 7
    assert all(1 <= d <= 4 for d in a.demands)


# =========================================================================== #
# Legacy tests of the generic toolkit (qopt/model.py).  They need pytest's
# `approx`/`raises`; a minimal shim keeps them runnable under scripts/run_tests.py.
# =========================================================================== #
try:  # pragma: no cover
    import pytest  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import contextlib

    class _Approx:
        def __init__(self, v, rel=1e-6, abs_=1e-12):
            self.v, self.rel, self.abs = v, rel, abs_

        def __eq__(self, other):
            if isinstance(self.v, (list, tuple)):
                return len(other) == len(self.v) and all(_Approx(a, self.rel, self.abs) == b for a, b in zip(self.v, other))
            return abs(other - self.v) <= max(self.rel * abs(self.v), self.abs)

    class pytest:  # type: ignore
        approx = staticmethod(lambda v, rel=1e-6, abs=1e-12: _Approx(v, rel, abs))

        @staticmethod
        @contextlib.contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"expected {exc.__name__}")

from qopt import model as qmodel  # noqa: E402
from qopt.model import Problem, QUBO, auto_penalty, brute_force_verify  # noqa: E402


def test_qopt_qubo_energy_conventions():
    Q = np.array([[1.0, 2.0], [2.0, -3.0]])
    q = QUBO(Q)
    # E = x^T Q x : x=(1,1) -> 1 + 2*2 - 3 = 2
    assert q.energy([1, 1]) == pytest.approx(2.0)
    assert q.linear.tolist() == [1.0, -3.0]
    assert q.couplings[0, 1] == pytest.approx(4.0)
    assert q.energies(np.array([[1, 1], [1, 0], [0, 0]])).tolist() == pytest.approx([2.0, 1.0, 0.0])


def test_qopt_knapsack_verifies():
    p = qmodel.knapsack([3, 4, 5, 6], [2, 3, 4, 5], 7)
    b = p.to_qubo()
    rep = brute_force_verify(p, b)
    assert rep.ok
    assert rep.constrained_optimum == 9.0
    assert b.qubo.project(rep.qubo_argmin).tolist() == [0, 1, 1, 0]
    assert len(b.slacks) == 1 and sum(b.slacks[0].coeffs) == 7


def test_qopt_slack_expansion_non_power_of_two():
    # range 0..5 -> coeffs 1,2,2
    assert qmodel._binary_expansion(5) == [1.0, 2.0, 2.0]
    assert qmodel._binary_expansion(7) == [1.0, 2.0, 4.0]
    assert qmodel._binary_expansion(0) == []


def test_qopt_equality_and_ge_constraints():
    p = Problem(["a", "b", "c"])
    p.set_objective(linear={"a": 1, "b": 2, "c": 3}, sense="min")
    p.add_eq({"a": 1, "b": 1, "c": 1}, 2, name="two")
    p.add_ge({"b": 1, "c": 1}, 1, name="bc")
    b = p.to_qubo()
    assert brute_force_verify(p, b).ok
    x, v = p.constrained_optimum()
    assert v == 3.0 and x.tolist() == [1, 1, 0]


def test_qopt_one_hot_and_quadratic_objective():
    p = Problem(["x", "y", "z"])
    p.set_objective(linear={"x": -1, "y": -1, "z": -1}, quadratic={("x", "y"): 3.0}, sense="min")
    p.add_one_hot(["y", "z"])
    b = p.to_qubo()
    assert brute_force_verify(p, b).ok


def test_qopt_penalty_too_small_is_detected_and_auto_fixed():
    p = qmodel.knapsack([10, 10], [1, 1], 1)  # optimum 10; penalty 0.5 is too small
    bad = p.to_qubo(penalty=0.5)
    assert not brute_force_verify(p, bad).ok
    good = auto_penalty(p, start=0.5)
    assert brute_force_verify(p, good).ok


def test_qopt_set_packing_and_mis():
    sp = qmodel.set_packing([[0, 1], [1, 2], [2, 3], [0, 3], [4]], 5)
    assert brute_force_verify(sp, sp.to_qubo()).ok
    mis = qmodel.max_independent_set([(0, 1), (1, 2), (2, 3), (3, 0)], 4)
    rep = brute_force_verify(mis, mis.to_qubo())
    assert rep.ok and rep.constrained_optimum == -2.0


def test_qopt_infeasible_constraint_raises():
    p = Problem(["a"])
    p.add_le({"a": 1}, -1)
    with pytest.raises(ValueError):
        p.to_qubo()
