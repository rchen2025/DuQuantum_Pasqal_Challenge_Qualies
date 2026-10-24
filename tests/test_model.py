import numpy as np
import pytest

from qopt import model
from qopt.model import Problem, QUBO, auto_penalty, brute_force_verify


def test_qubo_energy_conventions():
    Q = np.array([[1.0, 2.0], [2.0, -3.0]])
    q = QUBO(Q)
    # E = x^T Q x : x=(1,1) -> 1 + 2*2 - 3 = 2
    assert q.energy([1, 1]) == pytest.approx(2.0)
    assert q.linear.tolist() == [1.0, -3.0]
    assert q.couplings[0, 1] == pytest.approx(4.0)
    assert q.energies(np.array([[1, 1], [1, 0], [0, 0]])).tolist() == pytest.approx([2.0, 1.0, 0.0])


def test_knapsack_verifies():
    p = model.knapsack([3, 4, 5, 6], [2, 3, 4, 5], 7)
    b = p.to_qubo()
    rep = brute_force_verify(p, b)
    assert rep.ok
    assert rep.constrained_optimum == 9.0
    assert b.qubo.project(rep.qubo_argmin).tolist() == [0, 1, 1, 0]
    assert len(b.slacks) == 1 and sum(b.slacks[0].coeffs) == 7


def test_slack_expansion_non_power_of_two():
    # range 0..5 -> coeffs 1,2,2
    assert model._binary_expansion(5) == [1.0, 2.0, 2.0]
    assert model._binary_expansion(7) == [1.0, 2.0, 4.0]
    assert model._binary_expansion(0) == []


def test_equality_and_ge_constraints():
    p = Problem(["a", "b", "c"])
    p.set_objective(linear={"a": 1, "b": 2, "c": 3}, sense="min")
    p.add_eq({"a": 1, "b": 1, "c": 1}, 2, name="two")
    p.add_ge({"b": 1, "c": 1}, 1, name="bc")
    b = p.to_qubo()
    assert brute_force_verify(p, b).ok
    x, v = p.constrained_optimum()
    assert v == 3.0 and x.tolist() == [1, 1, 0]


def test_one_hot_and_quadratic_objective():
    p = Problem(["x", "y", "z"])
    p.set_objective(linear={"x": -1, "y": -1, "z": -1}, quadratic={("x", "y"): 3.0}, sense="min")
    p.add_one_hot(["y", "z"])
    b = p.to_qubo()
    assert brute_force_verify(p, b).ok


def test_penalty_too_small_is_detected_and_auto_fixed():
    p = model.knapsack([10, 10], [1, 1], 1)  # optimum 10; penalty 0.5 is too small
    bad = p.to_qubo(penalty=0.5)
    assert not brute_force_verify(p, bad).ok
    good = auto_penalty(p, start=0.5)
    assert brute_force_verify(p, good).ok


def test_set_packing_and_mis():
    sp = model.set_packing([[0, 1], [1, 2], [2, 3], [0, 3], [4]], 5)
    assert brute_force_verify(sp, sp.to_qubo()).ok
    mis = model.max_independent_set([(0, 1), (1, 2), (2, 3), (3, 0)], 4)
    rep = brute_force_verify(mis, mis.to_qubo())
    assert rep.ok and rep.constrained_optimum == -2.0


def test_infeasible_constraint_raises():
    p = Problem(["a"])
    p.add_le({"a": 1}, -1)
    with pytest.raises(ValueError):
        p.to_qubo()
