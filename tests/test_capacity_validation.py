"""Capacity validation protocol: exact algebra, ties and no held-out selection."""
import itertools
import math

import numpy as np
import pytest

from cvrp.common import Instance, all_bitstrings
from cvrp.instances import generate
from cvrp.model import get_formulation
from scripts.capacity_validation import (KAPPAS, capacity_binding_audit, ground_state_audit,
    literal_energy, select_kappa, weight_for, wilson)


def test_all_256_energies_and_recomputed_A():
    for kappa in KAPPAS:
        inst = generate("S", 1000)
        f = get_formulation("onehot-ub", inst, **weight_for(inst, kappa))
        W = inst.customer_dist() + 2*f.weights["lam2"]*np.outer(inst.demands, inst.demands)
        np.fill_diagonal(W, 0)
        assert f.weights["A"] == pytest.approx(W.sum(axis=1).max()+1)
        X = all_bitstrings(8)
        assert np.allclose(f.energies(X), [literal_energy(f, x) for x in X], rtol=0, atol=1e-8)
        row = ground_state_audit(inst, kappa, include_states=True)
        assert len(row["states"]) == 256
        assert row["all_ground_states_onehot"]
        states = [s for s in row["states"] if s["ground_state"]]
        assert row["all_ground_states_feasible"] == all(s["feasible"] for s in states)


def test_selection_difficulty_rule_checks_all_ties():
    # Identical coordinates deliberately create tied objectives; some minimizing
    # assignments overload and others are feasible, so this must NOT be selected.
    inst = Instance("tied", "S", 0, 4, 2, 7, (50., 50.), [(0., 0.)]*4,
                    [4, 4, 3, 3], (0, 1, 0, 1))
    row = capacity_binding_audit(inst)
    assert row["n_optimal_assignments"] == 16
    assert row["n_capacity_feasible_optimal_assignments"] > 0
    assert not row["all_unconstrained_optima_infeasible"]
    f = get_formulation("onehot-ub", inst, **weight_for(inst, 0))
    result = ground_state_audit(inst, 0)
    assert result["n_feasible_ground_states"] > 0
    assert not result["all_ground_states_feasible"]


def test_selected_instances_are_certified_and_capacity_binding():
    chosen = []
    for seed in range(1000, 1200):
        inst = generate("S", seed)
        if capacity_binding_audit(inst)["all_unconstrained_optima_infeasible"]:
            chosen.append(inst)
        if len(chosen) == 8:
            break
    assert len(chosen) == 8
    assert [x.seed for x in chosen] == sorted(x.seed for x in chosen)
    for inst in chosen:
        assert inst.is_valid_assignment(inst.certificate)
        row = ground_state_audit(inst, 0)
        assert row["n_feasible_ground_states"] == 0
        assert row["worst_feasible_ground_model_gap"] is None


def _selection_rows():
    return [{"split": "development", "instance": f"dev{i}", "kappa": k,
        "all_ground_states_feasible": True, "worst_feasible_ground_model_gap": 0.}
        for k in KAPPAS for i in range(4)]


def test_selection_priorities_and_missing_ground_state():
    rows = _selection_rows()
    for row in rows:
        if row["kappa"] == 0:
            row["all_ground_states_feasible"] = False
            row["worst_feasible_ground_model_gap"] = None
        elif row["kappa"] == .05:
            row["worst_feasible_ground_model_gap"] = 10.
    chosen, summary = select_kappa(rows)
    assert chosen == .15  # beats .05 on gap, then beats .3/.6/1 on kappa
    assert summary[0]["missing_feasible_ground_state"]
    assert summary[0]["worst_feasible_ground_model_gap"] is None
    rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="development rows only"):
        select_kappa(rows)
    with pytest.raises(ValueError, match="exactly one"):
        select_kappa(_selection_rows()[:-1])


def test_wilson_zero_does_not_claim_impossibility():
    lo, hi = wilson(0, 500)
    assert 0 <= lo < 1e-12
    assert 0.007 < hi < .008
    lo, hi = wilson(500, 500)
    assert .992 < lo < .993
    assert hi == pytest.approx(1.)
