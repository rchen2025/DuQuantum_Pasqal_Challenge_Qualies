"""Behavioral checks for bounded classical repair and honest shot scoring."""
from pathlib import Path
import itertools

import numpy as np
import pytest

from cvrp.common import Instance, bits_to_str, route_exactly
from cvrp.metrics import model_optimum, score_counts, wilson_interval
from cvrp.model import get_formulation
from cvrp.repair import capacity_local_repair
from cvrp.results import rows_from_record

ROOT = Path(__file__).resolve().parents[1]


def form():
    return get_formulation("onehot-ub", Instance.load(ROOT / "instances/S_seed00.json"))


def test_exhaustive_small_repair_invariants_and_determinism():
    f = form()
    repair = capacity_local_repair(f)
    independent = capacity_local_repair(f)
    for a in itertools.product(range(f.inst.K), repeat=f.inst.N):
        x = f.encode(a)
        y = repair(x)
        b = f.decode(y)
        assert b is not None
        assert np.array_equal(y, independent(x))
        assert np.array_equal(y, repair(x))
        assert len(repair.traces[a]) <= 2 * f.inst.N
        assert repair.overload(b) <= repair.overload(a)
        if f.feasible(a):
            assert f.feasible(b)
            assert route_exactly(f.inst, b)[0] <= route_exactly(f.inst, a)[0] + 1e-9
        for step in repair.traces[a]:
            if step["stage"] == "overload":
                assert step["overload_after"] < step["overload_before"]
            else:
                assert step["overload_after"] == 0
                assert step["route_after"] < step["route_before"] - 1e-9
    assert repair.cache_hits == f.inst.K ** f.inst.N


def test_malformed_rows_unchanged_and_failure_gap_undefined():
    f = form()
    repair = capacity_local_repair(f)
    for x in [np.zeros(f.n_vars, dtype=int), np.ones(f.n_vars, dtype=int)]:
        assert np.array_equal(repair(x), x)
    metrics, rows = score_counts({"0" * f.n_vars: 500}, f, f.inst, transform=repair)
    assert metrics.feasible_rate == 0
    assert metrics.best_model_gap is None and metrics.best_cvrp_gap is None
    assert not rows[0].feasible
    assert metrics.feasible_rate_ci95[1] > 0


def test_unsuccessful_capacity_repair_remains_failure():
    # 3+3+3+3 in two capacity-6 bins is feasible; limiting accepted moves to
    # zero creates an explicit exhausted-budget failure without a fallback.
    f = form()
    repair = capacity_local_repair(f)
    repair.max_moves = 0
    a = (0,) * f.inst.N
    assert not f.feasible(a)
    counts = {bits_to_str(f.encode(a)): 7}
    m, _ = score_counts(counts, f, f.inst, transform=repair)
    assert m.feasible_rate == 0 and m.best_model_gap is None


def test_repair_does_not_consult_reference_or_certificate():
    f = form()
    expected = capacity_local_repair(f)(f.encode((0,) * f.inst.N))
    f.inst.reference = {"cost": -1e50, "best_assignment": [999]}
    f.inst.certificate = (999,)
    actual = capacity_local_repair(f)(f.encode((0,) * f.inst.N))
    assert np.array_equal(actual, expected)


def test_wilson_boundaries_and_csv_backwards_compatibility():
    assert wilson_interval(0, 500)[0] == 0
    assert wilson_interval(500, 500)[1] == 1
    lo, hi = wilson_interval(250, 500)
    assert 0.45 < lo < 0.5 < hi < 0.55
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    old = {"method": "legacy", "instance": "S", "counts": {"00": 1},
           "metrics_raw": {"shots": 1, "feasible_rate": 0}}
    row = rows_from_record(old)[0]
    assert row["feasibility_rate"] == 0 and row["feasibility_ci95_low"] is None


def test_route_selection_is_distinct_and_raw_metrics_preserved():
    f = get_formulation("onehot-ub", Instance.load(ROOT / "instances/S_seed03.json"))
    counts = {bits_to_str(f.encode(a)): 1 for a in itertools.product(range(f.inst.K), repeat=f.inst.N)}
    original = counts.copy()
    opt = model_optimum(f)
    raw, _ = score_counts(counts, f, f.inst, opt)
    rep, _ = score_counts(counts, f, f.inst, opt, transform=capacity_local_repair(f))
    assert counts == original
    assert raw.best_routed_over_feasible < raw.best_routed_cost
    assert raw.repair_wall_s == 0 and rep.repair_wall_s > 0
    assert rep.decodable_rate == raw.decodable_rate
