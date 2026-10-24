"""Tests for cvrp.instances (generator) and cvrp.validator.

Plain ``def test_*():`` functions with no fixtures so they run both under real
pytest and under ``scripts/run_tests.py``.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from cvrp.common import SQUARE_SIDE, TIERS, Instance
from cvrp.instances import DEFAULT_SEEDS, GENERATOR_VERSION, generate, generate_default
from cvrp.validator import validate


def _roundtrip(inst: Instance) -> Instance:
    return Instance.from_dict(json.loads(json.dumps(inst.to_dict(), sort_keys=True)))


def test_generator_is_deterministic():
    """Same tier + seed -> byte-identical JSON; different seed -> different data."""
    for tier in TIERS:
        a = json.dumps(generate(tier, 3).to_dict(), sort_keys=True)
        b = json.dumps(generate(tier, 3).to_dict(), sort_keys=True)
        assert a == b, f"tier {tier}: generation is not deterministic"
        c = generate(tier, 4)
        assert c.locations != generate(tier, 3).locations


def test_generated_instances_match_tier_and_are_certified():
    for tier, t in TIERS.items():
        for seed in range(6):
            inst = generate(tier, seed)
            assert (inst.N, inst.K, inst.Q) == (t["N"], t["K"], t["Q"])
            assert len(inst.locations) == inst.N and len(inst.demands) == inst.N
            assert all(t["d_min"] <= d <= t["d_max"] for d in inst.demands)
            assert all(0 <= x <= SQUARE_SIDE and 0 <= y <= SQUARE_SIDE for x, y in inst.locations)
            assert len(set(inst.locations)) == inst.N
            assert inst.depot == (50.0, 50.0)
            # the certificate is a valid assignment and every vehicle is used
            assert inst.is_valid_assignment(inst.certificate)
            assert set(inst.certificate) == set(range(inst.K))
            assert all(L <= inst.Q for L in inst.loads(inst.certificate))
            assert abs(inst.rho - sum(inst.demands) / (inst.K * inst.Q)) < 1e-12
            assert abs(inst.dmax_over_Q - max(inst.demands) / inst.Q) < 1e-12
            assert inst.rho >= 0.6 or inst.generator["attempts"] >= inst.generator["max_attempts"]
            assert inst.feasibility_method.startswith("certificate-first")
            assert inst.generator["version"] == GENERATOR_VERSION


def test_default_set_validates_with_regeneration():
    with tempfile.TemporaryDirectory() as tmp:
        made = generate_default(tmp)
        assert len(made) == sum(len(v) for v in DEFAULT_SEEDS.values())
        for p in sorted(Path(tmp).glob("*.json")):
            inst = Instance.load(p)
            assert validate(inst, regenerate=True) == [], validate(inst, regenerate=True)


def test_json_roundtrip_is_exact():
    inst = generate("L", 1)
    again = _roundtrip(inst)
    assert again.to_dict() == inst.to_dict()
    assert validate(again) == []


def test_demand_bias_raises_load_ratio():
    """bias=1 always takes the largest demand that fits -> rho close to 1."""
    for tier in TIERS:
        hi = generate(tier, 0, demand_bias=1.0)
        lo = generate(tier, 0, demand_bias=0.0)
        assert hi.rho >= lo.rho
        assert hi.rho > 0.85
        assert validate(hi) == []


def test_unknown_option_rejected():
    try:
        generate("S", 0, foo=1)
    except TypeError:
        return
    raise AssertionError("unknown generator option should raise TypeError")


# --------------------------------------------------------------------------- #
# Validator must catch corrupted instances
# --------------------------------------------------------------------------- #
def _corrupt(**changes) -> Instance:
    d = generate("M", 0).to_dict()
    d.update(changes)
    return Instance.from_dict(d)


def test_validator_accepts_clean_instance():
    assert validate(generate("M", 0)) == []


def test_validator_catches_demand_over_Q():
    inst = generate("M", 0)
    d = list(inst.demands)
    d[0] = inst.Q + 1
    errs = validate(_corrupt(demands=d, rho=sum(d) / (inst.K * inst.Q), dmax_over_Q=max(d) / inst.Q))
    assert any("exceeds Q" in e for e in errs), errs


def test_validator_catches_demand_outside_tier_range():
    inst = generate("M", 0)
    d = list(inst.demands)
    d[1] = TIERS["M"]["d_max"] + 1   # 9 <= Q=20, so only the tier range catches it
    errs = validate(_corrupt(demands=d, rho=sum(d) / (inst.K * inst.Q), dmax_over_Q=max(d) / inst.Q))
    assert any("outside tier range" in e for e in errs), errs


def test_validator_catches_duplicate_location():
    inst = generate("M", 0)
    locs = list(inst.locations)
    locs[2] = locs[5]
    errs = validate(_corrupt(locations=locs))
    assert any("duplicate" in e for e in errs), errs


def test_validator_catches_location_outside_square():
    inst = generate("M", 0)
    locs = list(inst.locations)
    locs[0] = (101.0, 50.0)
    errs = validate(_corrupt(locations=locs))
    assert any("outside the square" in e for e in errs), errs


def test_validator_catches_bad_certificate():
    inst = generate("M", 0)
    # everyone on vehicle 0: load = sum d > Q (rho >= 0.6 with K=3 guarantees that)
    errs = validate(_corrupt(certificate=[0] * inst.N))
    assert any("certificate" in e and "exceeds Q" in e for e in errs), errs
    errs = validate(_corrupt(certificate=list(inst.certificate)[:-1]))
    assert any("certificate has length" in e for e in errs), errs
    bad = list(inst.certificate)
    bad[0] = inst.K
    errs = validate(_corrupt(certificate=bad))
    assert any("outside 0.." in e for e in errs), errs


def test_validator_catches_wrong_tier_parameters_and_ratios():
    errs = validate(_corrupt(Q=21))
    assert any("Q=21" in e for e in errs), errs
    errs = validate(_corrupt(rho=0.1))
    assert any("rho=" in e for e in errs), errs
    errs = validate(_corrupt(depot=(0.0, 0.0)))
    assert any("centre" in e for e in errs), errs
    errs = validate(_corrupt(depot_rule=""))
    assert any("depot_rule is empty" in e for e in errs), errs


def test_validator_catches_wrong_reference_cost_and_bad_reference_assignment():
    from cvrp.common import route_exactly

    inst = generate("S", 0)
    a = list(inst.certificate)
    cost, routes = route_exactly(inst, tuple(a))
    good_ref = {"cost": cost, "best_assignment": a, "reference_status": "proven_optimum", "method": "test",
                "best_clusters": [[i for i in range(inst.N) if a[i] == k] for k in range(inst.K)],
                "routes": [routes[k][1] if k in routes else [] for k in range(inst.K)]}
    assert validate(_corrupt_ref(inst, good_ref)) == []
    # wrong cost
    errs = validate(_corrupt_ref(inst, {**good_ref, "cost": cost + 1.0}))
    assert any("recomputed exact-TSP cost" in e for e in errs), errs
    # SA claims to beat a proven optimum
    errs = validate(_corrupt_ref(inst, {**good_ref, "sa_cost": cost - 5.0, "sa_matches_optimum": False}))
    assert any("below the proven optimum" in e for e in errs), errs
    # best_clusters inconsistent
    errs = validate(_corrupt_ref(inst, {**good_ref, "best_clusters": [list(range(inst.N))]}))
    assert any("best_clusters" in e for e in errs), errs
    # route not a permutation of the vehicle's customers
    bad_routes = [list(r) for r in good_ref["routes"]]
    bad_routes[0] = bad_routes[0][:-1]
    errs = validate(_corrupt_ref(inst, {**good_ref, "routes": bad_routes}))
    assert any("not a permutation" in e for e in errs), errs
    # infeasible reference assignment
    errs = validate(_corrupt_ref(inst, {**good_ref, "best_assignment": [0] * inst.N}))
    assert any("reference best_assignment" in e for e in errs), errs


def _corrupt_ref(inst: Instance, ref: dict) -> Instance:
    d = inst.to_dict()
    d["reference"] = ref
    return Instance.from_dict(d)


def test_triangle_inequality_check_runs_on_euclidean_matrix():
    """Euclidean distances always satisfy the inequality; the check must agree."""
    inst = generate("L", 2)
    D = inst.dist_matrix()
    via = D[:, :, None] + D[None, :, :]
    assert (D[:, None, :] - via).max() <= 1e-9
    assert np.allclose(D, D.T)
