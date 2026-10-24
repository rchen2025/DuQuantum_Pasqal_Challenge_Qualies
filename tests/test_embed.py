import numpy as np
import pytest

from qopt.embed import DeviceLimits, KNOWN_DEVICES, coupling_matrix, embed_qubo, optimize_positions, plan_detunings
from qopt.model import QUBO, knapsack, max_independent_set

LIM = KNOWN_DEVICES["WeightedAnalogDevice"]


def test_known_limits_match_pulser_if_installed():
    pulser = pytest.importorskip("pulser")
    for name, known in KNOWN_DEVICES.items():
        live = DeviceLimits.from_device_name(name).to_dict()
        for k, v in known.to_dict().items():
            if isinstance(v, float):
                assert live[k] == pytest.approx(v, rel=1e-6), (name, k)
            else:
                assert live[k] == v, (name, k)


def test_optimizer_recovers_realizable_geometry():
    coords = np.array([[0, 0], [7, 0], [0, 7], [7, 7]], float)
    J = coupling_matrix(coords, LIM.c6) / 3.0
    c, alpha, info = optimize_positions(J, LIM, n_starts=3)
    assert info["rel_error"] < 1e-3
    # geometry is recovered up to a global scale (alpha absorbs it)
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    d0 = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    ratio = d[d0 > 0] / d0[d0 > 0]
    assert np.allclose(ratio, ratio.mean(), rtol=1e-2)


def test_embedding_respects_limits_and_preserves_ground_state_on_mis():
    p = max_independent_set([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5)
    q = p.to_qubo().qubo
    emb = embed_qubo(q, LIM, n_starts=4)
    r = emb.report
    assert r.limit_violations == []
    assert r.min_distance >= LIM.min_atom_distance - 1e-6
    assert r.max_radial <= LIM.max_radial_distance
    assert r.ground_state_preserved
    assert r.uses_dmm
    assert all(0.0 <= w <= 1.0 for w in emb.weights.values())
    assert abs(emb.plan.delta_global) <= LIM.max_abs_detuning + 1e-9
    assert emb.plan.delta_dmm >= LIM.dmm_bottom_detuning - 1e-9


def test_realized_qubo_matches_plan_diagonal():
    q = knapsack([3, 4, 5, 6], [2, 3, 4, 5], 7).to_qubo().qubo
    emb = embed_qubo(q, LIM, n_starts=2, check_ground_state=False)
    rq = emb.realized_qubo()
    # diagonal realised exactly (DMM), couplings only approximately
    assert np.allclose(rq.linear, q.linear, rtol=1e-6)
    assert emb.report.rel_error_diag < 1e-6
    assert 0 < emb.report.rel_error_offdiag < 1  # dense penalty QUBO is not exactly embeddable


def test_decode_drops_slack():
    q = knapsack([3, 4], [1, 2], 2).to_qubo().qubo
    emb = embed_qubo(q, LIM, n_starts=1, check_ground_state=False)
    assert emb.decode("10" + "1" * (q.n - 2)).tolist() == [1, 0]


def test_no_dmm_fallback_uses_mean():
    lim = KNOWN_DEVICES["AnalogDevice"]
    plan = plan_detunings(np.array([-1.0, -3.0]), 2.0, lim)
    assert not plan.uses_dmm and plan.delta_global == pytest.approx(4.0)


def test_detuning_cap_rescales_distances():
    q = QUBO(np.array([[-100.0, 1.0], [1.0, -100.0]]))
    emb = embed_qubo(q, LIM, n_starts=1, check_ground_state=False)
    assert abs(emb.plan.delta_global) <= LIM.max_abs_detuning + 1e-9
    assert emb.report.rescale_factor >= 1.0
