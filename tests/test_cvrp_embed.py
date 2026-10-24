"""Tests for cvrp.embed (plain functions, no pytest fixtures; deterministic seeds).

Run:  PYTHONPATH=/home/claude/_pydeps:. python3 scripts/run_tests.py cvrp_embed
"""
import numpy as np

from pulser.devices import AnalogDevice

from cvrp.embed import (
    Embedding, FidelityReport, coupling_at, demo_embedding, demo_qubo, embed,
    one_worked_pair, target_distance,
)

CH = AnalogDevice.channels["rydberg_global"]
C6 = float(AnalogDevice.interaction_coeff)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


def _pair(J12: float) -> np.ndarray:
    return np.array([[0.0, J12], [J12, 0.0]])


# --------------------------------------------------------------------------- #
# refusal / input validation
# --------------------------------------------------------------------------- #
def test_negative_coupling_is_refused_naming_the_pair():
    J = np.array([[0, 0.5, -0.3], [0.5, 0, 0.2], [-0.3, 0.2, 0]], float)
    e = _raises(ValueError, embed, J, -np.ones(3), n_starts=1)
    msg = str(e)
    assert "negative" in msg and "(0,2)" in msg and "repulsive" in msg


def test_zero_coupling_refused_only_when_allow_zero_false():
    J = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], float)
    e = _raises(ValueError, embed, J, -np.ones(3), allow_zero=False, n_starts=1)
    assert "(0,1)" in str(e)
    emb = embed(J, -np.ones(3), n_starts=2)             # default allow_zero=True
    assert emb.fidelity.max_spurious_coupling is not None
    assert emb.fidelity.spurious_pair == (0, 1)


def test_too_many_atoms_raises():
    n = int(AnalogDevice.max_atom_num) + 1
    J = np.ones((n, n)); np.fill_diagonal(J, 0.0)
    e = _raises(ValueError, embed, J, -np.ones(n), n_starts=1)
    assert "max_atom_num" in str(e)


def test_asymmetric_or_mismatched_inputs_raise():
    _raises(ValueError, embed, np.array([[0, 1], [0.5, 0]]), -np.ones(2))
    _raises(ValueError, embed, _pair(0.5), -np.ones(3))


# --------------------------------------------------------------------------- #
# distance formula
# --------------------------------------------------------------------------- #
def test_target_distance_inverts_coupling():
    for J, a in [(0.5, 10.0), (1.0, 46.0), (0.2, 3.0)]:
        r = target_distance(J, a, C6)
        assert abs(coupling_at(r, C6) - a * J) < 1e-9 * a * J
    # at the 5 µm floor the device gives C6/5^6 = 55.4 rad/µs
    assert abs(coupling_at(AnalogDevice.min_atom_distance, C6) - C6 / 5.0 ** 6) < 1e-9


def test_two_atom_embedding_reproduces_target_distance():
    emb = embed(_pair(0.5), np.array([-1.0, -1.0]), n_starts=2)
    r_real = float(np.linalg.norm(emb.coords[0] - emb.coords[1]))
    r_target = target_distance(0.5, emb.alpha, C6)
    assert abs(r_real - r_target) < 1e-6, (r_real, r_target)
    assert emb.fidelity.rel_frobenius_offdiag < 1e-9
    assert emb.fidelity.worst_pair_rel_error < 1e-9
    # alpha is as large as the geometry allows: the pair sits at the min distance (x1.02 optimiser margin)
    assert AnalogDevice.min_atom_distance <= r_real < 1.03 * AnalogDevice.min_atom_distance
    # centred register
    assert np.allclose(emb.coords.mean(0), 0.0, atol=1e-9)
    # delta_final = -alpha h
    assert abs(emb.delta_final - (-emb.alpha * -1.0)) < 1e-12
    wp = one_worked_pair(emb)
    assert wp["pair"] == (0, 1) and abs(wp["realized_distance_um"] - r_target) < 1e-6 and wp["rel_error"] < 1e-9


# --------------------------------------------------------------------------- #
# alpha choice / caps
# --------------------------------------------------------------------------- #
def test_detuning_cap_scales_distances_up_and_keeps_delta_within_limit():
    emb = embed(_pair(0.5), np.array([-10.0, -10.0]), n_starts=2)
    assert abs(emb.delta_final) <= CH.max_abs_detuning * (1 + 1e-9)
    assert abs(abs(emb.delta_final) - CH.max_abs_detuning) < 1e-6      # cap is active
    assert abs(emb.alpha - CH.max_abs_detuning / 10.0) < 1e-9
    r = float(np.linalg.norm(emb.coords[0] - emb.coords[1]))
    assert abs(r - target_distance(0.5, emb.alpha, C6)) < 1e-6        # U ~ r^-6 rescaling exact
    assert emb.fidelity.limit_violations == []
    assert any("scaled" in n for n in emb.fidelity.notes)


def test_requested_alpha_is_honoured():
    emb = embed(_pair(0.5), np.array([-1.0, -1.0]), n_starts=2, alpha=10.0)
    assert abs(emb.alpha - 10.0) < 1e-12
    r = float(np.linalg.norm(emb.coords[0] - emb.coords[1]))
    assert abs(r - target_distance(0.5, 10.0, C6)) < 1e-6
    assert emb.fidelity.limit_violations == []


def test_impossible_requests_are_reported_not_hidden():
    # alpha beyond the 5 µm floor -> min-distance violation reported
    emb = embed(_pair(0.5), np.array([-1.0, -1.0]), n_starts=2, alpha=200.0)
    assert any("min distance" in v for v in emb.fidelity.limit_violations)
    # huge |h| needs atoms farther apart than the 38 µm radius allows -> radius fixed, detuning violation reported
    emb = embed(_pair(0.5), np.array([-1e8, -1e8]), n_starts=2)
    assert emb.fidelity.max_radius <= AnalogDevice.max_radial_distance + 1e-6
    assert any("max_abs_detuning" in v for v in emb.fidelity.limit_violations)


# --------------------------------------------------------------------------- #
# linear terms
# --------------------------------------------------------------------------- #
def test_non_uniform_linear_terms_set_flag_and_use_mean():
    emb = embed(_pair(0.5), np.array([-1.0, -3.0]), n_starts=2)
    f = emb.fidelity
    assert f.linear_terms_uniform is False
    assert abs(f.linear_spread - 2.0) < 1e-12
    assert abs(emb.delta_final - (-emb.alpha * -2.0)) < 1e-9        # mean h = -2
    assert any("NOT A FAITHFUL" in n for n in f.notes)
    assert not f.ok
    uni = embed(_pair(0.5), np.array([-1.0, -1.0 + 1e-12]), n_starts=2)
    assert uni.fidelity.linear_terms_uniform is True and uni.fidelity.ok


# --------------------------------------------------------------------------- #
# demo embedding: what we measure, honestly
# --------------------------------------------------------------------------- #
def test_demo_embedding_respects_device_and_reports_fidelity():
    emb = demo_embedding(seed=0, n_starts=8)
    f = emb.fidelity
    assert isinstance(f, FidelityReport) and f.n_atoms == 6
    assert f.limit_violations == []
    assert f.min_distance >= AnalogDevice.min_atom_distance - 1e-6
    assert f.max_radius <= AnalogDevice.max_radial_distance + 1e-6
    assert abs(emb.delta_final) <= CH.max_abs_detuning
    assert f.linear_terms_uniform
    # the dense random toy (12 of 15 couplings in [0.2,1]) is NOT faithfully embeddable in 2D with 1/r^6:
    # measured 0.370 on seed 0 (nine geometric degrees of freedom for twelve targets).
    assert 0.2 < f.rel_frobenius_offdiag < 0.45, f.rel_frobenius_offdiag
    assert f.worst_pair is not None and f.worst_pair_rel_error > f.rel_frobenius_offdiag
    assert f.max_spurious_coupling is not None and f.max_spurious_coupling < 0.05
    # formula check: rel Frobenius recomputed from the public arrays
    off = ~np.eye(6, dtype=bool)
    rel = np.linalg.norm((emb.U / emb.alpha - emb.J_target)[off]) / np.linalg.norm(emb.J_target[off])
    assert abs(rel - f.rel_frobenius_offdiag) < 1e-12
    # strongest pair sits at (about) the min distance: alpha as large as geometry allows
    wp = one_worked_pair(emb)
    assert wp["realized_distance_um"] < 1.05 * AnalogDevice.min_atom_distance
    assert abs(wp["U_realized"] - C6 / wp["realized_distance_um"] ** 6) < 1e-9


def test_demo_is_deterministic_and_serialisable():
    a = demo_embedding(seed=0, n_starts=3)
    b = demo_embedding(seed=0, n_starts=3)
    assert np.allclose(a.coords, b.coords) and a.alpha == b.alpha
    d = a.to_dict()
    assert d["qubit_ids"] == [f"q{i}" for i in range(6)]
    assert len(d["coords_um"]) == 6 and "rel_frobenius_offdiag" in d["fidelity"]
    import json
    json.dumps(d)


def test_realized_Q_matches_device_energy():
    emb = demo_embedding(seed=1, n_starts=3)
    Q = emb.realized_Q()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 2, 6).astype(float)
        e_qubo_units = x @ Q @ x
        assert abs(emb.alpha * e_qubo_units - emb.device_energy(x)) < 1e-9
    assert np.allclose(np.diag(Q), -emb.delta_final / emb.alpha)
    assert np.allclose(2 * Q - np.diag(np.diag(2 * Q)), emb.U / emb.alpha)


def test_register_matches_coords_and_layout_is_attached():
    emb = demo_embedding(seed=0, n_starts=3)
    reg = emb.register()
    assert list(reg.qubit_ids) == emb.qubit_ids
    for i, q in enumerate(reg.qubit_ids):
        assert np.allclose(np.asarray(reg.qubits[q], float), emb.coords[i])
    reg2 = emb.register(auto_layout=True)
    assert reg2.layout is not None
    for i, q in enumerate(reg2.qubit_ids):
        assert np.allclose(np.asarray(reg2.qubits[q], float), emb.coords[i], atol=1e-9)
