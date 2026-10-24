import numpy as np
import pytest

from qopt.classical import solve_best
from qopt.embed import KNOWN_DEVICES, embed_qubo
from qopt.model import knapsack, max_independent_set
from qopt.run import mock_boltzmann, repair, run_pipeline, score_counts
from qopt.schedule import Schedule, build_sequence, final_hamiltonian_from_sequence, scan

LIM = KNOWN_DEVICES["WeightedAnalogDevice"]


@pytest.fixture(scope="module")
def knap():
    p = knapsack([3, 4, 5, 6], [2, 3, 4, 5], 7)
    b = p.to_qubo()
    emb = embed_qubo(b.qubo, LIM, n_starts=2, check_ground_state=False)
    return p, b.qubo, emb


def test_schedule_default_respects_limits(knap):
    _, _, emb = knap
    s = Schedule.default_for(emb)
    assert s.duration % LIM.clock_period == 0
    assert s.duration + 200 <= LIM.max_sequence_duration
    assert s.omega_max <= LIM.max_amp
    assert abs(s.delta_init) <= LIM.max_abs_detuning and abs(s.delta_final) <= LIM.max_abs_detuning
    w = s.waveforms()
    assert (w["amp"] >= 0).all() and w["amp"][0] == pytest.approx(0) and w["amp"][-1] < 0.05 * s.omega_max
    assert w["det"][0] == pytest.approx(s.delta_init) and w["det"][-1] == pytest.approx(s.delta_final, rel=1e-3)
    assert "dmm" in w and (w["dmm"] <= 0).all()


def test_build_sequence_matches_plan(knap):
    pytest.importorskip("pulser")
    _, _, emb = knap
    s = Schedule.default_for(emb)
    seq = build_sequence(emb, s)
    assert seq.get_duration() == s.duration
    assert "dmm_0" in seq.declared_channels
    delta, U = final_hamiltonian_from_sequence(seq)
    assert np.allclose(delta, emb.plan.delta_realized, atol=1e-2)
    assert np.allclose(U, emb.U, rtol=1e-3)
    # serialisable for the cloud
    assert len(seq.to_abstract_repr()) > 100


def test_build_sequence_dmm_ramp_variant(knap):
    pytest.importorskip("pulser")
    _, _, emb = knap
    s = Schedule.default_for(emb, dmm_shape="ramp", sweep_shape="smooth")
    seq = build_sequence(emb, s)
    delta, _ = final_hamiltonian_from_sequence(seq)
    assert np.allclose(delta, emb.plan.delta_realized, atol=1e-2)


def test_repair_fixes_infeasible_and_keeps_feasible(knap):
    p, q, _ = knap
    x, ok = repair(np.array([1, 1, 1, 1]), p, q)
    assert ok and p.is_feasible(p.assignment(x))
    x2, ok2 = repair(np.array([0, 1, 1, 0]), p, q)
    assert ok2 and x2.tolist() == [0, 1, 1, 0]


def test_score_counts_identifies_optimum(knap):
    p, q, emb = knap
    opt_bits = "0110" + "0" * (q.n - 4)
    counts = {opt_bits: 60, "1111" + "0" * (q.n - 4): 40}
    sc = score_counts(counts, p, q, emb.decode, classical=solve_best(q.Q))
    assert sc.classical_objective == 9.0
    assert sc.p_optimal_raw == pytest.approx(0.6)
    assert sc.feasible_rate_raw == pytest.approx(0.6)
    assert sc.feasible_rate_repaired == pytest.approx(1.0)
    assert sc.approx_ratio == pytest.approx(1.0)
    assert sc.optimum_rank == 1


def test_run_pipeline_mock_backend(knap):
    p, q, emb = knap
    s = Schedule.default_for(emb)
    res = run_pipeline(p, q, emb, s, backend="mock", shots=200, seed=1)
    assert sum(res.counts.values()) == 200
    assert res.score.best_objective is not None
    assert res.score.feasible_rate_repaired == 1.0


def test_scan_loop_runs(knap):
    p, q, emb = knap
    base = Schedule.default_for(emb)
    calls = []

    def ev(s):
        calls.append(s.duration)
        return float(s.duration), {}

    out = scan(base, {"duration": [2000, 3000]}, ev, limits=LIM, verbose=False)
    assert [c for c in calls] == [2000, 3000] and out[0].params["duration"] == 3000


def test_mock_sampler_prefers_low_energy():
    p = max_independent_set([(0, 1), (1, 2)], 3)
    q = p.to_qubo().qubo
    emb = embed_qubo(q, LIM, n_starts=1)
    counts = mock_boltzmann(emb, shots=500, temperature=0.3)
    best = max(counts, key=counts.get)
    assert best == "101"
