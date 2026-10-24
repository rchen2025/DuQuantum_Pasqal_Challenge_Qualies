"""Tests that need the real emulators / qubo-solver. They are skipped where
those are not installed, so run `pytest -v` in the hackathon venv to see them
actually execute (STATUS.md lists what is still untested)."""
import numpy as np
import pytest

from qopt.classical import solve_best
from qopt.embed import KNOWN_DEVICES, embed_qubo
from qopt.model import max_independent_set
from qopt.run import run_pipeline
from qopt.schedule import Schedule

LIM = KNOWN_DEVICES["WeightedAnalogDevice"]


@pytest.fixture(scope="module")
def mis5():
    p = max_independent_set([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5)
    q = p.to_qubo().qubo
    emb = embed_qubo(q, LIM, n_starts=3)
    return p, q, emb


@pytest.mark.slow
def test_qutip_backend_finds_mis(mis5):
    pytest.importorskip("pulser_simulation")
    p, q, emb = mis5
    s = Schedule.default_for(emb, duration=3000)
    res = run_pipeline(p, q, emb, s, backend="qutip", shots=300, classical=solve_best(q.Q))
    assert sum(res.counts.values()) == 300
    assert res.score.approx_ratio == pytest.approx(1.0)
    assert res.score.p_optimal_repaired > 0.2


@pytest.mark.slow
def test_emu_sv_backend_runs(mis5):
    pytest.importorskip("emu_sv")
    p, q, emb = mis5
    s = Schedule.default_for(emb, duration=2000)
    res = run_pipeline(p, q, emb, s, backend="emu-sv", shots=200)
    assert sum(res.counts.values()) == 200 and res.score.best_objective is not None


@pytest.mark.slow
def test_emu_mps_backend_runs(mis5):
    pytest.importorskip("emu_mps")
    p, q, emb = mis5
    s = Schedule.default_for(emb, duration=2000)
    res = run_pipeline(p, q, emb, s, backend="emu-mps", shots=200, max_bond_dim=64)
    assert sum(res.counts.values()) == 200


@pytest.mark.slow
def test_qubo_solver_baseline(mis5):
    pytest.importorskip("qubosolver")
    from qopt.baseline import run_qubo_solver
    p, q, _ = mis5
    base = run_qubo_solver(p, q, shots=200, mode="functional", classical=solve_best(q.Q))
    assert sum(base.counts.values()) == 200
    assert base.score.best_objective is not None
    assert base.embedding_rel_error is not None
    sol = run_qubo_solver(p, q, shots=200, mode="solver", classical=solve_best(q.Q))
    assert sol.score.best_objective is not None


def test_cloud_submit_requires_env(monkeypatch):
    pytest.importorskip("pasqal_cloud")
    from qopt.run import _connection
    for k in ("PASQAL_USERNAME", "PASQAL_PASSWORD", "PASQAL_PROJECT_ID"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        _connection()
