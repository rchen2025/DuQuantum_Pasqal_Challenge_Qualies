import numpy as np
import pytest

from qopt import classical
from qopt.model import knapsack


def _random_qubo(n, seed):
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(n, n))
    return 0.5 * (Q + Q.T)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_milp_matches_brute_force(seed):
    Q = _random_qubo(10, seed)
    bf = classical.brute_force(Q)
    ml = classical.milp(Q)
    assert ml.energy == pytest.approx(bf.energy, abs=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_heuristics_reach_optimum_on_small(seed):
    Q = _random_qubo(12, seed)
    bf = classical.brute_force(Q)
    sa = classical.simulated_annealing(Q, seed=seed, n_sweeps=300, n_restarts=4)
    tb = classical.tabu_search(Q, seed=seed, n_iter=500)
    assert sa.energy == pytest.approx(bf.energy, abs=1e-6)
    assert tb.energy == pytest.approx(bf.energy, abs=1e-6)


def test_solve_best_and_exact_dispatch():
    b = knapsack([3, 4, 5, 6], [2, 3, 4, 5], 7).to_qubo()
    r = classical.solve_best(b.qubo.Q)
    assert r.exact and b.qubo.project(r.x).tolist() == [0, 1, 1, 0]
    assert classical.solve_exact(b.qubo.Q).energy == pytest.approx(r.energy)


def test_greedy_descent_is_local_min():
    Q = _random_qubo(8, 3)
    x = classical.greedy_descent(Q, np.zeros(8))
    d = classical._flip_deltas(Q, x.astype(float))
    assert (d >= -1e-9).all()
