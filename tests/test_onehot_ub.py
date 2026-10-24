"""Gate-1 formulation ``onehot-ub`` (docs/MODEL.md): algebra, signs, soundness."""
import glob
import itertools

import numpy as np

from cvrp.common import Instance
from cvrp.model import brute_force_check, get_formulation, make_toy_instance


def _literal_energy(f, x):
    """E(x) straight from MODEL.md eq. (M4): objective + A*sum_i(1-sum_k x_ik)^2 + 2 lam2 sum_k sum_{i<j} d_i d_j x_ik x_jk."""
    inst, N, K = f.inst, f.N, f.K
    A, lam2 = f.weights["A"], f.weights["lam2"]
    X = np.asarray(x).reshape(N, K)
    D, d = f.D, inst.demands
    e = 0.0
    for k in range(K):
        for i, j in itertools.combinations(range(N), 2):
            e += D[i, j] * X[i, k] * X[j, k] + 2.0 * lam2 * d[i] * d[j] * X[i, k] * X[j, k]
    e += A * sum((1 - X[i].sum()) ** 2 for i in range(N))
    return e


def test_Q_matches_literal_formula_on_random_bitstrings():
    inst = Instance.load(sorted(glob.glob("instances/S_seed*.json"))[0])
    f = get_formulation("onehot-ub", inst)
    rng = np.random.default_rng(0)
    for x in rng.integers(0, 2, size=(500, f.n_vars)):
        assert abs(f.energy(x) - _literal_energy(f, x)) < 1e-9


def test_device_realisable_signs_on_every_shipped_instance():
    for p in sorted(glob.glob("instances/*_seed*.json")):
        f = get_formulation("onehot-ub", Instance.load(p))
        h, J = f.linear(), f.couplings()
        assert np.allclose(h, h[0]), p                 # uniform linear term -> one global detuning
        assert h[0] < 0, p                             # delta_f > 0 favours |r> (exactly-one needs excitations)
        assert (J >= 0).all(), p                       # every coupling positive (repulsive C6/r^6)


def test_count_and_support_are_rook_graph():
    for N, K, Q in [(4, 2, 7), (10, 3, 20), (15, 4, 30)]:
        inst = make_toy_instance({4: "S", 10: "M", 15: "L"}[N], 0)
        f = get_formulation("onehot-ub", inst)
        assert f.n_vars == N * K == type(f).count_vars(N, K, Q)
        J = f.couplings()
        n_pairs = int((np.triu(J, 1) > 0).sum())
        assert n_pairs == N * K * (K - 1) // 2 + K * N * (N - 1) // 2, (N, K, n_pairs)


def test_balance_identity_on_one_hot_strings():
    """On one-hot strings E = f + lam2*(sum_k L_k^2 - sum_i d_i^2) (offset = A*N cancels the penalty)."""
    inst = Instance.load("instances/S_seed00.json")
    f = get_formulation("onehot-ub", inst)
    lam2 = f.weights["lam2"]
    for a in itertools.product(range(inst.K), repeat=inst.N):
        x = f.encode(a)
        want = f.objective(a) + lam2 * (f.balance(a) - sum(d * d for d in inst.demands))
        assert abs(f.energy(x) - want) < 1e-9


def test_sound_on_all_shipped_tier_S_instances():
    for p in sorted(glob.glob("instances/S_seed*.json")):
        r = brute_force_check(get_formulation("onehot-ub", Instance.load(p)), margins=False)
        assert r.sound, p


def test_A_rule_makes_every_argmin_one_hot_even_when_capacity_binds():
    """toy seed 14: capacity binds and lam2 is too weak to force feasibility (MODEL.md §6),
    but the argmin is still one-hot (decodable): the A rule is independent of lam2."""
    f = get_formulation("onehot-ub", make_toy_instance("S", 14))
    r = brute_force_check(f, margins=False)
    assert not r.sound
    assert all(f.decode(x) is not None for x in [f.encode(r.argmin_assignment)])
    assert r.argmin_assignment is not None


def test_tight_A_rule_every_argmin_is_one_hot():
    """onehot-ub-tight: A = R_w/K (1+1%) still makes every energy argmin one-hot (MODEL.md §6 proof),
    on the 5 shipped S instances, 20 toy S seeds, and random non-negative W at K = 2, 3, 4 (N = 4)."""
    from cvrp.common import all_bitstrings
    insts = [Instance.load(p) for p in sorted(glob.glob("instances/S_seed*.json"))] + [make_toy_instance("S", s) for s in range(20)]
    for inst in insts:
        f = get_formulation("onehot-ub-tight", inst)
        assert f.weights["A"] < get_formulation("onehot-ub", inst).weights["A"]
        X = all_bitstrings(f.n_vars)
        E = f.energies(X)
        for x in X[np.isclose(E, E.min())]:
            assert f.decode(x) is not None, inst.name
    rng = np.random.default_rng(1)
    for K in (2, 3, 4):
        for _ in range(5):
            N = 4
            W = np.triu(rng.uniform(0, 1, (N, N)) * (rng.uniform(0, 1, (N, N)) < 0.7), 1)
            W = W + W.T
            Rw = W.sum(1).max()
            A = Rw / K * 1.01
            n = N * K
            Q = np.zeros((n, n))
            for k in range(K):
                for i in range(N):
                    Q[i * K + k, i * K + k] -= A
                    for l in range(k + 1, K):
                        Q[i * K + k, i * K + l] += A
                        Q[i * K + l, i * K + k] += A
                    for j in range(i + 1, N):
                        Q[i * K + k, j * K + k] += W[i, j] / 2
                        Q[j * K + k, i * K + k] += W[i, j] / 2
            X = np.array(list(itertools.product([0, 1], repeat=n)))
            E = np.einsum("mi,ij,mj->m", X, Q, X)
            for x in X[np.isclose(E, E.min())]:
                assert (x.reshape(N, K).sum(1) == 1).all()


def test_random_onehot_control_is_assignment_valid():
    from cvrp.metrics import random_onehot_counts
    f = get_formulation("onehot-ub", Instance.load("instances/S_seed00.json"))
    counts = random_onehot_counts(f, 300, 0)
    assert sum(counts.values()) == 300
    for s in counts:
        assert f.decode(np.array([int(c) for c in s])) is not None
