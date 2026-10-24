"""cvrp.baseline: pure-python parts (Q convention, counts conversion, labels) run anywhere;
anything that needs torch/qubo-solver is a SkipTest without them and a real smoke test on
the Windows venv (dry-run embedding + compile, classical SA on S_seed00)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvrp import baseline as B  # noqa: E402
from cvrp.common import Instance  # noqa: E402
from cvrp.model import get_formulation  # noqa: E402


def _torch_stack():
    try:
        import torch  # noqa: F401
        import qubosolver  # noqa: F401
        import qoolqit  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(f"needs torch + qubo-solver (Windows venv): {e}")


def _inst():
    return Instance.load(ROOT / "instances/S_seed00.json")


# --------------------------------------------------------------------------- #
# Q convention
# --------------------------------------------------------------------------- #
def test_qubosolver_cost_pins_their_convention():
    # docstring example of qubosolver/utils/_costs.py::quadratic_cost: Q=[[1,-1],[-1,2]]
    Q = np.array([[1.0, -1.0], [-1.0, 2.0]])
    assert B.qubosolver_cost(Q, [1, 0]) == 1.0
    assert B.qubosolver_cost(Q, [0, 1]) == 2.0
    # both triangles count: x=[1,1] -> 1 + 2 - 1 - 1 = 1  (an upper-triangular reading would give 2)
    assert B.qubosolver_cost(Q, [1, 1]) == 1.0


def test_matrix_conversion_round_trips_energy_on_random_bitstrings():
    rng = np.random.default_rng(3)
    for n in (3, 8, 14):
        Q = rng.normal(size=(n, n)) * 10
        Q = 0.5 * (Q + Q.T)
        offset = float(rng.normal() * 100)
        M = B.to_qubosolver_matrix(Q)
        assert M.shape == (n, n) and np.array_equal(M, Q)
        for _ in range(50):
            x = rng.integers(0, 2, n)
            ours = float(x @ Q @ x + offset)
            theirs = B.qubosolver_cost(M, x)
            assert abs(B.energy_from_qubosolver_cost(theirs, offset) - ours) < 1e-9


def test_matrix_conversion_matches_formulation_energy():
    inst = _inst()
    for name in ("onehot-ub", "textbook"):
        f = get_formulation(name, inst)
        M = B.to_qubosolver_matrix(f.Q())
        rng = np.random.default_rng(1)
        for _ in range(40):
            x = rng.integers(0, 2, f.n_vars)
            assert abs(B.qubosolver_cost(M, x) + f.offset - f.energy(x)) < 1e-6 * max(1.0, abs(f.energy(x)))


def test_matrix_conversion_refuses_triangular_or_asymmetric():
    Q = np.triu(np.ones((3, 3)))
    try:
        B.to_qubosolver_matrix(Q)
        assert False, "upper-triangular Q must be refused (couplings would be halved)"
    except B.ConventionError:
        pass
    try:
        B.to_qubosolver_matrix(np.ones((2, 3)))
        assert False
    except B.ConventionError:
        pass


# --------------------------------------------------------------------------- #
# bitstrings / counts
# --------------------------------------------------------------------------- #
def test_counts_from_final_bitstrings_identity_order():
    fb = {"0101": 3, "1100": 2}
    out = B.counts_from_final_bitstrings(fb, 4, qubit_ids=("0", "1", "2", "3"))
    assert out == {"0101": 3, "1100": 2}
    assert B.counts_from_final_bitstrings(fb, 4) == out


def test_counts_from_final_bitstrings_reorders_permuted_register():
    # pulser register ids (2, 0, 1): bit 0 of the string is qubit "2" = variable 2, etc.
    fb = {"100": 5, "011": 1}
    out = B.counts_from_final_bitstrings(fb, 3, qubit_ids=("2", "0", "1"))
    # "100": x2=1, x0=0, x1=0 -> "001";  "011": x2=0, x0=1, x1=1 -> "110"
    assert out == {"001": 5, "110": 1}
    # duplicates after reordering are merged
    out2 = B.counts_from_final_bitstrings({"10": 1, "01": 2}, 2, qubit_ids=("1", "0"))
    assert out2 == {"01": 1, "10": 2}


def test_counts_from_final_bitstrings_rejects_bad_input():
    for fb, n, ids in (({"010": 1}, 4, None), ({"01a": 1}, 3, None), ({"01": 1}, 2, ("0", "0")),
                       ({"01": 1}, 2, ("a", "b")), ({}, 2, None)):
        try:
            B.counts_from_final_bitstrings(fb, n, qubit_ids=ids)
            assert False, (fb, n, ids)
        except B.ConventionError:
            pass


class _Cand:
    def __init__(self, s, c, cost=0.0):
        self.string, self.count, self.cost = s, c, cost


def test_counts_from_solution_duck_typed():
    sol = [_Cand("0110", 7), _Cand("0001", 0), _Cand("0110", 1)]
    assert B.counts_from_solution(sol, 4) == {"0110": 8, "0001": 1}
    try:
        B.counts_from_solution([_Cand("011", 1)], 4)
        assert False
    except B.ConventionError:
        pass


# --------------------------------------------------------------------------- #
# labels / choices
# --------------------------------------------------------------------------- #
def test_method_labels():
    assert B.method_label("onehot-ub", "quantum") == "onehot-ub+qubosolver-quantum"
    assert B.method_label("onehot-ub", "quantum", repair="qs-postprocess") == "onehot-ub+qubosolver-quantum+repair:qs-postprocess"
    assert B.method_label("onehot-ub", "quantum", pipeline="default", repair="qubosolver-builtin") == \
        "onehot-ub+qubosolver-quantum-default+repair:qubosolver-builtin"
    assert B.method_label("onehot-ub", "classical", classical_algorithm="cplex") == "onehot-ub+qubosolver-cplex"
    assert B.method_label("onehot-ub", "classical", classical_algorithm="simulated_annealing") == "onehot-ub+qubosolver-sa"
    assert B.method_label("textbook", "classical", classical_algorithm="tabu_search") == "textbook+qubosolver-tabu"
    # record file names stay distinct from our pipeline's and from each other
    from cvrp.run import record_path
    names = {record_path(Path("."), {"instance": "S_seed00", "method": m, "seed": 0}).name
             for m in ("onehot-ub+qutip", "onehot-ub+qubosolver-quantum", "onehot-ub+qubosolver-sa",
                       "onehot-ub+qubosolver-quantum+repair:qs-postprocess")}
    assert len(names) == 4


def test_classical_algorithm_choice():
    algo = B.choose_classical_algorithm("auto")
    assert algo in ("cplex", "simulated_annealing")
    assert (algo == "cplex") == B.cplex_available()
    assert B.choose_classical_algorithm("simulated_annealing") == "simulated_annealing"
    try:
        B.choose_classical_algorithm("annealing")
        assert False
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# fidelity in our convention
# --------------------------------------------------------------------------- #
def test_coupling_fidelity_exact_when_proportional():
    rng = np.random.default_rng(0)
    J = np.abs(rng.normal(size=(6, 6))); J = 0.5 * (J + J.T); np.fill_diagonal(J, 0.0)
    J[0, 1] = J[1, 0] = 0.0                      # one zero coupling -> spurious term defined
    U = 3.7 * J
    U[0, 1] = U[1, 0] = 0.37                     # spurious coupling 0.1 in J units
    fid = B.coupling_fidelity(U, J)
    assert abs(fid["alpha"] - 3.7) < 1e-12
    assert fid["rel_frobenius_offdiag"] < 0.2 and fid["worst_pair_rel_error"] < 1e-12
    assert abs(fid["max_spurious_coupling"] - 0.1) < 1e-12 and fid["spurious_pair"] == [0, 1]
    # J with the realised value halved on one pair -> that pair is the worst
    U2 = 2.0 * J; U2[2, 3] = U2[3, 2] = J[2, 3]
    fid2 = B.coupling_fidelity(U2, J)
    assert fid2["worst_pair"] == [2, 3] and fid2["worst_pair_rel_error"] > 0.3


# --------------------------------------------------------------------------- #
# needs torch + qubo-solver (skip here, real smoke test on Windows)
# --------------------------------------------------------------------------- #
def test_quantum_dry_run_builds_embedding_and_compiled_sequence():
    _torch_stack()
    inst = _inst()
    rec = B.run_quantum(inst, "onehot-ub", shots=10, seed=0, dry_run=True, verbose=False)
    assert rec["counts"] is None and rec["method"] == "onehot-ub+qubosolver-quantum"
    assert rec["atoms"] == rec["n_vars"] == 8 and rec["dmm"] is False
    assert rec["device"] == "AnalogDevice"
    emb, pulse = rec["embedding"], rec["pulse"]
    assert emb["embedder"] == "blade" and len(emb["coords_um"]) == 8
    assert emb["fidelity"]["rel_frobenius_offdiag"] is not None
    assert pulse["pulser_validated"] is True and pulse["duration_ns"] <= 6000
    assert rec["backend_params"]["convention_check"]["max_abs_diff"] <= rec["backend_params"]["convention_check"]["tol"]
    assert rec["sequence_duration_ns"] == pulse["duration_ns"]


def test_classical_sa_record_on_S_seed00():
    _torch_stack()
    inst = _inst()
    rec = B.run_classical(inst, "onehot-ub", seed=0, algorithm="simulated_annealing", sa_iters=2000, verbose=False)
    assert rec["method"] == "onehot-ub+qubosolver-sa"
    assert sum(rec["counts"].values()) == 1 and len(next(iter(rec["counts"]))) == 8
    m = rec["metrics_raw"]
    assert m["shots"] == 1 and m["feasible_rate"] in (0.0, 1.0)
    if m["feasible_rate"] == 0:
        assert m["best_model_gap"] is None
    assert rec["backend_params"]["algorithm"] == "simulated_annealing"
