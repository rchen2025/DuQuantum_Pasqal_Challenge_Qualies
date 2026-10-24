"""THROWAWAY textbook formulation (brief Derive 4.1) -- for plumbing only.

Two formulations are defined here, both using one-hot assignment variables
x_{ik} = 1  <=>  customer i rides on vehicle k   (index i*K + k):

* ``AssignmentOnly``  ("assignment-only"): the N*K assignment variables and the
  clustering objective, no constraints at all.  Exists only for the first row
  of the variable-count table; its energy minimum is the all-zero string.
* ``TextbookOneHotSlack`` ("textbook"): adds the two textbook penalties with
  binary-encoded slack, B = ceil(log2(Q+1)) bits per vehicle (index
  N*K + k*B + b), so n = N*K + K*B  (S: 14, M: 45, L: 80).

Objective ("closeness" measure, brief §5.1): total pairwise distance inside
each cluster,  f(x) = sum_k sum_{i<j} d_ij x_ik x_jk.  It is additive over
clusters (``cluster_cost``), so the exact set-partition DP applies.

Constraints -> penalties (full algebra in docs/DERIVATIONS.md §3):
  exactly-one  sum_k x_ik = 1           ->  A * sum_i (sum_k x_ik - 1)^2
  capacity     sum_i d_i x_ik <= Q      ->  B * sum_k (sum_i d_i x_ik + sum_b 2^b s_kb - Q)^2
Slack max = 2^B - 1 >= Q, so every load 0..Q has a slack value that zeroes
the capacity penalty; slack values above Q are simply never at the minimum.

Default weights: A = B = (sum_{i<j} d_ij) + 1, i.e. one more than the largest
value the objective can take (everything on one vehicle).  Any infeasible
bitstring pays at least min(A, B) * 1^2 while the objective is >= 0, so its
energy exceeds every feasible energy -> provably sound, verified by
``cvrp.model.brute_force_check``.  Tighter rules exist (DERIVATIONS.md §3.4).

WHY THIS IS NOT DEVICE-REALISABLE on Pulser's AnalogDevice (global detuning):
* The linear terms are non-uniform.  h(x_ik) = -A + B (d_i^2 - 2 Q d_i)
  depends on the customer's demand d_i, and h(s_kb) = B (4^b - 2 Q 2^b)
  depends on the bit position b.  A single global delta can realise only ONE
  value of h for all atoms (DERIVATIONS.md §2.3), so the register would see
  the *mean* linear term and the capacity bookkeeping is lost.
* The capacity penalty couples every pair (i, j) on the same vehicle with
  J = 2 B d_i d_j and every slack bit to every customer of its vehicle
  (J = 2 B d_i 2^b) and to every other slack bit (J = 2 B 2^b 2^b'): a dense,
  rank-1-like pattern per vehicle that 1/r^6 in a plane reproduces badly.
* What IS fine: every coupling is >= 0 (objective d_ij >= 0, one-hot 2A > 0,
  capacity 2 B * products of positive coefficients), checked in the tests.
"""
from __future__ import annotations

import math
from typing import FrozenSet, List, Optional

import numpy as np

from cvrp.common import Assignment, Formulation, Instance
from cvrp.model import register


class _PairwiseDistanceClustering(Formulation):
    """Shared machinery: one-hot x_{ik} at index i*K + k and the total
    pairwise within-cluster distance objective."""

    #: human label for the Derive 4.1 table (scripts/variable_count.py)
    table_label: str = ""
    #: closed-form variable count for the same table
    var_formula: str = ""

    def __init__(self, inst: Instance, **weights: float) -> None:
        super().__init__(inst, **weights)
        self.N, self.K = inst.N, inst.K
        self.D = inst.customer_dist()
        self._Q: Optional[np.ndarray] = None

    # ---- indices ---------------------------------------------------------- #
    def xidx(self, i: int, k: int) -> int:
        return i * self.K + k

    @property
    def n_assign(self) -> int:
        return self.N * self.K

    # ---- objective -------------------------------------------------------- #
    def cluster_cost(self, members: FrozenSet[int]) -> float:
        m = sorted(members)
        return float(sum(self.D[a, b] for ai, a in enumerate(m) for b in m[ai + 1:]))

    def objective(self, a: Assignment) -> float:
        groups = {}
        for i, k in enumerate(a):
            groups.setdefault(int(k), []).append(i)
        return float(sum(self.cluster_cost(frozenset(g)) for g in groups.values()))

    def objective_upper_bound(self) -> float:
        """sum_{i<j} d_ij: the objective when every customer shares one vehicle."""
        return float(np.triu(self.D, 1).sum())

    def _objective_Q(self, n: int) -> np.ndarray:
        """Q block of  sum_k sum_{i<j} d_ij x_ik x_jk  (symmetric: d_ij/2 each side)."""
        Q = np.zeros((n, n))
        for k in range(self.K):
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    Q[self.xidx(i, k), self.xidx(j, k)] += 0.5 * self.D[i, j]
                    Q[self.xidx(j, k), self.xidx(i, k)] += 0.5 * self.D[i, j]
        return Q

    # ---- decode ----------------------------------------------------------- #
    def decode(self, x: np.ndarray) -> Optional[Assignment]:
        X = np.asarray(x)[: self.n_assign].reshape(self.N, self.K)
        if not (X.sum(axis=1) == 1).all():
            return None
        return tuple(int(k) for k in X.argmax(axis=1))

    def decode_many(self, X: np.ndarray) -> np.ndarray:
        """(M, n) bitstrings -> (M, N) assignments, rows of -1 where invalid."""
        Xa = np.asarray(X)[:, : self.n_assign].reshape(-1, self.N, self.K)
        ok = (Xa.sum(axis=2) == 1).all(axis=1)
        A = Xa.argmax(axis=2).astype(int)
        A[~ok] = -1
        return A

    def var_names(self) -> List[str]:
        return [f"x[{i},{k}]" for i in range(self.N) for k in range(self.K)]


@register
class AssignmentOnly(_PairwiseDistanceClustering):
    """N*K one-hot assignment variables, objective only, NO constraints.
    Row 1 of the Derive 4.1 table.  Its minimum is x = 0 (nobody served), so
    ``brute_force_check`` reports it as not sound -- by design."""

    name = "assignment-only"
    table_label = "Assignment variables only, no constraints"
    var_formula = "N·K"

    @property
    def n_vars(self) -> int:
        return self.n_assign

    @classmethod
    def count_vars(cls, N: int, K: int, Q: int) -> int:
        return N * K

    def Q(self) -> np.ndarray:
        if self._Q is None:
            self._Q = self._objective_Q(self.n_vars)
        return self._Q


@register
class TextbookOneHotSlack(_PairwiseDistanceClustering):
    """One-hot assignment + binary-encoded slack, both constraints as squared
    penalties (module docstring; DERIVATIONS.md §3).  Weights: ``A`` (exactly-one),
    ``B`` (capacity); default A = B = sum_{i<j} d_ij + 1."""

    name = "textbook"
    table_label = "+ binary-encoded slack (textbook)"
    var_formula = "N·K + K·⌈log2(Q+1)⌉"

    @staticmethod
    def slack_bits(Q: int) -> int:
        """B = ceil(log2(Q+1)): smallest bit count with 2^B - 1 >= Q."""
        return int(math.ceil(math.log2(Q + 1)))

    def __init__(self, inst: Instance, **weights: float) -> None:
        super().__init__(inst, **weights)
        self.n_slack_bits = self.slack_bits(inst.Q)
        default = self.objective_upper_bound() + 1.0
        self.weights.setdefault("A", default)
        self.weights.setdefault("B", default)
        unknown = set(self.weights) - {"A", "B"}
        if unknown:
            raise ValueError(f"unknown weights {unknown}; textbook has A (exactly-one) and B (capacity)")
        self.Q()  # build eagerly so ``offset`` is set before any energy() call

    # ---- indices ---------------------------------------------------------- #
    def sidx(self, k: int, b: int) -> int:
        return self.n_assign + k * self.n_slack_bits + b

    @property
    def n_vars(self) -> int:
        return self.n_assign + self.K * self.n_slack_bits

    @classmethod
    def count_vars(cls, N: int, K: int, Q: int) -> int:
        return N * K + K * cls.slack_bits(Q)

    def var_names(self) -> List[str]:
        return super().var_names() + [f"s[{k},{b}]" for k in range(self.K) for b in range(self.n_slack_bits)]

    # ---- QUBO ------------------------------------------------------------- #
    def Q(self) -> np.ndarray:
        if self._Q is not None:
            return self._Q
        n = self.n_vars
        A, B = self.weights["A"], self.weights["B"]
        N, K, Qcap = self.N, self.K, self.inst.Q
        d = self.inst.demands
        Q = self._objective_Q(n)
        offset = 0.0

        # exactly-one:  A sum_i (sum_k x_ik - 1)^2
        #   = A sum_i [ -sum_k x_ik  +  2 sum_{k<l} x_ik x_il  +  1 ]      (x^2 = x)
        for i in range(N):
            for k in range(K):
                Q[self.xidx(i, k), self.xidx(i, k)] += -A
                for l in range(k + 1, K):
                    Q[self.xidx(i, k), self.xidx(i, l)] += A      # 2A x_ik x_il split symmetrically
                    Q[self.xidx(i, l), self.xidx(i, k)] += A
            offset += A

        # capacity:  B sum_k ( sum_m c_m z_m - Q )^2   with z = (x_0k..x_{N-1,k}, s_k0..s_k,B-1),
        #   c = (d_0..d_{N-1}, 1, 2, .., 2^(B-1))
        #   = B sum_k [ sum_m (c_m^2 - 2 Q c_m) z_m  +  2 sum_{m<m'} c_m c_m' z_m z_m'  +  Q^2 ]
        for k in range(K):
            idx = [self.xidx(i, k) for i in range(N)] + [self.sidx(k, b) for b in range(self.n_slack_bits)]
            c = [float(d[i]) for i in range(N)] + [float(2 ** b) for b in range(self.n_slack_bits)]
            for m, (im, cm) in enumerate(zip(idx, c)):
                Q[im, im] += B * (cm * cm - 2.0 * Qcap * cm)
                for im2, cm2 in zip(idx[m + 1:], c[m + 1:]):
                    Q[im, im2] += B * cm * cm2                     # 2B c c' split symmetrically
                    Q[im2, im] += B * cm * cm2
            offset += B * Qcap * Qcap

        self.offset = offset
        self._Q = Q
        return Q

    # ---- helpers for docs/tests ------------------------------------------- #
    def slack_value(self, x: np.ndarray, k: int) -> int:
        x = np.asarray(x)
        return int(sum((1 << b) * int(x[self.sidx(k, b)]) for b in range(self.n_slack_bits)))

    def encode(self, a: Assignment) -> np.ndarray:
        """Assignment -> bitstring with the slack set to Q - load (zero penalty
        when the assignment is feasible)."""
        x = np.zeros(self.n_vars, dtype=np.int8)
        for i, k in enumerate(a):
            x[self.xidx(i, k)] = 1
        for k, load in enumerate(self.inst.loads(tuple(a))):
            s = max(0, self.inst.Q - load)
            for b in range(self.n_slack_bits):
                x[self.sidx(k, b)] = (s >> b) & 1
        return x
