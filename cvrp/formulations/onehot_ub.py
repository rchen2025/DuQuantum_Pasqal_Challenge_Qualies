"""Gate-1 formulation ``onehot-ub``: one-hot assignment, NO slack, capacity by
unbalanced penalization reduced to a load-balance term.  Derivation:
docs/MODEL.md (full algebra) and docs/DERIVATIONS.md §4 (summary).

Variables (same as the textbook, DERIVATIONS §3.1): x_{ik} = 1 <=> customer i
rides on vehicle k, index i*K + k, n = N*K  (S: 8, M: 30, L: 60).

Objective (3.1): f(x) = sum_k sum_{i<j} d_ij x_ik x_jk  (additive over clusters).

Constraints:
  [=] exactly-one  sum_k x_ik = 1    ->  A * sum_i (1 - sum_k x_ik)^2
                                       = A*N + sum_i [ -A sum_k x_ik + 2A sum_{k<l} x_ik x_il ]
  [<=] capacity    L_k <= Q          ->  unbalanced penalization (Montanez-Barrera et al. 2024):
                                         sum_k [ -lam1 (Q - L_k) + lam2 (Q - L_k)^2 ],  L_k = sum_i d_i x_ik.

Expanding the capacity term gives, per variable x_ik, a linear coefficient
c_i = d_i (lam1 - 2 lam2 Q + lam2 d_i) that depends on the customer only, and
couplings +2 lam2 d_i d_j between customers of the same vehicle.  On every
one-hot string  sum_k c_i x_ik = c_i  is a CONSTANT, so the whole set of
per-customer linear terms (lam1 included) is invisible to the model: it only
changes the energy of strings that already violate exactly-one.  We drop it
(MODEL.md §3b, the "one-hot identity"), which leaves

  E(x) = f(x) + A sum_i (1 - sum_k x_ik)^2 + 2 lam2 sum_k sum_{i<j} d_i d_j x_ik x_jk
       = f(x) + A sum_i (1 - sum_k x_ik)^2 + lam2 ( sum_k L_k^2 - sum_i d_i^2 )   on one-hot strings.

So what "capacity" becomes is a LOAD-BALANCE term lam2 * sum_k L_k^2: it is
minimised by the most even split of demand across vehicles.  It does not
guarantee L_k <= Q (brief Pitfall 9.1 applies; feasibility rate is reported),
but for K = 2 its minimiser over feasible-splittable demand is always
capacity-feasible (MODEL.md §3b), and the brute-force checker certifies the
weights on every tier-S instance.

Device realisability (DERIVATIONS §2.3): every linear term is -A (uniform ->
one global detuning, delta_f = alpha*A > 0); every coupling is >= 0
(d_ij >= 0, 2A > 0, 2 lam2 d_i d_j > 0); the support is the N x K rook's graph
(row = the K atoms of one customer, column = the N atoms of one vehicle).

Weights (MODEL.md §6):
  A    = max_i sum_{j != i} w_ij + 1,  w_ij = d_ij + 2 lam2 d_i d_j
         (provably sound: from any non-one-hot string a one-hot repair step
         lowers the energy, so every energy argmin is one-hot).
  lam2 = kappa * R / (2 Q),  R = max_i sum_{j != i} d_ij,  kappa = 0.15.
         R/(2Q) is the "constraint-dominates" scale (the largest objective change
         one reassignment can cause, over the smallest balance change an
         overload-removing move makes, 2*d_min*Q with d_min = 1 on every tier);
         at kappa = 1 the balance term overrides the objective (27/40 toy S seeds
         sound), at kappa = 0 capacity is unenforced (34/40 sound = the seeds
         where it does not bind).  kappa = 0.15 is the scan optimum on 40 toy
         tier-S seeds (35/40 sound, 37/40 argmin feasible, 5/5 shipped S sound),
         MODEL.md §6.  Heuristic; the checker's half/double/zero margins are
         reported per instance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from cvrp.common import Instance
from cvrp.formulations.textbook import _PairwiseDistanceClustering
from cvrp.model import register


@register
class OneHotUnbalanced(_PairwiseDistanceClustering):
    name = "onehot-ub"
    KAPPA = 0.15      # tie-breaker fraction of the constraint-dominates scale (MODEL.md §6)
    #: how the exactly-one weight A is set from W (w_ij = d_ij + 2 lam2 d_i d_j), MODEL.md §6:
    #:   "row_sum"        A = max_i sum_{j!=i} w_ij + 1            (original, sufficient, conservative)
    #:   "row_sum_over_k" A = (max_i sum_{j!=i} w_ij) / K * (1 + 0.01)  (tight: every target-energy
    #:                     global minimiser is one-hot; proof MODEL.md §6; registered as `onehot-ub-tight`)
    PENALTY_RULE = "row_sum"
    table_label = "One-hot, no slack, capacity as load-balance (unbalanced penalization)"
    var_formula = "N·K"

    @classmethod
    def A_rule(cls, W: np.ndarray, K: int) -> float:
        Rw = float(W.sum(axis=1).max())
        if cls.PENALTY_RULE == "row_sum":
            return Rw + 1.0
        if cls.PENALTY_RULE == "row_sum_over_k":
            return Rw / K + max(1e-6, 0.01 * Rw / K)
        raise ValueError(f"unknown PENALTY_RULE {cls.PENALTY_RULE!r}")

    def __init__(self, inst: Instance, **weights: float) -> None:
        super().__init__(inst, **weights)
        unknown = set(self.weights) - {"A", "lam2"}
        if unknown:
            raise ValueError(f"unknown weights {unknown}; onehot-ub has A (exactly-one) and lam2 (load balance)")
        d = np.asarray(inst.demands, float)
        R = float(max(self.D[i].sum() for i in range(self.N)))        # largest row sum of d_ij
        self.weights.setdefault("lam2", self.KAPPA * R / (2.0 * inst.Q))
        lam2 = self.weights["lam2"]
        W = self.D + 2.0 * lam2 * np.outer(d, d)
        np.fill_diagonal(W, 0.0)
        self.weights.setdefault("A", self.A_rule(W, self.K))
        self.Q()

    def describe(self):
        return {**super().describe(), "penalty_rule": self.PENALTY_RULE, "kappa": self.KAPPA}

    @property
    def n_vars(self) -> int:
        return self.n_assign

    @classmethod
    def count_vars(cls, N: int, K: int, Q: int) -> int:
        return N * K

    def Q(self) -> np.ndarray:
        if self._Q is not None:
            return self._Q
        n = self.n_vars
        A, lam2 = self.weights["A"], self.weights["lam2"]
        N, K = self.N, self.K
        d = [float(v) for v in self.inst.demands]
        Q = self._objective_Q(n)                                  # f(x): d_ij/2 on same-vehicle pairs
        offset = 0.0
        # exactly-one: A sum_i (1 - sum_k x_ik)^2 = A N + sum_i [ -A sum_k x_ik + 2A sum_{k<l} x_ik x_il ]
        for i in range(N):
            for k in range(K):
                Q[self.xidx(i, k), self.xidx(i, k)] += -A
                for l in range(k + 1, K):
                    Q[self.xidx(i, k), self.xidx(i, l)] += A      # 2A split symmetrically
                    Q[self.xidx(i, l), self.xidx(i, k)] += A
            offset += A
        # capacity -> load balance: 2 lam2 sum_k sum_{i<j} d_i d_j x_ik x_jk
        #   (= lam2 sum_k L_k^2 - lam2 sum_i d_i^2 on one-hot strings; the dropped
        #    per-customer linear terms are constant there, see module docstring)
        for k in range(K):
            for i in range(N):
                for j in range(i + 1, N):
                    Q[self.xidx(i, k), self.xidx(j, k)] += lam2 * d[i] * d[j]   # 2 lam2 d_i d_j split
                    Q[self.xidx(j, k), self.xidx(i, k)] += lam2 * d[i] * d[j]
        self.offset = offset
        self._Q = Q
        return Q

    # ---- helpers for docs/tests ------------------------------------------- #
    def encode(self, a) -> np.ndarray:
        x = np.zeros(self.n_vars, dtype=np.int8)
        for i, k in enumerate(a):
            x[self.xidx(i, k)] = 1
        return x

    def balance(self, a) -> float:
        """sum_k L_k^2 of an assignment (the term capacity became)."""
        return float(sum(L * L for L in self.inst.loads(tuple(a))))


@register
class OneHotUnbalancedTight(OneHotUnbalanced):
    """Same QUBO as ``onehot-ub`` with the tight exactly-one weight A = R_w/K (1 + 1 %).

    Proof that every target-energy global minimiser is one-hot (MODEL.md §6, reviewer 2026-10-24):
    (1) a customer on m >= 2 vehicles: dropping one assignment lowers the penalty by A(2m-3) > 0 and
        cannot raise the non-negative pair terms -> a minimiser has no duplicates;
    (2) a missing customer i: with t_k = sum_{j!=i} w_ij x_jk and every other customer on <= 1 vehicle,
        sum_k t_k <= sum_j w_ij <= R_w, so some vehicle has t_k <= R_w/K; adding i there lowers the
        penalty by A and raises the pair terms by t_k < A.  Hence no minimiser has a missing customer.
    A target-model guarantee only: it says nothing about capacity, sampling, or embedding error.
    Named separately so that existing `onehot-ub` rows are not changed in place (ablation: before/after).
    """
    name = "onehot-ub-tight"
    PENALTY_RULE = "row_sum_over_k"
    table_row = False     # same variable count as onehot-ub; keep the Derive 4.1 table to one "ours" row
