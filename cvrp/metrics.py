"""Score a histogram of measured bitstrings against the brief's §9 metrics.

Rules (CLAUDE.md): every gap is computed on a *feasible decode*; shots whose
bits do not decode to a capacity-feasible clustering contribute only to the
feasibility rate.  We never report a mean cost.  Vehicles are identical, so
"reaches the reference" compares clusterings as sets of customer groups.

Gap levels (brief §9.2, Track A):
  model gap   : model objective of the best feasible shot  vs  optimum of the
                constrained clustering model (same objective)
  CVRP gap    : that same best shot, routed exactly (TSP per cluster)  vs  the
                instance's CVRP reference C*
  decomposition gap (informative, Track B name): model optimum routed exactly vs C*

"Best shot" = the feasible shot with the lowest *model* objective, because
that is what the device was asked to minimise.  We also record the best
*routed* cost over all distinct feasible clusterings seen (labelled
``cvrp_gap_best_routed``); that is classical post-selection by routing and
is reported separately, never as the headline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .common import (Assignment, Clustering, Formulation, Instance, clustering_of, gap_percent,
                     route_exactly, str_to_bits)

EPSILONS = (0.0, 1.0, 5.0, 10.0)   # % levels for P(gap <= eps)


@dataclass
class ModelOptimum:
    value: float
    assignment: Assignment
    clusterings: List[Clustering]          # every optimal clustering (mod relabeling)
    n_optima: int
    method: str


def model_optimum(f: Formulation, exact_partition: Optional[Callable] = None) -> ModelOptimum:
    """Constrained optimum of the clustering model.

    Small N·K: enumerate all K^N assignments (exact, also collects every
    optimal clustering).  Otherwise use the generic set-partition DP from
    cvrp.exact (needs an additive objective: Formulation.cluster_cost)."""
    inst = f.inst
    N, K = inst.N, inst.K
    if K ** N <= 200_000:
        best = np.inf
        bests: Dict[Clustering, Assignment] = {}
        for idx in range(K ** N):
            a = []
            v = idx
            for _ in range(N):
                a.append(v % K)
                v //= K
            a = tuple(a)
            if not inst.capacity_ok(a):
                continue
            val = f.objective(a)
            if val < best - 1e-9:
                best, bests = val, {clustering_of(a): a}
            elif abs(val - best) <= 1e-9:
                bests.setdefault(clustering_of(a), a)
        a0 = next(iter(bests.values()))
        return ModelOptimum(float(best), a0, list(bests), len(bests), "enumeration of all K^N assignments")
    if exact_partition is None:
        from .exact import exact_partition as _ep
        exact_partition = _ep
    if not f.additive:
        raise ValueError("model objective is not additive over clusters; cannot use the exact partition DP")

    def cost(members):
        if sum(inst.demands[i] for i in members) > inst.Q:
            return float("inf")
        return f.cluster_cost(frozenset(members))

    value, clusters, n_opt = exact_partition(N, K, cost)
    a = [-1] * N
    for k, g in enumerate(clusters):
        for i in g:
            a[i] = k
    a = tuple(a)
    return ModelOptimum(float(value), a, [clustering_of(a)], int(n_opt), "exact set-partition DP (cvrp.exact)")


@dataclass
class ShotRow:
    bits: str
    count: int
    energy: float
    decodable: bool
    feasible: bool
    assignment: Optional[Assignment]
    objective: Optional[float]
    model_gap: Optional[float]
    routed_cost: Optional[float]
    cvrp_gap: Optional[float]


@dataclass
class Metrics:
    shots: int
    n_distinct_bitstrings: int
    decodable_rate: float
    feasible_rate: float
    # best feasible shot by model objective
    best_model_objective: Optional[float]
    best_model_gap: Optional[float]
    best_assignment: Optional[List[int]]
    best_routed_cost: Optional[float]
    best_cvrp_gap: Optional[float]
    # post-selection by routing (labelled, not headline)
    best_routed_over_feasible: Optional[float]
    cvrp_gap_best_routed: Optional[float]
    # probabilities (fractions of ALL shots)
    p_reach_model_optimum: float
    p_reach_cvrp_reference: float
    p_gap_le: Dict[str, float]                 # model gap <= eps %, keyed "0", "1", "5", "10"
    p_cvrp_gap_le: Dict[str, float]
    n_distinct_feasible_clusterings: int
    # references
    model_optimum: float
    model_n_optima: int
    model_optimum_method: str
    decomposition_cost: float                  # model optimum routed exactly
    decomposition_gap: float
    cvrp_reference: float
    reference_status: str
    # the lowest-energy shot (is the device minimising what we think?)
    argmin_energy_bits: str
    argmin_energy_feasible: bool
    top: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        g = lambda v: "undefined" if v is None else f"{v:+.2f}%"
        return "\n".join([
            f"shots={self.shots} distinct={self.n_distinct_bitstrings} feasible rate={self.feasible_rate:.3f} (decodable {self.decodable_rate:.3f})",
            f"best feasible shot: model obj={self.best_model_objective} model gap={g(self.best_model_gap)} | routed {self.best_routed_cost} CVRP gap={g(self.best_cvrp_gap)}",
            f"post-selected by routing: {self.best_routed_over_feasible} gap={g(self.cvrp_gap_best_routed)}",
            f"P(reach model opt)={self.p_reach_model_optimum:.3f}  P(reach CVRP ref)={self.p_reach_cvrp_reference:.3f}  P(model gap<=5%)={self.p_gap_le['5']:.3f}",
            f"model optimum={self.model_optimum:.4f} ({self.model_n_optima} optima, {self.model_optimum_method}); decomposition gap={self.decomposition_gap:+.2f}%",
            f"CVRP reference={self.cvrp_reference:.4f} [{self.reference_status}]; lowest-energy shot feasible: {self.argmin_energy_feasible}",
        ])


def score_counts(counts: Dict[str, int], f: Formulation, inst: Instance, mopt: Optional[ModelOptimum] = None,
                 top: int = 10, transform: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Tuple[Metrics, List[ShotRow]]:
    """counts: bitstring -> count (bit i = variable i).  ``transform`` is an
    optional labelled post-processing step (e.g. repair) applied to the bits
    before decoding; the caller labels the resulting Metrics accordingly."""
    if not inst.reference or "cost" not in inst.reference:
        raise ValueError(f"instance {inst.name} has no reference; run cvrp.reference first")
    C_ref = float(inst.reference["cost"])
    ref_status = str(inst.reference.get("reference_status", "unknown"))
    ref_clusterings = {clustering_of(tuple(inst.reference["best_assignment"]))}
    for a in inst.reference.get("all_optimal_assignments", []):
        ref_clusterings.add(clustering_of(tuple(a)))
    if mopt is None:
        mopt = model_optimum(f)
    # Decomposition cost: the model optimum routed exactly, minimised over every optimal clustering we know
    # (the enumeration branch lists all of them; the DP branch only one, see model_optimum).
    dec_cost = min(route_exactly(inst, _assignment_from(cl, inst.N))[0] for cl in mopt.clusterings)

    shots = sum(counts.values())
    rows: List[ShotRow] = []
    routed_cache: Dict[Clustering, float] = {}
    n_dec = n_feas = 0
    p_reach_model = p_reach_cvrp = 0
    p_le = {str(int(e)): 0 for e in EPSILONS}
    p_cle = {str(int(e)): 0 for e in EPSILONS}
    best_row: Optional[ShotRow] = None
    best_routed: Optional[float] = None
    Qm = f.Q()
    for bits, cnt in counts.items():
        x = str_to_bits(bits)
        if transform is not None:
            x = transform(x)
        e = float(x @ Qm @ x + f.offset)
        a = f.decode(x)
        dec = a is not None
        feas = f.feasible(a)
        obj = mg = rc = cg = None
        if feas:
            obj = float(f.objective(a))
            mg = gap_percent(obj, mopt.value) if mopt.value != 0 else (0.0 if obj == 0 else float("inf"))
            cl = clustering_of(a)
            if cl not in routed_cache:
                routed_cache[cl] = route_exactly(inst, a)[0]
            rc = routed_cache[cl]
            cg = gap_percent(rc, C_ref)
            n_feas += cnt
            if mg <= 1e-9:                      # reaches a model optimum (by value: robust to ties the DP did not list)
                p_reach_model += cnt
            if cl in ref_clusterings or abs(rc - C_ref) <= 1e-6 * max(1.0, C_ref):
                p_reach_cvrp += cnt
            for eps in EPSILONS:
                if mg <= eps + 1e-9:
                    p_le[str(int(eps))] += cnt
                if cg <= eps + 1e-9:
                    p_cle[str(int(eps))] += cnt
            if best_routed is None or rc < best_routed:
                best_routed = rc
        if dec:
            n_dec += cnt
        row = ShotRow(bits, cnt, e, dec, feas, a, obj, mg, rc, cg)
        rows.append(row)
        if feas and (best_row is None or obj < best_row.objective - 1e-12 or
                     (abs(obj - best_row.objective) <= 1e-12 and cnt > best_row.count)):
            best_row = row
    rows.sort(key=lambda r: (-r.count, r.energy))
    argmin = min(rows, key=lambda r: r.energy)
    m = Metrics(
        shots=shots, n_distinct_bitstrings=len(rows),
        decodable_rate=n_dec / shots, feasible_rate=n_feas / shots,
        best_model_objective=best_row.objective if best_row else None,
        best_model_gap=best_row.model_gap if best_row else None,
        best_assignment=list(best_row.assignment) if best_row else None,
        best_routed_cost=best_row.routed_cost if best_row else None,
        best_cvrp_gap=best_row.cvrp_gap if best_row else None,
        best_routed_over_feasible=best_routed,
        cvrp_gap_best_routed=gap_percent(best_routed, C_ref) if best_routed is not None else None,
        p_reach_model_optimum=p_reach_model / shots, p_reach_cvrp_reference=p_reach_cvrp / shots,
        p_gap_le={k: v / shots for k, v in p_le.items()}, p_cvrp_gap_le={k: v / shots for k, v in p_cle.items()},
        n_distinct_feasible_clusterings=len(routed_cache),
        model_optimum=mopt.value, model_n_optima=mopt.n_optima, model_optimum_method=mopt.method,
        decomposition_cost=dec_cost, decomposition_gap=gap_percent(dec_cost, C_ref),
        cvrp_reference=C_ref, reference_status=ref_status,
        argmin_energy_bits=argmin.bits, argmin_energy_feasible=argmin.feasible,
        top=[{"bits": r.bits, "count": r.count, "energy": r.energy, "feasible": r.feasible,
              "objective": r.objective, "model_gap": r.model_gap, "cvrp_gap": r.cvrp_gap} for r in rows[:top]],
    )
    return m, rows


def _assignment_from(cl: Clustering, N: int) -> Assignment:
    a = [-1] * N
    for k, g in enumerate(cl):
        for i in g:
            a[i] = k
    return tuple(a)


def random_counts(n_vars: int, shots: int, seed: int) -> Dict[str, int]:
    """Uniform random bitstrings: the 'random sampler' method the brief asks
    us to compare the quantum part against."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(shots, n_vars))
    out: Dict[str, int] = {}
    for row in X:
        s = "".join(str(int(b)) for b in row)
        out[s] = out.get(s, 0) + 1
    return out
