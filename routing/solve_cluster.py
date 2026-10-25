"""Integration boundary of Track B: route ONE frozen Part A cluster, return a ``RoutingResult``.

    solve_cluster(instance, customer_ids, config) -> RoutingResult

The cluster is a frozen set of original customer indices (0..N-1 of ``cvrp.common.Instance``);
routing never changes it.  Validation happens here (ids exist, no duplicates, demand <= Q);
clustering-level validation (disjoint, covering, <= K vehicles) is ``validate_clustering``.
Local matrix index 0 is the depot, local index k (1..n) is ``customer_ids[k-1]``.

Policy (handoff "Physical execution and decision gates", initial policy):

* n <= 3     classical exact (``cvrp.common.tsp_exact``), ``method = "classical-exact"``;
* n == 4     both encodings (edge: 10 atoms, position: 16 atoms) through ``routing.pilot``
             (generic ``cvrp.embed.embed`` -> Part A pulse -> Pulser-validated sequence ->
             ``cvrp.svsim`` exact emulation).  The returned route is the best RAW valid shot over
             the two encodings (strict decode; no repair).  If no encoding yields a valid shot the
             route falls back to exact classical routing and is flagged ``fallback=True``;
* n >= 5     exact classical fallback, flagged (position model 25+ atoms is not run here; the
             edge model is a cycle-cover relaxation for n >= 5).

Costs of returned routes are always recomputed from the instance's own distance matrix.  Raw
samples (counts, embedded register, sequence, device-energy analysis) are written separately
under ``out/routing/`` and referenced by path from the result.  Emulation only - nothing here
ran on hardware.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cvrp.common import Instance, tsp_exact

from .encodings import build_edge, build_position, canonical_tour, tour_cost
from .fixtures import RoutingFixture
from .pilot import DEFAULT_OUT, PilotConfig, _json_default, _versions, run_encoding

QUANTUM_N = 4                         # the only cluster size routed through the quantum pilot here
MAX_EXACT_N = 15                      # Held-Karp reference cap (handoff: "up to 15 customers")

FALLBACK_REASON_LARGE = ("n>4: position model 25+ atoms not run here; "
                         "edge model is a cycle-cover relaxation for n>=5")
FALLBACK_REASON_NO_VALID = "n=4: no raw valid tour in {shots} shots of any emulated encoding ({emulated}){not_run}"


@dataclass
class RoutingConfig:
    shots: int = 1000
    seed: int = 0
    n_starts: int = 8
    embed_seeds: int = 6
    dt_ns: float = 4.0
    position_emulation: str = "yes"       # "yes" | "no" (sequence saved, counted as not run)
    out_dir: str = str(DEFAULT_OUT)
    tag: str = ""                         # prefix for record names (e.g. the Part A run it came from)

    def pilot_config(self, name: str) -> PilotConfig:
        return PilotConfig(fixture=name, shots=self.shots, seed=self.seed, n_starts=self.n_starts, dt_ns=self.dt_ns,
                           position_emulation=self.position_emulation, embed_seeds=self.embed_seeds, out_dir=self.out_dir)


@dataclass
class EncodingOutcome:
    """Per-encoding summary of the quantum attempt (raw decode only; no repair)."""
    encoding: str
    atoms: int
    emulated: bool
    shots: int
    n_raw_valid: int
    n_raw_optimal: int
    raw_best_cost: Optional[float]
    raw_best_tour_local: Optional[List[int]]
    raw_best_gap_percent: Optional[float]
    exact_p_valid: Optional[float]
    exact_p_optimal: Optional[float]
    target_ground_states_optimal: bool
    embedding: Dict[str, Any]
    device_argmin_decodes: bool
    rank_of_optimal_tour: int
    emulation_seconds: Optional[float]
    record_path: str


@dataclass
class RoutingResult:
    instance: str
    customer_ids: List[int]                       # original ids (0..N-1), order = local index 1..n
    n: int
    route: List[int]                              # local indices, depot (0) at both ends
    route_ids: List[Any]                          # original ids, "depot" at both ends
    cost: float                                   # recomputed from the instance distance matrix
    method: str                                   # classical-exact | quantum-raw | classical-exact-fallback
    encoding: Optional[str]                       # edge | position | None
    atoms: int
    shots: int
    n_raw_valid: int
    n_raw_optimal: int
    raw_best_cost: Optional[float]
    fallback: bool
    fallback_reason: Optional[str]
    exact_cost: float
    exact_route: List[int]                        # local indices, depot at both ends
    exact_method: str
    gap_percent_vs_exact: float                   # of the returned route
    quantum_attempted: bool
    runtime_seconds: float
    seed: int
    config: Dict[str, Any]
    backend: Optional[str]
    versions: Dict[str, str]
    embedding_diagnostics: Dict[str, Any] = field(default_factory=dict)
    encodings: List[Dict[str, Any]] = field(default_factory=list)
    raw_sample_paths: Dict[str, str] = field(default_factory=dict)
    selection_rule: str = ""
    label: str = "emulation only; no hardware"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=_json_default))


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_cluster(inst: Instance, customer_ids: Sequence[int]) -> List[int]:
    ids = [int(c) for c in customer_ids]
    if len(ids) == 0:
        raise ValueError("empty cluster")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate customer ids in cluster: {ids}")
    bad = [c for c in ids if not 0 <= c < inst.N]
    if bad:
        raise ValueError(f"customer ids {bad} are not customers of {inst.name} (0..{inst.N - 1})")
    load = sum(inst.demands[c] for c in ids)
    if load > inst.Q:
        raise ValueError(f"cluster {ids} has demand {load} > Q = {inst.Q}")
    return ids


def validate_clustering(inst: Instance, clusters: Iterable[Iterable[int]]) -> List[List[int]]:
    """Disjoint, covering, at most K non-empty clusters, each capacity-feasible.  Returns sorted lists."""
    cl = [sorted(validate_cluster(inst, g)) for g in clusters if len(list(g)) > 0]
    if len(cl) > inst.K:
        raise ValueError(f"{len(cl)} non-empty clusters > K = {inst.K}")
    seen: Dict[int, int] = {}
    for k, g in enumerate(cl):
        for c in g:
            if c in seen:
                raise ValueError(f"customer {c} in clusters {seen[c]} and {k}: not disjoint")
            seen[c] = k
    missing = sorted(set(range(inst.N)) - set(seen))
    if missing:
        raise ValueError(f"customers {missing} are in no cluster: not covering")
    return cl


# --------------------------------------------------------------------------- #
# local geometry
# --------------------------------------------------------------------------- #
def local_fixture(inst: Instance, ids: Sequence[int], name: str) -> RoutingFixture:
    """Frozen cluster as a ``RoutingFixture``: depot = local 0, local k = ids[k-1]; coordinates from the instance."""
    return RoutingFixture(name=name, customer_ids=tuple(int(c) for c in ids), depot=tuple(float(v) for v in inst.depot),
                          customers=tuple(tuple(float(v) for v in inst.locations[c]) for c in ids),
                          notes=f"frozen cluster of Part A instance {inst.name}", tags=("part-a-cluster",))


def local_dist_matrix(inst: Instance, ids: Sequence[int]) -> np.ndarray:
    """Rows/cols [depot, ids...] of ``inst.dist_matrix()`` (depot is row 0 there)."""
    D = inst.dist_matrix()
    idx = [0] + [int(c) + 1 for c in ids]
    return D[np.ix_(idx, idx)]


def route_cost_from_instance(inst: Instance, ids: Sequence[int], route_local: Sequence[int]) -> float:
    """Cost of a local route (depot at both ends) from the INSTANCE matrix, not from any model energy."""
    D = inst.dist_matrix()
    glob = [0] + [int(c) + 1 for c in ids]
    r = list(route_local)
    if r[0] != 0 or r[-1] != 0:
        raise ValueError("route must start and end at the depot (local 0)")
    if sorted(r[1:-1]) != list(range(1, len(ids) + 1)):
        raise ValueError("route must visit every customer of the cluster exactly once")
    return float(sum(D[glob[a], glob[b]] for a, b in zip(r[:-1], r[1:])))


def cluster_name(inst: Instance, ids: Sequence[int], tag: str = "") -> str:
    core = f"{inst.name}_c" + "-".join(str(c) for c in ids)
    return f"{tag}__{core}" if tag else core


# --------------------------------------------------------------------------- #
# solve
# --------------------------------------------------------------------------- #
def _exact(Dloc: np.ndarray, n: int) -> Tuple[float, List[int]]:
    if n > MAX_EXACT_N:
        raise ValueError(f"n = {n} > {MAX_EXACT_N}: Held-Karp reference not attempted")
    L, order = tsp_exact(Dloc, list(range(1, n + 1)))
    return float(L), [0, *[int(c) for c in order], 0]


def _raw_counts(enc, counts: Dict[str, int], opt_tours) -> Tuple[int, int]:
    """(# raw valid shots, # raw optimal shots) by strict decode of every sampled string."""
    from cvrp.common import str_to_bits
    opt = {tuple(t) for t in opt_tours}
    n_valid = n_opt = 0
    for s, c in counts.items():
        t = enc.decode(str_to_bits(s))
        if t is None:
            continue
        n_valid += c
        if canonical_tour(t) in opt:
            n_opt += c
    return n_valid, n_opt


def _outcome(enc, rec: Dict[str, Any], path: Path) -> EncodingOutcome:
    s = rec["samples"]
    emu = bool(s.get("backend"))
    dev = rec["device_energies"]
    fid = rec["embedding"]["fidelity"]
    n_valid, n_opt = _raw_counts(enc, rec["raw_counts"], rec["classical_targets"]["optimal_tours_undirected"]) if emu else (0, 0)
    return EncodingOutcome(
        encoding=enc.name, atoms=rec["encoding"]["n_vars"], emulated=emu,
        shots=s.get("shots", 0) if emu else 0, n_raw_valid=n_valid, n_raw_optimal=n_opt,
        raw_best_cost=s.get("best_feasible_cost") if emu else None,
        raw_best_tour_local=s.get("best_feasible_tour") if emu else None,
        raw_best_gap_percent=s.get("best_feasible_gap_percent") if emu else None,
        exact_p_valid=s.get("exact_p_feasible") if emu else None, exact_p_optimal=s.get("exact_p_optimal") if emu else None,
        target_ground_states_optimal=bool(rec["classical_targets"]["all_ground_states_optimal"]),
        embedding={"rel_frobenius_offdiag": fid["rel_frobenius_offdiag"], "worst_pair_rel_error": fid["worst_pair_rel_error"],
                   "max_spurious_coupling": fid["max_spurious_coupling"], "alpha": rec["embedding"]["alpha"],
                   "delta_final": rec["embedding"]["delta_final"], "selected_embed_seed": rec["embedding"]["selected_seed"],
                   "A": rec["encoding"]["A"], "validation_violations": rec["embedding"]["sequence"]["validation_violations"]},
        device_argmin_decodes=bool(dev["device_argmin_decodes"]),
        rank_of_optimal_tour=int(min(dev["rank_of_optimal_tour_strings"])),
        emulation_seconds=s.get("seconds") if emu else None, record_path=str(path))


def solve_cluster(inst: Instance, customer_ids: Sequence[int], config: Optional[RoutingConfig] = None) -> RoutingResult:
    cfg = config or RoutingConfig()
    t0 = time.time()
    ids = validate_cluster(inst, customer_ids)
    n = len(ids)
    Dloc = local_dist_matrix(inst, ids)
    exact_cost, exact_route = _exact(Dloc, n)
    versions = _versions()
    base = dict(instance=inst.name, customer_ids=ids, n=n, exact_cost=exact_cost, exact_route=exact_route,
                exact_method="cvrp.common.tsp_exact (Held-Karp)" if n > 1 else "depot out-and-back",
                seed=cfg.seed, config=asdict(cfg), versions=versions)

    def finish(route, method, encoding, atoms, shots, n_valid, n_opt, raw_best, fallback, reason, backend,
               emb_diag, outcomes, paths, rule, quantum):
        cost = route_cost_from_instance(inst, ids, route)
        return RoutingResult(route=[int(v) for v in route], route_ids=["depot", *[ids[k - 1] for k in route[1:-1]], "depot"],
                             cost=cost, method=method, encoding=encoding, atoms=atoms, shots=shots, n_raw_valid=n_valid,
                             n_raw_optimal=n_opt, raw_best_cost=raw_best, fallback=fallback, fallback_reason=reason,
                             gap_percent_vs_exact=100.0 * (cost - exact_cost) / exact_cost if exact_cost > 0 else 0.0,
                             quantum_attempted=quantum, runtime_seconds=time.time() - t0, backend=backend,
                             embedding_diagnostics=emb_diag, encodings=[asdict(o) for o in outcomes],
                             raw_sample_paths=paths, selection_rule=rule, **base)

    if n <= 3:
        return finish(exact_route, "classical-exact", None, 0, 0, 0, 0, None, False, None, None, {}, [], {},
                      "n<=3: classical exact by policy (no quantum attempt)", False)
    if n > QUANTUM_N:
        return finish(exact_route, "classical-exact-fallback", None, 0, 0, 0, 0, None, True, FALLBACK_REASON_LARGE,
                      None, {}, [], {}, "n>4: exact classical fallback by policy", False)

    # ---- n == 4: both encodings through the pilot machinery ---------------------------------------
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cluster_name(inst, ids, cfg.tag)
    fx = local_fixture(inst, ids, name)
    pcfg = cfg.pilot_config(name)
    outcomes: List[EncodingOutcome] = []
    paths: Dict[str, str] = {}
    for enc in (build_edge(Dloc), build_position(Dloc)):
        do_emu = enc.name == "edge" or cfg.position_emulation == "yes"
        rec = run_encoding(fx, enc, pcfg, out_dir, do_emu)
        rec["label"] = f"Part A cluster of {inst.name} routed through the quantum pilot (emulation; no hardware)"
        path = out_dir / f"cluster_{name}_{enc.name}.json"
        path.write_text(json.dumps(rec, indent=1, default=_json_default))
        paths[enc.name] = str(path)
        paths[f"{enc.name}_sequence"] = rec["paths"]["sequence_abstract_repr"]
        paths[f"{enc.name}_register"] = rec["paths"]["register"]
        outcomes.append(_outcome(enc, rec, path))

    emb_diag = {o.encoding: o.embedding for o in outcomes}
    atoms_total = {o.encoding: o.atoms for o in outcomes}
    rule = ("n=4: lowest raw best feasible cost over {edge, position} (strict decode, no repair); "
            "ties -> higher raw valid count, then edge (fewer atoms); no valid shot anywhere -> exact classical fallback")
    valid = [o for o in outcomes if o.emulated and o.raw_best_cost is not None]
    backend = "cvrp.svsim (exact state vector)"
    shots_total = sum(o.shots for o in outcomes)
    n_valid_total = sum(o.n_raw_valid for o in outcomes)
    n_opt_total = sum(o.n_raw_optimal for o in outcomes)
    if not valid:
        emulated = [o.encoding for o in outcomes if o.emulated]
        not_run = [o.encoding for o in outcomes if not o.emulated]
        reason = FALLBACK_REASON_NO_VALID.format(
            shots=cfg.shots, emulated=", ".join(emulated),
            not_run=f"; {', '.join(not_run)} not emulated (sequence saved)" if not_run else "")
        return finish(exact_route, "classical-exact-fallback", None, max(atoms_total.values()), shots_total, 0, 0, None,
                      True, reason, backend, emb_diag, outcomes, paths, rule, True)
    best = sorted(valid, key=lambda o: (o.raw_best_cost, -o.n_raw_valid, 0 if o.encoding == "edge" else 1))[0]
    route = [0, *best.raw_best_tour_local, 0]
    # sanity: the pilot's cost (from the local matrix) must agree with the instance matrix
    assert abs(tour_cost(Dloc, best.raw_best_tour_local) - route_cost_from_instance(inst, ids, route)) < 1e-9
    res = finish(route, "quantum-raw", best.encoding, best.atoms, shots_total, n_valid_total, n_opt_total,
                 best.raw_best_cost, False, None, backend, emb_diag, outcomes, paths, rule, True)
    return res


def is_optimal_route(res: RoutingResult, tol: float = 1e-7) -> bool:
    return abs(res.cost - res.exact_cost) <= tol * max(1.0, res.exact_cost)


__all__ = ["RoutingConfig", "RoutingResult", "EncodingOutcome", "solve_cluster", "validate_cluster",
           "validate_clustering", "local_fixture", "local_dist_matrix", "route_cost_from_instance", "cluster_name",
           "is_optimal_route", "QUANTUM_N", "FALLBACK_REASON_LARGE", "FALLBACK_REASON_NO_VALID", "canonical_tour"]
