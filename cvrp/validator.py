"""Validator for CVRP instance files -- deliverable D3's gate.

``validate(inst)`` returns a list of human-readable problems; an empty list
means the instance satisfies every assumption of brief §2, the tier table, the
feasibility certificate, and (if present) the reference block written by
``cvrp.reference``.  The CLI prints one line per JSON file and exits with
status 1 if anything fails::

    python -m cvrp.validator instances/            # also re-generates from the seed
    python -m cvrp.validator instances/ --no-regen # skip the reproducibility check

What is checked
---------------
Brief §2 assumptions
  (1) depot fixed and stated (``depot_rule`` non-empty, depot inside the square;
      with the default rule the depot must be the centre);
  (2) costs symmetric, non-negative, zero on the diagonal and obeying the
      triangle inequality -- checked numerically on the distance matrix even
      though Euclidean distances guarantee it;
  (3) every customer served by exactly one vehicle -- an assignment is a
      length-N vector with one vehicle index per customer, so "exactly once"
      is structural; we check length and index range;
  (4) one trip per vehicle -- the load of a vehicle is the sum over its
      customers and must fit in a single trip, i.e. <= Q;
  (5) no single demand exceeds Q;
  (6) fleet size K fixed; fewer vehicles may be used (an unused vehicle simply
      has no customers).
Tier table
  N, K, Q and the demand range must match ``cvrp.common.TIERS[tier]``;
  locations inside ``[0, side]^2`` and pairwise distinct; name convention.
Certificate
  valid assignment with all loads <= Q; rho and d_max/Q as reported.
Reproducibility (CLI default, ``validate(inst, regenerate=True)``)
  re-running ``cvrp.instances.generate(tier, seed, **knobs)`` must reproduce
  locations, demands and certificate exactly.
Reference block (only if ``inst.reference`` is non-empty)
  ``best_assignment`` valid; its cost recomputed with ``common.route_exactly``
  equals ``cost``; ``best_clusters`` is the same clustering; every route is a
  permutation of that vehicle's customers and the routes add up to the cost;
  the cross-checks (naive brute force, SA) cannot beat a proven optimum.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from cvrp.common import TIERS, Instance, clustering_of, route_exactly

NUM_TOL = 1e-6        # absolute tolerance for recomputed costs / ratios
TRI_TOL = 1e-9        # slack for the triangle inequality (floating point)
REFERENCE_STATUSES = {"proven_optimum", "best_known", "heuristic"}


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and math.isfinite(float(x))


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


# --------------------------------------------------------------------------- #
# Individual checks (each appends to ``errs``)
# --------------------------------------------------------------------------- #
def _check_tier(inst: Instance, errs: List[str]) -> None:
    if inst.tier not in TIERS:
        errs.append(f"unknown tier {inst.tier!r}")
        return
    t = TIERS[inst.tier]
    for key in ("N", "K", "Q"):
        if getattr(inst, key) != t[key]:
            errs.append(f"{key}={getattr(inst, key)} does not match tier {inst.tier} ({t[key]})")
    expected = f"{inst.tier}_seed{inst.seed:02d}"
    if inst.name != expected:
        errs.append(f"name {inst.name!r} should be {expected!r} (tier + seed convention)")
    if not _is_int(inst.seed) or inst.seed < 0:
        errs.append(f"seed must be a non-negative int, got {inst.seed!r}")


def _check_geometry(inst: Instance, errs: List[str]) -> None:
    side = inst.square_side
    if not (_is_num(side) and side > 0):
        errs.append(f"square_side must be positive, got {side!r}")
        return
    # (1) depot fixed and stated
    if not inst.depot_rule or not str(inst.depot_rule).strip():
        errs.append("depot_rule is empty: the depot position must be stated (§2 assumption 1)")
    if len(inst.depot) != 2 or not all(_is_num(c) for c in inst.depot):
        errs.append(f"depot must be two finite numbers, got {inst.depot!r}")
        return
    if not all(0.0 <= c <= side for c in inst.depot):
        errs.append(f"depot {inst.depot} lies outside the square [0, {side}]^2")
    if "centre" in str(inst.depot_rule).lower() or "center" in str(inst.depot_rule).lower():
        if any(abs(c - side / 2) > NUM_TOL for c in inst.depot):
            errs.append(f"depot_rule says centre but depot is {inst.depot}, not ({side/2}, {side/2})")
    # locations
    if len(inst.locations) != inst.N:
        errs.append(f"{len(inst.locations)} locations for N={inst.N} customers")
    for i, p in enumerate(inst.locations):
        if len(p) != 2 or not all(_is_num(c) for c in p):
            errs.append(f"location {i} is not a pair of finite numbers: {p!r}")
        elif not all(0.0 <= c <= side for c in p):
            errs.append(f"location {i} = {tuple(p)} outside the square [0, {side}]^2")
    pts = [tuple(float(c) for c in p) for p in inst.locations if len(p) == 2]
    if len(set(pts)) != len(pts):
        dup = sorted({p for p in pts if pts.count(p) > 1})
        errs.append(f"duplicate customer locations: {dup}")
    if tuple(float(c) for c in inst.depot) in set(pts):
        errs.append("a customer sits exactly on the depot")


def _check_cost_matrix(inst: Instance, errs: List[str]) -> None:
    """(2) symmetric, non-negative, zero diagonal, triangle inequality."""
    try:
        D = inst.dist_matrix()
    except Exception as e:  # noqa: BLE001  (malformed coordinates already reported)
        errs.append(f"could not build the distance matrix: {e}")
        return
    if not np.all(np.isfinite(D)):
        errs.append("distance matrix has non-finite entries")
        return
    if not np.allclose(D, D.T, atol=TRI_TOL):
        errs.append("cost matrix is not symmetric")
    if np.any(np.abs(np.diag(D)) > TRI_TOL):
        errs.append("cost matrix has a non-zero diagonal")
    if np.any(D < -TRI_TOL):
        errs.append("cost matrix has negative entries")
    # triangle inequality: D[i,j] <= D[i,k] + D[k,j] for all i, j, k
    via = D[:, :, None] + D[None, :, :]          # via[i,k,j] = D[i,k] + D[k,j]
    worst = (D[:, None, :] - via).max()           # > 0 means violated
    if worst > TRI_TOL:
        errs.append(f"triangle inequality violated by {worst:.3g}")


def _check_demands(inst: Instance, errs: List[str]) -> None:
    if len(inst.demands) != inst.N:
        errs.append(f"{len(inst.demands)} demands for N={inst.N} customers")
    t = TIERS.get(inst.tier)
    for i, d in enumerate(inst.demands):
        if not _is_int(d):
            errs.append(f"demand {i} = {d!r} is not an integer")
            continue
        if d > inst.Q:
            errs.append(f"demand {i} = {d} exceeds Q={inst.Q} (§2 assumption 5)")
        if d < 1:
            errs.append(f"demand {i} = {d} must be positive")
        if t is not None and not (t["d_min"] <= d <= t["d_max"]):
            errs.append(f"demand {i} = {d} outside tier range [{t['d_min']}, {t['d_max']}]")


def _check_assignment(inst: Instance, a: Sequence[int], label: str, errs: List[str]) -> bool:
    """Shared by the certificate and the reference assignment.  True if valid."""
    if len(a) != inst.N:
        errs.append(f"{label} has length {len(a)}, expected N={inst.N} (every customer exactly once)")
        return False
    bad = [k for k in a if not _is_int(k) or not (0 <= k < inst.K)]
    if bad:
        errs.append(f"{label} has vehicle indices outside 0..{inst.K - 1}: {bad}")
        return False
    if len(inst.demands) != inst.N:
        return False
    loads = inst.loads(tuple(int(k) for k in a))
    over = [(k, L) for k, L in enumerate(loads) if L > inst.Q]
    if over:
        errs.append(f"{label}: vehicle load exceeds Q={inst.Q}: {over} (one trip per vehicle)")
        return False
    return True


def _check_certificate_and_ratios(inst: Instance, errs: List[str]) -> None:
    _check_assignment(inst, inst.certificate, "certificate", errs)
    if not inst.feasibility_method or not str(inst.feasibility_method).strip():
        errs.append("feasibility_method is empty: state how feasibility was guaranteed")
    if len(inst.demands) == inst.N and inst.N > 0 and all(_is_int(d) for d in inst.demands):
        rho = sum(inst.demands) / (inst.K * inst.Q)
        if abs(rho - inst.rho) > NUM_TOL:
            errs.append(f"rho={inst.rho} but sum d/(K Q)={rho:.6f}")
        dq = max(inst.demands) / inst.Q
        if abs(dq - inst.dmax_over_Q) > NUM_TOL:
            errs.append(f"dmax_over_Q={inst.dmax_over_Q} but max d/Q={dq:.6f}")
    if not isinstance(inst.generator, dict) or "version" not in inst.generator:
        errs.append("generator block must record at least the generator version")


def _check_regeneration(inst: Instance, errs: List[str]) -> None:
    """Reproducibility: the seed + recorded knobs must reproduce the data."""
    from cvrp import instances as gen   # local import: validator stays usable without it

    g = inst.generator if isinstance(inst.generator, dict) else {}
    if g.get("version") != gen.GENERATOR_VERSION:
        errs.append(f"generator version {g.get('version')!r} != current {gen.GENERATOR_VERSION!r}; "
                    "cannot check reproducibility")
        return
    knobs = {k: g[k] for k in gen.DEFAULTS if k in g}
    try:
        again = gen.generate(inst.tier, inst.seed, **knobs)
    except Exception as e:  # noqa: BLE001
        errs.append(f"re-generation from seed failed: {e}")
        return
    if [tuple(p) for p in again.locations] != [tuple(p) for p in inst.locations]:
        errs.append("locations are not reproduced by generate(tier, seed)")
    if list(again.demands) != list(inst.demands):
        errs.append("demands are not reproduced by generate(tier, seed)")
    if tuple(again.certificate) != tuple(inst.certificate):
        errs.append("certificate is not reproduced by generate(tier, seed)")


def _check_reference(inst: Instance, errs: List[str]) -> None:
    ref = inst.reference
    if not ref:
        return
    if not isinstance(ref, dict):
        errs.append("reference must be a dict")
        return
    for key in ("cost", "best_assignment", "reference_status", "method"):
        if key not in ref:
            errs.append(f"reference lacks {key!r}")
    if ref.get("reference_status") not in REFERENCE_STATUSES:
        errs.append(f"reference_status {ref.get('reference_status')!r} not in {sorted(REFERENCE_STATUSES)}")
    if "cost" in ref and not _is_num(ref["cost"]):
        errs.append(f"reference cost is not a finite number: {ref['cost']!r}")
    a = ref.get("best_assignment")
    if a is None or not _check_assignment(inst, a, "reference best_assignment", errs) or not _is_num(ref.get("cost")):
        return
    a = tuple(int(k) for k in a)
    total, routes = route_exactly(inst, a)
    if abs(total - float(ref["cost"])) > NUM_TOL:
        errs.append(f"reference cost {ref['cost']} != recomputed exact-TSP cost {total:.6f} of best_assignment")
    # best_clusters must describe the same clustering
    if "best_clusters" in ref:
        try:
            cl = frozenset(frozenset(int(i) for i in g) for g in ref["best_clusters"])
        except Exception:  # noqa: BLE001
            cl = None
        if cl != clustering_of(a):
            errs.append("best_clusters is not the clustering of best_assignment")
    # routes: one list per vehicle, a permutation of that vehicle's customers
    if "routes" in ref:
        rts = ref["routes"]
        if not isinstance(rts, list) or len(rts) != inst.K:
            errs.append(f"routes must be a list of K={inst.K} visiting orders")
        else:
            D = inst.dist_matrix()
            route_total = 0.0
            for k, order in enumerate(rts):
                members = sorted(i for i in range(inst.N) if a[i] == k)
                if sorted(int(i) for i in order) != members:
                    errs.append(f"route of vehicle {k} {order} is not a permutation of its customers {members}")
                    continue
                if order:
                    seq = [0] + [int(i) + 1 for i in order] + [0]
                    route_total += float(sum(D[seq[j], seq[j + 1]] for j in range(len(seq) - 1)))
            if abs(route_total - float(ref["cost"])) > NUM_TOL:
                errs.append(f"routes add up to {route_total:.6f}, not the reference cost {ref['cost']}")
    n_opt = ref.get("n_distinct_optima")
    if n_opt is not None and (not _is_int(n_opt) or n_opt < 1):
        errs.append(f"n_distinct_optima must be a positive int, got {n_opt!r}")
    # nothing may beat a proven optimum
    if ref.get("reference_status") == "proven_optimum":
        for key in ("sa_cost", "brute_force_naive_cost"):
            v = ref.get(key)
            if _is_num(v) and float(v) < float(ref["cost"]) - NUM_TOL:
                errs.append(f"{key}={v} is below the proven optimum {ref['cost']}")
        bf = ref.get("brute_force_naive_cost")
        if _is_num(bf) and abs(float(bf) - float(ref["cost"])) > NUM_TOL:
            errs.append(f"naive brute force {bf} disagrees with the DP optimum {ref['cost']}")
        if ref.get("agree") is False:
            errs.append("reference reports agree=False between the two tier-S methods")
    if "sa_cost" in ref and "sa_matches_optimum" in ref and _is_num(ref["sa_cost"]):
        matches = abs(float(ref["sa_cost"]) - float(ref["cost"])) <= NUM_TOL
        if bool(ref["sa_matches_optimum"]) != matches:
            errs.append(f"sa_matches_optimum={ref['sa_matches_optimum']} inconsistent with sa_cost={ref['sa_cost']}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def validate(inst: Instance, regenerate: bool = False) -> List[str]:
    """Return a list of problems (empty list = the instance is valid).

    ``regenerate=True`` additionally re-runs the generator from the recorded
    seed and knobs and requires an exact match (reproducibility).
    """
    errs: List[str] = []
    _check_tier(inst, errs)
    _check_geometry(inst, errs)
    _check_cost_matrix(inst, errs)
    _check_demands(inst, errs)
    _check_certificate_and_ratios(inst, errs)
    if regenerate:
        _check_regeneration(inst, errs)
    _check_reference(inst, errs)
    return errs


def validate_file(path: str | Path, regenerate: bool = False) -> List[str]:
    try:
        inst = Instance.load(path)
    except Exception as e:  # noqa: BLE001
        return [f"cannot load: {e}"]
    return validate(inst, regenerate=regenerate)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate CVRP instance JSON files (D3 gate).")
    p.add_argument("paths", nargs="+", help="JSON files or folders containing them")
    p.add_argument("--no-regen", action="store_true", help="skip the regenerate-from-seed check")
    args = p.parse_args(argv)
    files: List[Path] = []
    for s in args.paths:
        q = Path(s)
        files += sorted(q.glob("*.json")) if q.is_dir() else [q]
    if not files:
        print("no instance files found", file=sys.stderr)
        return 1
    n_bad = 0
    for f in files:
        errs = validate_file(f, regenerate=not args.no_regen)
        if errs:
            n_bad += 1
            print(f"FAIL {f}")
            for e in errs:
                print(f"     - {e}")
        else:
            inst = Instance.load(f)
            ref = inst.reference or {}
            tail = (f"C*={ref['cost']:.4f} ({ref.get('reference_status')}, "
                    f"{ref.get('n_distinct_optima', '?')} optima)" if ref else "no reference yet")
            print(f"OK   {f}  rho={inst.rho:.3f} dmax/Q={inst.dmax_over_Q:.3f}  {tail}")
    print(f"\n{len(files) - n_bad}/{len(files)} instances valid")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
