"""Formulation registry, exact model optimum and the brute-force QUBO checker.

Everything here is *model level* and formulation agnostic: it only uses the
``Formulation`` interface from ``cvrp.common``.  Concrete formulations live in
``cvrp/formulations/`` and register themselves with ``@register``.

Vocabulary (see docs/DERIVATIONS.md §0):
* bitstring  x in {0,1}^n   -- what the register produces
* assignment a in {0..K-1}^N -- vehicle of every customer (``decode(x)``)
* feasible   -- decodable and every vehicle load <= Q
* model optimum -- the feasible assignment with the smallest ``objective``
  (the constrained optimum of the *model*, not the CVRP route length).

Run ``python -m cvrp.model`` for the checker numbers quoted in DERIVATIONS.md §3.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from cvrp import common
from cvrp.common import (Assignment, Clustering, Formulation, Instance, TIERS,
                         all_bitstrings, clustering_of)

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
REGISTRY: Dict[str, Type[Formulation]] = {}


def register(cls: Type[Formulation]) -> Type[Formulation]:
    """Class decorator: ``REGISTRY[cls.name] = cls``.  Names must be unique."""
    if cls.name == "abstract" or not cls.name:
        raise ValueError(f"{cls.__name__} must set a class attribute `name`")
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"formulation name {cls.name!r} already registered by {REGISTRY[cls.name].__name__}")
    REGISTRY[cls.name] = cls
    return cls


def _load_builtin_formulations() -> None:
    """Import ``cvrp.formulations`` so its ``@register`` decorators run."""
    import cvrp.formulations  # noqa: F401  (side effect: registration)


def get_formulation(name: str, inst: Instance, **weights: float) -> Formulation:
    """Instantiate a registered formulation by name.  ``weights`` override the
    formulation's default penalty weights (names are formulation specific,
    e.g. ``A=..., B=...`` for the textbook one)."""
    _load_builtin_formulations()
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown formulation {name!r}; registered: {sorted(REGISTRY)}") from None
    return cls(inst, **weights)


def registered_formulations() -> Dict[str, Type[Formulation]]:
    _load_builtin_formulations()
    return dict(REGISTRY)


# --------------------------------------------------------------------------- #
# Toy instance (tests and docs only; the official generator is cvrp.instances)
# --------------------------------------------------------------------------- #
def make_toy_instance(tier: str = "S", seed: int = 0) -> Instance:
    """Deterministic tier-shaped instance built *without* ``cvrp.instances``.

    Locations uniform in [0, 100]^2, demands uniform integers in
    [d_min, d_max], depot at the centre.  The seed is re-drawn (seed+1000,
    seed+2000, ...) until a capacity-feasible assignment exists; the first one
    found by enumeration is the certificate.  Only meant for N*K small.
    """
    t = TIERS[tier]
    N, K, Q = t["N"], t["K"], t["Q"]
    for attempt in range(50):
        rng = np.random.default_rng(seed + 1000 * attempt)
        locs = [tuple(float(v) for v in p) for p in rng.uniform(0, common.SQUARE_SIDE, size=(N, 2))]
        demands = [int(v) for v in rng.integers(t["d_min"], t["d_max"] + 1, size=N)]
        cert = _first_feasible_assignment(N, K, Q, demands)
        if cert is not None:
            return Instance(
                name=f"toy-{tier}-{seed}", tier=tier, seed=seed, N=N, K=K, Q=Q,
                depot=(common.SQUARE_SIDE / 2, common.SQUARE_SIDE / 2),
                locations=locs, demands=demands, certificate=cert,
                rho=sum(demands) / (K * Q), dmax_over_Q=max(demands) / Q,
                feasibility_method="resample until enumeration finds a feasible assignment",
                generator={"module": "cvrp.model.make_toy_instance", "attempt": attempt},
            )
    raise RuntimeError("could not build a feasible toy instance")


def _first_feasible_assignment(N: int, K: int, Q: int, demands: Sequence[int]) -> Optional[Assignment]:
    for a in itertools.product(range(K), repeat=N):
        loads = [0] * K
        for i, k in enumerate(a):
            loads[k] += demands[i]
        if max(loads) <= Q:
            return tuple(a)
    return None


# --------------------------------------------------------------------------- #
# Exact model optimum by enumeration
# --------------------------------------------------------------------------- #
MAX_ENUM_ASSIGNMENTS = 2_000_000


def _enumerate_feasible_assignments(inst: Instance):
    """Yield every capacity-feasible assignment with customer 0 on vehicle 0.

    Vehicles are identical, so fixing customer 0's label loses no clustering
    and divides the K^N enumeration by K."""
    N, K = inst.N, inst.K
    d = inst.demands
    for rest in itertools.product(range(K), repeat=N - 1):
        a = (0,) + rest
        loads = [0] * K
        ok = True
        for i, k in enumerate(a):
            loads[k] += d[i]
            if loads[k] > inst.Q:
                ok = False
                break
        if ok:
            yield a


def model_optima(f: Formulation, tol: float = 1e-9) -> Tuple[float, List[Assignment]]:
    """All optimal feasible assignments (one representative per clustering)."""
    inst = f.inst
    n_enum = inst.K ** (inst.N - 1)
    if n_enum > MAX_ENUM_ASSIGNMENTS:
        raise ValueError(
            f"K^(N-1) = {n_enum} assignments is too many to enumerate; use "
            "cvrp.exact.exact_partition(N, K, f.cluster_cost) (set-partition DP) "
            "for additive objectives instead")
    best = math.inf
    arg: List[Assignment] = []
    seen: set = set()
    for a in _enumerate_feasible_assignments(inst):
        c = clustering_of(a)
        if c in seen:
            continue
        seen.add(c)
        v = float(f.objective(a))
        if v < best - tol:
            best, arg = v, [a]
        elif abs(v - best) <= tol:
            arg.append(a)
    if not arg:
        raise ValueError("instance has no capacity-feasible assignment")
    return best, arg


def model_optimum(f: Formulation) -> Tuple[float, Assignment]:
    """(optimal objective value, one optimal assignment) by enumerating the
    K^(N-1) label-canonical assignments with ``capacity_ok``.  Raises for
    large N*K with a pointer to ``cvrp.exact``."""
    v, args = model_optima(f)
    return v, args[0]


# --------------------------------------------------------------------------- #
# Brute-force checker
# --------------------------------------------------------------------------- #
MAX_CHECK_VARS = 20


@dataclass
class WeightMargin:
    weight: str
    value: float
    ok_at_half: bool
    ok_at_double: bool
    halvings_until_failure: Optional[int]   # None = still sound after `max_halvings` halvings
    max_halvings: int


@dataclass
class CheckReport:
    formulation: str
    instance: str
    n_vars: int
    weights: Dict[str, float]
    n_bitstrings: int
    feasible_fraction: float                 # bitstrings whose decode is capacity-feasible
    decodable_fraction: float                # bitstrings whose decode is not None
    model_optimum: float
    model_optimum_assignment: Assignment
    n_optimal_clusterings: int
    min_energy: float
    min_feasible_energy: float
    min_infeasible_energy: float             # +inf if every bitstring is feasible
    energy_gap: float                        # min_infeasible - min_feasible  (> 0 is sound)
    n_argmin_bitstrings: int
    argmin_decodes_feasible: bool            # every energy-argmin bitstring is feasible
    argmin_all_same_clustering: bool
    argmin_is_model_optimum: bool            # every argmin decodes to an optimal assignment
    argmin_assignment: Optional[Assignment]
    argmin_objective: Optional[float]
    gap_landscape: Dict[str, Any] = field(default_factory=dict)
    penalty_margin: List[WeightMargin] = field(default_factory=list)

    @property
    def sound(self) -> bool:
        return self.argmin_decodes_feasible and self.argmin_is_model_optimum

    @property
    def ok(self) -> bool:  # alias used by tests
        return self.sound

    def summary(self) -> str:
        L = []
        L.append(f"brute-force check: formulation={self.formulation!r} instance={self.instance!r} "
                 f"n_vars={self.n_vars} (2^n = {self.n_bitstrings}) weights={self.weights}")
        L.append(f"  decodable fraction      : {self.decodable_fraction:.4f}")
        L.append(f"  feasible fraction       : {self.feasible_fraction:.4f}")
        L.append(f"  model optimum           : {self.model_optimum:.4f}  assignment={self.model_optimum_assignment}"
                 f"  ({self.n_optimal_clusterings} optimal clustering(s))")
        L.append(f"  min energy              : {self.min_energy:.4f}  over {self.n_argmin_bitstrings} argmin bitstring(s)")
        L.append(f"  argmin assignment       : {self.argmin_assignment}  objective={self.argmin_objective}")
        L.append(f"  argmin decodes feasible : {self.argmin_decodes_feasible}")
        L.append(f"  argmin same clustering  : {self.argmin_all_same_clustering}")
        L.append(f"  argmin is model optimum : {self.argmin_is_model_optimum}")
        L.append(f"  min feasible energy     : {self.min_feasible_energy:.4f}")
        L.append(f"  min infeasible energy   : {self.min_infeasible_energy:.4f}")
        L.append(f"  energy gap (inf - feas) : {self.energy_gap:.4f}   {'OK (>0)' if self.energy_gap > 0 else 'BROKEN (<=0)'}")
        gl = self.gap_landscape
        if gl:
            L.append(f"  landscape over {gl['n_points']} feasible assignment(s): "
                     f"pearson={gl['pearson']:.4f} spearman={gl['spearman']:.4f}")
            L.append("    rank  energy      objective   gap%     assignment")
            for row in gl["table"]:
                L.append(f"    {row['energy_rank']:>4d}  {row['energy']:>10.4f}  {row['objective']:>10.4f}  "
                         f"{row['gap_pct']:>6.2f}   {row['assignment']}")
        if self.penalty_margin:
            L.append("  penalty margins (sound = argmin feasible AND optimal):")
            for m in self.penalty_margin:
                h = ("never fails within %d halvings" % m.max_halvings if m.halvings_until_failure is None
                     else f"fails after {m.halvings_until_failure} halving(s) (x 1/{2 ** m.halvings_until_failure})")
                L.append(f"    {m.weight}={m.value:g}: ok_at_half={m.ok_at_half} ok_at_double={m.ok_at_double}; {h}")
        L.append(f"  => {'SOUND' if self.sound else 'NOT SOUND'}")
        return "\n".join(L)


def _decode_all(f: Formulation, X: np.ndarray) -> np.ndarray:
    """(M, N) int array of assignments, -1 rows where decode is None.
    Uses ``f.decode_many`` when the formulation provides one."""
    dm = getattr(f, "decode_many", None)
    if dm is not None:
        return np.asarray(dm(X), dtype=int)
    N = f.inst.N
    out = np.full((X.shape[0], N), -1, dtype=int)
    for m in range(X.shape[0]):
        a = f.decode(X[m])
        if a is not None:
            out[m] = a
    return out


def _feasible_mask(inst: Instance, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(decodable mask, feasible mask) for an (M, N) assignment array."""
    decodable = (A >= 0).all(axis=1)
    feas = np.zeros(A.shape[0], dtype=bool)
    if decodable.any():
        Ad = A[decodable]
        onehot = (Ad[:, :, None] == np.arange(inst.K)[None, None, :])     # (m, N, K)
        loads = np.einsum("i,mik->mk", np.asarray(inst.demands, float), onehot)
        feas[decodable] = (loads <= inst.Q + 1e-9).all(axis=1)
    return decodable, feas


def _core_check(f: Formulation, opt_value: float, opt_clusterings: set, tol: float):
    """Everything except the penalty-margin scan.  Returns a dict of fields."""
    n = f.n_vars
    X = all_bitstrings(n)
    E = f.energies(X)
    A = _decode_all(f, X)
    decodable, feas = _feasible_mask(f.inst, A)
    inst = f.inst

    Emin = float(E.min())
    atol = tol * max(1.0, abs(Emin))
    arg_idx = np.flatnonzero(E <= Emin + atol)
    arg_feas = bool(feas[arg_idx].all())
    arg_clusterings = {clustering_of(tuple(int(k) for k in A[m])) for m in arg_idx if decodable[m]}
    arg_same = arg_feas and len(arg_clusterings) == 1
    arg_opt = arg_feas and all(c in opt_clusterings for c in arg_clusterings)
    m0 = int(arg_idx[0])
    arg_assign = tuple(int(k) for k in A[m0]) if decodable[m0] else None
    arg_obj = float(f.objective(arg_assign)) if arg_assign is not None else None

    min_feas = float(E[feas].min()) if feas.any() else math.inf
    min_inf = float(E[~feas].min()) if (~feas).any() else math.inf
    gap = min_inf - min_feas

    # landscape: per feasible assignment, the lowest energy of any bitstring decoding to it
    landscape: Dict[str, Any] = {}
    if feas.any():
        best_e: Dict[Assignment, float] = {}
        for m in np.flatnonzero(feas):
            a = tuple(int(k) for k in A[m])
            e = float(E[m])
            if e < best_e.get(a, math.inf):
                best_e[a] = e
        assigns = list(best_e)
        es = np.array([best_e[a] for a in assigns])
        objs = np.array([float(f.objective(a)) for a in assigns])
        order = np.argsort(es, kind="stable")
        pearson = _pearson(es, objs)
        spearman = _pearson(_ranks(es), _ranks(objs))
        table = []
        shown: set = set()
        for rank, idx in enumerate(order, start=1):
            c = clustering_of(assigns[idx])
            if c in shown:
                continue
            shown.add(c)
            table.append({"energy_rank": rank, "energy": float(es[idx]), "objective": float(objs[idx]),
                          "gap_pct": common.gap_percent(float(objs[idx]), opt_value) if opt_value else float("nan"),
                          "assignment": assigns[idx]})
            if len(table) >= 6:
                break
        landscape = {"n_points": len(assigns), "pearson": pearson, "spearman": spearman, "table": table}

    return dict(
        n_vars=n, n_bitstrings=int(X.shape[0]),
        feasible_fraction=float(feas.mean()), decodable_fraction=float(decodable.mean()),
        min_energy=Emin, min_feasible_energy=min_feas, min_infeasible_energy=min_inf, energy_gap=gap,
        n_argmin_bitstrings=int(arg_idx.size), argmin_decodes_feasible=arg_feas,
        argmin_all_same_clustering=arg_same, argmin_is_model_optimum=arg_opt,
        argmin_assignment=arg_assign, argmin_objective=arg_obj, gap_landscape=landscape,
    )


def _ranks(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), as Spearman needs."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _sound_with_weights(f: Formulation, weights: Dict[str, float], opt_value: float, opt_clusterings: set, tol: float) -> bool:
    g = type(f)(f.inst, **weights)
    r = _core_check(g, opt_value, opt_clusterings, tol)
    return bool(r["argmin_decodes_feasible"] and r["argmin_is_model_optimum"])


def brute_force_check(f: Formulation, margins: bool = True, max_halvings: int = 10, tol: float = 1e-9) -> CheckReport:
    """Enumerate all 2^n bitstrings (n <= 20) and audit the QUBO against the model.

    * ``feasible_fraction``: share of bitstrings decoding to a capacity-feasible assignment.
    * ``argmin_decodes_feasible`` / ``argmin_is_model_optimum``: every lowest-energy
      bitstring is feasible / is a model optimum (clusterings compared modulo relabeling).
    * ``energy_gap``: E(best infeasible) - E(best feasible); must be > 0 for a sound penalty.
    * ``gap_landscape``: Pearson/Spearman between min-energy-per-assignment and the
      objective over feasible assignments, plus the best few by energy with their gap %.
    * ``penalty_margin``: for each weight, soundness after halving / doubling it alone,
      and how many halvings it survives.
    """
    n = f.n_vars
    if n > MAX_CHECK_VARS:
        raise ValueError(f"n_vars={n} > {MAX_CHECK_VARS}: brute force is for shrunken instances only")
    opt_value, opt_args = model_optima(f, tol)
    opt_clusterings = {clustering_of(a) for a in opt_args}
    core = _core_check(f, opt_value, opt_clusterings, tol)

    pm: List[WeightMargin] = []
    if margins:
        for w, val in f.weights.items():
            def ok(factor: float) -> bool:
                ww = dict(f.weights)
                ww[w] = val * factor
                return _sound_with_weights(f, ww, opt_value, opt_clusterings, tol)
            ok_half = ok(0.5)
            ok_double = ok(2.0)
            fails_at: Optional[int] = None
            if not ok_half:
                fails_at = 1
            else:
                for h in range(2, max_halvings + 1):
                    if not ok(0.5 ** h):
                        fails_at = h
                        break
            pm.append(WeightMargin(w, float(val), ok_half, ok_double, fails_at, max_halvings))

    return CheckReport(
        formulation=f.name, instance=f.inst.name, weights=dict(f.weights),
        model_optimum=opt_value, model_optimum_assignment=opt_args[0],
        n_optimal_clusterings=len(opt_clusterings), penalty_margin=pm, **core,
    )


# --------------------------------------------------------------------------- #
# CLI: numbers for DERIVATIONS.md §3
# --------------------------------------------------------------------------- #
def _main(argv: List[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="brute-force check a registered formulation on a toy tier-S instance")
    p.add_argument("--formulation", default="textbook")
    p.add_argument("--tier", default="S")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--instance", help="instance JSON (overrides --tier/--seed)")
    p.add_argument("--weight", action="append", default=[], metavar="NAME=VALUE")
    a = p.parse_args(argv)
    inst = Instance.load(a.instance) if a.instance else make_toy_instance(a.tier, a.seed)
    weights = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.weight}
    f = get_formulation(a.formulation, inst, **weights)
    print(f"instance {inst.name}: N={inst.N} K={inst.K} Q={inst.Q} demands={inst.demands} "
          f"locations={[tuple(round(v, 2) for v in p) for p in inst.locations]}")
    print(f"linear terms h: min={f.linear().min():.4f} max={f.linear().max():.4f}; "
          f"couplings J: min(nonzero)={f.couplings()[f.couplings() != 0].min():.4f} max={f.couplings().max():.4f}; "
          f"all couplings >= 0: {bool((f.couplings() >= 0).all())}")
    print(brute_force_check(f).summary())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
