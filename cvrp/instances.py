"""Instance generator for the CVRP challenge (brief §2, Task A2).

Every instance is produced from a single integer ``seed`` with
``numpy.random.default_rng(seed)``, so the committed JSON files in
``instances/`` can be regenerated bit-for-bit by anyone (see
``instances/SEEDS.md`` for the exact command).

How feasibility is guaranteed (brief: "state which you used")
--------------------------------------------------------------
We use a **certificate-first** construction:

1. Draw customer locations uniformly in the square ``[0, 100]^2``; the depot is
   fixed at the centre ``(50, 50)``.
2. Draw a customer -> vehicle assignment *before* the demands.  Group sizes are
   kept nearly balanced (they differ by at most one) and then the customers are
   shuffled into the groups, so every vehicle gets at least one customer
   whenever ``N >= K``.
3. Draw the demands vehicle by vehicle.  Inside one vehicle the customers are
   handled in order; each customer's demand is drawn uniformly from an integer
   range whose upper end never exceeds what is left of the capacity ``Q`` once
   every later customer of the same vehicle has been reserved its minimum
   demand ``d_min``.  Hence the vehicle's total load can never exceed ``Q``
   and the assignment of step 2 is a valid *certificate* of feasibility.
   The validator (``cvrp.validator``) re-checks that certificate.

Making the instances non-trivial
--------------------------------
A capacity constraint only matters when the fleet is reasonably full.  The
draw of step 3 is already "uniform within what still fits", which on its own
gives load ratios ``rho = sum(d) / (K * Q)`` of roughly 0.6-0.95 for the
default seeds.  Two knobs exist to push further: ``demand_bias = b`` moves the
lower end of the draw range from ``d_min`` to ``d_min + b * (hi - d_min)``
(rounded up; ``b = 0`` is the plain uniform draw used by default, ``b = 1``
always takes the largest demand that still fits), and if the load ratio ends
up below ``rho_min`` (default 0.6) the whole instance is redrawn from the
*same* random stream (so the result is still a deterministic function of the
seed); the number of attempts is recorded.  We deliberately do *not* push rho
to 1: at rho = 1 the certificate is essentially the only feasible clustering
and the optimisation problem becomes trivial.

All the knobs are written into ``Instance.generator`` together with the
generator version so that a JSON file documents how it was made.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from cvrp.common import SQUARE_SIDE, TIERS, Instance

GENERATOR_VERSION = "1.0"
FEASIBILITY_METHOD = ("certificate-first: assignment drawn, then demands drawn "
                      "within each vehicle's remaining capacity")
DEFAULT_SEEDS: Dict[str, List[int]] = {"S": [0, 1, 2, 3, 4], "M": [0, 1, 2], "L": [0, 1, 2]}

# Default knobs (also recorded in every JSON so nothing is implicit).
DEFAULTS: Dict[str, Any] = {
    "demand_bias": 0.0,        # 0 = uniform over the whole admissible range (0.5 = upper half)
    "rho_min": 0.6,            # redraw until the load ratio is at least this
    "balanced_sizes": True,    # certificate groups differ in size by at most one
    "coordinate_decimals": 2,  # locations are rounded so JSON <-> float is exact
    "max_attempts": 1000,
}


def instance_name(tier: str, seed: int) -> str:
    """File stem / instance name convention: ``S_seed03``."""
    return f"{tier}_seed{seed:02d}"


# --------------------------------------------------------------------------- #
# Building blocks (each is a pure function of the rng state)
# --------------------------------------------------------------------------- #
def _draw_locations(rng: np.random.Generator, N: int, side: float, decimals: int,
                    depot: Tuple[float, float]) -> List[Tuple[float, float]]:
    """N distinct points in [0, side]^2, none equal to the depot.

    Rounding to ``decimals`` places keeps the JSON round trip exact; the
    (astronomically unlikely) coincidence of two rounded points is handled by
    redrawing the offending point.
    """
    pts: List[Tuple[float, float]] = []
    seen = {tuple(depot)}
    while len(pts) < N:
        x, y = rng.uniform(0.0, side, size=2)
        p = (round(float(x), decimals), round(float(y), decimals))
        if p in seen:
            continue
        seen.add(p)
        pts.append(p)
    return pts


def _draw_assignment(rng: np.random.Generator, N: int, K: int, balanced: bool) -> List[int]:
    """One customer -> vehicle map; this becomes the feasibility certificate.

    ``balanced=True``: group sizes are N // K or N // K + 1 (the extra customers
    go to randomly chosen vehicles).  ``balanced=False``: the first K shuffled
    customers seed one vehicle each, the rest pick a vehicle uniformly.
    """
    perm = rng.permutation(N)
    a = [0] * N
    if balanced:
        sizes = np.full(K, N // K, dtype=int)
        extra = rng.choice(K, size=N % K, replace=False) if N % K else []
        sizes[list(extra)] += 1
        pos = 0
        for k in range(K):
            for i in perm[pos:pos + sizes[k]]:
                a[int(i)] = k
            pos += int(sizes[k])
    else:
        for k, i in enumerate(perm[:K]):
            a[int(i)] = k
        for i in perm[K:]:
            a[int(i)] = int(rng.integers(0, K))
    return a


def _draw_demands(rng: np.random.Generator, a: Sequence[int], K: int, Q: int,
                  d_min: int, d_max: int, bias: float) -> List[int]:
    """Demands that respect ``Q`` on every vehicle of assignment ``a``.

    For the j-th customer of a vehicle with ``r`` customers still to come after
    it and ``cap`` capacity left, the admissible range is
    ``[d_min, min(d_max, cap - r * d_min)]`` (so the later customers can still
    get d_min each).  Within that range we draw uniformly from
    ``[lo, hi]`` with ``lo = d_min + ceil(bias * (hi - d_min))``.
    """
    d = [0] * len(a)
    for k in range(K):
        members = [i for i, v in enumerate(a) if v == k]
        cap = Q
        for j, i in enumerate(members):
            remaining_after = len(members) - j - 1
            hi = min(d_max, cap - remaining_after * d_min)
            if hi < d_min:  # cannot happen when N <= K*Q/d_min; guard anyway
                raise ValueError("vehicle cannot fit its customers at minimum demand")
            lo = d_min + math.ceil(bias * (hi - d_min) - 1e-12)
            lo = min(max(lo, d_min), hi)
            d[i] = int(rng.integers(lo, hi + 1))
            cap -= d[i]
        assert cap >= 0
    return d


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate(tier: str, seed: int, **opts: Any) -> Instance:
    """Build one reproducible, certified-feasible instance of ``tier``.

    Parameters
    ----------
    tier : "S" | "M" | "L" (see ``cvrp.common.TIERS``)
    seed : int, feeds ``numpy.random.default_rng``
    opts : any key of ``DEFAULTS`` (demand_bias, rho_min, balanced_sizes,
           coordinate_decimals, max_attempts); unknown keys raise.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(TIERS)}")
    unknown = set(opts) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown generator option(s): {sorted(unknown)}")
    cfg = {**DEFAULTS, **opts}
    t = TIERS[tier]
    N, K, Q, d_min, d_max = t["N"], t["K"], t["Q"], t["d_min"], t["d_max"]
    side = SQUARE_SIDE
    depot = (side / 2.0, side / 2.0)

    rng = np.random.default_rng(seed)
    attempts = 0
    while True:
        attempts += 1
        locations = _draw_locations(rng, N, side, cfg["coordinate_decimals"], depot)
        a = _draw_assignment(rng, N, K, cfg["balanced_sizes"])
        demands = _draw_demands(rng, a, K, Q, d_min, d_max, cfg["demand_bias"])
        rho = sum(demands) / (K * Q)
        if rho >= cfg["rho_min"] or attempts >= cfg["max_attempts"]:
            break

    inst = Instance(
        name=instance_name(tier, seed),
        tier=tier, seed=int(seed), N=N, K=K, Q=Q,
        depot=depot,
        locations=locations,
        demands=demands,
        certificate=tuple(a),
        square_side=side,
        depot_rule="centre of the square (50, 50)",
        rho=rho,
        dmax_over_Q=max(demands) / Q,
        feasibility_method=FEASIBILITY_METHOD,
        generator={
            "version": GENERATOR_VERSION,
            "rng": "numpy.random.default_rng(seed)",
            "seed": int(seed),
            "attempts": attempts,
            "demand_range": [d_min, d_max],
            **{k: cfg[k] for k in DEFAULTS},
            "command": f"python -m cvrp.instances --tiers {tier} --seeds {seed} --out instances/",
        },
    )
    assert inst.is_valid_assignment(inst.certificate), "internal error: certificate invalid"
    return inst


def generate_default(out: str | Path = "instances", seeds: Dict[str, List[int]] | None = None,
                     tiers: Sequence[str] = ("S", "M", "L"), **opts: Any) -> List[Instance]:
    """Generate and save the default instance set; returns the instances."""
    seeds = seeds or DEFAULT_SEEDS
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for tier in tiers:
        for s in seeds[tier]:
            inst = generate(tier, s, **opts)
            inst.save(out / f"{inst.name}.json")
            made.append(inst)
    return made


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiers", nargs="+", default=["S", "M", "L"], choices=sorted(TIERS))
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="seeds for every listed tier (default: S 0-4, M 0-2, L 0-2)")
    p.add_argument("--out", default="instances")
    p.add_argument("--demand-bias", type=float, default=DEFAULTS["demand_bias"])
    p.add_argument("--rho-min", type=float, default=DEFAULTS["rho_min"])
    args = p.parse_args(argv)
    seeds = {t: (args.seeds if args.seeds is not None else DEFAULT_SEEDS[t]) for t in args.tiers}
    made = generate_default(args.out, seeds, args.tiers, demand_bias=args.demand_bias, rho_min=args.rho_min)
    for inst in made:
        print(f"{inst.name}: N={inst.N} K={inst.K} Q={inst.Q} rho={inst.rho:.3f} "
              f"dmax/Q={inst.dmax_over_Q:.3f} certificate loads={inst.loads(inst.certificate)} "
              f"attempts={inst.generator['attempts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
