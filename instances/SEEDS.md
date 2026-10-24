# Instance set (deliverable D3)

Every file in this folder is a deterministic function of `(tier, seed)` via
`cvrp/instances.py` (generator version **1.0**, `numpy.random.default_rng(seed)`).
`python -m cvrp.validator instances/` re-generates each one from its seed and
fails if a single coordinate, demand or certificate entry differs.

## Tiers (brief Table 1, fixed in `cvrp.common.TIERS`)

| tier | N customers | K vehicles | Q | demands | seeds | files |
|---|---|---|---|---|---|---|
| S | 4 | 2 | 7 | 1..4 | 0, 1, 2, 3, 4 | `S_seed00.json` .. `S_seed04.json` |
| M | 10 | 3 | 20 | 1..8 | 0, 1, 2 | `M_seed00.json` .. `M_seed02.json` |
| L | 15 | 4 | 30 | 1..10 | 0, 1, 2 | `L_seed00.json` .. `L_seed02.json` |

Depot: centre of the side-100 square, `(50, 50)` (stated in every file as `depot_rule`).
Customer locations: uniform in `[0, 100]^2`, rounded to 2 decimals, pairwise distinct.

## How feasibility is guaranteed — certificate first

Recorded in each file as `feasibility_method`:

> certificate-first: assignment drawn, then demands drawn within each vehicle's remaining capacity

1. a customer → vehicle assignment is drawn **before** the demands (group sizes
   differ by at most one, every vehicle gets ≥ 1 customer);
2. demands are then drawn vehicle by vehicle, uniformly on
   `[d_min, min(d_max, capacity left − d_min × customers still to come)]`,
   so no vehicle can exceed `Q`;
3. that assignment is shipped as `certificate`; the validator re-checks it
   (every customer exactly once, all loads ≤ Q).

Knobs (all written into the `generator` block of each JSON): `demand_bias = 0.0`
(plain uniform draw; `1.0` would always take the largest demand that fits),
`rho_min = 0.6` (redraw from the same stream if the load ratio is lower; every
default instance was accepted on its first attempt), `balanced_sizes = true`,
`coordinate_decimals = 2`.  Load ratios `ρ = Σd/(KQ)` of the default set lie in
0.60–0.95; we deliberately do not push ρ → 1, where the certificate becomes the
only feasible clustering and the optimisation is trivial.

## Exact commands

```bash
python -m cvrp.instances --out instances/                      # all tiers, default seeds (this set)
python -m cvrp.instances --tiers S --seeds 0 1 2 3 4 --out instances/   # equivalent, tier by tier
python -m cvrp.instances --tiers M L --seeds 0 1 2 --out instances/
python -m cvrp.reference instances/                            # fills the `reference` block (C*, #optima, SA cross-check)
python -m cvrp.validator instances/                            # D3 gate: exits 1 on any failure
```

## Reference values (`reference` block, brief §9.3)

`reference_status = "proven_optimum"` for every instance: C* comes from an
exhaustive set-partition dynamic programme over *all* capacity-feasible
clusterings with an exact Held–Karp tour per cluster (`cvrp/exact.py`), which
also counts the number of distinct optimal clusterings modulo vehicle
relabeling (`n_distinct_optima`).  CPLEX: not available.  Simulated annealing
(parameters under `solver_settings.sa`: T₀ = 20, T_f = 0.01, α = 0.995 per sweep
of N moves, ≤ 50 000 moves, 10 s limit, 10 restarts, seeds 0–9) is run on every
instance as a cross-check (`sa_cost`, `sa_matches_optimum`); on tier S a naive
enumeration of all K^N assignments is run as well (`brute_force_naive_cost`, `agree`).
