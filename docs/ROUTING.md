# Track B routing — first slice: two tour encodings, exhaustive checks, a 4-customer pilot

**Status: standalone fixtures, not an end-to-end Track B run.** Everything below is classical
algebra plus *emulation* (`cvrp.svsim`, exact state vector) on hand-made 4/5/6-customer
fixtures (`routing/fixtures.py`). Nothing ran on hardware; no Part A clustering has been routed
through this code yet. Numbers come from the records named in §6, produced by
`scripts/routing_demo.py`.

Code: `routing/encodings.py` (builders + strict decoders), `routing/check.py` (exhaustive
checks), `routing/fixtures.py`, `routing/pilot.py`, `scripts/routing_demo.py`,
`tests/test_routing_encodings.py`, `tests/test_routing_pilot.py`. Spec followed: the Part B
implementation handoff (24 Oct 2026); its equations are reproduced here so the code can be
checked against the text.

---

## 1. What we need from a routing QUBO

A cluster is a set of `n` customers plus the depot (local index 0; original customer ids are
kept separately by `RoutingFixture`). We want the shortest closed route
depot → customers → depot. The `AnalogDevice` realises, at the end of a pulse with Ω = 0,

    E_dev(x) = −δ_final Σ_u x_u + Σ_{u<v} U_uv x_u x_v ,     U_uv = C6 / r_uv⁶ > 0 .

So a QUBO `E = offset + Σ h_u x_u + Σ_{u<v} J_uv x_u x_v` is a *candidate* for the device only if

* **all `h_u` are equal** (one global detuning, no DMM on `AnalogDevice`), and
* **all `J_uv ≥ 0`** (the Rydberg interaction is repulsive; a negative coupling cannot exist).

Both conditions are necessary, not sufficient: `J_uv = 0` still means a small positive `U_uv`
(atoms far apart), and the wanted `J_uv > 0` have to be simultaneously realisable by a planar
set of distances. The two encodings below satisfy the necessary conditions by construction;
§5 measures how badly the sufficient one fails for a generic embedding.

Repo convention (`cvrp.common`): we also export a symmetric `Q` with `E = xᵀQx + offset`,
`Q_uu = h_u`, `Q_uv = J_uv / 2`. The pair terms are accumulated once per *unordered* pair and
stored symmetrically, so nothing is double counted.

## 2. Encoding 1 — depot-fixed position one-hot (n² bits)

Variables `y_{i,t}`: customer `i ∈ {1..n}` is visited at position `t ∈ {1..n}`. The depot has
**no** variables: it is at position 0 and position n+1 by definition. This saves the
`2n+1` bits that the textbook `(n+1)²` encoding spends on the depot.

### 2.1 Folding the depot legs into the transitions

The textbook cost has quadratic terms `c_ij y_{i,t} y_{j,t+1}` for the customer-to-customer legs
and *linear* boundary terms `c_0i y_{i,1}` and `c_i0 y_{i,n}` for the depot legs. Linear
terms that differ between variables would break the uniform-`h` requirement, so we move them
into the quadratic terms. For `n ≥ 2`, `i ≠ j`, `t = 1..n−1`:

    w(i,j,t) = c_ij + [t = 1]·c_0i + [t = n−1]·c_j0

    F_pair = Σ_{t=1}^{n−1} Σ_{i≠j} w(i,j,t) y_{i,t} y_{j,t+1}

On a permutation (every row and column of `y` has exactly one 1) each transition `t → t+1` is
selected exactly once. The transition out of position 1 carries the outgoing depot leg of the
customer in position 1; the transition into position n carries the returning leg of the customer
in position n; for `n = 2` the single transition carries both. Hence **`F_pair` equals the full
depot-to-depot tour length on every valid string** — this is exact, not an approximation. It
*does* change the energies of invalid strings (a string with no customer in position 1 pays no
depot leg), which is harmless because those strings are pushed up by the penalties anyway.

### 2.2 Penalties and the expansion

    E = F_pair + A Σ_i (Σ_t y_{i,t} − 1)² + A Σ_t (Σ_i y_{i,t} − 1)²

Expanding one squared one-hot constraint with `y² = y`:

    (Σ_t y_t − 1)² = Σ_t y_t + 2 Σ_{t<t'} y_t y_t' − 2 Σ_t y_t + 1 = −Σ_t y_t + 2 Σ_{t<t'} y_t y_t' + 1

Each variable sits in exactly one row constraint and one column constraint, so

* **offset** = `nA` (rows) + `nA` (columns) = **`2nA`**;
* **`h_u` = −A − A = −2A** for every variable — *uniform by construction*;
* **`J += 2A`** for two positions of the same customer, and for two customers in the same position;
* **`J += w(i,j,t) ≥ 0`** for `(i,t),(j,t+1)` with `i ≠ j`;
* all other pairs: `J = 0` (non-adjacent positions, different customers).

Every coupling is a sum of non-negative pieces, so **`J ≥ 0`**. A valid tour string has energy
exactly its tour cost (`routing.check` verifies `|E − cost| < 1e-7` for every permutation on
every seed, n = 2, 3, 4).

### 2.3 Why `A = C_ub·(1 + 1 %)` is enough

`F_pair ≥ 0` on every string (all `w ≥ 0`). Every invalid string violates at least one one-hot
constraint by an integer, so it pays at least `A·1² = A`. A valid tour of cost `C_ub` exists
(we take the nearest-neighbour tour from the depot, `routing.encodings.nearest_neighbour_tour`).
Therefore with `A > C_ub ≥ C_opt` no invalid string can tie or beat the best tour, and the
ground states are exactly the optimal tours. The bound needs no knowledge of the optimum, and
we do **not** tune `A` against held-out optimal tours (a smaller `A` would help the physical
energy scale, but would need its own soundness argument). The exhaustive check also records
`min_infeasible_energy > C_opt` on every instance.

Decoding is strict: the `n×n` bit matrix must be a permutation matrix; anything else is
"infeasible" and never contributes a cost. Two strings (the two orientations) describe the same
undirected tour; `canonical_tour` identifies them when counting distinct optima.

## 3. Encoding 2 — undirected edge variables (n(n+1)/2 bits, n ≤ 4)

One bit `x_e` per undirected edge `e = {u,v}` of the complete graph on the `m = n+1` vertices
(depot included). Degree `d_v = Σ_{e∋v} x_e`. A tour is a set of edges in which every vertex has
degree exactly 2 **and** the edges form a single cycle.

### 3.1 Cost spread over the two endpoints

A plain cost term `Σ_e c_e x_e` is linear and non-uniform. Instead each selected edge is charged
half its length *at each endpoint*, through pairs of edges that meet there:

    F_pair = Σ_v Σ_{unordered distinct e,f ∋ v} (c_e + c_f)/2 · x_e x_f

When `d_v = 2` for every `v`, the two edges `e, f` meeting at `v` contribute `(c_e + c_f)/2`,
i.e. each of the two edges at `v` contributes half its length; summed over both endpoints,
every selected edge contributes exactly `c_e`. So **`F_pair` = tour length on every degree-2
string**.

### 3.2 Penalty and expansion

    E = F_pair + A Σ_v (d_v − 2)²

`(d_v − 2)² = d_v² − 4d_v + 4 = Σ_{e∋v} x_e + 2 Σ_{e<f ∋ v} x_e x_f − 4 Σ_{e∋v} x_e + 4`, hence per
vertex: `−3 x_e` for each incident edge, `+2 x_e x_f` for each incident pair, `+4`. Every edge
has two endpoints, so

* **offset = 4Am**, **`h_e = −6A`** for every edge (uniform),
* **`J_ef = 2A + (c_e + c_f)/2 > 0`** when `e, f` share a vertex (in a simple graph two distinct
  edges share at most one vertex, so this is added once),
* **`J_ef = 0`** when `e, f` are disjoint.

`A > C_ub` works by the same argument as §2.3 (a degree violation is an integer, costs ≥ A).

### 3.3 Why this is exact only up to n = 4

A degree-2 subgraph is a disjoint union of cycles, each on ≥ 3 vertices. On `m = 5` vertices
(`n = 4`) two cycles would need ≥ 6 vertices, so every degree-2 string *is* a Hamiltonian cycle
and the 10-bit model is an exact TSP encoding. On `m = 6` (`n = 5`) two disjoint triangles are a
degree-2 configuration, and if they are cheaper than every tour the ground state is a **cycle
cover, not a tour**. `routing.check.edge_relaxation_failure` builds exactly that instance
(fixture `tri5_subtour`: depot + 2 customers in a unit triangle, 3 customers in another unit
triangle 100 away) and enumerates all 2¹⁵ strings:

| quantity | value |
|---|---|
| ground state (15 bits) | `110001000000111` = edges {0-1, 0-2, 1-2} ∪ {3-4, 3-5, 4-5} |
| all degrees 2 | yes |
| connected components | `[0,1,2]`, `[3,4,5]` — two triangles |
| decodes to a tour | **no** |
| ground energy | 6.000 (two unit triangles) |
| best energy of any real tour | 203.000 (= `tsp_exact`) |

Post-selecting connected samples does not repair this: the device would be asked to find a
state whose energy sits 197 units above the ground state. Therefore `build_edge` **refuses
`n > 4`** unless called with `allow_subtours=True` (only `routing.check` does). The position
encoding has no such failure (its optimum on the same instance *is* the tour); it is the general
formulation, at the price of `n²` bits (n² > 80 at n = 9).

Decoding is strict: every degree 2, one cycle through all `m` vertices starting and ending at the
depot; the returned tour is canonicalised (second vertex < last vertex), so the 12 undirected
tours of n = 4 are 12 distinct strings.

## 4. Exhaustive algebraic checks (classical, not a simulation)

`tests/test_routing_encodings.py` / `routing.check.check_seeds`: for n = 2, 3, 4 and three
coordinate seeds each, both encodings, all `2^{n_vars}` strings (up to 2¹⁶):

* `E(x_tour) = tour cost` for every permutation (max abs error < 1e-7);
* every decodable string's energy equals its decoded tour's cost;
* `h` uniform, every off-diagonal `J ≥ 0`, `Q` symmetric and consistent with `(h, J, offset)`;
* ground energy = `cvrp.common.tsp_exact` = brute force over permutations;
* the set of ground-state tours equals the set of optimal undirected tours (2 ground strings per
  optimal tour for the directed position model, 1 for the edge model);
* every infeasible string has energy > `C_opt`.

All 18 (encoding, n, seed) cases pass; `n_optimal_undirected_tours = 1` on all random seeds. The
tie fixture `hex6_ties` (regular hexagon around the depot) has 6 tied optimal tours and each of
them has energy `C_opt` in the position encoding (36 bits, not enumerated).
`13 passed, 0 failed` with `PYTHONPATH=/home/claude/_pydeps:. python3 scripts/run_tests.py routing_`.

## 5. Pilot on the 4-customer fixture `quad4`

`PYTHONPATH=/home/claude/_pydeps:. python3 scripts/routing_demo.py` (1000 shots, seed 0,
embed seeds 0–5 with 8 optimiser starts each, selected by **embedding fidelity only**).
Fixture: depot (35, 42), customers ids 101–104 at (10,20), (90,30), (80,85), (20,70);
`C_opt = 258.334`, unique optimal tour `[1, 4, 3, 2]` (= `tsp_exact`), nearest-neighbour
`C_ub = 281.52`, `A = 284.34`. Pipeline: `routing.encodings` → `cvrp.embed.embed` (generic,
`J = 0` pairs approximated by far atoms and reported as the largest spurious coupling) →
`cvrp.pulse.default_for` (5800 ns, Ω = 0.75 Ω_max, linear δ sweep to `δ_final = −α h`) →
`pulser.Sequence` on `AnalogDevice` (built, `to_abstract_repr()`, `cvrp.pulse.validate` = no
violations, layout attached) → `cvrp.svsim` exact state vector (4 ns slices).

Three layers, reported separately:

(generated into `out/routing/pilot_quad4.md`; copied verbatim)

| encoding | atoms | target GS optimal | rel-Frob | spurious J (QUBO) | device argmin decodes | rank(opt) | raw feasible | P(opt) | best gap % | rand-bits feasible | rand-bits P(opt) | rand-tour P(opt) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge | 10 | True | 0.794 | 389 / 2A=569 | False | 114 of 1024 | 0.001 | 0.000 | 15.0 | 0.014 | 0.003 | 0.083 |
| position | 16 | True | 0.979 | 10 / 2A=569 | False | 603 of 65536 | 0.000 | 0.000 | none | 0.000 | 0.000 | 0.083 |

Columns: *target GS optimal* = the QUBO's exhaustive ground states decode to the `tsp_exact` tour; *rel-Frob* = `cvrp.embed` fidelity ‖U/α − J‖_F/‖J‖_F; *spurious J* = largest realised coupling on a `J = 0` pair, against the 2A the wanted pairs carry; *device argmin decodes* / *rank(opt)* = zero-drive device energies `E_dev(x)` of **all** strings, rank 1 = lowest; *raw feasible*, *P(opt)*, *best gap* = 1000 svsim shots, strict decode, infeasible shots carry no cost; the last three columns are the two baselines (1000 uniform random bitstrings; uniform random valid tours, analytic 1/12 for a unique n = 4 optimum).

Reading of the table:

1. **Formulation**: both QUBOs have the optimal tour as their unique (undirected) ground state —
   the algebra is right.
2. **Geometry**: the generic planar embedding does not reproduce either QUBO. Edge model: the
   coupling graph is the line graph of K₅ (every variable must sit at the *same* distance from
   six others and far from three); rel. Frobenius error 0.79, and a `J = 0` pair receives
   389 QUBO units (target couplings are ≈ 2A = 569). Under the realised zero-drive device
   energies the optimal tour ranks 114th of 1024 strings; no valid tour is among the 12 lowest
   device states. Position model: rel. Frobenius 0.98 (worst wanted pair off by 147 %), spurious
   couplings small (10 units) but the wanted ones wrong; the two orientations of the optimal
   tour rank 603rd/604th of 65 536 and the best valid tour 453rd. The restart sweep (6 embed
   seeds, selected by fidelity only) never makes the device argmin a tour: edge ranks 40–368
   (seed 4 has the smallest spurious coupling, 42, and rank 40, but worse Frobenius fidelity, so
   the rule picks seed 0); position seeds are all within 1e-5 of each other (rank 602–603).
   Correlation between device and target energies over all strings: 0.63 (edge), 0.91 (position).
3. **Dynamics**: the Part A pulse *does* what it is asked. For the 10-atom edge register the most
   probable sample (20.6 %) is exactly the embedded register's own classical ground state
   `1001010011` (5 edges, degrees 2,2,1,2,3); for the 16-atom position register the most probable
   sample (12.9 %) is again the device argmin `1010101000010101` (a blockade "checkerboard": two
   positions for three of the customers). The adiabatic sweep works on the Hamiltonian it was
   given; that Hamiltonian is just not the TSP. Edge: raw feasibility 0.1 % (exact 0.04 %),
   P(optimal) ≈ 2e-7, best feasible gap 15 % from a single shot. Position: 0 feasible shots in
   1000 (exact P(feasible) 2e-7, P(optimal) 1e-9), best feasible cost `null`. Both are worse than
   uniform random *valid tours* (feasible 100 %, P(opt) = 1/12) and no better than uniform random
   bitstrings (edge: 1.4 % feasible, 0.3 % optimal; position: 0 of 1000 feasible). No quantum
   advantage claim can be made from these numbers, and none is. Runtime: 2 s (10 atoms, dense)
   and 140 s (16 atoms, Krylov) for the exact state-vector evolution on this sandbox; emu-sv on
   the Windows venv is not needed for n = 4.

Conclusion for the next step (as the handoff says): *do not spend hours on pulses for a bad
physical cost function*. The lever is the embedding — a structure-aware register (like
`cvrp.embed_onehot` does for Part A one-hot blocks), a smaller legal `A` with its own soundness
proof, or accepting that the edge model's line-graph couplings have no planar 1/r⁶ realisation.

## 6. Records

| file | content |
|---|---|
| `out/routing/pilot_quad4_edge.json` | edge model: targets, embedding (+ restart sweep), device energies, baselines, samples, raw counts |
| `out/routing/pilot_quad4_position.json` | position model, same layout |
| `out/routing/seq_quad4_{edge,position}.json` | validated Pulser sequences (`to_abstract_repr()`), consumable by emu-sv on the Windows venv |
| `out/routing/register_quad4_{edge,position}.json` | registers with variable names, α, δ_final, target and realised couplings |
| `out/routing/pilot_quad4.md` | the table above, generated |

`out/` is gitignored; regenerate with one command (`scripts/routing_demo.py`, ≈ 3 min on this
sandbox including the 16-atom exact run; `--position-emulation no` skips it and only writes the
sequence for emu-sv).

## 7. Not done / open

* No Part A clusterings routed; no `solve_cluster(...)` integration boundary (disjointness /
  coverage / capacity validation) yet — the fixtures are hand made.
* No structure-aware embedding for either routing encoding; the generic embedder is the
  measured failure point.
* No pulse tuning (one Part A default pulse); no equal-budget comparison beyond this single run.
* No n = 5 position (25 atoms) run; it needs emu-mps with convergence checks on the Windows venv.
* No hardware execution anywhere.
