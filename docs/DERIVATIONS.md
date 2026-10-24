# DERIVATIONS — the single source of truth for the model

Pasqal challenge, Track A: capacity-feasible clustering of customers onto vehicles
(cluster-first / route-second CVRP). **Code follows this file, never the reverse.** Every weight,
encoding, embedding and pulse decision is written here first, then implemented. Written for a
physics undergraduate: no step is skipped. Section numbers are referenced from the code
(`cvrp/model.py`, `cvrp/formulations/`, `cvrp/embed.py`, `cvrp/pulse.py`).

Brief references: `docs/CHALLENGE.pdf` v3.0 — Eq. (1) device Hamiltonian, Derive 2.1 / 3.1 / 4.1,
Table 1 tiers, §3.1 hardware envelope, §9 evaluation.

---

## 0. Notation and conventions

**Sets and indices.** Customers `i, j ∈ {0, …, N−1}`; vehicles `k, l ∈ {0, …, K−1}`; the depot is a
separate point (index 0 of the full distance matrix, `inst.depot`). Slack-bit positions `b ∈ {0, …, B−1}`.

**Binary variables and bitstrings.** A *bitstring* `x ∈ {0,1}^n` is what the register returns (`'1'` =
atom measured in the Rydberg state `|r⟩` = variable 1). Bitstring index `i` = atom `q{i}` = QUBO
variable `i`, slack atoms included.

**QUBO convention** (identical in `cvrp/common.py`, `qopt/`, and here):

    E(x) = xᵀ Q x + offset,      Q symmetric (n×n)
    h_i  = Q_ii                  (linear term of variable i; x_i² = x_i for binaries)
    J_ij = 2 Q_ij  (i ≠ j)       (coupling of the pair; the factor 2 because Q_ij and Q_ji both appear)

so equivalently `E(x) = Σ_i h_i x_i + Σ_{i<j} J_ij x_i x_j + offset`. When a derivation produces a
term `c · x_i x_j` we store `Q_ij = Q_ji = c/2`.

**Assignment vs clustering.** An *assignment* `a ∈ {0..K−1}^N` names the vehicle of every customer.
Vehicles are identical (brief §2, Pitfall 9.3), so two assignments differing only by a relabeling of
vehicles are the *same clustering*: a set of disjoint non-empty customer groups. We always compare
clusterings, never raw labels (`cvrp.common.clustering_of`). A clustering with `k` non-empty groups
appears `K!/(K−k)!` times among assignments.

**Feasible** = the bitstring *decodes* to an assignment (one-hot rows etc. are well formed) **and**
every vehicle load `L_k = Σ_{i: a_i = k} d_i` satisfies `L_k ≤ Q`. "Model optimum" = the feasible
assignment with the smallest clustering objective of the *model* (not the CVRP route length).

**Distances.** `d_ij = ‖p_i − p_j‖₂` between customers (symmetric, triangle inequality, brief
assumption 2); `d_0i` customer–depot. Locations in `[0, 100]²`. **Units for physics**: rad/µs for
energies (`H/ħ`), µm for positions, integer ns for time.

**Tiers** (brief Table 1; `cvrp.common.TIERS`):

| Tier | N | K | Q | demands | intended |
|---|---|---|---|---|---|
| S | 4 | 2 | 7 | {1..4} | pulser-simulation, emu-sv |
| M | 10 | 3 | 20 | {1..8} | emu-mps |
| L | 15 | 4 | 30 | {1..10} | emu-mps or QPU |

---

## 1. The CVRP, formally (Derive 2.1)

**Sets.** `C = {0,…,N−1}` customers, `V = {0,…,K−1}` identical vehicles, depot `0`.

**Parameters.** Locations `p_i ∈ [0,100]²` and `p_depot` (fixed before clustering, stated per instance);
demands `d_i ∈ ℤ_{>0}` with `d_i ≤ Q` (assumption 5); capacity `Q`; symmetric costs `c_uv = ‖p_u − p_v‖₂`.

**Decision variables (full CVRP).** Routing arcs `y^k_{uv} ∈ {0,1}` (vehicle `k` drives `u → v`) and
assignment `x_{ik} ∈ {0,1}` (customer `i` is served by vehicle `k`).

**Objective.** `min Σ_k Σ_{(u,v)} c_uv y^k_{uv}` — total distance driven by the fleet.

**Constraints**, marked **[=]** equality or **[≤]** inequality:

1. **[=] Exactly one vehicle per customer** (served once, never twice): `Σ_k x_{ik} = 1  ∀ i`.
2. **[≤] Capacity**: `Σ_i d_i x_{ik} ≤ Q  ∀ k`. A half-empty vehicle is valid.
3. **[=] Flow conservation / degree**: every customer `i` has exactly one incoming and one outgoing arc
   of the vehicle it is assigned to: `Σ_u y^k_{ui} = Σ_v y^k_{iv} = x_{ik}  ∀ i, k`.
4. **[≤] One trip per vehicle**: `Σ_v y^k_{0v} ≤ 1  ∀ k` (unused vehicle allowed, assumption 6; it leaves
   the depot at most once and returns once, no mid-route reload, assumption 4).
5. **[≤] Subtour elimination**: for every customer subset `S ⊆ C`, `S ≠ ∅`, `Σ_{u,v∈S} y^k_{uv} ≤ |S| − 1 ∀ k`
   (no cycle avoiding the depot).

**The cluster-first / route-second split** (brief §2.2) replaces 3–5 by an exact TSP per vehicle
(`cvrp.common.route_exactly`, Held–Karp) and leaves the **clustering problem** as the quantum part:

    min_x  f(x)                                  (a "closeness" objective, chosen in §3.1 / §4)
    s.t.   Σ_k x_{ik} = 1       ∀ i              [=]   exactly-one
           Σ_i d_i x_{ik} ≤ Q   ∀ k              [≤]   capacity
           x_{ik} ∈ {0,1}

The asymmetry **[=] vs [≤]** is the whole difficulty (brief §2, §4): an equality has one target value
to penalise deviation from; an inequality has a range of allowed values and no single target.

**Instance feasibility** is *not* implied by `Σ d_i ≤ K·Q` (bin packing, brief §2). Every instance ships
with a certificate assignment that the validator checks (`Instance.certificate`, `cvrp.validator`).

---

## 2. What the register minimises (Derive 3.1)

### 2.1 From Eq. (1) to a classical cost

The analog device evolves under (brief Eq. 1)

    H(t)/ħ = Σ_i (Ω(t)/2) σ̂ˣ_i  −  δ(t) Σ_i n̂_i  +  Σ_{i<j} (C6 / ‖r_i − r_j‖⁶) n̂_i n̂_j ,
    n̂_i = |r⟩⟨r|_i  (= 1 if atom i is in the Rydberg state, 0 if in the ground state).

Take a computational-basis state `|x⟩ = |x_0 x_1 … x_{n−1}⟩` with `x_i ∈ {0,1}` (`1` = Rydberg). It is an
eigenstate of every `n̂_i`: `n̂_i |x⟩ = x_i |x⟩`, hence also of every product `n̂_i n̂_j |x⟩ = x_i x_j |x⟩`.
The only term that is *not* diagonal in this basis is `σ̂ˣ_i`, which flips bit `i` (`σ̂ˣ|g⟩ = |r⟩`,
`σ̂ˣ|r⟩ = |g⟩`) and therefore mixes bitstrings: this is the term that "lets the system explore".

**At the end of the pulse `Ω` has been ramped back to 0.** Then `H_final/ħ = −δ_f Σ_i n̂_i + Σ_{i<j} U_ij n̂_i n̂_j`
with `δ_f = δ(t_final)` and `U_ij = C6/r_ij⁶`, `r_ij = ‖r_i − r_j‖`. Every `|x⟩` is an eigenstate and its
eigenvalue is the **classical cost of the bitstring**:

    E_dev(x) = −δ_f Σ_i x_i  +  Σ_{i<j} (C6 / r_ij⁶) x_i x_j            [rad/µs]           (2.1)

Compare with the QUBO form of §0, `E(x) = Σ_i h_i x_i + Σ_{i<j} J_ij x_i x_j + offset`. The register
*is* a QUBO with

    h_i   = −δ_f      for every i   (all linear terms EQUAL),
    J_ij  = C6 / r_ij⁶ > 0          (all couplings POSITIVE),
    offset = 0.

The adiabatic recipe (brief §8.2) prepares the ground state of `H_initial` (all atoms in `|g⟩`, i.e. `x=0`
at large negative `δ`), and if the sweep is slow compared with the instantaneous gap, the final state is
the ground state of `H_final`, i.e. the bitstring that minimises (2.1).

### 2.2 Who controls what

| term in (2.1) | symbol | controlled by | range on `AnalogDevice` (Pulser 1.9.1, `scripts/device_limits.py`) |
|---|---|---|---|
| linear coefficient, the same for every atom | `−δ_f` | **the laser**: final detuning | `|δ| ≤ 125.66 rad/µs` (2π·20 MHz). **Global**: one value for all atoms; no DMM on `AnalogDevice` |
| pairwise coupling | `U_ij = C6/r_ij⁶` | **the atom positions you choose** | `r_ij ≥ 5 µm` (min separation) and all atoms inside a disc of radius 38 µm, so `r_ij ≤ 76 µm`: `C6/76⁶ ≈ 4.5·10⁻⁶ ≤ U_ij ≤ C6/5⁶ = 55.4 rad/µs` |
| interaction constant | `C6` | **nobody**: fixed physics of the Rydberg level | `C6/ħ = 8.657·10⁵ rad/µs·µm⁶` |
| sign of the coupling | `+` | nobody (van der Waals repulsion between two Rydberg atoms) | always positive |
| `Ω`, duration | — | the laser; they set how adiabatic the sweep is, **not** the cost function | `Ω ≤ 12.57 rad/µs`, `T ≤ 6000 ns` |

Three roles, explicitly:

1. **Laser (δ_f)** → *one* uniform linear coefficient `h = −δ_f`. Its sign decides whether excitations
   are favoured (`δ_f > 0`, `h < 0`) or not. It cannot give atom `i` a different `h_i` than atom `j`.
2. **Positions (r_i)** → every coupling, through `1/r⁶`. Couplings are *all the same sign* and
   *geometric*: `J_ij` is not a free matrix but a function of 2D positions, so e.g. three mutually strongly
   coupled atoms must form a small triangle, and a pair can never be made exactly uncoupled (at 76 µm the
   residual is `4.5·10⁻⁶` rad/µs, which is tiny but not zero).
3. **C6** → the overall coupling scale, fixed. It only sets the conversion between "distance" and
   "energy"; together with `r_min = 5 µm` it bounds the largest realisable coupling at 55.4 rad/µs.

### 2.3 Consequence: which cost functions are directly realisable

A cost function `E(x) = Σ h_i x_i + Σ_{i<j} J_ij x_i x_j + offset` is realisable on the device, up to a
positive overall scale `α > 0` (multiplying every energy by `α` does not change which bitstring is lowest),
**if and only if**

* **every linear term is equal**, `h_i = h ∀ i`, with `|α h| ≤ 125.66 rad/µs` (then `δ_f = −α h`);
* **every coupling is strictly positive**, `J_ij > 0`, and `α J_ij ∈ [C6/76⁶, C6/5⁶]`;
* the matrix `(α J_ij)^{-1/6} · C6^{1/6}` is a *Euclidean distance matrix of points in a plane* inside a
  38 µm disc with all pairwise distances `≥ 5 µm`. Not every positive matrix is one: this is the embedding
  problem (§5), and its residual error is part of the result (brief §8.1).

Everything that violates this must be *approximated* (and the approximation error reported) or
*reformulated*. In particular, with a **global detuning, non-uniform linear terms collapse to their mean**:
the device sees `h̄ = (1/n) Σ_i h_i` on every atom and the per-atom differences `h_i − h̄` are simply
lost — which is what kills the textbook formulation (§3.3).

Two things that do **not** matter:

* **The constant `offset`** shifts every level by the same amount; it never changes the ground state and
  is dropped on the device. We keep it in code only so that `E(x)` equals the literal penalty expression
  (`test_textbook_Q_matches_literal_formula_on_random_bitstrings`).
* **Variables that `decode` ignores** (slack bits) still cost an atom each and still shape the energy
  landscape, but their measured value is discarded at readout: a shot is scored only through the
  assignment it decodes to.

### 2.4 Energy scales on the device (for §5–6)

`U(5 µm) = 55.4 rad/µs` is the largest coupling; `Ω_max = 12.57 rad/µs` sets the Rydberg-blockade radius
`R_b = (C6/Ω)^{1/6} ≈ (8.657·10⁵ / 12.57)^{1/6} ≈ 6.4 µm` at full power; `|δ| ≤ 125.66 rad/µs` is the
largest uniform linear term. A cost function must be scaled (`α`) so that its *coupling* range fits in
`[~0, 55.4]` while its linear term fits in `[−125.66, 125.66]`; the ratio `h/J` is fixed by the model and
`α` is the only freedom.

---

## 3. The textbook formulation and its cost (Derive 4.1)

This formulation is **throwaway**: it exists to exercise the plumbing (registry → QUBO → brute-force
check → embed → pulse → decode → score) and to fill the first two rows of the Derive 4.1 table. It is
*not* device-realisable (§3.3). Code: `cvrp/formulations/textbook.py` (`AssignmentOnly`, name
`assignment-only`; `TextbookOneHotSlack`, name `textbook`).

### 3.1 Variables and objective

**One-hot assignment variables** `x_{ik} ∈ {0,1}`, `x_{ik} = 1 ⇔` customer `i` rides on vehicle `k`;
QUBO index `i·K + k`. Count `N·K` (S: 8, M: 30, L: 60).

**Closeness measure** (brief §5.1 offers several; we pick one and say so): **total pairwise distance inside
each cluster**,

    f(x) = Σ_k Σ_{i<j} d_ij x_{ik} x_{jk} .                                                   (3.1)

Why this one: it is a genuine quadratic in the binaries with **no extra variables** (a "distance to a
cluster centre" objective needs centre variables or a non-quadratic form), its couplings `d_ij ≥ 0` are
already of the right sign for the device, and it is **additive over clusters**:
`f = Σ_k cost(cluster_k)` with `cost(S) = Σ_{i<j ∈ S} d_ij` (`Formulation.cluster_cost`), which is
exactly what the exact set-partition DP in `cvrp.exact` needs to produce proven references for all
tiers. Its known bias: it grows like `|S|²`, so it prefers many small clusters — see §3.4 for what this
does to the capacity penalty.

In QUBO form (3.1) contributes `Q[ik, jk] = Q[jk, ik] = d_ij / 2`, i.e. `J_{ik,jk} = d_ij` for `i<j`, same
vehicle, and nothing between different vehicles.

`AssignmentOnly` **is just (3.1)**: no constraints. Its minimum is `x = 0` (serve nobody, `E = 0`), so the
brute-force checker reports it as unsound by design (`test_checker_detects_the_unconstrained_formulation`).

### 3.2 Constraints → penalties, expanded step by step

**(a) Exactly-one [=].** `Σ_k x_{ik} = 1` has a single target, so penalise the squared deviation:

    P_A(x) = A Σ_i ( Σ_k x_{ik} − 1 )²

Expand one customer `i` (write `x_k` for `x_{ik}`):

    ( Σ_k x_k − 1 )² = (Σ_k x_k)² − 2 Σ_k x_k + 1
                      = Σ_k x_k² + 2 Σ_{k<l} x_k x_l − 2 Σ_k x_k + 1          (square of a sum: diagonal + cross terms)
                      = Σ_k x_k   + 2 Σ_{k<l} x_k x_l − 2 Σ_k x_k + 1          (x_k² = x_k for a binary)
                      = −Σ_k x_k  + 2 Σ_{k<l} x_k x_l + 1 .

So per customer: linear `−A` on each `x_{ik}` (`Q[ik,ik] += −A`), coupling `+2A` between any two vehicles
of the same customer (`Q[ik,il] = Q[il,ik] += A`), constant `+A`. Summing over `i`: `offset += A·N`.

Check: a valid one-hot row (exactly one `x_k = 1`) gives `−1 + 0 + 1 = 0`; an empty row gives `0+0+1 = 1`;
a row with two ones gives `−2 + 2 + 1 = 1`; three ones `−3 + 6 + 1 = 4 = (3−1)²`. ✓

**(b) Capacity [≤] via binary slack.** `Σ_i d_i x_{ik} ≤ Q` has no single target. Introduce a
non-negative integer slack `s_k ∈ {0, …, 2^B − 1}`, written in binary with `B` bits:

    s_k = Σ_{b=0}^{B−1} 2^b s_{kb},   s_{kb} ∈ {0,1},   B = ⌈log₂(Q+1)⌉ .

**Slack encoding convention**: `B` is the smallest bit count with `2^B − 1 ≥ Q`, so every load
`L_k ∈ {0,…,Q}` has a slack value `s_k = Q − L_k` that makes the constraint an equality. Values
`s_k > Q` exist (for `Q = 20`, `B = 5`, `s_k` goes to 31) but can never zero the penalty, so they are
never at the minimum; they only waste bitstrings, not atoms (the alternative, coefficients
`1,2,4,…,Q − 2^{B−1} + 1` as in `qopt.model._binary_expansion`, uses the same `B` bits). Tiers: `B = 3`
(S, `Q=7`, exactly `2³−1 = 7`), `B = 5` (M, `Q=20`), `B = 5` (L, `Q=30`).

The inequality becomes the equality `Σ_i d_i x_{ik} + Σ_b 2^b s_{kb} = Q`, penalised as

    P_B(x, s) = B_w Σ_k ( Σ_i d_i x_{ik} + Σ_b 2^b s_{kb} − Q )²

(`B_w` is the weight, written `B` in code as `weights["B"]`; the bit count is `n_slack_bits`). For one
vehicle `k` collect the `N + B` variables `z_m ∈ (x_{0k}, …, x_{N−1,k}, s_{k0}, …, s_{k,B−1})` with
coefficients `c_m ∈ (d_0, …, d_{N−1}, 2⁰, …, 2^{B−1})`. Then

    ( Σ_m c_m z_m − Q )² = (Σ_m c_m z_m)² − 2Q Σ_m c_m z_m + Q²
                         = Σ_m c_m² z_m² + 2 Σ_{m<m'} c_m c_m' z_m z_m' − 2Q Σ_m c_m z_m + Q²
                         = Σ_m (c_m² − 2Q c_m) z_m + 2 Σ_{m<m'} c_m c_m' z_m z_m' + Q² .       (z_m² = z_m)

Per vehicle: linear `B_w (c_m² − 2Q c_m)` on each of its `N+B` variables, coupling `2 B_w c_m c_m'`
between every pair of them (stored as `Q[m,m'] = Q[m',m] += B_w c_m c_m'`), constant `B_w Q²`. Summing
over `k`: `offset += B_w · K · Q²`.

Concretely the linear terms are

    h(x_{ik}) = −A + B_w ( d_i² − 2Q d_i )            (depends on the customer's demand d_i)
    h(s_{kb}) =      B_w ( 4^b − 2Q·2^b )              (depends on the bit position b)

and the couplings (J = 2Q_ij):

    J(x_{ik}, x_{jk}) = d_ij + 2 B_w d_i d_j         same vehicle, two customers   (objective + capacity)
    J(x_{ik}, x_{il}) = 2A                            same customer, two vehicles   (exactly-one)
    J(x_{ik}, s_{kb}) = 2 B_w d_i 2^b                 customer ↔ slack bit of its vehicle
    J(s_{kb}, s_{kb'}) = 2 B_w 2^b 2^{b'}             two slack bits of one vehicle
    J(x_{ik}, x_{jl}) = 0,  J(s_{kb}, s_{lb'}) = 0    different vehicle AND different customer / slack

**Total energy** `E(x) = f(x) + P_A(x) + P_B(x,s)`; `offset = A·N + B_w·K·Q²`. The test
`test_textbook_Q_matches_literal_formula_on_random_bitstrings` evaluates the three sums literally
(no algebra) and compares with `xᵀQx + offset` on random and on all `2^14` tier-S bitstrings;
`test_textbook_linear_terms_and_couplings_match_derivation` checks every line of the two displays above.

**Decode.** Read the `N` one-hot rows; a row with exactly one `1` names the vehicle; any row with 0 or ≥2
ones ⇒ the bitstring does not decode (`None`), it is infeasible. Slack bits are ignored at readout.
Feasible ⇔ decodes **and** `L_k ≤ Q ∀k` (the slack is not trusted: feasibility is re-checked from the
assignment).

**Sign check.** Every coupling above is `≥ 0`: `d_ij ≥ 0`, `A > 0`, `B_w > 0`, demands and powers of two
positive. So the sign requirement of §2.3 is met (asserted in the tests). The problem is elsewhere (§3.3).

### 3.3 Variable count and why it is not device-realisable

**Count.** `n = N·K + K·B = N·K + K·⌈log₂(Q+1)⌉`. The table is generated, never typed:
`python scripts/variable_count.py --write-md` writes it between the markers in `docs/MODEL_WRITEUP.md`
for every registered formulation; its output for the two rows defined here is

    | Assignment variables only, no constraints | N·K                    |  8 | 30 | 60 | yes/yes/yes        |
    | + binary-encoded slack (textbook)         | N·K + K·⌈log2(Q+1)⌉   | 14 | 45 | 80 | yes/yes/yes (=cap) |

Tier L lands exactly on the 80-atom cap: no room for anything, and 20 of the 80 atoms (25 %) carry slack
whose value is thrown away at readout. The third row ("Your formulation") is filled automatically once the
Gate-1 formulation is registered (§4).

**Not realisable on `AnalogDevice`, for two independent reasons:**

1. **Non-uniform linear terms vs a single global δ** (§2.3). `h(x_{ik}) = −A + B_w(d_i² − 2Q d_i)` differs
   between customers with different demands, and `h(s_{kb}) = B_w(4^b − 2Q 2^b)` differs between slack bits
   (`b = 0,1,2` at `Q = 7`, `B_w = 1`: `−13, −24, −40`). The device can apply only one `−δ_f` to all
   atoms, so it realises the *mean* linear term and loses exactly the information that does the capacity
   bookkeeping (which demand each customer carries, what each slack bit is worth). On the toy tier-S
   instance of §3.4 the linear terms span `h ∈ [−17033, −5401]` (in objective units) — a factor 3
   spread that a global detuning cannot represent. A DMM would help in emulation only (`ε_i δ_DMM ≤ 0`,
   one direction, must be labelled; QPU runs are global-only).
2. **Dense couplings.** The capacity penalty couples *every* pair of the `N + B` variables of a vehicle
   with `2 B_w c_m c_m'`: a rank-one-like all-to-all block per vehicle, plus the exactly-one block coupling
   the `K` copies of each customer. Both blocks overlap on every `x_{ik}`. A `1/r⁶` law in a plane
   cannot reproduce an all-to-all block of nearly equal weights for more than a handful of atoms
   (the brief §4: "exactly the dense, long-range interaction pattern that a flat atomic register handles
   worst"; `qopt` measured ~0.5 relative off-diagonal error on such blocks).

What *is* fine: all couplings positive (§3.2), and the weight rule is provably sound (§3.4).

### 3.4 Weights and the brute-force check on a tier-S example

**Weight rule.** Default `A = B_w = Σ_{i<j} d_ij + 1` (`TextbookOneHotSlack.objective_upper_bound() + 1`),
one more than the largest value the objective (3.1) can take (everybody on one vehicle). **Proof of
soundness:** every feasible assignment with the matching slack has `P_A = P_B = 0`, so
`E = f ≤ Σ_{i<j} d_ij`. Every infeasible bitstring violates at least one constraint by an integer `≥ 1`,
so it pays at least `min(A, B_w)·1² > Σ_{i<j} d_ij ≥ 0`, while its objective part is `≥ 0`. Hence
`E(infeasible) > E(any feasible)`: the energy argmin is feasible and, among feasible bitstrings, energy =
objective, so the argmin is the model optimum. ∎ The rule is deliberately loose (it ignores that the
objective part of an infeasible string is rarely 0); the checker measures how loose.

**Tighter rules we could use** (not needed for a throwaway): `A > max_i [ f* − f*_{−i} ]`, the largest
saving from *dropping* one customer (`f*_{−i}` = optimum with `i` removed); for seed 0 below that is
exactly `d_01 = 64.76`. `B_w > max over 1-unit-overloaded clusterings of (f* − f)`; for seed 14 that is
`67.97 − 42.23 = 25.74`. Bisection on the checker confirms both numbers (`A` threshold 64.757, `B_w`
threshold 25.739).

**Checker** (`cvrp.model.brute_force_check`, `n ≤ 20`): enumerate all `2^n` bitstrings, compute `E`, decode
each, and report feasible fraction, whether every energy-argmin decodes feasible and to a model optimum
(clusterings compared modulo relabeling against the exhaustive `K^{N−1}` enumeration of
`cvrp.model.model_optima`), the energy gap `E(best infeasible) − E(best feasible)`, the landscape
correlation between min-energy-per-assignment and objective, and for every weight whether the check still
passes at ½×, 2×, 0× and after how many halvings it breaks.

**Instance.** `instances/` did not exist when this was written, so the instance is built inline with a fixed
seed: `cvrp.model.make_toy_instance("S", seed)` (uniform locations in `[0,100]²`, demands uniform in
`{1..4}`, depot `(50,50)`, seed re-drawn as `seed + 1000·attempt` until a feasible assignment exists —
seed 0 and 14 both succeed at attempt 0). Reproduce with
`python -m cvrp.model --seed 0` / `--seed 14` (pasted verbatim below; labels in `assignment` are
arbitrary, `(0,0,1,1)` and `(1,1,0,0)` are the same clustering).

```
$ python -m cvrp.model --seed 0
instance toy-S-0: N=4 K=2 Q=7 demands=[3, 3, 3, 4] locations=[(63.7, 26.98), (4.1, 1.65), (81.33, 91.28), (60.66, 72.95)]
linear terms h: min=-17032.8530 max=-5400.6607; couplings J: min(nonzero)=830.8709 max=13293.9340; all couplings >= 0: True
brute-force check: formulation='textbook' instance='toy-S-0' n_vars=14 (2^n = 16384) weights={'A': 415.4354383646001, 'B': 415.4354383646001}
  decodable fraction      : 0.0625
  feasible fraction       : 0.0234
  model optimum           : 92.3758  assignment=(0, 0, 1, 1)  (1 optimal clustering(s))
  min energy              : 92.3758  over 2 argmin bitstring(s)
  argmin assignment       : (0, 0, 1, 1)  objective=92.3757614834281
  argmin decodes feasible : True
  argmin same clustering  : True
  argmin is model optimum : True
  min feasible energy     : 92.3758
  min infeasible energy   : 443.0546
  energy gap (inf - feas) : 350.6788   OK (>0)
  landscape over 6 feasible assignment(s): pearson=1.0000 spearman=1.0000
    rank  energy      objective   gap%     assignment
       1     92.3758     92.3758    0.00   (1, 1, 0, 0)
       3    157.6813    157.6813   70.70   (1, 0, 1, 0)
       5    164.3784    164.3784   77.95   (0, 1, 1, 0)
  penalty margins (sound = argmin feasible AND optimal):
    A=415.435: ok_at_half=True ok_at_double=True ok_at_zero=False; fails after 3 halving(s) (x 1/8)
    B=415.435: ok_at_half=True ok_at_double=True ok_at_zero=True (not binding here); never fails within 10 halvings
  => SOUND
```

Reading it:

* **Decodable fraction 1/16 = (K/2^K)^N = (2/4)⁴**: only one bitstring in 16 has all four one-hot rows
  well formed; **feasible fraction 2.3 %** (384 of 16384 strings; `3 clusterings × 2 labelings × 2^6 slack
  states`). This is the "feasibility is not optional" problem (brief Pitfall 9.1) in numbers: a sampler
  that returned uniform bitstrings would be feasible 2 % of the time.
* **2 argmin bitstrings** = the two labelings of the optimal clustering `{0,1}|{2,3}`, each with the unique
  slack that zeroes `P_B` (loads 6 and 7 → slacks 1 and 0). ✓ same clustering, feasible, optimal.
* **Energy gap 350.7 > 0**: the cheapest infeasible string costs 443.05 = `A` (415.44) + 27.62
  (= `d_23`: customers 2,3 together, customer 0 alone, customer **1 dropped** — the far-away customer at
  (4.1, 1.65)). That is the failure mode the exactly-one penalty guards against: dropping a customer saves
  `d_01 = 64.76` of objective, so `A` must exceed 64.76.
* **Landscape**: Pearson = Spearman = 1 over the 6 feasible assignments, because with the right slack
  `E = f` exactly. The best few by energy are the three feasible clusterings with gaps 0 %, 70.7 %, 78.0 %.
* **Margins**: `A` survives halving (A/2 = 207.7 > 64.76) and doubling, breaks at `A/8 = 51.9 < 64.76`
  (`python -m cvrp.model --seed 0 --weight A=51.9294` → `min energy 79.5485`, argmin decodes to `None`,
  gap −12.83, NOT SOUND; 79.55 = 51.93 + 27.62 as predicted). **`B_w` is not binding at all on this seed**
  (`ok_at_zero=True`): the only over-capacity clusterings put ≥ 3 customers on one vehicle, and the pairwise
  objective already makes those more expensive than the optimum. Honest finding: on a scan of toy seeds
  0–39 the capacity weight binds on only 6 (seeds 12, 14, 16, 20, 24, 35); `A` binds on all 40 and breaks
  after 2–5 halvings. **The exactly-one constraint is where the penalty does its work; capacity is mostly
  enforced by the objective's preference for small clusters.** This matters for Gate 1: the capacity
  mechanism should be tested on seeds where it binds.

Seed 14 is such a seed (demands `[4, 2, 4, 3]`, `ρ = 13/14`):

```
$ python -m cvrp.model --seed 14
instance toy-S-14: N=4 K=2 Q=7 demands=[4, 2, 4, 3] locations=[(83.1, 36.09), (70.27, 86.01), (64.13, 54.84), (76.23, 71.63)]
linear terms h: min=-7520.9082 max=-2384.6782; couplings J: min(nonzero)=366.8736 max=5896.6414; all couplings >= 0: True
brute-force check: formulation='textbook' instance='toy-S-14' n_vars=14 (2^n = 16384) weights={'A': 183.436785998489, 'B': 183.436785998489}
  decodable fraction      : 0.0625
  feasible fraction       : 0.0156
  model optimum           : 67.9690  assignment=(0, 1, 1, 0)  (1 optimal clustering(s))
  min energy              : 67.9690  over 2 argmin bitstring(s)
  argmin assignment       : (1, 0, 0, 1)  objective=67.96902766532756
  argmin decodes feasible : True
  argmin same clustering  : True
  argmin is model optimum : True
  min feasible energy     : 67.9690
  min infeasible energy   : 199.0024
  energy gap (inf - feas) : 131.0334   OK (>0)
  landscape over 4 feasible assignment(s): pearson=1.0000 spearman=1.0000
    rank  energy      objective   gap%     assignment
       1     67.9690     67.9690    0.00   (0, 1, 1, 0)
       3     72.2379     72.2379    6.28   (1, 1, 0, 0)
  penalty margins (sound = argmin feasible AND optimal):
    A=183.437: ok_at_half=True ok_at_double=True ok_at_zero=False; fails after 2 halving(s) (x 1/4)
    B=183.437: ok_at_half=True ok_at_double=True ok_at_zero=False; fails after 3 halving(s) (x 1/8)
  => SOUND
```

Here the clustering `{0,2}|{1,3}` (loads 8 and 5, objective 42.23) is cheaper than the optimum 67.97 but
over capacity by one unit; it is kept out only by `B_w·1² > 25.74`, so `B_w/8 = 22.9` fails and
`B_w/4 = 45.9` passes. Only 2 of 16 clusterings are feasible (feasible fraction 1.6 %).

Tests enforcing all of this: `tests/test_model.py` (`test_brute_force_check_passes_on_tier_S_with_default_weights`,
`test_halving_A_enough_times_breaks_the_penalty`, `test_capacity_weight_binds_on_some_seed_and_not_others`).

---

## 4. Our formulation (Gate 1)

*To be written before model code at Gate 1 — the formulation arrives in `docs/MODEL.md`. Writing this
section, then `cvrp/formulations/<name>.py`, then `python scripts/variable_count.py --write-md`, then
`brute_force_check` on three tier-S seeds (include a seed where capacity binds, e.g. toy seed 14 or its
`instances/` equivalent), is the Gate-1 order.*

It must contain, in this order:

1. **Variables**: what each bit means, index convention, count formula `n(N, K, Q)` and the S/M/L numbers
   (from `count_vars`, pasted from the script output, never typed).
2. **Objective**: the closeness measure, written as a polynomial in the bits, and whether it is additive
   over clusters (`cluster_cost`) so `cvrp.exact` can certify the references. If it differs from (3.1),
   say how the model gap changes meaning.
3. **Each constraint → how it is handled**: penalty (expanded to `Q` as in §3.2, cross terms, `x² = x`,
   offset), native encoding, or "left to decode/repair" — one paragraph per constraint, labelled [=]/[≤].
4. **Why every coupling is positive** (§2.3), term by term; if any sign had to be flipped (`x → 1−x` on a
   subset) show the substitution.
5. **Why every linear term is uniform** (or exactly what is lost under the mean, with the number), so a
   single global `δ_f` realises it. State the resulting `h/J` ratio the embedding must hit.
6. **Weights**: how each is set (rule + reason), then `brute_force_check` output pasted verbatim for ≥ 3
   tier-S seeds: feasible fraction, argmin feasible/optimal, energy gap, and what breaks at ½× and 2× of
   each weight (`ok_at_half`, `ok_at_double`, `halvings_until_failure`, `ok_at_zero`).
7. **What it costs**: variables vs §3, coupling structure (sparse? geometric?), feasible fraction of random
   bitstrings vs the textbook's 2 %, and the expected effect on the §9 metrics.

---

## 5. Embedding

*Placeholder — written by the embedding owner (`cvrp/embed.py`). Must contain: the scale `α` and `δ_f`
chosen from the §4 `h/J` ratio and the §2.2 ranges; the position optimisation and its fidelity formula
(relative Frobenius error on off-diagonal couplings + worst pair), the 5 µm / 38 µm / 80-atom checks,
how zero couplings are approximated and the residual, and the brute-force ground-state check on the
*realised* QUBO (`realized_qubo`) versus the target.*

## 6. Pulse

*Placeholder — written by the pulse owner (`cvrp/pulse.py`). Must contain: the three-point Ω ramp and
linear δ sweep with the device limits read from Pulser (`Ω ≤ 12.57`, `|δ| ≤ 125.66`, `T ≤ 6000 ns`,
clock 4 ns), the initial detuning argument (ground state = all `|g⟩`), the adiabaticity estimate from
the minimum gap on a tier-S emulation, the scan over duration / `Ω_max` / slow region scored by best
gap, feasibility rate and P(gap ≤ ε) (never the mean), and the Pulser `Sequence` validation.*

---

## 7. Questions a judge will ask

**Q1. Why cluster-first / route-second instead of the full CVRP?** The direct (customer, vehicle,
position) encoding needs `~N²K` variables before any slack: 32 / 300 / 900 on S / M / L against 80 atoms
(brief §2.2). Clustering alone needs `N·K` (8 / 30 / 60) and the per-vehicle TSPs are tiny (exact
Held–Karp in `cvrp.common.route_exactly`). We pay for this with the decomposition gap (Pitfall 2.1), which we
report separately. → §1.

**Q2. Where is the depot and why?** At the centre of the square, `(50, 50)` (`Instance.depot_rule`). The
clustering objective (3.1) does not see the depot at all — it only enters through the exact TSP of each
cluster, i.e. the CVRP gap. A central depot makes that routed cost least sensitive to which side of the
square a cluster lies on, so the CVRP gap reflects the clustering rather than an asymmetric depot choice.
The brief leaves the choice to us as long as it is stated. → §1, `cvrp.instances`.

**Q3. Why must every coupling be positive on this device?** Because the only interaction is the van der
Waals repulsion `+C6/r⁶ n̂_i n̂_j` between two Rydberg atoms, and `C6 > 0` is fixed physics. A negative
coupling (a reward for exciting two atoms together) has no knob; `cvrp.embed.embed` refuses any `J_ij < 0`.
A formulation with negative couplings must be rewritten (e.g. `x → 1 − x` on a subset). → §2.2–2.3.

**Q4. Why does slack cost atoms, and how many?** The device only has binary variables, so the integer
slack `s_k = Q − L_k ∈ {0..Q}` needs `⌈log₂(Q+1)⌉` bits per vehicle: 3 / 5 / 5 bits, i.e. 6 / 15 / 20 of
the 14 / 45 / 80 textbook variables are slack, each one an atom whose measured value we throw away. Tier
L lands exactly on the 80-atom cap. And the squared penalty couples each slack bit to every customer of
its vehicle. → §3.2–3.3, `scripts/variable_count.py`.

**Q5. What do the textbook formulation's linear terms do on a global-detuning device?** They vary with
the demand (`h(x_{ik}) = −A + B_w(d_i² − 2Q d_i)`) and with the bit position (`h(s_{kb}) = B_w(4^b − 2Q 2^b)`),
spanning a factor ~3 on the toy instance. `AnalogDevice` applies one `δ_f` to every atom, so it realises
their mean and loses the capacity bookkeeping entirely: the realised cost function is a different problem.
That, plus the dense per-vehicle blocks, is why the textbook model is plumbing only. → §2.3, §3.3.

**Q6. How do you know the penalty weights are right?** By exhaustive enumeration: `brute_force_check`
evaluates all `2^14` tier-S bitstrings, verifies the energy argmin decodes to a feasible assignment that
matches the independently enumerated model optimum (clusterings compared modulo relabeling), reports the
energy gap to the best infeasible string, and re-runs with each weight at ½×, 2×, 0× and successive
halvings. For the textbook rule `A = B_w = Σ d_ij + 1` the gap is 350.7 on seed 0, `A` survives 2
halvings and breaks at the third, exactly where the analytic threshold `d_01 = 64.76` predicts. → §3.4.

**Q7. Does the capacity penalty actually do anything?** On most tier-S seeds, no: the pairwise objective
grows like cluster size squared, so overloaded clusters are already expensive and `B_w = 0` is still
sound on 34 of 40 toy seeds. It binds when a cheap 1-unit-overloaded clustering exists (seed 14: `B_w`
must exceed 25.74). We therefore test capacity handling on seeds where it binds, and we report
`ok_at_zero` so a non-binding weight is never mistaken for a working one. → §3.4.

**Q8. How are reference values computed and what is their status?** Model gap: exact optimum of the
constrained clustering model — by enumeration of the `K^{N−1}` label-canonical assignments on tier S
(`cvrp.model.model_optimum`) and by an exact set-partition DP over clusters for all tiers (`cvrp.exact`,
possible because the objective is additive), with the number of optimal clusterings counted modulo
relabeling. CVRP gap: the best clustering routed by exact TSP, cross-checked against simulated annealing
with stated parameters. Every reference is labelled *proven* or *best known* (brief §9.3). → §0, §3.1,
`docs/MODEL_WRITEUP.md`.
