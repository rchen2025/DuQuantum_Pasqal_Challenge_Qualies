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
there is one δ for all atoms, so *we* must pick a single value; `cvrp.embed` sets it from the mean `h̄ = (1/n) Σ_i h_i`
(a choice, flagged `linear_terms_uniform=False`, not device physics) and the per-atom differences `h_i − h̄` are simply
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
`B_w/4 = 45.9` passes. Only 2 of the 8 clusterings (4 of the 16 assignments) are capacity-feasible (feasible fraction of bitstrings 1.6 %).

Tests enforcing all of this: `tests/test_model.py` (`test_brute_force_check_passes_on_tier_S_with_default_weights`,
`test_halving_A_enough_times_breaks_the_penalty`, `test_capacity_weight_binds_on_some_seed_and_not_others`).

---

## 4. Our formulation (Gate 1): `onehot-ub`

**Full derivation, weight scan, checker output and judge Q&A: [`MODEL.md`](MODEL.md)** (AI-drafted at the
team's request on 2026-10-24, to be re-derived by the team before judging — `AI_USE.md`). This section is
the summary that code follows (`cvrp/formulations/onehot_ub.py`, tests in `tests/test_onehot_ub.py`).

1. **Variables.** The textbook's one-hot `x_ik` (index `i·K + k`) and nothing else: `n = N·K` = S 8 / M 30 /
   L 60 (`scripts/variable_count.py`). No slack.
2. **Objective.** (3.1) unchanged, additive over clusters, so the model gap means the same as in §3.
3. **Constraints.** [=] exactly-one: squared penalty `A Σ_i (1 − Σ_k x_ik)²` → `h = −A`, `J = 2A` on each
   customer's K-clique, offset `A·N` (same algebra as §3.2). [≤] capacity: unbalanced penalization
   `Σ_k [−λ₁(Q − L_k) + λ₂(Q − L_k)²]` (Montañez-Barrera et al. 2024, arXiv:2211.13914), expanded to
   per-variable terms `c_i x_ik`, `c_i = d_i(λ₁ − 2λ₂Q + λ₂d_i)`, plus couplings `2λ₂ d_i d_j` on same-vehicle
   pairs. **One-hot identity:** `Σ_k c_i x_ik = c_i` on every one-hot string, so all per-customer linear terms
   (λ₁ included) are constants where the model lives and are dropped. What remains is
   `λ₂ Σ_k L_k² + const` — capacity *becomes a load-balance term*, soft, with no `L_k ≤ Q` guarantee
   (K = 2: the balance optimum is feasible whenever a feasible split exists; K ≥ 3: likely only).
   Final cost (MODEL.md M4):

       E(x) = Σ_k Σ_{i<j} d_ij x_ik x_jk + A Σ_i (1 − Σ_k x_ik)² + 2λ₂ Σ_k Σ_{i<j} d_i d_j x_ik x_jk .

4. **Signs.** `d_ij ≥ 0`, `2A > 0`, `2λ₂ d_i d_j > 0`: every coupling positive, no flip needed.
5. **Linear terms.** Only `−A`, on every variable: uniform, `δ_f = +αA` (27.7 rad/µs at the device scale that
   puts the clique at 5 µm). Nothing is lost under the mean — the dropped diagonal is exactly constant.
6. **Weights.** `A = max_i Σ_{j≠i} (d_ij + 2λ₂ d_i d_j) + 1` (provably makes every argmin one-hot: any
   exactly-one repair step lowers the energy). `λ₂ = κ·R/(2Q)`, `R = max_i Σ_j d_ij`, `κ = 0.15` from a
   40-seed toy scan (35/40 sound, 37/40 argmin feasible, 5/5 shipped S sound; `κ = 0`: 34/34; `κ = 1`:
   27/40 sound, 40/40 feasible). Checker verbatim for S_seed00/01/04 and the binding toy seed 14 (NOT
   SOUND: closest clustering overloads a vehicle; best *feasible* string is still the optimum) in MODEL.md §6
   and `out/gate1/check_*.txt`. `A` survives one halving, fails at ×1/4; `λ₂` is not binding on any
   shipped S instance (`ok_at_zero=True`).
7. **Cost.** Support = N×K rook's graph, S 16/28, M 165/435, L 510/1770 pairs; the capacity term adds no
   pairs. The embedding cannot make `(i,k)–(j,l≠k)` weak while `(i,k)–(j,k)` is strong (the K copies of a
   customer sit 5 µm apart): expect a relative Frobenius error of the textbook's order (0.55); measured by
   `cvrp.embed` per run (runner job 006). Random-bitstring feasible fraction 1.6–4.7 % on S, set by the
   decodable fraction `K^N/2^{NK}`, not by the capacity term.

---

### 4.1 M2 step 2 — weights on the device (scan, no change)

`docs/MODEL.md` §6 sets A by the provable repair-step rule and λ₂ = 0.15·R/(2Q) by a 40-seed brute-force scan.
The device adds one more question: do the *ratios* A : λ₂ : d_ij matter once the register, not the QUBO, is
what is sampled? We scanned multipliers of both rules on the three M1 instances (structure-aware embedding
recomputed per point, baseline pulse, numpy-sv, 500 shots):

<!-- M2_WEIGHT_SCAN -->
| point | feasibility | best_model_gap | best_cvrp_gap | P_reach_model | P_reach_ref | P_gap_le_5 | per_instance_P_reach_model |
|---|---|---|---|---|---|---|---|
| A x0.5, lam2 x0.5 | 0.845 | 0.000 | 7.243 | 0.634 | 0.325 | 0.634 | 0.42 / 0.55 / 0.94 |
| A x0.5, lam2 x1 | 0.854 | 0.000 | 7.243 | 0.207 | 0.200 | 0.207 | 0.04 / 0.56 / 0.02 |
| A x0.5, lam2 x2 | 0.885 | 0.000 | 7.243 | 0.382 | 0.323 | 0.382 | 0.02 / 0.87 / 0.25 |
| A x0.75, lam2 x0.5 | 0.891 | 0.000 | 7.243 | 0.813 | 0.559 | 0.813 | 0.95 / 0.69 / 0.80 |
| A x0.75, lam2 x1 | 0.837 | 0.000 | 7.243 | 0.581 | 0.342 | 0.581 | 0.46 / 0.52 / 0.76 |
| A x0.75, lam2 x2 | 0.983 | 0.000 | 7.243 | 0.553 | 0.563 | 0.553 | 0.71 / 0.95 / 0.00 |
| A x1, lam2 x0.5 | 0.918 | 0.000 | 7.243 | 0.577 | 0.402 | 0.577 | 0.38 / 0.73 / 0.62 |
| A x1, lam2 x1 | 0.869 | 0.000 | 7.243 | 0.603 | 0.397 | 0.603 | 0.51 / 0.61 / 0.68 |
| A x1, lam2 x2 | 0.821 | 0.000 | 7.243 | 0.479 | 0.486 | 0.479 | 0.96 / 0.46 / 0.01 |
| A x1.5, lam2 x0.5 | 0.808 | 0.000 | 7.243 | 0.471 | 0.285 | 0.471 | 0.29 / 0.54 / 0.58 |
| A x1.5, lam2 x1 | 0.830 | 0.000 | 7.243 | 0.425 | 0.312 | 0.425 | 0.32 / 0.49 / 0.47 |
| A x1.5, lam2 x2 | 0.970 | 0.000 | 7.243 | 0.599 | 0.598 | 0.599 | 0.85 / 0.91 / 0.03 |

(generated by `python scripts/scan.py` → `out/m2_weights/scan.md`; mean over the three M1 instances, per-instance P(reach model optimum) in the last column)
<!-- /M2_WEIGHT_SCAN -->

Reading: the per-instance columns swing by ±0.4 between neighbouring points with no monotone trend; the
embedding optimiser lands on a different local minimum for every (A, λ₂) and that choice moves P(reach) far
more than a factor 2 on either weight does (the same effect that gave 51 % vs 90 % on S_seed00 for one weight
setting, `DID_NOT_WORK.md`). No weight change is justified by this scan; the formulation's weights stay exactly
as MODEL.md states them, and the margin that matters on the device is the realised-register check of §5.7
(argmin feasible and optimal, single-flip defect gap), which is run and recorded for every row.

## 5. Embedding

### 5.1 What the device can compute

Pulser's `AnalogDevice` drives every atom with the same laser (one global Rydberg
channel, no detuning-map / DMM).  In Pulser's units (ħ = 1, frequencies in rad/µs,
lengths in µm) the Hamiltonian is

$$
H(t) = \sum_i \frac{\Omega(t)}{2}\,\sigma^x_i \;-\; \delta(t)\sum_i n_i \;+\; \sum_{i<j} \frac{C_6}{r_{ij}^6}\, n_i n_j ,
\qquad n_i = |r\rangle\langle r|_i .
$$

At the end of the pulse Ω = 0, so $H$ is diagonal in the computational basis.
Reading $n_i \to x_i \in \{0,1\}$ (atom $i$ in the Rydberg state ⇔ $x_i = 1$), a
bitstring $x$ has the energy

$$
E_{\text{dev}}(x) = -\delta_{\text{final}} \sum_i x_i + \sum_{i<j} U_{ij}\, x_i x_j ,
\qquad U_{ij} = \frac{C_6}{r_{ij}^6} > 0 .
$$

This is a QUBO with three built-in restrictions: (i) every coupling $U_{ij}$ is
**positive** (the van der Waals interaction is repulsive), (ii) no coupling is ever
exactly **zero**, and (iii) every linear coefficient is the **same** number
$-\delta_{\text{final}}$.

### 5.2 From the QUBO to distances

Our QUBO (common.py convention) is $E(x) = x^\mathsf{T} Q x = \sum_i h_i x_i + \sum_{i<j} J_{ij} x_i x_j$
with $h_i = Q_{ii}$, $J_{ij} = 2Q_{ij}$.  We want the device to minimise the same
function, so we ask for $E_{\text{dev}} = \alpha\, E$ for a single positive energy
scale $\alpha$ (rad/µs per QUBO unit; it only rescales the spectrum and does not
move the minimiser).  Matching term by term:

$$
U_{ij} = \alpha J_{ij}, \qquad \delta_{\text{final}} = -\alpha h .
$$

Solving $C_6 / r_{ij}^6 = \alpha J_{ij}$ for the distance gives the **target distance**

$$
\boxed{\; r_{ij} = \left(\frac{C_6}{\alpha\, J_{ij}}\right)^{1/6} \;}
\qquad (\texttt{cvrp.embed.target\_distance}) .
$$

Two consequences of the sixth root: a factor 2 in $J$ moves the atoms only by
$2^{1/6} = 1.12$, and scaling *all* coordinates by $s$ multiplies every $U_{ij}$
by $s^{-6}$, i.e. it changes $\alpha$ by $s^{-6}$ while leaving the *shape* of the
realised QUBO untouched.  We use this freedom below.

**Negative couplings** ($J_{ij} < 0$) would need an attractive interaction; there
is none, so `embed()` raises `ValueError` naming the pairs.  The formulation has to
be rewritten, not patched here.  **Zero couplings** can only be approximated by
putting the two atoms far apart (§5.5).

### 5.3 Choosing α

α is bounded on both sides by the hardware:

1. **Minimum distance (geometry).**  Atoms cannot be closer than
   $r_{\min}$ = `device.min_atom_distance` = 5 µm, so the strongest coupling obeys
   $\alpha J_{\max} \le C_6 / r_{\min}^6 = 865\,723 / 5^6 = 55.4$ rad/µs.  We start
   from the *largest* α this allows (the optimiser places the strongest pair at the
   floor; it keeps a 2 % margin, 5.1 µm, so a rounding error never trips the hard
   limit).  Larger α means larger energy gaps relative to the fixed Ω and the fixed
   6 µs budget, which is what adiabaticity wants.
2. **Detuning cap.**  $|\delta_{\text{final}}| = \alpha |h| \le$ `max_abs_detuning`
   = 125.66 rad/µs.  If the geometric α violates it, α is set to
   $\alpha_{\text{cap}} = 125.66 / |h|$ and every coordinate is multiplied by
   $(\alpha/\alpha_{\text{cap}})^{1/6}$ (exact because $U \propto r^{-6}$).
3. **Register radius.**  After step 2 all atoms must still lie within
   `max_radial_distance` = 38 µm of the origin.  If not, we shrink back onto the
   38 µm circle and *report* the resulting detuning violation in
   `fidelity.limit_violations` — such a QUBO (|h| ≫ |J|) simply does not fit this
   device and the run layer must refuse it.  Atom count > `max_atom_num` = 80
   raises immediately.

All four numbers are read from the Pulser device object at run time
(`DeviceLimits.from_pulser_device`), never typed into the code.

### 5.4 Fidelity: the number we report

Even with α free, a 2-D register has only $2n-3$ geometric degrees of freedom for
$n(n-1)/2$ couplings, and $1/r^6$ obeys triangle-type inequalities (two atoms both
close to a third cannot be far from each other).  So in general $U/\alpha \ne J$,
and we quantify this with exactly

$$
\boxed{\;\text{rel\_frobenius\_offdiag} = \frac{\lVert U/\alpha - J\rVert_F}{\lVert J\rVert_F}
\quad\text{over all } i \ne j\;}
\qquad (\texttt{FidelityReport.rel\_frobenius\_offdiag}),
$$

together with the worst single pair, $\max_{i<j,\,J_{ij}>0} |U_{ij}/\alpha - J_{ij}| / J_{ij}$
(`worst_pair_rel_error`, `worst_pair`), and the largest *spurious* coupling
$\max_{J_{ij}=0} U_{ij}/\alpha$ (`max_spurious_coupling`).  α itself is the
least-squares value $\alpha = \langle U, J\rangle / \langle J, J\rangle$ for the final
geometry (then capped as in §5.3), and positions come from a multi-start L-BFGS
(`qopt.embed.optimize_positions`: relative error on positive targets, far-apart
target on zero pairs, hinge penalties for the 5 µm floor and the 38 µm radius;
8 seeded starts, best kept).  `Embedding.realized_Q()` returns the QUBO the device
actually implements, $Q'_{ij} = U_{ij}/(2\alpha)$, $Q'_{ii} = -\delta_{\text{final}}/\alpha$,
so the classical checker can compare its minimiser with the target's.

### 5.5 One worked pair (from `python -m cvrp.embed --demo`, seed 0)

Toy: 6 variables, $J_{ij}$ uniform in [0.2, 1] with 3 of the 15 pairs set to 0,
$h_i = -1$ for all $i$.  The optimiser returns α = **46.38 rad/µs per QUBO unit**
(the strongest pair sits at the 5.1 µm floor; the detuning cap is not active since
$|\delta_{\text{final}}| = 46.38 < 125.66$).  Strongest pair (2, 3):

| quantity | value |
|---|---|
| target $J_{23}$ | 0.9481 |
| α | 46.38 rad/µs |
| target $U = \alpha J$ | 43.97 rad/µs |
| target distance $(C_6/\alpha J)^{1/6} = (865\,723 / 43.97)^{1/6}$ | 5.196 µm |
| realised distance | 5.101 µm |
| realised $U = C_6 / 5.101^6$ | 49.13 rad/µs → $U/\alpha$ = 1.059 |
| relative error on this pair | 11.7 % |

The optimiser deliberately leaves this pair 1.8 % too close because the same two
atoms also have to serve their other neighbours; the global figures for the toy are
**rel_frobenius_offdiag = 0.370**, worst pair (1, 3) with 94 % error (target 0.685,
realised 0.039 — the geometry sacrificed it), largest spurious coupling 0.022 on the
zero pair (0, 2).  A dense, nearly uniform 6-clique is close to the worst case for
$1/r^6$ in 2-D; graph-structured QUBOs (few, well-separated neighbours per atom)
embed with errors well below 0.1 (cf. the 2-atom test, error < 1e-9).

**Zero couplings.**  $U$ is never 0.  The best we can do is distance: inside the
38 µm trap radius the largest possible separation is 2 × 38 = 76 µm, where
$U = 865\,723 / 76^6 = 4.5\times10^{-6}$ rad/µs, i.e. $U/\alpha \approx 1\times10^{-7}$
QUBO units at α = 46.4 — negligible.  In practice the zero pairs are much closer
(the register is compact), e.g. 20 µm gives $U/\alpha = 2.9\times10^{-4}$; the toy's
0.022 comes from a zero pair that ended up at 9.74 µm.  The report always quotes the
realised worst case so the classical checker can decide whether it matters.

### 5.6 Why the linear terms must be uniform

The device has one knob for all diagonal terms: $-\delta_{\text{final}}\sum_i x_i$.
An `AnalogDevice` QUBO must therefore have $h_i = h$ for every variable (a
`WeightedAnalogDevice` could lower individual $\delta_i$ with its DMM; this device
cannot).  `embed()` checks `linear_spread = max h − min h` against a $10^{-9}$
tolerance and sets `fidelity.linear_terms_uniform`.  If it is **false** we do not
silently fix anything: δ_final is set from the *mean* of $h$ so a sequence can
still be built for plumbing tests, a loud `NOT A FAITHFUL DEVICE EMBEDDING` note is
added, `fidelity.ok` is false, and the run layer refuses to present such a run as a
device result.  The formulation must make the diagonal uniform itself (e.g. by
absorbing per-variable biases into the coupling structure or into the constant).

### 5.7 Structure-aware embedding for one-hot formulations (M2 step 1, `cvrp/embed_onehot.py`)

**What went wrong with the generic embedder.** `onehot-ub` asks for the N×K rook's graph: the K atoms of
customer i coupled by 2A (the exactly-one clique), atom (i,k) coupled to (j,k) by w_ij = d_ij + 2λ₂d_id_j,
and (i,k)–(j,l≠k) coupled by exactly 0. In a plane with U = C6/r⁶ the clique forces the K copies of a
customer to sit 5 µm apart, so any atom of customer j is at nearly the same distance from all K copies of
customer i: the "zero" couplings come out as strong as the wanted ones. Measured on the M1 rows (generic
embedder, baseline pulse; table below): relative Frobenius error ≈ 0.95, and on S_seed04 the realised
register's ground state is an *infeasible* string (model optimum at device-energy rank 9).

**Ask for less: the one-hot subspace.** On a one-hot string every customer has exactly one excited atom,
so between customers i and j exactly one of the K² couplings is active, U(i,a_i; j,a_j). Writing
Σ_ik x_ik = N (constant), the device energy on one-hot strings is

    E_dev(a) = −δ_f·N + Σ_{i<j} U(i,a_i; j,a_j) .                                                   (5.7a)

The model wants E_dev(a) = α·E_QUBO(a) + const with E_QUBO(a) = const + Σ_{i<j, a_i=a_j} w_ij. Split the
sum in (5.7a) into same-vehicle and cross-vehicle pairs. If, for each pair (i,j), the same-vehicle coupling
U(i,k;j,k) did not depend on k and all K(K−1) cross couplings were equal to a common value C_ij, then
Σ_{i<j} U(i,a_i;j,a_j) = Σ_{i<j} C_ij + Σ_{i<j, a_i=a_j} (U_same,ij − C_ij), i.e. only the *effective* coupling

    U_eff(i,j) = U(i,k; j,k) − mean_{l≠k} U(i,k; j,l)                                               (5.7b)

has to equal α w_ij; the cross couplings need not vanish, they only need to be *constant on the subspace*.
Neither condition holds exactly in a plane (for K = 2 both cross distances are equal only when the dimers are
parallel and displaced perpendicular to their axis), so both deviations are measured and reported: the
spread of the cross couplings ("anisotropy", max |U(i,k;j,l) − mean| / αw_ij) and the spread of the
same-vehicle couplings over k (`subspace_same_vehicle_asymmetry`; it makes the two vehicle labelings of one
clustering differ in device energy, which the scoring ignores because clusterings are compared as sets).

**The construction.** Each customer becomes a rigid regular K-gon of side s = 5 µm (K=2: a dimer), so the
clique coupling is exactly 2A·α with α = C6/s⁶/(2A) — the clique is pinned at the minimum separation.
The polygon centres c_i and orientations θ_i are optimised (L-BFGS, 6 seeded starts) so that the device
energies of all K^N one-hot strings (all K^N strings up to 4096, else a seeded sample of 2048) fit α·E_QUBO up to a constant:
we minimise the normalised residual Σ(E_dev − αE_QUBO − c)² / Σ(αE_QUBO − mean)² with the slope fixed at α
(0 = the register orders the clusterings exactly like the model; it is not an R², the slope is not free).
Two more terms: (i) **defect gaps** — every one-hot string with one bit flipped (a customer with no atom or
with two) must lie at least ½ of the model's own penalty gap above its parent, in device units (all 16
one-hot strings on S; the first 64 of the sample on M/L); without this the crowded cross couplings made
"customer with no atom" the ground state on two of three seeds; (ii) hinges keeping atoms of different
polygons ≥ 5 µm apart and the register inside 38 µm. **δ_f is a free fit parameter** here: by (5.7a) it is a
constant on the one-hot subspace, so it only sets the defect gaps; it starts at the QUBO value αA
(27.5 rad/µs) and lands a little above it (table) because the real cross couplings make an excitation cheaper
to drop than the algebra with zero cross terms assumes. All limits still come from the device object; the
sequence is still validated by Pulser. The optimiser is seeded but not bit-identical across platforms — and on
tier M not even across two concurrent runs on the same machine (M_seed01: two registers with δ_f 125.7 / 24.0 and
optimum rank 1 114 / 12 966 from the same seed, 18:50 ET) — so the register of every reported row is cached in
`embeddings/` and committed, every run loads it (`register loaded from cache` in the record notes), and a row
whose register differs from the cache is re-run rather than reported (`docs/DID_NOT_WORK.md`).

**What we report, per run record (`embedding.fidelity`, `realized_register`) and in results.csv.** (1) The
brief's fidelity formula on the full matrix — for this register it is dominated by the exact clique entries
and therefore *understates* the pair errors (worst positive-pair relative error and the largest spurious
coupling are reported next to it); (2) the subspace fidelity ||U_eff/α − w||_F/||w||_F, the energy-fit
residual, the anisotropy and the same-vehicle asymmetry; (3) the realised-register check: is the device's
own argmin feasible and equal to the model optimum, at which device-energy rank does the model optimum sit,
and the smallest single-flip defect gap. A judge should read (3) first, (2) second, (1) last.

**Before / after.** Every number below is generated from the run records (`scripts/derivation_numbers.py`);
the "after" registers are the cached ones, identical on Windows (qutip) and here (numpy-sv), which is why the
two backends agree to sampling noise.

<!-- M2_EMBEDDING_TABLE -->
| instance | embedder | backend | feasibility | P(reach model opt) | P(reach CVRP ref) | best model gap | rel. Frobenius (full) | subspace rel. Frob. | energy-fit residual | device rank of model optimum | δ_f (rad/µs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S_seed00 | generic | numpy-sv | 0.198 | 0.116 | 0.116 | 0.0 % | 0.948 | — | — | 1 | 13.0 |
| S_seed00 | generic | qutip | 0.216 | 0.126 | 0.126 | 0.0 % | 0.948 | — | — | 1 | 13.0 |
| S_seed00 | generic | random | 0.044 | 0.007 | 0.007 | 0.0 % | 0.948 | — | — | — | 13.0 |
| S_seed00 | polygons | numpy-sv | 0.990 | 0.902 | 0.902 | 0.0 % | 0.263 | 0.382 | 0.1531 | 1 | 35.1 |
| S_seed00 | polygons | qutip | 0.990 | 0.896 | 0.896 | 0.0 % | 0.263 | 0.382 | 0.1531 | 1 | 35.1 |
| S_seed01 | generic | numpy-sv | 0.088 | 0.078 | 0.078 | 0.0 % | 0.940 | — | — | 1 | 13.0 |
| S_seed01 | generic | qutip | 0.086 | 0.072 | 0.072 | 0.0 % | 0.940 | — | — | 1 | 13.0 |
| S_seed01 | generic | random | 0.017 | 0.007 | 0.007 | 0.0 % | 0.940 | — | — | — | 13.0 |
| S_seed01 | polygons | numpy-sv | 0.838 | 0.838 | 0.838 | 0.0 % | 0.212 | 0.566 | 0.3203 | 1 | 35.9 |
| S_seed01 | polygons | qutip | 0.880 | 0.878 | 0.878 | 0.0 % | 0.212 | 0.566 | 0.3203 | 1 | 35.9 |
| S_seed04 | generic | numpy-sv | 0.056 | 0.026 | 0.004 | 0.0 % | 0.946 | — | — | 9 | 13.0 |
| S_seed04 | generic | qutip | 0.060 | 0.028 | 0.002 | 0.0 % | 0.946 | — | — | 9 | 13.0 |
| S_seed04 | generic | random | 0.045 | 0.009 | 0.009 | 0.0 % | 0.946 | — | — | — | 13.0 |
| S_seed04 | polygons | numpy-sv | 0.982 | 0.650 | 0.078 | 0.0 % | 0.318 | 0.248 | 0.0621 | 1 | 37.7 |
| S_seed04 | polygons | qutip | 0.986 | 0.622 | 0.090 | 0.0 % | 0.318 | 0.248 | 0.0621 | 1 | 37.7 |

(generated by `python scripts/derivation_numbers.py` from out/m1, out/m2; 500 shots, seed 0, baseline pulse; random sampler 2000 shots)
<!-- /M2_EMBEDDING_TABLE -->

S_seed04: P(reach CVRP ref) stays low because the model optimum is not the CVRP-optimal clustering
(decomposition gap 21.7 %, §7). One consequence of the A rule worth stating: every one-hot string is a strict
single-flip local minimum of the QUBO (the cheapest uphill flip costs ≥ A − Σ_j w_ij ≥ 1), so any
single-flip post-processing — qubo-solver's greedy sweep, our energy-descent repair — can only snap a defective
string to the nearest one-hot string, never move between clusterings.

## 6. Pulse

### 6.1 Why we start at large negative detuning

For $\delta \ll -\Omega$ the single-atom Hamiltonian $\tfrac{\Omega}{2}\sigma^x - \delta n$
has its ground state at $|g\rangle$ (energy ≈ 0 versus ≈ $-\delta > 0$ for
$|r\rangle$), and the interaction term only adds energy to states with excitations.
So the many-body ground state of $H(0)$ is $|g\,g\cdots g\rangle$, exactly the state
the machine prepares by default.  The adiabatic theorem then says: if $H(t)$ changes
slowly — sweep rate $|\dot\delta|$ small compared with $\Delta_{\min}^2$ (Δ = instantaneous gap above the ground
state; Landau–Zener form, same statement as §6.4), the system stays in the instantaneous ground state and ends in the ground
state of $H(T)$, which (Ω(T) = 0) is the minimiser of $E_{\text{dev}} = \alpha E_{\text{QUBO}}$.

`default_delta_init` = $-\min\!\big(\max(|\delta_{\text{final}}|,\, 2\Omega_{\max}),\ \text{max\_abs\_detuning}\big)$:
at least $2\Omega_{\max}$ below resonance so the initial gap
($\sqrt{\delta^2 + \Omega^2} \ge 2.2\,\Omega$ per atom) is unambiguous, at least
$|\delta_{\text{final}}|$ so the sweep is symmetric about δ = 0 and the slope is the
same on both sides of the crossing, and never beyond the channel limit.  Toy:
δ_init = −46.38 rad/µs.

### 6.2 Where δ_final comes from, and its sign

§5.2: $\delta_{\text{final}} = -\alpha h$.  For the toy $h = -1$ (every customer
assignment variable *wants* to be 1), so δ_final = +46.38 rad/µs > 0: positive
detuning lowers the energy of $|r\rangle$, favouring excitation, and the repulsive
$U_{ij}$ then decide which subset of atoms may be excited together — the usual
"maximum-weight independent set" picture.  A QUBO with $h > 0$ would give
δ_final < 0 and the trivial all-ground answer; the formulation layer keeps $h < 0$.

### 6.3 The baseline shape (brief §8.2)

The brief asks for "a smooth three-point ramp for the amplitude and a linear sweep
for the detuning, over a few microseconds".  `PulseParams` implements exactly that,
in fractions of the duration $T$ (default 4000 ns):

| waveform | knots (time fraction → value) |
|---|---|
| Ω(t) | 0 → 0, `rise_frac`=0.15 → Ω, 1−`fall_frac`=0.85 → Ω, 1 → 0 |
| δ(t) | 0 → δ_init, `sweep_start_frac`=0.15 → δ_init, `sweep_end_frac`=0.85 → δ_final, 1 → δ_final |

with Ω = `omega_scale` × Ω_max (default 1.0 × 12.566 rad/µs).  Both are Pulser
`InterpolatedWaveform`s (PCHIP through the knots, like Pulser itself), which is
monotone between knots: Ω never overshoots Ω_max and δ never leaves
[δ_init, δ_final], so the sampled waveforms respect the limits whenever the knots
do.  The detuning sweeps only while Ω is on its plateau, so the gap is never probed
with Ω = 0.  `waveforms()` reproduces Pulser's samples (knots at frac × (T − 1),
1 ns steps) for plots and tests; `final_detuning_from_sequence()` reads δ(T) back
from `pulser.sampler.sample(seq)` and agrees with δ_final to machine precision
(toy: diff 0.00e+00).  Toy baseline sweep rate: 92.75 rad/µs over 2.8 µs = 33 rad/µs².

### 6.4 The slow-region knob

The gap is smallest where the ground state changes character — the first atoms
flip around δ ≈ 0 … U, and a Landau–Zener crossing is traversed safely only if the
sweep rate there is small compared with $\Delta_{\min}^2$.  Rather than slowing the
whole sweep (the 6 µs budget is fixed), `slow_region = (centre_frac, width_frac,
slowdown_factor)` inserts extra δ-knots so that over the part of the *detuning path*
centred at `centre_frac` (0 = δ_init, 1 = δ_final) and spanning `width_frac` of it,
the slope is `slowdown_factor` times smaller than outside, with the total sweep
time unchanged (so the rest is correspondingly faster).  Example
`(0.6, 0.3, 3.0)` on the toy: the window covers δ ∈ [−4.6, +23.2] rad/µs, slope
inside ≈ 16 rad/µs² versus ≈ 66 outside (chord ratio exactly 3; PCHIP rounds the
corners).  The sweep stays monotone and still ends exactly at δ_final.  The right
centre is problem dependent — scan it with the emulator (`cvrp.run`), the formula is
only the knob.

### 6.5 Device limits every sequence is validated against

`build_sequence` lets Pulser do the validation: `Sequence(register, AnalogDevice)`
checks atom count ≤ 80, pairwise distance ≥ 5 µm, radius ≤ 38 µm; `seq.add(pulse,
"ryd")` checks Ω ≤ 12.566 rad/µs, |δ| ≤ 125.66 rad/µs (both raise `ValueError` from
the channel's `validate_pulse`), and total duration ≤ 6000 ns (raises
`RuntimeError`); durations are snapped to the 4 ns clock and ≥ 16 ns.  We then call
`seq.to_abstract_repr()` (the JSON the cloud/QPU receives) and assert
`seq.get_duration()` equals the requested duration.  `validate(seq)` re-checks the
*sampled* waveforms and the register against the device object independently.
The register is attached to an automatic layout (`Register.with_automatic_layout`,
`requires_layout` is true for the QPU); this rounds coordinates at the 1e-6 µm
level, i.e. U changes by ~1e-6 relative, far below the embedding error.

### 6.6 M2 step 3 — pulse scan on the device (structure-aware registers, numpy-sv, 500 shots, seed 0)

Scored as the brief asks (§8.2): best gap, P(reach model optimum / CVRP reference), feasibility — never a
mean cost. Three knobs in the order the brief names them: how long the sweep is (duration), how strong the
drive is (Ω as a fraction of Ω_max), and the final detuning (a multiplier on the embedding's fitted δ_f).

**Duration × Ω scale** (δ_f ×1):

<!-- M2_PULSE_SCAN -->
| point | feasibility | best_model_gap | best_cvrp_gap | P_reach_model | P_reach_ref | P_gap_le_5 | per_instance_P_reach_model |
|---|---|---|---|---|---|---|---|
| T=2000 ns, Omega x0.5, delta_f x1 | 0.522 | 0.000 | 7.243 | 0.305 | 0.181 | 0.305 | 0.25 / 0.25 / 0.41 |
| T=2000 ns, Omega x0.75, delta_f x1 | 0.875 | 0.000 | 7.243 | 0.740 | 0.480 | 0.740 | 0.70 / 0.74 / 0.78 |
| T=2000 ns, Omega x1, delta_f x1 | 0.879 | 0.000 | 7.243 | 0.739 | 0.569 | 0.739 | 0.83 / 0.78 / 0.61 |
| T=4000 ns, Omega x0.5, delta_f x1 | 0.823 | 0.000 | 7.243 | 0.579 | 0.397 | 0.579 | 0.56 / 0.55 / 0.63 |
| T=4000 ns, Omega x0.75, delta_f x1 | 0.929 | 0.000 | 7.243 | 0.844 | 0.557 | 0.844 | 0.85 / 0.80 / 0.89 |
| T=4000 ns, Omega x1, delta_f x1 | 0.937 | 0.000 | 7.243 | 0.797 | 0.606 | 0.797 | 0.90 / 0.84 / 0.65 |
| T=5800 ns, Omega x0.5, delta_f x1 | 0.873 | 0.000 | 7.243 | 0.695 | 0.462 | 0.695 | 0.66 / 0.65 / 0.77 |
| T=5800 ns, Omega x0.75, delta_f x1 | 0.983 | 0.000 | 7.243 | 0.932 | 0.642 | 0.932 | 0.96 / 0.95 / 0.89 |
| T=5800 ns, Omega x1, delta_f x1 | 0.976 | 0.000 | 7.243 | 0.865 | 0.653 | 0.865 | 0.98 / 0.93 / 0.68 |

(generated by `python scripts/scan.py` → `out/m2_pulse/scan.md`; mean over the three M1 instances, per-instance P(reach model optimum) in the last column)
<!-- /M2_PULSE_SCAN -->

Reading: longer is monotonically better on every instance (2000 → 4000 → 5800 ns at fixed Ω), as the
adiabatic theorem predicts for a fixed gap; Ω at 0.75·Ω_max beats full power at 4000 and 5800 ns because the
plateau Ω = 9.4 rad/µs sits closer to the same-vehicle couplings (3–13 rad/µs) while still well above the
sweep rate, whereas Ω = Ω_max = 12.6 rad/µs dresses the one-hot cliques (U_clique = 55 rad/µs) harder and
costs feasibility on S_seed04; Ω/2 is too weak to open the gap. **Chosen baseline for tier S: T = 5800 ns
(the device maximum minus the 200 ns margin), Ω = 0.75·Ω_max.** qubo-solver's own drive shaper lands in the
same corner (5940 ns, Ω = 9.7–10.1 rad/µs, `out/m1/*qubosolver-quantum*`), which is reassuring but was not
used to pick ours.

**δ_f scale and slow region** at that (T, Ω):

<!-- M2_DELTA_SCAN -->
| point | feasibility | best_model_gap | best_cvrp_gap | P_reach_model | P_reach_ref | P_gap_le_5 | per_instance_P_reach_model |
|---|---|---|---|---|---|---|---|
| T=5800 ns, Omega x0.75, delta_f x0.8 | 0.970 | 0.000 | 7.243 | 0.915 | 0.643 | 0.915 | 0.96 / 0.95 / 0.84 |
| T=5800 ns, Omega x0.75, delta_f x1.25 | 0.958 | 0.000 | 7.243 | 0.884 | 0.600 | 0.884 | 0.90 / 0.88 / 0.87 |

(generated by `python scripts/scan.py` → `out/m2_delta/scan.md`; mean over the three M1 instances, per-instance P(reach model optimum) in the last column)
<!-- /M2_DELTA_SCAN -->

<!-- M2_SLOW_SCAN -->
| point | feasibility | best_model_gap | best_cvrp_gap | P_reach_model | P_reach_ref | P_gap_le_5 | per_instance_P_reach_model |
|---|---|---|---|---|---|---|---|
| no slow region (baseline) | 0.983 | 0.000 | 7.243 | 0.932 | 0.642 | 0.932 | 0.96 / 0.95 / 0.89 |
| slow centre=0.5 width=0.3 factor=3 | 0.927 | 0.000 | 7.243 | 0.851 | 0.539 | 0.851 | 0.81 / 0.79 / 0.95 |
| slow centre=0.65 width=0.3 factor=3 | 0.983 | 0.000 | 7.243 | 0.959 | 0.650 | 0.959 | 0.97 / 0.96 / 0.94 |
| slow centre=0.8 width=0.3 factor=3 | 0.984 | 0.000 | 7.243 | 0.957 | 0.650 | 0.957 | 0.98 / 0.96 / 0.93 |

(generated by `python scripts/scan.py` → `out/m2_slow/scan.md`; mean over the three M1 instances, per-instance P(reach model optimum) in the last column)
<!-- /M2_SLOW_SCAN -->

Reading: the fitted δ_f is the best of the three multipliers (×0.8 / ×1 / ×1.25 → 0.915 / 0.932 / 0.884). A slow
region in the last part of the sweep (centre 0.65–0.8 of the detuning path, slope ÷3 over 30 % of it) gains
+2.5 pp on average (0.932 → 0.959, consistent on all three seeds), a slow region in the middle loses 8 pp; the
gain is at the edge of the 500-shot sampling noise (±2 pp), so the committed default stays the plain linear
sweep and the slow-region rows stay in results.csv as a labelled option (`scan.py slow`).


### 6.7 The hard bound δ_final < U_clique (found on tier M, 18:10 ET)

Inside one customer's K-gon every pair of atoms sits at the side $s$ (5 µm), so the realised clique coupling
is $U_{\text{clique}} = C_6/s^6 = 55.1$ rad/µs. At the end of the sweep (Ω = 0) the clique's own energies are

$$E_0 = 0,\qquad E_1 = -\delta_f,\qquad E_2 = -2\delta_f + U_{\text{clique}},\qquad
E_3 = -3\delta_f + 3U_{\text{clique}}\ (K=3),$$

so the one-hot state $E_1$ is the clique ground state **only if** $0 < \delta_f < U_{\text{clique}}$ — and for a
general K-gon the binding number is the *weakest* intra-polygon pair, $U_{\min} = C_6/\ell_{\max}^6$ with
$\ell_{\max}$ the longest chord: for K ≤ 3 every chord is the 5 µm side ($U_{\min} = U_{\text{clique}} = 55.1$), but a
K = 4 square has diagonals $5\sqrt2 = 7.07$ µm, so $U_{\min} = U_{\text{side}}/8 = 6.93$ rad/µs and two *opposite*
corners are favoured for any δ_f above that. (Four points in a plane cannot all be mutually equidistant — the
max/min distance ratio is at least √2 — so a planar one-hot clique for K ≥ 4 is never uniform; a 1/r⁶ law turns
that √2 into a factor 8 in coupling. This was caught by the team's review of report_1850 before the first L
row was reported; the L run launched with the triangle bound was discarded.) The embedder uses $U_{\min}$ of the
actual polygon, and `scripts/fix_cached_delta.py` reads it off the register. On tier S the
one-hot-subspace fit of §5.7 returns δ_f = 33–38 rad/µs = 0.49–0.68 $U_{\text{clique}}$ (every cached S register,
`scripts/fix_cached_delta.py --dry-run`), safely inside. On tier M the same fit is ill-conditioned (one-hot-subspace
relative Frobenius error 0.96–1.55 against 0.25–0.57 on S, results.csv column `embedding_subspace_rel_frobenius`;
energy-fit residual 38 against 0.06–0.32, Q-M1) and ran to the device maximum, δ_f = 125.66 rad/µs = 2.28 $U_{\text{clique}}$:
the sweep then *rewards* two excitations per triangle, and the measured as-is row (M_seed00, emu-mps χ = 32)
had 0 of 500 shots decodable — not a feasibility failure of the encoding but a wrong end point of the sweep.
The embedder now enforces the bound: when the fit leaves $(0, 0.9\,U_{\min})$ it falls back to
$0.65\,U_{\min}$, the ratio the well-conditioned S fits land at; cached registers are migrated with
the same rule and unchanged geometry (`scripts/fix_cached_delta.py`), so before/after rows share the register.
Dry run on the M_seed00 register with δ_f = 35.8: the minimum single-flip defect gap turns from −70.4 to +19.4
rad/µs (one-hot strings are again below their defects) while the model optimum's rank among one-hot strings
stays 10 474 of 59 049 — the bound restores *decodability*, not solution quality, which is set by the distance
couplings the plane cannot realise (Q-M1).

Measured before/after (emu-mps χ = 32, 500 shots, seed 0, identical registers — results.csv, tags `|pre-6.7` =
before):

| instance | δ_f [rad/µs] | decodable | feasible | best model gap | best CVRP gap | optimum's one-hot rank |
|---|---|---|---|---|---|---|
| M_seed00 before | 125.66 (2.28 U) | 0.000 | 0.000 | undefined | undefined | 10 474 |
| M_seed00 after | 35.80 (0.65 U) | 0.994 | 0.044 | 67.9 % | 29.9 % | 10 474 |
| M_seed01 (fit already inside the bound) | 24.01 (0.44 U) | 0.984 | 0.388 | 37.9 % | 9.2 % | 12 966 |
| M_seed02 before | 125.66 (2.28 U) | 0.000 | 0.000 | undefined | undefined | 4 192 |
| M_seed02 after | 35.80 (0.65 U) | 0.996 | 0.446 | 23.0 % | 16.0 % | 4 192 |
| qubo-solver's pipeline, M_seed00 / M_seed01 (BLaDE register, χ = 32) | 23.9 / 24.1 | 0.99 / 1.00 | 0.016 / 0.37 | 72.1 % / 47.5 % | 35.2 % / 11.2 % | — |

Reading: the bound turns decodability from 0 to ≥ 98 % on both affected instances and feasibility from 0 to
4–45 %; the best feasible shot stays 23–68 % above the model optimum because the optimum's rank among one-hot
strings is unchanged — exactly the split predicted above (constraints embed, objective does not). Our rows and
qubo-solver's independent pipeline on the same instances land in the same regime (feasibility 2–37 %, gaps
11–35 % CVRP), which supports reading this as a property of one-hot CVRP clustering on a planar 1/r⁶ register
at K = 3 rather than of our embedder. The brief's "if no shot is feasible, the gap is undefined" rows are kept
under `|pre-6.7` as the record of the mistake.

**Tier L (K = 4), first measured row (L_seed00, emu-mps χ = 32, 500 shots, 48 min; register = cached, δ_f = 4.475
= 0.65 U_min):** decodable 0.014, capacity-feasible 0.008 (4 shots), best feasible model gap 140 %, CVRP gap 80 %,
P(reach) 0; random control 0/2000. Why so much worse than M even with the right end point: for K = 4 the bound
forces δ_f below the diagonal coupling 6.9 rad/µs, so the *entire* one-hot energy landscape of the 60-atom
register is compressed into a window of a few rad/µs — the single-flip defect gaps are 1.1 rad/µs (results.csv
`realized_register`), against 19–23 rad/µs on M. A sweep of δ from −25.1 to +4.5 rad/µs over 0.7 × 5800 ns has rate
|δ̇| ≈ 7 rad/µs², far above the Landau–Zener scale Δ² ≈ 1.2 (rad/µs)² (§6.1), so the evolution is not adiabatic
with respect to the one-hot manifold and most shots leave it (98.6 % non-decodable). Opening the squares does not
help: the diagonal/side ratio √2 is fixed by planar geometry, so U_min/U_side = 1/8 for every square size, and a
larger side only lowers both. The honest L statement is therefore: a planar one-hot encoding with K = 4 cannot
present the device with an energy scale it can resolve in 6 µs — the limit is the K-gon geometry, not the pulse
or the weights. L_seed01/02 rows follow with the same settings (realised ranks 1 063 / 14 862).


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

**Q-ref. How are the reference values C* obtained?** The way brief §9.3 prescribes, by the first applicable
method with a stated max solve time (`scripts/reference_qubosolver.py`, settings recorded in every
`instances/*.json` under `reference.solver_settings`): (2) qubo-solver's classical solver on our penalized
`onehot-ub` QUBO — CPLEX when the `cplex` package is importable (it is not here, `docs/DID_NOT_WORK.md`),
otherwise qubo-solver's simulated annealing with stated parameters (T₀ = max penalty weight, T_f = 10⁻³ T₀,
geometric cooling fixed by (T₀, T_f, max_iter), 10 seeded restarts, **max solve time 60 s**); every returned
bitstring is decoded and only feasible decodes count, routed exactly; and (3) our constrained simulated
annealing on the CVRP itself (stated T₀/T_f/α/max_iter, 10 seeded restarts, the same 60 s budget). C* is the
best value found by either method and is recorded as `reference_status = best_known`, with the method that
found it. On tier S the brief additionally asks for brute force *and* the solver with an agreement check, so
there the exhaustive set-partition DP (`cvrp/exact.py`, Held–Karp per cluster) is also run and `agree` is
recorded — and because an exhaustive search that agrees *is* the brief's level 1, the five S references carry
`proven_optimum` (team review of report_1850: an exact certificate is not downgraded because CPLEX is missing);
on M/L no exhaustive search enters the reference and they stay `best_known`. Measured (job 200, 17:50 ET, every number in the
instance JSONs): the two solvers agree on all 11 instances' best-known C* and the tier-S brute-force check
agrees 5/5. qubo-solver's SA on the *penalized QUBO* reaches C* on 5/5 S and 2/3 M instances but on tier L
its best feasible decode within 60 s is 23–34 % above the constrained SA (599.9 vs 447.0, 554.6 vs 434.9,
521.7 vs 424.7) — the penalty landscape of a 60-variable one-hot QUBO is a harder search space than the
constrained problem itself, which is why C* is defined as the best of the two and never the QUBO solver
alone. (The exact DP run earlier today — commit 6a8d0dd — returned the same 11 values; we report them as
best known because that is the §9.3 route the team chose.) Re-scoring every run record against the new
blocks (`scripts/rescore_records.py`) changed only the `reference_status` column of results.csv.
→ `scripts/reference_qubosolver.py`, `cvrp/reference.py`, `cvrp/exact.py`.

**Q-fleet. Your L optima use 3 of 4 vehicles — is that allowed?** Yes: brief §2 assumption 6 says a vehicle
may be left unused at zero cost, and by the triangle inequality merging two tours never costs more when
capacity allows. The reference therefore minimises over *at most* K non-empty clusters, and the clustering
model must not force exactly K non-empty groups (that would change the optimum). → §1, `cvrp.exact.exact_partition`.

**Q-random. Why is the random sampler a serious baseline on tier S?** With 14 textbook variables the space
has 16 384 strings and ~0.6 % of uniform samples already decode to the optimal clustering (`cvrp.run
--backend random`, 2000 shots, seed 0: P(reach ref) = 0.0065 raw — the exact uniform value is 2 labelings × 2⁶ slack states / 2¹⁴ = 0.78 % — and 0.080 after labelled energy-descent repair; record `out/S_seed00__textbook_random_repair-energy-descent__seed0.json`).
Any quantum result on S must beat these numbers at equal shots or it says nothing; the brief asks for exactly
this comparison (§9.2, "replace the quantum part with a random sampler"). → `cvrp/metrics.py`.

**Q-M1. Your tier-S numbers are excellent; why is there nothing comparable for M and L?** Because the
encoding stops being layable at K = 3. The exactly-one clique forces a customer's K atoms within 5 µm of each
other, so a neighbouring customer's atoms are at nearly equal distance from all of them and the "zero"
cross-vehicle couplings are as strong as the wanted ones and anisotropic. On S (K = 2, 4 dimers) a fit on the
one-hot subspace absorbs this (§5.7); on M (K = 3, 10 triangles) it does not: subspace residual 38, model optimum
at rank ~10 000 of 59 049 one-hot strings, defects below their parents (`DID_NOT_WORK.md`). We measured it
rather than hid it (results.csv, tier M rows: feasibility and gaps as found). The brief's §4 warning — "an
approach that halves the variable count but produces a coupling pattern you cannot lay out is not a saving" —
applies to one-hot beyond K = 2 on this device. → §5.7, STATUS Gate-2 options.

**Q-work. How much of the result is the quantum part and how much is the decoder (brief §9.2)?** Replace the
register by an assignment-valid random sampler — uniform random assignments encoded one-hot (`--backend
random-onehot`) — and score identically: P(reach model optimum) = 2/16 ≈ 0.12 on every tier-S instance
(two labelings of one optimal clustering among 16 assignments), capacity-feasible 25–63 %. The headline qutip
rows give 0.94 / 0.95 / 0.89 on S_seed00/01/04 with feasibility ≥ 0.95: a 7–8× factor from the register alone.
The plain uniform-bitstring sampler (0.7–1 %) is the weaker control and is also reported. → results.csv,
`cvrp/run.py` (`random-onehot`), §5.7.

**Q-A. Is the big exactly-one weight A just a safety factor? What happens with the tight one?** No — it is
physically load-bearing. `onehot-ub-tight` uses the tight bound A = R_w/K·(1+1 %) (MODEL.md P1, sound by the
same argument and checker-verified) with the same embedder and pulse: P(reach optimum) 0.94 → 0.03 on S_seed00
(the realised register's ground state becomes the third-best clustering), 0.95 → 0.52 on S_seed01, 0.89 → 0.94
on S_seed04 (qutip, 500 shots; the in-house exact state-vector rows give 0.03 / 0.56 / 0.96). A sets the single-flip defect gaps in device units (§5.7) and the margin the embedding error can eat;
the tight value leaves none on two of three seeds. We keep the provable rule for the headline and report tight
as the ablation; its pulse was not re-tuned. → MODEL.md §6, §5.7, results.csv (`onehot-ub-tight`).

### Embedding & pulse

**Q-pulse1. Why 5800 ns and 0.75·Ω_max, and is that over-fitted to three seeds?** The scan (§6.6) shows
monotone trends, not a sharp optimum: longer is always better up to the device limit (so we use the maximum
minus the 200 ns safety margin), and Ω/Ω_max = 0.75 beats 1.0 and 0.5 at every duration. The δ_f scan
(×0.8 / ×1 / ×1.25) brackets the fitted value with the maximum at ×1. S_seed02/03, which were not used in the
scan, reach the model optimum in 99 % / 96 % of raw shots on qutip with these settings; the three scan seeds give
94 / 95 / 89 % (qutip, `out/m3`). → §6.6, `results.csv`.

**Q-pulse2. Does the register actually stay adiabatic?** We cannot prove it from a gap we do not compute,
but the measurements say yes where it matters: P(reach model optimum) rises with duration at fixed Ω
(0.74 → 0.80 → 0.87 for 2000 → 4000 → 5800 ns at Ω_max), the lowest-energy strings of the final Hamiltonian
dominate the histogram, and the remaining shots are mostly the second-best clustering, not random strings
(feasibility 0.98). → §6.1, §6.6.

**Q-pulse3. Why does the final detuning from the fit (35–38 rad/µs) differ from αA (27.5)?** See Q-emb3:
on one-hot strings δ_f is a constant and only sets the single-flip defect gaps; the real cross couplings make
dropping an excitation cheaper than the zero-cross-term algebra assumes, so the fit raises δ_f. The scan
confirms ×1 is the best of the three multipliers tried. → §5.7, §6.6.

**Q-M2. Your tier-M row had zero decodable shots — is one-hot simply unrealisable at 30 atoms?** No; that row
had δ_f = 125.66 rad/µs, 2.3× the clique coupling, so the sweep ended in a regime where two excitations per
triangle are cheaper than one (§6.7). The bound δ_f < U_clique is a physics requirement we had not written down
because the tier-S fits never violated it. With δ_f = 0.65 U_clique the same register is one-hot at its device
ground state (defect gap +19.4). What *is* unrealisable at K = 3 is the ordering of the one-hot strings by the
distance couplings (optimum at rank ~10 000), so the honest tier-M statement is: the constraints embed, the
objective does not. qubo-solver's own pipeline on the same instance (BLaDE register, χ = 32) shows the same
split — 99 % one-hot shots, 1.6 % capacity-feasible, best feasible 72 % above the model optimum. → §6.7, results.csv.

**Q-M3. On tier M, how much of the loss is the soft capacity penalty, how much the register, how much the pulse?**
We enumerated all K^N one-hot assignments and compared three landscapes (`scripts/landscapes.py` →
`docs/checks/landscapes.md`): the original model f (closeness on capacity-feasible assignments), the target
QUBO E_Q (with the load-balance term), and the realised register E_dev at Ω = 0. (i) *Penalty:* on all 5 S
instances and on M_seed00/01 the QUBO argmin over one-hot strings is exactly the model optimum (rank 1,
capacity-feasible) — the load-balance term costs nothing there. On **M_seed02 it does not**: the QUBO argmin is a
feasible assignment 10.2 % above the model optimum, which sits at QUBO rank 331 — and qubo-solver's tabu row on
M_seed02 lands at precisely that 10.2 %, i.e. tabu solved the QUBO exactly. So on M_seed02, 10 of the 23 points of
measured model gap are the formulation's λ₂ = 0.15·R/(2Q) rule, not the device. (ii) *Register:* the model optimum
sits at device rank 10 474 / 12 966 / 4 192 of 59 049 on M_seed00/01/02 — this is where most of the loss is. On
M_seed01 the device's own one-hot ground state is capacity-feasible with a 101 % model gap but only a 2.6 % CVRP
gap after exact routing (the device likes a clustering that routes well but is far from "close"). (iii) *Pulse:*
the emulated best feasible shots (model gaps 67.9 / 37.9 / 23.0 %) are *better* than the register's best feasible
one-hot string under E_dev (82.0 / 101.3 / 93.9 %), so the sweep is not the bottleneck; it samples the low-energy
part of a landscape whose ordering is wrong. The equivalence "P(model gap ≤ 5 %) = P(model optimum)" used in the
report holds on every S instance and on M_seed00/01 (smallest non-zero feasible gap 11–62 %) but **not** on
M_seed02 (6 feasible assignments within 5 %, smallest 0.98 %) — the report now states it per instance. All of
this certifies the one-hot subspace only, never all 2^(NK) bitstrings. → `docs/checks/landscapes.md`.

**Q-emb1. Your full-matrix fidelity went from ≈0.95 to ≈0.2–0.3 — did you fix the embedding or the metric?** Neither; we changed
what we ask the register to reproduce. The full-matrix error is the brief's formula and we still report it,
but it is dominated by the exactly-one cliques (the four largest entries) and says little about the
same-vehicle couplings; the numbers that decided the design are the one-hot-subspace fit (1 − R² of device
vs model energies) and the realised-register check (is the device's own argmin the model optimum?). Both
are in every record. → §5.7.

**Q-emb2. If cross-vehicle couplings are not zero, how can the device compute the model?** Because the
model only lives on one-hot strings, where each customer pair contributes exactly one coupling; a constant
cross coupling shifts every one-hot string by the same amount. Only U_same − U_cross must match α·w_ij
(5.7b). For K = 2 parallel dimers this is exact; for K ≥ 3 the spread of the cross couplings is measured
and reported as anisotropy. → §5.7.

**Q-emb3. Why is δ_final no longer αA?** On the one-hot subspace Σ_ik x_ik = N, so δ_final is a constant
there and cannot change the ordering of clusterings; it only sets how far single-flip defects (a customer
with 0 or 2 atoms) sit above the one-hot strings, and the real cross couplings move that balance. We fit
it (starting from αA ≈ 27.5 and landing at 35–38 rad/µs on the cached tier-S registers, far inside ±125.66) against the explicit defect-gap
constraint, and the realised-register check confirms every argmin is one-hot and optimal. → §5.7, §6.2.

**Q: How do you know the register implements *your* QUBO and not something else?**
`Embedding.realized_Q()` is the QUBO the final Hamiltonian actually encodes
($Q'_{ij} = U_{ij}/2\alpha$, $Q'_{ii} = -\delta_{\text{final}}/\alpha$), and
`final_hamiltonian_from_sequence` rebuilds δ(T) and $U$ from the *built* Pulser
sequence, not from our parameters.  The gap between $Q'$ and $Q$ is one number,
rel_frobenius_offdiag (§5.4), plus the worst pair and the worst spurious coupling;
the classical checker runs on both $Q$ and $Q'$.

**Q: What about couplings you cannot realise?**
Negative ones are refused outright (`ValueError` listing the pairs, §5.2) — we never
drop or clip a sign.  Zeros are approximated by distance and the residual is
reported (§5.5: ≤ 1e-7 QUBO units at the 76 µm maximum separation, 0.022 in the
compact toy).  Non-uniform linear terms are flagged, not averaged away silently
(§5.6).

**Q: Why these pulse parameters, and are they legal on the device?**
δ_init is set so the start is unambiguously all-ground and the sweep is symmetric
about resonance (§6.1); δ_final is forced by the formulation (§6.2); the shape is the
brief's baseline (§6.3).  Legality is not our claim but Pulser's: the sequence is
built on `AnalogDevice` and serialised with `to_abstract_repr()`, and
`test_over_limit_pulses_are_rejected_by_pulser` shows an over-amplitude or
over-detuning request fails inside Pulser (§6.5).

**Q: Is the sweep adiabatic?**
Not guaranteed by construction — the minimum gap of a 10–15 atom problem is not
known analytically.  The baseline keeps the detuning sweep within the Ω plateau and
the slow-region knob (§6.4) lets us spend more of the fixed 6 µs budget where the
first atoms flip; whether a given setting is slow enough is measured on the
emulator (P(ground state) versus duration and slow-region centre), and the numbers
go into the results table, not into this document.
