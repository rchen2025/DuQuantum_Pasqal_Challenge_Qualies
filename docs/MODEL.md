# MODEL — Gate-1 formulation `onehot-ub` (Track A)

**Status: drafted by Claude at the team's request (2026-10-24 14:30 ET), reviewed and corrected 16:30 ET
(`AI_USE.md`).** Every number comes from a script call named next to it, never typed. Notation is DERIVATIONS.md §0; the physics is §2; the textbook it replaces is §3.

One sentence: *one-hot assignment variables, no slack; the capacity inequality is handled by unbalanced
penalization, which on one-hot strings collapses to a load-balance term; everything the device cannot
realise (per-customer linear terms) is provably invisible to the model and is dropped.*

---

## 1. Variables

`x_ik ∈ {0,1}`, `x_ik = 1 ⇔` customer `i` rides on vehicle `k`. QUBO index `i·K + k` (same as the
textbook, so the decoder, `encode`, and the checker are shared). **No other variables.**

    n(N, K, Q) = N·K          S: 8    M: 30    L: 60          (python scripts/variable_count.py --write-md)

Decode: a bitstring is decodable iff every row `(x_i0 … x_i,K−1)` has exactly one `1`; then `a_i = argmax_k x_ik`.

## 2. Objective

Total pairwise Euclidean distance inside each cluster (brief §5.1 "closeness"), identical to the textbook's (3.1):

    f(x) = Σ_k Σ_{i<j} d_ij x_ik x_jk .                                                        (M1)

It is additive over clusters, `cost(S) = Σ_{i<j∈S} d_ij`, so `cvrp.exact` certifies the model optimum on
every tier and the **model gap keeps the same meaning as for the textbook** (same model, different QUBO).
Known bias (§3.1): it grows like `|S|²`, so it already prefers small clusters — this does part of the
capacity constraint's work for free (§6).

## 3. Constraints

### 3a. Exactly-one, `Σ_k x_ik = 1` for every `i` — [=] — squared penalty

    P_1(x) = A Σ_i (1 − Σ_k x_ik)²
           = A Σ_i [ 1 − 2 Σ_k x_ik + (Σ_k x_ik)² ]
           = A Σ_i [ 1 − 2 Σ_k x_ik + Σ_k x_ik + 2 Σ_{k<l} x_ik x_il ]           (x² = x)
           = A·N  −  A Σ_i Σ_k x_ik  +  2A Σ_i Σ_{k<l} x_ik x_il .                            (M2)

QUBO: `h(x_ik) += −A` for every variable; `J(ik, il) = +2A` for every pair of vehicles of the same
customer (a K-clique per customer); `offset += A·N`.

### 3b. Capacity, `L_k = Σ_i d_i x_ik ≤ Q` for every `k` — [≤] — unbalanced penalization, then the one-hot identity

Textbook: slack bits (§3.2). Instead, write the inequality as `g_k = Q − L_k ≥ 0` and penalise it with the
two-term expansion of Montañez-Barrera, Willsch, Maldonado-Romo & Michielsen, *Quantum Sci. Technol.* 2024
(arXiv:2211.13914, their eqs. 25–27): `ζ(g) = −λ₁ g + λ₂ g²`, `λ₁, λ₂ > 0`, no auxiliary variables.

    P_2(x) = Σ_k [ −λ₁ (Q − L_k) + λ₂ (Q − L_k)² ]
           = Σ_k [ −λ₁ Q + λ₁ L_k + λ₂ Q² − 2 λ₂ Q L_k + λ₂ L_k² ] .

Expand `L_k` and `L_k²` in the bits (`x² = x`):

    L_k   = Σ_i d_i x_ik ,
    L_k²  = Σ_i d_i² x_ik  +  2 Σ_{i<j} d_i d_j x_ik x_jk .

Collect per variable and per pair:

    P_2(x) = K(λ₂ Q² − λ₁ Q)  +  Σ_i Σ_k c_i x_ik  +  2 λ₂ Σ_k Σ_{i<j} d_i d_j x_ik x_jk ,
    c_i    = d_i (λ₁ − 2 λ₂ Q + λ₂ d_i)        — depends on the customer i, NOT on the vehicle k.   (M3)

**The one-hot identity.** On every string that satisfies 3a, `Σ_k x_ik = 1`, hence

    Σ_i Σ_k c_i x_ik  =  Σ_i c_i  =  constant.

So the entire linear part of the capacity penalty — and with it **λ₁, which appears nowhere else** — is a
constant on the one-hot subspace. It can only change the energy of strings that already violate exactly-one,
which `A` already pushes above every one-hot string (§6). We therefore drop `Σ_i Σ_k c_i x_ik` and the
constant. (Check: dropping it changes the energy of a string by `Σ_i c_i (Σ_k x_ik − 1)`, zero on one-hot
strings.) This is what makes every linear term uniform (§5) without a DMM.

What survives of the capacity penalty is the coupling `+2 λ₂ d_i d_j` between every two customers on the
same vehicle. On one-hot strings this is exactly

    2 λ₂ Σ_k Σ_{i<j} d_i d_j x_ik x_jk  =  λ₂ ( Σ_k L_k² − Σ_i d_i² )  =  λ₂ Σ_k L_k²  +  const.

**So "capacity" becomes a load-balance term:** `λ₂ Σ_k L_k²` is minimised, for fixed total demand
`D = Σ_i d_i`, by the most even split of demand across the K vehicles. It is **not** a hard `L_k ≤ Q`:

* For `K = 2` and the **pure** balance term: `L_1² + (D − L_1)²` is minimised by the achievable `L_1` closest
  to `D/2`, which also minimises `max(L_1, L_2)`; so if any capacity-feasible split exists, the balance-optimal
  split is feasible. This does **not** carry over to `f + λ₂·balance`: the objective can outvote the balance
  term and the argmin can overload a vehicle (toy seed 14, §6, is exactly that).
* For `K ≥ 3`: minimising `Σ_k L_k²` does not in general minimise `max_k L_k`; feasibility of the argmin is
  not guaranteed, only made likely. Brief Pitfall 9.1 applies in full: we report the best **feasible** shot
  and the feasibility rate, never an infeasible cost.
* The term also rewards using **all** K vehicles (K equal loads beat K−1), whereas STATUS.md notes that the
  CVRP optima on all L instances use 3 of 4 vehicles. This bias is part of the decomposition/clustering
  loss and is measured, not hidden.

### 3c. Final cost function (what `cvrp/formulations/onehot_ub.py` builds)

    E(x) = Σ_k Σ_{i<j} d_ij x_ik x_jk  +  A Σ_i (1 − Σ_k x_ik)²  +  2 λ₂ Σ_k Σ_{i<j} d_i d_j x_ik x_jk     (M4)

QUBO entries (DERIVATIONS §0 convention `J_ij = 2 Q_ij`):

| term | `h` (diagonal) | `J` (pair) | support |
|---|---|---|---|
| objective (M1) | 0 | `+d_ij` | `(ik, jk)`, same vehicle, `i<j` |
| exactly-one (M2) | `−A` on every variable | `+2A` | `(ik, il)`, same customer, `k<l` |
| capacity → balance | 0 (dropped, constant) | `+2 λ₂ d_i d_j` | `(ik, jk)`, same vehicle, `i<j` |
| offset | `A·N` | | |

Verified against the literal formula on 500 random bitstrings and the one-hot identity on all `K^N`
assignments (`tests/test_onehot_ub.py`).

## 4. Why every coupling is positive

Term by term: `d_ij ≥ 0` (Euclidean distance); `2A > 0`; `2 λ₂ d_i d_j > 0` (demands ≥ 1, `λ₂ > 0`). No
sign flip `x → 1−x` was needed anywhere. Pairs not listed in §3c have `J = 0` and are *approximated* by far
atoms (§7). `cvrp.embed` refuses any `J < 0` and `test_device_realisable_signs_on_every_shipped_instance`
checks all 11 instances.

## 5. Why every linear term is uniform

Only (M2) contributes a diagonal, and it is `−A` on every variable. The capacity penalty's diagonal `c_i`
(M3) is non-uniform but constant on the one-hot subspace (§3b) and dropped; nothing is "lost under the
mean" — the dropped part is exactly zero where the model lives. Hence a single global final detuning

    δ_f = −α h = +α A  > 0        (favours excitations: exactly-one needs one atom per customer excited)

realises it, with `α` the device scale of §7. Ratio the embedding must hit: `h / J_clique = −A / 2A = −1/2`
and `h / J_objective = −A / (d_ij + 2λ₂ d_i d_j)`, i.e. `|h|` is 2–8× the largest objective coupling on S
and 7–100× on L (numbers in §7).

## 6. Weights

**`A` (exactly-one) — two proved rules.** Let `w_ij = d_ij + 2 λ₂ d_i d_j` (every same-vehicle coupling,
identical across vehicles) and `R_w = max_i Σ_{j≠i} w_ij`. Both rules guarantee that **every global
minimiser of the target energy is one-hot**; neither claims that every invalid string lies above every valid
one, and neither says anything about capacity, sampling, or embedding error.

*Rule `row_sum` (`onehot-ub`, original):* `A = R_w + 1`. Take any string violating exactly-one. A customer
with no vehicle: give it one — the penalty drops by `A`, the couplings rise by at most `Σ_{j≠i} w_ij ≤ R_w`.
A customer with `m ≥ 2` vehicles: remove one — the penalty drops from `A(m−1)²` to `A(m−2)²`, the couplings
can only drop (`J ≥ 0`). Every repair step strictly lowers the energy, for every `λ₂`
(`test_A_rule_makes_every_argmin_one_hot_even_when_capacity_binds`).

*Rule `row_sum_over_k` (`onehot-ub-tight`, reviewer 2026-10-24):* `A = (R_w/K)(1 + 0.01)`, a factor `K`
smaller. Proof: (1) a global minimiser has no customer on `m ≥ 2` vehicles — removing one lowers the penalty
by `A(2m−3) > 0` and cannot raise the pair terms. (2) So a hypothetical invalid minimiser has only missing
customers, and every other customer is on at most one vehicle. (3) For a missing customer `i` let
`t_k = Σ_{j≠i} w_ij x_jk`; then `Σ_k t_k ≤ Σ_{j≠i} w_ij ≤ R_w`, so some vehicle has `t_k ≤ R_w/K`. (4) Adding
`i` to that vehicle lowers the penalty by `A` and raises the pair terms by `t_k < A`. Contradiction. ∎
Checked by enumeration on the 5 shipped S instances, 40 toy S seeds and random non-negative `W` at
`K = 2, 3, 4` (`test_tight_A_rule_every_argmin_is_one_hot`). It is a 1 % target-model margin, so the
original rule stays as the large-margin ablation; on the device the smaller `A` makes every objective
coupling `K×` stronger relative to the clique coupling. The checker confirms the bound is tight-ish: the
tight `A` fails after one halving, the original after two.

**`λ₂` (capacity → balance) — empirical parameterisation, chosen by scan.** We write `λ₂` in units of
`R / (2Q)`, `R = max_i Σ_{j≠i} d_ij`, purely to make `κ` dimensionless and comparable across tiers; it is
*not* a bound of any kind. (The true balance change of moving a customer of demand `d` from vehicle `a` to
`b` is `2d(L_b − L_a + d)`: it depends on the actual loads, a feasible solution need not be one move away,
and nothing guarantees a demand-1 customer exists in an instance.) We set

    λ₂ = κ · R / (2Q),   κ = 0.15 ,

the optimum of a scan on 40 toy tier-S seeds (`cvrp.model.make_toy_instance`, brute force, all 256
bitstrings each):

| κ | shipped S sound | toy sound /40 | toy argmin feasible /40 | mean Pearson(energy, objective) |
|---|---|---|---|---|
| 0 (no capacity term) | 5/5 | 34 | 34 | 1.000 |
| 0.10 | 5/5 | 34 | 36 | 0.993 |
| **0.15** | **5/5** | **35** | **37** | **0.985** |
| 0.25 | 4/5 | 34 | 37 | 0.964 |
| 0.5 | 4/5 | 31 | 38 | 0.898 |
| 1.0 | 4/5 | 27 | 40 | 0.796 |

Reading: at `κ = 0` the 34 sound seeds are exactly those where capacity does not bind (the objective's
`|S|²` bias already splits them); every increase of `κ` buys feasibility on binding seeds and pays with
optimality on non-binding ones (the balance term re-ranks feasible clusterings). No `κ` is sound on all 40:
the two constraints "closest" and "balanced" conflict on the binding seeds, and a soft term cannot resolve
that — only post-selection on feasibility can (Pitfall 9.1). On the 5 shipped S instances capacity never
binds, so `ok_at_zero=True` below is expected and honest.

**Brute-force checker output, verbatim** (`python -m cvrp.model --formulation onehot-ub --instance …`;
full files in `out/gate1/check_*.txt`):

    S_seed00  weights={'lam2': 2.9366, 'A': 398.4189}
      decodable fraction 0.0625  feasible fraction 0.0391
      argmin decodes feasible True  argmin is model optimum True   energy gap (inf−feas) 197.42 OK
      landscape over 10 feasible assignments: pearson 0.9693 spearman 0.9000
      lam2: ok_at_half=True ok_at_double=True ok_at_zero=True (not binding here); never fails within 10 halvings
      A   : ok_at_half=True ok_at_double=True ok_at_zero=False; fails after 2 halvings (×1/4)      => SOUND
    S_seed01  weights={'lam2': 2.1574, 'A': 318.8505}
      feasible fraction 0.0156; argmin feasible+optimal; gap 42.39 OK; pearson 1.0000
      lam2: ok_at_half/double/zero = True/True/True(not binding); A: fails after 2 halvings        => SOUND
    S_seed04  weights={'lam2': 1.2415, 'A': 176.4641}
      feasible fraction 0.0391; argmin feasible+optimal; gap 74.57 OK; pearson 0.9666 spearman 1.0
      lam2: ok_at_half=True ok_at_double=False ok_at_zero=True; A: fails after 2 halvings           => SOUND
    toy-S-14 (capacity binds: demands [4,2,4,3], closest pair 0,2 sums to 8 > 7)
      argmin (1,0,1,0) is the closest but INFEASIBLE clustering; energy gap −20.84 BROKEN     => NOT SOUND
      among the 4 feasible assignments pearson = 1.0: the best *feasible* shot is the model optimum.

What breaks: halving `A` twice lets a customer vanish (string with a zero row) because the saved objective
couplings exceed the penalty; doubling `λ₂` on S_seed04 re-ranks two feasible clusterings (balance beats
closeness) and the argmin is feasible but not optimal; `λ₂ = 0` is sound wherever capacity does not bind
and fails exactly where it does (toy 14) — which `κ = 0.15` does not rescue either. Post-selection on
feasibility then *retains* the feasible strings that were sampled; it cannot recover an optimum that was not
sampled, and a run with zero feasible shots is reported as feasibility 0 with undefined gaps.

## 7. What it costs

**Variables** (brief Derive 4.1 table, `python scripts/variable_count.py --write-md`):

| Encoding | Variable count (formula) | S | M | L | Fits in 80 atoms? (S/M/L) |
|---|---|---|---|---|---|
| Assignment variables only, no constraints | N·K | 8 | 30 | 60 | yes/yes/yes |
| + binary-encoded slack (textbook) | N·K + K·⌈log2(Q+1)⌉ | 14 | 45 | 80 | yes/yes/yes (=cap) |
| One-hot, no slack, capacity as load-balance (unbalanced penalization) (ours: `onehot-ub`) | N·K | 8 | 30 | 60 | yes/yes/yes |

20 atoms saved on L (25 % of the budget), 15 on M, 6 on S; no atom whose value is discarded at readout.

**Coupling structure.** Support = the `N×K` rook's graph: `N·K(K−1)/2` clique pairs + `K·N(N−1)/2`
same-vehicle pairs = S 16/28, M 165/435, L 510/1770 nonzero target couplings. The capacity term adds *no*
new pairs (same support as the objective), unlike the textbook's slack block.

**Device scales** (α chosen so `J_max = 2A·α = C6/5⁶ = 55.4 rad/µs`, i.e. the one-hot clique sits at the
minimum separation; `python - <<…` snippet in `out/gate1/scales.txt`):

| instance | A | λ₂ | α | δ_f (rad/µs) | same-vehicle J (rad/µs) | ⇒ atom distances (µm) |
|---|---|---|---|---|---|---|
| S_seed00 | 398.4 | 2.94 | 0.0695 | +27.7 | 3.6 – 13.1 | 6.4 – 7.9 |
| M_seed00 | 2973 | 2.83 | 0.0093 | +27.7 | 1.1 – 4.1 | 7.7 – 9.6 |
| L_seed00 | 4315 | 2.87 | 0.0064 | +27.7 | 0.28 – 3.1 | 8.1 – 12.1 |

`δ_f = 27.7 rad/µs` is far inside the ±125.66 range. These rows assume the clique coupling `2A` is the
largest entry; with the tight `A` that is no longer automatic (`J_max = max(2A, max w_ij)`), and the
structure-aware embedder (DERIVATIONS §5.7) fits `α` and `δ_f` on the one-hot subspace instead of this
pairwise bound. Geometry: a `K = 3` clique is an equilateral triangle at 5 µm; a `K = 4` clique cannot have
six equal 5 µm distances in a plane (a square gives two diagonals at 7.07 µm, coupling ratio `1/8`), so even
the clique alone is realised only approximately on L.

**Embedding — the real cost, measured.** The rook's graph asks the K atoms of customer `i` to sit within
5 µm of each other *and* atom `(i,k)` to sit 6–12 µm from `(j,k)` while `(i,k)`–`(j,l≠k)` should be ≈ 0.
With the generic (all-pairs) embedder this fails as the brief §4 warns: rel. Frobenius 0.95 and, on
S_seed04, a physical ground state that is infeasible (DERIVATIONS §5.7, `results.csv`). The fix is to ask
for less: on one-hot strings each customer pair contributes exactly one coupling, so only
*same-vehicle − cross-vehicle* must match `α·w_ij`, and the cross couplings may be any constant. The
structure-aware embedder (`cvrp/embed_onehot.py`: rigid 5 µm K-gons, fit of device energies on the one-hot
subspace, single-flip defect gaps, fitted `δ_f`) is what the results use; its residual is reported per run
split by pair class (same-customer / same-vehicle / cross), and the register's own classical ground state is
enumerated on tier S and compared with the target's (`realized_register` in every run record).

**Feasible fraction of random bitstrings**: 1.6–4.7 % on S (checker), vs the textbook's 4.5 % raw — similar,
because both are dominated by the `K^N / 2^{NK}` decodable fraction (6.25 % on S). The capacity term does not
make random strings feasible; it moves energy away from infeasible one-hot strings.

**Effect on §9 metrics**: atom efficiency ↑ (table above); the feasibility rate is set jointly by the
formulation, the embedding, the pulse, the emulator settings and the shot count, so it is reported per
(instance, method, seed), never predicted; best model gap 0 % whenever a sampled one-hot string is the right
clustering; P(reach reference) is where the embedding residual shows.

---

## Judge questions (the three Claude would ask) — with the answers to give

**Q1. "Your capacity penalty has a λ₁ in the write-up but no λ₁ in the code. Where did it go?"**
It multiplies `Σ_k (Q − L_k) = KQ − D`, a constant on every one-hot string, so it only shifts the energy of
strings that already violate exactly-one — which `A` handles. It is dropped together with all other
per-customer linear terms (M3) by the one-hot identity `Σ_k x_ik = 1`. The honest consequence: what the
device sees is not an inequality penalty at all but a load-balance term `λ₂ Σ_k L_k²`; "unbalanced
penalization" is where it came from, "balance" is what it does. Derivation: §3b, test
`test_balance_identity_on_one_hot_strings`.

**Q2. "Then what guarantees `L_k ≤ Q`?"**
Nothing, by design. For `K = 2` the balance-optimal split is capacity-feasible whenever a feasible split
exists (one line, §3b); for `K ≥ 3` it is only likely. We score only feasible decodes and report the
feasibility rate (brief Pitfall 9.1, repo rule). The scan in §6 shows the trade-off explicitly: `κ = 0`
leaves 6/40 toy seeds infeasible, `κ = 1` makes all 40 feasible but 13 suboptimal; we chose `κ = 0.15` and
say which seeds it does not rescue (toy 14). On every binding seed we checked, the ordering of energy
among *feasible* strings still matches the objective (Pearson 1.0), so post-selection retains the right
candidate *if it was sampled*; it cannot recover an optimum the register never produced.

**Q3. "Why is `A` so large relative to the objective couplings, and what does that do to the embedding?"**
`A` must exceed the largest coupling a missing customer would pick up on the emptiest vehicle, `R_w/K`
(§6, tight rule); the original `R_w + 1` is `K×` more conservative. On S the original is 2–8× the largest
single objective coupling, on L 9–100×; the tight rule divides that by `K`. On the device the clique coupling
`2A` is pinned at 55.4 rad/µs (5 µm), so the objective couplings land at 0.3–13 rad/µs, i.e. 6–12 µm
distances, comfortably resolvable; `δ_f = 27.7 rad/µs` is uniform and in range. The cost is not the ratio
but the geometry: rook's-graph cross-vehicle pairs cannot be made weak while same-vehicle pairs are strong
(§7). The checker shows the original `A` can be halved once without harm (fails at ×1/4) and the tight `A`
cannot (fails at ×1/2) — the two rules bracket the true threshold.

**Also ask yourself before judging:** why `κ` and not a per-instance `λ₂`? (Because the reference values
are computed per instance and a per-instance weight would be fitted to the answer; a tier-level constant
is the honest choice.) And: what would a DMM buy here? (It is unnecessary to reproduce this model's uniform target fields;
using one to compensate embedding residuals would be a separate, labelled experiment.)
