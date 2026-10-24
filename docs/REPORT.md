---
title: "Capacity-feasible CVRP clustering on an analog neutral-atom register (Track A)"
subtitle: "Pasqal challenge — team DuQuantum — repository `qhack`"
date: "2026-10-25"
geometry: margin=1.8cm
fontsize: 10pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
---

# 1. Model

**Problem (brief §2).** Customers $i=1..N$ with demands $d_i$, $K$ identical vehicles of capacity $Q$, Euclidean costs.
Cluster-first / route-second: the quantum part decides *which* customers share a vehicle; each cluster is then
routed exactly (Held–Karp). Decision variables $x_{ik}\in\{0,1\}$ (customer $i$ on vehicle $k$), one-hot per
customer, $n = NK$ atoms: **8 / 30 / 60** on S / M / L, no slack. Every derivation is in `docs/DERIVATIONS.md`
(§1–§4) and `docs/MODEL.md`; this report states results and decisions.

**Objective.** Total pairwise distance inside each cluster, $f=\sum_k\sum_{i<j} d_{ij}x_{ik}x_{jk}$ — additive over
clusters, so the exact set-partition DP (`cvrp/exact.py`) certifies the *model* optimum on every tier.

**Constraints and what the device sees (DERIVATIONS §2–§4).** At $\Omega=0$, Eq. (1) assigns a bitstring the cost
$E(x) = -\delta_f\sum_i x_i + \sum_{i<j} C_6 r_{ij}^{-6} x_i x_j$: one *uniform* linear coefficient (laser) and
*positive* couplings (positions). Exactly-one [=]: squared penalty $A\sum_i(1-\sum_k x_{ik})^2$ → $h=-A$ on every
variable (uniform (OK)), $J=2A$ on each customer's $K$-clique (positive (OK)). Capacity [≤]: unbalanced penalization
$\sum_k[-\lambda_1(Q-L_k)+\lambda_2(Q-L_k)^2]$ (Montañez-Barrera et al. 2024), expanded; its per-customer linear part
$c_i\sum_k x_{ik}$ is a *constant* on one-hot strings (the one-hot identity), so it is dropped — $\lambda_1$
disappears and what remains is the coupling $+2\lambda_2 d_id_j$ on same-vehicle pairs, i.e. a **load-balance term**
$\lambda_2\sum_k L_k^2$. It is soft: for $K=2$ the balance optimum is capacity-feasible whenever a feasible split exists,
for $K\ge3$ only likely, so we score feasible decodes only and report the feasibility rate (Pitfall 9.1).
Weights: $A=\max_i\sum_{j\ne i}(d_{ij}+2\lambda_2 d_id_j)+1$ (provable: every argmin is one-hot), $\lambda_2=0.15\,R/(2Q)$
from a 40-seed brute-force scan. Brute-force checker on tier S: argmin feasible and optimal on 5/5 shipped
instances; $A$ survives one halving and fails at ¼; $\lambda_2$ is not binding on these instances (`out/gate1/`).

# 2. Variable accounting (brief Derive 4.1)

| Encoding | count | S | M | L | fits 80? |
|---|---|---|---|---|---|
| assignment variables only | $NK$ | 8 | 30 | 60 | yes |
| + binary slack (textbook) | $NK+K\lceil\log_2(Q{+}1)\rceil$ | 14 | 45 | 80 | yes (L = cap) |
| **ours** (`onehot-ub`) | $NK$ | **8** | **30** | **60** | yes |

(`scripts/variable_count.py`; the textbook rows are implemented and checked, DERIVATIONS §3.) Saving: 6 / 15 / 20 atoms,
and no atom whose readout is discarded; the capacity term adds no coupling pairs beyond the objective's.

# 3. Embedding (DERIVATIONS §5)

Fidelity formula reported everywhere: $\|U/\alpha - J\|_F/\|J\|_F$ over off-diagonals, with the worst-pair error next to it.
The generic least-squares embedder cannot realise the one-hot rook's graph: the $K$ copies of a customer sit 5 µm
apart, so the "zero" cross-vehicle couplings come out as large as the wanted ones (rel. Frobenius 0.95; on S_seed04 the
register's own ground state was infeasible). We therefore ask for less: on one-hot strings each customer pair contributes
exactly one coupling, so only *same-vehicle minus cross-vehicle* must equal $\alpha w_{ij}$ (identity 5.7b). Each customer
becomes a rigid 5 µm $K$-gon (clique pinned at the minimum separation, $\alpha = C_6/5^6/2A$); centres and orientations
are fitted so the device energies of all one-hot strings match $\alpha E_\text{QUBO}$, with explicit single-flip
defect gaps and a fitted $\delta_f$ (a constant on the subspace). Registers are cached and committed (`embeddings/`).
Diagnostics per run: full-matrix fidelity, subspace fidelity, and the realised-register check (device argmin feasible?
equal to the model optimum? defect gap). On tier S: argmin = model optimum on 5/5 instances.

**Where it breaks (tier M, $K=3$).** Triangles 5 µm on a side cannot be far from each other's neighbours: the
subspace-fit residual is 38 (S: 0.15–0.32), defects fall below their parents and the model optimum ranks ~10 000th among
one-hot strings; opening the polygons lowers the energy scale without fixing the ordering (DERIVATIONS §5.7, STATUS). This
is the brief's §4 warning in its sharpest form, and it is a property of one-hot encodings on a 2-D $1/r^6$ register.

# 4. Pulse (DERIVATIONS §6)

Three-point Ω ramp (15 % / 70 % / 15 %) and a linear δ sweep from $\delta_0=-\min(\max(|\delta_f|,2\Omega_{\max}),\delta_{\max})$
to $\delta_f$ while Ω is on its plateau; every sequence is built with Pulser on `AnalogDevice` and validated
(`to_abstract_repr()`, limits read from the device object). Scan on three tier-S instances, scored by P(reach), best gap and
feasibility (never mean cost): longer is monotonically better; $\Omega=0.75\,\Omega_{\max}$ beats full and half power;
$\delta_f\times\{0.8,1,1.25\}$ peaks at the fitted value. **Chosen: 5800 ns, $0.75\,\Omega_{\max}$, fitted $\delta_f$**
(35–38 rad/µs on S, inside ±125.66). Emulators: pulser-simulation (qutip) for every headline row, emu-sv as a cross-check,
and an in-house exact state-vector integrator built from Pulser's own samples for scans (agrees with qutip to sampling
noise, TV ≈ 0.05 at 500 shots); emu-mps for M/L with the bond dimension recorded.

# 5. Results (brief Table 3; `results.csv`, `python -m cvrp.results`)

<!-- RESULTS_TABLE -->
| instance | method | feas. | model gap | CVRP gap | P(reach C*) | P(gap≤5%) | fidelity | backend |
|---|---|---|---|---|---|---|---|---|
| S_seed00 | ours | 0.99 | 0.0 % | 0.0 % | 0.90 | 0.90 | 0.26 | qutip |
| S_seed00 | ours, generic emb. | 0.22 | 0.0 % | 0.0 % | 0.13 | 0.13 | 0.95 | qutip |
| S_seed00 | qubo-solver quantum | 1.00 | 0.0 % | 0.0 % | 0.78 | 0.78 | 0.12 | qs-quantum |
| S_seed00 | qubo-solver tabu | 1.00 | 0.0 % | 0.0 % | 1.00 | 1.00 | undef. | qs-classical |
| S_seed00 | random sampler | 0.04 | 0.0 % | 0.0 % | 0.01 | 0.01 | 0.95 | random |
| S_seed01 | ours | 0.88 | 0.0 % | 0.0 % | 0.88 | 0.88 | 0.21 | qutip |
| S_seed01 | ours, generic emb. | 0.09 | 0.0 % | 0.0 % | 0.07 | 0.07 | 0.94 | qutip |
| S_seed01 | qubo-solver quantum | 0.80 | 0.0 % | 0.0 % | 0.78 | 0.78 | 0.16 | qs-quantum |
| S_seed01 | qubo-solver tabu | 1.00 | 0.0 % | 0.0 % | 1.00 | 1.00 | undef. | qs-classical |
| S_seed01 | random sampler | 0.02 | 0.0 % | 0.0 % | 0.01 | 0.01 | 0.94 | random |
| S_seed02 | ours (in-house SV) | 1.00 | 0.0 % | 0.0 % | 0.98 | 0.98 | 0.31 | numpy-sv |
| S_seed02 | random sampler | 0.04 | 0.0 % | 0.0 % | 0.01 | 0.01 | 0.31 | random |
| S_seed03 | ours (in-house SV) | 1.00 | 0.0 % | 15.6 % | 0.01 | 0.97 | 0.29 | numpy-sv |
| S_seed03 | random sampler | 0.05 | 0.0 % | 15.6 % | 0.01 | 0.01 | 0.29 | random |
| S_seed04 | ours | 0.99 | 0.0 % | 21.7 % | 0.09 | 0.62 | 0.32 | qutip |
| S_seed04 | ours, generic emb. | 0.06 | 0.0 % | 21.7 % | 0.00 | 0.03 | 0.95 | qutip |
| S_seed04 | qubo-solver quantum | 1.00 | 27.3 % | 30.0 % | 0.00 | 0.00 | 0.19 | qs-quantum |
| S_seed04 | qubo-solver tabu | 1.00 | 0.0 % | 21.7 % | 0.00 | 1.00 | undef. | qs-classical |
| S_seed04 | random sampler | 0.04 | 0.0 % | 21.7 % | 0.01 | 0.01 | 0.95 | random |

(generated by `python scripts/report_tables.py` from `results.csv`; raw shots only — repair rows are in results.csv under their own label; shots: 500 per emulator row, 2000 for the random sampler, 1 for a classical solver)
<!-- /RESULTS_TABLE -->

References are proven optima for every instance (exhaustive set-partition DP × exact TSP; SA and naive enumeration agree;
CPLEX unavailable). Decomposition gap (model optimum routed exactly vs $C^*$):

<!-- DECOMPOSITION_TABLE -->
| instance | ρ | C* (proven) | model optimum routed exactly | decomposition gap |
|---|---|---|---|---|
| S_seed00 | 0.714 | 262.97 | 262.97 | 0.0 % |
| S_seed01 | 0.857 | 245.02 | 245.02 | 0.0 % |
| S_seed02 | 0.714 | 254.62 | 254.62 | 0.0 % |
| S_seed03 | 0.643 | 197.44 | 228.34 | 15.6 % |
| S_seed04 | 0.714 | 184.42 | 224.49 | 21.7 % |
<!-- /DECOMPOSITION_TABLE -->

Reading: with the structure-aware register and the scanned pulse, the device returns the model-optimal clustering in
≈ 90–98 % of raw shots on tier S (random sampler: 0.7–1 %; generic embedder: 7–13 %). Where P(reach $C^*$) is low with a
0 % model gap (S_seed03/04) the loss is the decomposition, not the device. qubo-solver's own pipeline (BLaDE + its drive
shaper, 5940 ns) reaches 0.78 on two instances and fails on S_seed04; its classical tabu search finds every model optimum,
and its default SA path does not (a single-start greedy descent at $T_0=5$ — we report a stated-parameter SA instead).

# 6. What did not work

See `docs/DID_NOT_WORK.md` (one sentence each): no PyPI/GitHub from the sandboxes (Windows job runner instead); CPLEX
not installable; a typed number caught by review; qubo-solver's default SA; the emu-mps calibration job blocking the queue;
cross-platform non-reproducibility of the embedding optimiser (fixed by committing registers); the one-hot rook's graph
not embedding for $K=3$ (tier M), which is the honest limit of this encoding on this device.

# 7. Use of AI

`docs/AI_USE.md`. Claude wrote the code, the plumbing and most of the prose from the team's decisions; the Gate-1
formulation (`docs/MODEL.md`) was AI-drafted at the team's request under time pressure and checked by brute force; the
team must re-derive it before judging. Every number in this report is produced by a script from committed records.
