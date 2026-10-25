<!-- length: temporarily allowed > 4 pages; final must be <= 4 (brief D5) -->

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

**What we chose, why, and what it costs.** Cluster-first / route-second: the register decides *which* customers share a
vehicle; each cluster is then routed exactly (Held–Karp). The full (customer, vehicle, position) encoding needs $\sim N^2K$
variables — 32 / 300 / 900 on S / M / L — against 80 atoms (DERIVATIONS Q1); clustering needs $NK$. The price is a
*decomposition gap* (our objective's best clustering is not always the CVRP optimum), reported in §5.

**Variables and objective (brief §2).** Customers $i=1..N$ with demands $d_i$, $K$ identical vehicles of capacity $Q$,
Euclidean costs $d_{ij}$. One-hot $x_{ik}\in\{0,1\}$ (customer $i$ on vehicle $k$), $n=NK$ atoms: **8 / 30 / 60** on
S / M / L, no slack. Objective: total pairwise distance inside each cluster, $f=\sum_k\sum_{i<j} d_{ij}x_{ik}x_{jk}$,
additive over clusters, so an exact set-partition DP (`cvrp/exact.py`) certifies the *model* optimum on every tier.

**Constraints → penalties (DERIVATIONS §4, `docs/MODEL.md`).** Exactly-one [=]: squared penalty
$A\sum_i(1-\sum_k x_{ik})^2$ → $h=-A$ on every variable, $J=2A$ on each customer's $K$-clique. Capacity [≤]: unbalanced
penalization $\sum_k[-\lambda_1(Q-L_k)+\lambda_2(Q-L_k)^2]$ (Montañez-Barrera et al. 2024), expanded. Its per-customer
linear part $c_i\sum_k x_{ik}$ is a *constant* on one-hot strings (the one-hot identity), so it is dropped: $\lambda_1$
disappears and what remains is $+2\lambda_2 d_id_j$ on same-vehicle pairs, a **soft load-balance term**
$\lambda_2\sum_k L_k^2$. For $K=2$ its optimum is capacity-feasible whenever a feasible split exists; for $K\ge3$ only
likely — so we score feasible decodes only and report the feasibility rate (Pitfall 9.1).

**What the device sees.** At $\Omega=0$ the final Hamiltonian is a QUBO with one *uniform* linear coefficient
($\delta_f$) and *positive* couplings ($C_6/r^6$) — exactly the pattern above: every $J>0$, every $h=-A$. Weights:
$A=\max_i\sum_{j\ne i}(d_{ij}+2\lambda_2 d_id_j)+1$ (provable: every argmin is one-hot), $\lambda_2=0.15\,R/(2Q)$ from a
40-seed brute-force scan. Checker on tier S (`out/gate1/`): argmin feasible and optimal on 5/5 shipped instances; $A$
survives one halving and fails at ¼; $\lambda_2$ is not binding on these instances.

# 2. Variable accounting (brief Derive 4.1)

| Encoding | count | S | M | L | fits 80? |
|---|---|---|---|---|---|
| assignment variables only | $NK$ | 8 | 30 | 60 | yes |
| + binary slack (textbook) | $NK+K\lceil\log_2(Q{+}1)\rceil$ | 14 | 45 | 80 | yes (L = cap) |
| **ours** (`onehot-ub`) | $NK$ | **8** | **30** | **60** | yes |

(`scripts/variable_count.py`; textbook rows implemented and checked, DERIVATIONS §3.) Saving: 6 / 15 / 20 atoms; no
discarded readouts; the capacity term adds no coupling pairs beyond the objective's.

# 3. Embedding (DERIVATIONS §5)

**What we chose, why, and what it costs.** A generic least-squares placement cannot realise the one-hot rook's graph, so
we fit the register on the one-hot subspace only (§5.7). For $K=2$ (tier S) the fit realises the model; for $K=3$ (tier M)
exactly-one assignments are produced with ≈ 98–100 % probability while capacity feasibility (4–45 %) and objective
quality remain unresolved (§5, DERIVATIONS §6.7, Q-M3).

**Why the generic embedder fails.** The $K$ copies of a customer form a $2A$-clique 5 µm apart, so a neighbouring
customer's atom is at nearly the same distance from all $K$ copies: the "zero" cross-vehicle couplings come out as large as
the wanted ones. Measured: relative coupling error 0.95 ($\|U/\alpha-J\|_F/\|J\|_F$ over all $i\ne j$, lower is better; `cvrp/embed.py`
`FidelityReport`), always reported with the worst-pair error $\max_{J_{ij}>0}|U_{ij}/\alpha-J_{ij}|/J_{ij}$ — zero-target
pairs are excluded from it and enter separately as the largest spurious coupling $\max_{J_{ij}=0}U_{ij}/\alpha$ — and, on
S_seed04, an infeasible device ground state (model optimum at rank 9).

**Ask for less.** On one-hot strings each customer pair contributes exactly one coupling, so only *same-vehicle minus
cross-vehicle* must equal $\alpha w_{ij}$ (identity 5.7b); cross couplings need only be constant. Each customer becomes a
rigid 5 µm $K$-gon (clique at the minimum separation, $\alpha=C_6/5^6/2A$); centres and orientations are fitted (L-BFGS, 6
seeded starts) so the device energies of all one-hot strings match $\alpha E_\text{QUBO}$, with explicit single-flip defect
gaps and a fitted $\delta_f$ (a constant on the subspace: it only sets the defect gaps). Registers are cached and committed
(`embeddings/`). Per run: full-matrix and subspace coupling error, and the *realised-register check* (device argmin feasible and
optimal? smallest defect gap). Tier S: 5/5 yes.

\begin{figure}[!t]
\centering
\includegraphics[width=0.30\textwidth]{../report/figures/register.png}\hfill
\includegraphics[width=0.36\textwidth]{../report/figures/couplings.png}\hfill
\includegraphics[width=0.30\textwidth]{../report/figures/pulse_scan.png}
\caption{\small Left: S\_seed00 register; each 5~µm dimer is one customer's exactly-one clique, filled atoms = excited atoms of the modal feasible shot (one per dimer: a one-hot string). Middle: target vs realised couplings; clique entries exact, cross-dimer entries not — the full-matrix coupling error (0.263) is dominated by what the fit does not need (worst pair 3.40), the subspace error (0.382) by what it does. Right: pulse scan, three tier-S instances (in-house SV, 500 shots, mean): both curves rise with duration at every Ω; 0.75~Ω$_{\max}$ is on top at 4000 and 5800~ns; Ω/2 loses feasibility.}
\end{figure}

**Where it breaks (tier M, $K=3$).** Triangles 5 µm on a side cannot be far from each other's neighbours: the
subspace-fit residual is 38 (S: 0.06–0.32) and the model optimum ranks ~10 000th of 59 049 one-hot strings; opening the
triangles to 6–10 µm does not fix the ordering (rank ≥ 1 279; DERIVATIONS §5.7) — the brief's §4 warning in its sharpest
form, a property of one-hot encodings on a 2-D $1/r^6$ register. The cliques themselves do embed, under a bound found on
M (§6.7): at $\Omega=0$ a $K$-gon's one-hot state is its ground state only if $0<\delta_f<U_{\min}$, the coupling of the
*weakest* intra-polygon pair. For $K\le3$ every pair is the 5 µm side, $U_{\min}=U_\text{clique}=55.1$ rad/µs; S fits land at
0.49–0.68 $U$, the ill-conditioned M fit ran to the device maximum 125.66 (2.28 $U$), so the embedder now enforces the
bound (fallback 0.65 $U_{\min}$). For $K=4$ (tier L) a 5 µm square has 7.07 µm diagonals with $U_\text{side}/8=6.9$ rad/µs,
so the L registers run at $\delta_f=4.5$ rad/µs (results.csv: 4.475); the first L run (δ_f 38.7) was killed before
reporting (DID_NOT_WORK). Four points in a plane are never equidistant (max/min distance ≥ √2, a factor 8 in $1/r^6$), so a
planar one-hot clique for $K\ge4$ is never uniform.

# 4. Pulse (DERIVATIONS §6)

**Chosen: $T=5800$ ns (device maximum minus 200 ns), $\Omega=0.75\,\Omega_{\max}$, fitted $\delta_f$** (35–38 rad/µs on
S, inside ±125.66). Shape: three-point Ω ramp (15 % / 70 % / 15 % of $T$), linear δ sweep from
$\delta_0=-\min(\max(|\delta_f|,2\Omega_{\max}),\delta_{\max})$ to $\delta_f$ while Ω is on its plateau; every sequence is
built with Pulser on `AnalogDevice` and validated (`to_abstract_repr()`, limits from the device object). Why: a
scan on three tier-S instances (Fig. 1 right, §6.6), scored by P(reach), best gap and feasibility, never mean cost: longer
is monotonically better (2000 → 4000 → 5800 ns at every Ω); $0.75\,\Omega_{\max}$ beats full and half power at 4000 and
5800 ns; $\delta_f\times\{0.8,1,1.25\}$ peaks at the fitted value. S_seed02/03, not used in the scan, reach the model optimum in
99 % / 96 % of shots with these settings. Emulators: pulser-simulation (qutip) for every S headline row; emu-sv cross-check (S_seed00: 0.96
vs 0.94); an in-house exact state-vector integrator from Pulser's own samples for the scans (TV 0.05–0.07 from qutip at
500 shots = sampling noise); emu-mps for M/L, bond dimension recorded.

# 5. Results (brief Table 3; `results.csv`, `python -m cvrp.results`)

> **Reading the table.** One row per (instance, method); raw shots, no repair. *feas.*: fraction of shots decoding to a
> capacity-feasible assignment. *model gap*: best feasible shot vs the exact clustering-model optimum (set-partition DP);
> *CVRP gap*: that shot routed exactly vs $C^*$ (proven on S, best known on M/L) — both on feasible decodes only,
> "undefined" if none. *P(reach $C^*$)*: shots whose clustering routes to $C^*$ (includes the decomposition loss);
> *P(gap ≤ 5 %)*: shots within 5 % of the model optimum — equal to P(reach model optimum) on every S instance and on
> M_seed00/01 (smallest non-zero feasible gap 11–62 %), not on M_seed02 (6 feasible assignments within 5 %, smallest
> 0.98 %; Q-M3). *fidelity*: relative coupling error $\|U/\alpha-J\|_F/\|J\|_F$ over the full matrix, lower is better (§3).
> Rows: **ours** = headline (`onehot-ub`, polygon register, 5800 ns / 0.75 Ω~max~; qutip on S, emu-mps on M); **tight A** =
> ablation; **random assignments (one-hot)** = fair control (assignment-valid random strings, same decoder); **random
> sampler** = uniform bitstrings; **qubo-solver quantum / tabu** = Pasqal's library on our QUBO, its own embedding and drive.

<!-- RESULTS_TABLE -->
| instance | method | feas. | model gap | CVRP gap | P(reach C*) | P(gap≤5%) | fidelity | backend |
|------|------------------------|----|------|------|------|------|-----|-----------|
| S_seed00 | ours | 1.00 | 0.0 % | 0.0 % | 0.94 | 0.94 | 0.26 | qutip |
| S_seed00 | ours, tight A (ablation) | 1.00 | 0.0 % | 0.0 % | 0.03 | 0.03 | 0.31 | qutip |
| S_seed00 | random assignments (one-hot) | 0.63 | 0.0 % | 0.0 % | 0.12 | 0.12 | 0.26 | random-onehot |
| S_seed00 | qubo-solver quantum | 1.00 | 0.0 % | 0.0 % | 0.78 | 0.78 | 0.12 | qs-quantum |
| S_seed00 | qubo-solver tabu | 1.00 | 0.0 % | 0.0 % | 1.00 | 1.00 | undef. | qs-classical |
| S_seed00 | random sampler | 0.04 | 0.0 % | 0.0 % | 0.01 | 0.01 | 0.95 | random |
| S_seed01 | ours | 0.95 | 0.0 % | 0.0 % | 0.95 | 0.95 | 0.21 | qutip |
| S_seed01 | ours, tight A (ablation) | 0.52 | 0.0 % | 0.0 % | 0.52 | 0.52 | 0.29 | qutip |
| S_seed01 | random assignments (one-hot) | 0.25 | 0.0 % | 0.0 % | 0.12 | 0.12 | 0.21 | random-onehot |
| S_seed01 | qubo-solver quantum | 0.80 | 0.0 % | 0.0 % | 0.78 | 0.78 | 0.16 | qs-quantum |
| S_seed01 | qubo-solver tabu | 1.00 | 0.0 % | 0.0 % | 1.00 | 1.00 | undef. | qs-classical |
| S_seed01 | random sampler | 0.02 | 0.0 % | 0.0 % | 0.01 | 0.01 | 0.94 | random |
| S_seed02 | ours | 1.00 | 0.0 % | 0.0 % | 0.99 | 0.99 | 0.31 | qutip |
| S_seed02 | qubo-solver quantum | 0.95 | 0.0 % | 0.0 % | 0.18 | 0.18 | 0.21 | qs-quantum |
| S_seed02 | qubo-solver tabu | 1.00 | 0.0 % | 0.0 % | 1.00 | 1.00 | undef. | qs-classical |
| S_seed02 | random sampler | 0.04 | 0.0 % | 0.0 % | 0.01 | 0.01 | 0.31 | random |
| S_seed03 | ours | 1.00 | 0.0 % | 15.6 % | 0.01 | 0.96 | 0.29 | qutip |
| S_seed03 | qubo-solver quantum | 0.99 | 0.0 % | 15.6 % | 0.18 | 0.79 | 0.14 | qs-quantum |
| S_seed03 | qubo-solver tabu | 1.00 | 0.0 % | 15.6 % | 0.00 | 1.00 | undef. | qs-classical |
| S_seed03 | random sampler | 0.05 | 0.0 % | 15.6 % | 0.01 | 0.01 | 0.29 | random |
| S_seed04 | ours | 1.00 | 0.0 % | 21.7 % | 0.02 | 0.89 | 0.32 | qutip |
| S_seed04 | ours, tight A (ablation) | 0.99 | 0.0 % | 21.7 % | 0.02 | 0.94 | 0.34 | qutip |
| S_seed04 | random assignments (one-hot) | 0.62 | 0.0 % | 21.7 % | 0.11 | 0.13 | 0.32 | random-onehot |
| S_seed04 | qubo-solver quantum | 1.00 | 27.3 % | 30.0 % | 0.00 | 0.00 | 0.19 | qs-quantum |
| S_seed04 | qubo-solver tabu | 1.00 | 0.0 % | 21.7 % | 0.00 | 1.00 | undef. | qs-classical |
| S_seed04 | random sampler | 0.04 | 0.0 % | 21.7 % | 0.01 | 0.01 | 0.95 | random |
| M_seed00 | ours (emu-mps) | 0.04 | 67.9 % | 29.9 % | 0.00 | 0.00 | 0.58 | emu-mps χ=32 |
| M_seed00 | qubo-solver quantum | 0.02 | 72.1 % | 35.2 % | 0.00 | 0.00 | 0.09 | qs-quantum χ=32 |
| M_seed00 | random sampler | 0.00 | undef. | undef. | 0.00 | 0.00 | 0.58 | random |
| M_seed01 | ours (emu-mps) | 0.39 | 37.9 % | 9.2 % | 0.00 | 0.00 | 0.08 | emu-mps χ=32 |
| M_seed01 | qubo-solver quantum | 0.37 | 47.5 % | 11.2 % | 0.00 | 0.00 | 0.08 | qs-quantum χ=32 |
| M_seed01 | qubo-solver tabu | 1.00 | 65.4 % | 17.6 % | 0.00 | 0.00 | undef. | qs-classical |
| M_seed01 | random sampler | 0.00 | undef. | undef. | 0.00 | 0.00 | 0.08 | random |
| M_seed02 | ours (emu-mps) | 0.45 | 23.0 % | 16.0 % | 0.00 | 0.00 | 0.47 | emu-mps χ=32 |
| M_seed02 | qubo-solver quantum | 0.39 | 34.2 % | 13.6 % | 0.00 | 0.00 | 0.09 | qs-quantum χ=32 |
| M_seed02 | qubo-solver tabu | 1.00 | 10.2 % | 2.8 % | 0.00 | 0.00 | undef. | qs-classical |
| M_seed02 | random sampler | 0.00 | undef. | undef. | 0.00 | 0.00 | 0.47 | random |
| L_seed00 | random sampler | 0.00 | undef. | undef. | 0.00 | 0.00 | 0.65 | random |
| L_seed01 | random sampler | 0.00 | undef. | undef. | 0.00 | 0.00 | 0.65 | random |
| L_seed02 | random sampler | 0.00 | undef. | undef. | 0.00 | 0.00 | 0.64 | random |

(generated by `python scripts/report_tables.py` from `results.csv`; raw shots only — repair rows are in results.csv under their own label; shots: 500 per emulator row, 2000 for the random sampler, 1 for a classical solver)
<!-- /RESULTS_TABLE -->

**Reference $C^*$ (brief §9.3).** CPLEX is not importable, so $C^*$ is the best of (i) qubo-solver's SA (stated
parameters) on our penalized `onehot-ub` QUBO and (ii) our constrained SA on the CVRP, each 10 seeded restarts, **60 s
max solve time** (`instances/*.json`). On tier S the exhaustive set-partition DP is also run and agrees 5/5, which is the
brief's level 1, so S references are `proven_optimum`; M/L are `best_known`; the CPLEX cross-check stays an outstanding
item (DID_NOT_WORK). The
QUBO-side SA reaches $C^*$ on 5/5 S and 2/3 M instances but stays 23–34 % above it on all three L (599.9 / 554.6 / 521.7 vs
447.0 / 434.9 / 424.7; Q-ref): the penalized one-hot QUBO is harder to search than the constrained problem, so $C^*$ is
never the QUBO solver alone. Decomposition gap (model optimum routed exactly vs $C^*$):

<!-- DECOMPOSITION_TABLE -->
| instance | ρ | C* (best known) | model optimum routed exactly | decomposition gap |
|---|---|---|---|---|
| S_seed00 | 0.714 | 262.97 | 262.97 | 0.0 % |
| S_seed01 | 0.857 | 245.02 | 245.02 | 0.0 % |
| S_seed02 | 0.714 | 254.62 | 254.63 | 0.0 % |
| S_seed03 | 0.643 | 197.44 | 228.34 | 15.6 % |
| S_seed04 | 0.714 | 184.42 | 224.49 | 21.7 % |
| M_seed00 | 0.950 | 432.00 | 432.00 | 0.0 % |
| M_seed01 | 0.733 | 368.36 | 368.36 | 0.0 % |
| M_seed02 | 0.700 | 360.86 | 385.62 | 6.9 % |
| L_seed00 | 0.650 | 446.97 | 525.71 | 17.6 % |
| L_seed01 | 0.708 | 434.91 | 470.67 | 8.2 % |
| L_seed02 | 0.600 | 424.66 | 480.17 | 13.1 % |
<!-- /DECOMPOSITION_TABLE -->

**Tier S, in four steps.** (1) *Register.* On every instance the realised register's own argmin is the model optimum
(device rank 1); relative coupling error 0.21–0.32 vs 0.94–0.95 for the generic embedder, which put the optimum at rank 9 on
S_seed04. (2) *Feasibility.* 0.95–1.00 of raw shots are capacity-feasible (generic register: 0.06–0.22; uniform random:
0.02–0.05). (3) *P(reach) vs controls — useful quantum work (brief §9.2).* The model-optimal clustering is returned in
89–99 % of raw shots on all five instances. The fair control, `random-onehot`, reaches it with P = 2/16 ≈ 0.12 on every
instance (two labelings of one optimal clustering among 16 assignments; feasible 25–63 %); the headline rows give 0.94 /
0.95 / 0.89 on S_seed00/01/04 at equal decoding — a 7–8× factor, so the register, not the decoder, does the work (uniform
bitstrings: 0.7–1.2 %; generic embedder: 3–13 %). qubo-solver's own pipeline (BLaDE + its drive shaper, 5940 ns): 0.18–0.79
on four instances, 27 % model gap on S_seed04; its tabu search finds every model optimum (its default SA, a single-start
greedy descent at $T_0=5$, does not; we report a stated-parameter SA instead). (4) *Decomposition.* Where P(reach $C^*$) is
low with a 0 % model gap (S_seed03/04: 0.01 / 0.02) the loss is the decomposition gap (15.6 % / 21.7 %; M_seed02: 6.9 %):
the closeness objective's optimum is not the CVRP optimum — the cost of cluster-first (§1), not of the device.

**Ablation — the exactly-one margin is physically load-bearing.** `onehot-ub-tight` replaces $A=\max_i\sum_j w_{ij}+1$
by the tight bound $A=R_w/K\,(1+1\%)$ (still sound, checker-verified), same register construction, pulse tuned for the
original $A$: P(reach model optimum) 0.94 → 0.03 on S_seed00 (device ground state = third-best clustering), 0.95 → 0.52
on S_seed01, 0.89 → 0.94 on S_seed04 (qutip; in-house SV 0.03 / 0.56 / 0.96): a smaller $A$ shrinks the single-flip defect
gaps the sweep must resolve and leaves less margin for the embedding error. The headline stays `onehot-ub`.

**Tier M — exactly-one assignments are produced with ≈ 98–100 % probability; capacity feasibility (4–45 %) and
objective quality remain unresolved (DERIVATIONS §6.7, Q-M2, Q-M3).** *Before:* our first M rows had $\delta_f=125.66$
rad/µs $=2.28\,U_\text{clique}$, so two excitations per triangle were favoured and 0 of 500 shots were decodable (kept as
`|pre-6.7` in results.csv) — a wrong sweep end point, not a feasibility limit of the encoding. *After* (same registers,
bounded $\delta_f$, emu-mps $\chi=32$, 500 shots; rows "ours (emu-mps)"): decodable 0 → 0.994 / 0.984 / 0.996, feasible
0 → 0.044 / 0.388 / 0.446 on M_seed00/01/02, best model gap 67.9 / 37.9 / 23.0 %, CVRP gap 29.9 / 9.2 / 16.0 %. A uniform
random sampler decodes 0 of 2000 shots on every M instance (table; M_seed01 re-measured on the cached register), so the
one-hot shots are the device's own work. *Where the remaining loss sits* (Q-M3, all 59 049 one-hot strings enumerated,
`docs/checks/landscapes.md`): (i) *penalty* — the load-balance term leaves the model optimum at QUBO rank 1 on every S
instance and on M_seed00/01, but on M_seed02 the QUBO argmin is a feasible assignment 10.2 % above the model optimum
(rank 331) — exactly qubo-solver's tabu row — so ~10 of M_seed02's 23 points of model gap are the $\lambda_2$ rule, not
the register; (ii) *register* — the optimum's device rank among one-hot strings is 10 474 / 12 966 / 4 192 and is the
main loss; (iii) *sweep* — the emulated best feasible shots (67.9 / 37.9 / 23.0 %) beat the register's own best feasible
one-hot string under $E_\text{dev}$ (82.0 / 101.3 / 93.9 %), so the pulse is not the bottleneck. qubo-solver's independent
pipeline (BLaDE register, $\chi=32$) lands in the same regime on all three M (feasible 0.016 / 0.37 / 0.388, decodable
0.99 / 0.998 / 0.992, model gap 72.1 / 47.5 / 34.2 %, CVRP gap 35.2 / 11.2 / 13.6 %), while its tabu search on the same
QUBO reaches 17.6 % / 2.8 % above $C^*$ on M_seed01/02: the QUBO is solvable classically; the planar register limits it.
A $K=2$ bisection does not help (GATE2_OPTIONS (f): rank 192 594 / 12 217).

**Tier L — measured rows pending.** Three 60-atom registers were built with the $K=4$ bound ($\delta_f=4.475$ rad/µs); the
model optimum sits at one-hot rank 16 534 / 1 063 / 14 862 of 1 048 576 (results.csv, random rows), and the uniform random
control decodes 0 of 2000 shots on all three; the decomposition gap is already known (17.6 / 8.2 / 13.1 %, table above). Emulations (emu-mps $\chi=32$) and an M_seed00 $\chi=64$ convergence run
are running; rows are reported as measured.

# 6. What did not work

`docs/DID_NOT_WORK.md`, one sentence each: no PyPI from the sandboxes; CPLEX not importable; a typed number caught by
review; qubo-solver's default SA; a blocking calibration job; a weight scan with no trend (the embedder's local minimum
moves P(reach) more than the weights); a non-reproducible embedder (fixed by committing registers); the tight-$A$ ablation;
an unbounded $\delta_f$ that made the first tier-M rows undecodable (§6.7); the triangle bound applied to tier-L squares
(first L run killed before reporting); a `|` in a run tag that lost one Windows record; the polygon optimiser giving two
registers from one seed when runs overlap (only cache-loaded registers are reported now); the CPLEX cross-check, still
outstanding; and the one-hot *ordering* not embedding for $K=3$ — the limit of this encoding on this device.

# 7. Use of AI

`docs/AI_USE.md`. Claude wrote the code, plumbing and most prose from the team's decisions; the Gate-1 formulation
(`docs/MODEL.md`) was AI-drafted at the team's request under time pressure, checked by brute force, and must be
re-derived by the team before judging. Every number here is produced by a script from committed records.
