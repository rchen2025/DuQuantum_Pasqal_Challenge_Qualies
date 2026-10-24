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
vehicle; each cluster is then routed exactly (Held–Karp, `cvrp/common.py`). The full (customer, vehicle, position)
encoding needs $\sim N^2K$ variables — 32 / 300 / 900 on S / M / L — against 80 atoms (DERIVATIONS Q1); clustering needs
$NK$. The price is a *decomposition gap* (the best clustering under our objective is not always the CVRP optimum), which
we report separately in §5.

**Variables and objective (brief §2).** Customers $i=1..N$ with demands $d_i$, $K$ identical vehicles of capacity $Q$,
Euclidean costs $d_{ij}$. One-hot $x_{ik}\in\{0,1\}$ (customer $i$ on vehicle $k$), $n=NK$ atoms: **8 / 30 / 60** on
S / M / L, no slack. Objective: total pairwise distance inside each cluster, $f=\sum_k\sum_{i<j} d_{ij}x_{ik}x_{jk}$.
It is additive over clusters, so an exact set-partition DP (`cvrp/exact.py`) certifies the *model* optimum on every tier.

**Constraints → penalties (DERIVATIONS §4, `docs/MODEL.md`).** Exactly-one [=]: squared penalty
$A\sum_i(1-\sum_k x_{ik})^2$ → $h=-A$ on every variable, $J=2A$ on each customer's $K$-clique. Capacity [≤]: unbalanced
penalization $\sum_k[-\lambda_1(Q-L_k)+\lambda_2(Q-L_k)^2]$ (Montañez-Barrera et al. 2024), expanded. Its per-customer
linear part $c_i\sum_k x_{ik}$ is a *constant* on one-hot strings (the one-hot identity), so it is dropped: $\lambda_1$
disappears and what remains is $+2\lambda_2 d_id_j$ on same-vehicle pairs, a **soft load-balance term**
$\lambda_2\sum_k L_k^2$. For $K=2$ its optimum is capacity-feasible whenever a feasible split exists; for $K\ge3$ only
likely. We therefore score feasible decodes only and report the feasibility rate (Pitfall 9.1).

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

(`scripts/variable_count.py`; textbook rows implemented and checked, DERIVATIONS §3.) Saving: 6 / 15 / 20 atoms, none
with a discarded readout; the capacity term adds no coupling pairs beyond the objective's.

# 3. Embedding (DERIVATIONS §5)

**What we chose, why, and what it costs.** A generic least-squares placement cannot realise the one-hot rook's graph, so
we fit the register on the one-hot subspace only (§5.7). This works for $K=2$ (tier S) and fails for $K=3$ (tier M).

**Why the generic embedder fails.** The $K$ copies of a customer form a $2A$-clique and must sit 5 µm apart, so a
neighbouring customer's atom is at nearly the same distance from all $K$ copies: the "zero" cross-vehicle couplings come
out as large as the wanted ones. Measured: rel. Frobenius error 0.95 (the brief's formula, $\|U/\alpha-J\|_F/\|J\|_F$ over
off-diagonals, always reported with the worst-pair error) and, on S_seed04, an infeasible device ground state (model
optimum at device-energy rank 9).

**Ask for less.** On one-hot strings each customer pair contributes exactly one coupling, so only *same-vehicle minus
cross-vehicle* must equal $\alpha w_{ij}$ (identity 5.7b); cross couplings need only be constant. Each customer becomes a
rigid 5 µm $K$-gon (clique pinned at the minimum separation, $\alpha=C_6/5^6/2A$); centres and orientations are fitted
(L-BFGS, 6 seeded starts) so that the device energies of all one-hot strings match $\alpha E_\text{QUBO}$, with explicit
single-flip defect gaps and a fitted $\delta_f$ (a constant on the subspace, so it only sets the defect gaps). Registers
are cached and committed (`embeddings/`). Per run we record full-matrix and subspace fidelity and the *realised-register
check*: is the device's own argmin feasible and the model optimum, and the smallest defect gap. Tier S: 5/5 yes.

\begin{figure}[H]
\centering
\includegraphics[width=0.30\textwidth]{../report/figures/register.png}\hfill
\includegraphics[width=0.36\textwidth]{../report/figures/couplings.png}\hfill
\includegraphics[width=0.30\textwidth]{../report/figures/pulse_scan.png}
\caption{\small Left: S\_seed00 register; each 5~µm dimer is one customer's exactly-one clique, filled atoms are the excited atoms of the modal feasible shot (one per dimer = a one-hot string). Middle: target vs realised couplings; clique entries are exact, cross-dimer entries are not — the full-matrix error (0.263) is dominated by what the fit does not need (worst pair 3.40), the subspace error (0.382) by what it does. Right: pulse scan on three tier-S instances (in-house SV, 500 shots, mean over instances): both curves rise with duration at every Ω, the 0.75~Ω$_{\max}$ curve is on top at 4000 and 5800~ns, Ω/2 loses feasibility.}
\end{figure}

**Where it breaks (tier M, $K=3$).** Triangles 5 µm on a side cannot be far from each other's neighbours: the
subspace-fit residual is 38 (S: 0.06–0.32), defects fall below their parents and the model optimum ranks ~10 000th of
59 049 one-hot strings; opening the triangles to 6–10 µm lowers the energy scale without fixing the ordering (rank ≥ 1 279;
DERIVATIONS §5.7, DID_NOT_WORK). This is the brief's §4 warning in its sharpest form, and it is a property of one-hot
encodings on a 2-D $1/r^6$ register, not of the weights.

# 4. Pulse (DERIVATIONS §6)

**Chosen: $T=5800$ ns (device maximum minus 200 ns), $\Omega=0.75\,\Omega_{\max}$, fitted $\delta_f$** (35–38 rad/µs on
S, inside ±125.66). Shape: three-point Ω ramp (15 % / 70 % / 15 % of $T$) and a linear δ sweep from
$\delta_0=-\min(\max(|\delta_f|,2\Omega_{\max}),\delta_{\max})$ to $\delta_f$ while Ω is on its plateau; every sequence is
built with Pulser on `AnalogDevice` and validated (`to_abstract_repr()`, limits read from the device object). Why: a
scan on three tier-S instances (Fig. 1 right, §6.6), scored by P(reach), best gap and feasibility — never mean cost. Longer is
monotonically better (2000 → 4000 → 5800 ns at every Ω); $0.75\,\Omega_{\max}$ beats full and half power at 4000 and
5800 ns; $\delta_f\times\{0.8,1,1.25\}$ peaks at the fitted value. S_seed02/03, not used in the scan, reach the model
optimum in 99 % / 96 % of shots with these settings. Emulators: pulser-simulation (qutip) for every headline row; emu-sv
as a cross-check (S_seed00: P = 0.96 vs 0.94); an in-house exact state-vector integrator built from Pulser's own samples
for the scans (agrees with qutip to sampling noise, TV 0.05–0.07 at 500 shots); emu-mps for M/L, bond dimension recorded.

# 5. Results (brief Table 3; `results.csv`, `python -m cvrp.results`)

> **Reading the table.** One row per (instance, method); raw shots, no repair. *feas.*: fraction of shots that decode to
> a capacity-feasible assignment. *model gap*: best feasible shot vs the proven optimum of the clustering objective;
> *CVRP gap*: that shot routed exactly vs the proven CVRP optimum $C^*$ — both on feasible decodes only, "undefined"
> when none exists. *P(reach $C^*$)*: fraction of shots whose clustering routes to $C^*$ (so it includes the
> decomposition loss); *P(gap ≤ 5 %)*: fraction within 5 % of the model optimum (on these instances equal to P(reach model
> optimum)). *fidelity*: full-matrix rel. Frobenius error, lower is better (§3: it understates the subspace error). Rows:
> **ours** = headline (`onehot-ub`, polygon register, 5800 ns / 0.75 Ω~max~, qutip); **tight A** = ablation; **random
> assignments (one-hot)** = the fair control (assignment-valid random strings through the same decoder); **random
> sampler** = uniform bitstrings; **qubo-solver quantum / tabu** = Pasqal's library on our QUBO with its own embedding and drive.

<!-- RESULTS_TABLE -->
| instance | method | feas. | model gap | CVRP gap | P(reach C*) | P(gap≤5%) | fidelity | backend |
|---|---|---|---|---|---|---|---|---|
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

**Reading, in four steps.** (1) *Register.* On every tier-S instance the realised register's own argmin is the model
optimum (device rank 1); full-matrix fidelity 0.21–0.32 against 0.94–0.95 for the generic embedder, which put the optimum
at rank 9 on S_seed04. (2) *Feasibility.* 0.95–1.00 of raw shots decode to a capacity-feasible assignment (generic
register: 0.06–0.22; uniform random: 0.02–0.05). (3) *P(reach) against the controls — useful quantum work (brief §9.2).*
The model-optimal clustering is returned in 89–99 % of raw shots on all five instances. The fair control,
`random-onehot`, reaches it with P = 2/16 ≈ 0.12 on every instance (two labelings of one optimal clustering among 16
assignments; capacity-feasible 25–63 % of the time); the headline rows give 0.94 / 0.95 / 0.89 on S_seed00/01/04 at equal
decoding — a 7–8× factor, so the register, not the decoder, does the work. Uniform random bitstrings: 0.7–1.2 %; generic
embedder: 3–13 %. qubo-solver's own pipeline (BLaDE + its drive shaper, 5940 ns) reaches 0.18–0.79 on four instances and
a 27 % model gap on S_seed04; its tabu search finds every model optimum (its default SA path, a single-start greedy
descent at $T_0=5$, does not — we report a stated-parameter SA instead). (4) *Decomposition.* Where P(reach $C^*$) is low
with a 0 % model gap (S_seed03/04: 0.01 / 0.02) the loss is the decomposition gap (15.6 % / 21.7 %): the closeness
objective's optimum is not the CVRP optimum. That is the cost of cluster-first (§1), not of the device.

**Ablation — the exactly-one margin is physically load-bearing.** `onehot-ub-tight` replaces $A=\max_i\sum_j w_{ij}+1$
by the tight bound $A=R_w/K\,(1+1\%)$ (still sound: the optimum stays the QUBO ground state, checker-verified) and changes
nothing else — same register construction, pulse tuned for the original $A$. P(reach model optimum) goes 0.94 → 0.03 on
S_seed00 (the realised register's ground state becomes the third-best clustering), 0.95 → 0.52 on S_seed01 and
0.89 → 0.94 on S_seed04 (qutip; in-house SV 0.03 / 0.56 / 0.96). A smaller $A$ shrinks the single-flip defect gaps the
sweep must resolve and leaves less margin for the embedding error. The headline stays `onehot-ub`.

**Tiers M and L — the measured limit of the encoding.** The one-hot register does not embed for $K=3$ (§3, DERIVATIONS
Q-M1): generic and polygon registers both put the model optimum at rank ~10 000 of 59 049 one-hot strings, and a uniform
random sampler decodes 0 of 2000 shots. The measured consequence on emu-mps, M_seed00 at $\chi=32$: **0 of 500 shots
feasible**, so model and CVRP gaps are undefined (STATUS, job 070; the row enters the table above as it is written to
`results.csv`). M_seed01/02, L_seed00–02 ($\chi=32$) and an M $\chi=64$ convergence run are queued; we expect
feasibility ≈ 0 and report what is measured. A $K=2$ bisection does not rescue M/L either (`docs/GATE2_OPTIONS.md` (f):
model optimum at rank 192 594 / 12 217).

# 6. What did not work

`docs/DID_NOT_WORK.md`, one sentence each: no PyPI from the sandboxes; CPLEX not installable; a typed number caught by
review; qubo-solver's default SA; a blocking calibration job; a 12-point weight scan with no trend (the embedding
optimiser's local minimum moves P(reach) more than the weights do); a non-reproducible embedding optimiser (fixed by
committing registers); the tight-$A$ ablation; the one-hot rook's graph not embedding for $K=3$ — the honest limit of
this encoding on this device.

# 7. Use of AI

`docs/AI_USE.md`. Claude wrote the code, the plumbing and most of the prose from the team's decisions; the Gate-1
formulation (`docs/MODEL.md`) was AI-drafted at the team's request under time pressure and checked by brute force; the
team must re-derive it before judging. Every number in this report is produced by a script from committed records.
