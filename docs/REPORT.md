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

**Bottom line.** The register decides which customers share a vehicle; each group is then routed exactly. One atom per
(customer, vehicle) pair: 8 / 30 / 60 atoms on tiers S / M / L. On tier S the device returns the best grouping in
89–99 % of shots. On M it returns valid groupings but poor ones. On L it returns almost nothing valid. §5 has the numbers,
§6 the reasons.

**Why cluster-first.** Encoding the full route (customer, vehicle, position) needs about $N^2K$ atoms: 32 / 300 / 900
for S / M / L, against 80 available (DERIVATIONS Q1). Deciding only the grouping needs $NK$. The cost is a
*decomposition gap*: the best grouping under our objective is not always the grouping of the shortest routes. It is
0 % on 5 of 11 instances and up to 21.7 % on the rest (§5, decomposition table).

**Variables.** $N$ customers with demands $d_i$; $K$ identical vehicles of capacity $Q$; $d_{ij}$ the distance between
customers $i$ and $j$. The bit $x_{ik}=1$ means customer $i$ rides on vehicle $k$. A valid grouping has exactly one
$1$ in every customer's row ("one-hot").

**Objective.** We want groups whose members are close to each other, so we add up the distances between every pair
of customers that share a vehicle:
\begin{equation}
f(x)=\sum_{k=1}^{K}\ \sum_{i<j} d_{ij}\,x_{ik}\,x_{jk}.
\end{equation}
It costs us the decomposition gap above, and it is additive over groups, which is what lets an exact dynamic programme
certify the model optimum on every tier (`cvrp/exact.py`).

**Constraint 1 — every customer on exactly one vehicle.** Enforced by a squared penalty with weight $A$:
\begin{equation}
A\sum_{i=1}^{N}\Big(1-\sum_{k=1}^{K}x_{ik}\Big)^2 .
\end{equation}
Expanded, it puts the same linear term $-A$ on every atom and a coupling $+2A$ between the $K$ atoms of one customer
(a "clique"). It costs nothing in atoms; it costs energy scale: $A$ must exceed every other term (below), and it is the
largest coupling the register has to realise.

**Constraint 2 — vehicle load $L_k=\sum_i d_i x_{ik}$ at most $Q$.** We use unbalanced penalization
(Montañez-Barrera et al. 2024), $\sum_k\big[-\lambda_1(Q-L_k)+\lambda_2(Q-L_k)^2\big]$, because it needs no slack
atoms. On one-hot strings its linear part is the same constant for every valid grouping, so it drops out together with
$\lambda_1$ (the "one-hot identity", DERIVATIONS §4); what survives is a load-balance term
\begin{equation}
\lambda_2\sum_{k=1}^{K} L_k^2 \;=\; \text{const}+2\lambda_2\sum_k\sum_{i<j} d_i d_j\,x_{ik}x_{jk}.
\end{equation}
It costs us a guarantee: for $K=2$ its minimum is capacity-feasible whenever a feasible split exists; for $K\ge3$ it is
not guaranteed. We therefore score feasible decodes only and report the feasibility rate (brief Pitfall 9.1).

**The whole cost function** is Eq. (1) + Eq. (2) + Eq. (3). Every coupling is positive ($d_{ij}\ge0$, $2A>0$,
$2\lambda_2 d_id_j>0$) and every linear term is $-A$: exactly the pattern an analog register with repulsive
$C_6/r^6$ interactions and one global detuning can produce. The pairs that interact are the $N{\times}K$ rook's graph
(same customer, or same vehicle); the capacity term adds no new pairs.

**Weights.** $A=\max_i\sum_{j\ne i}(d_{ij}+2\lambda_2 d_id_j)+1$, which provably makes every minimiser one-hot
(DERIVATIONS §4). $\lambda_2=0.15\,R/(2Q)$ with $R=\max_i\sum_j d_{ij}$, from a 40-seed brute-force scan. Brute force
on the five S instances: the minimiser is feasible and optimal on 5/5; $A$ survives one halving and fails at $A/4$;
$\lambda_2$ is not binding on any S instance.

# 2. Variable accounting (brief Derive 4.1)

| Encoding | count | S | M | L | fits 80? |
|---|---|---|---|---|---|
| assignment variables only | $NK$ | 8 | 30 | 60 | yes |
| + binary slack (textbook) | $NK+K\lceil\log_2(Q{+}1)\rceil$ | 14 | 45 | 80 | yes (L = cap) |
| **ours** (`onehot-ub`) | $NK$ | **8** | **30** | **60** | yes |

Tiers: S $N{=}4,K{=}2,Q{=}7$; M $N{=}10,K{=}3,Q{=}20$; L $N{=}15,K{=}4,Q{=}30$ (`scripts/variable_count.py`). We
save 6 / 15 / 20 atoms over the textbook encoding and never discard a readout bit.

# 3. Embedding (DERIVATIONS §5)

**Bottom line.** A generic least-squares placement cannot realise the rook's graph. We fit the register only on the
one-hot strings instead. On tier S ($K=2$) that reproduces the model exactly; on M ($K=3$) it reproduces the
exactly-one constraint but not the ordering of groupings; on L ($K=4$) the geometry of a square caps the energy
scale below what the sweep can resolve.

**Why generic placement fails.** The $K$ atoms of one customer must sit 5 µm apart (the clique). Any atom of another
customer is then at nearly the same distance from all $K$ of them, so the couplings that should be zero
(different customer *and* different vehicle) come out as large as the wanted ones. Measured: relative coupling error
0.95, defined as $\|U/\alpha-J\|_F/\|J\|_F$ over all pairs (lower is better; `cvrp/embed.py`), and on S_seed04 the
register's ground state was an infeasible string.

**Ask for less.** On a one-hot string each pair of customers has exactly one active coupling, so only the difference
*same-vehicle minus cross-vehicle* has to match the model (DERIVATIONS 5.7b); the cross couplings only need to be
constant. Each customer becomes a rigid 5 µm $K$-gon, which realises the clique exactly with
$\alpha=C_6/(5\,\mu\text{m})^6/2A$; the centres and orientations are fitted (L-BFGS, 6 seeded starts) so that the device
energies of all one-hot strings match $\alpha\,E_\text{QUBO}$, with single-flip defects forced above their parents and
the final detuning $\delta_f$ as a fit parameter. Registers are committed in `embeddings/`; every run checks the
realised register (is its own minimiser feasible and optimal? smallest defect gap). Tier S: 5/5 yes, coupling error
0.21–0.32.

\begin{figure}[!t]
\centering
\includegraphics[width=0.30\textwidth]{../report/figures/register.png}\hfill
\includegraphics[width=0.36\textwidth]{../report/figures/couplings.png}\hfill
\includegraphics[width=0.30\textwidth]{../report/figures/pulse_scan.png}
\caption{\small Left: S\_seed00 register; each 5~µm dimer is one customer's exactly-one clique, filled atoms are the excited atoms of the modal feasible shot (one per dimer). Middle: target vs realised couplings; clique entries exact, cross-dimer entries not — the full-matrix coupling error (0.263) is dominated by pairs the fit does not need (worst pair 3.40), the one-hot-subspace error (0.382) by pairs it does. Right: pulse scan on three tier-S instances (in-house state-vector emulator, 500 shots, mean): longer is better at every Ω; 0.75~Ω$_{\max}$ is on top at 4000 and 5800~ns; Ω/2 loses feasibility.}
\end{figure}

**A bound we found on tier M.** At the end of the sweep ($\Omega=0$) one atom of a $K$-gon excited costs $-\delta_f$;
two excited cost $-2\delta_f+U_{\min}$, where $U_{\min}$ is the coupling of the *weakest* pair inside the polygon. One
excitation per customer is the ground state only if
\begin{equation}
0<\delta_f<U_{\min}.
\end{equation}
For $K\le3$ every pair is the 5 µm side, $U_{\min}=55.1$ rad/µs; the S fits land at $\delta_f=35$–38 rad/µs
(0.64–0.68 $U_{\min}$). On M the fit is ill-conditioned and ran to the device maximum, 125.66 rad/µs
$=2.28\,U_{\min}$, which favours *two* vehicles per customer; the embedder now enforces Eq. (4) (fallback
$0.65\,U_{\min}$). For $K=4$ a 5 µm square has 7.07 µm diagonals, and $1/r^6$ turns that $\sqrt2$ into a factor 8:
$U_{\min}=6.9$ rad/µs, so the L registers run at $\delta_f=4.5$ rad/µs. Four points in a plane are never
equidistant, so a planar one-hot clique for $K\ge4$ is never uniform (DERIVATIONS §6.7).

# 4. Pulse (DERIVATIONS §6)

**Chosen: 5800 ns (device maximum minus 200 ns), $\Omega=0.75\,\Omega_{\max}$, fitted $\delta_f$.** Three-point
$\Omega$ ramp (15 % up, 70 % plateau, 15 % down) and a linear detuning sweep during the plateau from
$\delta_0=-\min(\max(|\delta_f|,2\Omega_{\max}),\delta_{\max})$ to $\delta_f$. Starting far below resonance puts every
atom in $|g\rangle$; ending at $\delta_f>0$ with $\delta_f<U_{\min}$ makes one excitation per customer the target
state. Every sequence is built with Pulser on `AnalogDevice` and validated against the device object.

**Why these values.** A scan on three S instances (Fig. 1 right; DERIVATIONS §6.6), scored by P(reach), best gap and
feasibility, never mean cost: longer is better at every $\Omega$ (2000 → 4000 → 5800 ns); $0.75\,\Omega_{\max}$ beats
full and half power at 4000 and 5800 ns; $\delta_f\times\{0.8,1,1.25\}$ gives 0.915 / 0.932 / 0.884, so the fitted value
stays. S_seed02/03, not used in the scan, reach the model optimum in 99 % / 96 % of shots with these settings.
Emulators: pulser-simulation (QuTiP) for every S headline row; emu-sv cross-check on S_seed00 (0.96 vs 0.94); an
in-house exact state-vector integrator built from Pulser's own samples for the scans (total-variation distance
0.05–0.07 from QuTiP at 500 shots, i.e. sampling noise); emu-mps for M and L with the bond dimension recorded.

# 5. Results (brief Table 3; full table: `results.csv`, 76 rows, `python -m cvrp.results`)

**How to read a row.** Raw shots, no repair. *feas.*: fraction of shots that decode to a capacity-feasible grouping.
*model gap*: best feasible shot against the exact model optimum; *CVRP gap*: that shot routed exactly against $C^*$
(proven on S, best known on M/L); both on feasible decodes only, "undefined" if there are none. *P(reach $C^*$)*:
shots whose grouping routes to $C^*$, so it includes the decomposition loss. *P(gap ≤ 5 %)*: shots within 5 % of the
model optimum; on every S instance and on M_seed00/01 this equals P(reach model optimum) because the next-best feasible
grouping is 11–62 % worse, on M_seed02 it does not (six groupings within 5 %; DERIVATIONS Q-M3). *fidelity*: relative
coupling error of the full matrix, lower is better.

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
| S_seed02 | ours | 1.00 | 0.0 % | 0.0 % | 0.99 | 0.99 | 0.31 | qutip |
| S_seed03 | ours | 1.00 | 0.0 % | 15.6 % | 0.01 | 0.96 | 0.29 | qutip |
| S_seed04 | ours | 1.00 | 0.0 % | 21.7 % | 0.02 | 0.89 | 0.32 | qutip |
| M_seed00 | ours (emu-mps) | 0.04 | 67.9 % | 29.9 % | 0.00 | 0.00 | 0.58 | emu-mps χ=32 |
| M_seed00 | qubo-solver quantum | 0.02 | 72.1 % | 35.2 % | 0.00 | 0.00 | 0.09 | qs-quantum χ=32 |
| M_seed01 | ours (emu-mps) | 0.39 | 37.9 % | 9.2 % | 0.00 | 0.00 | 0.08 | emu-mps χ=32 |
| M_seed01 | qubo-solver quantum | 0.37 | 47.5 % | 11.2 % | 0.00 | 0.00 | 0.08 | qs-quantum χ=32 |
| M_seed02 | ours (emu-mps) | 0.45 | 23.0 % | 16.0 % | 0.00 | 0.00 | 0.47 | emu-mps χ=32 |
| M_seed02 | qubo-solver quantum | 0.39 | 34.2 % | 13.6 % | 0.00 | 0.00 | 0.09 | qs-quantum χ=32 |
| L_seed00 | ours (emu-mps) | 0.01 | 140.0 % | 79.7 % | 0.00 | 0.00 | 0.65 | emu-mps χ=32 |

(generated by `python scripts/report_tables.py` from `results.csv`; raw shots only — repair rows are in results.csv under their own label; shots: 500 per emulator row, 2000 for the random sampler, 1 for a classical solver; controls and ablations are shown for S_seed00 only — every other row of every method (random controls, tight-A ablation, tabu, pre-6.7 rows) is in results.csv)
<!-- /RESULTS_TABLE -->

**Reference $C^*$ (brief §9.3).** CPLEX is not importable, so $C^*$ is the best of qubo-solver's simulated annealing on
our penalized QUBO and our constrained annealing on the CVRP, each 10 seeded restarts with a 60 s max solve time
(settings in `instances/*.json`). On S the exhaustive dynamic programme is also run and agrees 5/5, so S references
are proven; M/L are best known. The QUBO-side annealer reaches $C^*$ on 5/5 S and 2/3 M instances but stays 23–34 %
above it on all three L (599.9 / 554.6 / 521.7 vs 447.0 / 434.9 / 424.7), so $C^*$ is never the QUBO solver alone.

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

**Tier S — the device does the work.** The model-optimal grouping is returned in 89–99 % of raw shots on all five
instances, with 0.95–1.00 of shots feasible. The fair control — random assignment-valid strings through the same
decoder — reaches it with probability 2/16 ≈ 0.12 on every instance; uniform random bitstrings 0.7–1.2 %; the generic
embedder 3–13 %. qubo-solver's own pipeline (its embedding and drive, 5940 ns) reaches 0.78 / 0.78 / 0.18 / 0.79 / 0.00
and has a 27 % model gap on S_seed04; its tabu search finds every model optimum classically. Where P(reach $C^*$) is low
with a 0 % model gap (S_seed03/04: 0.01 / 0.02) the loss is the decomposition gap (15.6 % / 21.7 %): the closeness
objective's best grouping is not the shortest-route grouping. That is the price of §1, not of the device.

**Ablation — the margin in $A$ is physical.** Replacing $A$ by its tight provable bound (DERIVATIONS Q-A), same register
construction and pulse, moves P(reach model optimum) from 0.94 to 0.03 on S_seed00, 0.95 to 0.52 on S_seed01 and
0.89 to 0.94 on S_seed04: a smaller $A$ shrinks the defect gaps the sweep must resolve.

**Tier M — valid groupings, poor ones.** With $\delta_f$ bounded by Eq. (4), 98–100 % of shots are one-hot and
4.4 % / 38.8 % / 44.6 % are capacity-feasible on M_seed00/01/02; the best feasible shots are 67.9 / 37.9 / 23.0 % above
the model optimum (29.9 / 9.2 / 16.0 % CVRP gap). A uniform random sampler decodes 0 of 2000 shots on every M
instance. We enumerated all 59 049 one-hot groupings to locate the loss (DERIVATIONS Q-M3, `docs/checks/landscapes.md`):
(i) the load-balance term leaves the model optimum at QUBO rank 1 on M_seed00/01, but on M_seed02 the QUBO minimum is a
feasible grouping 10.2 % above it — exactly what qubo-solver's tabu search returns — so on M_seed02 about 10 of the
23 points are the $\lambda_2$ rule; (ii) under the realised register the model optimum sits at rank 10 474 / 12 966 /
4 192 of 59 049, which is the main loss; (iii) the emulated best shots beat the register's own best feasible string
(82.0 / 101.3 / 93.9 %), so the sweep is not the bottleneck. qubo-solver's independent pipeline lands in the same
regime (feasible 0.016 / 0.37 / 0.388, CVRP gap 35.2 / 11.2 / 13.6 %), while its tabu search on the same QUBO reaches
17.6 % / 2.8 % above $C^*$ on M_seed01/02: the QUBO is solvable classically; the planar register limits it.

**Tier L — almost nothing valid.** L_seed00 (60 atoms, emu-mps $\chi=32$, 500 shots, $\delta_f=4.475$ rad/µs):
1.4 % of shots one-hot, 0.8 % feasible, best feasible shot 140 % above the model optimum (79.7 % CVRP gap). The
cause is Eq. (4): with $U_{\min}=6.9$ rad/µs the whole one-hot landscape of 60 atoms is compressed into a few rad/µs
(smallest defect gap 1.1 rad/µs against 19–23 on M), and a sweep of 30 rad/µs in 4 µs is far from adiabatic there
(rate ≈ 7 rad/µs² against a gap-squared of ≈ 1.2). The model optimum sits at one-hot rank 16 534 / 1 063 / 14 862 of
1 048 576 on L_seed00/01/02; L_seed01/02 rows and an M_seed00 $\chi=64$ convergence run are in `results.csv` as
measured.

# 6. What did not work

- **Generic embedding of the one-hot graph.** Tried least-squares placement of all $NK$ atoms; coupling error 0.95, an
  infeasible ground state on S_seed04, P(reach) 3–13 %. Cause: the clique forces the zero couplings to be as large as
  the wanted ones. Replaced by the one-hot-subspace fit (§3): P(reach) 0.89–0.99.
- **Unbounded $\delta_f$ on tier M.** The first M rows had $\delta_f=125.66$ rad/µs $=2.28\,U_{\min}$: 0 of 500 shots
  decodable on M_seed00/02. Cause: two excitations per triangle were favoured. Fixed by Eq. (4); the old rows stay in
  `results.csv` under `|pre-6.7`. Then the triangle version of the bound was applied to tier-L squares: the first L run
  ($\delta_f=38.7$) was killed before reporting and the bound rewritten for the weakest pair.
- **The one-hot ordering for $K=3$ and $K=4$.** Opening the M triangles from 5 to 6–10 µm lowers the energy scale
  without fixing the ordering (optimum rank ≥ 1 279); a $K=2$ bisection puts it at rank 192 594 / 12 217
  (`docs/GATE2_OPTIONS.md`). We did not find a planar register that orders one-hot groupings by Eq. (1) for $K\ge3$.
- **The $\lambda_2$ rule on M_seed02.** The load-balance weight moves the QUBO optimum 10.2 % above the model optimum
  on one of three M instances (0 % elsewhere). We kept the rule because changing weights after seeing results was
  ruled out; the effect is quantified in Q-M3.
- **Register reproducibility.** The polygon optimiser gave two different 30-atom registers from one seed when two runs
  overlapped (M_seed01, optimum rank 1 114 vs 12 966). Rule adopted: every reported row loads the committed register;
  one control row was re-measured.
- **Classical references.** CPLEX is not importable; the S agreement check is brute force vs two annealers. qubo-solver's
  default annealer is a single-start descent at $T_0=5$ and misses the S optima, so a stated-parameter annealer is used.
- **Weight scan.** A 12-point scan of $(A,\lambda_2)$ around the rule showed no trend: the embedder's local minimum
  moves P(reach) more than the weights do (DERIVATIONS §4.1).
- **Tooling.** A `|` in a run tag lost one Windows record; a silent LaTeX failure left a stale PDF once. Both are caught
  by the build scripts now.
