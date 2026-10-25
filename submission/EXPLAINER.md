# The whole story, in plain language

*A human-readable companion to the report and the README. It explains the problem we were given, the machine we were asked to use, every model we chose and why, how we turned a model into atom positions and a laser pulse, every decision taken along the way, and every error we hit. No prior knowledge of quantum hardware is assumed. Where a number appears, it comes from a run record or `results.csv` in the repository.*

---

## 1. The problem we were given

A delivery company has one depot, a list of customers, and a fleet of identical vans. Every customer wants a known number of parcels (their *demand*); every van can carry at most *Q* parcels; every van leaves the depot once, visits some customers, and returns. The company wants the plan that serves everyone exactly once, never overloads a van, and drives the fewest total kilometres. That is the **Capacitated Vehicle Routing Problem (CVRP)**. Parcel carriers, grocery distributors, waste collection and hospital supply chains solve a version of it every night, and it is NP-hard: exact solvers give up long before the sizes that appear in daily operations.

The challenge used three sizes ("tiers"):

| tier | customers N | vans K | capacity Q | demand per customer |
|---|---|---|---|---|
| S | 4 | 2 | 7 | 1–4 |
| M | 10 | 3 | 20 | 1–8 |
| L | 15 | 4 | 30 | 1–10 |

Two constraints matter and they are different in kind:

* **"Every customer on exactly one van"** is an *equality*. There is one right answer (one), and any deviation is wrong.
* **"No van above capacity"** is an *inequality*. A van carrying 3 of 7 parcels is perfectly fine; so is one carrying 7. There is no single target to aim at.

That asymmetry is where almost all the difficulty of this challenge lives, because the machine we were asked to use can only minimise an *unconstrained* function. Equalities are easy to fold into such a function; inequalities are not.

**Cluster first, route second.** Deciding, for every customer, both *which* van and *in what order* needs a variable for every (customer, van, position) triple — roughly $N^2 K$ of them: 32, 300 and 900 on the three tiers, against a machine that holds 80. So the challenge prescribes a two-step heuristic: first decide which customers share a van (a *clustering* problem, $N K$ variables), then order each van's customers separately (a tiny travelling-salesman problem per van, solved exactly on a laptop). The price is that the best clustering under whatever "closeness" measure we choose is not always the clustering whose routes are shortest. That difference is called the **decomposition gap**, and we measure it on every instance instead of pretending it is zero.

Track A of the challenge is the clustering step on the quantum device. Track B adds the ordering step on the device as well.

---

## 2. The machine: a neutral-atom register

Pasqal's device is an array of individual atoms, each held in place by a focused laser beam (an "optical tweezer"). **You choose where every atom sits** in a plane, subject to three limits: at most 80 atoms, no two closer than 5 micrometres, all within a disc of radius 38 micrometres. A single global laser then drives every atom between its ground state $\lvert g\rangle$ and a highly excited "Rydberg" state $\lvert r\rangle$. At the end, each atom is measured: ground = 0, Rydberg = 1. So one run of the machine returns a string of $N$ bits — exactly the kind of thing a binary optimisation problem wants.

The physics during the run is governed by the Hamiltonian in the brief,

$$
\frac{H(t)}{\hbar} = \sum_i \frac{\Omega(t)}{2}\, \hat\sigma^x_i \;-\; \delta(t) \sum_i \hat n_i \;+\; \sum_{i<j} \frac{C_6}{\lVert \mathbf r_i - \mathbf r_j \rVert^6}\, \hat n_i \hat n_j
$$

Read it term by term:

* **The Rabi term,** $\tfrac{\Omega(t)}{2}\,\hat\sigma^x_i$. The laser amplitude $\Omega$ "mixes" the two states of every atom: it is what lets the system explore different bit strings. It is the *only* term that changes which strings are present; the other two only assign energies. The device allows $\Omega$ up to 12.57 rad/µs.
* **The detuning term,** $-\delta(t) \sum_i \hat n_i$. Here $\hat n_i$ is 1 if atom $i$ is excited and 0 otherwise. A positive $\delta$ lowers the energy of every excited atom by the same amount (makes excitation attractive); a negative $\delta$ raises it. Crucially, $\delta$ is **global**: the same for every atom. The range is $\pm 125.66$ rad/µs.
* **The interaction term,** $\tfrac{C_6}{r_{ij}^6}\, \hat n_i \hat n_j$. Two excited atoms repel each other with an energy that falls off as the sixth power of their distance. $C_6$ is a fixed constant of the atom species; you cannot change it, and you cannot make the interaction attractive. At the minimum distance of 5 µm the interaction is 55.4 rad/µs; at 10 µm it is $2^6 = 64$ times smaller.

**What the register minimises.** If you switch the laser amplitude off at the end ($\Omega = 0$), the only terms left assign a plain number to every bit string $x$:

$$
E_{\mathrm{dev}}(x) = -\delta_f \sum_i x_i \;+\; \sum_{i<j} \frac{C_6}{r_{ij}^6}\, x_i x_j
$$

The first term is $-\delta_f$ times the number of 1s; the second adds a repulsion for every pair of atoms that are both 1.

This is a **QUBO** — a quadratic function of binary variables — with three rules baked in: every variable has the **same** linear coefficient ($-\delta_f$), every pairwise coupling is **positive**, and no coupling is ever exactly zero (atoms are at most 76 µm apart). The adiabatic recipe is: start with the laser far below resonance so every atom is in $\lvert g\rangle$ (that is the ground state there), slowly turn $\Omega$ on, sweep $\delta$ up to the value you want at the end, turn $\Omega$ off. If "slowly" is slow enough compared with the energy gap between the ground state and the next state, the system stays in its ground state the whole way and ends in the lowest-energy bit string of $E_{\mathrm{dev}}$. Measure it, and you have a sample of the minimum of your QUBO.

**What you control and what you don't.** The laser controls one number for all atoms ($\delta_f$) and how adiabatic the sweep is ($\Omega$ and duration, up to 6000 ns). The positions control every coupling, but only as a planar distance pattern. $C_6$ and the sign are physics. Everything we did for two days was an attempt to make the CVRP clustering look like $E_{\mathrm{dev}}$.

---

## 3. The clustering model we chose, and the ones we did not

### 3.1 The textbook approach and why it is a trap

The standard way to put the clustering on a QUBO is: one bit $x_{ik}$ per (customer, van); an objective that rewards "close" customers sharing a van; a squared penalty $A\big(1 - \sum_k x_{ik}\big)^2$ for "exactly one van"; and, for capacity, a **slack variable** per van — a non-negative integer $s_k$ that absorbs the difference $Q - L_k$ (where $L_k$ is the van's load), written in binary with $\lceil \log_2(Q+1) \rceil$ bits, so that the inequality becomes an equality ($L_k + s_k = Q$) that can be squared and penalised.

We implemented exactly this (`textbook`) to make the pipeline run end to end on day one. It is provably sound and it costs 14 / 45 / 80 atoms — tier L lands exactly on the 80-atom cap with 20 atoms (a quarter of the machine) holding slack bits whose measured values are simply thrown away. Worse, it is not realisable on this device for two independent reasons: the squared capacity penalty gives every atom a *different* linear coefficient (it depends on the customer's demand and the slack bit's weight — a spread of a factor three), which a single global detuning collapses to the average, destroying exactly the capacity bookkeeping; and it couples every pair of a van's $N + B$ variables with nearly equal weight, which is the dense all-to-all pattern a flat $1/r^6$ register handles worst.

### 3.2 The alternatives we weighed

The team ranked the known ways around slack variables before coding:

* **Domain-wall encoding** of "which van" — $K-1$ bits per customer with a chain of couplings. Fewer bits than one-hot for $K \ge 3$, but it needs *negative* (ferromagnetic) couplings along the chain, which this device cannot produce.
* **Binary encoding** of the van index — $\lceil \log_2 K \rceil$ bits per customer. Fewest bits, but the objective becomes a high-order polynomial that must be reduced with auxiliary variables, and the reduction introduces negative couplings and non-uniform linear terms.
* **One-hot + unbalanced penalization** — $K$ bits per customer, the capacity inequality penalised by a two-term expansion with no auxiliary variable. Every coupling positive, every linear term uniform (after the identity below), sparsest coupling graph, fits the atom budget on all tiers.

The third option won on every device-facing criterion, and it is what we built (`onehot-ub`).

### 3.3 The model, in words

* **Variables.** $x_{ik} = 1$ means "customer $i$ rides on van $k$". $N K$ bits: 8 / 30 / 60. Nothing else.
* **Objective ("closeness").** The sum of the distances between every pair of customers that share a van, $f(x) = \sum_k \sum_{i<j} d_{ij}\, x_{ik} x_{jk}$. We chose this over "distance to a cluster centre" because it needs no extra variables, its couplings are already positive, and it is *additive over clusters* — which lets us compute the exact model optimum on every tier by a dynamic programme and so measure every gap against a certified number. Its known bias: it grows like the square of the cluster size, so it prefers small clusters, which incidentally does part of the capacity constraint's work for free.
* **Exactly-one.** The squared penalty $A\sum_i\big(1 - \sum_k x_{ik}\big)^2$. Expanding it gives the same linear term $-A$ on *every* bit and a coupling $+2A$ between the $K$ bits of each customer (a little clique). Good: uniform linear term, positive couplings, no new atoms.
* **Capacity.** Write the slack as $g_k = Q - L_k \ge 0$ and penalise it by $-\lambda_1 g_k + \lambda_2 g_k^2$ (unbalanced penalization, Montañez-Barrera et al. 2024). Expanding gives a linear term per bit that depends on the customer's demand, $c_i\, x_{ik}$ with $c_i = d_i(\lambda_1 - 2\lambda_2 Q + \lambda_2 d_i)$, plus a coupling $2\lambda_2 d_i d_j$ between any two customers on the same van.

**The one-hot identity — the heart of the model.** On any string that satisfies exactly-one, each customer has exactly one bit set, so $\sum_k c_i x_{ik} = c_i$, and the sum over customers is a *constant*:

$$
\sum_i \sum_k c_i\, x_{ik} = \sum_i c_i \qquad \text{on every one-hot string.}
$$

The entire linear part of the capacity penalty — and $\lambda_1$ with it — is invisible on the strings where the model lives. We drop it. What survives is the coupling $2\lambda_2 d_i d_j$, which on one-hot strings equals

$$
2\lambda_2 \sum_k \sum_{i<j} d_i d_j\, x_{ik} x_{jk} = \lambda_2 \sum_k L_k^2 + \text{const},
$$

a **load-balance** term that prefers evenly loaded vans. So "capacity" on this device is not a hard wall at $Q$; it is a soft push toward balance. For two vans, the balance term alone always has a feasible minimiser if any feasible split exists; once you add the distance objective that guarantee goes away, and for three or more vans there never was one. The honest consequence, which the brief itself anticipates: we **score only shots that decode to a capacity-feasible assignment** and always report the fraction that do. Infeasible shots never contribute a cost.

* **Weights.** $A = \max_i \sum_{j\ne i} w_{ij} + 1$ (the largest row sum of the same-van couplings $w_{ij} = d_{ij} + 2\lambda_2 d_i d_j$, plus one). We proved that with this $A$ every global minimiser is one-hot: adding a missing van to a customer always lowers the energy, removing a duplicate van always lowers it. A tighter $A$, $K$ times smaller, is also proved and was tested as an ablation. $\lambda_2 = \kappa\, R/(2Q)$ with $R = \max_i \sum_j d_{ij}$ and $\kappa = 0.15$, chosen by brute-forcing all 256 bit strings of 40 small random instances: $\kappa = 0$ fails exactly on the seeds where capacity binds, $\kappa = 1$ makes every seed feasible but re-ranks 13 of them so the minimiser is no longer the closest clustering, and no $\kappa$ is right on all 40. The brute-force checker is run on every shipped tier-S instance before any emulation: the minimiser is feasible and optimal on all five, $A$ survives one halving and breaks at a quarter, $\lambda_2$ is not even binding there.

**Atom count:** 8 / 30 / 60 — 6, 15 and 20 atoms saved against the textbook, and not one readout bit discarded.

---

## 4. The ordering model (Track B)

Once a van has its customers, the question is the order to visit them. Ordering is harder to encode than grouping: the cost depends on which customers are *adjacent* in the sequence, not on which share a label. Two encodings were built, both with uniform linear terms and positive couplings by construction:

* **Depot-fixed position encoding,** $\boldsymbol{n^2}$ **bits.** Bit $y_{i,t} = 1$ means "customer $i$ is visited $t$-th". The textbook version also gives the depot a position and spends $(n+1)^2$ bits; ours fixes the depot at the start and end and folds the two depot legs into the weights of the first and last transitions, $w(i,j,t) = c_{ij} + [t{=}1]\,c_{0i} + [t{=}n-1]\,c_{j0}$. On every valid tour the energy is then *exactly* the tour length, and every bit gets the same linear term. Works for any $n$, but $n^2$ bits means a 9-customer cluster already exceeds 80 atoms ($9^2 = 81$).
* **Undirected edge encoding,** $\boldsymbol{n(n+1)/2}$ **bits.** One bit per road segment between any two of the $n+1$ points; the constraint is that every point has exactly two chosen segments; the segment cost is split half-and-half between its two end points so that the linear terms stay uniform. Fewer bits, but it is only an exact TSP for $n \le 4$: from $n = 5$ on, two separate little loops also satisfy "degree two" and can be cheaper than any real tour (we built the instance that shows it: two triangles far apart, ground energy 6 against a best tour of 203). The code refuses $n > 4$.

Both were verified exhaustively — every bit string of every small instance enumerated — to have the optimal tour as their ground state. The policy for the full pipeline: clusters of 1–3 customers are "trivial" (the brief's word: a random tour is optimal with probability $\ge 1/3$) and are routed exactly; clusters of 4 go to the device with both encodings (10 and 16 atoms); clusters of 5 or more are routed classically and labelled so.

---

## 5. Embedding: turning the model into atom positions

### 5.1 What the generic approach does and why it failed

The direct recipe: for every wanted coupling $J_{ij}$, place the atoms at distance $r_{ij} = \left(C_6/\alpha J_{ij}\right)^{1/6}$ for some overall scale $\alpha$; put atoms with a "zero" coupling far apart; solve the resulting geometry problem by least squares with hinges for the 5 µm floor and the 38 µm radius. We report the brief's fidelity formula, the relative Frobenius error $\lVert U/\alpha - J\rVert_F \,/\, \lVert J\rVert_F$ over all pairs, plus the worst single pair and the largest spurious coupling on a pair that should be zero.

For our model the wanted coupling graph is the *rook's graph*: the $K$ bits of each customer form a clique (coupling $2A$, the strongest), bits of different customers on the *same* van are coupled by the distance (weaker), and bits of different customers on *different* vans should be uncoupled. In a plane this is a contradiction. The clique forces a customer's $K$ atoms within 5 µm of each other; any atom of another customer is then at almost the same distance from all $K$ of them, so the couplings that should be zero come out as large as the ones that should not. Measured on tier S: error 0.95, the optimum reached in only 3–13 % of shots, and on S_seed04 the physical ground state of the register was an *infeasible* string.

### 5.2 Ask for less: fit on the one-hot subspace

The way out was to notice what the model actually needs. On a one-hot string each customer has exactly one excited atom, so between two customers exactly one of the $K^2$ couplings is "active". If the cross-van couplings were all equal to some constant, they would shift every one-hot string by the same amount and change nothing. So only the *difference* "same-van coupling minus average cross-van coupling" has to match the model:

$$
U(i,k;\,j,k) - \operatorname*{mean}_{l\ne k} U(i,k;\,j,l) \;=\; \alpha\, w_{ij};
$$

the cross couplings need not be zero, only constant.

The structure-aware embedder therefore makes each customer a **rigid regular polygon of side 5 µm** (a dimer for $K = 2$, a triangle for $K = 3$, a square for $K = 4$) — which realises the exactly-one clique exactly at the minimum separation — and then optimises only the positions and orientations of the polygons so that the device energies of all one-hot strings fit the model energies, with two extra conditions: every "defect" string (a customer with zero or two excited atoms) must sit well above its parent one-hot string, and polygons must not overlap. The final detuning $\delta_f$ is a free parameter of this fit (on the one-hot subspace it is a constant, so it only sets how far defects sit above valid strings). Every register is cached and committed to the repository, because the optimiser is seeded but not bit-identical across machines.

Result on tier S: the realised register's own ground state **is** the model optimum on all five instances; with the baseline pulse the optimum went from 13 / 7 / 3 % of shots to 90 / 88 / 62 % on the committed registers, and feasibility from 22 / 9 / 6 % to 99 / 88 / 99 %.

### 5.3 The bound we found on tier M, and the square problem on tier L

Inside one polygon, exciting one atom costs $-\delta_f$, exciting two costs $-2\delta_f$ plus their repulsion. One excitation is the cheaper option only if $\delta_f$ is smaller than the *weakest* repulsion inside the polygon:

$$
0 < \delta_f < U_{\min} .
$$ For a dimer or a triangle every pair is 5 µm apart and that number is 55.1 rad/µs. The tier-S fits landed at $\delta_f = 35$–$38$ rad/µs, comfortably inside. On tier M the fit was ill-conditioned and ran to the device maximum, 125.66 — more than twice the bound — so the sweep ended in a regime that *rewards* two excitations per triangle, and not a single one of 500 shots decoded. Once the bound was enforced (fallback $0.65\,U_{\min}$), 98–100 % of shots were one-hot again.

A square is worse: its diagonal is $\sqrt2$ times its side, and $1/r^6$ turns $\sqrt2$ into a factor $(\sqrt2)^6 = \mathbf{8}$, so the weakest pair in a 5 µm square repels at only 6.9 rad/µs. The final detuning on tier L is thus capped at about 4.5 rad/µs, and the whole energy landscape of a 60-atom register is squeezed into a few rad/µs. Four points in a plane can never be equidistant, so a planar one-hot clique for $K \ge 4$ is never uniform; this is a limit of the geometry, not of the weights or the pulse.

### 5.4 The overnight tier-L fix

Two geometric changes addressed it. First, atoms of *different* polygons were kept at least 8 µm apart instead of 5, which caps every spurious cross-polygon coupling at $C_6/(8\,\mu\mathrm{m})^6 = 3.3$ rad/µs and roughly doubles the energy gap between valid strings and their defects (1.1 → 2.4 rad/µs). Second, the laser amplitude was scaled down to that energy scale: $\Omega = 0.15\,\Omega_{\max} = 1.9$ rad/µs. The amplitude scan on L_seed00 shows a sharp optimum — at 0.10 nothing decodes at all, at 0.15 feasibility is 20.8 %, at 0.25 it is 6 %, at the tier-S value 0.75 it is 1.6 %. Across the three L instances feasibility rose from under 1 % to 14–23 % and the best route found was 36–47 % above the reference — level with tabu search on the same QUBO. These are the tier-L headline rows.

---

## 6. Pulse shaping: what the laser does

The pulse is the brief's baseline shape, built with Pulser and validated by Pulser against the real device description on every run: the amplitude rises over the first 15 % of the duration, holds, and falls over the last 15 %; during the plateau the detuning sweeps linearly from a large negative value (every atom starts in $\lvert g\rangle$) to the fitted positive $\delta_f$ (one excitation per customer becomes favourable, and the repulsions decide which). The whole thing lasts 5800 ns, 200 ns under the device maximum.

The two knobs — duration and amplitude — were scanned on three tier-S instances and scored the way the brief insists: by the probability of reaching the optimum, the best gap and the feasibility rate, never by the mean cost (which throws away what the good shots tell you). Longer was better at every amplitude, as the adiabatic theorem predicts for a fixed gap. Three-quarters of maximum amplitude beat full power: the plateau value of 9.4 rad/µs sits close to the same-van couplings (3–13 rad/µs) while disturbing the 55 rad/µs cliques less. Scaling $\delta_f$ by 0.8 or 1.25 made things worse, so the fitted value stands. Slowing the sweep over its last third gained 2.5 points — within sampling noise — so the committed default stays the plain linear sweep. Two instances that were not in the scan (S_seed02/03) reach the optimum in 99 % and 96 % of shots with these settings, which is the evidence that the pulse was not over-fitted.

Three emulators were used and cross-checked: Pasqal's pulser-simulation (QuTiP, exact) for every tier-S headline row; emu-sv as a check; an in-house exact state-vector integrator, built from Pulser's own per-nanosecond samples, for the scans (it agrees with QuTiP to sampling noise); and emu-mps, the tensor-network emulator, for 30 and 60 atoms, with the bond dimension and the *peak* bond dimension the state actually needed recorded on every row (M_seed00, M_seed01 and M_seed02 are converged; the L rows run at $\chi = 32$).

---

## 7. What the results mean

The point of this work is not to beat a classical solver — no analog device does that on CVRP today, and the brief says so. The point is how efficiently the problem can be put onto the hardware and how much each implementation step improves what comes out. So the results are best read as a ladder of improvements, each measured before and after on the same instances:

| step | what we changed | what it did |
|---|---|---|
| 1 | Textbook slack encoding → one-hot + unbalanced penalization | atoms 14 / 45 / 80 → **8 / 30 / 60**; every linear term now identical, so one global detuning is enough |
| 2 | Generic embedder → rigid polygons fitted on the one-hot strings | tier S: optimal clustering in 13 / 7 / 3 % of shots → **90 / 88 / 62 %** |
| 3 | Pulse scan: 4000 ns at full power → 5800 ns at three-quarters power | tier S: 90 / 88 / 62 % → **94 / 95 / 89 %** |
| 4 | Enforce the derived bound $\delta_f < U_{\mathrm{clique}}$ | tier M: 0 % of shots decodable → **99–100 %** |
| 5 | Use the weakest pair of the polygon for that bound | tier L: a run that would have favoured two corners per customer was caught and discarded |
| 6–7 | 8 µm spacing between polygons + drive matched to the $K = 4$ energy scale | tier L: feasible shots 0.8 / 0.4 / 0 % → **20.8 / 14.0 / 22.6 %**; best route 80 % → **36–47 %** above the reference |
| 8 | Larger emulator bond dimension on M_seed01 | best route 9.2 % → **5.9 %** above the reference |

What each tier tells us:

* **Tier S** shows the full method working. All five instances return the model-optimal, feasible clustering in 89–99 % of raw shots. On two instances the best clustering routes 16 % and 22 % longer than the true CVRP optimum, but the model gap is zero — the register found the closest clustering, and the "closest" objective simply isn't the shortest-route one there. That is the decomposition gap, a property of the objective, not of the device.
* **Tier M** shows the first physical limit and the fix for it: once the final detuning respects the derived bound, almost every shot is a valid assignment. By enumerating all 59 049 valid strings we located the remaining loss precisely: not the capacity penalty (free on two of three instances), not the pulse (the sampled shots beat the register's own best valid string), but the way a plane of triangles orders the clusterings.
* **Tier L** is the geometry result. A square of atoms has an 8× weaker diagonal, which squeezes the whole energy landscape of a 60-atom register into a few rad/µs. Spacing the polygons and turning the laser down to that scale took feasible shots from under 1 % to 14–23 %, with routes 36–47 % above the reference.
* **Track B** closes the loop end to end and splits the loss exactly into decomposition, clustering and routing parts. Both routing encodings are exactly correct; the generic embedding is what limits them, and a structure-aware routing register — built the way Track A's was — is the next step.

For context, the brief asks how much useful work the quantum part does compared with a random sampler; those reference numbers (random bitstrings $\approx 1$ %, random valid assignments 12.5 % on tier S) and Pasqal's own `qubo-solver` pipeline are in the README's §9, next to the full tables.

---

## 8. Every decision, in order, with the reason

1. **Build the ugliest working pipeline first** (Sat 13:00). Textbook formulation, random sampler, one instance, so that every later change had a before/after row. Reason: the brief's own advice; a working baseline is the only way to know whether an improvement helps.
2. **Depot at the centre (50, 50).** The clustering objective never sees the depot; it enters only through the exact routing. A central depot makes the routed cost least sensitive to which side of the square a cluster lies on, so the CVRP gap reflects the clustering, not an asymmetric depot choice.
3. **Certificate-first instances.** Draw the assignment before the demands so feasibility is guaranteed by construction and the certificate is free; keep load ratios at 0.6–0.95 rather than pushing to 1, where the certificate would be the only feasible clustering and the optimisation trivial.
4. **Pairwise-distance objective**, for additivity (exact references on every tier) and positive couplings.
5. **One-hot + unbalanced penalization** over domain-wall and binary encodings, from a ranked comparison on device constraints: all couplings positive, uniform linear terms, sparsest graph, fits the budget.
6. **Drop the per-customer linear terms by the one-hot identity** rather than approximating them with a DMM. They are exactly constant where the model lives; nothing is lost. No DMM is used anywhere, so every row is comparable with QPU runs.
7. $\boldsymbol{A}$ **by the provable row-sum rule,** $\boldsymbol{\lambda_2}$ **by a brute-force scan at** $\boldsymbol{\kappa = 0.15}$, and a tier-level constant rather than a per-instance weight, because the reference values are computed per instance and a per-instance weight would be fitted to the answer.
8. **Score feasible decodes only; never a mean cost; compare clusterings as sets; repair is a separately labelled method.** All four are brief requirements made into repository rules enforced in code.
9. **Switch from the generic embedder to the one-hot-subspace polygon fit** (Sat 15:00) after measuring error 0.95 and an infeasible physical ground state. Reason: the model only needs the effective same-minus-cross coupling.
10. **Cache and commit every register** after discovering the optimiser is not bit-identical across platforms (51 % vs 90 % from one seed). Only cache-loaded registers are reported.
11. **Keep the weights as derived** after a 12-point scan showed no trend (the embedding's local-minimum choice dominates).
12. **Pulse: 5800 ns,** $\boldsymbol{0.75\,\Omega_{\max}}$**, fitted** $\boldsymbol{\delta_f}$**, plain linear sweep** from the scan; slow region kept as a labelled option because its gain is within noise.
13. **Keep the large provable A, report the tight A as an ablation** after it dropped P(reach) from 0.94 to 0.03.
14. **Add the assignment-valid random control** (uniform random valid assignments through the same decoder) so the "useful quantum work" question of the brief has a fair answer; the uniform-bitstring sampler is the weaker control and is also reported.
15. **References via the brief's §9.3 route** (qubo-solver with a stated 60 s max solve time plus a constrained annealer, best known), with tier S also brute-forced and proven; CPLEX could not be installed.
16. **Enforce** $\boldsymbol{\delta_f < U_{\min}}$ after tier M returned zero decodable shots; migrate cached registers with unchanged geometry so before/after rows share the register.
17. **Use the weakest intra-polygon pair for the bound** after the team's review caught that a square's diagonal is 8× weaker; discard the mis-set first L run before reporting.
18. **Track A first, Track B as a standalone, honestly labelled slice** (Gate 2, Sat evening): the routing encodings and the end-to-end integration were built, but the one quantum tour obtained is reported as what it is.
19. **Overnight: 8 µm inter-polygon spacing and an amplitude scan on tier L**, because the diagnosis said the limit was energy scale and spurious couplings, not the weights.
20. **Sunday morning, a frozen two-hour protocol** for three follow-ups — exact comparison of three clustering objectives, a held-out capacity-weight study with the selection frozen before the validation set was looked at, and a bounded repair with matched random controls — so that none of them could be tuned to the answer. Their outcomes (depot-MST helps L and hurts M; $\kappa = 0.6$ buys feasibility and pays with optimality; repair helps random controls just as much) are reported as ablations, not as new headlines.

---

## 9. Every error we hit, and what we did about it

1. **No internet from the sandbox.** The cloud sandbox and the laptop's Linux shell could not reach PyPI or GitHub, so the emulators only run in the Windows virtual environment. A small job runner (`win_runner.bat`) was written that watches a queue folder; everything else was developed against a copy of pulser-core.
2. **No CPLEX.** The package could not be installed, so the §9.3 reference hierarchy uses qubo-solver's annealer plus our constrained annealer, with the exhaustive DP as the tier-S proof. `make references` picks CPLEX up automatically if it ever appears.
3. **A hand-typed number.** A baseline value (0.066) measured on a toy instance slipped into the docs in place of the real 0.080. A fresh-context reviewer caught it; from then on every number in a document is generated from a run record.
4. **Windows console encoding.** Printing "δ" crashed a job with `UnicodeEncodeError` under cp1252; every job now sets `PYTHONUTF8=1`.
5. **No single** $\boldsymbol{\lambda_2}$ **is sound on all toy seeds** (best 35/40). Where capacity binds, "closest" and "most balanced" conflict; a soft term cannot force the inequality. Accepted and documented; feasibility is reported, not assumed.
6. **The emulator timing calibration** (12–40 atoms × three bond dimensions) did not finish in 55 minutes and blocked the serial queue; killed and re-queued last with a smaller grid.
7. **`QutipConfig(noise_model=None)` crashes** inside pulser-simulation; passing nothing is the fix, found by the first Windows pytest run (111/112).
8. **qubo-solver's simulated annealing** returned the same non-optimal string on three tier-S instances regardless of iterations or pre/post-processing; its tabu search finds the optimum at once, so tabu is the honest classical baseline and the SA rows are reported as what they are. A stated-parameter SA path was added for the reference computation.
9. **The polygon optimiser is not bit-identical across platforms** ($\delta_f$ 32.8 vs 35.1 rad/µs from the same seed, P(reach) 51 % vs 90 %) — and not even across two overlapping runs on one machine. Registers are cached and committed; a row whose register differs from the cache is re-run, not reported. The metric difference between the two solutions is itself a finding: the fit objective does not fully predict P(reach).
10. **One-hot does not embed at** $\boldsymbol{K = 3}$ with the generic fit (residual 38 against 0.06–0.32 on S; defects below parents; optimum at rank ~10 000); opening the triangles to 6–10 µm lowers the energy scale without fixing the ordering. A limit of one-hot on a 2-D $1/r^6$ register, reported as such.
11. **The weight scan showed no trend** because every weight setting produces a different embedding local minimum. Weights left as derived.
12. **A tarball sync overwrote a parallel session's edits** (15:45). Replaced by git patches in both directions with checksum comparison after every exchange.
13. **The tight A loses the device margin** (0.94 → 0.03) despite being provably sound. Headline keeps the larger A.
14. **qubo-solver's uncapped emu-mps run on tier M** let the bond dimension grow past 300 ($\approx 50$ s per step) and was killed after 80 minutes; re-run with $\chi = 32$.
15. **Track B's generic embedding fails for both routing encodings** (error 0.79 and 0.98, optimum at device rank 114 of 1 024 / 603 of 65 536 on the pilot fixture, 129–146 / 429–668 on the real tier-M clusters); pulse samples decode to 0.1 % / 0 % valid tours, worse than a random valid tour. No routing advantage claimed; the next lever is a routing-specific register.
16. **A `|` in a `--tag` produced a file name Windows rejects**, so a complete 500-shot tier-M probe lost its record; the printed metrics are kept in the log and the row was re-measured under a safe label.
17. **Two overlapping optimiser runs on M_seed01 produced two different registers from one seed**, one with $\delta_f$ at the device maximum; the random control measured on the wrong register was re-measured on the cached one.
18. **The** $\boldsymbol{\delta_f}$ **bound was first written for triangles**; a square's diagonal pair is 8× weaker, so the first tier-L run ($\delta_f = 35.8$ vs a bound of 6.9 rad/µs) would have favoured two opposite corners per customer. Caught in the team's review of the 18:50 report and killed before reporting; the bound now uses the weakest pair of the actual polygon.
19. **The CPLEX cross-check on tier S could not be run**; the agreement check is brute force vs qubo-solver's SA vs our constrained SA (5/5 agree).
20. **Track B integration** produced exactly one raw valid quantum tour (20 % longer than exact) in five attempts; every other attempt fell back to exact classical routing, labelled with its reason.
21. **A stale PDF was reported as fresh**: an HTML comment above the YAML header made pandoc emit a `\usepackage` into the body, XeLaTeX failed, and the build script read the page count of the previous PDF. The script now deletes the old PDF, fails on a non-zero LaTeX exit, and refuses a body containing `\usepackage`.
22. **The depot-MST objective** improves all three L gaps (17.6/8.2/13.1 % → 0/0/7.5 %) but worsens all three M instances; reported as a promising L-specific alternative, not a replacement.
23. $\boldsymbol{\kappa = 0.6}$ raises held-out feasibility from 0.48 to 0.80 but lowers P(reach $C^*$) from 0.39 to 0.19 and hurts one instance badly; kept as an ablation.
24. **Capacity-local repair** makes almost every M or random sample feasible and often reaches the reference route — for the quantum samples *and* for the matched random controls alike. So repaired results establish nothing about the quantum part and stay separate from every raw metric.
25. **Tier L with the tier-S drive** gave $\le 0.8$ % feasible shots: the $K = 4$ bound compresses the landscape to an energy scale the 5800 ns sweep cannot follow adiabatically. Fixed overnight by the 8 µm spacing and the $\Omega = 0.15\,\Omega_{\max}$ drive (Section 5.4), which became the tier-L headline.

---

## 10. Small glossary

* **QUBO** — a cost function of binary variables with linear and pairwise terms; what the register minimises.
* **One-hot** — a string where each customer has exactly one bit set (one van).
* **Decodable / feasible** — a shot is decodable if it is one-hot; feasible if it is decodable *and* no van is overloaded.
* **Model gap** — how far the best feasible shot's clustering objective is above the exact optimum of our clustering model. **CVRP gap** — how far that clustering, routed exactly, is above the best known CVRP cost $C^*$. **Decomposition gap** — how far the model's own optimum, routed exactly, is above $C^*$.
* **P(reach)** — the fraction of shots whose clustering equals the optimum (model optimum or $C^*$), with vehicle labels ignored.
* **Fidelity** — relative error between the couplings the register realises and the ones the model asked for; we report the brief's full-matrix version and the one-hot-subspace version, plus whether the register's own ground state is the model optimum.
* **Defect gap** — how far a string with one bit flipped away from a valid one-hot string sits above it in device energy; must be positive for the sweep to end on valid strings.
* $\boldsymbol{\chi}$ **(chi)** — the bond dimension of the tensor-network emulator; a cap on how much entanglement it keeps. A row is converged only if the peak $\chi$ actually needed stays below the cap.
* **Repair** — classical post-processing that moves a sample to a feasible neighbour; always a separately labelled row, never the headline.
