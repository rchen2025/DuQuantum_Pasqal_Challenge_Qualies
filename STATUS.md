# STATUS — Sat 2026-10-24, 18:45 ET (hackathon day; freeze Sun 11:00 ET)

**18:45 ET.** The tier-M fix is measured and confirmed (DERIVATIONS §6.7 table): with the final detuning bounded
below the triangle coupling, M_seed00 goes from 0 % decodable / 0 % feasible to 99 % / 4.4 %, M_seed02 to
100 % / 45 %, and M_seed01 (whose fit was already inside the bound) sits at 98 % / 39 %; best feasible shots
are still 23–68 % above the model optimum because the one-hot ordering is set by distance couplings the plane
cannot realise. qubo-solver's own pipeline on M_seed00/01 is in the same regime (1.6 % / 37 % feasible, 35 % /
11 % CVRP gap). The two unbounded rows are kept under `|pre-6.7` as the before. results.csv: 71 rows. Running:
tier L with the bound (60 atoms, emu-mps χ = 32, the slow runs), then M_seed00 at χ = 64 for convergence;
lane b: qubo-solver quantum on M_seed02. Report refresh 3 requested (table now compact for M/L: ours +
qubo-solver rows only; controls and pre-6.7 rows stay in results.csv). A `|` in a `--tag` cost one lost record
on Windows (fixed, DID_NOT_WORK).


**18:10 ET.** References switched (see 17:40): all 11 instances `best_known` via qubo-solver (60 s) + constrained SA,
tier-S brute-force check agrees 5/5, values unchanged, results.csv re-scored (66 rows, only the status column
moved). Report refresh 2 is in (4 pages; C* labelled best known; L finding: qubo-solver's SA on the penalized
QUBO is 23–34 % above C* within 60 s). New tier-M rows: **qubo-solver quantum on M_seed00 (its BLaDE register,
AnalogDevice, emu-mps χ=32): 99 % of shots one-hot, 1.6 % capacity-feasible, best feasible 72 % model gap /
35 % CVRP gap** — versus our polygon register: 0 % decodable. Diagnosis: our one-hot-subspace fit of δ_f is
broken at M (residual 1.55) and returned the device maximum 125.66 rad/µs, 2.3× the clique coupling
U = 55.1 rad/µs, so two excitations per triangle are favoured and no shot is one-hot. Dry-run with
δ_f = 0.65 U = 35.8 rad/µs: single-flip defect gap turns positive (+19.4) while the optimum's rank among
one-hot strings stays 10 474 (that part is the distance couplings). Before/after row queued (lane b job 240,
`|dfscale0.285`); if it decodes, embed_onehot gets a hard bound δ_f ∈ (0, U_clique) with a DERIVATIONS note.
Main lane resumed (100 = M_seed01/02 as-is running, then 110 L, 120 M χ=64). Track B slice 1 is in
`routing/` + `docs/ROUTING.md` (standalone, negative result, not in the report).


**17:40 ET.** Reference values are being switched to the brief's §9.3 route at Richard's request: C* now comes
from qubo-solver's classical solver on the `onehot-ub` QUBO (CPLEX if importable, else its SA with stated
parameters) under a **60 s max solve time**, plus our constrained SA with the same budget — best known of the
two, `reference_status = best_known`, every setting in the instance JSON (`scripts/reference_qubosolver.py`,
lane b job 200, running). Tier S keeps the brute-force agreement check the brief demands; M/L use no
exhaustive search. Afterwards `scripts/rescore_records.py` re-scores every record against the new blocks and
results.csv is regenerated (if the best-known values equal the earlier DP optima — expected — no gap or
P(reach) changes, only the status column). Also landed: tier-M rows in results.csv (64 rows): M_seed00 as-is
on emu-mps χ=32 — **0/500 feasible, gaps undefined**, model optimum at realised rank 10 474; tabu on
M_seed01/02 reaches feasible clusterings 17.6 % / 2.8 % above C*; random sampler 0/2000 feasible. Job 070
(qubo-solver quantum on M, uncapped χ > 300, ~50 s/step) was killed after 80 min — a χ-capped version will be
re-queued after the as-is M/L rows (100/110/120, now running on the main lane). Report editor pass 1 is in
(`docs/REPORT.md` restructured, 4 pages). Track B: a standalone slice (two routing encodings, exhaustive
n ≤ 4 checks, 10-atom pilot) is being built in `routing/` by a subagent that owns only that package; it enters
the report only if Richard says so at Gate 2 — Track A stays first.


**17:12 ET.** Two runner lanes now (`win_runner.bat` heavy, `win_runner_lane.bat` light). Job 080 landed on qutip:
`onehot-ub-tight` ablation P(reach) 0.03 / 0.52 / 0.94 on S_seed00/01/04 (sandbox SV: 0.03 / 0.56 / 0.96) and
`random-onehot` 0.11 / 0.11 / 0.12 (500 shots; 2000-shot rows 0.12) — both in results.csv (59 rows) and the report.
Tier-M as-is row (job 070 part 1, χ = 32): feasibility 0/500. qubo-solver's uncapped emu-mps on M is still
running (χ ≈ 300, 37 s/step); overnight queue holds M_seed01/02, L_seed00–02 (χ = 32), an M χ = 64 convergence
run, M tabu/random baselines, full pytest and the README demo on Windows.

**16:40 ET addendum (Claude, chat session) — how to read the six new rows in results.csv (a56980e).**
(1) *Useful quantum work (brief §9.2):* the assignment-valid random control (`random-onehot`) reaches the model
optimum with P ≈ 0.12 on S_seed00/01/04; the headline `onehot-ub[emb=polygons]+qutip` rows reach it with
P ≈ 0.94 / 0.95 / 0.89 — a 7–8× factor. Put this in REPORT.md and as a judge Q&A. (2) *`onehot-ub-tight` is
an ablation, not a replacement:* the tight `A = R_w/K` keeps the optimum as the ground state (proved, checked)
but shrinks the single-flip defect gaps the sweep must resolve; P(reach optimum) goes 0.94→0.03 (S_seed00),
0.95→0.56 (S_seed01), 0.89→0.96 (S_seed04) with the pulse tuned for the original `A`. Headline stays
`onehot-ub`; report `tight` as the experiment showing the exactly-one margin is physically load-bearing; one
sentence in DID_NOT_WORK.md. Do not re-tune the pulse for `tight` unless everything else is finished.

**What changed and why.** Tier S is done end to end with final settings and every deliverable now exists:
`make deliverables` prints **ALL PASS** (D1 README + pinned requirements, D2 write-up with the generated table,
D3 11 validated instances with proven references, D4 `results.csv` with 47 rows, D5 `report/report.pdf`
3 pages). Headline rows (pulser-simulation/qutip, structure-aware registers, 5800 ns, 0.75·Ω_max, 500 shots):
P(reach model optimum) = **0.94 / 0.95 / 0.99 / 0.96 / 0.89** on S_seed00–04 with feasibility 0.95–1.00 and
0 % model gap everywhere; the random sampler sits at 0.007–0.011 and qubo-solver's own quantum pipeline at
0.18–0.79 (and a 27 % model gap on S_seed04). Where P(reach C*) is low (S_seed03/04) it is the decomposition
gap (15.6 % / 21.7 %: the closeness objective's optimum is not the CVRP optimum), reported as such. The pulse
defaults in code are now the scan result (DERIVATIONS §6.6); the weights are unchanged (§4.1); the δ_f scan peaks
at the fitted value; the slow-region scan is finishing. Rows from before each change keep their own labels
(`[emb=generic]`, `|pulse=4000ns_Omega1.0(pre-scan)`), so every before/after pair is in results.csv.
`cvrp/report.py` (figures) is being written. Everything is committed and synced to the Windows folder.

**Open for Gate 2 (your decision, see the tier-M paragraph in the previous entry):** the one-hot encoding does
not embed for K = 3; the overnight run can (a) measure M/L anyway on emu-mps and report honestly, (b) switch M/L
to a K = 2 bisection (model change), or (c) spend the night on Track B for tier S.

**Previous entry (15:50 ET).** M2 is through its three steps on tier S. (1) Embedding: the structure-aware
polygon registers are now cached and committed (`embeddings/`), so Windows (qutip) and the sandbox
(numpy-sv) sample the *same* register and agree to sampling noise; P(reach model optimum) with the baseline
pulse is 0.90 / 0.84 / 0.65 on S_seed00/01/04 (generic embedder: 0.12 / 0.07 / 0.03; random: 0.0075).
(2) Weights: a 12-point scan of multipliers on A and λ₂ shows no trend — the embedding optimiser's choice of
local minimum moves P(reach) more than a factor 2 on either weight — so the weights stay exactly as MODEL.md
states them (DERIVATIONS §4.1). (3) Pulse: duration and Ω scale have clean monotone trends; **T = 5800 ns,
Ω = 0.75·Ω_max** gives P(reach) 0.96 / 0.95 / 0.89 and feasibility 0.98 (DERIVATIONS §6.6); δ_f and
slow-region scans are running now. A fresh-context reviewer found no blockers (details below); its main
catch — qubo-solver's classical SA is a single-start greedy descent at T₀ = 5 — is fixed with a
stated-parameter SA path, and `qubosolver-tabu` (finds the optimum on 3/3) is the honest classical row.
qubo-solver's quantum pipeline (BLaDE + local_energy_scale, 5940 ns) reaches 0.78 / 0.78 / 0.00 on the same
instances — a strong baseline that ours now beats on all three seeds.

**Bad news for M/L, to decide at Gate 2:** the one-hot rook's graph does not embed for K = 3 (tier M, 30
atoms). With 5 µm triangles the one-hot-subspace fit residual is 38 (S: 0.06–0.32), the single-flip
defects sit *below* their parents and the model optimum ranks 10 000th among one-hot strings; opening the
polygons to 6–10 µm (lower energy scale) does not help (residual 0.86–0.95, rank ≥ 1279, register radius at
the 38 µm limit). Cause: K copies 5 µm apart cannot be far from a neighbouring customer's K copies, so the
cross-vehicle couplings are of the order of the wanted ones and strongly anisotropic — exactly the brief's §4
warning, and the A rule pins the clique/objective coupling ratio to ≈ 1.5 in distance. Options for the
overnight run: (a) run M on emu-mps anyway and report the measured (poor) feasibility and gaps honestly with
the realised-register diagnostics; (b) a K = 2 decomposition for M/L (hierarchical bisection) — a model
change, so your call; (c) Track B on tier S only. Timing (`docs/EMU_CALIBRATION.md`): emu-sv 20 atoms 167 s,
emu-mps 30 atoms χ = 32 ≈ 215 s per 2000 ns, 100 shots — a 5800 ns, 500-shot M run is ≈ 10–15 min.

**Previous entry (15:00 ET).** M1 is in (3 tier-S seeds, `onehot-ub` on qutip / emu-sv / our exact state-vector
emulator, plus the random sampler; all three emulators agree within sampling noise; Windows pytest is fully
green) and M2 step 1 — embedding fidelity — produced the biggest jump of the day. The generic embedder cannot
realise the one-hot rook's graph (cross-vehicle couplings come out as strong as the wanted ones, rel. Frobenius
0.95, and on S_seed04 the register's own ground state was infeasible). The fix is to ask for less: on one-hot
strings each customer pair contributes exactly one coupling, so only *same-vehicle minus cross-vehicle* has to
match α·w_ij; cross couplings may be anything constant. `cvrp/embed_onehot.py` builds each customer as a rigid
5 µm K-gon and fits the device energies of all one-hot strings to the model energies (1 − R²), with explicit
"single-flip defect" gaps and a fitted δ_final (DERIVATIONS §5.7). Result, baseline pulse, 500 shots:
P(reach model optimum) 11.6 / 7.8 / 2.6 % → **51 / 61 / 68 %**, feasibility 20 / 9 / 6 % → **99.6 / 63 / 98 %**,
realised argmin = model optimum on 3/3 seeds (random sampler: 0.75 %, 2–4 %). qutip confirmation of the
"after" rows and the qubo-solver baselines (BLaDE + local_energy_scale, same scorer) are queued on the runner
(`cvrp/baseline.py` is written, API verified against the installed source). Next: reviewer pass on M1/M2-1,
then weights via the brute-force checker, then the pulse scan (Ω scale / δ_final / slow region) — all on the
in-house emulator with qutip confirmation of the final rows.

**Previous entry (14:45 ET, M1 first rows).** Gate 1 is implemented exactly as `docs/MODEL.md` says (`onehot-ub`: one-hot, no slack,
capacity → load-balance via the one-hot identity; DERIVATIONS §4 ↔ code ↔ regenerated D2 table agree, 8/30/60
atoms). M1 has its first quantum rows: an in-house exact state-vector emulator (`cvrp/svsim.py`, backend
`numpy-sv`, built from Pulser's own per-ns samples) lets tier S run here while the Windows runner re-runs the
same sequences on pulser-simulation (qutip) for confirmation — the qutip adapter had a `noise_model=None` bug
(caught by the Windows pytest run: 111/112 pass) that is now fixed and re-queued. With the **baseline pulse and
the generic embedder**, onehot-ub on S_seed00 reaches the CVRP reference in 11.6 % of raw shots (random
sampler: 0.75 %), feasibility 19.8 % (random 4.4 %); S_seed01 7.8 % vs 0.75 %; S_seed04 only 2.6 % reach the
*model* optimum and the realised register's ground state is an **infeasible** string — the embedding is the
bottleneck: relative Frobenius error 0.94–0.95 on all three seeds, exactly the rook's-graph problem MODEL.md §7
predicted. That is M2 step 1. `results.csv` (D4) now exists and is regenerated by `python -m cvrp.results`.
S_seed04 also shows a 21.7 % decomposition gap (the closeness objective's optimum is not the CVRP-optimal
clustering) — reported, not hidden. qubo-solver baseline (`cvrp/baseline.py`) is being written; reviewer pass
follows M1.

**Previous entry (14:40 ET, Gate 1 on paper).** Gate 1 passed on paper: the real formulation `onehot-ub` is in `docs/MODEL.md`
(one-hot, no slack, capacity by unbalanced penalization → collapses to a load-balance term via the one-hot
identity; all couplings positive, all linear terms uniform, N·K atoms = 8/30/60). DERIVATIONS §4, the
formulation module, 6 new tests (all pass), the regenerated D2 table and the brute-force checker on
S_seed00/01/04 (SOUND) + binding toy seed 14 (NOT SOUND, documented) are done. **It was AI-drafted at the
team's request under time pressure — `AI_USE.md` says so; re-derive MODEL.md before judging.** Runner jobs
006 (onehot-ub on 3 S seeds, random 2000 + qutip 500 shots → `out/gate1/`) and 007 (assert-brief with
UTF-8, job 002 had crashed on cp1252) are queued behind the calibration job. Next: M1 rows from 006,
`cvrp.baseline` (qubo-solver) on the same seeds, reviewer pass, then M2 (embedding fidelity → pulse scan).

**Previous entry (13:00 ET).** Items 0–6 of the build are done and committed: the whole Track-A pipeline runs end
to end on a real tier-S instance with a throwaway textbook formulation (`python -m cvrp.run ... --backend
random` works everywhere; `--backend qutip` builds and validates the Pulser sequence here and will sample
once the Windows job runner is up). References are *proven optima* for all 11 instances (exhaustive
set-partition DP × exact TSP, cross-checked by SA and, on S, by naive enumeration). The derivations document
(`docs/DERIVATIONS.md`) has §0–3, §5–6 and 13 judge Q&As. A fresh-context reviewer checked physics signs,
textbook algebra, metrics and the exact solver (findings below).

**16:30 ET addendum (Claude, chat session) — second application; the 15:45 edits were lost to a tree reset.**
Reviewer handoff on MODEL.md applied where not already covered: (P1) tight exactly-one rule
`A = (R_w/K)(1+1%)` with proof, registered as **`onehot-ub-tight`** so existing `onehot-ub` rows stay
untouched (before/after ablation); checker SOUND on S, `A` fails at one halving as a tight bound should;
`test_onehot_ub.py` 8/8. (P4) assignment-valid random control: `--backend random-onehot` (uniform random
assignments encoded one-hot). (P2) already covered by `embed_onehot.py` + `realized_register`. MODEL.md
corrected (K=4 clique not planar-exact; `J_max = max(2A, max w_ij)`; cross-coupling claim replaced by the
measured §5.7 numbers; K=2 feasibility scoped to the pure balance term; `R/(2Q)` demoted to a
parameterisation; "post-selection retains, does not recover"; feasibility is joint; DMM wording).
Patch of exactly these changes: `patches/2026-10-24_reviewer_p1_p4.patch` (`git apply` if the tree is
reset again). Suggested rows: `onehot-ub-tight` on S_seed00/01/04, `emb=polygons`, qutip 500; `random-onehot`
on the same seeds. **Runner appears dead since 15:42 (queue holds `070_tierM_probe.cmd`): relaunch
`scripts\win_runner.bat`.**

## Build order — state
| # | item | state |
|---|---|---|
| 0 | env | runner bat shipped (**not yet launched — needs your double-click**); `device_limits.py --assert-brief` ✔; `requirements.txt` pinned (pytest/pulp/dotenv pins to be confirmed from Windows `pip freeze`); emu-mps calibration script written, queued |
| 1 | instances + validator | ✔ 11 instances (S×5, M×3, L×3), certificate-first, ρ 0.60–0.95, `validator` 11/11 |
| 2 | reference | ✔ proven optimum all tiers (L: 1.3 s), SA 10 restarts agrees 11/11, naive brute force agrees on S; CPLEX not available |
| 3 | model + textbook + variable_count | ✔ checker: feasible fraction, argmin→feasible/optimal, energy gap, weight half/double/zero scan, landscape correlation; D2 table 8/30/60, 14/45/80 (L = cap), **ours 8/30/60** |
| 3b | **Gate 1 formulation `onehot-ub`** | ✔ `docs/MODEL.md` + DERIVATIONS §4 + `cvrp/formulations/onehot_ub.py` + `tests/test_onehot_ub.py` (6 pass); SOUND on 5/5 shipped S, 35/40 toy S; runner job 006 queued for M1 rows |
| 4 | embed | ✔ refuses J<0, 5/38/80 enforced, α capped by \|δ\|≤125.66, fidelity = rel. Frobenius off-diag + worst pair, uniform-h check |
| 5 | pulse | ✔ 3-point Ω ramp + linear δ sweep + slow-region knob; every sequence built on AnalogDevice, `to_abstract_repr()`, δ(T) read-back exact |
| 6 | run | ✔ CLI → run-record JSON; metrics on feasible decodes only; labelled `repair=energy-descent`; backends qutip/emu-sv/emu-mps/random/counts |
| — | Makefile `demo`/`deliverables`/`report`, README (D1), `check_deliverables.py` | ✔ (checklist currently FAILs D2-ours, D4, D5 as expected pre-gate) |
| 7–10 | results.py (D4), baseline.py (qubo-solver), report.py figures, report.pdf | unblocked: run on `onehot-ub` once job 006 lands |

Tests: 80 pass (`scripts/run_tests.py`, pytest stand-in; real pytest once Windows runner installs it).

## Current best rows (textbook = throwaway, random sampler only — NOT a result)
S_seed00, textbook (14 atoms), random 2000 shots: feasible 4.5 % raw, best model gap 0 %, CVRP gap 0 %,
P(reach ref) 0.0065 raw (exact: 0.78 %) / 0.080 with energy-descent repair (feasible 31.7 % after repair). Textbook linear terms are non-uniform → flagged
`device_faithful=False`; rel. Frobenius embedding error 0.55. This is the bar any quantum run must beat.

## Needs you
1. Runner is alive (13:36); jobs 004–007 queued. Check `.runner/out/006_*.log` for the first onehot-ub emulator rows.
2. **Re-derive `docs/MODEL.md` §3b (the one-hot identity) and §6 (the `A` rule) by hand** — judges will ask exactly those (MODEL.md Q1–Q3).
3. `git add -A && git commit` (the Linux shell cannot commit: deletion is disabled, git leaves lock files).
4. Facts from the instances that shaped the design (kept for the report):
   - optima on all L instances use only 3 of 4 vehicles → the model must allow empty vehicles (brief §2 assumption 6);
   - with the pairwise-distance objective, the capacity constraint is non-binding on most tier-S seeds (34/40 toy
     seeds) — pick M1 seeds where it binds (S_seed01 ρ=0.86, S_seed00 ρ=0.71) and check the checker's `B` margin.

## Decisions taken (all in CLAUDE.md / DERIVATIONS.md)
Depot (50,50); certificate-first instances; references proven by enumeration; gaps only on feasible decodes;
"best shot" = lowest model objective, routing-post-selected best reported separately; never mean cost;
clusterings compared as sets.

## Open judge questions — see DERIVATIONS.md §7 (13 answered)
§4 answered in MODEL.md: linear terms uniform by the one-hot identity (the dropped diagonal is constant on one-hot strings); `A` by a repair-step argument (survives one halving, fails at ×1/4); `λ₂ = 0.15·R/(2Q)` by a 40-seed scan, with the binding seeds it does not rescue named.

## Reviewer pass on items 0–6 (fresh context, 13:10)
No blockers. Physics signs, Eq.(1)→QUBO mapping, textbook algebra (max err 3e-11 on 2000 random bitstrings), device-limit reading, exact DP (verified against brute force on 40 random instances) all confirmed. Fixed: one stale typed number (0.066 → 0.080), model-optimum tie handling in metrics (P(reach) by value), decomposition cost minimised over all optima, strict-radius rescale margin in embed, three wording slips in DERIVATIONS (§2.3 mean-h is our choice; §3.4 "2 of 8 clusterings"; §6.1 adiabatic criterion in Landau–Zener form).

## Reviewer pass on M1 / M2-step-1 (fresh context, 15:20)
No blockers. Confirmed: gaps only on feasible decodes; fidelity formula reproduced from records to 6 decimals; D2 table consistent; §4 ↔ `onehot_ub.py` line by line; `svsim` Hamiltonian/bit order match pulser-simulation source, qutip vs numpy-sv histograms within sampling noise (TV 0.05–0.07 on the cached registers, exact probabilities reproduce qutip counts to TV 0.03); one-hot identity holds (R² 0.81/0.53/0.92, all single-flip defects above parents, all pairs ≥ 5 µm, no reference data used in the fit). Fixed: stale §5.7/STATUS numbers (now generated by `scripts/derivation_numbers.py` from records), results.csv regenerated; qubo-solver SA rows exposed as a T₀=5 single-start greedy descent → new stated-parameter SA path (`qubosolver-sa-stated`, T₀ = max weight, 10 restarts) + tabu rows queued; embedder label on baseline rows; MODEL_WRITEUP prose (C* is the full CVRP optimum, not the routed model optimum); §5.7 wording (k-independence of same-vehicle couplings now measured as `subspace_same_vehicle_asymmetry`; "normalised residual", not R²); δ_final-scan knob now feeds the realised-register check and results.csv.
