# Gate 2 — what to run overnight (decision sheet, 16:00 ET)

Facts on the table (all in results.csv / DERIVATIONS.md):

* Tier S is solved as well as it can be: `onehot-ub` + polygon register + 5800 ns / 0.75 Ω_max gives
  P(reach model optimum) 0.89–0.99 on 5/5 instances (qutip), feasibility 0.95–1.00, 0 % model gap. Remaining
  loss is the decomposition gap on two instances (15.6 %, 21.7 %) — a property of the closeness objective, not
  of the device.
* Tier M (K = 3, 30 atoms): the one-hot rook's graph does not embed (DERIVATIONS Q-M1). The random sampler
  decodes 0/2000 shots; the generic and polygon registers both put the model optimum at rank ~10 000 of 59 049
  one-hot strings. A measured emu-mps row (job 070) is queued; expect feasibility ≈ 0.
* Emulator budget (docs/EMU_CALIBRATION.md, laptop CPU): emu-mps 30 atoms χ = 32 ≈ 215 s per 2000 ns at 100
  shots → a 5800 ns, 500-shot M run ≈ 10–15 min; L (60 atoms) untested, expect 30–60 min at χ = 16–32.

## (a) Run M/L as-is on emu-mps and report honestly
Cost: 3 M + 3 L runs ≈ 1.5–3 h of runner time; no new code. Result: rows with feasibility ≈ 0, gaps undefined,
plus the realised-register diagnostics that explain why. Brief-compatible (Pitfall 9.2: "if no shot is
feasible, the gap is undefined: report it as such") and it is the honest measurement of the encoding's limit.
Judging story: "we found where one-hot stops being layable and measured it."

## (b) Hierarchical K = 2 bisection for M/L (model change — needs your go)
Idea: every clustering decision becomes a 2-way split, which is the case the device handles (dimers, exact
one-hot identity). M (K = 3): level 1 splits the 10 customers into a 1-vehicle group (capacity Q) and a
2-vehicle group (capacity 2Q) — 20 atoms; level 2 splits the 2-vehicle group — ≤ 2·9 = 18 atoms. L (K = 4):
level 1: 15 customers into two 2-vehicle groups (capacity 2Q each) — 30 atoms; level 2: two problems of ≤ 8
customers — 16 atoms each. Every sub-problem is a `onehot-ub` instance with K = 2 and an adjusted capacity
(and an adjusted λ₂ scale), so the existing code, checker and embedder apply unchanged; only a driver and the
derivation of the hierarchical objective are new. What changes: the "model" is the hierarchy (its optimum is
computed by exact enumeration at each level, so the model gap stays defined), the decomposition gap grows
(bisection is a heuristic on top of cluster-first), and the atom count row in D2 becomes "20 / 30 at the
largest level" (still N·K = 30 / 60 atoms total used across levels — state both). Cost: ~1.5 h to derive, code,
checker, before/after; then the same emulation budget as (a) but with expected feasibility ≫ 0. Risk: level-1
groups with capacity 2Q make the load-balance term weaker relative to closeness (K = 2 guarantee still holds
for the split itself).

## (c) Track B on tier S
Clusters on S have ≤ 3 customers, so every cluster is trivial (brief §9.2): Track B on S "validates the
pipeline but says nothing about routing". Only worth it if combined with (b) to get non-trivial clusters on
M/L. Cost: two routing encodings + end-to-end driver ≈ 3 h; routing emulation on 4–6-customer clusters is
cheap (≤ 25 atoms with a position-based one-hot, fewer with a conflict-graph encoding).

## (d) DMM-assisted single-bit encoding for K = 2 levels (emulation only, labelled)
One atom per customer (x_i = which side of a bisection), objective couplings 2w_ij > 0, linear terms
−Σ_j w_ij are non-uniform → needs the DMM, allowed in emulation only and compared separately. Halves the atoms
of (b) and removes the clique problem entirely, but cannot run on the QPU as the brief describes it. Cost ≈ 1 h
on top of (b). Mention in the report as the road not taken unless you want the DMM rows.

Recommendation: (a) tonight regardless (it is cheap and honest), plus (b) if you accept the model change —
it is the only path to non-zero feasibility on M/L with this device.
