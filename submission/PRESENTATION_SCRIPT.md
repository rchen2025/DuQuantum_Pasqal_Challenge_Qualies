# Four-minute presentation script — Team Qualies, Pasqal challenge

*$\approx 620$ spoken words: 4:00 at a natural pace ($\approx 155$ words/min). Slide cues in brackets; timings are cumulative. The talk follows the improvement ladder: what we built, what broke, what we changed, and what each change did. Every number is from `results.csv` or a run record.*

---

**[0:00 — Slide 1: title. "8 / 30 / 60 atoms. Zero slack."]**

Every extra binary variable costs an atom. So our goal was to put the clustering half of vehicle routing onto Pasqal's analog device with as few atoms as possible — and then make the register actually sample it. We got to one atom per customer–vehicle pair, no slack atoms at all: 8, 30 and 60 atoms on the three tiers, where the textbook encoding needs 14, 45 and 80. This talk is the sequence of improvements that got us there.

**[0:25 — Slide 2: Eq. (1) → $E_{\mathrm{dev}}(x)$]**

First, the constraint the device imposes. When the laser switches off, the register is a QUBO: one global detuning gives every atom the same linear term, and the atom positions give every coupling, all positive. Any model we run has to look like that.

**[0:45 — Slide 3: the model — improvement 1]**

Our variables are one-hot: customer i rides on vehicle k. Closeness is total pairwise distance inside each cluster; "exactly one vehicle" is a squared penalty. For capacity — the inequality — we used unbalanced penalization, which needs no slack bits. Then the key step: on any valid string, each customer has exactly one bit set, so all the uneven linear terms add up to a constant and drop out. What's left is a load-balance coupling. Every coupling positive, every linear term identical, one global detuning. That is improvement one: fourteen, forty-five, eighty atoms down to eight, thirty, sixty.

**[1:25 — Slide 4: embedding — improvement 2]**

Then we tried to place the atoms, and the register fought back. A generic embedding had a coupling error of 0.95, and the optimal clustering came out in only 3 to 13 percent of shots. The reason: a customer's atoms sit five microns apart, so every other atom is equidistant from them, and couplings that should be zero aren't.

Our fix was to ask the register for less. On valid strings only the *difference* between same-vehicle and cross-vehicle couplings matters. So each customer became a rigid five-micron polygon, and we fitted only the polygon positions. Optimal shots went from 3–13 percent to 62–90 percent.

**[2:00 — Slide 5: pulse scan and the ladder figure — improvement 3]**

Next, the pulse. We scanned duration and amplitude and scored each by how often it hits the optimum, not by mean cost. A longer sweep at three-quarters power won. That took tier S to **89 to 99 percent optimal shots on all five instances**, feasibility 95 to 100 percent.

**[2:25 — Slide 6: tier M — improvement 4]**

On tier M, every single shot was invalid. We traced it to physics: one excitation per triangle only beats two if the final detuning stays below the triangle's coupling — and our fit had overshot by more than double. We derived that bound and made it a rule in the code. Decodable shots went from zero to 99 percent.

**[2:50 — Slide 7: tier L — improvements 5 to 7]**

Tier L uses squares, and a square's diagonal is eight times weaker than its side, so the bound gets eight times tighter and the whole energy landscape shrinks to a few rad per microsecond. We caught that before reporting a single wrong run. Then we spaced the polygons to eight microns and turned the laser down to match that scale. Feasible shots on sixty atoms went from under one percent to **14 to 23 percent**, with best routes 36 to 47 percent above the reference.

**[3:25 — Slide 8: Track B and close]**

For Track B we built two routing encodings — a depot-fixed position model with n-squared atoms and an edge model with n(n+1)/2 — verified both exhaustively, and closed the loop end to end with an exact split of the loss. The next step there is the same one that worked for Track A: a structure-aware register.

So: a slack-free model, an embedder that asks for only what the model needs, two physical bounds turned into code, and every improvement measured before and after. Thank you.

**[4:00 — end]**

---

### If asked in Q&A (30-second answers)

* **"Where did $\boldsymbol{\lambda_1}$ go?"** It multiplies $\sum_k (Q - L_k) = KQ - D$ (with $D = \sum_i d_i$), a constant on every one-hot string. It only moves strings that already violate exactly-one, which $A$ handles. The device sees a load-balance term, not an inequality — by construction.
* **"What guarantees $\boldsymbol{L_k \le Q}$?"** Nothing, by design; we score feasible decodes only and report feasibility. For $K = 2$ the pure balance term is feasible whenever a split exists; adding distance removes that guarantee; $K \ge 3$ has none.
* **"Why is $\boldsymbol{A}$ so large?"** The tight provable bound $A = R_w/K$ is sound but leaves no device margin: P(reach) falls from 0.94 to 0.03 with the same register and pulse. $A$ sets the single-flip defect gaps in device units.
* **"How does it compare with classical or random?"** For context, as the brief asks: random valid assignments hit the tier-S optimum 12.5 % of the time and random bitstrings about 1 %; on tier L, tabu search on the same QUBO lands at 41–47 %. The full tables are in the README.
* **"Are your M/L emulations converged?"** M_seed00 peaks at $\chi = 12$ under caps of 32 and 64, M_seed02 at 7, M_seed01 at 14 under $\chi = 64$; the L rows run at $\chi = 32$ with peak $\chi$ recorded on every record.
