# What did not work (one honest sentence each)

- The cloud sandbox and the Linux shell on the laptop cannot reach PyPI or GitHub (egress policy 403), so emulators run only in the Windows venv through `scripts/win_runner.bat`; everything else is developed against pulser-core copied from that venv.
- CPLEX is not installed in the hackathon venv and could not be installed in one try, so the §9.3 hierarchy is brute force (exact set-partition DP, proven optimum) cross-checked by simulated annealing, not CPLEX.
- A hand-typed baseline number (P(reach ref) after repair = 0.066, actually measured on a toy instance) slipped into DERIVATIONS.md/STATUS.md and was caught by the reviewer; the correct S_seed00 value is 0.080 and numbers now come only from run records.
- Runner job 002 (`device_limits.py --assert-brief`) crashed on Windows with `UnicodeEncodeError: 'charmap' codec` (cp1252 console, printing `δ`); re-queued as 007 with `set PYTHONUTF8=1`, which every later `.cmd` now sets — the running `win_runner.bat` was not edited because cmd reads batch files incrementally.
- No single load-balance weight `λ₂` makes `onehot-ub` sound on all 40 toy tier-S seeds (max 35/40 at κ=0.15): on seeds where capacity binds, "closest" and "most balanced" conflict and a soft term cannot force `L_k ≤ Q`; we rely on reporting the best *feasible* shot (brief Pitfall 9.1) and say so in MODEL.md §6.
