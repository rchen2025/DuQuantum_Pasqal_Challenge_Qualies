# AI use, per deliverable

Honest accounting as the brief asks (§"On the use of AI"). Everything below was reviewed by the team; the
derivations in `docs/DERIVATIONS.md` are the team's to defend at judging.

| Deliverable | What AI (Claude) did | What the team did |
|---|---|---|
| D1 README / requirements | wrote the README, pinned versions from the venv's `pip freeze` | chose the stack, built the venv |
| D2 model write-up | typed up the derivations from the whiteboard model, wrote `scripts/variable_count.py` | chose the formulation and every penalty weight rule (Gate 1) |
| D3 instances | wrote generator/validator/exact reference solver + tests | specified depot rule, feasibility method, seeds |
| D4 results.csv | wrote run/results pipeline and the metrics code against brief §9 | decided what is reported (feasible decodes only, labeled repair) |
| D5 report | drafted `docs/REPORT.md` from the derivations and results; built the PDF | edited, cut to 4 pages, owns every claim |
| Plumbing | Windows job runner, emulator timing calibration, Makefile | — |

Known AI failure modes we checked for (brief): negative couplings (refused in code), min-spacing violations
(Pulser validation on every sequence), over-long pulses (Pulser validation), hallucinated citations (none cited
without a link in `docs/research/`).
