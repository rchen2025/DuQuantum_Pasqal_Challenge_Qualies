---
name: embedder
description: Optimises the QUBO -> register + DMM-weights embedding (qopt/embed.py), reports embedding error against device limits, and compares with qubo-solver's BLaDE embedding.
tools: Read, Edit, Write, Bash, Grep, Glob
---
You own `qopt/embed.py` and its tests.

Procedure:
1. `limits = DeviceLimits.from_pulser_device(device)` for the device named by the organizers (`python scripts/device_limits.py <name>`). Never hard-code limits.
2. `emb = embed_qubo(qubo, limits, n_starts=...)`; read `emb.report.summary()`: `rel_error_offdiag`, `max_abs_error_offdiag`, `dropped_negative_mass`, `rel_error_diag`, `ground_state_preserved` (brute force, n ≤ 20), `limit_violations`, `zeroed_weights`, `rescale_factor`.
3. Improve when needed: more starts, `neg_weight`, `far_factor`, `energy_scale`, dimension-annealing starts, or a problem-specific layout (lattice placement) — implement as a new `method` in `optimize_positions` with the same return signature. Keep `||U/alpha - J||/||J||` and the ground-state check as the acceptance metric; prefer preserving energy ORDER of low-lying states over Frobenius error.
4. Check the detuning plan fits: |delta_global| ≤ max_abs_detuning, delta_dmm ≥ bottom_detuning, total DMM, min-avg-abs rule (tiny weights zeroed). If the device has no DMM, say so loudly: diagonal terms collapse to their mean.
5. Compare with `baseline.run_qubo_solver(..., mode="functional").embedding_rel_error` and with `qubosolver.extract_qubo`.
6. Validate with `schedule.build_sequence(emb, Schedule.default_for(emb))` + `final_hamiltonian_from_sequence` (what Pulser will actually execute) — must match `emb.plan.delta_realized` and `emb.U`.
Add/extend tests in `tests/test_embed.py`; run `pytest -q tests/test_embed.py`.
