# Pulser 1.9.1 — web/source-verified notes (2026-10-24)

Verified against the `v1.9.1` tag of github.com/pasqal-io/Pulser and the installed package (`pulser/` in the venv). Docs: https://pulser.readthedocs.io/en/stable/ (stable = 1.9.1). Releases: https://github.com/pasqal-io/Pulser/releases.

## Install / packages
- `pip install pulser` → meta-package pinning `pulser-core==1.9.1` + `pulser-simulation==1.9.1` + `pasqal-cloud>=0.23.0` (pulser-pasqal is deprecated/replaced).
- `pip install pulser-core` only: no QuTiP. Extras: `pulser[torch]` (torch ~2.6), `pulser-core[fast-validation]`.
- Python ≥ 3.10 (CI 3.10–3.14). Deps: numpy (≥2 on py≥3.13), scipy<2, matplotlib≥3.10.5, jsonschema; simulation: qutip 5.
- https://pulser.readthedocs.io/en/stable/installation.html

## Signatures (v1.9.1)
```python
Register(qubits: Mapping[id, (x,y)])                      # NOT centred
Register.from_coordinates(coords, center=True, prefix=None, labels=None)   # centred
Register.rectangle(rows, columns, spacing=4.0) / square / rectangular_lattice / triangular_lattice / hexagon / max_connectivity(n, device)
reg.with_automatic_layout(device, layout_slug=None) -> Register
reg.define_detuning_map({qid: weight in [0,1]}) -> DetuningMap
Sequence(register, device)                                 # validates register (min distance, radial, atom count)
seq.declare_channel(name, channel_id, initial_target=None)
seq.add(pulse, channel, protocol="min-delay"|"no-delay"|"wait-for-all")
seq.delay(duration, channel); seq.measure(basis="ground-rydberg")
seq.config_detuning_map(det_map, dmm_id=None); seq.add_dmm_detuning(waveform, dmm_name, protocol="no-delay")
seq.config_slm_mask(qubits, dmm_id="dmm_0")
seq.get_duration(channel=None, include_fall_time=False); seq.to_abstract_repr(); Sequence.from_abstract_repr(s)
seq.with_new_register(reg) / with_new_device(dev, strict=True)
Pulse(amplitude_wf, detuning_wf, phase, post_phase_shift=0.0); Pulse.ConstantDetuning(amp_wf, det, phase); Pulse.ConstantAmplitude(amp, det_wf, phase); Pulse.ConstantPulse(T, amp, det, phase)
ConstantWaveform(T, v); RampWaveform(T, start, stop); BlackmanWaveform(T, area); InterpolatedWaveform(T, values, times=None (fractions of T))
CompositeWaveform(*wfs); CustomWaveform(samples); wf.with_new_duration(n)
QutipBackendV2(sequence, *, config=None, mimic_qpu=False).run() -> pulser.backend.Results
QutipConfig(*, sampling_rate=1.0, solver="default"|"MasterEquation"|"MonteCarlo", observables=(), default_evaluation_times=(1.0,), with_modulation=False, noise_model=None, n_trajectories=None, default_num_shots=1000)
BitStrings(evaluation_times=None, num_shots=None, one_state=None, tag_suffix=None)
results.final_bitstrings (Counter), results.final_state, results.get_result("bitstrings", 1.0), results.atom_order, results.total_duration
# legacy, DeprecationWarning: QutipEmulator.from_sequence(seq, sampling_rate=1.0, ...).run().sample_final_state(N)
```

## Device attributes
Fields: `name, dimensions, rydberg_level, min_atom_distance, max_atom_num, max_radial_distance, supports_slm_mask, min_layout_filling, max_layout_filling, optimal_layout_filling, min_layout_traps, max_layout_traps, max_sequence_duration, max_runs, requires_layout, reusable_channels, channel_ids, channel_objects, dmm_objects, noise_model, short_description, pre_calibrated_layouts, accepts_new_layouts`.
Properties/methods: `channels`, `dmm_channels`, `interaction_coeff` (C6/ħ rad/µs·µm⁶), `interaction_coeff_xy`, `rydberg_blockade_radius(Ω)`, `rabi_from_blockade(r)`, `validate_register`, `validate_layout`, `validate_layout_filling`, `specs`/`print_specs()`, `calibrated_register_layouts`, `to_virtual()`, `to_abstract_repr()`/`Device.from_abstract_repr`.
Channel fields: `addressing, max_abs_detuning, max_amp, min_retarget_interval, fixed_retarget_t, max_targets, clock_period, min_duration, max_duration, min_avg_amp, mod_bandwidth, custom_phase_jump_time, eom_config`; `rise_time`, `validate_pulse`, `modulate`.
DMM fields: `bottom_detuning, total_bottom_detuning, min_avg_abs_detuning, clock_period, min_duration, max_duration, mod_bandwidth`.
Devices in 1.9.1: `AnalogDevice, DigitalAnalogDevice, MockDevice, WeightedAnalogDevice` (+ `Device`, `VirtualDevice`). No Fresnel object: real QPU specs come from `PasqalCloudConnection.fetch_available_devices()`.

## Gotchas
1. Units: rad/µs, µm, int ns; `mod_bandwidth` in MHz; `evaluation_times` relative in [0,1] for `EmulationConfig` (µs for legacy QutipEmulator).
2. `validate_duration` rounds up to `clock_period` with a warning and raises below `min_duration`.
3. Amplitude ≥ 0 enforced; `WeightedAnalogDevice.min_avg_amp = 2π·0.3` rejects pulses whose average amplitude is too low.
4. δ<0 favours ground; interaction +C6/r⁶; detuning capped at `max_abs_detuning`.
5. DMM ≤ 0 and floors (`bottom`, `total_bottom`, `min_avg_abs`); local = w·δ_DMM; positive local detuning = positive global + negative DMM.
6. `Sequence()` checks radial distance from the **origin** → centre coordinates.
7. `requires_layout` is only enforced by `mimic_qpu=True` / QPU submission; use `with_automatic_layout`.
8. `max_sequence_duration` enforced at `add` time (RuntimeError).
9. Modulation bandwidth low-passes the output; `with_modulation=True` to simulate; 1.6.5 changed the transfer function (√2 slower).
10. Serialisation: `to_abstract_repr()` (parametrized sequences need `build()`/defaults).
11. Default shots 1000; results stochastic.
12. QutipBackendV2 practical ≈ 15 atoms.

## Breaking changes (12 months)
| version | change |
|---|---|
| 1.9.0 | `QutipEmulator.run()` deprecated → `QutipBackendV2`; `InterpolatedWaveform(interpolator="interp1d")` deprecated; `pulser.result.Result` deprecated; `pulser-pasqal` → `pasqal-cloud`; new `WeightedAnalogDevice` |
| 1.8.0 | `Device.default_noise_model` → `noise_model` |
| 1.7.0 | Python 3.9 dropped; `NoiseModel.runs` → `EmulationConfig.n_trajectories`; `BackendConfig` read-only (`with_changes`); `default_num_shots` |
| 1.6.5 | modulation transfer function corrected |
| 1.6.0 | `SimConfig` → `NoiseModel`; `QutipBackend` → `QutipBackendV2`; `switch_register/switch_device` → `with_new_register/with_new_device`; `change_duration` → `with_new_duration`; `EmuFreeBackend/EmuTNBackend` removed, `EmuMPSBackend→RemoteMPSBackend` etc. |
| ≤1.5 | `Chadoq2`, `IroiseMVP`, `Simulation` removed |

Links: https://github.com/pasqal-io/Pulser/tree/v1.9.1 (`pulser-core/pulser/devices/_devices.py`, `_device_datacls.py`, `channels/dmm.py`, `sequence/sequence.py`, `pulser-simulation/pulser_simulation/qutip_backend.py`), https://pulser.readthedocs.io/en/stable/apidoc/core.html, `tutorials/applications/QAA to solve a QUBO problem.ipynb`, `tutorials/advanced_features/Local addressability with DMM.ipynb`.
