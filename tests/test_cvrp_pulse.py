"""Tests for cvrp.pulse (plain functions, deterministic).

Run:  PYTHONPATH=/home/claude/_pydeps:. python3 scripts/run_tests.py cvrp_pulse
"""
import numpy as np

from pulser.devices import AnalogDevice

from cvrp.embed import demo_embedding
from cvrp.pulse import (
    PulseParams, build_sequence, default_delta_init, default_for, final_detuning_from_sequence,
    final_hamiltonian_from_sequence, knots, snap_duration, sweep_slope, validate, waveforms,
)

CH = AnalogDevice.channels["rydberg_global"]
EMB = demo_embedding(seed=0, n_starts=4)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
def test_default_params_follow_embedding_and_device():
    p = default_for(EMB)
    assert p.delta_final == EMB.delta_final
    assert p.omega_device_max == CH.max_amp and abs(p.omega - CH.max_amp) < 1e-12
    assert p.duration_ns == 4000 and p.duration_ns % CH.clock_period == 0
    assert p.duration_ns <= AnalogDevice.max_sequence_duration - 200
    # delta_init = -min(max(|delta_final|, 2 Omega_max), max_abs_detuning)
    expect = -min(max(abs(EMB.delta_final), 2 * CH.max_amp), CH.max_abs_detuning)
    assert p.delta_init == expect and p.delta_init < 0
    assert default_delta_init(1.0, AnalogDevice) == -2 * CH.max_amp
    assert default_delta_init(-1e6, AnalogDevice) == -CH.max_abs_detuning


def test_duration_is_snapped_and_capped():
    assert snap_duration(4001, AnalogDevice) == 4000
    assert snap_duration(4002, AnalogDevice) % CH.clock_period == 0
    assert snap_duration(9000, AnalogDevice) == AnalogDevice.max_sequence_duration - 200
    assert snap_duration(1, AnalogDevice) >= CH.min_duration
    p = default_for(EMB, duration_ns=3001)
    assert p.duration_ns == 3000
    p = default_for(EMB, duration_ns=10_000)
    assert p.duration_ns == 5800


def test_bad_fractions_rejected():
    _raises(ValueError, default_for, EMB, rise_frac=0.6, fall_frac=0.6)
    _raises(ValueError, default_for, EMB, sweep_start_frac=0.9, sweep_end_frac=0.2)
    _raises(ValueError, default_for, EMB, slow_region=(0.5, 0.2, 0.5))


# --------------------------------------------------------------------------- #
# waveforms
# --------------------------------------------------------------------------- #
def test_waveform_endpoints_and_bounds():
    p = default_for(EMB)
    w = waveforms(p)
    assert len(w["t"]) == p.duration_ns
    assert w["amp"][0] == 0.0 and abs(w["amp"][-1]) < 1e-12
    assert (w["amp"] >= 0).all() and w["amp"].max() <= p.omega * (1 + 1e-12)
    assert abs(w["det"][0] - p.delta_init) < 1e-12 and abs(w["det"][-1] - p.delta_final) < 1e-12
    assert np.all(np.diff(w["det"]) >= -1e-9)                      # monotone sweep
    assert w["det"].min() >= p.delta_init - 1e-9 and w["det"].max() <= p.delta_final + 1e-9
    # plateau: amplitude at Omega between rise and fall
    T = p.duration_ns
    mid = w["amp"][int(0.3 * T):int(0.7 * T)]
    assert np.allclose(mid, p.omega, rtol=1e-6)
    K = knots(p)
    assert len(K["amp"][0]) == 4 and len(K["det"][0]) == 4


def test_slow_region_reduces_slope_in_window_and_stays_monotone():
    k = 3.0
    base = default_for(EMB)
    p = default_for(EMB, slow_region=(0.6, 0.3, k))
    w = waveforms(p)
    d = w["det"]
    assert abs(d[0] - p.delta_init) < 1e-12 and abs(d[-1] - p.delta_final) < 1e-12
    assert np.all(np.diff(d) >= -1e-9)
    span = p.delta_final - p.delta_init
    lo, hi = p.delta_init + 0.45 * span, p.delta_init + 0.75 * span
    sl = sweep_slope(p)
    inside = sl[(d > lo + 0.02 * span) & (d < hi - 0.02 * span)]
    before = sl[(d > p.delta_init + 0.1 * span) & (d < lo - 0.1 * span)]
    assert inside.size and before.size
    slope_in, slope_out = float(np.median(inside)), float(np.median(before))
    base_slope = float(np.median(sweep_slope(base)[int(0.3 * base.duration_ns):int(0.7 * base.duration_ns)]))
    assert slope_in < slope_out / 2.0, (slope_in, slope_out)       # chord ratio is exactly k=3; PCHIP keeps > 2
    assert slope_in < base_slope < slope_out                       # same total sweep time -> outside is faster
    # knot bookkeeping: 2 holds + 4 sweep knots
    tk, vk = knots(p)["det"]
    assert np.all(np.diff(tk) > 0) and np.all(np.diff(vk) >= 0) and len(tk) == 6
    # the knob does not change where the sweep starts or ends
    assert tk[1] == base.sweep_start_frac and tk[-2] == base.sweep_end_frac


def test_slow_region_at_the_edges_drops_degenerate_knots():
    p = default_for(EMB, slow_region=(0.0, 0.4, 2.0))     # window clipped at the start of the path
    tk, vk = knots(p)["det"]
    assert np.all(np.diff(tk) > 0) and len(tk) == 5
    p = default_for(EMB, slow_region=(1.0, 0.4, 2.0))
    tk, vk = knots(p)["det"]
    assert np.all(np.diff(tk) > 0) and len(tk) == 5
    assert abs(vk[-1] - p.delta_final) < 1e-12


# --------------------------------------------------------------------------- #
# Pulser sequence
# --------------------------------------------------------------------------- #
def test_sequence_builds_validates_and_reads_back():
    p = default_for(EMB)
    seq = build_sequence(EMB, p)
    assert seq.device.name == "AnalogDevice"
    assert seq.get_duration() == p.duration_ns and p.duration_ns % CH.clock_period == 0
    assert validate(seq) == []
    assert seq.register.layout is not None                        # requires_layout satisfied
    assert list(seq.register.qubit_ids) == EMB.qubit_ids
    dT = final_detuning_from_sequence(seq)
    assert abs(dT - p.delta_final) < 1e-6, (dT, p.delta_final)
    dT2, U = final_hamiltonian_from_sequence(seq)
    # automatic layout rounds coordinates to ~1e-6 µm, so U agrees to ~1e-6 relative
    assert dT2 == dT and np.allclose(U, EMB.U, rtol=1e-5, atol=1e-9)
    s = seq.to_abstract_repr()
    import json
    rep = json.loads(s)
    assert rep["operations"][0]["channel"] == "ryd" and "rydberg_global" in s
    assert rep["operations"][0]["amplitude"]["kind"] == "interpolated"
    assert rep["operations"][0]["detuning"]["kind"] == "interpolated"


def test_sequence_from_bare_register_and_without_layout():
    p = default_for(EMB)
    seq = build_sequence(EMB.register(), p, device=AnalogDevice, auto_layout=False)
    assert seq.register.layout is None and validate(seq) == []
    seq2 = build_sequence(EMB.register(), p, device=AnalogDevice)
    assert seq2.register.layout is not None


def test_slow_region_sequence_validates_too():
    p = default_for(EMB, slow_region=(0.6, 0.3, 3.0), duration_ns=3000)
    seq = build_sequence(EMB, p)
    assert validate(seq) == [] and seq.get_duration() == 3000
    assert abs(final_detuning_from_sequence(seq) - p.delta_final) < 1e-6


def test_over_limit_pulses_are_rejected_by_pulser():
    # amplitude above max_amp: rejected by Pulser's channel validation inside seq.add (ValueError)
    p = default_for(EMB, omega_scale=1.5)
    e = _raises(ValueError, build_sequence, EMB, p)
    assert "amplitude" in str(e).lower()
    # detuning beyond max_abs_detuning: also Pulser's channel validation
    p = default_for(EMB, delta_init=-(CH.max_abs_detuning + 10.0))
    e = _raises(ValueError, build_sequence, EMB, p)
    assert "detuning" in str(e).lower()
    p = default_for(EMB, delta_final=CH.max_abs_detuning + 5.0)
    e = _raises(ValueError, build_sequence, EMB, p)
    assert "detuning" in str(e).lower()
    # an un-snapped over-long duration is rejected by Pulser as well (PulseParams built by hand, bypassing
    # snap): Pulser's _schedule._check_duration raises RuntimeError, not ValueError
    p = PulseParams(delta_final=EMB.delta_final, delta_init=-30.0, omega_device_max=CH.max_amp,
                    duration_ns=AnalogDevice.max_sequence_duration + 400)
    e = _raises(RuntimeError, build_sequence, EMB, p)
    assert "duration" in str(e).lower()


def test_validate_flags_a_hand_built_bad_sequence():
    # Build a sequence on a permissive device so Pulser lets it through, then validate against AnalogDevice
    from pulser import Pulse, Register, Sequence
    from pulser.devices import MockDevice
    from pulser.waveforms import InterpolatedWaveform
    reg = Register.from_coordinates(EMB.coords, prefix="q", center=False)
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ryd", "rydberg_global")
    T = 4002
    amp = InterpolatedWaveform(T, [0.0, 20.0, 20.0, 0.0])
    det = InterpolatedWaveform(T, [-200.0, -200.0, 50.0, 50.0])
    seq.add(Pulse(amp, det, 0.0), "ryd")
    v = validate(seq, device=AnalogDevice)
    assert any("amplitude" in x for x in v) and any("detuning" in x for x in v)
    assert any("clock_period" in x for x in v)
