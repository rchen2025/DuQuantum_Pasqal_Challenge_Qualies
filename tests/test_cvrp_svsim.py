"""Sanity tests for the in-house exact state-vector emulator (cvrp.svsim)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvrp import svsim  # noqa: E402


def _seq(coords, amp_vals, det_vals, T=1000):
    from pulser import Pulse, Register, Sequence
    from pulser.devices import MockDevice
    from pulser.waveforms import ConstantWaveform
    reg = Register.from_coordinates(coords, prefix="q")
    s = Sequence(reg, MockDevice)
    s.declare_channel("r", "rydberg_global")
    s.add(Pulse(ConstantWaveform(T, amp_vals), ConstantWaveform(T, det_vals), 0.0), "r")
    return s


def test_single_atom_pi_pulse_flips():
    # Omega * T = pi with T = 1 µs -> Omega = pi rad/µs; resonant: P(r) = 1
    psi = svsim.evolve(_seq([[0.0, 0.0]], np.pi, 0.0))
    assert abs(abs(psi[1]) ** 2 - 1.0) < 1e-6


def test_single_atom_detuned_rabi_formula():
    # P(r)(t) = Omega^2/(Omega^2+delta^2) sin^2(sqrt(Omega^2+delta^2) t / 2)
    om, de, T = 4.0, 3.0, 700
    psi = svsim.evolve(_seq([[0.0, 0.0]], om, de, T=T), dt_ns=1.0)
    W = np.hypot(om, de); t = T * 1e-3
    expect = om ** 2 / W ** 2 * np.sin(W * t / 2) ** 2
    assert abs(abs(psi[1]) ** 2 - expect) < 1e-4


def test_two_atoms_in_blockade_never_both_excited():
    # 5 µm apart on MockDevice: U = C6/5^6 ~ 347 rad/µs >> Omega -> |rr> suppressed
    psi = svsim.evolve(_seq([[0.0, 0.0], [5.0, 0.0]], 4.0, 0.0, T=800))
    p = np.abs(psi) ** 2
    assert p[3] < 1e-3                 # '11'
    assert abs(p.sum() - 1) < 1e-9


def test_bit_order_matches_pulser_convention():
    # Drive only atom 0 into |r> via a position-dependent trick is not possible with a global
    # channel; instead check the energy table: state '10' (atom 0 excited) has energy -delta,
    # '01' the same, '11' = -2 delta + U.  Index of '10' must be 2 (MSB = atom 0).
    s = _seq([[0.0, 0.0], [8.0, 0.0]], 1.0, 2.0, T=100)
    E = svsim.final_energies(s)
    U = svsim.interaction_matrix(s)[0, 1]
    assert abs(E[0] - 0) < 1e-9 and abs(E[1] + 2.0) < 1e-9 and abs(E[2] + 2.0) < 1e-9
    assert abs(E[3] - (-4.0 + U)) < 1e-9


def test_sample_counts_sum_and_format():
    psi = np.zeros(4, complex); psi[2] = 1.0  # '10'
    c = svsim.sample_counts(psi, 50, seed=1)
    assert c == {"10": 50}
