"""In-house exact state-vector emulator for small registers (<= ~14 atoms).

Why this exists: the sandbox where most of this repo is developed has
pulser-core but no qutip / torch, so pulser-simulation, emu-sv and emu-mps
can only run on the Windows venv through a job runner.  For tier S (8 atoms
with ``onehot-ub``) the Schrödinger equation is a 256-dimensional linear ODE
and is cheap to integrate exactly here.  Every headline number is still
re-run on pulser-simulation (qutip); ``tests/test_cvrp_svsim.py`` and the
Windows cross-check job compare the two (they must agree to sampling noise).

Physics (Pulser conventions, hbar = 1, rad/µs, ns):

    H(t) = sum_i Omega(t)/2 (|g><r|_i + |r><g|_i)  -  delta(t) sum_i n_i  +  sum_{i<j} U_ij n_i n_j

with Omega(t), delta(t) taken from ``pulser.sampler.sample(seq)`` — the SAME
per-nanosecond samples Pulser hands its own backends — and U_ij = C6/r_ij^6
from the sequence's register and device.  The evolution is piecewise constant
on slices of ``dt`` ns (default 4 ns = the device clock; Omega and delta are
averaged over the slice) and each slice is applied with a dense matrix
exponential (scipy.linalg.expm) for n <= 10 atoms, or with scipy's
``expm_multiply`` on the sparse Hamiltonian above that.

Basis / bit order: computational basis index b has bit i (i = 0 is the most
significant, i.e. the first character of the bitstring) = 1 when atom i
(= register qubit i = QUBO variable i) is in |r>.  This matches Pulser's
bitstring convention (first qubit = first character, '1' = Rydberg).

Sampling: ``shots`` bitstrings are drawn from |psi(T)|^2 with a seeded
numpy Generator (Pulser's backends also sample; seeds are not comparable).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def samples_from_sequence(seq) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-ns (amp, det, phase) arrays of the global Rydberg channel."""
    from pulser.sampler import sample
    s = sample(seq).to_nested_dict()
    g = s["Global"]["ground-rydberg"]
    return np.asarray(g["amp"], float), np.asarray(g["det"], float), np.asarray(g["phase"], float)


def interaction_matrix(seq) -> np.ndarray:
    reg = seq.register
    qids = list(reg.qubit_ids)
    coords = np.array([np.asarray(reg.qubits[q], float) for q in qids])
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    U = float(seq.device.interaction_coeff) / d ** 6
    np.fill_diagonal(U, 0.0)
    return U


def _diagonal_terms(n: int, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For every basis state: number of excitations and interaction energy."""
    dim = 1 << n
    idx = np.arange(dim)
    bits = ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(float)   # bit 0 = MSB = atom 0
    n_exc = bits.sum(1)
    inter = 0.5 * np.einsum("bi,ij,bj->b", bits, U, bits)
    return n_exc, inter


def _sigma_x_sum(n: int):
    """Sparse sum_i sigma^x_i as (rows, cols) index arrays (each entry 1)."""
    dim = 1 << n
    idx = np.arange(dim)
    rows = np.concatenate([idx for _ in range(n)])
    cols = np.concatenate([idx ^ (1 << (n - 1 - i)) for i in range(n)])
    return rows, cols


def evolve(seq, dt_ns: float = 4.0, psi0: Optional[np.ndarray] = None, progress: bool = False) -> np.ndarray:
    """Return the final state vector |psi(T)> for a Pulser sequence."""
    amp, det, phase = samples_from_sequence(seq)
    T = len(amp)
    U = interaction_matrix(seq)
    n = U.shape[0]
    dim = 1 << n
    n_exc, inter = _diagonal_terms(n, U)
    rows, cols = _sigma_x_sum(n)
    psi = np.zeros(dim, complex) if psi0 is None else np.asarray(psi0, complex).copy()
    if psi0 is None:
        psi[0] = 1.0           # |g g ... g>  (index 0 = no excitations)
    dense = n <= 8   # dense expm is O(2^{3n}) per step; Krylov expm_multiply is far cheaper beyond 8 atoms
    if dense:
        from scipy.linalg import expm
        X = np.zeros((dim, dim))
        X[rows, cols] = 1.0
    else:
        import scipy.sparse as sp
        from scipy.sparse.linalg import expm_multiply
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(dim, dim))
    step = max(1, int(round(dt_ns)))
    edges = list(range(0, T, step)) + [T]
    for a, b in zip(edges[:-1], edges[1:]):
        om = float(amp[a:b].mean())
        de = float(det[a:b].mean())
        ph = float(phase[a:b].mean())
        tau = (b - a) * 1e-3     # ns -> µs (energies in rad/µs)
        diag = -de * n_exc + inter
        if dense:
            H = 0.5 * om * (np.cos(ph) * X) + np.diag(diag)   # phase 0 in all our pulses; keep cos for generality
            if abs(np.sin(ph)) > 1e-12:
                # sigma^x cos(phi) + sigma^y sin(phi): add the imaginary part
                Y = np.zeros((dim, dim), complex)
                # sigma^y = i(|g><r| - |r><g|) per atom: entries depend on the direction of the flip
                for i in range(n):
                    m = 1 << (n - 1 - i)
                    idx = np.arange(dim)
                    up = (idx & m) == 0          # g -> r
                    Y[idx[up], idx[up] ^ m] = -1j
                    Y[idx[~up], idx[~up] ^ m] = 1j
                H = H + 0.5 * om * np.sin(ph) * Y
            psi = expm(-1j * tau * H) @ psi
        else:
            H = 0.5 * om * X + sp.diags(diag)
            psi = expm_multiply(-1j * tau * H, psi)
    psi /= np.linalg.norm(psi)
    return psi


def probabilities(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


def sample_counts(psi: np.ndarray, shots: int, seed: int = 0) -> Dict[str, int]:
    n = int(round(np.log2(len(psi))))
    p = probabilities(psi)
    p /= p.sum()
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(p), size=shots, p=p)
    out: Dict[str, int] = {}
    for b in np.bincount(draws, minlength=len(p)).nonzero()[0]:
        out[format(int(b), f"0{n}b")] = int(np.sum(draws == b))
    return out


def emulate(seq, shots: int = 500, seed: int = 0, dt_ns: float = 4.0) -> Tuple[Dict[str, int], np.ndarray]:
    """Sequence -> (counts, exact probability vector)."""
    psi = evolve(seq, dt_ns=dt_ns)
    return sample_counts(psi, shots, seed), probabilities(psi)


def final_energies(seq) -> np.ndarray:
    """Classical device energy E_dev(x) = -delta(T) sum x + sum U x x for every basis state (rad/µs)."""
    amp, det, _ = samples_from_sequence(seq)
    U = interaction_matrix(seq)
    n_exc, inter = _diagonal_terms(U.shape[0], U)
    return -float(det[-1]) * n_exc + inter
