"""QUBO -> neutral-atom register + local detunings.

Physics (Pulser conventions, hbar = 1, energies in rad/µs, lengths in µm):

    H(t) = sum_i [ Omega(t)/2 sigma^x_i  -  delta_i(t) n_i ]  +  sum_{i<j} U_ij n_i n_j
    U_ij = C6 / r_ij^6,        n_i = |r><r|_i  (= x_i, 1 if atom i is Rydberg)

At the end of an adiabatic sweep Omega -> 0 and the energy of bitstring x is

    H_f(x) = -sum_i delta_i x_i + sum_{i<j} U_ij x_i x_j .

The QUBO is E(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j with h_i = Q_ii and
J_ij = 2 Q_ij.  We want H_f = alpha * E for some alpha > 0, i.e.

    U_ij = alpha * J_ij          (only possible for J_ij > 0: interactions are repulsive)
    delta_i = -alpha * h_i        (via global detuning + DMM local shift <= 0)

Position optimisation minimises || U(r)/alpha - J ||_F subject to
min-distance and radial-extent limits, with alpha chosen in closed form.
Negative couplings cannot be realised; they are pushed to ~0 (atoms far
apart) and reported as *dropped coupling mass*.

Local detuning: delta_i = delta_global + w_i * delta_dmm with w_i in [0, 1] and
delta_dmm <= 0 (DMM can only *lower* the detuning).  We set
delta_global = max_i(-alpha h_i) and give each atom the non-positive shift
needed to reach -alpha h_i.  Without a DMM channel the diagonal collapses
to the mean (documented fallback, error reported).

Uniform interface:  embed_qubo(QUBO, limits) -> Embedding with
    .coords (n, 2) µm centred,  .weights {qubit_id: w_i},  .decode(bitstring),
    .report (EmbeddingReport),  .register() -> pulser Register (lazy import)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .model import QUBO


# --------------------------------------------------------------------------- #
# Device limits (read from Pulser device objects; defaults baked in as fallback)
# --------------------------------------------------------------------------- #
@dataclass
class DeviceLimits:
    name: str
    c6: float                         # rad/µs * µm^6
    min_atom_distance: float          # µm
    max_radial_distance: Optional[float]  # µm, from origin
    max_atom_num: Optional[int]
    max_amp: Optional[float]          # rad/µs, global Rydberg channel
    max_abs_detuning: Optional[float]  # rad/µs, global Rydberg channel
    min_avg_amp: float = 0.0
    max_sequence_duration: Optional[int] = None  # ns
    min_duration: int = 16            # ns
    clock_period: int = 4             # ns
    mod_bandwidth: Optional[float] = None  # MHz
    max_runs: Optional[int] = None
    has_dmm: bool = False
    dmm_id: Optional[str] = None
    dmm_bottom_detuning: Optional[float] = None        # rad/µs per atom (<= 0)
    dmm_total_bottom_detuning: Optional[float] = None  # rad/µs summed over atoms
    dmm_min_avg_abs_detuning: float = 0.0              # rad/µs (weighted)
    requires_layout: bool = False
    min_layout_traps: int = 1
    max_layout_traps: Optional[int] = None
    min_layout_filling: float = 0.0
    max_layout_filling: float = 0.5
    rydberg_level: Optional[int] = None

    # ---- derived ------------------------------------------------------- #
    @property
    def max_coupling(self) -> float:
        """Largest realisable U_ij (atoms at min distance), rad/µs."""
        return self.c6 / self.min_atom_distance ** 6

    def coupling(self, r: float) -> float:
        return self.c6 / r ** 6

    def distance_for(self, U: float) -> float:
        return (self.c6 / U) ** (1.0 / 6.0)

    def blockade_radius(self, omega: float) -> float:
        return (self.c6 / omega) ** (1.0 / 6.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ---- constructors --------------------------------------------------- #
    @classmethod
    def from_pulser_device(cls, device: Any) -> "DeviceLimits":
        """Read every limit we need from a pulser Device/VirtualDevice."""
        ryd = device.channels["rydberg_global"]
        dmm_id = next(iter(device.dmm_channels), None)
        dmm = device.dmm_channels[dmm_id] if dmm_id else None
        return cls(
            name=device.name,
            c6=float(device.interaction_coeff),
            min_atom_distance=float(device.min_atom_distance),
            max_radial_distance=device.max_radial_distance,
            max_atom_num=device.max_atom_num,
            max_amp=ryd.max_amp,
            max_abs_detuning=ryd.max_abs_detuning,
            min_avg_amp=float(ryd.min_avg_amp or 0.0),
            max_sequence_duration=device.max_sequence_duration,
            min_duration=int(ryd.min_duration),
            clock_period=int(ryd.clock_period),
            mod_bandwidth=ryd.mod_bandwidth,
            max_runs=device.max_runs,
            has_dmm=dmm is not None,
            dmm_id=dmm_id,
            dmm_bottom_detuning=getattr(dmm, "bottom_detuning", None) if dmm else None,
            dmm_total_bottom_detuning=getattr(dmm, "total_bottom_detuning", None) if dmm else None,
            dmm_min_avg_abs_detuning=float(getattr(dmm, "min_avg_abs_detuning", 0.0) or 0.0) if dmm else 0.0,
            requires_layout=bool(device.requires_layout),
            min_layout_traps=int(device.min_layout_traps),
            max_layout_traps=device.max_layout_traps,
            min_layout_filling=float(device.min_layout_filling),
            max_layout_filling=float(device.max_layout_filling),
            rydberg_level=device.rydberg_level,
        )

    @classmethod
    def from_device_name(cls, name: str) -> "DeviceLimits":
        """Load a pulser device by name (``pulser.devices.<name>``), falling
        back to baked-in numbers (pulser 1.9.1, 2026-10) if pulser is absent."""
        try:
            import pulser.devices as pd
            dev = getattr(pd, name)
            return cls.from_pulser_device(dev)
        except (ImportError, AttributeError):
            if name in KNOWN_DEVICES:
                return KNOWN_DEVICES[name]
            raise

    @classmethod
    def default(cls) -> "DeviceLimits":
        import os
        return cls.from_device_name(os.environ.get("QOPT_DEVICE", "WeightedAnalogDevice"))


_2PI = 2 * math.pi
# Values read programmatically from pulser 1.9.1 (scripts/device_limits.py).
KNOWN_DEVICES: Dict[str, DeviceLimits] = {
    "AnalogDevice": DeviceLimits(
        "AnalogDevice", 865723.02, 5.0, 38.0, 80, 2.0 * _2PI, 20.0 * _2PI, 0.0, 6000, 16, 4, 8.0, 2000,
        False, None, None, None, 0.0, True, 1, None, 0.0, 0.5, 60),
    "WeightedAnalogDevice": DeviceLimits(
        "WeightedAnalogDevice", 12241414.53, 5.0, 80.0, 256, 2.0 * _2PI, 10.0 * _2PI, 0.3 * _2PI, 6000, 16, 4, 50.0, 500,
        True, "dmm_0", -10.0 * _2PI, -1000.0 * _2PI, 0.1 * _2PI, True, 150, 512, 0.35, 0.5, 75),
    "DigitalAnalogDevice": DeviceLimits(
        "DigitalAnalogDevice", 5420158.53, 4.0, 50.0, 100, 2.5 * _2PI, 20.0 * _2PI, 0.0, None, 16, 4, None, None,
        True, "dmm_0", -20.0 * _2PI, -2000.0 * _2PI, 0.0, True, 1, None, 0.0, 0.5, 70),
    "MockDevice": DeviceLimits(
        "MockDevice", 5420158.53, 0.0, None, None, None, None, 0.0, None, 1, 1, None, None,
        True, "dmm_0", None, None, 0.0, False, 1, None, 0.0, 0.5, 70),
}


# --------------------------------------------------------------------------- #
# Position optimisation
# --------------------------------------------------------------------------- #
def _pairwise(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return diff, d


def coupling_matrix(coords: np.ndarray, c6: float) -> np.ndarray:
    _, d = _pairwise(np.asarray(coords, float))
    U = c6 / d ** 6
    np.fill_diagonal(U, 0.0)
    return U


def best_alpha(U: np.ndarray, Jpos: np.ndarray) -> float:
    """alpha minimising ||U - alpha Jpos||_F (Jpos = positive part of J)."""
    den = float((Jpos * Jpos).sum())
    return float((U * Jpos).sum() / den) if den > 0 else 1.0


def _objective(flat: np.ndarray, n: int, T: np.ndarray, W: np.ndarray, c6: float,
               d_min: float, r_max: Optional[float], mu: float) -> Tuple[float, np.ndarray]:
    """f = sum_{i<j} W_ij (U_ij - T_ij)^2 + mu * hinge(d_min - d_ij)^2 + mu * hinge(|r_i| - r_max)^2
    with analytic gradient (vectorised)."""
    coords = flat.reshape(n, 2)
    diff, d = _pairwise(coords)
    U = c6 / d ** 6
    np.fill_diagonal(U, 0.0)
    R = U - T
    f = 0.5 * float((W * R * R).sum())  # each pair counted twice -> 0.5
    # dU/dr_i = -6 c6 d^-8 * diff  ;  df/dr_i = sum_j W_ij R_ij dU_ij/dr_i  (x2 symmetric, cancels the 0.5)
    g_pair = (W * R * (-6.0 * c6 / d ** 8))[:, :, None] * diff
    np.fill_diagonal(g_pair[:, :, 0], 0.0)
    np.fill_diagonal(g_pair[:, :, 1], 0.0)
    grad = 2.0 * g_pair.sum(axis=1)
    # min-distance hinge
    viol = np.maximum(0.0, d_min - d)
    np.fill_diagonal(viol, 0.0)
    f += 0.5 * mu * float((viol ** 2).sum())
    gh = (mu * viol * (-1.0 / np.where(d > 0, d, 1.0)))[:, :, None] * diff
    grad += 2.0 * gh.sum(axis=1)
    # radial hinge
    if r_max is not None:
        rad = np.sqrt((coords ** 2).sum(-1))
        v = np.maximum(0.0, rad - r_max)
        f += mu * float((v ** 2).sum())
        grad += (2.0 * mu * v / np.where(rad > 0, rad, 1.0))[:, None] * coords
    return f, grad.ravel()


def _objective_log(flat: np.ndarray, n: int, k: int, c6: float, logT: np.ndarray, pos: np.ndarray,
                   neg: np.ndarray, far_logU: float, d_min: float, lam: float, mu: float) -> Tuple[float, np.ndarray]:
    """Log-space objective in k dims: sum_{J>0} (log U_ij - log T_ij)^2 + hinge(log U_ij - far) over J<=0
    + hinge(log d_min - log d_ij) + lam * |extra dims|^2 (dimension annealing, BLaDE-like)."""
    X = flat.reshape(n, k)
    diff = X[:, None, :] - X[None, :, :]
    d2 = (diff ** 2).sum(-1)
    np.fill_diagonal(d2, 1.0)
    logU = math.log(c6) - 3.0 * np.log(d2)
    r = np.where(pos, logU - logT, 0.0)
    f = 0.5 * float((r ** 2).sum())
    g = ((r * (-6.0 / d2))[:, :, None] * diff).sum(1) * 2.0
    v = np.where(neg, np.maximum(0.0, logU - far_logU), 0.0)
    f += 0.5 * float((v ** 2).sum())
    g += ((v * (-6.0 / d2))[:, :, None] * diff).sum(1) * 2.0
    w = np.maximum(0.0, math.log(d_min) - 0.5 * np.log(d2))
    np.fill_diagonal(w, 0.0)
    f += 0.5 * mu * float((w ** 2).sum())
    g += ((mu * w * (-1.0 / d2))[:, :, None] * diff).sum(1) * 2.0
    if k > 2:
        f += 0.5 * lam * float((X[:, 2:] ** 2).sum())
        g[:, 2:] += lam * X[:, 2:]
    return f, g.ravel()


def _anneal_init(J: np.ndarray, c6: float, alpha0: float, d_min: float, d_far: float,
                 rng: np.random.Generator, k: int = 4) -> np.ndarray:
    """Start in k dims, anneal the extra dims away, return 2D coords."""
    from scipy.optimize import minimize
    n = J.shape[0]
    Jpos = np.maximum(J, 0.0)
    pos = Jpos > 0
    neg = (~pos) & ~np.eye(n, dtype=bool)
    logT = np.where(pos, np.log(alpha0 * Jpos + 1e-300), 0.0)
    far_logU = math.log(c6) - 6.0 * math.log(d_far)
    X = rng.uniform(-d_far, d_far, (n, k))
    for lam in (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        res = minimize(_objective_log, X.ravel(), args=(n, k, c6, logT, pos, neg, far_logU, d_min, lam, 10.0),
                       jac=True, method="L-BFGS-B", options={"maxiter": 1000})
        X = res.x.reshape(n, k)
    return X[:, :2].copy()


def _mds_init(J: np.ndarray, c6: float, alpha: float, d_far: float, rng: np.random.Generator) -> np.ndarray:
    n = J.shape[0]
    D = np.full((n, n), d_far)
    pos = J > 0
    D[pos] = (c6 / (alpha * J[pos])) ** (1.0 / 6.0)
    np.fill_diagonal(D, 0.0)
    # classical MDS
    D2 = D ** 2
    Jc = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * Jc @ D2 @ Jc
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    X = v[:, idx] * np.sqrt(np.maximum(w[idx], 1e-9))
    X += 0.05 * d_far * rng.standard_normal(X.shape)
    return X


def optimize_positions(
    J: np.ndarray,
    limits: DeviceLimits,
    n_starts: int = 8,
    seed: Optional[int] = 0,
    d_min_margin: float = 1.02,
    neg_weight: float = 1.0,
    far_factor: float = 0.02,
    maxiter: int = 2000,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Find 2D coordinates so that C6/r^6 ~ alpha * J.

    Returns (coords, alpha, info).  Couplings J_ij <= 0 get target 0 with
    weight ``neg_weight`` (relative to the mean positive target^2 weighting).
    Scale is set so the *largest* coupling uses the min distance (x margin);
    the caller may rescale afterwards for detuning limits.
    """
    from scipy.optimize import minimize

    J = 0.5 * (np.asarray(J, float) + np.asarray(J, float).T)
    np.fill_diagonal(J, 0.0)
    n = J.shape[0]
    rng = np.random.default_rng(seed)
    d_min = limits.min_atom_distance * d_min_margin if limits.min_atom_distance > 0 else 1.0
    Jpos = np.maximum(J, 0.0)
    jmax = float(Jpos.max())
    if jmax <= 0:
        # no positive couplings: place atoms far apart on a ring
        rad = max(d_min, 2.0 * d_min * n / (2 * math.pi))
        th = np.linspace(0, 2 * math.pi, n, endpoint=False)
        coords = np.stack([rad * np.cos(th), rad * np.sin(th)], 1)
        return coords, 1.0, {"note": "no positive couplings", "rel_error": 0.0}
    alpha0 = limits.coupling(d_min) / jmax  # largest coupling at min distance
    T = alpha0 * Jpos
    # weights: relative error on positive targets, absolute on zero targets
    W = np.where(Jpos > 0, 1.0 / (T ** 2 + 1e-30), neg_weight / (T[Jpos > 0].mean() ** 2))
    np.fill_diagonal(W, 0.0)
    far_U = far_factor * T[Jpos > 0].min()
    d_far = limits.distance_for(far_U)
    r_max = limits.max_radial_distance
    mu = 10.0 * float(W.max()) * limits.coupling(d_min) ** 2 / d_min ** 2

    best = None
    for s in range(n_starts):
        # starts cycle through: MDS init, dimension-annealed log-space init, random
        if s % 3 == 0:
            x0 = _mds_init(J, limits.c6, alpha0, d_far, rng)
        elif s % 3 == 1:
            x0 = _anneal_init(J, limits.c6, alpha0, d_min, d_far, rng)
        else:
            x0 = rng.uniform(-d_far, d_far, (n, 2))
        x0 -= x0.mean(0)
        res = minimize(_objective, x0.ravel(), args=(n, T, W, limits.c6, d_min, r_max, mu),
                       jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        coords = res.x.reshape(n, 2)
        coords -= coords.mean(0)
        coords = _enforce_min_distance(coords, limits.min_atom_distance)
        U = coupling_matrix(coords, limits.c6)
        alpha = best_alpha(U, Jpos)
        err = float(np.linalg.norm(U / alpha - J) / max(np.linalg.norm(J), 1e-30))
        if best is None or err < best[2]:
            best = (coords, alpha, err)
    coords, alpha, err = best
    return coords, alpha, {"rel_error": err, "n_starts": n_starts, "d_far": d_far}


def _enforce_min_distance(coords: np.ndarray, d_min: float, iters: int = 200) -> np.ndarray:
    """Push apart any pair closer than d_min (tiny corrections after L-BFGS)."""
    c = coords.copy()
    if d_min <= 0:
        return c
    for _ in range(iters):
        diff, d = _pairwise(c)
        close = d < d_min * (1 + 1e-9)
        if not close.any():
            break
        i, j = np.argwhere(np.triu(close))[0]
        v = c[i] - c[j]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = np.array([1.0, 0.0]); nv = 1.0
        push = (d_min * (1 + 1e-6) - nv) / 2 * v / nv
        c[i] += push; c[j] -= push
    return c - c.mean(0)


# --------------------------------------------------------------------------- #
# Detuning plan (diagonal terms)
# --------------------------------------------------------------------------- #
@dataclass
class DetuningPlan:
    delta_global: float                 # rad/µs, final value of the global detuning
    delta_dmm: float                    # rad/µs, final value of the DMM waveform (<= 0)
    weights: np.ndarray                 # w_i in [0, 1]
    delta_target: np.ndarray            # -alpha * h_i
    delta_realized: np.ndarray          # delta_global + w_i * delta_dmm
    zeroed_weights: List[int]           # atoms whose tiny weight was zeroed for DMM min-avg rule
    uses_dmm: bool


def plan_detunings(h: np.ndarray, alpha: float, limits: DeviceLimits) -> DetuningPlan:
    h = np.asarray(h, float)
    target = -alpha * h
    if not limits.has_dmm:
        dg = float(target.mean())
        return DetuningPlan(dg, 0.0, np.zeros_like(h), target, np.full_like(h, dg), [], False)
    dg = float(target.max())
    shift = target - dg  # <= 0
    dmm = float(shift.min())
    if dmm >= -1e-12:
        return DetuningPlan(dg, 0.0, np.zeros_like(h), target, np.full_like(h, dg), [], True)
    w = shift / dmm  # in [0, 1]
    zeroed: List[int] = []
    # DMM rule (WeightedAnalogDevice): min non-zero weight * avg|delta_dmm| >= min_avg_abs_detuning.
    # The avg is over the DMM *waveform*; at the end-point it is |dmm|; be conservative with 0.5*|dmm| for ramps.
    thr = limits.dmm_min_avg_abs_detuning
    if thr > 0:
        w_min_ok = thr / (0.5 * abs(dmm))
        for i in np.where((w > 0) & (w < w_min_ok))[0]:
            zeroed.append(int(i)); w[i] = 0.0
    realized = dg + w * dmm
    return DetuningPlan(dg, dmm, w, target, realized, zeroed, True)


# --------------------------------------------------------------------------- #
# Embedding object
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingReport:
    n_atoms: int
    alpha: float                         # rad/µs per QUBO-energy unit
    rel_error_offdiag: float             # ||U/alpha - J|| / ||J||
    max_abs_error_offdiag: float         # in QUBO units
    dropped_negative_mass: float         # sum |J_ij| over J_ij < 0 (cannot be realised)
    rel_error_diag: float                # ||delta_realized/(-alpha) - h|| / ||h||
    min_distance: float
    max_radial: float
    U_min_positive_target: float         # smallest realised coupling for a J>0 pair (rad/µs)
    U_max: float
    delta_global: float
    delta_dmm: float
    uses_dmm: bool
    zeroed_weights: List[int]
    limit_violations: List[str]
    ground_state_preserved: Optional[bool] = None   # brute force check (n <= 20)
    rescale_factor: float = 1.0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"atoms={self.n_atoms} alpha={self.alpha:.4g} rad/µs per unit",
            f"offdiag rel err={self.rel_error_offdiag:.3f} (max abs {self.max_abs_error_offdiag:.3g}), dropped negative mass={self.dropped_negative_mass:.3g}",
            f"diag rel err={self.rel_error_diag:.3f} dmm={'yes' if self.uses_dmm else 'NO (mean fallback)'} zeroed={self.zeroed_weights}",
            f"geometry: min dist {self.min_distance:.2f} µm, max radial {self.max_radial:.2f} µm, U in [{self.U_min_positive_target:.3g}, {self.U_max:.3g}] rad/µs",
            f"detuning final: global {self.delta_global:.3g}, dmm {self.delta_dmm:.3g} rad/µs",
            f"ground state preserved (brute force): {self.ground_state_preserved}",
        ]
        if self.limit_violations:
            lines.append("LIMIT VIOLATIONS: " + "; ".join(self.limit_violations))
        lines += self.notes
        return "\n".join(lines)


@dataclass
class Embedding:
    qubo: QUBO
    limits: DeviceLimits
    coords: np.ndarray           # (n, 2) µm, centred
    alpha: float
    plan: DetuningPlan
    report: EmbeddingReport
    qubit_ids: List[str]

    # ---- uniform interface ------------------------------------------------ #
    @property
    def weights(self) -> Dict[str, float]:
        return {q: float(w) for q, w in zip(self.qubit_ids, self.plan.weights)}

    @property
    def U(self) -> np.ndarray:
        return coupling_matrix(self.coords, self.limits.c6)

    def decode(self, bitstring: str) -> np.ndarray:
        """Pulser bitstring (qubit order = register order, '1' = Rydberg) ->
        original-variable assignment (slack variables dropped)."""
        x = np.array([int(c) for c in bitstring], dtype=int)
        return self.qubo.project(x)

    def decode_full(self, bitstring: str) -> np.ndarray:
        return np.array([int(c) for c in bitstring], dtype=int)

    def realized_qubo(self) -> QUBO:
        """QUBO actually implemented by the final Hamiltonian, in QUBO units:
        h'_i = -delta_i/alpha, J'_ij = U_ij/alpha."""
        U = self.U / self.alpha
        Q = 0.5 * U
        np.fill_diagonal(Q, -self.plan.delta_realized / self.alpha)
        return QUBO(Q, 0.0, list(self.qubo.names), list(self.qubo.original))

    def final_hamiltonian_energy(self, bits: np.ndarray) -> float:
        """H_f(x) in rad/µs for a full bit vector."""
        x = np.asarray(bits, float)
        return float(-self.plan.delta_realized @ x + 0.5 * x @ self.U @ x)

    def register(self, auto_layout: bool = True, device: Any = None):
        """Build the pulser Register (lazy import)."""
        from pulser import Register
        reg = Register.from_coordinates(self.coords, prefix="q", center=True)
        if auto_layout and self.limits.requires_layout:
            if device is None:
                import pulser.devices as pd
                device = getattr(pd, self.limits.name)
            reg = reg.with_automatic_layout(device)
        return reg

    def detuning_map(self, register):
        return register.define_detuning_map(self.weights)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coords": self.coords.tolist(),
            "alpha": self.alpha,
            "qubit_ids": self.qubit_ids,
            "weights": self.weights,
            "delta_global": self.plan.delta_global,
            "delta_dmm": self.plan.delta_dmm,
            "device": self.limits.name,
            "report": asdict(self.report),
        }


def _check_limits(coords: np.ndarray, plan: DetuningPlan, limits: DeviceLimits) -> List[str]:
    v: List[str] = []
    _, d = _pairwise(coords)
    dmin = float(d.min())
    if limits.min_atom_distance and dmin < limits.min_atom_distance - 1e-6:
        v.append(f"min distance {dmin:.3f} < {limits.min_atom_distance}")
    rad = float(np.sqrt((coords ** 2).sum(1)).max())
    if limits.max_radial_distance and rad > limits.max_radial_distance + 1e-6:
        v.append(f"radial extent {rad:.2f} > {limits.max_radial_distance}")
    if limits.max_atom_num and len(coords) > limits.max_atom_num:
        v.append(f"{len(coords)} atoms > max {limits.max_atom_num}")
    if limits.max_abs_detuning is not None and abs(plan.delta_global) > limits.max_abs_detuning + 1e-9:
        v.append(f"|delta_global| {abs(plan.delta_global):.3g} > {limits.max_abs_detuning:.3g}")
    if plan.uses_dmm and limits.dmm_bottom_detuning is not None:
        if plan.delta_dmm < limits.dmm_bottom_detuning - 1e-9:
            v.append(f"delta_dmm {plan.delta_dmm:.3g} < bottom {limits.dmm_bottom_detuning:.3g}")
        if limits.dmm_total_bottom_detuning is not None and plan.delta_dmm * plan.weights.sum() < limits.dmm_total_bottom_detuning:
            v.append("total DMM detuning below total_bottom_detuning")
    return v


def embed_qubo(
    qubo: QUBO,
    limits: Optional[DeviceLimits] = None,
    energy_scale: Optional[float] = None,
    n_starts: int = 8,
    seed: Optional[int] = 0,
    check_ground_state: bool = True,
    **opt_kwargs: Any,
) -> Embedding:
    """QUBO -> Embedding.

    energy_scale: desired alpha (rad/µs per QUBO unit).  Default: as large as
    the device allows (largest coupling at min distance), then reduced to fit
    the detuning limits by scaling all distances up.
    """
    limits = limits or DeviceLimits.default()
    h = qubo.linear
    J = qubo.couplings
    n = qubo.n
    coords, alpha, info = optimize_positions(J, limits, n_starts=n_starts, seed=seed, **opt_kwargs)
    notes: List[str] = []

    # --- fit alpha to detuning limits by scaling distances (U ~ s^-6) ------ #
    alpha_cap = math.inf
    if energy_scale is not None:
        alpha_cap = energy_scale
    if limits.max_abs_detuning is not None and np.abs(h).max() > 0:
        alpha_cap = min(alpha_cap, limits.max_abs_detuning / float(np.abs(h).max()))
    if limits.has_dmm and limits.dmm_bottom_detuning is not None and (h.max() - h.min()) > 0:
        alpha_cap = min(alpha_cap, abs(limits.dmm_bottom_detuning) / float(h.max() - h.min()))
    rescale = 1.0
    if alpha > alpha_cap:
        rescale = (alpha / alpha_cap) ** (1.0 / 6.0)
        coords = coords * rescale
        alpha = alpha_cap
        notes.append(f"distances scaled x{rescale:.3f} to respect detuning limits (alpha -> {alpha:.4g})")
    plan = plan_detunings(h, alpha, limits)

    U = coupling_matrix(coords, limits.c6)
    Jpos = np.maximum(J, 0)
    err_off = float(np.linalg.norm(U / alpha - J) / max(np.linalg.norm(J), 1e-30))
    max_abs = float(np.abs(U / alpha - J).max()) if n > 1 else 0.0
    neg_mass = float(np.abs(J[J < 0]).sum() / 2)
    diag_err = float(np.linalg.norm(-plan.delta_realized / alpha - h) / max(np.linalg.norm(h), 1e-30))
    _, d = _pairwise(coords)
    pos_pairs = (Jpos > 0)
    U_min_pos = float(U[pos_pairs].min()) if pos_pairs.any() else 0.0
    rep = EmbeddingReport(
        n_atoms=n, alpha=alpha, rel_error_offdiag=err_off, max_abs_error_offdiag=max_abs,
        dropped_negative_mass=neg_mass, rel_error_diag=diag_err,
        min_distance=float(d.min()) if n > 1 else 0.0,
        max_radial=float(np.sqrt((coords ** 2).sum(1)).max()),
        U_min_positive_target=U_min_pos, U_max=float(U.max()) if n > 1 else 0.0,
        delta_global=plan.delta_global, delta_dmm=plan.delta_dmm, uses_dmm=plan.uses_dmm,
        zeroed_weights=plan.zeroed_weights, limit_violations=_check_limits(coords, plan, limits),
        rescale_factor=rescale, notes=notes,
    )
    if not limits.has_dmm and np.ptp(h) > 0:
        rep.notes.append("device has no DMM: diagonal terms replaced by their mean (global detuning only)")
    qubit_ids = [f"q{i}" for i in range(n)]
    emb = Embedding(qubo, limits, coords, alpha, plan, rep, qubit_ids)
    if check_ground_state and n <= 20:
        from .classical import brute_force
        gs_q = brute_force(qubo.Q)
        gs_r = brute_force(emb.realized_qubo().Q)
        same = bool(np.array_equal(gs_q.x, gs_r.x))
        if not same:
            # degenerate optimum? accept if the realized argmin is also a QUBO argmin
            same = abs(qubo.energy(gs_r.x) - qubo.energy(gs_q.x)) < 1e-9 * max(1.0, abs(gs_q.energy))
        rep.ground_state_preserved = same
    return emb
