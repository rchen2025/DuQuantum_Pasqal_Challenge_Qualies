"""QUBO -> neutral-atom register on Pulser's ``AnalogDevice`` (global detuning only).

Physics (Pulser conventions, hbar = 1; rad/µs, µm, ns):

    H(t) = sum_i Omega(t)/2 sigma^x_i  -  delta(t) sum_i n_i  +  sum_{i<j} C6/r_ij^6 n_i n_j

At the end of the sweep Omega = 0 and bitstring x (n_i -> x_i) has energy

    E_dev(x) = -delta_final sum_i x_i + sum_{i<j} U_ij x_i x_j ,   U_ij = C6 / r_ij^6 > 0 .

The QUBO (cvrp.common convention, E = x^T Q x, h_i = Q_ii, J_ij = 2 Q_ij) is

    E(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j .

We want E_dev = alpha * E for one positive energy scale alpha (rad/µs per QUBO
unit).  Matching term by term:

    U_ij        = alpha J_ij     ->  r_ij = (C6 / (alpha J_ij))^(1/6)     (needs J_ij > 0)
    delta_final = -alpha h       ->  needs h_i identical for all i (single global detuning,
                                     AnalogDevice has no DMM)

Interactions are repulsive, so J_ij < 0 is *impossible* and ``embed`` refuses
it.  J_ij = 0 can only be *approximated* (U is never exactly zero; atoms are
placed far apart and the residual is reported as ``max_spurious_coupling``).

Embedding fidelity (THE number we report, also in docs/DERIVATIONS.md §5):

    rel_frobenius_offdiag = || U/alpha - J ||_F / || J ||_F     over i != j .

alpha is chosen as large as the geometry allows (the strongest coupling sits
at the 5 µm minimum distance), then capped so |delta_final| = alpha |h| stays
below the channel's max_abs_detuning (all distances scaled by
(alpha/alpha_cap)^(1/6), because U ~ r^-6) and so the register fits inside the
max radial distance.  Every limit is read from the Pulser device object at
runtime, never hard-coded.

Position optimisation reuses ``qopt.embed.optimize_positions`` (multi-start
L-BFGS with min-distance and radial hinges).  Qubit ``q{i}`` = QUBO variable i.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qopt.embed import DeviceLimits, coupling_matrix, optimize_positions, _pairwise

UNIFORM_TOL = 1e-9        # |h_i - h_j| <= UNIFORM_TOL * max(1, |h|_max)  counts as uniform
GEOM_TOL = 1e-6           # µm slack when checking the device geometry


def _default_device():
    from pulser.devices import AnalogDevice
    return AnalogDevice


# --------------------------------------------------------------------------- #
# Formulas
# --------------------------------------------------------------------------- #
def target_distance(J_ij: float, alpha: float, C6: float) -> float:
    """Distance r at which C6/r^6 = alpha*J_ij, i.e. r = (C6/(alpha J_ij))^(1/6).  µm."""
    if J_ij <= 0 or alpha <= 0:
        raise ValueError("target_distance needs J_ij > 0 and alpha > 0 (repulsive interaction)")
    return float((C6 / (alpha * J_ij)) ** (1.0 / 6.0))


def coupling_at(r: float, C6: float) -> float:
    """U = C6 / r^6 (rad/µs)."""
    return float(C6 / r ** 6)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class FidelityReport:
    """How well the register reproduces the QUBO.

    rel_frobenius_offdiag = ||U/alpha - J||_F / ||J||_F over all i != j
        (U = realised couplings in rad/µs, alpha = energy scale, J = target).
        This is the single embedding-fidelity number we report.
    worst_pair_rel_error  = max_{i<j, J_ij>0} |U_ij/alpha - J_ij| / J_ij, on ``worst_pair``.
    max_spurious_coupling = max_{i<j, J_ij==0} U_ij/alpha (QUBO units) on ``spurious_pair``;
        None if the QUBO has no zero off-diagonal couplings.
    linear_terms_uniform  = all h_i equal (within UNIFORM_TOL); otherwise delta_final uses
        the mean and the run layer must not present the result as a device result.
    """
    n_atoms: int
    alpha: float
    rel_frobenius_offdiag: float
    worst_pair_rel_error: float
    worst_pair: Optional[Tuple[int, int]]
    max_spurious_coupling: Optional[float]
    spurious_pair: Optional[Tuple[int, int]]
    min_distance: float
    max_radius: float
    linear_terms_uniform: bool
    linear_spread: float
    delta_final: float
    limit_violations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # Optional, filled by structure-aware embedders (cvrp.embed_onehot): fidelity measured on the
    # subspace where the model lives (one-hot strings), see DERIVATIONS §5.7.
    embedder: str = "generic"
    subspace_rel_frobenius: Optional[float] = None
    subspace_worst_pair_rel_error: Optional[float] = None
    subspace_anisotropy: Optional[float] = None
    subspace_same_vehicle_asymmetry: Optional[float] = None

    @property
    def ok(self) -> bool:
        return not self.limit_violations and self.linear_terms_uniform

    def summary(self) -> str:
        lines = [
            f"atoms={self.n_atoms}  alpha={self.alpha:.4g} rad/µs per QUBO unit  delta_final={self.delta_final:.4g} rad/µs",
            f"fidelity: rel Frobenius (off-diag) = {self.rel_frobenius_offdiag:.4f}",
            f"worst positive pair {self.worst_pair}: rel error {self.worst_pair_rel_error:.4f}",
            (f"largest spurious coupling (J=0 pair {self.spurious_pair}): {self.max_spurious_coupling:.3g} QUBO units"
             if self.max_spurious_coupling is not None else "no zero couplings requested"),
            f"geometry: min distance {self.min_distance:.3f} µm, max radius {self.max_radius:.3f} µm",
            f"linear terms uniform: {self.linear_terms_uniform} (spread {self.linear_spread:.3g})",
        ]
        if self.subspace_rel_frobenius is not None:
            lines.append(f"one-hot-subspace fidelity ({self.embedder}): rel Frobenius = {self.subspace_rel_frobenius:.4f}, "
                         f"worst pair rel error = {self.subspace_worst_pair_rel_error:.4f}, "
                         f"cross-vehicle anisotropy = {self.subspace_anisotropy:.4f}")
        if self.limit_violations:
            lines.append("LIMIT VIOLATIONS: " + "; ".join(self.limit_violations))
        lines += self.notes
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["worst_pair"] = list(self.worst_pair) if self.worst_pair is not None else None
        d["spurious_pair"] = list(self.spurious_pair) if self.spurious_pair is not None else None
        return d


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
@dataclass
class Embedding:
    coords: np.ndarray            # (n, 2) µm, centred on the origin
    alpha: float                  # rad/µs per QUBO energy unit
    U: np.ndarray                 # (n, n) realised couplings C6/r^6, rad/µs, zero diagonal
    J_target: np.ndarray          # (n, n) target couplings (QUBO units), zero diagonal
    h: np.ndarray                 # (n,) target linear terms (QUBO units)
    delta_final: float            # rad/µs, = -alpha * h_uniform (mean of h if not uniform)
    device: str                   # pulser device name
    fidelity: FidelityReport
    C6: float                     # rad/µs µm^6 (copied from the device)

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])

    @property
    def qubit_ids(self) -> List[str]:
        return [f"q{i}" for i in range(self.n)]

    def register(self, auto_layout: bool = False, device: Any = None):
        """Pulser Register; qubit ``q{i}`` is QUBO variable i.  With
        ``auto_layout`` the register is attached to a layout (needed by the QPU:
        AnalogDevice.requires_layout is True)."""
        from pulser import Register
        reg = Register.from_coordinates(self.coords, prefix="q", center=False)
        if auto_layout:
            if device is None:
                import pulser.devices as pd
                device = getattr(pd, self.device)
            reg = reg.with_automatic_layout(device)
        return reg

    def realized_Q(self) -> np.ndarray:
        """QUBO actually implemented by the final Hamiltonian, in QUBO units:
        J'_ij = U_ij/alpha (-> Q'_ij = J'_ij/2), h'_i = -delta_final/alpha (all equal)."""
        Q = 0.5 * self.U / self.alpha
        np.fill_diagonal(Q, -self.delta_final / self.alpha)
        return Q

    def realized_J(self) -> np.ndarray:
        return self.U / self.alpha

    def device_energy(self, x: np.ndarray) -> float:
        """E_dev(x) = -delta_final sum x + sum_{i<j} U_ij x_i x_j  (rad/µs)."""
        x = np.asarray(x, float)
        return float(-self.delta_final * x.sum() + 0.5 * x @ self.U @ x)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Embedding":
        """Inverse of to_dict (used by the embedding cache so a register is reproducible bit for bit)."""
        fd = dict(d["fidelity"])
        for k in ("worst_pair", "spurious_pair"):
            if fd.get(k) is not None:
                fd[k] = tuple(fd[k])
        known = {f.name for f in FidelityReport.__dataclass_fields__.values()}
        fid = FidelityReport(**{k: v for k, v in fd.items() if k in known})
        return cls(coords=np.asarray(d["coords_um"], float), alpha=float(d["alpha"]), U=np.asarray(d["U_realized"], float),
                   J_target=np.asarray(d["J_target"], float), h=np.asarray(d["h"], float), delta_final=float(d["delta_final"]),
                   device=d["device"], fidelity=fid, C6=float(d["C6"]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "C6": self.C6,
            "n_atoms": self.n,
            "qubit_ids": self.qubit_ids,
            "coords_um": self.coords.tolist(),
            "alpha": self.alpha,
            "delta_final": self.delta_final,
            "h": self.h.tolist(),
            "J_target": self.J_target.tolist(),
            "U_realized": self.U.tolist(),
            "fidelity": self.fidelity.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _check_inputs(J: np.ndarray, h: np.ndarray, allow_zero: bool) -> Tuple[np.ndarray, np.ndarray]:
    J = np.array(J, dtype=float)
    h = np.array(h, dtype=float).ravel()
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"J must be square, got shape {J.shape}")
    n = J.shape[0]
    if h.shape[0] != n:
        raise ValueError(f"h has length {h.shape[0]} but J is {n}x{n}")
    if not np.allclose(J, J.T, atol=1e-9, rtol=1e-9):
        raise ValueError("J must be symmetric (J_ij = 2 Q_ij with Q symmetric)")
    if not (np.isfinite(J).all() and np.isfinite(h).all()):
        raise ValueError("J and h must be finite")
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    off = ~np.eye(n, dtype=bool)
    neg = np.argwhere(np.triu(J < 0, 1))
    if len(neg):
        pairs = ", ".join(f"({i},{j}): J={J[i, j]:.4g}" for i, j in neg[:10])
        more = "" if len(neg) <= 10 else f" ... and {len(neg) - 10} more"
        raise ValueError(
            f"{len(neg)} negative coupling(s) cannot be realised on a neutral-atom device "
            f"(Rydberg interaction C6/r^6 is repulsive, so U_ij = alpha J_ij needs J_ij > 0): {pairs}{more}. "
            "Reformulate so every off-diagonal coupling is >= 0."
        )
    if not allow_zero:
        zero = np.argwhere(np.triu((J == 0) & off, 1))
        if len(zero):
            pairs = ", ".join(f"({i},{j})" for i, j in zero[:10])
            raise ValueError(f"{len(zero)} zero coupling(s) with allow_zero=False (U = C6/r^6 is never 0): {pairs}")
    return J, h


def _geometry_violations(coords: np.ndarray, limits: DeviceLimits) -> Tuple[List[str], float, float]:
    n = len(coords)
    if n > 1:
        _, d = _pairwise(coords)
        dmin = float(d.min())
    else:
        dmin = math.inf
    rad = float(np.sqrt((coords ** 2).sum(1)).max()) if n else 0.0
    v: List[str] = []
    if n > 1 and dmin < limits.min_atom_distance - GEOM_TOL:
        v.append(f"min distance {dmin:.4f} µm < device min_atom_distance {limits.min_atom_distance}")
    if limits.max_radial_distance is not None and rad > limits.max_radial_distance + GEOM_TOL:
        v.append(f"max radius {rad:.4f} µm > device max_radial_distance {limits.max_radial_distance}")
    return v, dmin, rad


def _fidelity(U: np.ndarray, alpha: float, J: np.ndarray) -> Dict[str, Any]:
    n = J.shape[0]
    off = ~np.eye(n, dtype=bool)
    Jr = U / alpha
    num = float(np.linalg.norm((Jr - J)[off]))
    den = float(np.linalg.norm(J[off]))
    if den > 0:
        rel = num / den
    else:
        rel = 0.0 if num == 0 else math.inf
    pos = (J > 0) & off
    if pos.any():
        rel_pair = np.where(pos, np.abs(Jr - J) / np.where(pos, J, 1.0), -1.0)
        i, j = np.unravel_index(int(np.argmax(rel_pair)), rel_pair.shape)
        worst, worst_pair = float(rel_pair[i, j]), (int(min(i, j)), int(max(i, j)))
    else:
        worst, worst_pair = 0.0, None
    zero = (J == 0) & off
    if zero.any():
        sp = np.where(zero, Jr, -1.0)
        i, j = np.unravel_index(int(np.argmax(sp)), sp.shape)
        spur, spur_pair = float(sp[i, j]), (int(min(i, j)), int(max(i, j)))
    else:
        spur, spur_pair = None, None
    return {"rel_frobenius_offdiag": rel, "worst_pair_rel_error": worst, "worst_pair": worst_pair,
            "max_spurious_coupling": spur, "spurious_pair": spur_pair}


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def embed(
    J: np.ndarray,
    h: np.ndarray,
    device: Any = None,
    alpha: Optional[float] = None,
    seed: int = 0,
    n_starts: int = 8,
    allow_zero: bool = True,
) -> Embedding:
    """QUBO couplings J (J_ij = 2 Q_ij, zero diagonal) and linear terms h (h_i = Q_ii)
    -> register coordinates + global detuning on ``device`` (default AnalogDevice).

    Raises ValueError for negative couplings (not realisable), for zero couplings when
    ``allow_zero`` is False, and for more atoms than the device holds.  Geometry /
    detuning problems that survive the automatic fixes are *reported* in
    ``fidelity.limit_violations`` (Pulser will refuse such a register anyway).

    alpha: desired energy scale (rad/µs per QUBO unit).  None = as large as the
    geometry allows (strongest coupling at the min distance), then capped by the
    detuning limit |alpha h| <= max_abs_detuning and by the radial limit.
    """
    if device is None:
        device = _default_device()
    limits = DeviceLimits.from_pulser_device(device)
    J, h = _check_inputs(J, h, allow_zero)
    n = J.shape[0]
    if limits.max_atom_num is not None and n > limits.max_atom_num:
        raise ValueError(f"{n} variables > {limits.name}.max_atom_num = {limits.max_atom_num}")
    if n == 0:
        raise ValueError("empty QUBO")
    C6 = limits.c6
    notes: List[str] = []

    # ---- linear terms: must be uniform (one global detuning) --------------- #
    spread = float(h.max() - h.min()) if n else 0.0
    uniform = spread <= UNIFORM_TOL * max(1.0, float(np.abs(h).max()) if n else 1.0)
    h_eff = float(h.mean())
    if not uniform:
        notes.append(
            f"NOT A FAITHFUL DEVICE EMBEDDING: linear terms are not uniform (spread {spread:.4g}); "
            f"{limits.name} has one global detuning and no DMM, so delta_final uses the MEAN h = {h_eff:.4g}. "
            "The realised QUBO differs from the target on the diagonal; do not present a run of this "
            "embedding as a device result for the original QUBO."
        )

    # ---- positions ---------------------------------------------------------- #
    if n == 1:
        coords = np.zeros((1, 2))
        alpha_geom = 1.0
    else:
        coords, alpha_geom, info = optimize_positions(J, limits, n_starts=n_starts, seed=seed)
        coords = coords - coords.mean(0)
        if not (J > 0).any():
            notes.append("no positive couplings: atoms placed on a ring far apart, alpha is arbitrary")
            alpha_geom = 1.0

    # ---- choose alpha -------------------------------------------------------- #
    alpha_val = float(alpha_geom)
    caps: List[Tuple[str, float]] = []
    if alpha is not None:
        caps.append(("requested alpha", float(alpha)))
    if limits.max_abs_detuning is not None and abs(h_eff) > 0:
        caps.append((f"|delta_final| = alpha*|h| <= max_abs_detuning {limits.max_abs_detuning:.4g}",
                     limits.max_abs_detuning / abs(h_eff)))
    for why, cap in caps:
        if alpha_val > cap * (1 + 1e-12):
            s = (alpha_val / cap) ** (1.0 / 6.0)
            coords = coords * s
            notes.append(f"alpha {alpha_val:.4g} -> {cap:.4g} ({why}): all distances scaled x{s:.4f}")
            alpha_val = cap
    if alpha is not None and alpha_val < alpha * (1 - 1e-12):
        # asked for more than the geometry allows: shrink anyway and let the limit check speak
        s = (alpha_val / alpha) ** (1.0 / 6.0)
        coords = coords * s
        notes.append(f"requested alpha {alpha:.4g} exceeds the geometric maximum {alpha_val:.4g}: "
                     f"distances scaled x{s:.4f}, expect a min-distance violation")
        alpha_val = float(alpha)
    # radial cap: if scaling up pushed atoms outside the trap area, shrink back to the edge
    if n > 1 and limits.max_radial_distance is not None:
        rad = float(np.sqrt((coords ** 2).sum(1)).max())
        if rad > limits.max_radial_distance + GEOM_TOL:
            s = (limits.max_radial_distance - 1e-6) / rad   # Pulser's radius check is a strict '>' with no tolerance
            coords = coords * s
            alpha_val = alpha_val / s ** 6
            notes.append(f"register radius {rad:.3f} µm > {limits.max_radial_distance}: distances scaled x{s:.4f}, "
                         f"alpha -> {alpha_val:.4g} (detuning limit may now be violated)")

    delta_final = -alpha_val * h_eff
    U = coupling_matrix(coords, C6) if n > 1 else np.zeros((1, 1))
    fid = _fidelity(U, alpha_val, J) if n > 1 else {
        "rel_frobenius_offdiag": 0.0, "worst_pair_rel_error": 0.0, "worst_pair": None,
        "max_spurious_coupling": None, "spurious_pair": None}
    viol, dmin, rad = _geometry_violations(coords, limits)
    if limits.max_abs_detuning is not None and abs(delta_final) > limits.max_abs_detuning * (1 + 1e-9):
        viol.append(f"|delta_final| {abs(delta_final):.4g} > max_abs_detuning {limits.max_abs_detuning:.4g}")
    report = FidelityReport(
        n_atoms=n, alpha=alpha_val, min_distance=dmin if n > 1 else 0.0, max_radius=rad,
        linear_terms_uniform=uniform, linear_spread=spread, delta_final=delta_final,
        limit_violations=viol, notes=notes, **fid,
    )
    return Embedding(coords=coords, alpha=alpha_val, U=U, J_target=J, h=h, delta_final=delta_final,
                     device=limits.name, fidelity=report, C6=C6)


# --------------------------------------------------------------------------- #
# Docs helpers / demo
# --------------------------------------------------------------------------- #
def one_worked_pair(emb: Embedding) -> Dict[str, Any]:
    """Numbers for the worked example in docs/DERIVATIONS.md §5: the strongest coupling."""
    J = emb.J_target
    n = J.shape[0]
    if n < 2 or not (J > 0).any():
        raise ValueError("need at least one positive coupling")
    i, j = np.unravel_index(int(np.argmax(np.triu(J, 1))), J.shape)
    i, j = int(i), int(j)
    r_real = float(np.linalg.norm(emb.coords[i] - emb.coords[j]))
    r_target = target_distance(J[i, j], emb.alpha, emb.C6)
    U_real = float(emb.U[i, j])
    U_target = emb.alpha * J[i, j]
    return {
        "pair": (i, j),
        "J_target": float(J[i, j]),
        "alpha": emb.alpha,
        "C6": emb.C6,
        "U_target": float(U_target),
        "target_distance_um": r_target,
        "realized_distance_um": r_real,
        "U_realized": U_real,
        "J_realized": U_real / emb.alpha,
        "rel_error": abs(U_real / emb.alpha - J[i, j]) / J[i, j],
    }


def demo_qubo(n: int = 6, seed: int = 0, n_zero: int = 3, h_value: float = -1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Toy positive-coupling QUBO: J_ij ~ U[0.2, 1] on a symmetric matrix with
    ``n_zero`` pairs set to exactly zero, uniform h = h_value."""
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    vals = rng.uniform(0.2, 1.0, len(iu[0]))
    zero_idx = rng.choice(len(iu[0]), size=min(n_zero, len(iu[0])), replace=False)
    vals[zero_idx] = 0.0
    J[iu] = vals
    J = J + J.T
    h = np.full(n, float(h_value))
    return J, h


def demo_embedding(seed: int = 0, n_starts: int = 8, **kw: Any) -> Embedding:
    J, h = demo_qubo(seed=seed)
    return embed(J, h, seed=seed, n_starts=n_starts, **kw)


def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="QUBO -> AnalogDevice register (global detuning only)")
    p.add_argument("--demo", action="store_true", help="embed the 6-variable toy QUBO and print the report")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-starts", type=int, default=8)
    p.add_argument("--json", type=str, default=None, help="write the embedding to this JSON file")
    a = p.parse_args(argv)
    if not a.demo:
        p.print_help()
        return 0
    J, h = demo_qubo(seed=a.seed)
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print("target J (QUBO units):\n", J)
    print("target h:", h)
    emb = embed(J, h, seed=a.seed, n_starts=a.n_starts)
    print("\n" + emb.fidelity.summary())
    print("\ncoords (µm):\n", emb.coords)
    print("\nrealised J = U/alpha:\n", emb.realized_J())
    wp = one_worked_pair(emb)
    print("\nworked pair (strongest coupling):")
    for k, v in wp.items():
        print(f"  {k:22s} {v}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(emb.to_dict(), f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
