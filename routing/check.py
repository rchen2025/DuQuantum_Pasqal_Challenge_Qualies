"""Exhaustive algebraic checks of the two routing encodings (no Pulser, no emulation).

What is verified for an encoding on a distance matrix D (n <= 4, all 2^n_vars strings):

* every valid tour x_tour has  E(x_tour) = depot-to-depot tour cost  (penalties vanish,
  F_pair reproduces the cost including both depot legs);
* every decodable bitstring decodes to a tour whose cost equals its energy;
* h is uniform and every off-diagonal J is >= 0;
* every ground state decodes to a valid tour of cost C_opt = ``cvrp.common.tsp_exact``;
* the set of ground-state tours (canonical, undirected) equals the set of optimal
  tours found by brute force over permutations;
* every infeasible string has energy > C_opt (the A > C_ub argument).

Plus the deliberate failure case: a 6-vertex instance of two tight, well-separated
triangles where the degree-only edge model's ground state is a 2-cycle cover.
These are classical checks of the equations; they are NOT a Pulser simulation.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cvrp.common import all_bitstrings, bits_to_str, tsp_exact

from .encodings import Encoding, EdgeEncoding, PositionEncoding, TIE_TOL, build_edge, build_position, canonical_tour, tour_cost


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #
def dist_matrix(points: np.ndarray) -> np.ndarray:
    P = np.asarray(points, float)
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def random_points(n: int, seed: int, side: float = 100.0) -> np.ndarray:
    """(n+1, 2): depot at a random (not central) point so depot legs are unequal."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, side, size=(n + 1, 2))


def two_triangles_points(side: float = 1.0, separation: float = 100.0) -> np.ndarray:
    """6 vertices: depot + 2 customers form a tight equilateral triangle; the other 3 customers
    form another, ``separation`` away.  Edge-model ground state = both triangles (cost 6*side)."""
    h = side * math.sqrt(3) / 2
    tri = np.array([[0.0, 0.0], [side, 0.0], [side / 2, h]])
    return np.vstack([tri, tri + np.array([separation, 0.0])])


# --------------------------------------------------------------------------- #
# Brute-force tours
# --------------------------------------------------------------------------- #
def all_tours(n: int) -> List[Tuple[int, ...]]:
    return [tuple(p) for p in itertools.permutations(range(1, n + 1))]


def optimal_tours(D: np.ndarray) -> Tuple[float, List[Tuple[int, ...]]]:
    """(C_opt, distinct undirected optimal tours in canonical form) by brute force."""
    n = D.shape[0] - 1
    costs = {t: tour_cost(D, t) for t in all_tours(n)}
    best = min(costs.values())
    opt = sorted({canonical_tour(t) for t, c in costs.items() if abs(c - best) <= TIE_TOL})
    return float(best), opt


# --------------------------------------------------------------------------- #
# Exhaustive check of one encoding
# --------------------------------------------------------------------------- #
def exhaustive_check(enc: Encoding, atol: float = 1e-7) -> Dict[str, Any]:
    """Run every check in the module docstring; returns a report dict with ``ok`` and ``failures``."""
    D, n = enc.D, enc.n_customers
    fails: List[str] = []
    N = enc.n_vars
    X = all_bitstrings(N)
    E = enc.energies(X)

    # coefficient structure
    h_uniform = float(enc.h.max() - enc.h.min()) <= 1e-12 * max(1.0, float(abs(enc.h).max()))
    off = ~np.eye(N, dtype=bool)
    J_min = float(enc.J[off].min())
    if not h_uniform:
        fails.append("h not uniform")
    if J_min < 0:
        fails.append(f"negative coupling {J_min}")
    if not np.allclose(enc.J, enc.J.T) or np.abs(np.diag(enc.J)).max() > 0:
        fails.append("J not symmetric / non-zero diagonal")
    if not np.allclose(enc.Q() + enc.Q().T, 2 * enc.Q()):
        fails.append("Q not symmetric")
    # Q convention agrees with (h, J)
    xr = (np.random.default_rng(0).uniform(size=N) < 0.5).astype(float)
    if abs(float(xr @ enc.Q() @ xr + enc.offset) - enc.energy(xr)) > atol:
        fails.append("x^T Q x + offset != h/J energy")

    # valid tours: energy == cost
    tours = all_tours(n)
    max_err = 0.0
    for t in tours:
        x = enc.encode(t)
        e, c = enc.energy(x), tour_cost(D, t)
        max_err = max(max_err, abs(e - c))
        dec = enc.decode(x)
        if dec is None or canonical_tour(dec) != canonical_tour(t):
            fails.append(f"encode/decode round trip failed for tour {t} -> {dec}")
    if max_err > atol:
        fails.append(f"valid-tour energy != cost (max abs err {max_err:.3g})")

    # every decodable string: energy == cost of decoded tour; count decodable strings
    decodable = 0
    for k in range(len(X)):
        dec = enc.decode(X[k])
        if dec is None:
            continue
        decodable += 1
        if abs(E[k] - tour_cost(D, dec)) > atol:
            fails.append(f"decodable string {bits_to_str(X[k])} energy {E[k]} != cost {tour_cost(D, dec)}")
            break
    expect_decodable = len(tours) if isinstance(enc, PositionEncoding) else len(tours) // 2
    if decodable != expect_decodable:
        fails.append(f"{decodable} decodable strings, expected {expect_decodable}")

    # ground states
    C_opt, opt_tours = optimal_tours(D)
    C_hk, _ = tsp_exact(D, list(range(1, n + 1)))
    if abs(C_hk - C_opt) > atol:
        fails.append(f"brute force C_opt {C_opt} != Held-Karp {C_hk}")
    E_min = float(E.min())
    gs_idx = np.where(E <= E_min + TIE_TOL)[0]
    gs_tours = set()
    for k in gs_idx:
        dec = enc.decode(X[k])
        if dec is None:
            fails.append(f"ground state {bits_to_str(X[k])} (E={E[k]:.6g}) does not decode to a tour")
        else:
            gs_tours.add(canonical_tour(dec))
    if abs(E_min - C_opt) > atol:
        fails.append(f"ground energy {E_min} != C_opt {C_opt}")
    if gs_tours != set(opt_tours):
        fails.append(f"ground-state tours {sorted(gs_tours)} != optimal tours {opt_tours}")
    # infeasible strings strictly above the optimum
    infeasible_mask = np.ones(len(X), dtype=bool)
    for k in range(len(X)):
        if enc.decode(X[k]) is not None:
            infeasible_mask[k] = False
    E_inf_min = float(E[infeasible_mask].min()) if infeasible_mask.any() else math.inf
    if E_inf_min <= C_opt + TIE_TOL:
        fails.append(f"an infeasible string has energy {E_inf_min} <= C_opt {C_opt}")

    return {
        "encoding": enc.name, "n_customers": n, "n_vars": N, "n_strings": int(len(X)),
        "h_uniform": h_uniform, "h_value": float(enc.h[0]), "J_min_offdiag": J_min, "J_max": float(enc.J.max()),
        "A": enc.A, "C_ub": enc.C_ub, "C_opt": C_opt, "ground_energy": E_min,
        "n_ground_states": int(len(gs_idx)), "n_optimal_undirected_tours": len(opt_tours),
        "optimal_tours": [list(t) for t in opt_tours], "n_decodable_strings": decodable,
        "min_infeasible_energy": E_inf_min, "max_valid_tour_energy_error": max_err,
        "ok": not fails, "failures": fails,
    }


def check_seeds(n_values=(2, 3, 4), seeds=(0, 1, 2)) -> List[Dict[str, Any]]:
    out = []
    for n in n_values:
        for s in seeds:
            D = dist_matrix(random_points(n, s))
            out.append({"seed": s, **exhaustive_check(build_position(D))})
            out.append({"seed": s, **exhaustive_check(build_edge(D))})
    return out


# --------------------------------------------------------------------------- #
# The n = 5 subtour failure of the edge model
# --------------------------------------------------------------------------- #
def edge_relaxation_failure(points: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Build the degree-only edge model on the two-triangles instance (n = 5, 15 bits) and show
    that its ground state is a cycle cover, not a tour.  Returns a report dict."""
    P = two_triangles_points() if points is None else np.asarray(points, float)
    D = dist_matrix(P)
    n = D.shape[0] - 1
    enc: EdgeEncoding = build_edge(D, allow_subtours=True)
    X = all_bitstrings(enc.n_vars)
    E = enc.energies(X)
    k = int(np.argmin(E))
    x = X[k]
    C_opt, opt = optimal_tours(D)
    deg2 = enc.is_degree_two(x)
    comps = enc.components(x)
    dec = enc.decode(x)
    refused = False
    try:
        build_edge(D)
    except ValueError:
        refused = True
    # best energy among strings that DO decode to a tour
    tour_E = [E[i] for i in range(len(X)) if enc.decode(X[i]) is not None]
    return {
        "n_customers": n, "n_vars": enc.n_vars, "ground_state": bits_to_str(x), "ground_energy": float(E[k]),
        "ground_state_degree_two": bool(deg2), "ground_state_components": comps,
        "ground_state_selected_edges": enc.selected_edges(x), "ground_state_decodes_to_tour": dec is not None,
        "C_opt_tour": C_opt, "optimal_tours": [list(t) for t in opt], "best_tour_energy": float(min(tour_E)),
        "production_builder_refuses_n5": refused,
        "is_failure_demonstrated": bool(deg2 and dec is None and len(comps) == 2 and E[k] < C_opt - TIE_TOL),
    }
