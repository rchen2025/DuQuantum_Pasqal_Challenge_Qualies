"""Structure-aware embedding for one-hot formulations (``onehot-ub``): exact on the subspace where the model lives.

The generic embedder (cvrp.embed) tries to make every realised coupling U_ij/alpha equal to its
target J_ij.  For the N x K rook's graph of a one-hot encoding that is impossible in a plane with
1/r^6 interactions: the K atoms of one customer must sit 5 µm apart (the exactly-one clique),
so an atom of customer j is at almost the same distance from all K copies of customer i, and the
"should be zero" cross-vehicle couplings (i,k)-(j,l != k) come out as strong as the wanted
same-vehicle ones (i,k)-(j,k).  Measured: relative Frobenius error 0.94-0.95 on tier S (M1 rows).

The way out is to ask for less.  On a one-hot string every customer has exactly one excited atom,
so for each customer pair (i, j) exactly ONE of the K^2 couplings between their atoms is active:
U(i,k; j,l) with k = a_i, l = a_j.  The device energy on the one-hot subspace is therefore

    E_dev(a) = const  +  sum_{i<j} [ U(i,a_i; j,a_j) ]

and what the model needs is only that  same-vehicle minus cross-vehicle  reproduces alpha*w_ij:

    U_eff(i,j) := U(i,k; j,k)  -  mean_{l != k} U(i,k; j,l)      (averaged over k)      = alpha * w_ij ,

with w_ij = d_ij + 2*lam2*d_i*d_j the same-vehicle coupling of the formulation.  The cross terms are
then a constant on the subspace (exactly so for K = 2 with parallel dimers, where both cross
distances are equal; approximately for K >= 3, the residual is reported as "anisotropy").  The
linear term -A is also constant on the subspace (sum_ik x_ik = N), so delta_final and the 5 µm
cliques only have to keep non-one-hot strings above every one-hot string, which the generic
checker (`realized_register` in the run record) verifies.

Construction (per instance):
  1. alpha = U(5 µm) / (2A): the clique coupling 2A is pinned at the minimum separation.
  2. Each customer i becomes a rigid regular K-gon of side 5 µm (K = 2: a dimer) with centre c_i
     and orientation theta_i; atom (i,k) = c_i + R*(cos, sin)(theta_i + 2*pi*k/K).
  3. c_i, theta_i are optimised (L-BFGS, multiple seeded starts) so that U_eff(i,j) = alpha*w_ij
     in relative error, with hinge penalties keeping every inter-polygon atom distance >= 5 µm and
     every atom inside the 38 µm radius.  Starts: classical MDS of the target centre distances
     Delta_ij that solve U_eff(Delta) = alpha*w_ij in isolation (bisection), plus random.
  4. Fidelity: the brief's formula on the FULL matrix (will stay poor by construction — the cross
     couplings are not zero and we do not pretend they are) AND the subspace fidelity
     ||U_eff/alpha - w||_F / ||w||_F over customer pairs, worst pair, and the anisotropy
     max |U(i,k;j,l) - mean_l U| / (alpha*w_ij) over cross pairs.  All three go into the record.

Limits are read from the device object (DeviceLimits.from_pulser_device), never hard-coded.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qopt.embed import DeviceLimits, coupling_matrix, _pairwise

from .embed import Embedding, FidelityReport, _default_device, _fidelity, _geometry_violations, GEOM_TOL


def _polygon(K: int, side: float) -> np.ndarray:
    """Vertices of a regular K-gon with the given side, centred at 0 (K=2: a dimer)."""
    if K == 1:
        return np.zeros((1, 2))
    R = side / (2.0 * math.sin(math.pi / K))
    ang = 2 * math.pi * np.arange(K) / K
    return R * np.stack([np.cos(ang), np.sin(ang)], 1)


def _rot(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)   # (..., 2, 2)


def atom_coords(centres: np.ndarray, thetas: np.ndarray, K: int, side: float) -> np.ndarray:
    """(N*K, 2) atom coordinates, variable index i*K + k."""
    V = _polygon(K, side)                                   # (K, 2)
    Rm = _rot(thetas)                                       # (N, 2, 2)
    pts = centres[:, None, :] + np.einsum("nab,kb->nka", Rm, V)
    return pts.reshape(-1, 2)


def effective_couplings(coords: np.ndarray, N: int, K: int, C6: float) -> Tuple[np.ndarray, np.ndarray]:
    """U_eff(i,j) = mean_k [U(ik,jk) - mean_{l!=k} U(ik,jl)] and the anisotropy of the cross terms."""
    U = coupling_matrix(coords, C6).reshape(N, K, N, K)     # U[i,k,j,l]
    eye = np.eye(K, dtype=bool)
    same_k = U[:, :, :, :][:, np.arange(K), :, np.arange(K)]          # (K, N, N): U(i,k; j,k)
    same_k = np.transpose(same_k, (1, 2, 0))                           # (N, N, K)
    cross = np.where(eye[None, :, None, :], np.nan, U)                 # mask l == k
    cross_mean = np.nanmean(cross, axis=3)                             # (N, K, N): mean over l != k
    cross_mean = np.transpose(cross_mean, (0, 2, 1))                   # (N, N, K)
    Ueff = (same_k - cross_mean).mean(axis=2)                          # (N, N)
    np.fill_diagonal(Ueff, 0.0)
    # anisotropy: spread of ALL K(K-1) cross couplings of a customer pair around their common mean
    # (for K = 2 this compares U(i,0;j,1) with U(i,1;j,0): equal only if the dimers are parallel and
    # displaced perpendicular to their axis)
    cross_all_mean = np.nanmean(cross, axis=(1, 3), keepdims=True)                  # (N, 1, N, 1)
    dev = np.nanmax(np.abs(cross - cross_all_mean), axis=(1, 3))                    # (N, N)
    np.fill_diagonal(dev, 0.0)
    return Ueff, dev


def _target_centre_distance(w: float, alpha: float, K: int, side: float, C6: float) -> float:
    """Solve U_eff(Delta) = alpha*w for two parallel K-gons whose centres are Delta apart
    (direction-averaged), by bisection.  Used only to build a starting layout."""
    target = alpha * w
    # K = 2: centres displaced perpendicular to the dimer axis (both cross distances equal, the
    # arrangement the optimiser converges to); K >= 3: average over directions, starting beyond
    # the polygon size so that no cross pair is closer than the side.
    th = np.array([math.pi / 2]) if K == 2 else np.linspace(0, 2 * math.pi, 24, endpoint=False)

    def ueff(D: float) -> float:
        vals = []
        for t in th:
            c = np.array([[0.0, 0.0], [D * math.cos(t), D * math.sin(t)]])
            pts = atom_coords(c, np.zeros(2), K, side)
            Ue, _ = effective_couplings(pts, 2, K, C6)
            vals.append(Ue[0, 1])
        return float(np.mean(vals))

    lo, hi = (side * 1.0 if K == 2 else 1.6 * side), 200.0
    if ueff(lo) < target:
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ueff(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mds(Dt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = Dt.shape[0]
    D2 = Dt ** 2
    Jc = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * Jc @ D2 @ Jc
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    X = v[:, idx] * np.sqrt(np.maximum(w[idx], 1e-9))
    return X + 0.02 * Dt.mean() * rng.standard_normal(X.shape)


def embed_onehot(J: np.ndarray, h: np.ndarray, N: int, K: int, device: Any = None, seed: int = 0,
                 n_starts: int = 6, side_margin: float = 1e-3, maxiter: int = 3000, _debug: Optional[dict] = None) -> Embedding:
    """One-hot QUBO (variable i*K+k) -> register exact on the one-hot subspace.  See module doc."""
    from scipy.optimize import minimize

    if device is None:
        device = _default_device()
    limits = DeviceLimits.from_pulser_device(device)
    J = np.asarray(J, float); h = np.asarray(h, float)
    n = N * K
    if J.shape != (n, n):
        raise ValueError(f"J must be ({n},{n}) for N={N}, K={K}")
    if (np.triu(J, 1) < 0).any():
        raise ValueError("negative couplings cannot be realised (repulsive interaction)")
    if limits.max_atom_num is not None and n > limits.max_atom_num:
        raise ValueError(f"{n} atoms > {limits.name}.max_atom_num = {limits.max_atom_num}")
    C6 = limits.c6
    side = limits.min_atom_distance * (1 + side_margin)
    notes: List[str] = []

    # --- read the structure off J: clique coupling and same-vehicle couplings --------------- #
    Jr = J.reshape(N, K, N, K)
    clique = np.array([Jr[i, k, i, l] for i in range(N) for k in range(K) for l in range(K) if k != l]) if K > 1 else np.array([0.0])
    if K > 1 and not np.allclose(clique, clique[0], rtol=1e-9, atol=1e-9):
        raise ValueError("one-hot structure expected: all intra-customer couplings equal (2A)")
    J_clique = float(clique[0]) if K > 1 else 0.0
    w = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                vals = [Jr[i, k, j, k] for k in range(K)]
                if not np.allclose(vals, vals[0], rtol=1e-9, atol=1e-9):
                    raise ValueError("one-hot structure expected: same-vehicle couplings independent of k")
                w[i, j] = vals[0]
                cross = [Jr[i, k, j, l] for k in range(K) for l in range(K) if k != l]
                if np.abs(cross).max() > 1e-12:
                    raise ValueError("one-hot structure expected: zero cross-vehicle couplings in the target")
    spread = float(h.max() - h.min())
    uniform = spread <= 1e-9 * max(1.0, float(np.abs(h).max()))
    if not uniform:
        notes.append(f"linear terms not uniform (spread {spread:.3g}); delta_final uses the mean")
    h_eff = float(h.mean())

    # --- alpha: clique at the minimum separation ------------------------------------------- #
    if K > 1:
        alpha = C6 / side ** 6 / J_clique
    else:
        alpha = 1.0
    if limits.max_abs_detuning is not None and abs(h_eff) > 0:
        cap = limits.max_abs_detuning / abs(h_eff)
        if alpha > cap:
            notes.append(f"alpha {alpha:.4g} capped to {cap:.4g} by |delta_final| <= {limits.max_abs_detuning:.4g}; "
                         f"cliques then sit above the minimum separation")
            alpha = cap
            side = (C6 / (alpha * J_clique)) ** (1.0 / 6.0)
    T = alpha * w                                             # target effective couplings (rad/µs)

    # --- optimise centres + orientations ---------------------------------------------------- #
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(N, 1)
    Tpos = T[iu]
    wts = 1.0 / np.maximum(Tpos, 1e-12) ** 2                 # relative error
    d_min = limits.min_atom_distance
    r_max = limits.max_radial_distance
    mu = 50.0 * float(wts.max()) * (C6 / d_min ** 6) ** 2 / d_min ** 2

    # --- the quantity we actually fit: device energies of one-hot strings vs alpha * model energies --- #
    # On one-hot strings sum_ik x_ik = N, so the detuning is a constant and E_dev(x) = const + 1/2 x^T U x.
    # We minimise the residual of  1/2 x^T U(pos) x  =  alpha * x^T Q_model x + c  (c free, closed form),
    # normalised by the variance of the model energies, over all K^N one-hot strings (<= 60k) or a seeded
    # sample of 4000 of them.  This is exactly "the register orders the clusterings like the model does",
    # which is what the measurement needs; per-pair effective couplings are reported as a diagnostic.
    n_strings = K ** N
    if n_strings <= 4096:
        idx = np.arange(n_strings)
    else:
        # the fit objective is evaluated thousands of times with numerical gradients: keep it to a seeded
        # sample of one-hot strings (the reported residual is recomputed on the same sample)
        idx = rng.integers(0, n_strings, size=2048)
    A_mat = np.zeros((len(idx), N), dtype=int)
    v = idx.copy()
    for i in range(N):
        A_mat[:, i] = v % K
        v //= K
    Xoh = np.zeros((len(idx), n))
    Xoh[np.arange(len(idx))[:, None], np.arange(N)[None, :] * K + A_mat] = 1.0
    Em = alpha * np.einsum("bi,ij,bj->b", Xoh, 0.5 * J, Xoh)          # alpha * model coupling energy (QUBO: J = 2Q)
    Em = Em - Em.mean()
    Em_var = float((Em ** 2).sum()) or 1.0

    def energy_fit(pts):
        U = coupling_matrix(pts, C6)
        Ed = 0.5 * np.einsum("bi,ij,bj->b", Xoh, U, Xoh)
        Ed = Ed - Ed.mean()
        return float(((Ed - Em) ** 2).sum()) / Em_var

    # --- defects: every one-hot string with ONE bit flipped must stay above its parent by at least half the
    # model's own penalty gap (in device units), otherwise the register's ground state leaves the one-hot
    # subspace (observed: cross couplings from crowded dimers made a "customer with no atom" cheaper).
    # delta_final is a free fit parameter here (it is a constant on the one-hot subspace, so it only
    # controls these defect gaps); it starts at alpha*A, the value the QUBO algebra suggests.
    n_def_parents = min(len(idx), 64 if n > 20 else 256)   # all one-hot strings on S (16); the first 64 of the sample on M/L
    par = Xoh[:n_def_parents]
    Xdef = np.repeat(par, n, axis=0)
    flip = np.tile(np.arange(n), n_def_parents)
    Xdef[np.arange(len(Xdef)), flip] = 1.0 - Xdef[np.arange(len(Xdef)), flip]
    Eq_par = np.einsum("bi,ij,bj->b", par, 0.5 * J, par) + par @ h
    Eq_def = np.einsum("bi,ij,bj->b", Xdef, 0.5 * J, Xdef) + Xdef @ h
    model_gap = alpha * (Eq_def - np.repeat(Eq_par, n))                       # > 0 by the A rule, device units
    dscale = (alpha * float(np.abs(h).max()) or 1.0) ** 2

    def defect_penalty(pts, delta):
        U = coupling_matrix(pts, C6)
        Ed_par = -delta * par.sum(1) + 0.5 * np.einsum("bi,ij,bj->b", par, U, par)
        Ed_def = -delta * Xdef.sum(1) + 0.5 * np.einsum("bi,ij,bj->b", Xdef, U, Xdef)
        short = np.maximum(0.0, 0.5 * model_gap - (Ed_def - np.repeat(Ed_par, n)))
        return float((short ** 2).sum()) / dscale

    d_lo, d_hi = 0.0, (limits.max_abs_detuning or 1e9)

    def unpack(z):
        c = z[: 2 * N].reshape(N, 2)
        th = z[2 * N: 3 * N]
        delta = float(np.clip(z[3 * N], d_lo, d_hi))
        return c, th, delta

    def objective(z):
        c, th, delta = unpack(z)
        pts = atom_coords(c, th, K, side)
        f = energy_fit(pts) * float(wts.max()) * float(Tpos.max()) ** 2   # same units as the pair term below
        f += defect_penalty(pts, delta) * float(wts.max()) * float(Tpos.max()) ** 2
        Ue, _ = effective_couplings(pts, N, K, C6)
        f += 0.1 * float((wts * (Ue[iu] - Tpos) ** 2).sum())                 # weak pair-wise regulariser
        # min-distance hinge between atoms of DIFFERENT polygons (same-polygon distances are the side by construction)
        _, d = _pairwise(pts)
        owner = np.repeat(np.arange(N), K)
        other = owner[:, None] != owner[None, :]
        viol = np.where(other, np.maximum(0.0, d_min * (1 + 2e-3) - d), 0.0)
        f += 0.5 * mu * float((viol ** 2).sum())
        if r_max is not None:
            rad = np.sqrt((pts ** 2).sum(1))
            f += mu * float((np.maximum(0.0, rad - (r_max - 0.05)) ** 2).sum())
        return f

    # ---- analytic gradient version (same terms minus the weak pair regulariser), used for n > 20 where
    # numerical gradients over 3N+1 parameters are too slow.  Chain rule: f depends on the atom positions only
    # through U_ij = C6 d_ij^-6, and dU_ij/dp_i = -6 U_ij (p_i - p_j)/d_ij^2 ; positions depend on (c_a, theta_a)
    # through p_i = c_a + R(theta_a) v_k, so dp_i/dc_a = I and dp_i/dtheta_a = R'(theta_a) v_k.
    scale_f = float(wts.max()) * float(Tpos.max()) ** 2
    owner_idx = np.repeat(np.arange(N), K)
    kk = np.tile(np.arange(K), N)
    V = _polygon(K, side)

    def objective_grad(z):
        c, th, delta = unpack(z)
        pts = atom_coords(c, th, K, side)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = (diff ** 2).sum(-1); np.fill_diagonal(d2, 1.0)
        U = C6 / d2 ** 3; np.fill_diagonal(U, 0.0)
        G = np.zeros((n, n))          # dF/dU_ij (symmetric, each unordered pair's weight appears in both entries)
        f = 0.0
        g_delta = 0.0
        # (1) energy fit
        Ed = 0.5 * np.einsum("bi,ij,bj->b", Xoh, U, Xoh)
        r = (Ed - Ed.mean()) - Em
        f += scale_f * float((r ** 2).sum()) / Em_var
        G += scale_f * (Xoh.T @ (r[:, None] * Xoh)) / Em_var        # d/dU_ij of sum r^2 = 2 r X_i X_j, and 1/2 from E
        # (2) defect gaps
        Ed_par = -delta * par.sum(1) + 0.5 * np.einsum("bi,ij,bj->b", par, U, par)
        Ed_def = -delta * Xdef.sum(1) + 0.5 * np.einsum("bi,ij,bj->b", Xdef, U, Xdef)
        gapv = Ed_def - np.repeat(Ed_par, n)
        short = np.maximum(0.0, 0.5 * model_gap - gapv)
        f += scale_f * float((short ** 2).sum()) / dscale
        wgt = -2.0 * short * scale_f / dscale                           # d/dgap of short^2
        # dgap/dU_ij = (X_def X_def - X_par X_par)_ij / 2 * 2 -> X X products; dgap/ddelta = -(sum X_def - sum X_par)
        G += 0.5 * (Xdef.T @ (wgt[:, None] * Xdef) - par.T @ ((np.add.reduceat(wgt, np.arange(0, len(wgt), n)))[:, None] * par))
        g_delta += float((wgt * (-(Xdef.sum(1) - np.repeat(par.sum(1), n)))).sum())
        # (3) min-distance hinge (different polygons) and radius hinge: direct position gradients
        d = np.sqrt(d2)
        other = owner_idx[:, None] != owner_idx[None, :]
        viol = np.where(other, np.maximum(0.0, d_min * (1 + 2e-3) - d), 0.0)
        f += 0.5 * mu * float((viol ** 2).sum())
        gp = ((-mu * viol / d)[:, :, None] * diff).sum(1) * 2.0 * 0.5 * 2.0   # each pair twice in the matrix -> factor handled
        if r_max is not None:
            rad = np.sqrt((pts ** 2).sum(1))
            vr = np.maximum(0.0, rad - (r_max - 0.05))
            f += mu * float((vr ** 2).sum())
            gp += (2.0 * mu * vr / np.where(rad > 0, rad, 1.0))[:, None] * pts
        # chain rule through U: dF/dp_i = sum_j (G_ij + G_ji) * dU_ij/dp_i
        Gs = G + G.T
        coef = Gs * (-6.0 * U / d2)                                       # (n, n)
        gp += (coef[:, :, None] * diff).sum(1)
        # positions -> centres / orientations
        gc = np.zeros((N, 2)); gth = np.zeros(N)
        np.add.at(gc, owner_idx, gp)
        Rp = _rot(th + np.pi / 2)                                         # R'(theta) v = R(theta + 90deg) v
        dpdth = np.einsum("nab,nb->na", Rp[owner_idx], V[kk])             # (n, 2)
        np.add.at(gth, owner_idx, (gp * dpdth).sum(1))
        grad = np.concatenate([gc.ravel(), gth, [g_delta if d_lo < delta < d_hi else 0.0]])
        return f, grad

    if _debug is not None:        # test hook: expose the closures for gradient checks
        _debug.update(objective=objective, objective_grad=objective_grad, unpack=unpack, alpha=alpha, h_eff=h_eff)
        return None  # type: ignore[return-value]

    # starting centre distances from the isolated-pair solution
    Dt = np.zeros((N, N))
    for i, j in zip(*iu):
        Dt[i, j] = Dt[j, i] = _target_centre_distance(w[i, j], alpha, K, side, C6) if w[i, j] > 0 else 3 * d_min * N ** 0.5
    best = None
    if n > 20:
        n_starts = max(n_starts, 8)      # analytic gradients: starts are cheap
    for s in range(n_starts):
        if s % 2 == 0:
            c0 = _mds(Dt, rng)
        else:
            c0 = rng.uniform(-Dt.mean(), Dt.mean(), (N, 2))
        th0 = np.zeros(N) if s < 2 else rng.uniform(0, 2 * math.pi / K, N)
        z0 = np.concatenate([(c0 - c0.mean(0)).ravel(), th0, [-alpha * h_eff]])
        if n > 20:
            res = minimize(objective_grad, z0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        else:
            res = minimize(objective, z0, method="L-BFGS-B", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    c, th, delta_fit = unpack(best.x)
    # Hard guarantee of the device's minimum separation between atoms of different polygons: push the two
    # polygons of the closest violating pair apart along the line joining their centres (tiny moves; the
    # soft hinge normally leaves nothing to do).  Pulser rejects any pair below 5 µm, so this is not optional.
    owner = np.repeat(np.arange(N), K)
    c = c - c.mean(0)
    pts = atom_coords(c, th, K, side)
    # Hard guarantee of the device radius too (Pulser's check is a strict '>' from the ORIGIN with no
    # tolerance): alternate the two projections — push the closest violating pair of polygons apart along
    # the line joining their centres, pull the polygon owning the outermost atom radially inwards by its
    # excess — until both constraints hold.  Each move is tiny; the soft hinges normally leave nothing to do.
    r_lim = (r_max - 2e-3) if r_max is not None else np.inf
    for _ in range(5000):
        pts = atom_coords(c, th, K, side)
        _, d = _pairwise(pts)
        d = np.where(owner[:, None] != owner[None, :], d, np.inf)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        rad = np.sqrt((pts ** 2).sum(1))
        k = int(np.argmax(rad))
        sep_ok = d[i, j] >= d_min * (1 + 2e-3)
        rad_ok = rad[k] <= r_lim
        if sep_ok and rad_ok:
            break
        if not rad_ok:
            a = owner[k]
            u = pts[k] / rad[k]
            c[a] -= (rad[k] - r_lim + 1e-3) * u
            notes_push = True
        if not sep_ok:
            a, b = owner[i], owner[j]
            v = c[a] - c[b]
            nv = np.linalg.norm(v)
            v = v / nv if nv > 1e-9 else np.array([1.0, 0.0])
            push = 0.5 * (d_min * (1 + 3e-3) - d[i, j])
            c[a] += push * v
            c[b] -= push * v
            notes_push = True
    pts = atom_coords(c, th, K, side)
    _, d = _pairwise(pts)
    d = np.where(owner[:, None] != owner[None, :], d, np.inf)
    if float(d.min()) < d_min or (r_max is not None and float(np.sqrt((pts ** 2).sum(1)).max()) > r_max):
        notes.append("register violates the device's min separation / radius after 5000 projection steps "
                     "(polygons too large for N); Pulser will reject it")

    # --- reports ------------------------------------------------------------------------------ #
    U = coupling_matrix(pts, C6)
    fid = _fidelity(U, alpha, J)
    Ue, dev = effective_couplings(pts, N, K, C6)
    sub_err = float(np.linalg.norm((Ue / alpha - w)[iu]) / max(np.linalg.norm(w[iu]), 1e-30))
    rel_pair = np.abs(Ue / alpha - w)[iu] / np.maximum(w[iu], 1e-30)
    sub_worst = float(rel_pair.max()) if len(rel_pair) else 0.0
    aniso = float((dev[iu] / np.maximum(T[iu], 1e-30)).max()) if len(iu[0]) else 0.0
    # same-vehicle couplings should not depend on k either (the two labelings of one clustering then share
    # an energy); report the spread relative to the target
    Ur = U.reshape(N, K, N, K)
    same_k = np.stack([Ur[:, k, :, k] for k in range(K)], -1)                       # (N, N, K)
    same_asym = float((np.ptp(same_k, axis=2)[iu] / np.maximum(T[iu], 1e-30)).max()) if K > 1 and len(iu[0]) else 0.0
    efit = energy_fit(pts)
    notes.append(f"one-hot-subspace energy fit: normalised residual = {efit:.4f} (sum of squared residuals of device vs "
                 f"alpha*model energies, slope fixed at alpha, over the model-energy variance; {len(idx)} one-hot strings)")
    notes.append(f"same-vehicle coupling spread over k, relative to target (max over pairs): {same_asym:.3f}")
    viol, dmin, rad = _geometry_violations(pts, limits)
    delta_final = float(delta_fit)
    notes.append(f"delta_final fitted to {delta_final:.4g} rad/µs (QUBO algebra value alpha*A = {-alpha * h_eff:.4g}); "
                 f"it is constant on the one-hot subspace and only sets the single-flip defect gaps")
    dpen = defect_penalty(pts, delta_final)
    notes.append(f"single-flip defect shortfall (0 = every one-bit defect of every one-hot string sits >= half the "
                 f"model's penalty gap above it): {dpen:.4g}")
    if limits.max_abs_detuning is not None and abs(delta_final) > limits.max_abs_detuning * (1 + 1e-9):
        viol.append(f"|delta_final| {abs(delta_final):.4g} > max_abs_detuning {limits.max_abs_detuning:.4g}")
    notes.append(f"structure-aware one-hot embedding: {N} regular {K}-gons of side {side:.3f} µm; "
                 f"full-matrix error is expected to be large (cross-vehicle couplings are constant on the one-hot "
                 f"subspace, not zero); the subspace fidelity is the relevant number")
    report = FidelityReport(n_atoms=n, alpha=alpha, min_distance=dmin, max_radius=rad, linear_terms_uniform=uniform,
                            linear_spread=spread, delta_final=delta_final, limit_violations=viol, notes=notes,
                            embedder="onehot-polygons", subspace_rel_frobenius=sub_err,
                            subspace_worst_pair_rel_error=sub_worst, subspace_anisotropy=aniso,
                            subspace_same_vehicle_asymmetry=same_asym, **fid)
    return Embedding(coords=pts, alpha=alpha, U=U, J_target=J, h=h, delta_final=delta_final,
                     device=limits.name, fidelity=report, C6=C6)
