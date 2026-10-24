"""Plots and tables: register with excited atoms, embedding-error heatmap,
schedule waveforms, score vs shots, and the classical / qubo-solver / ours table.

All plotting functions take an optional ``ax`` and return it.  ``make_report``
assembles everything into one figure and saves it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embed import Embedding
from .model import Problem, QUBO
from .run import Score, score_counts
from .schedule import Schedule


def _ax(ax=None, **kw):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(**kw)
    return ax


def excitation_probabilities(counts: Dict[str, int]) -> np.ndarray:
    tot = sum(counts.values())
    n = len(next(iter(counts)))
    p = np.zeros(n)
    for bs, c in counts.items():
        p += c * np.array([int(ch) for ch in bs])
    return p / max(tot, 1)


def plot_register(emb: Embedding, counts: Optional[Dict[str, int]] = None, best: Optional[np.ndarray] = None,
                  ax=None, show_couplings: bool = True, title: str = "register"):
    """Atoms coloured by Rydberg-excitation probability (from counts); atoms in
    ``best`` (full bit vector) outlined; slack atoms drawn as squares."""
    ax = _ax(ax)
    c = emb.coords
    n = len(c)
    if show_couplings:
        J = emb.qubo.couplings
        Jmax = max(float(np.abs(J).max()), 1e-12)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] > 0:
                    ax.plot([c[i, 0], c[j, 0]], [c[i, 1], c[j, 1]], color="gray", alpha=0.6 * J[i, j] / Jmax + 0.05, lw=1)
    p = excitation_probabilities(counts) if counts else np.zeros(n)
    orig = set(emb.qubo.original)
    for i in range(n):
        marker = "o" if i in orig else "s"
        edge = "red" if (best is not None and best[i]) else "black"
        lw = 2.5 if (best is not None and best[i]) else 0.8
        ax.scatter(c[i, 0], c[i, 1], s=260, c=[p[i]], cmap="viridis", vmin=0, vmax=1, marker=marker,
                   edgecolors=edge, linewidths=lw, zorder=3)
        ax.annotate(emb.qubo.names[i], (c[i, 0], c[i, 1]), ha="center", va="center", fontsize=7, color="white", zorder=4)
    lim = emb.limits
    ext = float(np.abs(c).max()) * 1.3 + lim.min_atom_distance
    if lim.max_radial_distance and lim.max_radial_distance < 2 * ext:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(lim.max_radial_distance * np.cos(th), lim.max_radial_distance * np.sin(th), "k--", lw=0.5, alpha=0.5)
    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
    ax.set_aspect("equal"); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(f"{title}: colour = P(rydberg), red = best sample, squares = slack")
    return ax


def plot_embedding_error(emb: Embedding, ax=None):
    ax = _ax(ax)
    J = emb.qubo.couplings
    E = emb.U / emb.alpha - J
    v = float(np.abs(E).max()) or 1.0
    im = ax.imshow(E, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(f"U/alpha - J  (rel err {emb.report.rel_error_offdiag:.2f})")
    ax.set_xticks(range(emb.qubo.n)); ax.set_yticks(range(emb.qubo.n))
    ax.set_xticklabels(emb.qubo.names, rotation=90, fontsize=6); ax.set_yticklabels(emb.qubo.names, fontsize=6)
    import matplotlib.pyplot as plt
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_schedule(sched: Schedule, ax=None, weights: Optional[Sequence[float]] = None):
    ax = _ax(ax)
    w = sched.waveforms()
    t = w["t"] / 1000.0
    ax.plot(t, w["amp"], label="Omega (rad/µs)")
    ax.plot(t, w["det"], label="delta global (rad/µs)")
    if "dmm" in w:
        ax.plot(t, w["dmm"], label="delta DMM (x weight)")
        if weights is not None:
            for wi in sorted(set(np.round(weights, 3))):
                if wi > 0:
                    ax.plot(t, w["det"] + wi * w["dmm"], color="gray", alpha=0.3, lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t (µs)"); ax.set_title(f"schedule '{sched.name}' T={sched.duration} ns"); ax.legend(fontsize=7)
    return ax


def score_vs_shots(counts: Dict[str, int], problem: Problem, qubo: QUBO, decode: Callable[[str], np.ndarray],
                   classical=None, points: Sequence[int] = (10, 20, 50, 100, 200, 500, 1000), seed: int = 0,
                   n_rep: int = 5) -> List[Tuple[int, float, float]]:
    """Subsample the measured distribution: (shots, mean P(opt after repair), mean approx ratio)."""
    rng = np.random.default_rng(seed)
    keys = list(counts); w = np.array([counts[k] for k in keys], float); w /= w.sum()
    out = []
    for m in points:
        if m > sum(counts.values()):
            break
        pr, ar = [], []
        for _ in range(n_rep):
            draw = rng.choice(len(keys), size=m, p=w)
            sub = {}
            for d in draw:
                sub[keys[d]] = sub.get(keys[d], 0) + 1
            sc = score_counts(sub, problem, qubo, decode, classical=classical)
            pr.append(sc.p_optimal_repaired); ar.append(sc.approx_ratio or 0.0)
        out.append((m, float(np.mean(pr)), float(np.mean(ar))))
    return out


def plot_score_vs_shots(curves: Dict[str, List[Tuple[int, float, float]]], ax=None):
    ax = _ax(ax)
    for name, pts in curves.items():
        if pts:
            ax.plot([p[0] for p in pts], [p[2] for p in pts], "o-", label=f"{name} approx ratio")
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "s--", alpha=0.6, label=f"{name} P(opt)")
    ax.set_xscale("log"); ax.set_xlabel("shots"); ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=7)
    ax.set_title("score vs shots (subsampled)")
    return ax


def comparison_table(rows: Dict[str, Dict[str, Any]]) -> str:
    """rows: {"classical": {...}, "qubo-solver": {...}, "ours": {...}} with keys
    objective, feasible, approx_ratio, p_opt, time_s, note."""
    cols = ["method", "objective", "feasible", "approx ratio", "P(opt) raw/rep", "time (s)", "note"]
    data = [cols]
    for name, r in rows.items():
        data.append([name,
                     _fmt(r.get("objective")), str(r.get("feasible", "")), _fmt(r.get("approx_ratio"), 4),
                     r.get("p_opt", ""), _fmt(r.get("time_s"), 1), str(r.get("note", ""))])
    widths = [max(len(str(row[i])) for row in data) for i in range(len(cols))]
    lines = ["  ".join(str(row[i]).ljust(widths[i]) for i in range(len(cols))) for row in data]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.{nd}f}"


def row_from_score(sc: Score, time_s: float, note: str = "") -> Dict[str, Any]:
    return {"objective": sc.best_objective, "feasible": sc.best_objective is not None,
            "approx_ratio": sc.approx_ratio, "p_opt": f"{sc.p_optimal_raw:.2f}/{sc.p_optimal_repaired:.2f}",
            "time_s": time_s, "note": note}


def make_report(emb: Embedding, sched: Schedule, counts: Optional[Dict[str, int]], score: Optional[Score],
                curves: Optional[Dict[str, List[Tuple[int, float, float]]]] = None, path: str = "out/report.png",
                table: str = ""):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    best = None
    if score is not None and score.best_x is not None and counts:
        # highlight the most frequent sample that decodes to the best objective
        for bs, *_ in score.top_counts:
            x = emb.decode(bs)
            if np.array_equal(x, score.best_x):
                best = emb.decode_full(bs); break
    plot_register(emb, counts, best, ax=axes[0, 0])
    plot_embedding_error(emb, ax=axes[0, 1])
    plot_schedule(sched, ax=axes[1, 0], weights=list(emb.weights.values()))
    if curves:
        plot_score_vs_shots(curves, ax=axes[1, 1])
    else:
        axes[1, 1].axis("off")
    if table:
        fig.text(0.01, 0.005, table, family="monospace", fontsize=7, va="bottom")
        fig.subplots_adjust(bottom=0.2)
    fig.suptitle(f"{emb.limits.name}: {emb.report.n_atoms} atoms, alpha={emb.alpha:.3g}")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
