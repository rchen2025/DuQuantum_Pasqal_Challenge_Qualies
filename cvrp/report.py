"""Deliverable D5 figures from run records (cvrp.run / cvrp.baseline), results.csv and scan.md.

    python -m cvrp.report out/m1 out/m2 out/m3 --figures report/figures \
        [--results results.csv] [--instance S_seed00] [--pulse-scan out/m2_pulse/scan.md] [--instances instances/]

Writes, for the headline instance (default S_seed00; the best "ours" record = method containing
'polygons', backend qutip preferred over numpy-sv):

    register.png             atom positions, variable labels i,k, exactly-one cliques, excited atoms of the modal feasible shot
    couplings.png            target J vs realised U/alpha (shared log colour scale) + relative error, fidelity numbers
    pulse.png                Omega(t), delta(t) with the device limits
    gap_hist_<instance>.png  model-gap distribution over shots per method (feasible shots only; infeasible as its own bar)
    summary.png              P(reach reference) and feasibility per instance and method, from results.csv
    pulse_scan.png           P(reach model optimum) and feasibility vs duration for each Omega scale, from scan.md

Every figure function returns the saved path.  Matplotlib only (Agg), 150 dpi, tab10 / viridis colours,
fonts >= 8 pt so the PNGs stay readable at 8 cm column width.  Records without counts (dry runs) and
baseline records lacking an embedding are skipped with a one-line warning, never a crash.

Per-shot model gaps are recomputed here from the raw counts (the record keeps only the top-10 shots):
the formulation is rebuilt from ``rec['formulation']`` (name + ``w_*`` weights) on the instance JSON, and
the gap is measured against ``rec['metrics_raw']['model_optimum']``, exactly as cvrp.metrics does.  When the
instance file is missing the histogram falls back to the top-10 rows and says so.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from .common import Instance, str_to_bits  # noqa: E402

DPI = 150
FIG_W = 3.4                      # inches ~ 8.6 cm: one column
RC = {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
      "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False}
TAB10 = plt.get_cmap("tab10").colors
GAP_BINS = ["0", "0–5 %", "5–20 %", ">20 %", "infeasible"]
DEFAULT_INSTANCES_DIR = "instances"


def _warn(msg: str) -> None:
    print(f"[cvrp.report] warning: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def load_records(dirs: Iterable[str | Path], skip_no_counts: bool = True) -> List[Dict[str, Any]]:
    """Every ``*.json`` run record under the given directories (non-recursive), in path order.
    Files that are not records (no 'method' key) and, by default, dry runs (counts=None) are skipped."""
    out: List[Dict[str, Any]] = []
    for d in dirs:
        d = Path(d)
        paths = sorted(d.glob("*.json")) if d.is_dir() else ([d] if d.suffix == ".json" and d.exists() else [])
        for p in paths:
            try:
                rec = json.loads(p.read_text())
            except Exception as e:  # noqa: BLE001
                _warn(f"cannot read {p}: {e}")
                continue
            if not isinstance(rec, dict) or "method" not in rec or "instance" not in rec:
                continue
            if skip_no_counts and not rec.get("counts"):
                _warn(f"skipping {p.name}: no counts (dry run)")
                continue
            rec.setdefault("_path", str(p))
            out.append(rec)
    return out


def short_method(method: str) -> str:
    """'onehot-ub[emb=polygons]+qutip+repair:x' -> 'polygons+qutip'; 'onehot-ub+qubosolver-tabu' -> 'qubosolver-tabu'.
    A '|tag' suffix (scan points, 'pre-scan' labels) is kept, minus a 'pulse=' prefix."""
    m = method.split("+repair:")[0]
    m, _, tag = m.partition("|")
    mm = re.match(r"^[^\[\+]*\[emb=([^\]]+)\]\+(.*)$", m)
    if mm:
        m = f"{mm.group(1)}+{mm.group(2)}"
    elif "+" in m:
        m = m.split("+", 1)[1]
    if tag:
        m += "|" + tag.replace("pulse=", "")
    return m


def _method_order(label: str) -> Tuple[int, str]:
    """Sort key: random sampler, then qubo-solver baselines, then generic embedding, then ours (polygons)."""
    rank = 0 if "random" in label and "polygons" not in label else 1 if "qubosolver" in label else 2 if "generic" in label else 3
    return rank, label


def load_instance(name: str, instances_dir: str | Path = DEFAULT_INSTANCES_DIR) -> Optional[Instance]:
    p = Path(instances_dir) / f"{name}.json"
    if not p.exists():
        return None
    try:
        return Instance.load(p)
    except Exception as e:  # noqa: BLE001
        _warn(f"cannot load instance {p}: {e}")
        return None


def _formulation_for(rec: Dict[str, Any], inst: Instance):
    """Rebuild the record's formulation (name + w_* weights) on the instance; None if impossible."""
    form = rec.get("formulation") or {}
    name = form.get("formulation") if isinstance(form, dict) else None
    if not name:
        return None
    try:
        from .model import get_formulation
        weights = {k[2:]: float(v) for k, v in form.items() if k.startswith("w_")}
        return get_formulation(name, inst, **weights)
    except Exception as e:  # noqa: BLE001
        _warn(f"cannot rebuild formulation {name!r}: {e}")
        return None


def shot_gaps(rec: Dict[str, Any], inst: Optional[Instance]) -> Tuple[List[Tuple[float, int]], int, int, str]:
    """Per-bitstring (model gap %, count) for feasible shots, the infeasible count, total shots and the source
    ('counts' = every shot decoded; 'top' = only the recorded top rows were available)."""
    counts = rec.get("counts") or {}
    m = rec.get("metrics_raw") or {}
    opt = m.get("model_optimum")
    f = _formulation_for(rec, inst) if inst is not None else None
    if f is not None and counts and opt is not None and opt != 0:
        feas: List[Tuple[float, int]] = []
        n_inf = 0
        for bits, cnt in counts.items():
            try:
                a = f.decode(str_to_bits(bits))
            except Exception:  # noqa: BLE001
                a = None
            if f.feasible(a):
                feas.append((100.0 * (float(f.objective(a)) - opt) / opt, int(cnt)))
            else:
                n_inf += int(cnt)
        return feas, n_inf, sum(int(c) for c in counts.values()), "counts"
    top = m.get("top") or []
    feas = [(float(r["model_gap"]), int(r["count"])) for r in top if r.get("feasible") and r.get("model_gap") is not None]
    n_inf = sum(int(r["count"]) for r in top if not r.get("feasible"))
    total = int(m.get("shots") or sum(int(r["count"]) for r in top) or 1)
    return feas, n_inf, total, "top"


def modal_feasible_shot(rec: Dict[str, Any], inst: Optional[Instance]) -> Optional[str]:
    """Bitstring of the most frequent feasible shot (decoded from counts when the formulation is available,
    else from the recorded top rows)."""
    counts = rec.get("counts") or {}
    f = _formulation_for(rec, inst) if inst is not None else None
    if f is not None and counts:
        for bits, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
            try:
                if f.feasible(f.decode(str_to_bits(bits))):
                    return bits
            except Exception:  # noqa: BLE001
                continue
        return None
    for r in (rec.get("metrics_raw") or {}).get("top") or []:
        if r.get("feasible"):
            return r["bits"]
    return None


def pick_headline(recs: Sequence[Dict[str, Any]], instance: str) -> Optional[Dict[str, Any]]:
    """Best 'ours' record of an instance: method contains 'polygons', backend qutip (preferred) or numpy-sv."""
    cands = [r for r in recs if r.get("instance") == instance and "polygons" in str(r.get("method", ""))
             and r.get("backend") in ("qutip", "numpy-sv") and r.get("counts") and r.get("embedding")
             and r.get("repair", "none") == "none"]
    if not cands:
        return None
    cands.sort(key=lambda r: (r.get("backend") != "qutip", "|" in str(r.get("method")), -int(r.get("shots") or 0),
                              -int((r.get("pulse") or {}).get("duration_ns") or 0)))
    return cands[0]


def pulse_params_from_record(rec: Dict[str, Any]):
    """cvrp.pulse.PulseParams rebuilt from rec['pulse'] (None for baselines whose pulse dict has another shape)."""
    from .pulse import PulseParams
    pd = rec.get("pulse") or {}
    fields = {f.name for f in dataclasses.fields(PulseParams)}
    kw = {k: v for k, v in pd.items() if k in fields}
    if not {"delta_final", "delta_init", "omega_device_max"} <= set(kw):
        return None
    if kw.get("slow_region") is not None:
        kw["slow_region"] = tuple(kw["slow_region"])
    kw["duration_ns"] = int(kw.get("duration_ns", PulseParams.duration_ns))
    return PulseParams(**kw)


def _device_limits(rec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """(Omega_max, |delta|_max) read from the Pulser device named in the record; falls back to the record."""
    name = rec.get("device") or (rec.get("embedding") or {}).get("device") or "AnalogDevice"
    try:
        import pulser.devices as pdv
        ch = getattr(pdv, str(name)).channels["rydberg_global"]
        return (float(ch.max_amp) if ch.max_amp is not None else None,
                float(ch.max_abs_detuning) if ch.max_abs_detuning is not None else None)
    except Exception:  # noqa: BLE001
        pd = rec.get("pulse") or {}
        om = pd.get("omega_device_max")
        return (float(om) if om is not None else None, None)


def _finish(fig, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def _infer_NK(rec: Dict[str, Any], inst: Optional[Instance], n: int) -> Tuple[int, int]:
    if inst is not None and inst.N * inst.K == n:
        return inst.N, inst.K
    return n, 1


# --------------------------------------------------------------------------- #
# 1. Register
# --------------------------------------------------------------------------- #
def plot_register(rec: Dict[str, Any], path: str | Path, inst: Optional[Instance] = None,
                  instances_dir: str | Path = DEFAULT_INSTANCES_DIR) -> str:
    """Atom positions (µm) labelled 'i,k' (variable i*K+k), one colour per vehicle k, the exactly-one clique of
    each customer joined by thin lines, filled = excited in the most frequent feasible shot; 5 µm scale bar and
    the device radius (dashed circle, in an inset when the register is much smaller than the device)."""
    emb = rec.get("embedding") or {}
    if not emb.get("coords_um"):
        raise ValueError("record has no embedding coordinates")
    coords = np.asarray(emb["coords_um"], float)
    n = len(coords)
    if inst is None:
        inst = load_instance(rec["instance"], instances_dir)
    N, K = _infer_NK(rec, inst, n)
    bits = modal_feasible_shot(rec, inst)
    excited = np.array([c == "1" for c in bits]) if bits and len(bits) == n else np.zeros(n, bool)
    om, _ = _device_limits(rec)
    try:
        import pulser.devices as pdv
        dev = getattr(pdv, str(rec.get("device") or "AnalogDevice"))
        R_dev, d_min = float(dev.max_radial_distance), float(dev.min_atom_distance)
    except Exception:  # noqa: BLE001
        R_dev, d_min = 38.0, 5.0
    r_max = float(np.sqrt((coords ** 2).sum(1)).max()) if n else 1.0

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_W), constrained_layout=True)
        # exactly-one cliques
        for i in range(N):
            idx = [i * K + k for k in range(K) if i * K + k < n]
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    ax.plot(coords[[idx[a], idx[b]], 0], coords[[idx[a], idx[b]], 1], color="0.6", lw=0.6, zorder=1)
        for v in range(n):
            i, k = divmod(v, K)
            col = TAB10[k % 10]
            ax.scatter(*coords[v], s=70, facecolor=col if excited[v] else "white", edgecolor=col, lw=1.4, zorder=3)
            ax.annotate(f"{i},{k}" if K > 1 else str(v), coords[v], xytext=(3, 3), textcoords="offset points", fontsize=8)
        pad = max(3.0, 0.25 * r_max)
        lim = r_max + pad
        if R_dev <= 1.5 * lim:            # register fills a fair share of the device: draw the radius here
            ax.add_patch(plt.Circle((0, 0), R_dev, fill=False, ls="--", lw=0.8, color="0.4"))
            lim = R_dev * 1.05
        else:                              # small register: inset with the device radius for scale
            ins = ax.inset_axes([0.72, 0.02, 0.26, 0.26])
            ins.add_patch(plt.Circle((0, 0), R_dev, fill=False, ls="--", lw=0.8, color="0.4"))
            ins.scatter(coords[:, 0], coords[:, 1], s=2, color="k")
            ins.set_xlim(-R_dev * 1.1, R_dev * 1.1); ins.set_ylim(-R_dev * 1.1, R_dev * 1.1)
            ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([])
            ins.set_title(f"R = {R_dev:g} µm", fontsize=7, pad=1)
        # scale bar = min separation
        x0, y0 = -lim + 0.08 * lim, -lim + 0.08 * lim
        ax.plot([x0, x0 + d_min], [y0, y0], color="k", lw=2)
        ax.text(x0 + d_min / 2, y0 + 0.02 * lim, f"{d_min:g} µm", ha="center", va="bottom", fontsize=8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        emb_name = rec.get("embedder") or emb.get("embedder") or (emb.get("fidelity") or {}).get("embedder") or "?"
        ax.set_title(f"{rec['instance']}, {emb_name}, {n} atoms")
        # legend: vehicle colours + filled/hollow meaning
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", ls="", color=TAB10[k % 10], label=f"vehicle {k}") for k in range(min(K, 10))]
        handles.append(Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="excited (modal feasible shot)"))
        handles.append(Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="ground"))
        ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
        return _finish(fig, path)


# --------------------------------------------------------------------------- #
# 2. Coupling heat maps
# --------------------------------------------------------------------------- #
def plot_coupling_heatmaps(rec: Dict[str, Any], path: str | Path, eps: float = 1e-9) -> str:
    """Target J (QUBO units) and realised U/alpha on one log colour scale, plus |U/alpha - J| / max(J, eps)
    clipped at 2; the rel. Frobenius fidelity (and the subspace one, if present) annotated."""
    emb = rec.get("embedding") or {}
    if emb.get("J_target") is None or emb.get("U_realized") is None or emb.get("alpha") is None:
        raise ValueError("record embedding lacks J_target / U_realized / alpha")
    J = np.asarray(emb["J_target"], float)
    Ua = np.asarray(emb["U_realized"], float) / float(emb["alpha"])
    n = J.shape[0]
    off = ~np.eye(n, dtype=bool)
    pos = np.concatenate([J[off], Ua[off]])
    pos = pos[pos > 0]
    vmax = float(pos.max()) if pos.size else 1.0
    vmin = max(float(pos.min()) if pos.size else vmax * 1e-3, vmax * 1e-4)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    rel = np.minimum(np.abs(Ua - J) / np.maximum(J, eps), 2.0)
    fid = emb.get("fidelity") or {}

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(FIG_W, FIG_W * 1.1), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[0.08, 1.0, 0.9])
        cax = fig.add_subplot(gs[0, :])
        axJ, axU = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        axT, axE = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
        Jm = np.where(off & (J > 0), J, np.nan)
        Um = np.where(off & (Ua > 0), Ua, np.nan)
        im = axJ.imshow(Jm, cmap="viridis", norm=norm)
        axU.imshow(Um, cmap="viridis", norm=norm)
        axJ.set_title("target J"); axU.set_title("realised U/α")
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top"); cb.ax.tick_params(labelsize=7, length=2)
        cb.set_label("coupling (QUBO units, log)", fontsize=7, labelpad=2)
        imE = axE.imshow(np.where(off, rel, np.nan), cmap="viridis", vmin=0, vmax=2)
        axE.set_title("|U/α − J| / J (clip 2)")
        fig.colorbar(imE, ax=axE, shrink=0.85, pad=0.03).ax.tick_params(labelsize=7, length=2)
        for ax in (axJ, axU, axE):
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            if n > 12:
                ax.set_xticks(range(0, n, max(1, n // 6))); ax.set_yticks(range(0, n, max(1, n // 6)))
            ax.tick_params(length=2, labelsize=6.5)
        axT.axis("off")
        lines = [f"{rec.get('instance', '')}  {rec.get('embedder') or fid.get('embedder') or ''}",
                 f"α = {float(emb['alpha']):.4g} rad/µs per unit",
                 f"δ_f = {float(emb.get('delta_final', np.nan)):.3g} rad/µs"]
        if fid.get("rel_frobenius_offdiag") is not None:
            lines.append(f"rel. Frobenius err. = {float(fid['rel_frobenius_offdiag']):.3f}")
        if fid.get("subspace_rel_frobenius") is not None:
            lines.append(f"one-hot subspace err. = {float(fid['subspace_rel_frobenius']):.3f}")
        if fid.get("worst_pair_rel_error") is not None:
            lines.append(f"worst pair rel. err. = {float(fid['worst_pair_rel_error']):.2f}")
        axT.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8, transform=axT.transAxes)
        return _finish(fig, path)


# --------------------------------------------------------------------------- #
# 3. Pulse
# --------------------------------------------------------------------------- #
def plot_pulse(rec: Dict[str, Any], path: str | Path, dt: float = 4.0) -> str:
    """Omega(t) and delta(t) of the record's pulse (cvrp.pulse.waveforms) in two stacked panels with the device
    limits Omega_max and ±|delta|_max dashed and delta_f annotated."""
    from .pulse import waveforms
    p = pulse_params_from_record(rec)
    if p is None:
        raise ValueError("record pulse dict cannot be turned into PulseParams (baseline record?)")
    w = waveforms(p, dt)
    t = w["t"] * 1e-3
    om_max, de_max = _device_limits(rec)
    with plt.rc_context(RC):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_W * 0.95), sharex=True, constrained_layout=True)
        a1.plot(t, w["amp"], color=TAB10[0], lw=1.6)
        if om_max is not None:
            a1.axhline(om_max, ls="--", lw=0.8, color="0.4")
            a1.text(0.0, om_max, f" Ω_max = {om_max:.2f}", va="bottom", fontsize=7.5, color="0.3")
            a1.set_ylim(0, om_max * 1.25)
        a1.set_ylabel("Ω (rad/µs)")
        a1.set_title(f"{rec.get('instance', '')}: T = {p.duration_ns} ns, Ω = {p.omega_scale:g}·Ω_max")
        a2.plot(t, w["det"], color=TAB10[1], lw=1.6)
        a2.axhline(0, lw=0.6, color="0.6")
        if de_max is not None:
            for s in (+1, -1):
                a2.axhline(s * de_max, ls="--", lw=0.8, color="0.4")
            a2.text(0.0, de_max, f" ±δ_max = {de_max:.1f}", va="top", fontsize=7.5, color="0.3")
            a2.set_ylim(-de_max * 1.12, de_max * 1.12)
        a2.annotate(f"δ_f = {p.delta_final:.2f}", (t[-1], p.delta_final), xytext=(-4, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8)
        a2.annotate(f"δ_i = {p.delta_init:.2f}", (t[0], p.delta_init), xytext=(4, -4), textcoords="offset points",
                    ha="left", va="top", fontsize=8)
        if p.slow_region is not None:
            a2.text(0.5, 0.9, f"slow region {p.slow_region}", transform=a2.transAxes, ha="center", fontsize=7.5)
        a2.set_ylabel("δ (rad/µs)"); a2.set_xlabel("t (µs)")
        a2.set_xlim(0, t[-1])
        return _finish(fig, path)


# --------------------------------------------------------------------------- #
# 4. Gap histogram
# --------------------------------------------------------------------------- #
def gap_bins(rec: Dict[str, Any], inst: Optional[Instance]) -> Tuple[np.ndarray, str]:
    """Fractions of all shots in GAP_BINS = [exact, 0–5 %, 5–20 %, >20 %, infeasible]."""
    feas, n_inf, total, src = shot_gaps(rec, inst)
    total = max(total, 1)
    b = np.zeros(len(GAP_BINS))
    for g, c in feas:
        if g <= 1e-9:
            b[0] += c
        elif g <= 5.0:
            b[1] += c
        elif g <= 20.0:
            b[2] += c
        else:
            b[3] += c
    b[4] = n_inf
    return b / total, src


def plot_gap_histogram(recs: Sequence[Dict[str, Any]], path: str | Path, inst: Optional[Instance] = None,
                       instances_dir: str | Path = DEFAULT_INSTANCES_DIR) -> str:
    """Side-by-side bars per method: fraction of shots per model-gap bin (feasible shots only) and the infeasible
    fraction as its own bar.  Records of the same base method are merged by keeping the one with most shots."""
    recs = [r for r in recs if r.get("counts")]
    if not recs:
        raise ValueError("no records with counts")
    inst_name = recs[0]["instance"]
    if inst is None:
        inst = load_instance(inst_name, instances_dir)
    by_label: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        lab = short_method(r["method"])
        if lab not in by_label or int(r.get("shots") or 0) > int(by_label[lab].get("shots") or 0):
            by_label[lab] = r
    labels = sorted(by_label, key=_method_order)
    data, sources = [], set()
    for lab in labels:
        b, src = gap_bins(by_label[lab], inst)
        data.append(b); sources.add(src)
    data = np.array(data)
    nm = len(labels)
    with plt.rc_context(RC):
        fig_w = FIG_W if nm <= 4 else 2 * FIG_W
        fig, ax = plt.subplots(figsize=(fig_w, max(FIG_W * 0.75, 0.55 + 0.16 * nm)), constrained_layout=True)
        x = np.arange(len(GAP_BINS))
        width = 0.84 / nm
        for j, lab in enumerate(labels):
            ax.bar(x + (j - (nm - 1) / 2) * width, data[j], width, color=TAB10[j % 10], label=lab,
                   hatch="////" if j >= 10 else None, edgecolor="white", lw=0.3)
        ax.set_xticks(x); ax.set_xticklabels(GAP_BINS)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("fraction of shots")
        ax.set_xlabel("model gap")
        ax.grid(axis="y", lw=0.4, color="0.85"); ax.set_axisbelow(True)
        title = f"{inst_name}: model gap per shot"
        if "top" in sources:
            title += " (top-10 shots only)"
        ax.set_title(title)
        fig.legend(frameon=False, loc="outside center right", fontsize=7, handlelength=1.5)
        return _finish(fig, path)


# --------------------------------------------------------------------------- #
# 5. Summary from results.csv
# --------------------------------------------------------------------------- #
def read_results_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(v: Any) -> Optional[float]:
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def plot_summary(results_csv_path: str | Path, path: str | Path,
                 metrics: Sequence[Tuple[str, str]] = (("p_reach_reference", "P(reach reference)"),
                                                       ("feasibility_rate", "feasibility rate"))) -> str:
    """Grouped bars per instance for every method in results.csv (repair rows excluded, random kept as baseline),
    one subplot per metric; missing values ('') are skipped."""
    rows = [r for r in read_results_csv(results_csv_path) if (r.get("repair") or "none") == "none"]
    if not rows:
        raise ValueError(f"no raw rows in {results_csv_path}")
    instances = sorted({r["instance"] for r in rows})
    methods = sorted({short_method(r["method"]) for r in rows}, key=_method_order)
    val: Dict[Tuple[str, str, str], float] = {}
    for r in rows:
        lab = short_method(r["method"])
        for col, _ in metrics:
            v = _f(r.get(col))
            if v is not None:
                key = (col, r["instance"], lab)
                val[key] = max(val.get(key, -np.inf), v)   # several seeds of one method: show the best
    nm, ni = len(methods), len(instances)
    with plt.rc_context(RC):
        fig_w = FIG_W if nm * ni <= 20 else 2 * FIG_W
        fig, axes = plt.subplots(len(metrics), 1, figsize=(fig_w, max(1.45 * len(metrics) + 0.6, 0.55 + 0.16 * nm)),
                                 sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        x = np.arange(ni)
        width = 0.84 / nm
        for ax, (col, title) in zip(axes, metrics):
            for j, lab in enumerate(methods):
                ys = [val.get((col, inst, lab), np.nan) for inst in instances]
                ax.bar(x + (j - (nm - 1) / 2) * width, ys, width, color=TAB10[j % 10], label=lab,
                       hatch="////" if j >= 10 else None, edgecolor="white", lw=0.3)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(title)
            ax.grid(axis="y", lw=0.4, color="0.85")
            ax.set_axisbelow(True)
        axes[-1].set_xticks(x); axes[-1].set_xticklabels(instances, rotation=0 if ni <= 6 else 45)
        axes[0].set_title("results.csv: raw shots, no repair (best seed per method)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, loc="outside center right", fontsize=7, handlelength=1.5)
        return _finish(fig, path)


# --------------------------------------------------------------------------- #
# 6. Pulse scan
# --------------------------------------------------------------------------- #
_POINT_RE = re.compile(r"T\s*=\s*(\d+)\s*ns.*?Omega\s*x\s*([\d.]+)(?:.*?delta_f\s*x\s*([\d.]+))?", re.I)


def parse_scan_md(text: str) -> List[Dict[str, Any]]:
    """Rows of a scan.md markdown table -> dicts.  Numeric columns become floats, 'point' is parsed into
    duration_ns / omega_scale / delta_f_scale, and 'per_instance_*' columns ('a / b / c') into lists."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    rows: List[Dict[str, Any]] = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        if len(cells) != len(header):
            continue
        row: Dict[str, Any] = {}
        for h, c in zip(header, cells):
            if h.startswith("per_instance"):
                row[h] = [_f(v.strip()) for v in c.split("/")]
            elif h == "point":
                row[h] = c
                m = _POINT_RE.search(c)
                if m:
                    row["duration_ns"] = int(m.group(1))
                    row["omega_scale"] = float(m.group(2))
                    row["delta_f_scale"] = float(m.group(3)) if m.group(3) else None
            else:
                v = _f(c)
                row[h] = v if v is not None else c
        rows.append(row)
    return rows


def plot_pulse_scan(scan_md_path: str | Path, path: str | Path, metric: str = "P_reach_model",
                    feas_col: str = "feasibility") -> Optional[str]:
    """P(reach model optimum) (top) and feasibility (bottom) vs duration, one line per Omega scale.
    Returns None when the scan file is missing or has no parsable points."""
    p = Path(scan_md_path)
    if not p.exists():
        _warn(f"pulse scan {p} not found; skipping pulse_scan figure")
        return None
    rows = [r for r in parse_scan_md(p.read_text()) if "duration_ns" in r and "omega_scale" in r]
    if not rows:
        _warn(f"no parsable points in {p}; skipping pulse_scan figure")
        return None
    scales = sorted({r["omega_scale"] for r in rows})
    dfs = sorted({r.get("delta_f_scale") for r in rows}, key=lambda v: (v is None, v))
    with plt.rc_context(RC):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_W * 0.95), sharex=True, constrained_layout=True)
        for j, s in enumerate(scales):
            for jj, dfs_ in enumerate(dfs):
                pts = sorted([r for r in rows if r["omega_scale"] == s and r.get("delta_f_scale") == dfs_],
                             key=lambda r: r["duration_ns"])
                if not pts:
                    continue
                T = [r["duration_ns"] for r in pts]
                lab = f"Ω × {s:g}" + (f", δ_f × {dfs_:g}" if len(dfs) > 1 and dfs_ is not None else "")
                ls = ["-", "--", ":", "-."][jj % 4]
                a1.plot(T, [r.get(metric, np.nan) for r in pts], marker="o", ms=3.5, ls=ls, color=TAB10[j % 10], label=lab)
                a2.plot(T, [r.get(feas_col, np.nan) for r in pts], marker="o", ms=3.5, ls=ls, color=TAB10[j % 10])
        a1.set_ylabel("P(reach model opt.)"); a2.set_ylabel("feasibility rate")
        a2.set_xlabel("duration T (ns)")
        for ax in (a1, a2):
            ax.set_ylim(0, 1.02); ax.grid(lw=0.4, color="0.85")
        a1.set_title("pulse scan")
        a1.legend(frameon=False)
        return _finish(fig, path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def make_all(record_dirs: Sequence[str | Path], figures: str | Path = "report/figures", results_csv: Optional[str] = None,
             instance: str = "S_seed00", pulse_scan: Optional[str] = None,
             instances_dir: str | Path = DEFAULT_INSTANCES_DIR) -> List[str]:
    """Produce every figure; returns the list of files written (figures that could not be built are warned about)."""
    figures = Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    recs = load_records(record_dirs)
    written: List[str] = []
    inst = load_instance(instance, instances_dir)
    if inst is None:
        _warn(f"instance file {Path(instances_dir) / (instance + '.json')} not found; labels fall back to variable indices")
    head = pick_headline(recs, instance)
    if head is None:
        _warn(f"no 'ours' record (method with 'polygons', backend qutip/numpy-sv) for {instance}; skipping register/couplings/pulse")
    else:
        for fn, name in ((plot_register, "register.png"), (plot_coupling_heatmaps, "couplings.png"), (plot_pulse, "pulse.png")):
            try:
                kw = {"inst": inst} if fn is plot_register else {}
                written.append(fn(head, figures / name, **kw))
            except Exception as e:  # noqa: BLE001
                _warn(f"{name} skipped ({Path(head.get('_path', '?')).name}): {e}")
    same = [r for r in recs if r.get("instance") == instance]
    if same:
        try:
            written.append(plot_gap_histogram(same, figures / f"gap_hist_{instance}.png", inst=inst))
        except Exception as e:  # noqa: BLE001
            _warn(f"gap_hist_{instance}.png skipped: {e}")
    else:
        _warn(f"no records for {instance}; skipping gap histogram")
    if results_csv:
        if Path(results_csv).exists():
            try:
                written.append(plot_summary(results_csv, figures / "summary.png"))
            except Exception as e:  # noqa: BLE001
                _warn(f"summary.png skipped: {e}")
        else:
            _warn(f"{results_csv} not found; skipping summary.png")
    if pulse_scan:
        out = plot_pulse_scan(pulse_scan, figures / "pulse_scan.png")
        if out:
            written.append(out)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="directories of run records (out/m1 out/m2 ...)")
    ap.add_argument("--figures", default="report/figures")
    ap.add_argument("--results", default=None, help="results.csv for summary.png")
    ap.add_argument("--instance", default="S_seed00", help="headline instance")
    ap.add_argument("--pulse-scan", default=None, help="scan.md of the pulse scan for pulse_scan.png")
    ap.add_argument("--instances", default=DEFAULT_INSTANCES_DIR, help="directory of instance JSONs")
    a = ap.parse_args(argv)
    written = make_all(a.dirs, a.figures, a.results, a.instance, a.pulse_scan, a.instances)
    for w in written:
        print(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
