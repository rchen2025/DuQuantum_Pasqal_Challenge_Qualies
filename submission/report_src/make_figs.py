"""Figures for the DuQuantum / Pasqal challenge report (Team Qualies). All numbers come from
_figdata.json, which is extracted from the repo's run records, results.csv and instance files."""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import PchipInterpolator

D = json.load(open("_figdata.json"))   # compact extract of the repo records (see BUILD.md)
OUT = "."

# reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.5,
    "axes.titlesize": 7.8, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
CUST = [BLUE, ORANGE, AQUA, VIOLET, YELLOW, MAGENTA, GREEN, RED]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1: pipeline on S_seed00
inst = D["instances"]["S_seed00"]
emb = D["emb_S00"]
counts = D["S00_counts_top"]
shots = D["S00_shots"]
locs = np.array(inst["locations"]); dem = inst["demands"]; depot = inst["depot"]
clusters = inst["best_clusters"]; routes = inst["routes"]

fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.25], wspace=0.35))
ax = axs[0]
ax.set_title("(a) S_seed00, routed optimum")
for ci, (cl, rt) in enumerate(zip(clusters, routes)):
    col = [BLUE, ORANGE][ci]
    path = [depot] + [inst["locations"][i] for i in rt] + [depot]
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "-", color=col, lw=1.4, zorder=1)
for i, (x, y) in enumerate(locs):
    ci = 0 if i in clusters[0] else 1
    ax.scatter(x, y, s=28 + 22 * dem[i], color=[BLUE, ORANGE][ci], edgecolor="white", lw=0.8, zorder=3)
    ax.annotate(f"c{i} (d={dem[i]})", (x, y), xytext=(5, 4 if i != 1 else 9), textcoords="offset points", fontsize=6.6, color=INK2)
ax.scatter(*depot, marker="s", s=48, color=INK, zorder=4)
ax.annotate("depot", depot, xytext=(5, -9), textcoords="offset points", fontsize=7, color=INK)
ax.set_xlim(-6, 106); ax.set_ylim(-6, 118); ax.set_aspect("equal")
ax.set_xlabel("x (distance units)"); ax.set_ylabel("y")
ax.text(0.02, 0.97, f"C* = {inst['ref_cost']:.1f} (proven)\nloads 5 / 5 of Q = 7",
        transform=ax.transAxes, fontsize=6.6, color=INK2, va="top")

ax = axs[1]
ax.set_title("(b) register, modal shot")
coords = np.array(emb["coords"]); K = 2
best = counts[0][0]  # most frequent bitstring
for a, (x, y) in enumerate(coords):
    i = a // K
    excited = best[a] == "1"
    ax.add_patch(Circle((x, y), 2.5, facecolor="none", edgecolor=CUST[i], lw=0.9, ls="--", alpha=0.7))
    ax.scatter(x, y, s=70 if excited else 34, color=CUST[i], edgecolor=INK if excited else "white", lw=1.4 if excited else 0.8, zorder=3)
    ax.annotate(f"x{i}{a % K}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6.3, color=INK2)
for i in range(4):
    p, q = coords[2 * i], coords[2 * i + 1]
    ax.plot([p[0], q[0]], [p[1], q[1]], color=CUST[i], lw=2.2, alpha=0.5, zorder=2)
ax.set_aspect("equal"); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
lim = np.abs(coords).max() + 4
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)


ax = axs[2]
ax.set_title("(c) 500 QuTiP shots")
labels, vals, cols = [], [], []
opt_strings = {"01011010", "10100101"}
feasible_strings = {"01011010", "10100101", "01100110", "10011001", "01100101", "10011010", "10010110", "01101001", "01011001", "10100110"}
top = counts[:4]
other = shots - sum(v for _, v in top)
for s, v in top:
    labels.append(s); vals.append(v / shots)
    cols.append(BLUE if s in opt_strings else (AQUA if s in feasible_strings else RED))
labels.append("other"); vals.append(other / shots); cols.append(MUTED)
y = np.arange(len(labels))[::-1]
ax.barh(y, vals, color=cols, height=0.62)
for yi, v in zip(y, vals):
    ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labels, family="monospace", fontsize=7)
ax.set_xlim(0, 1.0); ax.set_xlabel("fraction of shots")
ax.grid(axis="y", visible=False)

save(fig, "fig1_pipeline")

# ---------------------------------------------------------------- Fig 2: embedding + pulse
J = np.array(emb["J"]); U = np.array(emb["U"]); alpha = emb["alpha"]
n = len(J); K = 2
same_cust, same_veh, cross = [], [], []
for a in range(n):
    for b in range(a + 1, n):
        ia, ka, ib, kb = a // K, a % K, b // K, b % K
        pt = (J[a, b], U[a, b] / alpha)
        if ia == ib: same_cust.append(pt)
        elif ka == kb: same_veh.append(pt)
        else: cross.append(pt)
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.3), gridspec_kw=dict(width_ratios=[1, 1.15, 0.95], wspace=0.42))
ax = axs[0]
ax.set_title("(a) couplings, S_seed00")
for pts, col, lab, mk in [(same_cust, INK, "same customer (clique, 2A)", "D"), (same_veh, BLUE, "same vehicle (w_ij)", "o"), (cross, ORANGE, "cross pair (target 0)", "x")]:
    pts = np.array(pts)
    ax.scatter(np.maximum(pts[:, 0], 0.4), pts[:, 1], s=26, color=col, marker=mk, label=lab, zorder=3, lw=1.2)
xx = np.logspace(-0.5, 3.1, 50)
ax.plot(xx, xx, color=MUTED, lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("target J (QUBO units; 0 at 0.4)"); ax.set_ylabel("realised U/α")
ax.legend(loc="upper left", fontsize=6.0, handlelength=1, labelspacing=0.2)


ax = axs[1]
ax.set_title("(b) pulse, S_seed00")
T = 5800; om = 0.75 * 12.566370614359172; df = emb["delta_final"]; di = -max(df, 2 * 12.566370614359172)
t = np.linspace(0, T, 600)
Om = PchipInterpolator([0, 0.15 * T, 0.85 * T, T], [0, om, om, 0])(t)
De = PchipInterpolator([0, 0.15 * T, 0.85 * T, T], [di, di, df, df])(t)
ax.plot(t / 1000, Om, color=BLUE, lw=1.8, label="Ω(t)")
ax.plot(t / 1000, De, color=ORANGE, lw=1.8, label="δ(t)")
ax.axhline(0, color=MUTED, lw=0.6)
ax.axhline(55.1, color=INK2, lw=0.7, ls="--"); ax.text(0.05, 57, "U(5 µm) = 55.1 rad/µs", fontsize=6.3, color=INK2)
ax.set_xlabel("time (µs)"); ax.set_ylabel("rad/µs")
ax.set_ylim(-42, 66); ax.set_xlim(0, 5.8)
ax.legend(loc="lower right", ncol=2)


ax = axs[2]
ax.set_title("(c) scan: P(model optimum)")
Ts = ["2000", "4000", "5800"]; Oms = ["0.5", "0.75", "1.0"]
M = np.array([[0.305, 0.740, 0.739], [0.579, 0.844, 0.797], [0.695, 0.932, 0.865]])
im = ax.imshow(M, cmap="Blues", vmin=0.2, vmax=1.0, aspect="auto")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.7 else INK,
                fontweight="bold" if (i, j) == (2, 1) else "normal")
ax.set_xticks(range(3)); ax.set_xticklabels(Oms); ax.set_yticks(range(3)); ax.set_yticklabels(Ts)
ax.set_xlabel("Ω / Ω_max"); ax.set_ylabel("duration (ns)"); ax.grid(False)

save(fig, "fig2_embedding_pulse")

# ---------------------------------------------------------------- Fig 3: tier-S improvement ladder
rows = D["results_csv"]
def row(inst, method, backend):
    for r in rows:
        if r["instance"] == inst and r["method"] == method and r["backend"] == backend and r["repair"] == "none":
            return r
    raise KeyError((inst, method))
stages = [("random\nbitstrings", "onehot-ub[emb=generic]+random", "random"),
          ("generic\nembedder", "onehot-ub[emb=generic]+qutip", "qutip"),
          ("polygon\nembedder", "onehot-ub[emb=polygons]+qutip|pulse=4000ns_Omega1.0(pre-scan)", "qutip"),
          ("+ tuned\npulse", "onehot-ub[emb=polygons]+qutip", "qutip")]
seeds3 = [("S_seed00", BLUE, -2), ("S_seed01", ORANGE, 7), ("S_seed04", AQUA, -6)]
fig, ax = plt.subplots(figsize=(3.45, 2.25))
ax.set_title("Tier S: P(shot = model optimum) per step")
for inst, col, dy in seeds3:
    vals = [float(row(inst, m, be)["p_reach_model_optimum"]) for _, m, be in stages]
    ax.plot(range(4), vals, "o-", color=col, lw=1.8, ms=4.5, label=inst)
    ax.annotate(f"{vals[-1]:.2f}", (3, vals[-1]), xytext=(6, dy), textcoords="offset points", fontsize=6.6, va="center", color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels([s for s, _, _ in stages], fontsize=6.8)
ax.set_ylim(0, 1.05); ax.set_xlim(-0.25, 3.45); ax.set_ylabel("fraction of 500 shots")
ax.grid(axis="x", visible=False)
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=6.4, handlelength=1.2)
save(fig, "fig3_tierS")

# ---------------------------------------------------------------- Fig 4: tier M/L
recs = D["records"]
Lscan = sorted([(r["omega"], r["feasible"], r["cvrp_gap"]) for r in recs if r["instance"] == "L_seed00" and r["embedder"] == "polygons-cross8" and r["backend"] == "emu-mps"])
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.25), gridspec_kw=dict(width_ratios=[1, 1, 1.25], wspace=0.42))
om = [a for a, _, _ in Lscan]; fe = [b for _, b, _ in Lscan]; gp = [c if c is not None else np.nan for _, _, c in Lscan]
ax = axs[0]
ax.set_title("(a) L_seed00: feasibility")
ax.plot(om, fe, "o-", color=BLUE, lw=1.8, ms=5)
ax.annotate("Ω = 0.15 Ω_max: 0.208", (0.15, 0.208), xytext=(10, -2), textcoords="offset points", fontsize=6.6, color=INK)
ax.axhline(0.008, color=MUTED, lw=0.9, ls="--"); ax.text(0.33, 0.075, "5 µm packing, Ω = 0.75: 0.008", fontsize=6.2, color=INK2)
ax.set_xlabel("Ω / Ω_max"); ax.set_ylabel("capacity-feasible fraction"); ax.set_ylim(0, 0.25); ax.set_xlim(0.05, 0.8)
ax = axs[1]
ax.set_title("(b) L_seed00: best CVRP gap")
ax.plot(om, gp, "o-", color=ORANGE, lw=1.8, ms=5)
ax.annotate("46.9 %", (0.15, 46.9), xytext=(8, -10), textcoords="offset points", fontsize=6.6, color=INK)
ax.set_xlabel("Ω / Ω_max"); ax.set_ylabel("best feasible CVRP gap (%)"); ax.set_ylim(0, 100); ax.set_xlim(0.05, 0.8)
ax.text(0.09, 6, "Ω = 0.10: no decodable shot", fontsize=6.2, color=INK2)
ax = axs[2]
ax.set_title("(c) tiers M and L: feasible fraction")
cats = ["M00", "M01", "M02", "L00", "L01", "L02"]
before = [0.0, 0.388, 0.0, 0.008, 0.004, 0.0]
after = [0.044, 0.388, 0.446, 0.208, 0.140, 0.226]
x = np.arange(6); w = 0.34
ax.bar(x - w / 2, before, w * 0.9, color=AQUA, label="before fix")
ax.bar(x + w / 2, after, w * 0.9, color=BLUE, label="after fix")
for xi, v in zip(x, after):
    ax.text(xi + w / 2, v + 0.012, f"{v:.3f}" if v < 0.1 else f"{v:.2f}", ha="center", fontsize=6.2, color=INK)
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=7); ax.set_ylabel("capacity-feasible fraction"); ax.set_ylim(0, 0.6)
ax.legend(loc="upper left", fontsize=6.3, handlelength=1.2, ncol=1)
ax.grid(axis="x", visible=False)
save(fig, "fig4_tierML")

# ---------------------------------------------------------------- Fig 5: Track B
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.2), gridspec_kw=dict(width_ratios=[0.9, 1.3], wspace=0.35))
ax = axs[0]
ax.set_title("(a) routing encodings: atoms vs cluster size")
ns = np.arange(2, 11)
ax.plot(ns, (ns + 1) ** 2, "o-", color=MUTED, lw=1.4, ms=3.5, label="textbook position, depot encoded: (n+1)²")
ax.plot(ns, ns ** 2, "o-", color=BLUE, lw=1.8, ms=4, label="ours: depot-fixed position, n²")
ax.plot(ns[ns <= 4], (ns[ns <= 4] * (ns[ns <= 4] + 1)) // 2, "s-", color=ORANGE, lw=1.8, ms=4, label="ours: undirected edge, n(n+1)/2 (exact to n = 4)")
ax.plot(ns[ns >= 4], (ns[ns >= 4] * (ns[ns >= 4] + 1)) // 2, "s:", color=ORANGE, lw=1.2, ms=3, alpha=0.6)
ax.axhline(80, color=RED, lw=0.9, ls="--"); ax.text(2.05, 84, "80-atom cap", fontsize=6.5, color=RED)
ax.set_xlabel("customers in cluster, n"); ax.set_ylabel("atoms"); ax.set_ylim(0, 130)
ax.legend(fontsize=6, loc="upper left")
ax = axs[1]
ax.set_title("(b) Track B end-to-end loss split, tier M")
labels = ["M01 best shot", "M01 model opt.", "M02 best shot", "M02 model opt."]
Cref = [368.360, 368.360, 360.859, 360.859]
dec = [0.0, 0.0, 24.758, 24.758]; clu = [34.009, 0.0, 32.971, 0.0]; rou = [38.101, 0.0, 0.0, 0.0]
dec = np.array(dec) / np.array(Cref) * 100; clu = np.array(clu) / np.array(Cref) * 100; rou = np.array(rou) / np.array(Cref) * 100
y = np.arange(4)[::-1]
ax.barh(y, dec, color=AQUA, label="decomposition loss", height=0.6)
ax.barh(y, clu, left=dec, color=BLUE, label="clustering loss (sampler)", height=0.6)
ax.barh(y, rou, left=dec + clu, color=ORANGE, label="routing loss (quantum tour)", height=0.6)
for yi, tot in zip(y, dec + clu + rou):
    ax.text(tot + 0.4, yi, f"{tot:.1f} %", va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("end-to-end gap (% of CVRP reference)"); ax.set_xlim(0, 26); ax.grid(axis="y", visible=False)
ax.legend(loc="center right", fontsize=6.2, bbox_to_anchor=(1.0, 0.62))
save(fig, "fig5_trackB")
print("done")
