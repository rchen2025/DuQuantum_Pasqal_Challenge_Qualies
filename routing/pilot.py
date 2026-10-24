"""Four-customer pilot: both routing encodings -> generic embedding -> Part A pulse -> exact emulation.

Three things are reported SEPARATELY, so formulation, geometry and dynamics failures can be
told apart (handoff spec, "Physical execution and decision gates"):

1. ``classical_targets``   the QUBO's own ground states (exhaustive) vs ``cvrp.common.tsp_exact``;
2. ``device_energies``     the zero-drive classical energies  E_dev(x) = -delta_final sum x + sum U_ij x_i x_j
                           of the *embedded* register for all 2^n strings (rank of the optimal tour,
                           whether the device argmin decodes, correlation with the target QUBO);
3. ``samples``             bitstrings sampled from the exactly emulated pulse (``cvrp.svsim``), scored
                           by raw feasibility, P(optimal tour), best feasible gap - against uniform random
                           bitstrings AND uniform random valid tours (1/12 per shot for a unique n = 4 optimum).

Everything is emulation on standalone fixtures.  Nothing here ran on hardware, and nothing here
is an end-to-end Track B run on Part A clusterings.  Records go to ``out/routing/`` (gitignored).
"""
from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cvrp.common import all_bitstrings, bits_to_str, str_to_bits, tsp_exact
from cvrp.embed import Embedding, embed
from cvrp.pulse import PulseParams, build_sequence, default_for, final_detuning_from_sequence, validate
from cvrp import svsim

from .check import optimal_tours
from .encodings import Encoding, TIE_TOL, build_edge, build_position, canonical_tour, tour_cost
from .fixtures import RoutingFixture, fixture

DEFAULT_OUT = Path("out/routing")


@dataclass
class PilotConfig:
    fixture: str = "quad4"
    shots: int = 1000
    seed: int = 0
    n_starts: int = 8
    dt_ns: float = 4.0
    position_emulation: str = "auto"        # "auto" (probe runtime first) | "yes" | "no"
    position_time_budget_s: float = 600.0
    embed_seeds: int = 6                    # restart sweep: embed with seeds 0..embed_seeds-1, keep the best FIDELITY
    out_dir: str = str(DEFAULT_OUT)


# --------------------------------------------------------------------------- #
# 1. classical targets
# --------------------------------------------------------------------------- #
def classical_targets(enc: Encoding) -> Dict[str, Any]:
    D, n = enc.D, enc.n_customers
    C_opt, opt = optimal_tours(D)
    C_hk, order_hk = tsp_exact(D, list(range(1, n + 1)))
    X = all_bitstrings(enc.n_vars)
    E = enc.energies(X)
    gs = np.where(E <= E.min() + TIE_TOL)[0]
    gs_bits = [bits_to_str(X[k]) for k in gs]
    gs_tours = [enc.decode(X[k]) for k in gs]
    n_valid = sum(1 for k in range(len(X)) if enc.decode(X[k]) is not None)
    return {
        "C_opt_bruteforce": C_opt, "C_opt_held_karp": C_hk, "held_karp_order": [int(c) for c in order_hk],
        "optimal_tours_undirected": [list(t) for t in opt], "n_optimal_undirected_tours": len(opt),
        "ground_energy": float(E.min()), "ground_states": gs_bits,
        "ground_states_decode": [list(t) if t is not None else None for t in gs_tours],
        "all_ground_states_optimal": all(t is not None and canonical_tour(t) in set(opt) for t in gs_tours),
        "n_strings": int(len(X)), "n_valid_tour_strings": n_valid,
        "C_ub_nearest_neighbour": enc.C_ub, "A": enc.A, "nn_tour": list(enc.nn_tour),
        "A_over_C_opt": enc.A / C_opt,
    }


# --------------------------------------------------------------------------- #
# embedding + pulse
# --------------------------------------------------------------------------- #
def embed_and_pulse(enc: Encoding, seed: int, n_starts: int, device: Any = None):
    """Generic embedder (refuses J < 0; J = 0 pairs are approximated by far atoms and reported as
    ``max_spurious_coupling``), then the Part A default pulse, Pulser-validated."""
    if device is None:
        from pulser.devices import AnalogDevice
        device = AnalogDevice
    t0 = time.time()
    emb = embed(enc.J, enc.h, device=device, seed=seed, n_starts=n_starts, allow_zero=True)
    t_embed = time.time() - t0
    params = default_for(emb, device=device)
    seq = build_sequence(emb, params, device=device, auto_layout=True)
    viol = validate(seq, device)
    dT = final_detuning_from_sequence(seq)
    ch = device.channels["rydberg_global"]
    info = {
        "embed_seconds": t_embed, "embedder": "cvrp.embed.embed (generic)", "n_starts": n_starts, "seed": seed,
        "fidelity": emb.fidelity.to_dict(), "alpha": emb.alpha, "delta_final": emb.delta_final,
        "alpha_times_A": emb.alpha * enc.A, "pulse": params.to_dict(),
        "sequence": {"duration_ns": seq.get_duration(), "validation_violations": viol,
                     "delta_T_readback": dT, "layout_attached": seq.register.layout is not None,
                     "n_atoms": len(seq.register.qubit_ids), "qubit_order": list(seq.register.qubit_ids)},
        "device": {"name": device.name, "max_atom_num": device.max_atom_num, "min_atom_distance": device.min_atom_distance,
                   "max_radial_distance": device.max_radial_distance, "max_sequence_duration": device.max_sequence_duration,
                   "omega_max": ch.max_amp, "max_abs_detuning": ch.max_abs_detuning, "C6": float(device.interaction_coeff)},
        "coords_um": emb.coords.tolist(),
    }
    return emb, params, seq, info


# --------------------------------------------------------------------------- #
# 2. zero-drive device energies of the embedded register
# --------------------------------------------------------------------------- #
def device_energy_analysis(enc: Encoding, emb: Embedding, seq, targets: Dict[str, Any]) -> Dict[str, Any]:
    n = enc.n_vars
    E_dev = svsim.final_energies(seq)              # index k has bit i = MSB-first, same as bitstring order
    X = all_bitstrings(n)                           # column i = bit i, index k = sum x_i 2^i (LSB-first!)
    # bring both to the same labelling: string s -> svsim index int(s, 2)
    strings = [bits_to_str(x) for x in X]
    idx_dev = np.array([int(s, 2) for s in strings])
    E_dev_s = E_dev[idx_dev]                        # device energy of string strings[k]
    E_tgt = enc.energies(X)
    order = np.argsort(E_dev_s, kind="stable")
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    valid = np.array([enc.decode(x) is not None for x in X])
    costs = np.array([tour_cost(enc.D, enc.decode(x)) if v else np.nan for x, v in zip(X, valid)])
    opt_set = {tuple(t) for t in targets["optimal_tours_undirected"]}
    is_opt = np.array([v and canonical_tour(enc.decode(x)) in opt_set for x, v in zip(X, valid)])
    k_argmin = int(order[0])
    dec_argmin = enc.decode(X[k_argmin])
    best_valid_rank = int(rank[valid].min())
    corr = float(np.corrcoef(E_dev_s / emb.alpha, E_tgt)[0, 1])
    # lowest device-energy strings
    top = [{"bits": strings[k], "E_dev": float(E_dev_s[k]), "E_dev_over_alpha": float(E_dev_s[k] / emb.alpha),
            "E_target": float(E_tgt[k]), "tour": list(enc.decode(X[k])) if valid[k] else None,
            "cost": None if not valid[k] else float(costs[k])} for k in order[:10]]
    return {
        "device_argmin": strings[k_argmin], "device_argmin_decodes": dec_argmin is not None,
        "device_argmin_tour": list(dec_argmin) if dec_argmin is not None else None,
        "device_argmin_cost": float(costs[k_argmin]) if valid[k_argmin] else None,
        "device_argmin_is_optimal": bool(is_opt[k_argmin]),
        "rank_of_optimal_tour_strings": sorted(int(r) for r in rank[is_opt]),
        "best_rank_of_any_valid_tour": best_valid_rank,
        "n_valid_among_lowest_n_valid": int(valid[order[: int(valid.sum())]].sum()),
        "n_valid_total": int(valid.sum()),
        "corr_device_vs_target_energy_all_strings": corr,
        "E_dev_min": float(E_dev_s.min()), "E_dev_of_optimal_min": float(E_dev_s[is_opt].min()),
        "E_dev_gap_optimal_minus_min_rad_per_us": float(E_dev_s[is_opt].min() - E_dev_s.min()),
        "lowest_device_strings": top,
    }


# --------------------------------------------------------------------------- #
# 3. sample metrics
# --------------------------------------------------------------------------- #
def score_counts(enc: Encoding, counts: Dict[str, int], targets: Dict[str, Any]) -> Dict[str, Any]:
    """Raw-decode scoring.  Infeasible shots never contribute a cost; no mean cost is reported."""
    C_opt = targets["C_opt_bruteforce"]
    opt_set = {tuple(t) for t in targets["optimal_tours_undirected"]}
    shots = sum(counts.values())
    n_feas = n_opt = 0
    best_cost: Optional[float] = None
    best_tour: Optional[Tuple[int, ...]] = None
    rows = []
    for s, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        t = enc.decode(str_to_bits(s))
        cost = tour_cost(enc.D, t) if t is not None else None
        rows.append({"bits": s, "count": c, "tour": list(t) if t else None, "cost": cost, "energy": enc.energy_of_str(s)})
        if t is None:
            continue
        n_feas += c
        if canonical_tour(t) in opt_set:
            n_opt += c
        if best_cost is None or cost < best_cost:
            best_cost, best_tour = cost, t
    return {
        "shots": shots, "n_distinct": len(counts),
        "raw_feasible_rate": n_feas / shots, "p_optimal_tour": n_opt / shots,
        "best_feasible_cost": best_cost, "best_feasible_tour": list(best_tour) if best_tour else None,
        "best_feasible_gap_percent": None if best_cost is None else 100.0 * (best_cost - C_opt) / C_opt,
        "top_strings": rows[:12],
    }


def random_bitstring_counts(n_vars: int, shots: int, seed: int) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(shots, n_vars))
    out: Dict[str, int] = {}
    for row in bits:
        s = bits_to_str(row)
        out[s] = out.get(s, 0) + 1
    return out


def random_tour_counts(enc: Encoding, shots: int, seed: int) -> Dict[str, int]:
    """Uniform random valid tours (uniform over directed permutations = uniform over undirected tours)."""
    rng = np.random.default_rng(seed)
    out: Dict[str, int] = {}
    for _ in range(shots):
        t = tuple(int(v) for v in rng.permutation(np.arange(1, enc.n_customers + 1)))
        s = bits_to_str(enc.encode(t))
        out[s] = out.get(s, 0) + 1
    return out


def analytic_random_tour_baseline(enc: Encoding, targets: Dict[str, Any]) -> Dict[str, Any]:
    n = enc.n_customers
    n_und = math.factorial(n) // 2 if n >= 3 else 1
    n_opt = targets["n_optimal_undirected_tours"]
    return {"n_undirected_tours": n_und, "n_optimal_undirected_tours": n_opt, "p_optimal_per_shot": n_opt / n_und,
            "raw_feasible_rate": 1.0, "best_feasible_gap_percent": 0.0,
            "note": "a uniform random valid tour is feasible by construction; P(optimal) = #optimal / #tours"}


# --------------------------------------------------------------------------- #
# runtime probe for the 16-atom exact run
# --------------------------------------------------------------------------- #
def probe_evolve_seconds(seq, dt_ns: float, n_probe_steps: int = 10) -> float:
    """Time ``n_probe_steps`` Krylov steps of the actual Hamiltonian and extrapolate to the whole pulse.
    Steps late in the sweep (more entangled psi) can be slower, so this is a lower estimate."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply
    amp, det, _ = svsim.samples_from_sequence(seq)
    U = svsim.interaction_matrix(seq)
    n = U.shape[0]
    dim = 1 << n
    n_exc, inter = svsim._diagonal_terms(n, U)
    rows, cols = svsim._sigma_x_sum(n)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(dim, dim))
    rng = np.random.default_rng(0)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)     # generic (not a basis) state: realistic Krylov cost
    psi /= np.linalg.norm(psi)
    step = max(1, int(round(dt_ns)))
    mid = len(amp) // 2
    t0 = time.time()
    for _ in range(n_probe_steps):
        H = 0.5 * float(amp[mid]) * X + sp.diags(-float(det[mid]) * n_exc + inter)
        psi = expm_multiply(-1j * step * 1e-3 * H, psi)
    per = (time.time() - t0) / n_probe_steps
    return per * math.ceil(len(amp) / step)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("pulser", "numpy", "scipy"):
        try:
            out[m] = __import__(m).__version__
        except Exception:                         # noqa: BLE001
            out[m] = "n/a"
    return out


def restart_sweep(enc: Encoding, targets: Dict[str, Any], cfg: PilotConfig) -> Tuple[int, List[Dict[str, Any]]]:
    """Embed with several optimiser seeds; SELECT by embedding fidelity (rel. Frobenius) only.
    The rank of the optimal tour under each register is recorded as a diagnostic, never used
    for the selection (that would tune the geometry against the answer)."""
    rows = []
    for s in range(cfg.embed_seeds):
        emb, params, seq, info = embed_and_pulse(enc, s, cfg.n_starts)
        dev = device_energy_analysis(enc, emb, seq, targets)
        rows.append({"seed": s, "rel_frobenius_offdiag": info["fidelity"]["rel_frobenius_offdiag"],
                     "worst_pair_rel_error": info["fidelity"]["worst_pair_rel_error"],
                     "max_spurious_coupling": info["fidelity"]["max_spurious_coupling"], "alpha": emb.alpha,
                     "diag_rank_of_optimal": min(dev["rank_of_optimal_tour_strings"]),
                     "diag_device_argmin_decodes": dev["device_argmin_decodes"],
                     "diag_corr_device_vs_target": dev["corr_device_vs_target_energy_all_strings"]})
    best = min(rows, key=lambda r: r["rel_frobenius_offdiag"])["seed"]
    return best, rows


def run_encoding(fx: RoutingFixture, enc: Encoding, cfg: PilotConfig, out_dir: Path,
                 do_emulate: bool) -> Dict[str, Any]:
    t_start = time.time()
    targets = classical_targets(enc)
    best_seed, sweep = restart_sweep(enc, targets, cfg)
    emb, params, seq, info = embed_and_pulse(enc, best_seed, cfg.n_starts)
    dev = device_energy_analysis(enc, emb, seq, targets)
    rec: Dict[str, Any] = {
        "label": "standalone routing pilot (emulation; no hardware; not an end-to-end Track B run)",
        "fixture": fx.to_dict(), "encoding": enc.describe(), "classical_targets": targets,
        "embedding": {**info, "restart_sweep": sweep, "selected_seed": best_seed,
                      "selection_rule": "lowest rel_frobenius_offdiag over embed seeds (optimum never used)"},
        "device_energies": dev, "versions": _versions(), "config": asdict(cfg),
    }
    # legal register + sequence, always saved (also what a Windows emu-sv job would consume)
    seq_path = out_dir / f"seq_{fx.name}_{enc.name}.json"
    seq_path.write_text(seq.to_abstract_repr())
    reg_path = out_dir / f"register_{fx.name}_{enc.name}.json"
    reg_path.write_text(json.dumps({"var_names": enc.var_names, **emb.to_dict()}, indent=1, default=_json_default))
    rec["paths"] = {"sequence_abstract_repr": str(seq_path), "register": str(reg_path)}

    # baselines (always cheap)
    rand_bits = score_counts(enc, random_bitstring_counts(enc.n_vars, cfg.shots, cfg.seed + 1), targets)
    rand_tours = score_counts(enc, random_tour_counts(enc, cfg.shots, cfg.seed + 2), targets)
    rec["baselines"] = {"uniform_random_bitstrings": rand_bits, "uniform_random_valid_tours_sampled": rand_tours,
                        "uniform_random_valid_tours_analytic": analytic_random_tour_baseline(enc, targets)}

    emu: Dict[str, Any]
    if do_emulate:
        t0 = time.time()
        counts, probs = svsim.emulate(seq, shots=cfg.shots, seed=cfg.seed, dt_ns=cfg.dt_ns)
        t_emu = time.time() - t0
        scored = score_counts(enc, counts, targets)
        # exact probabilities (no sampling noise) of the same quantities
        n = enc.n_vars
        p_feas = p_opt = 0.0
        opt_set = {tuple(t) for t in targets["optimal_tours_undirected"]}
        for k in range(len(probs)):
            if probs[k] < 1e-12:
                continue
            t = enc.decode(str_to_bits(format(k, f"0{n}b")))
            if t is not None:
                p_feas += probs[k]
                if canonical_tour(t) in opt_set:
                    p_opt += probs[k]
        emu = {"backend": "cvrp.svsim (in-house exact state vector, piecewise-constant 4 ns slices)",
               "seconds": t_emu, "dt_ns": cfg.dt_ns, "shots": cfg.shots, "seed": cfg.seed,
               "exact_p_feasible": p_feas, "exact_p_optimal": p_opt, **scored}
        rec["raw_counts"] = counts
    else:
        emu = {"backend": None, "status": "NOT RUN here: needs emu-sv (Windows venv) - sequence saved at "
                                           + str(seq_path)}
    rec["samples"] = emu
    rec["runtime_seconds"] = time.time() - t_start
    return rec


def summary_table(records: List[Dict[str, Any]]) -> str:
    lines = ["| encoding | atoms | target GS optimal | rel-Frob | spurious J (QUBO) | device argmin decodes | rank(opt) | "
             "raw feasible | P(opt) | best gap % | rand-bits feasible | rand-bits P(opt) | rand-tour P(opt) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        fid = r["embedding"]["fidelity"]
        dev = r["device_energies"]
        s = r["samples"]
        b = r["baselines"]
        rb = b["uniform_random_bitstrings"]
        rt = b["uniform_random_valid_tours_analytic"]
        if s.get("backend"):
            feas, popt = f"{s['raw_feasible_rate']:.3f}", f"{s['p_optimal_tour']:.3f}"
            gap = "none" if s["best_feasible_gap_percent"] is None else f"{s['best_feasible_gap_percent']:.1f}"
        else:
            feas = popt = gap = "not run"
        spur = fid["max_spurious_coupling"]
        lines.append(
            f"| {r['encoding']['encoding']} | {r['encoding']['n_vars']} | {r['classical_targets']['all_ground_states_optimal']} | "
            f"{fid['rel_frobenius_offdiag']:.3f} | {spur:.0f} / 2A={2 * r['encoding']['A']:.0f} | "
            f"{dev['device_argmin_decodes']} | {min(dev['rank_of_optimal_tour_strings'])} of {r['classical_targets']['n_strings']} | "
            f"{feas} | {popt} | {gap} | {rb['raw_feasible_rate']:.3f} | {rb['p_optimal_tour']:.3f} | {rt['p_optimal_per_shot']:.3f} |")
    return "\n".join(lines)


def run_pilot(cfg: PilotConfig) -> Tuple[List[Dict[str, Any]], Path]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fx = fixture(cfg.fixture)
    D = fx.dist_matrix()
    records: List[Dict[str, Any]] = []
    encs: List[Encoding] = [build_edge(D), build_position(D)]
    for enc in encs:
        if enc.name == "edge":
            do_emu = True
        elif cfg.position_emulation == "yes":
            do_emu = True
        elif cfg.position_emulation == "no":
            do_emu = False
        else:
            emb, params, seq, _ = embed_and_pulse(enc, cfg.seed, cfg.n_starts)
            est = probe_evolve_seconds(seq, cfg.dt_ns)
            print(f"[pilot] {enc.name}: {enc.n_vars} atoms, estimated exact-emulation time {est:.0f} s "
                  f"(budget {cfg.position_time_budget_s:.0f} s)")
            do_emu = est < cfg.position_time_budget_s
        print(f"[pilot] running {enc.name} ({enc.n_vars} atoms), emulate={do_emu}")
        rec = run_encoding(fx, enc, cfg, out_dir, do_emu)
        path = out_dir / f"pilot_{fx.name}_{enc.name}.json"
        path.write_text(json.dumps(rec, indent=1, default=_json_default))
        rec["paths"]["record"] = str(path)
        records.append(rec)
        print(f"[pilot] wrote {path}  ({rec['runtime_seconds']:.1f} s)")
    md = out_dir / f"pilot_{fx.name}.md"
    header = (f"# Routing pilot on fixture `{fx.name}` (n = {fx.n}, {cfg.shots} shots, seed {cfg.seed})\n\n"
              "Standalone routing fixture, exact state-vector emulation (`cvrp.svsim`), generic embedder "
              "(`cvrp.embed.embed`), Part A default pulse (`cvrp.pulse.default_for`).  No hardware, no Part A clusters.\n\n")
    md.write_text(header + summary_table(records) + "\n\nRecords: " + ", ".join(f"`{r['paths']['record']}`" for r in records) + "\n")
    print(f"[pilot] wrote {md}")
    return records, md
