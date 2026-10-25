"""End-to-end: instance + formulation -> QUBO -> register -> pulse -> backend -> bitstrings -> metrics.

    python -m cvrp.run --instance instances/S_seed00.json --formulation textbook \
        --backend qutip --shots 500 --seed 0 --out out/

Writes one *run record* JSON per call (``out/<instance>__<formulation>__<method>__seed<k>.json``)
holding everything needed to rebuild the results.csv row (cvrp.results) and the
figures (cvrp.report): formulation + weights, embedding summary (alpha, delta_final,
fidelity numbers, coordinates), pulse parameters, backend + its parameters (bond
dimension for emu-mps), the raw counts, and the §9 metrics computed on feasible
decodes only (cvrp.metrics).  An optional *labelled* repair pass is scored
separately and never mixed into the raw numbers.

Backends
    qutip     pulser_simulation.QutipBackendV2   (exact, <= ~12 atoms)
    emu-sv    emu_sv.SVBackend                    (exact, <= ~20 atoms)
    emu-mps   emu_mps.MPSBackend                  (approximate; --bond-dim recorded)
    numpy-sv  cvrp.svsim                          (in-house exact state vector, <= ~14 atoms, runs anywhere
                                                   with pulser-core; cross-checked against qutip)
    random    uniform random bitstrings           (the brief's random-sampler comparison)
    counts    --counts file.json (bitstring -> count) from somewhere else (e.g. a cloud job)

Everything but the backend call runs in any environment with pulser-core; the
emulators need torch/qutip (the Windows venv).  ``--dry-run`` builds and
validates the full sequence and writes the record without counts.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .common import Instance, bits_to_str, clustering_of, str_to_bits
from .embed import Embedding, embed
from .metrics import Metrics, ModelOptimum, model_optimum, random_counts, random_onehot_counts, score_counts
from .model import get_formulation
from .pulse import PulseParams, build_sequence, default_for, validate

EMULATORS = ("qutip", "emu-sv", "emu-mps", "numpy-sv")
EMBEDDING_CACHE = "embeddings"


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def run_backend(seq, backend: str, shots: int, seed: int, bond_dim: int = 64, dt: float = 10.0,
                precision: float = 1e-5, dt_sv: float = 4.0) -> Dict[str, Any]:
    """Sequence -> {"counts": {...}, "backend_params": {...}}.  Emulator
    adapters live in qopt.run (yesterday's toolkit); only the thin dispatch
    is here so the labels in the record are explicit."""
    from qopt import run as qrun
    info: Dict[str, Any] = {"backend": backend, "shots": shots}
    t0 = time.perf_counter()
    if backend == "qutip":
        counts = qrun.emulate_qutip(seq, shots=shots, sampling_rate=0.1)
        import pulser_simulation
        info.update(sampling_rate=0.1, version=pulser_simulation.__version__)
    elif backend == "emu-sv":
        counts = qrun.emulate_emu_sv(seq, shots=shots, dt=dt)
        import emu_sv
        info.update(dt_ns=dt, version=emu_sv.__version__)
    elif backend == "emu-mps":
        counts, stats = qrun.emulate_emu_mps(seq, shots=shots, dt=dt, precision=precision, max_bond_dim=bond_dim,
                                             return_stats=True)
        import emu_mps
        peak = None
        try:
            last = stats[-1] if isinstance(stats, (list, tuple)) else stats
            peak = last.get("max_bond_dimension") if hasattr(last, "get") else getattr(last, "max_bond_dimension", None)
        except Exception:  # noqa: BLE001
            peak = None
        info.update(dt_ns=dt, precision=precision, bond_dim=bond_dim, peak_bond_dim=peak, version=emu_mps.__version__)
    elif backend == "numpy-sv":
        from . import svsim
        counts, prob = svsim.emulate(seq, shots=shots, seed=seed, dt_ns=dt_sv)
        info.update(dt_ns=dt_sv, seed=seed, version="in-house exact state vector (cvrp.svsim)",
                    exact_top_probabilities={format(int(i), f"0{int(round(np.log2(len(prob))))}b"): float(prob[i])
                                             for i in np.argsort(prob)[::-1][:16]})
    else:
        raise ValueError(f"unknown emulator backend {backend!r}")
    info["wall_s"] = time.perf_counter() - t0
    return {"counts": {str(k): int(v) for k, v in counts.items()}, "backend_params": info}


# --------------------------------------------------------------------------- #
# Labelled repair (generic): greedy descent on the QUBO energy
# --------------------------------------------------------------------------- #
def energy_descent_repair(Q: np.ndarray, offset: float, max_passes: int = 50) -> Callable[[np.ndarray], np.ndarray]:
    """Return a transform x -> x' that flips, one bit at a time, the bit that
    lowers the QUBO energy the most until no single flip helps.  This is a
    classical post-processing step; results that use it are labelled
    ``repair=energy-descent`` and reported next to, never instead of, raw."""
    Q = np.asarray(Q, float)
    diag = np.diag(Q).copy()

    def transform(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float).copy()
        for _ in range(max_passes):
            # dE_i for flipping bit i:  (1-2x_i) * (Q_ii + 2 * sum_{j!=i} Q_ij x_j)
            g = Q @ x
            dE = (1 - 2 * x) * (diag + 2 * (g - diag * x))
            i = int(np.argmin(dE))
            if dE[i] >= -1e-12:
                break
            x[i] = 1 - x[i]
        return x.astype(np.int8)

    return transform


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def parse_kv(s: Optional[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in (s or "").split(","):
        if part.strip():
            k, v = part.split("=")
            out[k.strip()] = float(v)
    return out


def method_label(formulation: str, backend: str, repair: str, embedder: str = "generic") -> str:
    base = f"{formulation}[emb={embedder}]+{backend}"
    return base if repair == "none" else f"{base}+repair:{repair}"


def run_one(inst: Instance, formulation: str, backend: str, shots: int, seed: int,
            weights: Optional[Dict[str, float]] = None, pulse_overrides: Optional[Dict[str, Any]] = None,
            embed_kw: Optional[Dict[str, Any]] = None, repair: str = "none", bond_dim: int = 64,
            counts: Optional[Dict[str, int]] = None, dry_run: bool = False,
            mopt: Optional[ModelOptimum] = None, verbose: bool = True) -> Dict[str, Any]:
    t_start = time.perf_counter()
    f = get_formulation(formulation, inst, **(weights or {}))
    J, h = f.couplings(), f.linear()

    # --- embedding (raises on negative couplings) ------------------------- #
    embed_kw = dict(embed_kw or {})
    embedder = embed_kw.pop("embedder", "generic")
    cache_dir = embed_kw.pop("cache_dir", EMBEDDING_CACHE)
    # The position optimiser is seeded but not bit-identical across platforms (BLAS / scipy versions), so the
    # register of every headline row is cached under embeddings/ and committed; a run reuses it when the
    # instance, formulation, weights, embedder and seed match.  Delete the file to force a recomputation.
    wkey = "_".join(f"{k}={v:.6g}" for k, v in sorted(f.weights.items()))
    cache_file = (Path(cache_dir) / f"{inst.name}__{formulation}__{wkey}__{embedder}__seed{seed}.json") if cache_dir else None
    emb: Optional[Embedding] = None
    if cache_file is not None and cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if np.allclose(np.asarray(cached["J_target"]), J, rtol=1e-9, atol=1e-9) and np.allclose(np.asarray(cached["h"]), h):
            emb = Embedding.from_dict(cached)
            emb.fidelity.notes.append(f"register loaded from cache {cache_file.name}")
    if emb is None:
        if embedder == "generic":
            emb = embed(J, h, seed=seed, **embed_kw)
        elif embedder == "polygons" or embedder.startswith("polygons-cross"):
            from .embed_onehot import embed_onehot
            if f.n_vars != inst.N * inst.K:
                raise ValueError("embedder=polygons needs a one-hot formulation with N*K variables")
            if embedder.startswith("polygons-cross"):
                # 'polygons-cross8' = polygon embedding with atoms of different polygons kept >= 8 um apart
                # (DERIVATIONS 6.7 geometry experiment); its own cache key and method label
                embed_kw["cross_min_um"] = float(embedder[len("polygons-cross"):])
            emb = embed_onehot(J, h, inst.N, inst.K, seed=seed, **embed_kw)
        else:
            raise ValueError(f"unknown embedder {embedder!r} (generic | polygons | polygons-cross<um>)")
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(emb.to_dict(), indent=1, default=_json_default))
    fid = emb.fidelity
    device_faithful = bool(fid.linear_terms_uniform and not fid.limit_violations)


    # --- pulse + Pulser-validated sequence --------------------------------- #
    pulse_overrides = dict(pulse_overrides or {})
    ds = pulse_overrides.pop("delta_final_scale", None)
    if ds is not None:                       # scan knob: scale the embedding's final detuning
        pulse_overrides["delta_final"] = float(emb.delta_final) * float(ds)
    p: PulseParams = default_for(emb, **pulse_overrides)
    seq = build_sequence(emb, p)
    problems = validate(seq)
    if problems:
        raise RuntimeError(f"sequence failed validation: {problems}")

    # What the realised register minimises, classically, with the pulse's final detuning: is the device's own argmin the model optimum?
    realized: Dict[str, Any] = {}
    if f.n_vars <= 20:
        from .common import all_bitstrings
        X = all_bitstrings(f.n_vars).astype(float)
        Edev = -p.delta_final * X.sum(1) + 0.5 * np.einsum("bi,ij,bj->b", X, emb.U, X)
        order = np.argsort(Edev)
        feas_mask = np.array([f.feasible(f.decode(x.astype(np.int8))) for x in X])
        if mopt is None:
            mopt = model_optimum(f)
        objs = np.array([f.objective(f.decode(x.astype(np.int8))) if fm else np.inf for x, fm in zip(X, feas_mask)])
        i0 = int(order[0])
        best_feas_E = float(Edev[feas_mask].min()) if feas_mask.any() else None
        best_infeas_E = float(Edev[~feas_mask].min()) if (~feas_mask).any() else None
        realized = {
            "argmin_bits": bits_to_str(X[i0].astype(int)),
            "argmin_feasible": bool(feas_mask[i0]),
            "argmin_is_model_optimum": bool(feas_mask[i0] and abs(objs[i0] - mopt.value) <= 1e-9),
            "energy_gap_infeasible_minus_feasible": (best_infeas_E - best_feas_E) if (best_feas_E is not None and best_infeas_E is not None) else None,
            "rank_of_model_optimum": int(np.where(np.isin(order, np.where(np.abs(objs - mopt.value) <= 1e-9)[0]))[0][0]) + 1
                                      if np.any(np.abs(objs - mopt.value) <= 1e-9) else None,
        }
    elif f.n_vars == inst.N * inst.K:
        # Too many bitstrings to enumerate: check the one-hot subspace (all K^N strings, or a seeded sample
        # plus the model optimum) and the single-flip defects of the lowest one-hot strings.
        if mopt is None:
            mopt = model_optimum(f)
        N, K = inst.N, inst.K
        rng = np.random.default_rng(seed)
        if K ** N <= 65536:
            A_mat = np.array(np.unravel_index(np.arange(K ** N), (K,) * N)).T
        else:
            A_mat = rng.integers(0, K, size=(20000, N))
        A_mat = np.vstack([A_mat, np.asarray(mopt.assignment)[None, :]])
        Xoh = np.zeros((len(A_mat), f.n_vars))
        Xoh[np.arange(len(A_mat))[:, None], np.arange(N)[None, :] * K + A_mat] = 1.0
        Edev = -p.delta_final * N + 0.5 * np.einsum("bi,ij,bj->b", Xoh, emb.U, Xoh)
        order = np.argsort(Edev)
        a0 = tuple(int(v) for v in A_mat[order[0]])
        feas0 = inst.capacity_ok(a0)
        obj0 = f.objective(a0)
        # single-flip defects of the 50 lowest one-hot strings
        low = Xoh[order[:50]]
        Xd = np.repeat(low, f.n_vars, axis=0)
        flip = np.tile(np.arange(f.n_vars), len(low))
        Xd[np.arange(len(Xd)), flip] = 1 - Xd[np.arange(len(Xd)), flip]
        Ed = -p.delta_final * Xd.sum(1) + 0.5 * np.einsum("bi,ij,bj->b", Xd, emb.U, Xd)
        min_defect_gap = float((Ed - np.repeat(Edev[order[:50]], f.n_vars)).min())
        opt_rows = np.where([clustering_of(tuple(int(v) for v in a)) in set(mopt.clusterings) for a in A_mat])[0]
        realized = {
            "subspace": "one-hot strings" + (" (all)" if K ** N <= 65536 else " (20000 sampled + model optimum)"),
            "argmin_onehot_assignment": list(a0),
            "argmin_feasible": bool(feas0),
            "argmin_is_model_optimum": bool(feas0 and abs(obj0 - mopt.value) <= 1e-9),
            "rank_of_model_optimum_among_onehot": int(np.where(np.isin(order, opt_rows))[0][0]) + 1 if len(opt_rows) else None,
            "min_single_flip_defect_gap": min_defect_gap,
            "energy_gap_infeasible_minus_feasible": None,
            "rank_of_model_optimum": int(np.where(np.isin(order, opt_rows))[0][0]) + 1 if len(opt_rows) else None,
        }

    rec: Dict[str, Any] = {
        "realized_register": realized,
        "instance": inst.name, "tier": inst.tier, "instance_seed": inst.seed,
        "formulation": f.describe(), "n_vars": f.n_vars, "atoms": emb.n,
        "method": method_label(formulation, backend, repair, embedder), "backend": backend, "embedder": embedder, "seed": seed, "shots": shots,
        "repair": repair, "dmm": False, "device": emb.device, "device_faithful": device_faithful,
        "embedding": emb.to_dict(), "pulse": p.to_dict(), "sequence_duration_ns": seq.get_duration(),
        "versions": {"python": platform.python_version(), "pulser": __import__("pulser").__version__},
    }
    if verbose:
        print(f"[{inst.name}] {formulation}: {f.n_vars} vars -> {emb.n} atoms; alpha={emb.alpha:.3f} delta_final={emb.delta_final:.3f}")
        print(f"   fidelity rel_frob={fid.rel_frobenius_offdiag:.3f} worst pair={fid.worst_pair_rel_error:.3f} uniform h={fid.linear_terms_uniform} violations={fid.limit_violations}")
        print(f"   pulse {p.duration_ns} ns, Omega={p.omega:.3f}, delta {p.delta_init:.2f} -> {p.delta_final:.2f}; sequence validated")
        if realized:
            print(f"   realised register argmin feasible={realized['argmin_feasible']} is model optimum={realized['argmin_is_model_optimum']} "
                  f"(model optimum at device-energy rank {realized['rank_of_model_optimum']}, gap infeas-feas {realized['energy_gap_infeasible_minus_feasible']}, "
                  f"min single-flip defect gap {realized.get('min_single_flip_defect_gap')})")
    if dry_run:
        rec["counts"] = None
        rec["wall_s"] = time.perf_counter() - t_start
        return rec

    # --- sample ------------------------------------------------------------ #
    if counts is not None:
        rec["backend_params"] = {"backend": "counts", "shots": sum(counts.values())}
    elif backend == "random":
        counts = random_counts(f.n_vars, shots, seed)
        rec["backend_params"] = {"backend": "random", "shots": shots, "seed": seed}
    elif backend == "random-onehot":
        counts = random_onehot_counts(f, shots, seed)
        rec["backend_params"] = {"backend": "random-onehot", "shots": shots, "seed": seed,
                                 "note": "uniform random assignments encoded one-hot (assignment-valid control)"}
    else:
        out = run_backend(seq, backend, shots, seed, bond_dim=bond_dim)
        counts, rec["backend_params"] = out["counts"], out["backend_params"]
    rec["counts"] = counts

    # --- score (feasible decodes only) ------------------------------------- #
    if mopt is None:
        mopt = model_optimum(f)
    m_raw, _ = score_counts(counts, f, inst, mopt=mopt)
    rec["metrics_raw"] = m_raw.to_dict()
    if repair != "none":
        if repair == "energy-descent":
            tr = energy_descent_repair(f.Q(), f.offset)
        elif repair == "capacity-local":
            from .repair import capacity_local_repair
            tr = capacity_local_repair(f)
        else:
            raise ValueError("unknown repair (none | energy-descent | capacity-local)")
        m_rep, _ = score_counts(counts, f, inst, mopt=mopt, transform=tr)
        rec["metrics_repaired"] = m_rep.to_dict()
        if hasattr(tr, "stats"):
            rec["repair_stats"] = tr.stats()
    rec["wall_s"] = time.perf_counter() - t_start
    if verbose:
        print("   raw:      " + m_raw.summary().replace("\n", "\n             "))
        if repair != "none":
            print("   repaired: " + m_rep.summary().replace("\n", "\n             "))
    return rec


def record_path(out_dir: Path, rec: Dict[str, Any]) -> Path:
    # '|' (tag separator in the method label) is illegal in Windows file names -> '~' in the file name only
    name = f"{rec['instance']}__{rec['method'].replace('+', '_').replace(':', '-').replace('|', '~')}__seed{rec['seed']}.json"
    return out_dir / name


def save_record(rec: Dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pth = record_path(out_dir, rec)
    pth.write_text(json.dumps(rec, indent=1, default=_json_default))
    return pth


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--formulation", required=True)
    ap.add_argument("--backend", default="random", help="qutip | emu-sv | emu-mps | numpy-sv | random | random-onehot")
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="run seed (embedding starts, random sampler)")
    ap.add_argument("--weights", default="", help="penalty weights, e.g. A=12,B=3 (default: formulation's rule)")
    ap.add_argument("--pulse", default="", help="pulse overrides, e.g. duration_ns=3000,omega_scale=0.8")
    ap.add_argument("--slow", nargs=3, type=float, metavar=("CENTRE", "WIDTH", "FACTOR"), default=None)
    ap.add_argument("--bond-dim", type=int, default=64)
    ap.add_argument("--embedder", default="generic", help="generic | polygons (structure-aware one-hot embedding, cvrp.embed_onehot) | polygons-cross<um> (same, atoms of different polygons >= <um> apart)")
    ap.add_argument("--embedding-cache", default=EMBEDDING_CACHE, help="directory of cached registers ('' to disable)")
    ap.add_argument("--repair", default="none", choices=("none", "energy-descent", "capacity-local"),
                    help="classical post-processing (labelled separately)")
    ap.add_argument("--counts", default=None, help="JSON file bitstring->count to score instead of sampling")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", default="", help="appended to the method label (e.g. a scan point) so rows stay unique")
    a = ap.parse_args(argv)

    inst = Instance.load(a.instance)
    pulse_over: Dict[str, Any] = parse_kv(a.pulse)
    if "duration_ns" in pulse_over:
        pulse_over["duration_ns"] = int(pulse_over["duration_ns"])
    if a.slow:
        pulse_over["slow_region"] = tuple(a.slow)
    counts = json.loads(Path(a.counts).read_text()) if a.counts else None
    rec = run_one(inst, a.formulation, a.backend, a.shots, a.seed, weights=parse_kv(a.weights),
                  pulse_overrides=pulse_over, repair=a.repair, bond_dim=a.bond_dim, counts=counts, dry_run=a.dry_run,
                  embed_kw={"embedder": a.embedder, "cache_dir": a.embedding_cache or None})
    if a.tag:
        rec["method"] += "|" + a.tag
    pth = save_record(rec, a.out)
    print(f"record -> {pth}")


if __name__ == "__main__":
    main()
