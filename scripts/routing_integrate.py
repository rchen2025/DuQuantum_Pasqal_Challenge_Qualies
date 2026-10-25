"""Track B slice 2: route genuine Part A clusterings through ``routing.solve_cluster`` (emulation only).

    PYTHONPATH=/home/claude/_pydeps:. python3 scripts/routing_integrate.py [--records out/m_tier/M_seed01__onehot-ub[emb=polygons]_emu-mps__seed0.json ...]
        [--instances instances] [--shots 1000] [--seed 0] [--embed-seeds 6] [--position-emulation yes|no] [--out out/routing]

For every Part A run record (JSON with ``counts`` {bitstring: count} and ``formulation`` weights):

1. decode every shot with the record's own formulation (``cvrp.model.get_formulation``), keep the
   feasible ones, pick the best by MODEL objective (ties -> lexicographically smallest assignment;
   the number of tied assignments is reported) - the clustering Part A actually returned;
2. compute the exact model-optimal clustering (``cvrp.metrics.model_optimum``) as a second input;
3. freeze each clustering, validate it (disjoint, covering, <= K, capacity) and route every cluster
   with ``solve_cluster``: n <= 3 exact classical, n = 4 both quantum encodings (edge 10 atoms,
   position 16 atoms; cvrp.svsim), n >= 5 exact classical fallback (flagged);
4. report per cluster and end to end, with the brief's three-term split
       C_end - C_ref = (C_dec - C_ref) + (C_clust - C_dec) + (C_end - C_clust)
   C_ref = instance reference, C_dec = model optimum routed exactly, C_clust = exact routing of the
   selected clustering, C_end = the routes actually returned (fallbacks included; they are excluded
   from the quantum routing gaps).

Identical clusters (same instance, same customer set) are routed once per run and the result is
reused (deterministic seed); the table says where.  Writes ``<out>/integration.md`` and
``<out>/integration_<source>.json`` + ``<out>/integration.json``.  Nothing here ran on hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvrp.common import Instance, clustering_of, gap_percent, route_exactly, str_to_bits   # noqa: E402
from cvrp.metrics import model_optimum                                                        # noqa: E402
from cvrp.model import get_formulation                                                        # noqa: E402
from routing.pilot import _json_default                                                       # noqa: E402
from routing.solve_cluster import RoutingConfig, RoutingResult, is_optimal_route, solve_cluster, validate_clustering  # noqa: E402

DEFAULT_RECORDS = [
    "out/m_tier/M_seed01__onehot-ub[emb=polygons]_emu-mps__seed0.json",
    "out/m_tier/M_seed02__onehot-ub[emb=polygons]_emu-mps__seed0.json",
]


# --------------------------------------------------------------------------- #
# Part A record -> clusterings
# --------------------------------------------------------------------------- #
def formulation_of(rec: Dict[str, Any], inst: Instance):
    fw = rec["formulation"]
    weights = {k[2:]: v for k, v in fw.items() if k.startswith("w_")}
    return get_formulation(fw["formulation"], inst, **weights), weights


def best_feasible_shot(rec: Dict[str, Any], f) -> Dict[str, Any]:
    """Best feasible shot by model objective; ties broken by the lexicographically smallest assignment."""
    rows = []
    n_feas = 0
    for bits, c in rec["counts"].items():
        a = f.decode(str_to_bits(bits))
        if not f.feasible(a):
            continue
        a = tuple(int(k) for k in a)
        n_feas += c
        rows.append((float(f.objective(a)), a, bits, int(c)))
    if not rows:
        raise ValueError("no feasible shot in the record")
    rows.sort(key=lambda r: (r[0], r[1]))
    best_obj = rows[0][0]
    tied = [r for r in rows if abs(r[0] - best_obj) <= 1e-9]
    tied_clusterings = {clustering_of(r[1]) for r in tied}
    return {"objective": best_obj, "assignment": list(rows[0][1]), "bits": rows[0][2], "count": rows[0][3],
            "n_feasible_shots": n_feas, "shots": sum(rec["counts"].values()), "n_distinct_feasible_assignments": len(rows),
            "n_tied_best_assignments": len(tied), "n_tied_best_clusterings": len(tied_clusterings),
            "tie_rule": "lexicographically smallest assignment tuple among feasible shots with the minimal model objective"}


def sorted_clusters(a) -> List[List[int]]:
    return sorted((sorted(int(i) for i in g) for g in clustering_of(tuple(a))), key=lambda g: (len(g), g))


# --------------------------------------------------------------------------- #
# routing of one clustering
# --------------------------------------------------------------------------- #
def route_clustering(inst: Instance, clusters: List[List[int]], cfg: RoutingConfig,
                     cache: Dict[Tuple[str, Tuple[int, ...]], Tuple[RoutingResult, str]], source: str,
                     log=print) -> List[Dict[str, Any]]:
    rows = []
    for g in clusters:
        key = (inst.name, tuple(g))
        if key in cache:
            res, first = cache[key]
            reused = first
        else:
            log(f"[integrate] {source}: routing cluster {g} (n = {len(g)}) ...")
            res = solve_cluster(inst, g, cfg)
            cache[key] = (res, source)
            reused = None
            log(f"[integrate]   -> {res.method} cost {res.cost:.3f} (exact {res.exact_cost:.3f}) in {res.runtime_seconds:.0f} s")
        row = res.to_dict()
        row["optimal"] = is_optimal_route(res)
        row["reused_from"] = reused
        rows.append(row)
    return rows


def summarize(inst: Instance, a_sel, rows: List[Dict[str, Any]], a_opt, source: str) -> Dict[str, Any]:
    C_end = sum(r["cost"] for r in rows)
    C_clust, _ = route_exactly(inst, tuple(a_sel))
    C_dec, _ = route_exactly(inst, tuple(a_opt))
    C_ref = float(inst.reference["cost"])
    assert abs(C_clust - sum(r["exact_cost"] for r in rows)) < 1e-6
    split = {"C_end": C_end, "C_clust": C_clust, "C_dec": C_dec, "C_ref": C_ref,
             "decomposition_loss": C_dec - C_ref, "clustering_loss": C_clust - C_dec, "routing_loss": C_end - C_clust,
             "total_loss": C_end - C_ref,
             "identity_abs_error": abs((C_end - C_ref) - ((C_dec - C_ref) + (C_clust - C_dec) + (C_end - C_clust))),
             "gap_percent_end_vs_ref": gap_percent(C_end, C_ref),
             "gap_percent_clust_vs_ref": gap_percent(C_clust, C_ref), "gap_percent_dec_vs_ref": gap_percent(C_dec, C_ref),
             "percent_denominator": "C_ref (brief Eq. 2); routing_loss_percent uses C_clust",
             "routing_loss_percent_of_C_clust": 100.0 * (C_end - C_clust) / C_clust}
    quantum = [r for r in rows if r["quantum_attempted"]]
    q_ok = [r for r in quantum if r["method"] == "quantum-raw"]
    counts = {"n_clusters": len(rows), "cluster_sizes": [r["n"] for r in rows],
              "trivial_n_le_3": sum(1 for r in rows if r["n"] <= 3),
              "attempted_quantum_n_eq_4": len(quantum),
              "successful_raw_quantum_routes": len(q_ok),
              "raw_quantum_routes_optimal": sum(1 for r in q_ok if r["optimal"]),
              "classical_fallbacks": sum(1 for r in rows if r["fallback"]),
              "fallbacks_after_quantum_attempt": sum(1 for r in quantum if r["fallback"]),
              "fallbacks_n_ge_5": sum(1 for r in rows if r["fallback"] and r["n"] >= 5),
              "quantum_shots_total": sum(r["shots"] for r in quantum),
              "quantum_raw_valid_total": sum(r["n_raw_valid"] for r in quantum),
              "quantum_raw_optimal_total": sum(r["n_raw_optimal"] for r in quantum),
              "quantum_routing_gap_percent": [r["gap_percent_vs_exact"] for r in q_ok],
              "note": "fallback routes count in C_end but are excluded from quantum routing gaps"}
    return {"source": source, "instance": inst.name, "selected_assignment": list(a_sel), "clusters": sorted_clusters(a_sel),
            "model_optimal_assignment": list(a_opt), "split": split, "counts": counts, "per_cluster": rows}


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _f(x, nd=3):
    if x is None:
        return "null"
    if abs(x) < 0.5 * 10 ** (-nd):
        x = 0.0                                   # no "-0.000"
    return f"{x:.{nd}f}"


def per_cluster_table(summaries: List[Dict[str, Any]]) -> str:
    lines = ["| source | cluster (ids) | n | method | encoding | atoms | shots | raw valid (edge/pos) | raw optimal (edge/pos) | "
             "raw best cost | returned cost | exact TSP | gap % | fallback reason | record |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        for r in s["per_cluster"]:
            encs = {e["encoding"]: e for e in r["encodings"]}
            if encs:
                rv = "/".join(str(encs[k]["n_raw_valid"]) if encs[k]["emulated"] else "nr" for k in ("edge", "position") if k in encs)
                ro = "/".join(str(encs[k]["n_raw_optimal"]) if encs[k]["emulated"] else "nr" for k in ("edge", "position") if k in encs)
                rec = ", ".join(f"`{Path(encs[k]['record_path']).name}`" for k in ("edge", "position") if k in encs)
            else:
                rv = ro = "-"
                rec = "-"
            reason = r["fallback_reason"] or "-"
            if r["reused_from"]:
                reason += f" (same cluster as in `{r['reused_from']}`: result reused)"
            lines.append(f"| {s['source']} | {r['customer_ids']} | {r['n']} | {r['method']} | {r['encoding'] or '-'} | "
                         f"{r['atoms'] or '-'} | {r['shots'] or '-'} | {rv} | {ro} | {_f(r['raw_best_cost'])} | {_f(r['cost'])} | "
                         f"{_f(r['exact_cost'])} | {_f(r['gap_percent_vs_exact'], 2)} | {reason} | {rec} |")
    return "\n".join(lines)


def end_to_end_table(summaries: List[Dict[str, Any]]) -> str:
    lines = ["| source | instance | clusters | C_ref | C_dec | C_clust | C_end | C_dec−C_ref | C_clust−C_dec | C_end−C_clust | "
             "C_end−C_ref | gap_end % | trivial | attempted quantum | raw quantum routes | fallbacks |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        p, c = s["split"], s["counts"]
        lines.append(f"| {s['source']} | {s['instance']} | {s['clusters']} | {_f(p['C_ref'])} | {_f(p['C_dec'])} | {_f(p['C_clust'])} | "
                     f"{_f(p['C_end'])} | {_f(p['decomposition_loss'])} | {_f(p['clustering_loss'])} | {_f(p['routing_loss'])} | "
                     f"{_f(p['total_loss'])} | {_f(p['gap_percent_end_vs_ref'], 2)} | {c['trivial_n_le_3']} | "
                     f"{c['attempted_quantum_n_eq_4']} | {c['successful_raw_quantum_routes']} | {c['classical_fallbacks']} |")
    return "\n".join(lines)


def diagnostics_table(summaries: List[Dict[str, Any]]) -> str:
    """One row per (distinct n = 4 cluster, encoding), read from the saved cluster records."""
    lines = ["| cluster | encoding | atoms | C_opt | A | target GS optimal | rel-Frob | spurious J / 2A | device argmin decodes | "
             "rank(opt) | raw valid | raw optimal | exact P(valid) | exact P(opt) | rand-bits valid | rand-bits P(opt) | rand-tour P(opt) | emu s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    seen = set()
    for s in summaries:
        for r in s["per_cluster"]:
            key = (r["instance"], tuple(r["customer_ids"]))
            if not r["quantum_attempted"] or key in seen:
                continue
            seen.add(key)
            for e in r["encodings"]:
                rec = json.loads(Path(e["record_path"]).read_text())
                t, b = rec["classical_targets"], rec["baselines"]
                rb, rt = b["uniform_random_bitstrings"], b["uniform_random_valid_tours_analytic"]
                emb = e["embedding"]
                lines.append(f"| {r['instance']} {r['customer_ids']} | {e['encoding']} | {e['atoms']} | {_f(t['C_opt_bruteforce'])} | "
                             f"{_f(emb['A'], 1)} | {e['target_ground_states_optimal']} | {_f(emb['rel_frobenius_offdiag'])} | "
                             f"{_f(emb['max_spurious_coupling'], 0)} / {_f(2 * emb['A'], 0)} | {e['device_argmin_decodes']} | "
                             f"{e['rank_of_optimal_tour']} of {t['n_strings']} | {e['n_raw_valid']}/{e['shots']} | {e['n_raw_optimal']}/{e['shots']} | "
                             f"{_f(e['exact_p_valid'], 5)} | {_f(e['exact_p_optimal'], 6)} | {_f(rb['raw_feasible_rate'])} | "
                             f"{_f(rb['p_optimal_tour'])} | {_f(rt['p_optimal_per_shot'])} | {_f(e['emulation_seconds'], 0)} |")
    return "\n".join(lines)


def totals(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("n_clusters", "trivial_n_le_3", "attempted_quantum_n_eq_4", "successful_raw_quantum_routes",
            "raw_quantum_routes_optimal", "classical_fallbacks", "fallbacks_after_quantum_attempt", "fallbacks_n_ge_5",
            "quantum_shots_total", "quantum_raw_valid_total", "quantum_raw_optimal_total")
    tot = {k: sum(s["counts"][k] for s in summaries) for k in keys}
    # distinct quantum attempts (a cluster shared by two sources is emulated once)
    seen = set()
    distinct = []
    for s in summaries:
        for r in s["per_cluster"]:
            if r["quantum_attempted"] and (r["instance"], tuple(r["customer_ids"])) not in seen:
                seen.add((r["instance"], tuple(r["customer_ids"])))
                distinct.append(r)
    per_enc: Dict[str, Dict[str, int]] = {}
    for r in distinct:
        for e in r["encodings"]:
            d = per_enc.setdefault(e["encoding"], {"clusters": 0, "emulated": 0, "shots": 0, "raw_valid": 0, "raw_optimal": 0,
                                                     "clusters_with_a_valid_shot": 0})
            d["clusters"] += 1
            if e["emulated"]:
                d["emulated"] += 1
                d["shots"] += e["shots"]
                d["raw_valid"] += e["n_raw_valid"]
                d["raw_optimal"] += e["n_raw_optimal"]
                d["clusters_with_a_valid_shot"] += int(e["n_raw_valid"] > 0)
    for d in per_enc.values():
        d["raw_valid_rate"] = d["raw_valid"] / d["shots"] if d["shots"] else None
        d["raw_optimal_rate"] = d["raw_optimal"] / d["shots"] if d["shots"] else None
    tot["distinct_quantum_clusters"] = len(distinct)
    tot["per_encoding_over_distinct_quantum_clusters"] = per_enc
    tot["quantum_raw_valid_rate"] = tot["quantum_raw_valid_total"] / tot["quantum_shots_total"] if tot["quantum_shots_total"] else None
    return tot


def write_markdown(path: Path, summaries: List[Dict[str, Any]], cfg: RoutingConfig, tot: Dict[str, Any], paths: Dict[str, str]) -> None:
    pe = tot["per_encoding_over_distinct_quantum_clusters"]
    enc_lines = ["| encoding | distinct n=4 clusters | emulated | shots | raw valid | raw optimal | raw valid rate | clusters with >= 1 valid shot |",
                 "|---|---|---|---|---|---|---|---|"]
    for k, d in pe.items():
        enc_lines.append(f"| {k} | {d['clusters']} | {d['emulated']} | {d['shots']} | {d['raw_valid']} | {d['raw_optimal']} | "
                         f"{_f(d['raw_valid_rate'], 4)} | {d['clusters_with_a_valid_shot']} |")
    md = [
        "# Track B slice 2: Part A clusterings routed through `routing.solve_cluster` (emulation; no hardware)",
        "",
        f"Config: {cfg.shots} shots per encoding per n = 4 cluster, seed {cfg.seed}, {cfg.embed_seeds} embed seeds x {cfg.n_starts} starts "
        f"(selected by embedding fidelity only), position emulation = {cfg.position_emulation}, backend `cvrp.svsim` (exact state vector, "
        f"{cfg.dt_ns:g} ns slices), generic embedder `cvrp.embed.embed`, Part A default pulse.  Policy: n <= 3 exact classical; n = 4 both "
        "encodings, route = best RAW valid shot (strict decode, no repair), none -> exact classical fallback (flagged); n >= 5 exact "
        "classical fallback (flagged).  Costs recomputed from the instance distance matrix.",
        "",
        "## Per cluster", "", per_cluster_table(summaries), "",
        "`nr` = not run.  Raw valid / raw optimal are shot counts out of the per-encoding shots (edge / position).",
        "",
        "## End to end  (C_end − C_ref = (C_dec − C_ref) + (C_clust − C_dec) + (C_end − C_clust))", "", end_to_end_table(summaries), "",
        "`best-shot`: the feasible Part A shot with the lowest model objective (ties -> lexicographically smallest assignment); "
        "`model-opt`: the exact constrained optimum of the same model.  Fallback routes count in C_end and are excluded from quantum routing gaps.",
        "",
        "## Totals", "",
        f"* clusters routed: {tot['n_clusters']} (trivial n <= 3: {tot['trivial_n_le_3']}; attempted quantum n = 4: "
        f"{tot['attempted_quantum_n_eq_4']}, of which {tot['distinct_quantum_clusters']} distinct customer sets were emulated; "
        f"successful raw quantum routes: {tot['successful_raw_quantum_routes']} (optimal: {tot['raw_quantum_routes_optimal']}); "
        f"classical fallbacks: {tot['classical_fallbacks']} = {tot['fallbacks_after_quantum_attempt']} after a failed quantum attempt + "
        f"{tot['fallbacks_n_ge_5']} with n >= 5)",
        f"* quantum shots over attempted clusters (both encodings): {tot['quantum_shots_total']}, raw valid {tot['quantum_raw_valid_total']} "
        f"(rate {_f(tot['quantum_raw_valid_rate'], 4)}), raw optimal {tot['quantum_raw_optimal_total']}",
        "", "\n".join(enc_lines), "",
        "## Per-encoding diagnostics (distinct n = 4 clusters; formulation / geometry / dynamics separately)", "",
        diagnostics_table(summaries), "",
        "*target GS optimal*: the QUBO's exhaustive ground states decode to the `tsp_exact` tour; *rel-Frob*: `cvrp.embed` fidelity "
        "‖U/α − J‖_F/‖J‖_F; *spurious J*: largest realised coupling on a J = 0 pair vs the 2A the wanted pairs carry; *rank(opt)*: rank of the "
        "optimal-tour string under the zero-drive device energies of all strings (1 = lowest); *exact P(..)*: from the emulated final state, no "
        "sampling noise; *rand-bits*: 1000 uniform random bitstrings; *rand-tour*: uniform random valid tours (analytic 1/12 for a unique optimum).",
        "",
        "## Records", "",
        *[f"* `{p}`" for p in paths.values()],
    ]
    path.write_text("\n".join(md) + "\n")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="*", default=DEFAULT_RECORDS)
    ap.add_argument("--instances", default="instances")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--embed-seeds", type=int, default=6)
    ap.add_argument("--position-emulation", choices=("yes", "no"), default="yes")
    ap.add_argument("--out", default="out/routing")
    ap.add_argument("--skip-model-opt", action="store_true", help="only route the best-shot clusterings")
    ap.add_argument("--rebuild-md", action="store_true", help="rewrite <out>/integration.md from <out>/integration.json (no emulation)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.rebuild_md:
        saved = json.loads((out / "integration.json").read_text())
        cfg = RoutingConfig(**saved["config"])
        paths = {s["source"]: str(out / f"integration_{s['instance']}_{s['source'].split('/')[1]}.json") for s in saved["summaries"]}
        paths["all"] = str(out / "integration.json")
        write_markdown(out / "integration.md", saved["summaries"], cfg, saved["totals"], paths)
        print((out / "integration.md").read_text())
        return 0
    cfg = RoutingConfig(shots=a.shots, seed=a.seed, n_starts=a.n_starts, embed_seeds=a.embed_seeds,
                        position_emulation=a.position_emulation, out_dir=str(out))
    t0 = time.time()
    cache: Dict[Tuple[str, Tuple[int, ...]], Tuple[RoutingResult, str]] = {}
    summaries: List[Dict[str, Any]] = []
    paths: Dict[str, str] = {}
    for rp in a.records:
        rec = json.loads(Path(rp).read_text())
        inst = Instance.load(Path(a.instances) / f"{rec['instance']}.json")
        f, weights = formulation_of(rec, inst)
        best = best_feasible_shot(rec, f)
        mo = model_optimum(f)
        a_opt = tuple(int(k) for k in mo.assignment)
        inputs = [("best-shot", tuple(best["assignment"]))]
        if not a.skip_model_opt:
            inputs.append(("model-opt", a_opt))
        for kind, a_sel in inputs:
            source = f"{inst.name}/{kind}"
            clusters = validate_clustering(inst, clustering_of(a_sel))
            clusters = sorted(clusters, key=lambda g: (len(g), g))
            rows = route_clustering(inst, clusters, cfg, cache, source)
            s = summarize(inst, a_sel, rows, a_opt, source)
            s["part_a_record"] = rp
            s["part_a_method"] = rec.get("method")
            s["formulation"] = {"name": rec["formulation"]["formulation"], "weights": weights}
            s["best_shot"] = best
            s["model_optimum"] = {"value": mo.value, "n_optima": mo.n_optima, "method": mo.method,
                                  "clusterings": [sorted(sorted(g) for g in c) for c in mo.clusterings]}
            s["config"] = cfg.__dict__
            s["label"] = "emulation only (cvrp.svsim); no hardware; generic embedder; Part A default pulse"
            p = out / f"integration_{inst.name}_{kind}.json"
            p.write_text(json.dumps(s, indent=1, default=_json_default))
            paths[source] = str(p)
            summaries.append(s)
            sp = s["split"]
            print(f"[integrate] {source}: C_end {sp['C_end']:.3f}  C_clust {sp['C_clust']:.3f}  C_dec {sp['C_dec']:.3f}  "
                  f"C_ref {sp['C_ref']:.3f}  split {sp['decomposition_loss']:+.3f} {sp['clustering_loss']:+.3f} {sp['routing_loss']:+.3f}")
    tot = totals(summaries)
    allp = out / "integration.json"
    allp.write_text(json.dumps({"summaries": summaries, "totals": tot, "config": cfg.__dict__, "wall_seconds": time.time() - t0,
                                "label": "emulation only; no hardware"}, indent=1, default=_json_default))
    paths["all"] = str(allp)
    md = out / "integration.md"
    write_markdown(md, summaries, cfg, tot, paths)
    print()
    print(per_cluster_table(summaries))
    print()
    print(end_to_end_table(summaries))
    print(f"\n[integrate] totals: {json.dumps({k: v for k, v in tot.items() if k != 'per_encoding_over_distinct_quantum_clusters'})}")
    print(f"[integrate] wrote {md}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
