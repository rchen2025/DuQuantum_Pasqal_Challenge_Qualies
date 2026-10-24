"""cvrp.report: every D5 figure builds from the real records under out/ (skipped when out/ is empty),
the scan.md parser on an inline table, and the skip-gracefully paths (counts=None, baseline records)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvrp import report  # noqa: E402

MIN_PNG_BYTES = 5 * 1024
RECORD_DIRS = [ROOT / "out" / d for d in ("m1", "m2", "m3")]


def _records():
    recs = report.load_records([d for d in RECORD_DIRS if d.is_dir()])
    if not recs:
        raise unittest.SkipTest("no run records under out/ (run cvrp.run first)")
    return recs


def _ours(recs):
    cands = [r for r in recs if (r.get("embedding") or {}).get("J_target") is not None and report.pulse_params_from_record(r)]
    if not cands:
        raise unittest.SkipTest("no record with a cvrp.embed embedding under out/")
    return cands[0]


def _assert_png(path):
    assert path is not None and os.path.exists(path), path
    assert os.path.getsize(path) > MIN_PNG_BYTES, (path, os.path.getsize(path))


def test_short_method_labels():
    assert report.short_method("onehot-ub[emb=polygons]+qutip") == "polygons+qutip"
    assert report.short_method("onehot-ub[emb=generic]+random+repair:energy-descent") == "generic+random"
    assert report.short_method("onehot-ub+qubosolver-tabu") == "qubosolver-tabu"
    assert report.short_method("onehot-ub[emb=polygons]+numpy-sv|pulse=4000ns_Omega1.0(pre-scan)") == \
        "polygons+numpy-sv|4000ns_Omega1.0(pre-scan)"


def test_parse_scan_md_inline_table():
    md = """# scan
| point | feasibility | best_model_gap | P_reach_model | P_reach_ref | per_instance_P_reach_model |
|---|---|---|---|---|---|
| T=2000 ns, Omega x0.5, delta_f x1 | 0.522 | 0.000 | 0.305 | 0.181 | 0.25 / 0.25 / 0.41 |
| T=4000 ns, Omega x1, delta_f x1 | 0.937 | 0.000 | 0.797 | 0.606 | 0.90 / 0.84 / 0.65 |
| T=5800 ns, Omega x0.75, delta_f x1 | 0.983 | 0.000 | 0.932 | 0.642 | 0.96 / 0.95 / 0.89 |
"""
    rows = report.parse_scan_md(md)
    assert len(rows) == 3
    assert rows[0]["duration_ns"] == 2000 and rows[0]["omega_scale"] == 0.5 and rows[0]["delta_f_scale"] == 1.0
    assert rows[1]["duration_ns"] == 4000 and rows[1]["omega_scale"] == 1.0
    assert abs(rows[2]["P_reach_model"] - 0.932) < 1e-12 and abs(rows[2]["feasibility"] - 0.983) < 1e-12
    assert rows[0]["per_instance_P_reach_model"] == [0.25, 0.25, 0.41]
    with tempfile.TemporaryDirectory() as d:
        md_path = Path(d) / "scan.md"
        md_path.write_text(md)
        _assert_png(report.plot_pulse_scan(md_path, Path(d) / "scan.png"))
        assert report.plot_pulse_scan(Path(d) / "missing.md", Path(d) / "none.png") is None


def test_plot_register_from_records():
    rec = _ours(_records())
    with tempfile.TemporaryDirectory() as d:
        _assert_png(report.plot_register(rec, Path(d) / "register.png"))


def test_plot_coupling_heatmaps_from_records():
    rec = _ours(_records())
    with tempfile.TemporaryDirectory() as d:
        _assert_png(report.plot_coupling_heatmaps(rec, Path(d) / "couplings.png"))


def test_plot_pulse_from_records():
    rec = _ours(_records())
    with tempfile.TemporaryDirectory() as d:
        _assert_png(report.plot_pulse(rec, Path(d) / "pulse.png"))


def test_plot_gap_histogram_from_records():
    recs = _records()
    inst_name = recs[0]["instance"]
    same = [r for r in recs if r["instance"] == inst_name]
    with tempfile.TemporaryDirectory() as d:
        _assert_png(report.plot_gap_histogram(same, Path(d) / "gap.png"))
    # bins are fractions of all shots and sum to one when every shot is decoded from counts
    inst = report.load_instance(inst_name)
    if inst is not None:
        b, src = report.gap_bins(same[0], inst)
        assert src == "counts" and abs(b.sum() - 1.0) < 1e-9, (b, src)


def test_plot_summary_from_results_csv():
    csv_path = ROOT / "results.csv"
    if not csv_path.exists():
        raise unittest.SkipTest("results.csv not written yet")
    with tempfile.TemporaryDirectory() as d:
        _assert_png(report.plot_summary(csv_path, Path(d) / "summary.png"))


def test_plot_summary_skips_missing_values():
    csv_text = ("instance,method,repair,backend,feasibility_rate,p_reach_reference\n"
                "S_seed00,onehot-ub[emb=polygons]+qutip,none,qutip,0.99,0.9\n"
                "S_seed00,onehot-ub[emb=generic]+random,none,random,0.05,\n"
                "S_seed00,onehot-ub[emb=generic]+random,energy-descent,random,0.95,0.45\n"
                "S_seed01,onehot-ub+qubosolver-tabu,none,qubosolver-classical,1,1\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "results.csv"
        p.write_text(csv_text)
        _assert_png(report.plot_summary(p, Path(d) / "summary.png"))


def test_make_all_skips_bad_records_gracefully():
    recs = _records()
    rec = next((r for r in (report.pick_headline(recs, name) for name in sorted({r["instance"] for r in recs})) if r), None)
    if rec is None:
        raise unittest.SkipTest("no polygons record under out/")
    inst_name = rec["instance"]
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "recs"
        src.mkdir()
        (src / "good.json").write_text(json.dumps({k: v for k, v in rec.items() if not k.startswith("_")}))
        dry = dict(rec); dry["counts"] = None
        (src / "dry.json").write_text(json.dumps({k: v for k, v in dry.items() if not k.startswith("_")}))
        base = {"instance": inst_name, "method": "onehot-ub+qubosolver-tabu", "backend": "qubosolver-classical",
                "seed": 0, "shots": 1, "repair": "none", "embedding": None, "pulse": None,
                "counts": {next(iter(rec["counts"])): 1}, "metrics_raw": {"shots": 1, "top": []}}
        (src / "baseline.json").write_text(json.dumps(base))
        (src / "not_a_record.json").write_text("[1, 2, 3]")
        written = report.make_all([src], Path(d) / "figs", results_csv=None, instance=inst_name, pulse_scan=None)
        names = {Path(w).name for w in written}
        assert {"register.png", "couplings.png", "pulse.png", f"gap_hist_{inst_name}.png"} <= names, names
        for w in written:
            _assert_png(w)
        # a baseline record alone cannot give register/couplings/pulse but must not crash
        src2 = Path(d) / "recs2"; src2.mkdir()
        (src2 / "baseline.json").write_text(json.dumps(base))
        written2 = report.make_all([src2], Path(d) / "figs2", instance=inst_name)
        assert {Path(w).name for w in written2} == {f"gap_hist_{inst_name}.png"}, written2
