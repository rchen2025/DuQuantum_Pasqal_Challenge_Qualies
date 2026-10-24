"""End-to-end plumbing test for cvrp.run with the random sampler (no emulator needed)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvrp.common import Instance  # noqa: E402
from cvrp.run import run_one, save_record, energy_descent_repair  # noqa: E402
import numpy as np  # noqa: E402


def _inst():
    return Instance.load(ROOT / "instances/S_seed00.json")


def test_random_backend_record_is_complete():
    inst = _inst()
    rec = run_one(inst, "textbook", "random", shots=400, seed=1, repair="energy-descent", verbose=False)
    for k in ("instance", "method", "backend", "seed", "shots", "repair", "dmm", "device", "device_faithful",
              "embedding", "pulse", "sequence_duration_ns", "counts", "metrics_raw", "metrics_repaired"):
        assert k in rec, k
    m = rec["metrics_raw"]
    assert m["shots"] == 400
    assert 0 <= m["feasible_rate"] <= 1
    # gaps only on feasible decodes: if nothing feasible, gaps are None (never a number)
    if m["feasible_rate"] == 0:
        assert m["best_model_gap"] is None and m["best_cvrp_gap"] is None
    else:
        assert m["best_model_gap"] is not None and m["best_model_gap"] >= -1e-9
        assert m["best_cvrp_gap"] is not None
    assert m["reference_status"] == "proven_optimum"
    assert rec["method"] == "textbook+random+repair:energy-descent"
    # textbook formulation has non-uniform linear terms -> not a faithful device embedding
    assert rec["device_faithful"] is False
    with tempfile.TemporaryDirectory() as d:
        p = save_record(rec, d)
        back = json.loads(Path(p).read_text())
        assert back["metrics_raw"]["shots"] == 400


def test_dry_run_builds_validated_sequence():
    inst = _inst()
    rec = run_one(inst, "textbook", "qutip", shots=10, seed=0, dry_run=True, verbose=False)
    assert rec["counts"] is None
    assert rec["sequence_duration_ns"] == rec["pulse"]["duration_ns"]
    assert rec["atoms"] == rec["n_vars"] == 14


def test_energy_descent_never_raises_energy():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(8, 8)); Q = 0.5 * (Q + Q.T)
    tr = energy_descent_repair(Q, 0.0)
    for _ in range(20):
        x = rng.integers(0, 2, 8)
        y = tr(x)
        assert y @ Q @ y <= x @ Q @ x + 1e-9
