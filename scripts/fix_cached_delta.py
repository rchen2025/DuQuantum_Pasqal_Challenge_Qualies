#!/usr/bin/env python
"""Apply the delta_final physics bound (DERIVATIONS §6.7) to cached polygon registers in embeddings/.

    python scripts/fix_cached_delta.py [--dry-run]

The register geometry is left untouched (so before/after rows share the register); only `delta_final` is
rewritten when it lies outside (0, 0.9 U_clique), to 0.65 U_clique, exactly as `cvrp.embed_onehot` now does
for new registers.  U_clique is read off the register itself: C6 / (shortest intra-polygon distance)^6.
Prints one line per file; tier-S registers (delta_final ~ 0.65 U already) are reported as unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvrp.common import Instance  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="embeddings")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    for p in sorted((ROOT / a.dir).glob("*__polygons__*.json")):
        d = json.loads(p.read_text())
        inst_name = p.name.split("__")[0]
        inst = Instance.load(ROOT / "instances" / f"{inst_name}.json")
        N, K = inst.N, inst.K
        pts = np.asarray(d["coords_um"], float)
        if len(pts) != N * K:
            print(f"{p.name}: {len(pts)} atoms != N*K = {N * K}, skipped")
            continue
        owner = np.repeat(np.arange(N), K)
        D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        intra = D[(owner[:, None] == owner[None, :]) & (np.arange(N * K)[:, None] != np.arange(N * K)[None, :])]
        side = float(intra.min())
        U_clique = float(d["C6"]) / side ** 6
        old = float(d["delta_final"])
        if 0.0 < old < 0.9 * U_clique:
            print(f"{p.name}: delta_final {old:.3f} = {old / U_clique:.2f} U_clique ({U_clique:.2f}) — unchanged")
            continue
        new = 0.65 * U_clique
        print(f"{p.name}: delta_final {old:.3f} = {old / U_clique:.2f} U_clique -> {new:.3f} (0.65 U_clique)")
        if not a.dry_run:
            d["delta_final"] = new
            d.setdefault("notes", [])
            if isinstance(d["notes"], list):
                d["notes"].append(f"delta_final rewritten from {old:.4g} to 0.65 U_clique = {new:.4g} rad/µs by "
                                  "scripts/fix_cached_delta.py (DERIVATIONS §6.7 bound); geometry unchanged")
            p.write_text(json.dumps(d, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
