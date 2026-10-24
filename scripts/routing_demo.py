"""One-command four-customer routing comparison (Track B first slice, standalone fixture).

    PYTHONPATH=/home/claude/_pydeps:. python3 scripts/routing_demo.py [--fixture quad4] [--shots 1000]
        [--seed 0] [--position-emulation auto|yes|no] [--out out/routing]

Builds both encodings (edge: 10 atoms, position: 16 atoms) for the fixture, embeds them with the
generic Part A embedder, builds and validates the Part A pulse on AnalogDevice, analyses the
zero-drive device energies of every bitstring, emulates the 10-atom edge model exactly with
cvrp.svsim (and the 16-atom position model when the runtime probe says it fits the budget),
and writes JSON records + a markdown table under --out.  Emulation only; nothing ran on hardware.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing.pilot import PilotConfig, run_pilot, summary_table  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="quad4")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--position-emulation", choices=("auto", "yes", "no"), default="auto")
    ap.add_argument("--position-budget-s", type=float, default=600.0)
    ap.add_argument("--out", default="out/routing")
    a = ap.parse_args(argv)
    cfg = PilotConfig(fixture=a.fixture, shots=a.shots, seed=a.seed, n_starts=a.n_starts,
                      position_emulation=a.position_emulation, position_time_budget_s=a.position_budget_s, out_dir=a.out)
    records, md = run_pilot(cfg)
    print()
    print(summary_table(records))
    for r in records:
        t = r["classical_targets"]
        print(f"\n{r['encoding']['encoding']}: C_opt = {t['C_opt_bruteforce']:.3f}, optimal tours {t['optimal_tours_undirected']}, "
              f"A = {t['A']:.2f} (C_ub = {t['C_ub_nearest_neighbour']:.2f}); "
              f"alpha = {r['embedding']['alpha']:.4g}, delta_final = {r['embedding']['delta_final']:.3g} rad/us; "
              f"validation: {r['embedding']['sequence']['validation_violations'] or 'OK'}")
        d = r["device_energies"]
        print(f"  device argmin {d['device_argmin']} decodes={d['device_argmin_decodes']} tour={d['device_argmin_tour']}; "
              f"rank of optimal-tour strings {d['rank_of_optimal_tour_strings']}; best valid rank {d['best_rank_of_any_valid_tour']}; "
              f"corr(E_dev, E_target) = {d['corr_device_vs_target_energy_all_strings']:.3f}")
        s = r["samples"]
        if s.get("backend"):
            print(f"  samples: feasible {s['raw_feasible_rate']:.3f} (exact {s['exact_p_feasible']:.4f}), "
                  f"P(opt) {s['p_optimal_tour']:.3f} (exact {s['exact_p_optimal']:.4f}), best gap {s['best_feasible_gap_percent']}, "
                  f"{s['seconds']:.0f} s")
        else:
            print("  samples:", s["status"])
    print(f"\nmarkdown table: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
