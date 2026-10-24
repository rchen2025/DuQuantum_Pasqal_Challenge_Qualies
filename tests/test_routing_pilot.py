"""Scoring / baseline / device-energy bookkeeping of the routing pilot (no emulation; ~5 s).

Run:  PYTHONPATH=/home/claude/_pydeps:. python3 scripts/run_tests.py routing_pilot
"""
import numpy as np

from cvrp.common import bits_to_str
from cvrp.pulse import validate
from routing.encodings import build_edge, build_position, canonical_tour
from routing.fixtures import fixture
from routing.pilot import (analytic_random_tour_baseline, classical_targets, device_energy_analysis, embed_and_pulse,
                           random_bitstring_counts, random_tour_counts, score_counts)

FX = fixture("quad4")
D = FX.dist_matrix()
EDGE = build_edge(D)
POS = build_position(D)


def test_classical_targets_agree_with_held_karp_and_are_unique():
    for enc in (EDGE, POS):
        t = classical_targets(enc)
        assert abs(t["C_opt_bruteforce"] - t["C_opt_held_karp"]) < 1e-7
        assert t["n_optimal_undirected_tours"] == 1 and t["all_ground_states_optimal"]
        assert canonical_tour(t["held_karp_order"]) == tuple(t["optimal_tours_undirected"][0])
        assert abs(t["ground_energy"] - t["C_opt_bruteforce"]) < 1e-7
        assert t["A"] > t["C_ub_nearest_neighbour"] >= t["C_opt_bruteforce"]
    assert classical_targets(EDGE)["n_valid_tour_strings"] == 12
    assert classical_targets(POS)["n_valid_tour_strings"] == 24


def test_score_counts_is_strict_and_never_averages():
    t = classical_targets(EDGE)
    opt = tuple(t["optimal_tours_undirected"][0])
    other = (1, 2, 3, 4) if opt != (1, 2, 3, 4) else (1, 3, 2, 4)
    counts = {bits_to_str(EDGE.encode(opt)): 3, bits_to_str(EDGE.encode(other)): 5, "0" * 10: 2, "1" * 10: 10}
    s = score_counts(EDGE, counts, t)
    assert s["shots"] == 20 and abs(s["raw_feasible_rate"] - 8 / 20) < 1e-12 and abs(s["p_optimal_tour"] - 3 / 20) < 1e-12
    assert abs(s["best_feasible_gap_percent"]) < 1e-9 and tuple(s["best_feasible_tour"]) == opt
    assert "mean" not in " ".join(s.keys())
    s2 = score_counts(EDGE, {"0" * 10: 7}, t)
    assert s2["raw_feasible_rate"] == 0 and s2["best_feasible_cost"] is None and s2["best_feasible_gap_percent"] is None


def test_random_tour_baseline_is_always_feasible_and_near_one_twelfth():
    t = classical_targets(EDGE)
    a = analytic_random_tour_baseline(EDGE, t)
    assert a["n_undirected_tours"] == 12 and abs(a["p_optimal_per_shot"] - 1 / 12) < 1e-12
    s = score_counts(EDGE, random_tour_counts(EDGE, 2400, 0), t)
    assert s["raw_feasible_rate"] == 1.0 and abs(s["best_feasible_gap_percent"]) < 1e-9
    assert 0.04 < s["p_optimal_tour"] < 0.14            # 200 +- ~14 of 2400 expected
    # random bitstrings: 12 valid strings out of 1024 for the edge model
    sb = score_counts(EDGE, random_bitstring_counts(10, 4000, 0), t)
    assert sb["raw_feasible_rate"] < 0.03
    # ties: hex6 has 6 optimal tours out of 360
    a6 = analytic_random_tour_baseline(build_position(fixture("hex6_ties").dist_matrix()),
                                       {"n_optimal_undirected_tours": 6})
    assert a6["n_undirected_tours"] == 360 and abs(a6["p_optimal_per_shot"] - 1 / 60) < 1e-12


def test_embedding_pulse_and_device_energy_bookkeeping():
    t = classical_targets(EDGE)
    emb, params, seq, info = embed_and_pulse(EDGE, seed=0, n_starts=2)
    assert emb.n == 10 and info["fidelity"]["linear_terms_uniform"] and not validate(seq)
    assert info["sequence"]["validation_violations"] == [] and info["sequence"]["duration_ns"] <= 6000
    assert abs(info["sequence"]["delta_T_readback"] - emb.delta_final) < 1e-6
    assert abs(emb.delta_final - (-emb.alpha * EDGE.h[0])) < 1e-9          # delta_final = 6 alpha A
    assert abs(emb.delta_final - 6 * emb.alpha * EDGE.A) < 1e-9
    # zero couplings are approximated, not refused
    assert info["fidelity"]["max_spurious_coupling"] is not None and info["fidelity"]["max_spurious_coupling"] > 0
    dev = device_energy_analysis(EDGE, emb, seq, t)
    assert dev["n_valid_total"] == 12 and 1 <= min(dev["rank_of_optimal_tour_strings"]) <= 1024
    assert dev["best_rank_of_any_valid_tour"] <= min(dev["rank_of_optimal_tour_strings"])
    # the device energy of a string must match Embedding.device_energy with the same bit order
    s = dev["lowest_device_strings"][0]
    x = np.array([int(c) for c in s["bits"]], float)
    assert abs(emb.device_energy(x) - s["E_dev"]) < 1e-3      # register layout rounds coordinates at ~1e-7 um
    # the device argmin has the per-string energy -delta*k + interactions; a consistency check of the sign convention:
    # all-zero string has energy 0, so the argmin must be <= 0
    assert s["E_dev"] <= 0.0
