"""qopt — analog neutral-atom QUBO optimisation toolkit (Pasqal stack).

Pipeline:  Problem -> QUBO (model) -> Embedding (embed) -> Schedule (schedule)
           -> emulator / cloud bitstrings (run) -> decode + repair -> score
           vs. exact / heuristic classical solvers (classical) and Pasqal's
           qubo-solver (baseline). report.py plots everything.

Pure-numpy modules (no Pulser needed): model, classical, embed (except
DeviceLimits.from_pulser_device), schedule.waveforms, run.decode/repair/score.
Pulser / emulator / cloud code is imported lazily inside the functions that
need it so the core stays importable and testable anywhere.
"""

from .model import QUBO, Problem, brute_force_verify, qubo_from_matrix  # noqa: F401
from .classical import solve_exact, simulated_annealing, tabu_search  # noqa: F401
from .embed import DeviceLimits, Embedding, embed_qubo  # noqa: F401
from .schedule import Schedule  # noqa: F401

__version__ = "0.1.0"
