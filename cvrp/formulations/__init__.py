"""Concrete QUBO formulations of the capacity-feasible clustering model.

Importing this package registers every built-in formulation in
``cvrp.model.REGISTRY`` (``cvrp.model.get_formulation`` imports it lazily).
Add a new formulation by creating a module here, decorating the class with
``@cvrp.model.register`` and importing the module below.

Derivations: docs/DERIVATIONS.md (§3 textbook, §4 ours).
"""
from cvrp.formulations import textbook  # noqa: F401  (registers "assignment-only" and "textbook")

__all__ = ["textbook"]
