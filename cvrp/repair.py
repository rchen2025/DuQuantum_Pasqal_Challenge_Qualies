"""Bounded, deterministic classical repair for slack-free one-hot samples.

Only instance geometry/demands and the clustering objective are consulted.
References and feasibility certificates are never used to construct a solution.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from .common import Assignment, Formulation
from .exact import tsp_tour_table


class CapacityLocalRepair:
    """Callable bitstring transform with assignment caching and measured cost.

    Relocations precede swaps in the final stable tie-break. Within each kind,
    customer indices and destination vehicle indices increase lexicographically.
    The same 2N accepted-move budget covers both overload and route improvement.
    """

    def __init__(self, formulation: Formulation):
        self.f = formulation
        self.inst = formulation.inst
        if formulation.n_vars != self.inst.N * self.inst.K or not hasattr(formulation, "encode"):
            raise ValueError("capacity-local requires a slack-free one-hot formulation with encode")
        self.max_moves = 2 * self.inst.N
        self.cache: Dict[Assignment, Assignment] = {}
        self.traces: Dict[Assignment, list] = {}
        self.wall_s = 0.0
        self.cache_hits = 0
        self.invalid_inputs = 0
        self._tours = None

    def overload(self, a: Assignment) -> int:
        return sum(max(0, load - self.inst.Q) for load in self.inst.loads(a))

    def route_cost(self, a: Assignment) -> float:
        if self._tours is None:
            # Exact subset tours accelerate repeated local moves. This table
            # optimizes only the route inside a group, never the partition.
            self._tours = tsp_tour_table(self.inst.dist_matrix())
        masks = [0] * self.inst.K
        for i, k in enumerate(a):
            masks[k] |= 1 << i
        return float(sum(self._tours[m] for m in masks))

    def neighbors(self, a: Assignment):
        for i in range(self.inst.N):
            for k in range(self.inst.K):
                if k != a[i]:
                    b = list(a)
                    b[i] = k
                    yield tuple(b), (0, i, k)
        for i in range(self.inst.N):
            for j in range(i + 1, self.inst.N):
                if a[i] != a[j]:
                    b = list(a)
                    b[i], b[j] = b[j], b[i]
                    yield tuple(b), (1, i, j)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        try:
            x = np.asarray(x)
            if (x.shape != (self.f.n_vars,) or not np.isin(x, [0, 1]).all()
                    or not np.all(x.reshape(self.inst.N, self.inst.K).sum(axis=1) == 1)):
                self.invalid_inputs += 1
                return x.copy()
            initial = self.f.decode(x)
            if initial is None:
                self.invalid_inputs += 1
                return x.copy()
            initial = tuple(int(k) for k in initial)
            if initial in self.cache:
                self.cache_hits += 1
                return self.f.encode(self.cache[initial])
            current = initial
            trace = []
            for _ in range(self.max_moves):
                overload = self.overload(current)
                before_route = self.route_cost(current) if overload == 0 else None
                candidates = []
                for candidate, move in self.neighbors(current):
                    after_overload = self.overload(candidate)
                    if overload:
                        if after_overload < overload:
                            candidates.append(((after_overload, self.f.objective(candidate), move), candidate))
                    elif after_overload == 0:
                        route = self.route_cost(candidate)
                        if route < before_route - 1e-9:
                            candidates.append(((route, self.f.objective(candidate), move), candidate))
                if not candidates:
                    break
                key, candidate = min(candidates, key=lambda item: item[0])
                trace.append({"stage": "overload" if overload else "route", "move": key[-1],
                              "overload_before": overload, "overload_after": self.overload(candidate),
                              "route_before": before_route,
                              "route_after": self.route_cost(candidate) if not self.overload(candidate) else None})
                current = candidate
            self.cache[initial] = current
            self.traces[initial] = trace
            return self.f.encode(current)
        finally:
            self.wall_s += time.perf_counter() - start

    def stats(self) -> dict:
        return {"max_accepted_moves_per_assignment": self.max_moves,
                "distinct_assignments": len(self.cache), "cache_hits": self.cache_hits,
                "invalid_distinct_inputs": self.invalid_inputs,
                "accepted_moves": sum(len(t) for t in self.traces.values()),
                "max_accepted_moves_observed": max((len(t) for t in self.traces.values()), default=0),
                "failed_distinct_assignments": sum(self.overload(a) > 0 for a in self.cache.values()),
                "repair_wall_s": self.wall_s}


def capacity_local_repair(formulation: Formulation) -> CapacityLocalRepair:
    return CapacityLocalRepair(formulation)
