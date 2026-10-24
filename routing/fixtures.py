"""Deterministic standalone routing fixtures (NOT Part A clusters).

Each fixture is one frozen customer set with its depot.  Original customer ids are
kept apart from the local matrix indices: local index 0 is the depot, local index k
(1..n) is ``customer_ids[k-1]``.  Ids are deliberately >= 100 so that a mix-up with
local indices is caught immediately.

* ``quad4``        4 customers, depot off-centre (unequal depot legs), customers in convex
                   position so crossing candidate tours exist; unique optimum.  Both
                   encodings (16 and 10 bits) run on it.
* ``tri5_subtour`` 5 customers: depot + 2 customers form a tight unit triangle, 3 more form
                   another 100 away.  The degree-only edge model's ground state is the
                   2-cycle cover (cost 6), not the tour (cost 203) - see routing.check.
* ``hex6_ties``    6 customers on a regular hexagon around the depot: 6 tied optimal tours
                   (drop any one hexagon edge), so "P(optimal)" must account for ties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class RoutingFixture:
    name: str
    customer_ids: Tuple[int, ...]            # original ids, order = local index 1..n
    depot: Tuple[float, float]
    customers: Tuple[Tuple[float, float], ...]
    notes: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def n(self) -> int:
        return len(self.customer_ids)

    def points(self) -> np.ndarray:
        return np.array([self.depot, *self.customers], dtype=float)

    def dist_matrix(self) -> np.ndarray:
        P = self.points()
        diff = P[:, None, :] - P[None, :, :]
        return np.sqrt((diff ** 2).sum(-1))

    def local_to_id(self, k: int) -> int:
        if not 1 <= k <= self.n:
            raise IndexError(f"local index {k} is not a customer (1..{self.n}); 0 is the depot")
        return self.customer_ids[k - 1]

    def id_to_local(self, cid: int) -> int:
        return self.customer_ids.index(cid) + 1

    def tour_ids(self, tour) -> List[int]:
        """Local tour (1..n) -> original ids with the depot (as 'depot') at both ends."""
        return ["depot", *[self.local_to_id(int(k)) for k in tour], "depot"]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "n_customers": self.n, "customer_ids": list(self.customer_ids),
                "depot": list(self.depot), "customers": [list(c) for c in self.customers],
                "notes": self.notes, "tags": list(self.tags), "label": "standalone routing fixture"}


def _hexagon(radius: float, centre: Tuple[float, float]) -> Tuple[Tuple[float, float], ...]:
    cx, cy = centre
    return tuple((cx + radius * math.cos(k * math.pi / 3), cy + radius * math.sin(k * math.pi / 3)) for k in range(6))


def _two_triangles(side: float = 1.0, separation: float = 100.0):
    h = side * math.sqrt(3) / 2
    tri = [(0.0, 0.0), (side, 0.0), (side / 2, h)]
    far = [(x + separation, y) for x, y in tri]
    return tri[0], tuple(tri[1:] + far)


_depot5, _cust5 = _two_triangles()

FIXTURES: Dict[str, RoutingFixture] = {
    "quad4": RoutingFixture(
        name="quad4", customer_ids=(101, 102, 103, 104), depot=(35.0, 42.0),
        customers=((10.0, 20.0), (90.0, 30.0), (80.0, 85.0), (20.0, 70.0)),
        notes="depot off-centre (unequal depot legs), customers in convex position (crossing candidate tours), unique optimum",
        tags=("both-encodings", "unequal-depot", "crossing-tours", "unique-optimum")),
    "tri5_subtour": RoutingFixture(
        name="tri5_subtour", customer_ids=(201, 202, 203, 204, 205), depot=_depot5, customers=_cust5,
        notes="two tight unit triangles 100 apart: degree-only edge model ground state is a 2-cycle cover (cost 6) not the tour (203)",
        tags=("edge-subtour-failure", "position-only")),
    "hex6_ties": RoutingFixture(
        name="hex6_ties", customer_ids=(301, 302, 303, 304, 305, 306), depot=(50.0, 50.0), customers=_hexagon(30.0, (50.0, 50.0)),
        notes="regular hexagon around the depot: six tied optimal tours (any hexagon edge may be the one replaced by the two depot legs)",
        tags=("tied-costs", "position-only")),
}


def fixture(name: str) -> RoutingFixture:
    return FIXTURES[name]
