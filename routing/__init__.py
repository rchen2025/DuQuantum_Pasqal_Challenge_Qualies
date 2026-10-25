"""Track B (routing) — standalone first slice.

Two QUBO encodings of the single-vehicle depot-to-depot tour (TSP) with
*uniform* linear terms and *non-negative* couplings, so that they are at
least candidates for the global-detuning-only ``AnalogDevice``:

* ``encodings.build_position``  depot-fixed position one-hot, n^2 variables
  (handoff spec "Encoding 1"), depot legs folded into transition weights;
* ``encodings.build_edge``      undirected edge variables, n(n+1)/2 variables
  (handoff spec "Encoding 2"), exact TSP only for n <= 4 (cycle-cover
  relaxation beyond that; the production builder refuses n > 4).

``routing.fixtures`` / ``routing.check`` / ``routing.pilot`` are the standalone
fixture study (slice 1); ``routing.solve_cluster`` is the integration boundary
that routes frozen Part A clusters (slice 2, driven by
``scripts/routing_integrate.py``).  Emulation only, no hardware.  See
``docs/ROUTING.md``.
"""
