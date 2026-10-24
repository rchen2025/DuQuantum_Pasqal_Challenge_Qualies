"""Track B (routing) — standalone first slice.

Two QUBO encodings of the single-vehicle depot-to-depot tour (TSP) with
*uniform* linear terms and *non-negative* couplings, so that they are at
least candidates for the global-detuning-only ``AnalogDevice``:

* ``encodings.build_position``  depot-fixed position one-hot, n^2 variables
  (handoff spec "Encoding 1"), depot legs folded into transition weights;
* ``encodings.build_edge``      undirected edge variables, n(n+1)/2 variables
  (handoff spec "Encoding 2"), exact TSP only for n <= 4 (cycle-cover
  relaxation beyond that; the production builder refuses n > 4).

Everything here is a *standalone routing fixture* study (``routing.fixtures``,
``routing.check``, ``routing.pilot``); it is not an end-to-end Track B run on
Part A clusterings.  See ``docs/ROUTING.md``.
"""
