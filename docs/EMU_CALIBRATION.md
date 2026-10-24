# Emulator timing calibration

Machine: Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 189 Stepping 1, GenuineIntel; python 3.14.2, pulser 1.9.1, torch 2.14.1+cpu (8 threads).
Register: generic grid at 7 µm; pulse: three-point Ω ramp to 0.8·Ω_max + linear δ sweep, 2000 ns; 100 shots; emu dt = 10 ns, emu-mps precision 1e-5.

Timing only — says nothing about solution quality. TV = total-variation distance between bitstring histograms (sampling noise alone gives roughly sqrt(#distinct/shots)/2).

| atoms | backend | bond dim | wall time (s) | peak bond dim | note |
|---|---|---|---|---|---|
| 12 | pulser-simulation (qutip) | - | 1.9 | - | - |
| 12 | emu-sv | - | 2.1 | - | - |
| 12 | emu-mps | 16 | 38.1 | 16 | TV vs emu-sv: 0.840 |
| 12 | emu-mps | 32 | 39.5 | 32 | TV vs chi=16: 0.740; TV vs emu-sv: 0.830 |
| 20 | emu-sv | - | 166.9 | - | - |
| 20 | emu-mps | 16 | 109.0 | 16 | TV vs emu-sv: 1.000 |
| 20 | emu-mps | 32 | 125.6 | 32 | TV vs chi=16: 0.990; TV vs emu-sv: 1.000 |
| 30 | emu-mps | 16 | 180.6 | 16 | - |
| 30 | emu-mps | 32 | 215.4 | 32 | TV vs chi=16: 1.000 |

## How to read this

* Tier S (≤ 16 atoms with any sensible encoding) → pulser-simulation or emu-sv, exact.
* Tier M/L → emu-mps; a result is only trusted when TV between successive bond dimensions is small (< ~0.05 beyond sampling noise) and the peak bond dimension stayed below the cap. Report the bond dimension with every emu-mps row in results.csv.
