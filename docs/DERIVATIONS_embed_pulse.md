<!-- Fragment for docs/DERIVATIONS.md: sections 5 (Embedding) and 6 (Pulse).
     Code: cvrp/embed.py, cvrp/pulse.py.  Numbers below are copied from
     `python -m cvrp.embed --demo` and `python -m cvrp.pulse --demo` (pulser 1.9.1, seed 0). -->

## 5. Embedding

### 5.1 What the device can compute

Pulser's `AnalogDevice` drives every atom with the same laser (one global Rydberg
channel, no detuning-map / DMM).  In Pulser's units (ħ = 1, frequencies in rad/µs,
lengths in µm) the Hamiltonian is

$$
H(t) = \sum_i \frac{\Omega(t)}{2}\,\sigma^x_i \;-\; \delta(t)\sum_i n_i \;+\; \sum_{i<j} \frac{C_6}{r_{ij}^6}\, n_i n_j ,
\qquad n_i = |r\rangle\langle r|_i .
$$

At the end of the pulse Ω = 0, so $H$ is diagonal in the computational basis.
Reading $n_i \to x_i \in \{0,1\}$ (atom $i$ in the Rydberg state ⇔ $x_i = 1$), a
bitstring $x$ has the energy

$$
E_{\text{dev}}(x) = -\delta_{\text{final}} \sum_i x_i + \sum_{i<j} U_{ij}\, x_i x_j ,
\qquad U_{ij} = \frac{C_6}{r_{ij}^6} > 0 .
$$

This is a QUBO with three built-in restrictions: (i) every coupling $U_{ij}$ is
**positive** (the van der Waals interaction is repulsive), (ii) no coupling is ever
exactly **zero**, and (iii) every linear coefficient is the **same** number
$-\delta_{\text{final}}$.

### 5.2 From the QUBO to distances

Our QUBO (common.py convention) is $E(x) = x^\mathsf{T} Q x = \sum_i h_i x_i + \sum_{i<j} J_{ij} x_i x_j$
with $h_i = Q_{ii}$, $J_{ij} = 2Q_{ij}$.  We want the device to minimise the same
function, so we ask for $E_{\text{dev}} = \alpha\, E$ for a single positive energy
scale $\alpha$ (rad/µs per QUBO unit; it only rescales the spectrum and does not
move the minimiser).  Matching term by term:

$$
U_{ij} = \alpha J_{ij}, \qquad \delta_{\text{final}} = -\alpha h .
$$

Solving $C_6 / r_{ij}^6 = \alpha J_{ij}$ for the distance gives the **target distance**

$$
\boxed{\; r_{ij} = \left(\frac{C_6}{\alpha\, J_{ij}}\right)^{1/6} \;}
\qquad (\texttt{cvrp.embed.target\_distance}) .
$$

Two consequences of the sixth root: a factor 2 in $J$ moves the atoms only by
$2^{1/6} = 1.12$, and scaling *all* coordinates by $s$ multiplies every $U_{ij}$
by $s^{-6}$, i.e. it changes $\alpha$ by $s^{-6}$ while leaving the *shape* of the
realised QUBO untouched.  We use this freedom below.

**Negative couplings** ($J_{ij} < 0$) would need an attractive interaction; there
is none, so `embed()` raises `ValueError` naming the pairs.  The formulation has to
be rewritten, not patched here.  **Zero couplings** can only be approximated by
putting the two atoms far apart (§5.5).

### 5.3 Choosing α

α is bounded on both sides by the hardware:

1. **Minimum distance (geometry).**  Atoms cannot be closer than
   $r_{\min}$ = `device.min_atom_distance` = 5 µm, so the strongest coupling obeys
   $\alpha J_{\max} \le C_6 / r_{\min}^6 = 865\,723 / 5^6 = 55.4$ rad/µs.  We start
   from the *largest* α this allows (the optimiser places the strongest pair at the
   floor; it keeps a 2 % margin, 5.1 µm, so a rounding error never trips the hard
   limit).  Larger α means larger energy gaps relative to the fixed Ω and the fixed
   6 µs budget, which is what adiabaticity wants.
2. **Detuning cap.**  $|\delta_{\text{final}}| = \alpha |h| \le$ `max_abs_detuning`
   = 125.66 rad/µs.  If the geometric α violates it, α is set to
   $\alpha_{\text{cap}} = 125.66 / |h|$ and every coordinate is multiplied by
   $(\alpha/\alpha_{\text{cap}})^{1/6}$ (exact because $U \propto r^{-6}$).
3. **Register radius.**  After step 2 all atoms must still lie within
   `max_radial_distance` = 38 µm of the origin.  If not, we shrink back onto the
   38 µm circle and *report* the resulting detuning violation in
   `fidelity.limit_violations` — such a QUBO (|h| ≫ |J|) simply does not fit this
   device and the run layer must refuse it.  Atom count > `max_atom_num` = 80
   raises immediately.

All four numbers are read from the Pulser device object at run time
(`DeviceLimits.from_pulser_device`), never typed into the code.

### 5.4 Fidelity: the number we report

Even with α free, a 2-D register has only $2n-3$ geometric degrees of freedom for
$n(n-1)/2$ couplings, and $1/r^6$ obeys triangle-type inequalities (two atoms both
close to a third cannot be far from each other).  So in general $U/\alpha \ne J$,
and we quantify this with exactly

$$
\boxed{\;\text{rel\_frobenius\_offdiag} = \frac{\lVert U/\alpha - J\rVert_F}{\lVert J\rVert_F}
\quad\text{over all } i \ne j\;}
\qquad (\texttt{FidelityReport.rel\_frobenius\_offdiag}),
$$

together with the worst single pair, $\max_{i<j,\,J_{ij}>0} |U_{ij}/\alpha - J_{ij}| / J_{ij}$
(`worst_pair_rel_error`, `worst_pair`), and the largest *spurious* coupling
$\max_{J_{ij}=0} U_{ij}/\alpha$ (`max_spurious_coupling`).  α itself is the
least-squares value $\alpha = \langle U, J\rangle / \langle J, J\rangle$ for the final
geometry (then capped as in §5.3), and positions come from a multi-start L-BFGS
(`qopt.embed.optimize_positions`: relative error on positive targets, far-apart
target on zero pairs, hinge penalties for the 5 µm floor and the 38 µm radius;
8 seeded starts, best kept).  `Embedding.realized_Q()` returns the QUBO the device
actually implements, $Q'_{ij} = U_{ij}/(2\alpha)$, $Q'_{ii} = -\delta_{\text{final}}/\alpha$,
so the classical checker can compare its minimiser with the target's.

### 5.5 One worked pair (from `python -m cvrp.embed --demo`, seed 0)

Toy: 6 variables, $J_{ij}$ uniform in [0.2, 1] with 3 of the 15 pairs set to 0,
$h_i = -1$ for all $i$.  The optimiser returns α = **46.38 rad/µs per QUBO unit**
(the strongest pair sits at the 5.1 µm floor; the detuning cap is not active since
$|\delta_{\text{final}}| = 46.38 < 125.66$).  Strongest pair (2, 3):

| quantity | value |
|---|---|
| target $J_{23}$ | 0.9481 |
| α | 46.38 rad/µs |
| target $U = \alpha J$ | 43.97 rad/µs |
| target distance $(C_6/\alpha J)^{1/6} = (865\,723 / 43.97)^{1/6}$ | 5.196 µm |
| realised distance | 5.101 µm |
| realised $U = C_6 / 5.101^6$ | 49.13 rad/µs → $U/\alpha$ = 1.059 |
| relative error on this pair | 11.7 % |

The optimiser deliberately leaves this pair 1.8 % too close because the same two
atoms also have to serve their other neighbours; the global figures for the toy are
**rel_frobenius_offdiag = 0.370**, worst pair (1, 3) with 94 % error (target 0.685,
realised 0.039 — the geometry sacrificed it), largest spurious coupling 0.022 on the
zero pair (0, 2).  A dense, nearly uniform 6-clique is close to the worst case for
$1/r^6$ in 2-D; graph-structured QUBOs (few, well-separated neighbours per atom)
embed with errors well below 0.1 (cf. the 2-atom test, error < 1e-9).

**Zero couplings.**  $U$ is never 0.  The best we can do is distance: inside the
38 µm trap radius the largest possible separation is 2 × 38 = 76 µm, where
$U = 865\,723 / 76^6 = 4.5\times10^{-6}$ rad/µs, i.e. $U/\alpha \approx 1\times10^{-7}$
QUBO units at α = 46.4 — negligible.  In practice the zero pairs are much closer
(the register is compact), e.g. 20 µm gives $U/\alpha = 2.9\times10^{-4}$; the toy's
0.022 comes from a zero pair that ended up at 9.74 µm.  The report always quotes the
realised worst case so the classical checker can decide whether it matters.

### 5.6 Why the linear terms must be uniform

The device has one knob for all diagonal terms: $-\delta_{\text{final}}\sum_i x_i$.
An `AnalogDevice` QUBO must therefore have $h_i = h$ for every variable (a
`WeightedAnalogDevice` could lower individual $\delta_i$ with its DMM; this device
cannot).  `embed()` checks `linear_spread = max h − min h` against a $10^{-9}$
tolerance and sets `fidelity.linear_terms_uniform`.  If it is **false** we do not
silently fix anything: δ_final is set from the *mean* of $h$ so a sequence can
still be built for plumbing tests, a loud `NOT A FAITHFUL DEVICE EMBEDDING` note is
added, `fidelity.ok` is false, and the run layer refuses to present such a run as a
device result.  The formulation must make the diagonal uniform itself (e.g. by
absorbing per-variable biases into the coupling structure or into the constant).

## 6. Pulse

### 6.1 Why we start at large negative detuning

For $\delta \ll -\Omega$ the single-atom Hamiltonian $\tfrac{\Omega}{2}\sigma^x - \delta n$
has its ground state at $|g\rangle$ (energy ≈ 0 versus ≈ $-\delta > 0$ for
$|r\rangle$), and the interaction term only adds energy to states with excitations.
So the many-body ground state of $H(0)$ is $|g\,g\cdots g\rangle$, exactly the state
the machine prepares by default.  The adiabatic theorem then says: if $H(t)$ changes
slowly compared with $1/\Delta_{\min}^2$ (Δ = instantaneous gap above the ground
state), the system stays in the instantaneous ground state and ends in the ground
state of $H(T)$, which (Ω(T) = 0) is the minimiser of $E_{\text{dev}} = \alpha E_{\text{QUBO}}$.

`default_delta_init` = $-\min\!\big(\max(|\delta_{\text{final}}|,\, 2\Omega_{\max}),\ \text{max\_abs\_detuning}\big)$:
at least $2\Omega_{\max}$ below resonance so the initial gap
($\sqrt{\delta^2 + \Omega^2} \ge 2.2\,\Omega$ per atom) is unambiguous, at least
$|\delta_{\text{final}}|$ so the sweep is symmetric about δ = 0 and the slope is the
same on both sides of the crossing, and never beyond the channel limit.  Toy:
δ_init = −46.38 rad/µs.

### 6.2 Where δ_final comes from, and its sign

§5.2: $\delta_{\text{final}} = -\alpha h$.  For the toy $h = -1$ (every customer
assignment variable *wants* to be 1), so δ_final = +46.38 rad/µs > 0: positive
detuning lowers the energy of $|r\rangle$, favouring excitation, and the repulsive
$U_{ij}$ then decide which subset of atoms may be excited together — the usual
"maximum-weight independent set" picture.  A QUBO with $h > 0$ would give
δ_final < 0 and the trivial all-ground answer; the formulation layer keeps $h < 0$.

### 6.3 The baseline shape (brief §8.2)

The brief asks for "a smooth three-point ramp for the amplitude and a linear sweep
for the detuning, over a few microseconds".  `PulseParams` implements exactly that,
in fractions of the duration $T$ (default 4000 ns):

| waveform | knots (time fraction → value) |
|---|---|
| Ω(t) | 0 → 0, `rise_frac`=0.15 → Ω, 1−`fall_frac`=0.85 → Ω, 1 → 0 |
| δ(t) | 0 → δ_init, `sweep_start_frac`=0.15 → δ_init, `sweep_end_frac`=0.85 → δ_final, 1 → δ_final |

with Ω = `omega_scale` × Ω_max (default 1.0 × 12.566 rad/µs).  Both are Pulser
`InterpolatedWaveform`s (PCHIP through the knots, like Pulser itself), which is
monotone between knots: Ω never overshoots Ω_max and δ never leaves
[δ_init, δ_final], so the sampled waveforms respect the limits whenever the knots
do.  The detuning sweeps only while Ω is on its plateau, so the gap is never probed
with Ω = 0.  `waveforms()` reproduces Pulser's samples (knots at frac × (T − 1),
1 ns steps) for plots and tests; `final_detuning_from_sequence()` reads δ(T) back
from `pulser.sampler.sample(seq)` and agrees with δ_final to machine precision
(toy: diff 0.00e+00).  Toy baseline sweep rate: 92.75 rad/µs over 2.8 µs = 33 rad/µs².

### 6.4 The slow-region knob

The gap is smallest where the ground state changes character — the first atoms
flip around δ ≈ 0 … U, and a Landau–Zener crossing is traversed safely only if the
sweep rate there is small compared with $\Delta_{\min}^2$.  Rather than slowing the
whole sweep (the 6 µs budget is fixed), `slow_region = (centre_frac, width_frac,
slowdown_factor)` inserts extra δ-knots so that over the part of the *detuning path*
centred at `centre_frac` (0 = δ_init, 1 = δ_final) and spanning `width_frac` of it,
the slope is `slowdown_factor` times smaller than outside, with the total sweep
time unchanged (so the rest is correspondingly faster).  Example
`(0.6, 0.3, 3.0)` on the toy: the window covers δ ∈ [−4.6, +23.2] rad/µs, slope
inside ≈ 16 rad/µs² versus ≈ 66 outside (chord ratio exactly 3; PCHIP rounds the
corners).  The sweep stays monotone and still ends exactly at δ_final.  The right
centre is problem dependent — scan it with the emulator (`cvrp.run`), the formula is
only the knob.

### 6.5 Device limits every sequence is validated against

`build_sequence` lets Pulser do the validation: `Sequence(register, AnalogDevice)`
checks atom count ≤ 80, pairwise distance ≥ 5 µm, radius ≤ 38 µm; `seq.add(pulse,
"ryd")` checks Ω ≤ 12.566 rad/µs, |δ| ≤ 125.66 rad/µs (both raise `ValueError` from
the channel's `validate_pulse`), and total duration ≤ 6000 ns (raises
`RuntimeError`); durations are snapped to the 4 ns clock and ≥ 16 ns.  We then call
`seq.to_abstract_repr()` (the JSON the cloud/QPU receives) and assert
`seq.get_duration()` equals the requested duration.  `validate(seq)` re-checks the
*sampled* waveforms and the register against the device object independently.
The register is attached to an automatic layout (`Register.with_automatic_layout`,
`requires_layout` is true for the QPU); this rounds coordinates at the 1e-6 µm
level, i.e. U changes by ~1e-6 relative, far below the embedding error.

## Questions a judge will ask (embedding & pulse)

**Q: How do you know the register implements *your* QUBO and not something else?**
`Embedding.realized_Q()` is the QUBO the final Hamiltonian actually encodes
($Q'_{ij} = U_{ij}/2\alpha$, $Q'_{ii} = -\delta_{\text{final}}/\alpha$), and
`final_hamiltonian_from_sequence` rebuilds δ(T) and $U$ from the *built* Pulser
sequence, not from our parameters.  The gap between $Q'$ and $Q$ is one number,
rel_frobenius_offdiag (§5.4), plus the worst pair and the worst spurious coupling;
the classical checker runs on both $Q$ and $Q'$.

**Q: What about couplings you cannot realise?**
Negative ones are refused outright (`ValueError` listing the pairs, §5.2) — we never
drop or clip a sign.  Zeros are approximated by distance and the residual is
reported (§5.5: ≤ 1e-7 QUBO units at the 76 µm maximum separation, 0.022 in the
compact toy).  Non-uniform linear terms are flagged, not averaged away silently
(§5.6).

**Q: Why these pulse parameters, and are they legal on the device?**
δ_init is set so the start is unambiguously all-ground and the sweep is symmetric
about resonance (§6.1); δ_final is forced by the formulation (§6.2); the shape is the
brief's baseline (§6.3).  Legality is not our claim but Pulser's: the sequence is
built on `AnalogDevice` and serialised with `to_abstract_repr()`, and
`test_over_limit_pulses_are_rejected_by_pulser` shows an over-amplitude or
over-detuning request fails inside Pulser (§6.5).

**Q: Is the sweep adiabatic?**
Not guaranteed by construction — the minimum gap of a 10–15 atom problem is not
known analytically.  The baseline keeps the detuning sweep within the Ω plateau and
the slow-region knob (§6.4) lets us spend more of the fixed 6 µs budget where the
first atoms flip; whether a given setting is slow enough is measured on the
emulator (P(ground state) versus duration and slow-region centre), and the numbers
go into the results table, not into this document.
