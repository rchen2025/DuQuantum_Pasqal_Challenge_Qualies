"""Parameterised adiabatic schedules (Omega ramp + delta sweep + DMM) that
respect device limits, plus a scan / optimise loop.

Units: durations in ns (multiples of the device clock period), frequencies
in rad/µs (Pulser convention; 2π·1 MHz = 6.283 rad/µs).

Default shape (fractions of the total duration T):

    Omega:   0 --rise--> Omega_max ---- plateau ---- --fall--> 0
    delta:   delta_init (hold) --------- sweep -----> delta_final (hold)
    DMM:     constant delta_dmm   (or 0 -> delta_dmm ramp during the sweep)

The sweep from delta_init < 0 (all atoms ground) to delta_final =
Embedding.plan.delta_global happens while Omega is on the plateau.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .embed import DeviceLimits, Embedding


@dataclass
class Schedule:
    duration: int                       # ns
    omega_max: float                    # rad/µs
    delta_init: float                   # rad/µs (negative)
    delta_final: float                  # rad/µs (from embedding plan)
    delta_dmm: float = 0.0              # rad/µs (<= 0), final/constant DMM value
    rise_frac: float = 0.15
    fall_frac: float = 0.15
    sweep_start_frac: float = 0.15      # delta sweep window (fractions of T)
    sweep_end_frac: float = 0.85
    dmm_shape: str = "constant"         # "constant" | "ramp"
    sweep_shape: str = "linear"         # "linear" | "smooth" (tanh-like via PCHIP knots)
    name: str = "adiabatic"
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def default_for(
        cls,
        emb: Embedding,
        duration: Optional[int] = None,
        omega_max: Optional[float] = None,
        delta_init: Optional[float] = None,
        **kw: Any,
    ) -> "Schedule":
        lim = emb.limits
        T = duration or (min(4000, (lim.max_sequence_duration or 4000) - 400))
        om = omega_max if omega_max is not None else (lim.max_amp or 2 * math.pi * 2.0)
        df = emb.plan.delta_global
        di = delta_init if delta_init is not None else -max(abs(df), 2.0 * om, 2 * math.pi * 2.0)
        if lim.max_abs_detuning is not None:
            di = max(di, -lim.max_abs_detuning)
        return cls(T, om, di, df, emb.plan.delta_dmm, **kw).snapped(lim)

    def snapped(self, limits: DeviceLimits) -> "Schedule":
        """Round durations to the clock period and clip to device limits."""
        cp = max(1, limits.clock_period)
        T = int(round(self.duration / cp) * cp)
        if limits.max_sequence_duration:
            T = min(T, int((limits.max_sequence_duration - 200) // cp * cp))
        T = max(T, 4 * max(limits.min_duration, cp))
        om = self.omega_max
        if limits.max_amp is not None:
            om = min(om, limits.max_amp)
        di, df = self.delta_init, self.delta_final
        if limits.max_abs_detuning is not None:
            di = float(np.clip(di, -limits.max_abs_detuning, limits.max_abs_detuning))
            df = float(np.clip(df, -limits.max_abs_detuning, limits.max_abs_detuning))
        return replace(self, duration=T, omega_max=om, delta_init=di, delta_final=df)

    def with_params(self, **kw: Any) -> "Schedule":
        return replace(self, **kw)

    # ---- knot representation --------------------------------------------- #
    def knots(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        T = self.duration
        t_amp = np.array([0.0, self.rise_frac, 1 - self.fall_frac, 1.0]) * T
        v_amp = np.array([0.0, self.omega_max, self.omega_max, 0.0])
        if self.sweep_shape == "smooth":
            s0, s1 = self.sweep_start_frac, self.sweep_end_frac
            ts = np.array([0.0, s0, s0 + 0.25 * (s1 - s0), s0 + 0.5 * (s1 - s0), s0 + 0.75 * (s1 - s0), s1, 1.0]) * T
            frac = np.array([0, 0, 0.1, 0.5, 0.9, 1, 1])
            vs = self.delta_init + (self.delta_final - self.delta_init) * frac
        else:
            ts = np.array([0.0, self.sweep_start_frac, self.sweep_end_frac, 1.0]) * T
            vs = np.array([self.delta_init, self.delta_init, self.delta_final, self.delta_final])
        out = {"amp": (t_amp, v_amp), "det": (ts, vs)}
        if self.delta_dmm < 0:
            if self.dmm_shape == "ramp":
                out["dmm"] = (ts.copy(), np.array([0.0, 0.0, self.delta_dmm, self.delta_dmm]) if self.sweep_shape != "smooth"
                              else self.delta_dmm * frac)
            else:
                out["dmm"] = (np.array([0.0, T]), np.array([self.delta_dmm, self.delta_dmm]))
        return out

    def waveforms(self, dt: float = 1.0) -> Dict[str, np.ndarray]:
        """Pure-numpy samples (for plots / checks). Uses PCHIP like Pulser."""
        from scipy.interpolate import PchipInterpolator
        t = np.arange(0, self.duration, dt)
        out = {"t": t}
        for k, (tk, vk) in self.knots().items():
            if len(tk) == 2:
                out[k] = np.full_like(t, vk[0], dtype=float)
            else:
                out[k] = PchipInterpolator(tk, vk)(t)
        out["amp"] = np.maximum(out["amp"], 0.0)
        return out

    # ---- diagnostics ------------------------------------------------------ #
    def adiabaticity_hints(self, emb: Embedding) -> Dict[str, float]:
        """Crude indicators: coupling scales vs Omega and sweep rate."""
        U = emb.U
        pos = emb.qubo.couplings > 0
        U_min = float(U[pos].min()) if pos.any() else float("nan")
        sweep_ns = (self.sweep_end_frac - self.sweep_start_frac) * self.duration
        rate = abs(self.delta_final - self.delta_init) / (sweep_ns * 1e-3)  # rad/µs per µs
        return {
            "omega_max": self.omega_max,
            "U_min_positive/omega": U_min / self.omega_max if self.omega_max else float("nan"),
            "sweep_rate_rad_per_us2": rate,
            "landau_zener_proxy_omega2/rate": (self.omega_max ** 2) / rate if rate else float("inf"),
            "T_us": self.duration * 1e-3,
        }

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        return d


# --------------------------------------------------------------------------- #
# Pulser sequence construction
# --------------------------------------------------------------------------- #
def build_sequence(emb: Embedding, sched: Schedule, device: Any = None, register: Any = None,
                   use_dmm: bool = True):
    """Embedding + Schedule -> pulser.Sequence (global Rydberg + optional DMM).

    Bitstring order of results = register qubit order = QUBO variable order.
    """
    from pulser import Pulse, Sequence
    from pulser.waveforms import ConstantWaveform, InterpolatedWaveform

    if device is None:
        import pulser.devices as pd
        device = getattr(pd, emb.limits.name)
    sched = sched.snapped(emb.limits)
    reg = register if register is not None else emb.register(device=device)
    seq = Sequence(reg, device)
    seq.declare_channel("ryd", "rydberg_global")
    K = sched.knots()
    T = sched.duration
    amp = InterpolatedWaveform(T, K["amp"][1], times=K["amp"][0] / T)
    det = InterpolatedWaveform(T, K["det"][1], times=K["det"][0] / T)
    seq.add(Pulse(amp, det, 0.0), "ryd")
    if use_dmm and emb.plan.uses_dmm and emb.plan.delta_dmm < 0 and device.dmm_channels:
        dmm_id = emb.limits.dmm_id or next(iter(device.dmm_channels))
        dmap = reg.define_detuning_map(emb.weights)
        seq.config_detuning_map(dmap, dmm_id)
        tk, vk = K["dmm"]
        if len(tk) == 2:
            wf = ConstantWaveform(T, vk[0])
        else:
            wf = InterpolatedWaveform(T, vk, times=tk / T)
        seq.add_dmm_detuning(wf, dmm_id, protocol="no-delay")
    return seq


def final_hamiltonian_from_sequence(seq) -> Tuple[np.ndarray, np.ndarray]:
    """Read back per-atom final detunings and the U matrix from an actual
    Sequence (independent check of what will be executed)."""
    from pulser.sampler import sample
    s = sample(seq).to_nested_dict()
    reg = seq.register
    qids = list(reg.qubit_ids)
    n = len(qids)
    delta = np.zeros(n)
    g = s.get("Global", {}).get("ground-rydberg", {})
    if g:
        delta += float(np.asarray(g["det"])[-1])
    loc = s.get("Local", {}).get("ground-rydberg", {})
    for i, q in enumerate(qids):
        if q in loc:
            delta[i] += float(np.asarray(loc[q]["det"])[-1])
    coords = np.array([np.asarray(reg.qubits[q], float) for q in qids])
    from .embed import coupling_matrix
    U = coupling_matrix(coords, float(seq.device.interaction_coeff))
    return delta, U


# --------------------------------------------------------------------------- #
# Scan / optimise
# --------------------------------------------------------------------------- #
@dataclass
class ScanPoint:
    params: Dict[str, Any]
    score: float
    info: Dict[str, Any]


def scan(
    base: Schedule,
    grid: Dict[str, Iterable[Any]],
    evaluate: Callable[[Schedule], Tuple[float, Dict[str, Any]]],
    limits: Optional[DeviceLimits] = None,
    verbose: bool = True,
) -> List[ScanPoint]:
    """Grid scan over schedule parameters. ``evaluate(schedule)`` returns
    (score, info) where HIGHER score is better (e.g. approximation ratio or
    probability of optimum)."""
    import itertools
    keys = list(grid)
    results: List[ScanPoint] = []
    for combo in itertools.product(*[list(grid[k]) for k in keys]):
        params = dict(zip(keys, combo))
        s = base.with_params(**params)
        if limits is not None:
            s = s.snapped(limits)
        score, info = evaluate(s)
        results.append(ScanPoint(params, score, info))
        if verbose:
            print(f"  scan {params} -> score {score:.4f}")
    results.sort(key=lambda r: -r.score)
    return results


def optimize(
    base: Schedule,
    bounds: Dict[str, Tuple[float, float]],
    evaluate: Callable[[Schedule], Tuple[float, Dict[str, Any]]],
    limits: Optional[DeviceLimits] = None,
    n_random: int = 6,
    n_nelder: int = 20,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[Schedule, List[ScanPoint]]:
    """Random search followed by Nelder-Mead on continuous schedule params.
    Integer params (duration) are rounded by ``snapped``."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    keys = list(bounds)
    lo = np.array([bounds[k][0] for k in keys], float)
    hi = np.array([bounds[k][1] for k in keys], float)
    history: List[ScanPoint] = []

    def make(x: np.ndarray) -> Schedule:
        p = {k: (int(round(v)) if k == "duration" else float(v)) for k, v in zip(keys, np.clip(x, lo, hi))}
        s = base.with_params(**p)
        return s.snapped(limits) if limits is not None else s

    def f(x: np.ndarray) -> float:
        s = make(x)
        score, info = evaluate(s)
        history.append(ScanPoint({k: getattr(s, k) for k in keys}, score, info))
        if verbose:
            print(f"  opt {history[-1].params} -> {score:.4f}")
        return -score

    X = [lo + (hi - lo) * rng.random(len(keys)) for _ in range(n_random)]
    X.append(np.array([getattr(base, k) for k in keys], float))
    vals = [f(x) for x in X]
    x0 = X[int(np.argmin(vals))]
    res = minimize(f, x0, method="Nelder-Mead", options={"maxfev": n_nelder, "xatol": 1e-2, "fatol": 1e-3})
    best = min(history, key=lambda h: -h.score)
    return base.with_params(**best.params), history
