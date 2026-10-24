#!/usr/bin/env python
"""Print every constraint we care about, read PROGRAMMATICALLY from a Pulser
device object.

    python scripts/device_limits.py                      # all shipped devices
    python scripts/device_limits.py WeightedAnalogDevice # one device
    python scripts/device_limits.py --json device.json   # a device serialised with to_abstract_repr()
    python scripts/device_limits.py --cloud FRESNEL_CAN1 # fetch specs from Pasqal cloud (env creds)
    python scripts/device_limits.py X --omega 6.28       # also print blockade radius at a given Omega
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys


def load_device(name: str | None, json_path: str | None, cloud: str | None):
    import pulser
    if json_path:
        return pulser.devices.Device.from_abstract_repr(open(json_path).read())
    if cloud:
        import os
        from pasqal_cloud import PasqalCloudConnection
        conn = PasqalCloudConnection(username=os.environ["PASQAL_USERNAME"], password=os.environ["PASQAL_PASSWORD"],
                                     project_id=os.environ["PASQAL_PROJECT_ID"])
        devs = conn.fetch_available_devices()
        if cloud not in devs:
            sys.exit(f"cloud device {cloud!r} not available; have: {list(devs)}")
        return devs[cloud]
    import pulser.devices as pd
    if not hasattr(pd, name):
        sys.exit(f"no device {name!r} in pulser.devices; have: {[n for n in pd.__all__ if 'Device' in n]}")
    return getattr(pd, name)


def mhz(x):
    return "None" if x is None else f"{x:.4g} rad/µs = 2π·{x / (2 * math.pi):.4g} MHz"


def describe(dev, omega: float | None = None) -> dict:
    out = {}
    print(f"\n===== {dev.name}  ({type(dev).__name__}) — {getattr(dev, 'short_description', '')}")
    print("-- register")
    for f in ("dimensions", "min_atom_distance", "max_radial_distance", "max_atom_num", "requires_layout",
              "min_layout_traps", "max_layout_traps", "min_layout_filling", "max_layout_filling", "optimal_layout_filling",
              "supports_slm_mask", "rydberg_level", "max_sequence_duration", "max_runs"):
        v = getattr(dev, f, None)
        out[f] = v
        print(f"   {f:<26} {v}")
    c6 = dev.interaction_coeff
    out["interaction_coeff"] = c6
    print(f"   {'interaction_coeff (C6)':<26} {c6:.6g} rad/µs·µm^6   -> U at min distance = {c6 / max(dev.min_atom_distance, 1e-9) ** 6:.4g} rad/µs")
    if getattr(dev, "pre_calibrated_layouts", None):
        for L in dev.pre_calibrated_layouts:
            print(f"   pre-calibrated layout: {L} ({L.number_of_traps} traps)")
    print("-- channels")
    for cid, ch in dev.channels.items():
        d = {f.name: getattr(ch, f.name) for f in dataclasses.fields(ch) if f.name != "eom_config"}
        out[f"channel:{cid}"] = d
        print(f"   {cid}: {type(ch).__name__} addressing={ch.addressing}")
        print(f"      max_amp          {mhz(ch.max_amp)}")
        print(f"      max_abs_detuning {mhz(ch.max_abs_detuning)}")
        print(f"      min_avg_amp      {mhz(ch.min_avg_amp)}")
        print(f"      clock_period {ch.clock_period} ns, min_duration {ch.min_duration} ns, max_duration {ch.max_duration} ns, mod_bandwidth {ch.mod_bandwidth} MHz, rise_time {ch.rise_time} ns")
        if getattr(ch, "eom_config", None):
            print(f"      EOM: {ch.eom_config}")
    print("-- DMM (local detuning)")
    if not dev.dmm_channels:
        print("   NONE -> no local detuning on this device (diagonal QUBO terms collapse to a global detuning)")
    for cid, dmm in dev.dmm_channels.items():
        d = {f.name: getattr(dmm, f.name) for f in dataclasses.fields(dmm)}
        out[f"dmm:{cid}"] = d
        print(f"   {cid}: bottom_detuning {mhz(dmm.bottom_detuning)} (per atom, weight x waveform >= this)")
        print(f"         total_bottom_detuning {mhz(dmm.total_bottom_detuning)} (sum over atoms)")
        print(f"         min_avg_abs_detuning {mhz(dmm.min_avg_abs_detuning)} (min non-zero weight x avg|det| >= this)")
        print(f"         clock_period {dmm.clock_period} ns, min_duration {dmm.min_duration} ns, mod_bandwidth {dmm.mod_bandwidth} MHz")
    om = omega or dev.channels["rydberg_global"].max_amp
    if om:
        rb = dev.rydberg_blockade_radius(om)
        print(f"-- blockade radius at Omega={om:.4g} rad/µs: {rb:.3f} µm  (rabi_from_blockade({rb:.2f}) = {dev.rabi_from_blockade(rb):.4g})")
        out["blockade_radius"] = rb
    return out


# The hackathon brief's hardware-envelope table (§3.1) for AnalogDevice.
# Everything the cvrp/ package relies on is read from the device object at
# runtime; this table only exists so we can PROVE the brief and Pulser agree.
BRIEF_TABLE = {
    "max_atom_num": 80,
    "min_atom_distance": 5.0,          # µm
    "max_radial_distance": 38.0,       # µm
    "max_amp": 12.57,                  # rad/µs  (2π·2 MHz)
    "max_abs_detuning": 125.66,        # rad/µs  (2π·20 MHz)
    "max_sequence_duration": 6000,     # ns
    "interaction_coeff": 8.657e5,      # rad/µs·µm^6
}


def assert_brief(dev) -> None:
    """Assert that Pulser's AnalogDevice matches the brief's §3.1 table
    (relative tolerance 1e-3 for the rounded numbers).  Exits non-zero on
    mismatch so `make deliverables` can gate on it."""
    ch = dev.channels["rydberg_global"]
    got = {
        "max_atom_num": dev.max_atom_num,
        "min_atom_distance": dev.min_atom_distance,
        "max_radial_distance": dev.max_radial_distance,
        "max_amp": ch.max_amp,
        "max_abs_detuning": ch.max_abs_detuning,
        "max_sequence_duration": dev.max_sequence_duration,
        "interaction_coeff": dev.interaction_coeff,
    }
    bad = []
    for k, want in BRIEF_TABLE.items():
        g = got[k]
        ok = g is not None and math.isclose(float(g), float(want), rel_tol=1e-3)
        print(f"   {'OK ' if ok else 'BAD'} {k:<24} brief={want!s:<10} pulser={g}")
        if not ok:
            bad.append(k)
    if dev.dmm_channels:
        bad.append("dmm_channels (brief: global detuning only)")
    print(f"   {'OK ' if not dev.dmm_channels else 'BAD'} {'no DMM (global δ only)':<24} pulser dmm_channels={list(dev.dmm_channels)}")
    if bad:
        sys.exit(f"device/brief mismatch: {bad}")
    print("   brief table matches pulser's AnalogDevice")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("device", nargs="?", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--cloud", default=None)
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--dump", default=None, help="write the numbers to this JSON file")
    ap.add_argument("--assert-brief", action="store_true",
                    help="assert pulser's AnalogDevice matches the hackathon brief table (exit 1 if not)")
    a = ap.parse_args(argv)
    import pulser
    print(f"pulser {pulser.__version__}")
    if a.assert_brief:
        import pulser.devices as pd
        print("-- brief §3.1 table vs pulser AnalogDevice")
        assert_brief(pd.AnalogDevice)
        return
    if a.device or a.json or a.cloud:
        devs = [load_device(a.device, a.json, a.cloud)]
    else:
        import pulser.devices as pd
        devs = [getattr(pd, n) for n in ("AnalogDevice", "WeightedAnalogDevice", "DigitalAnalogDevice", "MockDevice") if hasattr(pd, n)]
    dump = {}
    for d in devs:
        dump[d.name] = describe(d, a.omega)
    if a.dump:
        json.dump(dump, open(a.dump, "w"), indent=2, default=str)
        print(f"\nwritten {a.dump}")


if __name__ == "__main__":
    main()
