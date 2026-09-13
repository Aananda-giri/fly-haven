"""Measure how each sensory current drives its readout, and derive decoder thresholds.

The connectome kernel is deterministic given its input (no RNG in kernel.cpp): the
same frame and currents from the same state always give the same spikes. So this
calibrates over a few different *visual* warm-up states, not over random seeds.
"""

import argparse
import json
import time

import numpy as np

from .brain import RUNS, TICK_MS, ConnectomeBrain

# sense -> (primary readout, expected direction on that readout)
ROUTES = {
    "sugar": "MN9",
    "antenna": "DNg12",
    "loom": "GF",
    "heat": "MDN",
    "cold": "MDN",
    "female": "P1",
    "male": "P1",
}
PROBE_MV = [10.0, 20.0]
WARMUP_S = 0.5
PROBE_S = 0.3


def _frame(kind):
    frame = np.zeros((160, 90, 3), np.uint8)
    if kind == "bright":
        frame[:] = (235, 235, 220)
    elif kind == "dark":
        frame[:] = (15, 15, 20)
    else:  # dappled: bright sky over dim ground, like a forest floor
        frame[:70] = (200, 220, 235)
        frame[70:] = (60, 80, 45)
    return frame


def run(warmups=("dappled", "bright", "dark"), out=None, verbose=True):
    brain = ConnectomeBrain()
    rows = []
    thresholds = {}
    for warmup in warmups:
        frame = _frame(warmup)
        brain.brain.reset()
        brain.step(frame, {}, ms=WARMUP_S * 1000)
        base_state = brain.snapshot()
        baseline = brain.step(frame, {}, ms=PROBE_S * 1000)
        for sense, readout in ROUTES.items():
            for mv in PROBE_MV:
                brain.restore(base_state)
                t0 = time.perf_counter()
                evoked = brain.step(frame, {sense: mv}, ms=PROBE_S * 1000)
                rows.append(
                    {
                        "warmup": warmup,
                        "sense": sense,
                        "readout": readout,
                        "mV": mv,
                        "baseline_hz": baseline[readout],
                        "evoked_hz": evoked[readout],
                        "baseline_rates": baseline,
                        "evoked_rates": evoked,
                        "seconds": round(time.perf_counter() - t0, 2),
                    }
                )
                if verbose:
                    print(f"{warmup:8s} {sense:8s}@{mv:g}mV -> {readout}: {baseline[readout]:.1f} -> {evoked[readout]:.1f} Hz")
    # The full network's baseline rate shifts with ambient light alone (a flat white
    # warm-up lifted the MN9 and MDN baselines well above their normal ~0 Hz rest,
    # confirming fly-wirehead's own note that "motor firing was nonzero" from vision
    # alone). So every sense is judged by its *paired* delta over that same warm-up's
    # own baseline, at the strongest (20 mV) probe, with the median across warm-ups
    # rather than the min: one noisy 0.3 s probe should not veto an otherwise clear
    # route. The full per-warmup, per-mV table above stays in the report either way.
    strong_mv = PROBE_MV[-1]

    def paired_deltas(sense, readout=None, mv=strong_mv):
        readout = readout or ROUTES[sense]
        out = []
        for w in warmups:
            base = next(rr["baseline_rates"][readout] for rr in rows if rr["sense"] == sense and rr["warmup"] == w)
            ev = next(rr["evoked_rates"][readout] for rr in rows if rr["sense"] == sense and rr["warmup"] == w and rr["mV"] == mv)
            out.append(ev - base)
        return out

    for sense in ROUTES:
        thresholds[sense] = round(max(0.1, float(np.median(paired_deltas(sense))) / 2), 3)

    readout_thresholds = {
        "MN9": thresholds["sugar"],
        "DNg12": thresholds["antenna"],
        "GF": thresholds["loom"],
        "MDN": min(thresholds["heat"], thresholds["cold"]),
        "P1": thresholds["female"],
        "pIP10": round(max(0.1, float(np.median(paired_deltas("female", "pIP10"))) / 2), 3),
    }

    def majority_positive(sense):
        return sum(d > 0 for d in paired_deltas(sense)) >= 2

    checks = {
        "sugar->MN9": majority_positive("sugar"),
        "antenna->DNg12": majority_positive("antenna"),
        "loom->GF": majority_positive("loom"),
        "heat->MDN": majority_positive("heat"),
        "female_P1 > male_P1": float(np.median(paired_deltas("female"))) > float(np.median(paired_deltas("male"))),
    }
    failed = [k for k, ok in checks.items() if not ok]
    result = {
        "kernel": brain.brain.build,
        "senses_and_readouts": brain.describe(),
        "thresholds_hz": thresholds,
        "readout_thresholds_hz": readout_thresholds,
        "checks": checks,
        "rows": rows,
    }
    out = out or RUNS / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    if verbose:
        print(f"\nthresholds: {thresholds}")
        print(f"checks: {checks}")
        print(f"wrote {out}")
    if failed:
        raise SystemExit(f"Calibration routes failed: {failed}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    run(out=args.out)
