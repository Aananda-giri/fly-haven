"""Run the closed loop: connectome -> behaviour -> world -> connectome, and log it.

Each 50 ms tick: the world reports sensory currents from the fly's situation,
the full MaleCNS brain advances 50 ms of neural time under those currents,
`heaven.behaviour` decides a motor program from the resulting firing rates,
and the world applies it. Nothing about *when* the fly feeds, grooms, escapes
or courts is scripted; only the female, navigation and cosmetic poses are.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import world as W
from .behaviour import FlyBehavior, load_thresholds
from .brain import RUNS, TICK_MS, ConnectomeBrain

DT = TICK_MS / 1000.0


def simulate(seconds=240.0, seed=0, out=None, lesion=(), hunger_start=0.35, verbose=True, thresholds=None, brain_factory=ConnectomeBrain):
    thresholds = thresholds if thresholds is not None else load_thresholds()
    brain = brain_factory(silence=lesion)
    world = W.World(seed=seed, hunger_start=hunger_start)
    behavior = FlyBehavior(thresholds=thresholds, seed=seed)

    rows, events = [], []
    n_ticks = int(round(seconds / DT))
    last_state = (behavior.state, behavior.subphase)
    started = time.perf_counter()
    for tick in range(n_ticks):
        currents = world.sense(DT)
        frame = world.retina_frame()
        rates = brain.step(frame, currents, ms=TICK_MS)
        snapshot = world.snapshot()
        action = behavior.tick(rates, DT, snapshot)
        world.act(DT, action)

        state_key = (action["state"], action["subphase"])
        if state_key != last_state:
            events.append({"t": round(snapshot["t"], 3), "from": last_state, "to": state_key, "rates": {k: round(v, 2) for k, v in rates.items() if k != "network_spikes"}})
            last_state = state_key

        rows.append(
            {
                "t": snapshot["t"],
                "state": action["state"],
                "subphase": action["subphase"],
                "pose": action["pose"],
                "fly_x": snapshot["fly_x"],
                "fly_y": snapshot["fly_y"],
                "fly_heading": snapshot["fly_heading"],
                "fly_altitude": snapshot["fly_altitude"],
                "female_x": snapshot["female_x"],
                "female_y": snapshot["female_y"],
                "female_present": snapshot["female_present"],
                "leaf_x": snapshot["leaf_x"],
                "leaf_y": snapshot["leaf_y"],
                "leaf_height": snapshot["leaf_height"],
                "proboscis": action["proboscis"],
                "grooming": action["grooming"],
                "wing_song": action["wing_song"],
                "flying": action["flying"],
                "turn_bias": action["turn_bias"],
                "satiety": snapshot["satiety"],
                "temperature_c": snapshot["temperature_c"],
                "dust": snapshot["dust"],
                "song_time": snapshot["song_time"],
                "MN9_hz": rates["MN9"],
                "DNg12_hz": rates["DNg12"],
                "GF_hz": rates["GF"],
                "P1_hz": rates["P1"],
                "pIP10_hz": rates["pIP10"],
                "MDN_hz": rates["MDN"],
                "motor_hz": rates["motor"],
                "DNa02_L_hz": rates["DNa02_L"],
                "DNa02_R_hz": rates["DNa02_R"],
                "network_spikes": rates["network_spikes"],
            }
        )
        if verbose and tick % (20 * 10) == 0:
            print(f"t={snapshot['t']:6.1f}s  state={action['state']:10s} {action['subphase'] or '':10s}  "
                  f"({time.perf_counter() - started:.0f}s wall, {tick + 1}/{n_ticks} ticks)", flush=True)

    timeline = pd.DataFrame(rows)
    out = Path(out) if out else RUNS / "day1"
    out.mkdir(parents=True, exist_ok=True)
    timeline.to_parquet(out / "timeline.parquet", index=False)
    (out / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")

    counts = timeline.state.value_counts().to_dict()
    summary = {
        "seconds": seconds,
        "seed": seed,
        "lesion": list(lesion),
        "hunger_start": hunger_start,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "state_seconds": {k: round(v * DT, 1) for k, v in counts.items()},
        "fed": bool(timeline.state.eq("FEED").any()),
        "groomed": bool(timeline.state.eq("GROOM").any()),
        "escaped": bool(timeline.state.eq("ESCAPE").any()),
        "courted": bool(timeline.state.eq("COURT").any()),
        "mated": bool((timeline.subphase == "MATE").any()),
        "final_satiety": float(timeline.satiety.iloc[-1]),
        "kernel": brain.brain.build,
        "senses_and_readouts": brain.describe(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(json.dumps(summary, indent=2))
        print(f"wrote {out}")
    return timeline, summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=240.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--lesion", type=str, nargs="*", default=[], help="readouts to silence, e.g. MN9 P1")
    p.add_argument("--hunger-start", type=float, default=0.35)
    args = p.parse_args()
    simulate(seconds=args.seconds, seed=args.seed, out=args.out, lesion=args.lesion, hunger_start=args.hunger_start)
