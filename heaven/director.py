"""Create continuous chronological coverage; --highlights opts into the legacy cut.

Every clip boundary comes from the timeline's own `state`/`subphase` columns --
i.e. from where the connectome actually gated a behaviour on or off, not a
handpicked timestamp. This only decides how much lead-in/lead-out context to
keep around each real event, one clip per behaviour, in story order.

Run with the project's venv (pandas, no Blender needed):
    uv run python -m heaven.director runs/fly-heaven/final/timeline.parquet runs/fly-heaven/final/edl.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 24  # heaven.blender.animate.FPS
LEAD_IN_S = 2.0
LEAD_OUT_S = 2.0
MATE_TIMELAPSE_STEP = 5  # render every 5th frame during a held MATE bout -> a 5x time-lapse


def _runs(df, mask):
    """Contiguous [start_t, end_t] spans where `mask` is true."""
    idx = np.flatnonzero(mask.to_numpy())
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(idx) - 1]
    t = df["t"].to_numpy()
    return [(float(t[idx[s]]), float(t[idx[e]])) for s, e in zip(starts, ends)]


def _longest(runs):
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def plan_clips(df):
    clips = []
    end_of_sim = float(df["t"].max())

    clips.append({"label": "A fly wakes up", "start": 0.0, "end": min(9.0, end_of_sim)})

    feed = _longest(_runs(df, df.state.eq("FEED")))
    if feed:
        clips.append({"label": "Feeding", "start": max(0, feed[0] - LEAD_IN_S), "end": min(end_of_sim, feed[1] + LEAD_OUT_S)})

    groom = _longest(_runs(df, df.state.eq("GROOM")))
    if groom:
        clips.append({"label": "Grooming", "start": max(0, groom[0] - LEAD_IN_S), "end": min(end_of_sim, groom[1] + LEAD_OUT_S)})

    bask = _longest(_runs(df, df.state.eq("BASK")))
    if bask and bask[1] - bask[0] > 1.5:
        clips.append({"label": "Basking", "start": max(0, bask[0] - LEAD_IN_S), "end": min(end_of_sim, bask[1] + 1.0)})

    escape = _longest(_runs(df, df.state.eq("ESCAPE")))
    if escape:
        clips.append({"label": "A leaf falls -- escape!", "start": max(0, escape[0] - 3.0), "end": min(end_of_sim, escape[1] + LEAD_OUT_S)})

    court_runs = _runs(df, df.state.eq("COURT"))
    court = _longest(court_runs)
    if court:
        mate_runs = _runs(df, df.subphase.eq("MATE") & df.state.eq("COURT"))
        mate = _longest(mate_runs)
        if mate and mate[1] - mate[0] > 3.0:
            # Split around the held MATE bout so only that stretch is time-lapsed.
            clips.append({"label": "Courtship", "start": max(0, court[0] - LEAD_IN_S), "end": mate[0]})
            clips.append({"label": f"Mating ({MATE_TIMELAPSE_STEP}x)", "start": mate[0], "end": mate[1], "step": MATE_TIMELAPSE_STEP})
            clips.append({"label": "Afterward", "start": mate[1], "end": min(end_of_sim, court[1] + LEAD_OUT_S)})
        else:
            clips.append({"label": "Courtship", "start": max(0, court[0] - LEAD_IN_S), "end": min(end_of_sim, court[1] + LEAD_OUT_S)})

    for c in clips:
        c["start_frame"] = int(round(c["start"] * FPS))
        c["end_frame"] = max(c["start_frame"], int(round(c["end"] * FPS)))
    return clips


def build(timeline_path, out_path, highlights=False):
    df = pd.read_parquet(timeline_path)
    clips = plan_clips(df) if highlights else [{"label": "Fly Heaven", "start": 0.0, "end": float(df.t.max()), "start_frame": 0, "end_frame": int(round(float(df.t.max()) * FPS))}]
    ranges = [[c["start_frame"], c["end_frame"], c.get("step", 1)] for c in clips]
    total_out_frames = sum(max(1, (b - a) // s + 1) for a, b, s in ranges)
    edl = {
        "mode": "highlights" if highlights else "continuous",
        "clips": clips,
        "ranges": ranges,
        "fps": FPS,
        "total_output_frames": total_out_frames,
        "estimated_seconds": round(total_out_frames / FPS, 1),
    }
    Path(out_path).write_text(json.dumps(edl, indent=2))
    print(f"{len(clips)} clips, {total_out_frames} output frames (~{edl['estimated_seconds']}s) -> {out_path}")
    for c in clips:
        print(f"  {c['label']:28s} {c['start']:7.1f}s - {c['end']:7.1f}s" + (f"  [{c['step']}x]" if c.get("step", 1) > 1 else ""))
    return edl


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("timeline_path")
    p.add_argument("out_path")
    p.add_argument("--highlights", action="store_true", help="Opt into the legacy reordered highlight edit")
    args = p.parse_args()
    build(args.timeline_path, args.out_path, args.highlights)
