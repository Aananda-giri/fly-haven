"""Convert a heaven.simulate timeline.parquet to .npz, since Blender's bundled
Python has numpy but not pandas/pyarrow. Run with the project's own venv:

    uv run python -m heaven.blender.export_timeline runs/fly-heaven/day1/timeline.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def export(parquet_path, out_path=None):
    df = pd.read_parquet(parquet_path)
    out_path = Path(out_path) if out_path else Path(parquet_path).with_suffix(".npz")
    arrays = {c: df[c].to_numpy() for c in df.columns}
    np.savez(out_path, **arrays)
    print(f"wrote {out_path} ({len(df)} rows, {len(df.columns)} columns)")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("parquet_path")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    export(args.parquet_path, args.out)
