"""Train a bounded V3 pilot using a saved V2 run's verified fly interface.

This reuses the graph/data and selected neurons, not incompatible graft weights.
The output checkpoint can be evaluated with evaluate_fly_chess.py. The pilot is
small by design; held-out results, not its memorization check, measure learning.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flychess.checkpoint import load_run
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("runs/fly-chess-v2/data"))
    parser.add_argument("--graph", type=Path, default=Path("data/fruitless/full-graph.npz"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--pool", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--minutes", type=float, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--relay-kind", choices=("sensory","central"), default="sensory")
    parser.add_argument("--relay-steps", type=int, default=3)
    parser.add_argument("--value-weight", type=float, default=4)
    args = parser.parse_args()
    if min(args.steps, args.pool, args.batch, args.minutes, args.relay_steps, args.value_weight) <= 0 or args.pool < 64:
        parser.error("Use positive settings and a pool of at least 64")
    args.out.mkdir(parents=True, exist_ok=True)
    ns = load_run(args.source_run, args.data, args.graph, args.device)
    old_model = ns["model"]
    # Load definitions only: no full notebook stage is executed here.
    notebook = Path(__file__).resolve().parents[1] / "fly_chess_colab_V3.ipynb"
    pinned_source = args.out / "model-source.ipynb"
    if pinned_source.exists():
        notebook = pinned_source
    else:
        shutil.copyfile(notebook, pinned_source)
    cells = json.loads(notebook.read_text())["cells"]
    ns.update(os=os, random=random, SEED=0, ROOT=args.out, RUN_ID="pilot", MODE="full",
              RUN_DIR=args.out, BATCH_SIZE=args.batch, pickle=__import__("pickle"))
    for tag in ("recovery", "brain", "models", "training_definitions"):
        cell = next(c for c in cells if tag in c["metadata"].get("tags", []))
        nodes = [n for n in ast.parse("".join(cell["source"])).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), "exec"), ns)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    brain = ns["FlyBrain"](old_model.brain.Wt, old_model.brain.gain, old_model.brain.alpha)
    if args.relay_kind == "sensory":
        ns["RELAY_IDX"] = np.random.RandomState(4).choice(ns["SENSORY_IDX"], min(8000,len(ns["SENSORY_IDX"])), replace=False)
    model = ns["FlyChessModel"](brain, ns["RELAY_IDX"], ns["PREMOTOR_IDX"], ns["MOTOR_IDX"],
        old_model.active_mask, old_model.T_p, old_model.T_a, relay_steps=args.relay_steps).to(args.device)
    del ns["model"], old_model
    manifest = dict(ns["manifest"])
    manifest.update(version=3, current_amplitude=.5,
        architecture=json.loads(notebook.read_text()).get("metadata",{}).get("fly_chess",{}).get("architecture",
            "early-relay-summary-residual-attention-linear-motor-v3"),
        pilot_source_run=str(args.source_run), batch_size=args.batch, steps_target=args.steps,
        pool_size=args.pool, seed=0, graft={}, torch=torch.__version__, python=sys.version, device=args.device,
        relay_steps=model.relay_steps, relay_kind=args.relay_kind)
    node_ids = np.load(args.data / "connectome/full-nodes.npz")["ids"]
    manifest["relay_ids"] = node_ids[ns["RELAY_IDX"]].tolist()
    update_kwargs = {}
    if "value_weight" in inspect.signature(ns["training_update"]).parameters:
        update_kwargs["value_weight"] = args.value_weight
        manifest["value_weight"] = args.value_weight
    else:
        print("Pinned experiment uses its original value-loss weight", flush=True)
    path = args.out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Pilot configuration changed; choose another output directory")
    ns["atomic_json"](path, manifest)
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    positions_path = args.data / "fly-chess/positions-v2-full-1000000.npz"
    with positions_path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != manifest["dataset_sha256"]:
            raise ValueError("Training dataset differs from the source run")
    positions = np.load(positions_path, allow_pickle=False)
    ns.update(fens=positions["fen"], labels_uci=positions["move"], values=positions["value"],
              train_idx=np.flatnonzero(positions["split"] == "train"))
    pool = np.random.RandomState(0).choice(ns["train_idx"], min(args.pool, len(ns["train_idx"])), replace=False)
    validation = np.flatnonzero(positions["split"] == "validation")[:256]
    checkpoint = ns["load_checkpoint"](args.out / "main.pt", identity)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        ns["initialize_signal_stats"](model, pool[:64])
    cache = ns["cache_perception"](model, "main", pool, identity)
    if not checkpoint:
        ns["learning_preflight"](model, cache)
    sampler = np.random.RandomState(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    step, active_s = 0, 0.0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        ns["restore_rng"](checkpoint["rng"], sampler)
        step, active_s = checkpoint["step"], checkpoint["active_s"]
    report_path = args.out / "pilot-results.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "purpose": "Small real-graph validation pilot; not an Elo measurement", "pool": len(pool),
        "before": ns["move_match"](model, validation), "validation": []}
    started = time.monotonic()
    last_save = started
    def save():
        ns["save_checkpoint"](args.out / "main.pt", {"identity": identity, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "step": step, "active_s": active_s, "rng": ns["rng_state"](sampler)})
        ns["atomic_json"](report_path, report)
    try:
        while step < args.steps and time.monotonic() - started < args.minutes*60:
            tick = time.monotonic()
            metrics = ns["training_update"](model, optimizer, sampler, cache, **update_kwargs)
            active_s += time.monotonic() - tick
            step += 1
            if step % 100 == 0 or step == args.steps:
                heldout = ns["move_match"](model, validation)
                report["validation"].append({"step": step, "training": metrics, **heldout})
                print("validation", step, heldout, "training", metrics, flush=True)
            if time.monotonic()-last_save >= 60:
                save(); last_save=time.monotonic()
        report["status"] = "complete" if step == args.steps else "paused"
    except KeyboardInterrupt:
        report["status"] = "paused"
    finally:
        report.update(step=step, active_training_seconds=active_s)
        save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
