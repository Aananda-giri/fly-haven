"""Load V2 weights directly, without rerunning calibration or training cells."""
import ast
import contextlib
import hashlib
import json
import math
from pathlib import Path
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.sparse import load_npz


def load_run(run_dir, data_dir, graph_path, device="cpu", notebook=None, checkpoint_name="main.pt"):
    run_dir, data_dir, graph_path = map(Path, (run_dir, data_dir, graph_path))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["version"] not in (2, 3, 4, 5, 5.5) or not manifest["real_graph"]:
        raise ValueError("This loader accepts real-graph V2–V5.5 runs")
    saved_source = run_dir / "model-source.ipynb"
    default_source = saved_source if saved_source.exists() else Path(__file__).resolve().parents[1] / f"fly_chess_colab_V{manifest['version']}.ipynb"
    notebook = Path(notebook or default_source)
    def digest(path):
        with Path(path).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    # Verify the reconstructed interface before loading a checkpoint.
    if digest(graph_path) != manifest["graph_sha256"]:
        raise ValueError("Graph hash differs from the trained graph")
    nodes_path = data_dir / "connectome/full-nodes.npz"
    if digest(nodes_path) != manifest["nodes_sha256"]:
        raise ValueError("Node ordering/signs differ from the trained interface")
    ns = dict(np=np, torch=torch, nn=nn, F=F, chess=chess, contextlib=contextlib,
              hashlib=hashlib, math=math, time=time, json=json, Path=Path,
              DEVICE=torch.device(device), MODE=manifest["mode"], RUN_DIR=run_dir)
    def check_stop():
        if (run_dir / "STOP").exists():
            raise KeyboardInterrupt("Run STOP file is present")
    ns["check_stop"] = check_stop
    def atomic_json(path, value):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
        temporary.replace(path)
    ns["atomic_json"] = atomic_json
    source_document = json.loads(notebook.read_text())
    if manifest["version"] in (4,5,5.5):
        source_hash = hashlib.sha256(json.dumps(source_document,sort_keys=True).encode()).hexdigest()
        if source_hash != manifest["source_sha256"]:
            raise ValueError("Saved model implementation differs from the trained source")
    cells = source_document["cells"]
    tags = ["encoding"]
    if manifest["version"] == 5.5:tags.append("ranked_definitions")
    tags += ["brain", "injection", "models", "control_flyonly", "training_definitions", "search", "scoring", "matches"]
    if manifest["version"] == 2:tags.remove("control_flyonly")
    for tag in tags:
        # The corrected match loop resolves a terminal final ply. It does not alter weights.
        definition_cells = cells
        if tag == "matches":
            definition_cells = json.loads((Path(__file__).resolve().parents[1] / "fly_chess_colab_V3.ipynb").read_text())["cells"]
        cell = next(c for c in definition_cells if tag in c.get("metadata", {}).get("tags", []))
        tree = ast.parse("".join(cell["source"]))
        body = tree.body if tag in ("encoding","ranked_definitions") else [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(notebook), "exec"), ns)
    nodes = np.load(nodes_path, allow_pickle=False)
    ids = nodes["ids"]
    for field, name in (("sensory_ids", "SENSORY_IDX"), ("relay_ids", "RELAY_IDX"),
                        ("premotor_ids", "PREMOTOR_IDX"), ("motor_ids", "MOTOR_IDX")):
        indices = np.searchsorted(ids, manifest[field])
        if np.any(indices >= len(ids)) or not np.array_equal(ids[indices], manifest[field]):
            raise ValueError(f"Unknown neuron in {field}")
        ns[name] = indices
    ns["N_NEURONS"] = len(ids)
    matrix, _ = ns["build_signed_transpose"](load_npz(graph_path), nodes["signs"], manifest["threshold"])
    brain = ns["FlyBrain"](matrix.to(device), manifest["gain"], manifest["alpha"])
    ns["INJECT_MAP"] = ns["build_injection_map"](ns["SENSORY_IDX"], ns["N_FEATURES"], manifest["injection_seed"])
    ns["INJECT_AMPLITUDE"] = 1.0
    mask = torch.zeros(len(ids), dtype=torch.bool, device=device)
    mask[ns["PREMOTOR_IDX"]] = mask[ns["MOTOR_IDX"]] = True
    model_kwargs = dict(manifest["graft"])
    if "relay_steps" in manifest:
        model_kwargs["relay_steps"] = manifest["relay_steps"]
    if manifest["version"] == 5.5:
        model = ns["FlyOnlyModel"](brain,ns["MOTOR_IDX"],manifest["perception_steps"]).to(device)
    else:
        model = ns["FlyChessModel"](brain, ns["RELAY_IDX"], ns["PREMOTOR_IDX"], ns["MOTOR_IDX"],
            mask, manifest["perception_steps"], manifest["motor_steps"],
            current_amplitude=manifest.get("current_amplitude", 2.0), **model_kwargs).to(device)
    # The file is user-supplied experiment data, not a downloaded checkpoint.
    checkpoint = torch.load(run_dir / checkpoint_name, map_location="cpu", weights_only=False)
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if checkpoint["identity"] != identity:
        raise ValueError("Checkpoint identity differs from manifest")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ns.update(model=model, manifest=manifest, BATCH_SIZE=32, STOCKFISH_BINARY_SHA=manifest["engine_sha256"])
    return ns
