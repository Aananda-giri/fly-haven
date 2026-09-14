"""Shared V4/V5 data, curriculum, checkpoint and self-play implementation.

The notebook builder embeds this file verbatim. The namespace argument supplies
the notebook's frozen fly simulator, model, encoding, and tested MCTS functions.
"""
import copy
import gzip
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import shutil
import sqlite3
import time

import chess
import chess.engine
import numpy as np
import torch
from torch.nn import functional as F


def policy_distribution(scores, temperature=80.0):
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0 or not np.isfinite(scores).all() or temperature <= 0:
        raise ValueError("Finite, nonempty move scores and positive temperature required")
    weights = np.exp((scores - scores.max()) / temperature)
    return (weights / weights.sum()).astype(np.float32)


def ranking_score(cp=None, mate=None):
    if mate is not None:
        return (10000 - min(abs(mate), 90)*100) * (1 if mate > 0 else -1)
    return float(np.clip(cp, -2000, 2000))


def soft_policy_loss(logits, move_indices, probabilities):
    indices = torch.as_tensor(move_indices, device=logits.device, dtype=torch.long)
    weights = torch.as_tensor(probabilities, device=logits.device, dtype=torch.float32)
    if indices.ndim != 2 or weights.shape != indices.shape or indices.shape[0] != len(logits):
        raise ValueError("Policy targets must have shape (batch, candidates)")
    if torch.any(weights < 0) or not torch.allclose(weights.sum(1), torch.ones(len(logits), device=logits.device, dtype=torch.float32), atol=.002):
        raise ValueError("Target probabilities must sum to one")
    gathered = F.log_softmax(logits.float(), dim=1).gather(1, indices)
    if not torch.isfinite(gathered[weights > 0]).all():
        raise ValueError("A target move is illegal or its probability is nonfinite")
    return -(torch.where(weights > 0, gathered, torch.zeros_like(gathered)) * weights).sum(1).mean()


def curriculum_weights(phase):
    # Foundation, tactics, endgame, ordinary. Earlier skills remain in later phases.
    return {"foundation": (.55, .20, .20, .05),
            "general": (.15, .25, .25, .35),
            "consolidation": (.10, .25, .25, .40)}[phase]


def stratified_pool(groups, count, rng, phase="general"):
    chosen = []
    weights = curriculum_weights(phase)
    available = [np.asarray(g, dtype=np.int64) for g in groups]
    if not any(len(g) for g in available):
        raise ValueError("Empty training split")
    for group, weight in zip(available, weights):
        if len(group):
            chosen.extend(rng.choice(group, min(len(group), int(count*weight)), replace=False).tolist())
    chosen = np.unique(np.asarray(chosen, dtype=np.int64))
    total = np.unique(np.concatenate(available))
    if len(chosen) < min(count, len(total)):
        rest = np.setdiff1d(total, chosen)
        chosen = np.concatenate([chosen, rng.choice(rest, min(count-len(chosen), len(rest)), replace=False)])
    rng.shuffle(chosen)
    return chosen


def training_groups(data, train_indices):
    groups = [[], [], [], []]
    fens, moves = data["fen"], data["move"]
    for offset, i in enumerate(train_indices):
        board = chess.Board(str(fens[i]))
        move = chess.Move.from_uci(str(moves[i]))
        pieces = len(board.piece_map())
        short_mate = bool(data["is_mate"][i]) and abs(int(data["raw_mate"][i])) <= 2
        category = 0 if short_mate else 1 if board.is_check() or board.is_capture(move) or move.promotion else 2 if pieces <= 10 else 3
        groups[category].append(int(i))
        if offset and offset % 100000 == 0:
            print("curriculum indexing", offset, "/", len(train_indices), flush=True)
    return [np.asarray(g, dtype=np.int64) for g in groups]


def ingest_advanced(lines, limit, min_depth, ns):
    rows, seen, rejected = [], set(), 0
    for raw in lines:
        ns["check_stop"]()
        try:
            source = json.loads(raw)
            board = chess.Board(source["fen"])
            if not board.is_valid() or ns["terminal_value"](board) is not None:
                raise ValueError("invalid_or_terminal")
            key = ns["position_key"](board)
            if key in seen:
                continue
            best = max(source["evals"], key=lambda e: (e["depth"], e.get("knodes", 0)))
            if best["depth"] < min_depth:
                continue
            candidates, scores = [], []
            sign = 1 if board.turn else -1
            for pv in best["pvs"]:
                move = board.parse_uci(pv["line"].split()[0])
                if move.uci() in candidates:
                    continue
                candidates.append(move.uci())
                scores.append(ranking_score(sign*pv["cp"] if "cp" in pv else None,
                                           sign*pv["mate"] if "mate" in pv else None))
                if len(candidates) == 3:
                    break
            probabilities = policy_distribution(scores)
            pv = best["pvs"][0]
            rows.append((board.fen(en_passant="fen"), candidates[0],
                         ns["source_value"](board, pv.get("cp"), pv.get("mate")),
                         best["depth"], pv.get("cp", 0), pv.get("mate", 0), "mate" in pv,
                         key, ns["partition"](board),
                         candidates + [candidates[0]]*(3-len(candidates)),
                         probabilities.tolist() + [0.]*(3-len(candidates))))
            seen.add(key)
        except (ValueError, KeyError, IndexError, TypeError):
            rejected += 1
        if len(rows) >= limit:
            break
    return rows, rejected


def prepare_advanced_data(ns):
    count = int(os.environ.get("FLY_CHESS_POSITIONS", "2000" if ns["MODE"] == "smoke" else "1000000"))
    path = ns["CHESS_DATA"] / f"positions-v4-{ns['MODE']}-{count}.npz"
    pin_path = path.with_suffix(".json")
    if not path.exists():
        legacy = Path(os.environ.get("FLY_CHESS_SOURCE_DATASET", str(ns["CHESS_DATA"] / f"positions-v2-{ns['MODE']}-{count}.npz")))
        if legacy.exists():
            legacy_pin = json.loads(legacy.with_suffix(".json").read_text())
            if ns["sha256"](legacy) != legacy_pin["sha256"]:
                raise ValueError("Source dataset integrity mismatch")
            with np.load(legacy, allow_pickle=False) as old:
                arrays = {k:old[k] for k in old.files}
            if "policy_moves" not in arrays:
                arrays["policy_moves"] = np.repeat(arrays["move"][:, None], 3, axis=1)
                arrays["policy_probs"] = np.tile(np.array([1, 0, 0], np.float32), (len(arrays["move"]), 1))
            # V4/V5 use the same canonical split as V3. Cached source labels are
            # explicitly identified as single-move/proxy labels until relabelled.
            origin = {"source": str(legacy), "source_sha256": legacy_pin["sha256"], "policy": "copied_source_multipv" if legacy_pin.get("schema") == 4 else "single_move_source"}
        else:
            import contextlib
            with contextlib.closing(ns["stream_zst_lines"]("https://database.lichess.org/lichess_db_eval.jsonl.zst")) as lines:
                rows, rejected = ingest_advanced(lines, count, 10 if ns["MODE"] == "smoke" else 14, ns)
            if not rows:
                raise ValueError("No accepted positions")
            names = ("fen", "move", "value", "depth", "raw_cp", "raw_mate", "is_mate", "key", "split", "policy_moves", "policy_probs")
            arrays = {name: np.array(col) for name, col in zip(names, zip(*rows))}
            origin = {"source": "https://database.lichess.org/lichess_db_eval.jsonl.zst", "rejected": rejected, "policy": "available_source_multipv"}
        np.savez_compressed(path, **arrays)
        ns["atomic_json"](pin_path, {"schema": 4, "sha256": ns["sha256"](path), "origin": origin,
                                      "source_value": "side_to_move_centipawn_proxy; pinned teacher supplies WDL"})
    pin = json.loads(pin_path.read_text())
    if pin["schema"] != 4 or ns["sha256"](path) != pin["sha256"]:
        raise ValueError("V4 dataset integrity mismatch")
    with np.load(path, allow_pickle=False) as archive:
        data = {k:archive[k] for k in archive.files}
    ns.update(positions_path=path, positions=data, fens=data["fen"], labels_uci=data["move"], values=data["value"],
              train_idx=np.flatnonzero(data["split"] == "train"),
              validation_idx=np.flatnonzero(data["split"] == "validation"), test_idx=np.flatnonzero(data["split"] == "test"))
    if not all(len(ns[k]) for k in ("train_idx", "validation_idx", "test_idx")):
        raise ValueError("All three dataset splits must be populated")
    print("dataset", path, "positions", len(data["fen"]), "split sizes", *[len(ns[k]) for k in ("train_idx", "validation_idx", "test_idx")], flush=True)
    return data


class TeacherStore:
    """Local SQLite with atomic persistent snapshots; one writer per run."""
    def __init__(self, local, persistent, config):
        self.local, self.persistent = Path(local), Path(persistent)
        self.local.parent.mkdir(parents=True, exist_ok=True)
        self.persistent.parent.mkdir(parents=True, exist_ok=True)
        if not self.local.exists() and self.persistent.exists():
            shutil.copyfile(self.persistent, self.local)
        self.db = sqlite3.connect(self.local)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS labels (key TEXT PRIMARY KEY, value TEXT)")
        encoded = json.dumps(config, sort_keys=True)
        old = self.db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
        if old and old[0] != encoded:
            raise ValueError("Teacher configuration changed; start a new run")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('config', ?)", (encoded,))
        self.db.commit()
    def get(self, key):
        row = self.db.execute("SELECT value FROM labels WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
    def put(self, key, value):
        self.db.execute("INSERT OR IGNORE INTO labels VALUES (?, ?)", (key, json.dumps(value, allow_nan=False)))
        self.db.commit()
    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    def stage_done(self, key):
        return self.db.execute("SELECT 1 FROM metadata WHERE key=?", ("stage:"+key,)).fetchone() is not None
    def mark_stage(self, key):
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES (?, 'done')", ("stage:"+key,))
        self.db.commit()
    def snapshot(self):
        temporary = self.persistent.with_suffix(".tmp.sqlite")
        with sqlite3.connect(temporary) as destination:
            self.db.backup(destination)
        temporary.replace(self.persistent)
    def close(self):
        self.db.close()


def teacher_label(engine, board, nodes=50000, candidates=3):
    information = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=min(candidates, board.legal_moves.count()), game=object())
    if isinstance(information, dict):
        information = [information]
    moves, ranks = [], []
    for info in information:
        move = info["pv"][0]
        score = info["score"].pov(board.turn)
        moves.append(move.uci())
        ranks.append(ranking_score(score.score(), score.mate()))
    best = information[0]
    if "wdl" not in best:
        raise ValueError("Pinned teacher did not produce WDL; enable UCI_ShowWDL")
    wdl = best["wdl"].pov(board.turn)
    return {"moves": moves, "probs": policy_distribution(ranks).tolist(), "value": wdl.expectation(),
            "wdl": [wdl.wins, wdl.draws, wdl.losses], "depth": best.get("depth"),
            "nodes": best.get("nodes"), "value_source": "pinned_stockfish_wdl"}


def replay_board(record):
    board = chess.Board(record["start_fen"])
    history = record["history"].split() if isinstance(record["history"], str) else record["history"]
    for uci in history:
        board.push_uci(uci)
    return board


def record_moves(record):
    return record["moves"].split() if isinstance(record["moves"],str) else record["moves"]


def clone_checkpoint_model(ns, state, path):
    rng = ns["rng_state"](state["sampler"])
    try:
        saved = ns["load_checkpoint"](path,state["identity"])
        model = ns["make_advanced_model"]().to(ns["DEVICE"])
        model.load_state_dict(saved["model"])
        model.eval()
        return model
    finally:
        ns["restore_rng"](rng,state["sampler"])


def outcome_targets(samples, winner):
    return [dict(sample, value=.5 if winner is None else float(sample["turn"] == winner),
                 value_source="completed_selfplay_outcome") for sample in samples]


def search_target(ns, model, board, simulations, rng, exploration=False):
    priors, _ = ns["evaluate_leaves"](model, [board])
    if exploration:
        indices = list(priors[0])
        noise = rng.dirichlet(np.full(len(indices), .3))
        priors[0] = {i:.75*priors[0][i] + .25*float(n) for i,n in zip(indices,noise)}
    root = ns["search"](model, board, n_simulations=simulations, batch=min(16, simulations), root_priors=priors[0])
    indices = list(root.children)
    counts = np.array([root.children[i].visits for i in indices], np.float64)
    if not len(indices) or counts.sum() == 0:
        raise ValueError("Search produced no visit targets")
    return indices, (counts/counts.sum()).astype(np.float32), root


class AdvancedTrainer:
    def __init__(self, ns, config):
        self.ns, self.config = ns, config
        self.base = Path(ns["RUN_DIR"])
        self.device = ns["DEVICE"]
        self.smoke = ns["MODE"] == "smoke"
        self.round_size = config["round_size"]
        self.data = ns["positions"]
        self.groups = training_groups(self.data, ns["train_idx"])
        self.validation = np.random.RandomState(101).choice(ns["validation_idx"], min(config["validation_size"], len(ns["validation_idx"])), replace=False)
        teacher_config = {"engine_sha256": ns["STOCKFISH_BINARY_SHA"], "nodes": config["teacher_nodes"],
                          "multipv": 3, "threads": 1, "temperature_cp": 80}
        self.cache_scope = hashlib.sha256(str(self.base.resolve()).encode()).hexdigest()[:12]
        self.teacher = TeacherStore(ns["ROOT"] / "teacher-cache" / self.cache_scope / "labels.sqlite",
                                    self.base / "teacher.sqlite", teacher_config)
        physical = {"version": 4, "mode": ns["MODE"], "real_graph": ns["USE_REAL_GRAPH"],
                    "dataset_sha256": ns["sha256"](ns["positions_path"]),
                    "graph_sha256": ns["sha256"](ns["GRAPH_PATH"]) if ns["USE_REAL_GRAPH"] else "synthetic",
                    "nodes_sha256": ns["sha256"](ns["NODES_PATH"]) if ns["USE_REAL_GRAPH"] else "synthetic",
                    "threshold": ns["EDGE_MIN_SYNAPSES"], "gain": ns["GAIN"], "alpha": ns["ALPHA"],
                    "perception_steps": ns["T_P"], "motor_steps": ns["T_A"], "relay_steps": ns["RELAY_STEPS"],
                    "injection_seed": 0, "current_amplitude": .5, "features": ns["N_FEATURES"], "moves": ns["N_MOVES"],
                    "graft": ns["GRAFT_KW"], "architecture": "history-sensory-curriculum-v4-v5",
                    "source_sha256": ns["MODEL_SOURCE_SHA256"], "config": config,
                    "engine_sha256": ns["STOCKFISH_BINARY_SHA"], "batch_size": ns["BATCH_SIZE"]}
        for field, name in (("sensory_ids","SENSORY_IDX"),("relay_ids","RELAY_IDX"),("premotor_ids","PREMOTOR_IDX"),("motor_ids","MOTOR_IDX")):
            physical[field] = ns["ids"][ns[name]].tolist()
        self.cache_identity = hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()
        self.states = {}
        initial = copy.deepcopy(ns["model"].state_dict())
        for branch in config["branches"]:
            directory = self.base / branch
            directory.mkdir(exist_ok=True)
            manifest = dict(physical, version={"v4":4,"v5":5,"v55":5.5}[branch], branch=branch)
            if branch == "v55":
                manifest.update(architecture="ranked-lookahead-fly-only-v55", graft={}, current_amplitude=0,
                                assistance={"source":"material_minimax", "depth_plies":2, "ranked_k":4,
                                            "route":"sensory neurons only", "policy_choices":"all legal moves"})
            path = directory / "manifest.json"
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError("Run configuration changed; use FLY_CHESS_NEW_RUN=1 once")
            ns["atomic_json"](path, manifest)
            # Save the exact implementation alongside every branch checkpoint.
            if ns.get("NOTEBOOK_SOURCE"):
                ns["atomic_json"](directory / "model-source.ipynb", ns["NOTEBOOK_SOURCE"])
            identity = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
            construction_rng = ns["rng_state"](np.random.RandomState(0))
            model = ns["make_advanced_model"]().to(self.device)
            ns["restore_rng"](construction_rng, np.random.RandomState(0))
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=.01)
            scaler = torch.amp.GradScaler("cuda",enabled=self.device.type == "cuda",init_scale=128)
            state = {"model":model,"optimizer":optimizer,"scaler":scaler,"sampler":np.random.RandomState(0),
                     "step":0,"round":0,"phase":"foundation","best_loss":None,"stale":0,"status":"active",
                     "directory":directory,"identity":identity,"replay":[],"selfplay_games":0,"search_gate":False,
                     "collection_round":-1,"round_games":0,"collected_round":-1,"arena_pending":False,"champion_step":0}
            saved = ns["load_checkpoint"](directory / "main.pt",identity)
            if saved:
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scaler.load_state_dict(saved["scaler"])
                state.update({k:saved[k] for k in ("step","round","phase","best_loss","stale","status","selfplay_games","search_gate",
                                                     "collection_round","round_games","collected_round","arena_pending","champion_step")})
                state["sampler"].set_state(saved["sampler"])
                replay_path = directory / saved["replay_file"]
                if ns["sha256"](replay_path) != saved["replay_sha256"]:
                    raise ValueError("Replay snapshot integrity mismatch")
                with gzip.open(replay_path,"rt") as handle:state["replay"] = json.load(handle)
                if self.teacher.count() < saved["teacher_count"]:
                    raise ValueError("Teacher snapshot is older than the checkpoint")
                state["rng"] = saved["rng"]
            else:
                ns["initialize_signal_stats"](model, ns["train_idx"][:min(64,len(ns["train_idx"]))])
                # Every branch begins with the same stochastic training stream.
                torch.manual_seed(ns["SEED"])
                np.random.seed(ns["SEED"])
                state["rng"] = ns["rng_state"](state["sampler"])
                ns["save_checkpoint"](directory / "main.champion.pt",{
                    "identity":identity,"model":model.state_dict(),"step":0})
            self.states[branch] = state
        del initial
        # Remove the construction model to keep only the two active branches.
        ns.pop("model", None)
        self.cache = None
        self.cache_key = None

    def source_record(self, index):
        board = chess.Board(str(self.data["fen"][index]))
        key = board.fen(en_passant="fen")
        label = self.teacher.get(key)
        if not label:
            label = {"moves":self.data["policy_moves"][index].tolist(),
                     "probs":self.data["policy_probs"][index].tolist(),"value":float(self.data["value"][index]),
                     "value_source":"source_cp_proxy"}
        return {"start_fen":key,"history":[],"turn":board.turn,**label,"index":int(index)}

    def save(self, state):
        self.teacher.snapshot()
        replay_path = state["directory"] / f"replay-{state['selfplay_games']:06d}.json.gz"
        if not replay_path.exists():
            temporary = replay_path.with_suffix(".tmp")
            with gzip.open(temporary,"wt") as handle:json.dump(state["replay"],handle,allow_nan=False)
            temporary.replace(replay_path)
        saved = {k:state[k] for k in ("identity","step","round","phase","best_loss","stale","status","selfplay_games","search_gate",
                                     "collection_round","round_games","collected_round","arena_pending","champion_step")}
        saved.update(model=state["model"].state_dict(),optimizer=state["optimizer"].state_dict(),
                     scaler=state["scaler"].state_dict(),sampler=state["sampler"].get_state(),
                     rng=self.ns["rng_state"](state["sampler"]),teacher_count=self.teacher.count(),
                     replay_file=replay_path.name,replay_sha256=self.ns["sha256"](replay_path))
        self.ns["save_checkpoint"](state["directory"] / "main.pt",saved)
        self.ns["atomic_json"](state["directory"] / "progress.json",{k:saved[k] for k in ("step","round","phase","best_loss","stale","status","selfplay_games","search_gate","teacher_count")})

    def loss_batch(self, state, records, cache=None):
        boards = [replay_board(r) for r in records]
        features = None if cache else torch.tensor(np.stack([self.ns["encode_board"](b) for b in boards]),device=self.device)
        width = max(len(record_moves(r)) for r in records)
        indices = np.zeros((len(records),width),np.int64)
        probs = np.zeros((len(records),width),np.float32)
        for row,(record,board) in enumerate(zip(records,boards)):
            choices = [self.ns["move_index"](chess.Move.from_uci(m),board.turn) for m in record_moves(record)]
            indices[row] = choices[0]
            indices[row,:len(choices)] = choices
            probs[row,:len(choices)] = record["probs"]
        target = torch.tensor([r["value"] for r in records],device=self.device,dtype=torch.float32)
        state["model"].train()
        with torch.autocast(device_type=self.device.type,enabled=self.device.type == "cuda",dtype=torch.float16):
            if cache:
                logits,value = self.ns["cached_forward"](state["model"],cache,[r["index"] for r in records])
            else:
                logits,value = state["model"](features)
            policy = soft_policy_loss(self.ns["mask_logits"](logits,boards),indices,probs)
            value_loss = F.binary_cross_entropy_with_logits(value.float(),target)
            loss = policy + self.config["value_weight"]*value_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss; previous checkpoint retained")
        state["optimizer"].zero_grad(set_to_none=True)
        state["scaler"].scale(loss).backward()
        state["scaler"].unscale_(state["optimizer"])
        gradients = [p.grad for p in state["model"].parameters() if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in gradients):
            if state["scaler"].is_enabled() and state["scaler"].get_scale() > 1:
                state["scaler"].step(state["optimizer"])  # scaler skips the overflowed update
                state["scaler"].update()
                state["optimizer"].zero_grad(set_to_none=True)
                print("AMP overflow skipped; scale",state["scaler"].get_scale(),flush=True)
                return None
            raise ValueError("Nonfinite gradients at minimum loss scale; previous checkpoint retained")
        torch.nn.utils.clip_grad_norm_(state["model"].parameters(),5,error_if_nonfinite=True)
        state["scaler"].step(state["optimizer"])
        state["scaler"].update()
        return float(loss.detach())

    def relabel(self, indices, deadline, limit=None):
        missing = [i for i in indices if self.teacher.get(chess.Board(str(self.data["fen"][i])).fen(en_passant="fen")) is None]
        if not missing:
            return
        with self.ns["open_stockfish"]() as engine:
            engine.configure({"Threads":1,"Hash":128,"Skill Level":20,"UCI_LimitStrength":False,"UCI_ShowWDL":True})
            for offset, i in enumerate(missing[:limit or self.config["relabel_per_round"]]):
                self.ns["check_stop"]()
                if time.monotonic() >= deadline:
                    break
                board = chess.Board(str(self.data["fen"][i]))
                self.teacher.put(board.fen(en_passant="fen"),teacher_label(engine,board,self.config["teacher_nodes"]))
                if offset % 32 == 0:
                    print("teacher labels",self.teacher.count(),flush=True)
        self.teacher.snapshot()

    def search_gate(self, state):
        # A small training-only probe. Validation/test positions never enter replay.
        indices = stratified_pool(self.groups,min(self.config["gate_positions"],len(self.ns["train_idx"])),np.random.RandomState(313),"general")
        policy_score, search_score = [], []
        with self.ns["open_stockfish"]() as engine:
            engine.configure({"Threads":1,"Hash":128,"Skill Level":20,"UCI_LimitStrength":False,"UCI_ShowWDL":True})
            for i in indices:
                board = chess.Board(str(self.data["fen"][i]))
                raw = self.ns["choose_move"](state["model"],board,False)
                choices,visits,_ = search_target(self.ns,state["model"],board,self.config["simulations"],state["sampler"])
                improved = self.ns["index_to_move"](choices[int(visits.argmax())],board)
                def evaluate(move):
                    info = engine.analyse(board,chess.engine.Limit(nodes=self.config["teacher_nodes"]),root_moves=[move],game=object())
                    return info["score"].pov(board.turn).score(mate_score=10000)
                policy_score.append(evaluate(raw)); search_score.append(evaluate(improved))
        delta = np.asarray(search_score)-np.asarray(policy_score)
        accepted = float(delta.mean()) > 0 and np.count_nonzero(delta > 0) > np.count_nonzero(delta < 0)
        report = {"n":len(indices),"mean_search_gain_cp":float(delta.mean()),"accepted":bool(accepted),
                  "note":"Small training-only engineering gate, not a strength estimate"}
        self.ns["atomic_json"](state["directory"] / "search-gate.json",report)
        print("v5 search gate",report,flush=True)
        return bool(accepted)

    def collect_selfplay(self, state, deadline):
        path = state["directory"] / "selfplay-in-progress.json"
        if state["collection_round"] != state["round"]:
            state["collection_round"],state["round_games"] = state["round"],0
        while state["round_games"] < self.config["games_per_round"]:
            saved = json.loads(path.read_text()) if path.exists() else None
            if saved:
                if saved["game_number"] <= state["selfplay_games"]:
                    path.unlink();continue
                if saved["generator_step"] != state["step"]:
                    raise ValueError("In-progress game belongs to different generator weights")
                board = replay_board(saved); samples = saved["samples"]
                self.ns["restore_rng"](pickle.loads(bytes.fromhex(saved["rng"])),state["sampler"])
                opponent_info = saved.get("opponent")
            else:
                board = chess.Board()
                opening = self.ns["make_opening_pairs"](max(1,self.config["games_per_round"]),seed=41+state["selfplay_games"])[0]
                for move in opening: board.push_uci(move)
                samples = []
                opponent_info = None
                champion = state["directory"] / "main.champion.pt"
                if state["champion_step"] > 0 and state["sampler"].rand() < .25:
                    opponent_info = {"checkpoint":champion.name,"sha256":self.ns["sha256"](champion),
                                     "agent_color":bool(state["sampler"].randint(2))}
            opponent = None
            if opponent_info:
                checkpoint_path = state["directory"] / opponent_info["checkpoint"]
                if self.ns["sha256"](checkpoint_path) != opponent_info["sha256"]:
                    raise ValueError("Self-play opponent changed during an unfinished game")
                opponent = clone_checkpoint_model(self.ns,state,checkpoint_path)
            while self.ns["terminal_value"](board) is None and len(board.move_stack) < self.config["game_cap"]:
                self.ns["check_stop"]()
                if time.monotonic() >= deadline:
                    return False
                actor_turn = opponent is None or board.turn == opponent_info["agent_color"]
                generator = state["model"] if actor_turn else opponent
                choices,probs,_ = search_target(self.ns,generator,board,self.config["simulations"],state["sampler"],True)
                sample = {"start_fen":chess.STARTING_FEN,"history":" ".join(m.uci() for m in board.move_stack),
                          "turn":board.turn,"moves":" ".join(self.ns["index_to_move"](i,board).uci() for i in choices),"probs":probs.tolist(),
                          "generator_step":state["step"],"simulations":self.config["simulations"]}
                if actor_turn:samples.append(sample)
                choice = state["sampler"].choice(len(choices),p=probs.astype(np.float64)/probs.astype(np.float64).sum()) if len(board.move_stack) < 30 else int(probs.argmax())
                board.push(self.ns["index_to_move"](choices[choice],board))
                self.ns["atomic_json"](path,{"start_fen":chess.STARTING_FEN,"history":[m.uci() for m in board.move_stack],
                    "samples":samples,"generator_step":state["step"],"opponent":opponent_info,
                    "game_number":state["selfplay_games"]+1,"rng":pickle.dumps(self.ns["rng_state"](state["sampler"])).hex()})
            outcome = board.outcome(claim_draw=True)
            records = outcome_targets(samples,outcome.winner) if outcome is not None else []
            # Split by canonical board family even for history-rich self-play.
            records = [r for r in records if self.ns["partition"](replay_board(r)) == "train"]
            if records:
                state["replay"] = (state["replay"] + records)[-self.config["replay_size"]:]
            state["selfplay_games"] += 1
            state["round_games"] += 1
            game_dir = state["directory"] / "selfplay-games"
            game_dir.mkdir(exist_ok=True)
            self.ns["atomic_json"](game_dir / f"{state['selfplay_games']:06d}.json",{
                "status":"complete" if outcome is not None else "unresolved", "result":outcome.result() if outcome else None,
                "reason":outcome.termination.name if outcome else "safety_cap", "moves":[m.uci() for m in board.move_stack],
                "generator_step":state["step"],"train_records":len(records)})
            self.save(state)
            if path.exists():path.unlink()
            print("selfplay",state["selfplay_games"],"replay",len(state["replay"]),"result",outcome.result() if outcome else "unresolved",flush=True)
        state["collected_round"] = state["round"]
        return True

    def validate(self, state):
        policy_sum,value_sum,squared,correct = 0.,0.,0.,0
        target_values = []
        with self.ns["inference_mode"](state["model"]):
            for offset in range(0,len(self.validation),self.ns["BATCH_SIZE"]):
                records = [self.source_record(i) for i in self.validation[offset:offset+self.ns["BATCH_SIZE"]]]
                boards = [replay_board(r) for r in records]
                features = torch.tensor(np.stack([self.ns["encode_board"](b) for b in boards]),device=self.device)
                logits,value = state["model"](features)
                logits = self.ns["mask_logits"](logits,boards)
                width = max(len(r["moves"]) for r in records)
                choices = np.zeros((len(records),width),np.int64)
                probabilities = np.zeros((len(records),width),np.float32)
                for row,(record,board) in enumerate(zip(records,boards)):
                    ids = [self.ns["move_index"](chess.Move.from_uci(m),board.turn) for m in record["moves"]]
                    choices[row] = ids[0];choices[row,:len(ids)] = ids
                    probabilities[row,:len(ids)] = record["probs"]
                targets = torch.tensor([r["value"] for r in records],device=self.device,dtype=torch.float32)
                policy_sum += float(soft_policy_loss(logits,choices,probabilities))*len(records)
                value_sum += float(F.binary_cross_entropy_with_logits(value,targets,reduction="sum"))
                squared += float(((value.sigmoid()-targets)**2).sum())
                correct += int((logits.argmax(1)==torch.tensor(choices[:,0],device=self.device)).sum())
                target_values.extend(targets.cpu().tolist())
        baseline = float(np.var(target_values))
        metrics = {"legal_move_match":correct/len(self.validation),"value_mse":squared/len(self.validation),
                   "validation_loss":(policy_sum+self.config["value_weight"]*value_sum)/len(self.validation),
                   "value_skill_vs_validation_mean":1-squared/len(self.validation)/baseline if baseline>1e-10 else None,
                   "value_source":"fixed_pinned_teacher_wdl","n":len(self.validation)}
        loss = metrics["validation_loss"]
        if state["best_loss"] is None or loss < state["best_loss"]-self.config["min_delta"]:
            state["best_loss"],state["stale"] = loss,0
            self.ns["save_checkpoint"](state["directory"] / "main.best.pt",{
                "identity":state["identity"],"model":state["model"].state_dict(),"step":state["step"],"metrics":metrics})
        else:
            state["stale"] += 1
        self.ns["atomic_json"](state["directory"] / "validation-latest.json",dict(metrics,step=state["step"],phase=state["phase"]))
        print(state["directory"].name,"round",state["round"],"step",state["step"],"phase",state["phase"],metrics,flush=True)
        return metrics

    def arena(self, branch, state, deadline):
        directory = state["directory"] / "arenas" / f"step-{state['step']:08d}"
        directory.mkdir(parents=True,exist_ok=True)
        records_path = directory / "records.json"
        records = json.loads(records_path.read_text()) if records_path.exists() else []
        opponent = clone_checkpoint_model(self.ns,state,state["directory"] / "main.champion.pt")
        def mover(model):
            if branch != "v5":return self.ns["model_opponent"](model,False)
            def play(board,clock,increment):
                choices,probs,_ = search_target(self.ns,model,board,self.config["simulations"],state["sampler"])
                return self.ns["index_to_move"](choices[int(probs.argmax())],board)
            return play
        current,old = mover(state["model"]),mover(opponent)
        previous_dir = self.ns["RUN_DIR"]
        self.ns["RUN_DIR"] = directory
        try:
            for pair,opening in enumerate(self.ns["make_opening_pairs"](self.config["arena_pairs"],seed=79)):
                for color in ("white","black"):
                    if time.monotonic()>=deadline:return False
                    game_id = f"{pair}-{color}"
                    if any(r["id"]==game_id for r in records):continue
                    white,black = (current,old) if color=="white" else (old,current)
                    game = self.ns["play_game"](white,black,opening,game_id,initial_clock=120,increment=1,max_plies=500)
                    score = game["score"]
                    if score is not None and color=="black":score=1-score
                    records.append({"id":game_id,"pair":pair,"color":color,"score":score,"reason":game["reason"]})
                    self.ns["atomic_json"](records_path,records)
            summary = self.ns["strength_summary"](records)
            summary.update(candidate_step=state["step"],opponent_step=state["champion_step"],
                           simulations=self.config["simulations"] if branch=="v5" else 0,
                           promotion_rule="At least 62.5% paired score; engineering gate, not statistical proof")
            promoted = summary["complete_pairs"]==self.config["arena_pairs"] and summary["score"]>=.625
            summary["promoted"] = bool(promoted)
            self.ns["atomic_json"](directory / "summary.json",summary)
            if promoted:
                self.ns["save_checkpoint"](state["directory"] / "main.champion.pt",{
                    "identity":state["identity"],"model":state["model"].state_dict(),"step":state["step"]})
                state["champion_step"],state["stale"] = state["step"],0
            state["arena_pending"] = False
            print(branch,"arena",summary,flush=True)
            return True
        finally:
            self.ns["RUN_DIR"] = previous_dir

    def run(self, minutes):
        deadline = time.monotonic()+minutes*60
        ns = self.ns
        try:
            # Freeze the whole validation label set before the first update.
            self.relabel(self.validation,deadline,limit=len(self.validation))
            if any(self.teacher.get(chess.Board(str(self.data["fen"][i])).fen(en_passant="fen")) is None for i in self.validation):
                print("Validation labelling paused; rerun to finish before training",flush=True)
                return
            while time.monotonic()<deadline and any(s["status"]=="active" for s in self.states.values()):
                for branch,state in self.states.items():
                    if state["status"]!="active" or time.monotonic()>=deadline:continue
                    if state["round"] > min(s["round"] for s in self.states.values() if s["status"]=="active"):
                        continue
                    ns["restore_rng"](state["rng"],state["sampler"])
                    ns["RUN_DIR"] = state["directory"]
                    if state["arena_pending"]:
                        if not self.arena(branch,state,deadline):return
                    old_phase = state["phase"]
                    state["phase"] = "foundation" if state["round"]<self.config["foundation_rounds"] else "general" if state["round"]<self.config["pretrain_rounds"] else "consolidation"
                    if state["phase"] != old_phase:state["stale"] = 0
                    # The deterministic shard seed is branch independent. Both branches
                    # see the same supervised curriculum; shards rotate every round.
                    pool = stratified_pool(self.groups,self.config["pool_size"],np.random.RandomState(1000+state["round"]),state["phase"])
                    shard_key = hashlib.sha256(pool.tobytes()).hexdigest()
                    if not self.teacher.stage_done(shard_key):
                        self.relabel(pool,deadline)
                        if time.monotonic()<deadline:self.teacher.mark_stage(shard_key)
                    if time.monotonic()>=deadline:break
                    if branch=="v5" and state["round"]>=self.config["pretrain_rounds"]:
                        if not state["search_gate"] and state["round"]%self.config["gate_interval"]==0:
                            state["search_gate"] = self.search_gate(state)
                        if state["search_gate"] and state["collected_round"] != state["round"]:
                            if not self.collect_selfplay(state,deadline):break
                    # Frozen cache is shared between branches with the same shard.
                    key = hashlib.sha256(pool.tobytes()).hexdigest()
                    if self.cache_key!=key:
                        cache_root = ns["ROOT"] / "perception-cache" / ns["RUN_ID"]
                        tag = "shared-"+self.cache_scope+"-"+key[:12]
                        if cache_root.exists():
                            for old in cache_root.glob("shared-"+self.cache_scope+"-*"):
                                if old.name != tag:shutil.rmtree(old)
                        self.cache = ns["cache_perception"](state["model"],tag,pool,key+self.cache_identity)
                        self.cache_key = key
                    preflight_path = state["directory"] / "learning-preflight.json"
                    if state["step"] == 0:
                        if preflight_path.exists():
                            if not json.loads(preflight_path.read_text())["passed"] and not self.smoke and branch != "v55":
                                raise RuntimeError("Saved learning preflight failed; long training remains stopped")
                        else:
                            ns["learning_preflight"](state["model"],self.cache)
                    last_save = time.monotonic()
                    while state["step"] < (state["round"]+1)*self.round_size:
                        ns["check_stop"]()
                        if time.monotonic()>=deadline:
                            self.save(state);return
                        use_replay = branch=="v5" and bool(state["replay"]) and state["sampler"].rand()<self.config["selfplay_fraction"]
                        if use_replay:
                            selected = state["sampler"].choice(len(state["replay"]),min(ns["BATCH_SIZE"],len(state["replay"])),replace=False)
                            update = self.loss_batch(state,[state["replay"][i] for i in selected])
                        else:
                            selected = state["sampler"].choice(pool,min(ns["BATCH_SIZE"],len(pool)),replace=False)
                            update = self.loss_batch(state,[self.source_record(i) for i in selected],self.cache)
                        if update is None:continue
                        state["step"]+=1
                        if time.monotonic()-last_save>=60:
                            self.save(state);last_save=time.monotonic()
                    state["round"]+=1
                    self.validate(state)
                    if not self.smoke and state["round"]%self.config["arena_every"]==0:
                        state["arena_pending"] = True
                        self.save(state)
                        if not self.arena(branch,state,deadline):return
                    # A phase gets its own patience. Self-play gets a complete patience
                    # window after the search gate succeeds, then returns control.
                    if state["round"]>=self.config["pretrain_rounds"] and state["stale"]>=self.config["patience"]:
                        state["status"]="plateau"
                    if self.smoke and state["round"]>=self.config["smoke_rounds"]:state["status"]="smoke_complete"
                    self.save(state)
                    state["rng"] = ns["rng_state"](state["sampler"])
        except (KeyboardInterrupt, ns["RunPaused"]) as exc:
            print("Paused:",str(exc),flush=True)
        finally:
            # Each branch retains its own RNG stream, including dropout/CUDA RNG.
            for state in self.states.values():
                if ns.get("RUN_DIR")==state["directory"]:
                    state["rng"] = ns["rng_state"](state["sampler"])
            for state in self.states.values():
                ns["restore_rng"](state["rng"],state["sampler"])
                self.save(state)
            ns["RUN_DIR"] = self.base


def advanced_config(branches, smoke):
    def integer(name,default):return int(os.environ.get(name,str(default)))
    config = {"branches":list(branches),"round_size":2 if smoke else integer("FLY_CHESS_ROUND_STEPS",500),
        "pool_size":64 if smoke else integer("FLY_CHESS_TRAIN_POOL",32768),
        "foundation_rounds":1 if smoke else 2,"pretrain_rounds":1 if smoke else 6,
        "validation_size":32 if smoke else 512,"validation_relabel":2 if smoke else 32,
        "teacher_nodes":100 if smoke else integer("FLY_CHESS_TEACHER_NODES",50000),
        "relabel_per_round":4 if smoke else integer("FLY_CHESS_RELABEL_PER_ROUND",256),
        "lr":float(os.environ.get("FLY_CHESS_LR","0.0003")),"value_weight":2.0,
        "patience":integer("FLY_CHESS_PATIENCE",10),"min_delta":.002,
        "simulations":4 if smoke else integer("FLY_CHESS_SIMULATIONS",64),
        "gate_positions":2 if smoke else 32,"gate_interval":1 if smoke else 2,
        "games_per_round":1 if smoke else integer("FLY_CHESS_SELFPLAY_GAMES",8),
        "game_cap":16 if smoke else 500,"replay_size":256 if smoke else 50000,
        "selfplay_fraction":.35,"smoke_rounds":2}
    config.update(arena_every=4,arena_pairs=4)
    if any(config[k]<=0 for k in ("round_size","pool_size","teacher_nodes","patience","simulations","games_per_round")):
        raise ValueError("Training configuration values must be positive")
    return config


def benchmark_advanced(trainer, minutes=15, pairs=20):
    """Paired, immutable, interleaved network/search/helper comparisons."""
    ns = trainer.ns
    deadline = time.monotonic()+minutes*60
    base = ns["RUN_DIR"]
    entries = []
    snapshots = []
    for branch,state in trainer.states.items():
        checkpoint = state["directory"] / "main.best.pt"
        if not checkpoint.exists():checkpoint=state["directory"] / "main.pt"
        snapshot = clone_checkpoint_model(ns,state,checkpoint)
        snapshots.append(snapshot)
        fingerprint = ns["sha256"](checkpoint)
        modes = ["network_search","network_raw"] if branch == "v5" else ["network_raw"]
        if branch == "v55":modes=["fly_ranked","ranker_top1","no_candidates","reverse_candidates","no_senses"]
        for mode in modes:
            def agent(board,clock,increment,mode=mode,snapshot=snapshot,state=state):
                if mode == "ranker_top1":
                    return chess.Move.from_uci(ns["ranked_lookahead"](board.fen(en_passant="fen"))[0][0])
                if mode == "network_search":
                    choices,probabilities,_ = search_target(ns,snapshot,board,trainer.config["simulations"],state["sampler"])
                    return ns["index_to_move"](choices[int(probabilities.argmax())],board)
                if mode in ("no_candidates","reverse_candidates","no_senses"):
                    features = torch.tensor(ns["encode_board"](board)[None],device=ns["DEVICE"])
                    with ns["inference_mode"](snapshot):
                        logits,_ = snapshot(features,intervention=mode)
                    return ns["index_to_move"](int(ns["mask_logits"](logits,[board]).argmax(1)[0]),board)
                return ns["choose_move"](snapshot,board,False)
            entries.append((branch,state,fingerprint,mode,agent))
    with ns["open_stockfish"]() as engine:
        minimum,maximum = engine.options["UCI_Elo"].min,engine.options["UCI_Elo"].max
        ratings = [minimum] if trainer.smoke else sorted(set([minimum,min(maximum,1600),min(maximum,2000)]))
        try:
            openings = ns["make_opening_pairs"](1 if trainer.smoke else pairs,seed=113)
            for pair,opening in enumerate(openings):
                for rating in ratings:
                    for branch,state,fingerprint,mode,agent in entries:
                        directory=state["directory"] / "strength" / fingerprint[:16] / mode / f"sf-{rating}"
                        directory.mkdir(parents=True,exist_ok=True)
                        config={"checkpoint_sha256":fingerprint,"engine_sha256":ns["STOCKFISH_BINARY_SHA"],
                                "reference_elo":rating,"clock":120,"increment":1,"opening_seed":113,
                                "simulations":trainer.config["simulations"] if mode=="network_search" else 0,
                                "assist_mode":"material_minimax_depth2_top4" if branch=="v55" else "none", "mode":mode,
                                "max_plies":12 if trainer.smoke else 600}
                        config_path=directory / "config.json"
                        if config_path.exists() and json.loads(config_path.read_text())!=config:
                            raise ValueError("Benchmark configuration changed")
                        ns["atomic_json"](config_path,config)
                        record_path=directory / "records.json"
                        records=json.loads(record_path.read_text()) if record_path.exists() else []
                        ns["RUN_DIR"]=directory
                        engine.configure({"Threads":1,"Hash":64,"Skill Level":20,"UCI_LimitStrength":True,"UCI_Elo":rating})
                        def reference(board,clock,increment):
                            return engine.play(board,chess.engine.Limit(white_clock=clock if board.turn else 120,
                                black_clock=clock if not board.turn else 120,white_inc=1,black_inc=1),game=game_token).move
                        for color in ("white","black"):
                            game_id=f"{pair}-{color}"
                            if any(r["id"]==game_id for r in records):continue
                            if time.monotonic()>=deadline:return
                            game_token = object()
                            white,black=(agent,reference) if color=="white" else (reference,agent)
                            game=ns["play_game"](white,black,opening,game_id,initial_clock=120,increment=1,max_plies=config["max_plies"])
                            score=game["score"]
                            if score is not None and color=="black":score=1-score
                            records.append({"id":game_id,"pair":pair,"color":color,"score":score,"reason":game["reason"]})
                            ns["atomic_json"](record_path,records)
                            summary=ns["strength_summary"](records,rating,"Stockfish UCI_Elo benchmark at 120+1")
                            summary.update(checkpoint_sha256=fingerprint,configuration=config,target_pairs=len(openings))
                            ns["atomic_json"](directory / "summary.json",summary)
                            print(branch,mode,"benchmark",rating,game_id,score,game["reason"],flush=True)
        finally:
            ns["RUN_DIR"]=base
