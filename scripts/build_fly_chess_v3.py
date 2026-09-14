"""Build the standalone V3 Colab notebook from the saved V2 source.

The historical notebooks/results stay intact. Definitions remain tagged so tests
can execute the exact code Colab will run, without downloading or training.
"""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
notebook = copy.deepcopy(json.loads((ROOT / "fly_chess_colab_V2.ipynb").read_text()))


def cell(tag):
    return next(c for c in notebook["cells"] if tag in c["metadata"].get("tags", []))


def replace(tag, old, new):
    c = cell(tag)
    source = "".join(c["source"])
    assert old in source, (tag, old)
    c["source"] = source.replace(old, new).splitlines(keepends=True)


def set_source(tag, source):
    cell(tag)["source"] = source.strip().splitlines(keepends=True)


notebook["cells"][0]["source"] = [
    "# Fly Chess V3: learn first, measure strength, save every stage\n",
    "\n",
    "This notebook fixes the V2 training bottlenecks and resume budget. The frozen MaleCNS\n",
    "connectome still carries all board information into the graft and all graft output into\n",
    "the motor decoder. Train-only affine signal normalization, a learned relay summary, and richer attention blocks\n",
    "improve that path; no chess engine contributes to the model's decisions. The relay\n",
    "can read early sensory-neuron activity, before recurrent diffusion loses board detail.\n",
    "\n",
    "Frozen perception is cached for a representative training pool. A small memorization\n",
    "check must pass before the longer training stage. Value MSE is compared with a constant\n",
    "training-mean predictor on validation positions. The default trains main + C0; extra\n",
    "architecture controls are optional. Matches run before expensive position analysis.\n",
    "\n",
    "Drive checkpoint storage is on by default. Re-running uses the active run pointer.\n",
    "Every invocation gets a fresh time allowance; completed work remains saved.\n",
    "Stockfish UCI_Elo is an explicitly named benchmark reference, never a claim of a\n",
    "FIDE, Chess.com, or Lichess rating. All-loss samples produce a bound, not a made-up point.\n",
    "For existing V2 checkpoints use scripts/evaluate_fly_chess.py in the repository; V3\n",
    "changes the architecture and uses new checkpoints.\n",
]
notebook["cells"][1]["source"] = '''import os
# Edit these before Run all. Use smoke first to check the pipeline.
os.environ.setdefault("FLY_CHESS_MODE", "smoke")
os.environ.setdefault("FLY_CHESS_STEPS", "4000")
os.environ.setdefault("FLY_CHESS_TRAIN_POOL", "32768")
os.environ.setdefault("FLY_CHESS_TRAIN_MINUTES", "120")
os.environ.setdefault("FLY_CHESS_EVAL_MINUTES", "75")
os.environ.setdefault("FLY_CHESS_CONTROLS", "main,c0")
os.environ.setdefault("FLY_CHESS_MATCH_PAIRS", "20")
os.environ.setdefault("FLY_CHESS_VALUE_WEIGHT", "4")
# Set FLY_CHESS_NEW_RUN=1 once to start another run, then remove it to resume.
'''.splitlines(keepends=True)
replace("setup", "USE_DRIVE = False", 'USE_DRIVE = os.environ.get("FLY_CHESS_USE_DRIVE", "1") == "1"')
replace("setup", 'ROOT = Path("/content/drive/MyDrive/fly-chess-v2")',
        'PERSISTENT_ROOT = Path("/content/drive/MyDrive/fly-chess-v3")\n    ROOT = Path("/content/fly-chess-v3")')
replace("setup", 'ROOT = Path("/content/fly-chess-v2")', 'ROOT = Path(os.environ.get("FLY_CHESS_ROOT", "/content/fly-chess-v3"))\n    PERSISTENT_ROOT = ROOT')
replace("setup", 'RUNS = ROOT / "runs" / "fly-chess-v2"', 'RUNS = PERSISTENT_ROOT / "runs" / "fly-chess-v3"')
replace("setup", 'DATA = ROOT / "data"', 'DATA = Path(os.environ.get("FLY_CHESS_DATA_ROOT", str(ROOT / "data")))')
replace("setup", 'SEED = 0', 'SEED = int(os.environ.get("FLY_CHESS_SEED", "0"))')
replace("setup", 'if DEVICE.type != "cuda":', 'if MODE == "full" and DEVICE.type != "cuda":\n    raise RuntimeError("Full training requires a GPU. Select a GPU runtime before Run all.")\nif DEVICE.type != "cuda":')
replace("recovery", "EXPERIMENT_VERSION = 2", "EXPERIMENT_VERSION = 3")
replace("recovery", 'RUN_ID = os.environ.get("FLY_CHESS_RUN_ID", f"{MODE}-{time.strftime(\'%Y%m%d-%H%M%S\')}-{uuid.uuid4().hex[:8]}")', '''active_pointer = RUNS / f"active-{MODE}.json"
if os.environ.get("FLY_CHESS_RUN_ID"):
    RUN_ID = os.environ["FLY_CHESS_RUN_ID"]
elif active_pointer.exists() and os.environ.get("FLY_CHESS_NEW_RUN") != "1":
    RUN_ID = json.loads(active_pointer.read_text())["run_id"]
else:
    RUN_ID = f"{MODE}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"''')
replace("recovery", 'print("run", RUN_ID)', 'atomic_json(active_pointer, {"run_id": RUN_ID})\nprint("run", RUN_ID, "checkpoints", RUN_DIR)')

# Explicit prepared act state avoids repeating the full, frozen perception phase.
replace("brain", "    def run_active(self, r0, current_active, active_idx, blocks, steps):", '''    def run_active_prepared(self, initial, background, current_active, internal, steps):
        r = initial
        for _ in range(steps):
            target = torch.clamp(self.gain * (torch.sparse.mm(internal, r) + background)
                                 + current_active, 0, 1)
            r = (1 - self.alpha) * r + self.alpha * target
        return r

    def run_active(self, r0, current_active, active_idx, blocks, steps):''')

replace("models", '        self.relay_embed = nn.Embedding(n_relay, d_model)', '''        self.register_buffer("relay_mean", torch.zeros(n_relay))
        self.register_buffer("relay_scale", torch.ones(n_relay))
        self.rate_embed = nn.Linear(1, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.relay_summary = nn.Sequential(nn.Linear(n_relay, 4*d_model), nn.LayerNorm(4*d_model), nn.GELU())
        self.latent_from_summary = nn.Linear(4*d_model, n_latents*d_model)
        self.premotor_from_summary = nn.Linear(4*d_model, n_premotor)
        self.n_latents, self.d_model = n_latents, d_model
        nn.init.normal_(self.premotor_from_summary.weight, std=0.001)
        nn.init.zeros_(self.premotor_from_summary.bias)
        self.relay_embed = nn.Embedding(n_relay, d_model)''')
# Group the known sensory input populations using neuronal rates, not board features.
replace("models", 'class CortexGraft(nn.Module):', '''def make_relay_pool(relay_idx, injection_map):
    mapping = injection_map.coalesce().indices().cpu().numpy()
    lookup = np.full(injection_map.shape[0], -1, dtype=np.int64)
    lookup[np.asarray(relay_idx)] = np.arange(len(relay_idx))
    selected = lookup[mapping[0]] >= 0
    rows, cols = mapping[1, selected], lookup[mapping[0, selected]]
    counts = np.bincount(rows, minlength=injection_map.shape[1])
    if (counts > 0).mean() < .95:
        return None
    values = (1/np.maximum(counts[rows],1)).astype(np.float32)
    return torch.sparse_coo_tensor(torch.tensor(np.stack([rows,cols])), torch.tensor(values),
        (injection_map.shape[1],len(relay_idx))).coalesce().to(injection_map.device)


class CortexGraft(nn.Module):''')
replace("models", 'd_model=64, n_latents=48, n_layers=2, n_heads=4):',
        'd_model=64, n_latents=48, n_layers=2, n_heads=4, relay_pool=None):')
replace("models", '        n_relay, n_premotor = len(relay_idx), len(premotor_idx)', '''        self.register_buffer("relay_pool", torch.empty(0) if relay_pool is None else relay_pool)
        n_relay = len(relay_idx) if relay_pool is None else relay_pool.shape[0]
        n_premotor = len(premotor_idx)''')
replace("models", '    def forward(self, relay_rate):', '''    def pool_relay(self, rates):
        return torch.sparse.mm(self.relay_pool, rates.t()).t() if self.relay_pool.numel() else rates

    def forward(self, relay_rate):''')
replace("models", '        B = relay_rate.shape[0]', '        relay_rate = self.pool_relay(relay_rate)\n        B = relay_rate.shape[0]')
replace("models", '        self.rate_embed = nn.Linear(1, d_model)', '''        self.square_tokens = relay_pool is not None and n_relay == 780
        if self.square_tokens:
            self.piece_embed = nn.Linear(12, d_model)
            self.side_embed = nn.Linear(12, d_model)
        else:
            self.rate_embed = nn.Linear(1, d_model)''')
replace("models", 'self.relay_summary = nn.Sequential(nn.Linear(n_relay, 4*d_model), nn.LayerNorm(4*d_model), nn.GELU())',
        'self.relay_summary = nn.Sequential(nn.Linear(n_relay, 4*d_model), nn.LayerNorm(4*d_model), nn.GELU(), nn.Dropout(0.2))')
replace("models", 'self.relay_embed = nn.Embedding(n_relay, d_model)',
        'self.relay_embed = nn.Embedding(64 if self.square_tokens else n_relay, d_model)')
replace("models", 'batch_first=True)', 'batch_first=True, dropout=0.1)')
replace("models", '        self.policy = nn.Linear(n_motor, n_moves)',
        '        self.dropout = nn.Dropout(0.1)\n        self.policy = nn.Linear(n_motor, n_moves)')
replace("models", '        self.graft = CortexGraft(relay_idx, premotor_idx, **graft_kw)', '''        pooling = make_relay_pool(relay_idx, INJECT_MAP) if "INJECT_MAP" in globals() else None
        self.graft = CortexGraft(relay_idx, premotor_idx, relay_pool=pooling, **graft_kw)''')
replace("models", '        self.current_head = nn.Linear(d_model, 1)', '''        self.output_norm = nn.LayerNorm(d_model)
        self.ff_layers = nn.ModuleList(nn.Sequential(nn.Linear(d_model, 2*d_model), nn.GELU(),
            nn.Linear(2*d_model, d_model)) for _ in range(n_layers))
        self.ff_norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.current_head = nn.Linear(d_model, 1)
        nn.init.normal_(self.current_head.weight, std=0.01)
        nn.init.constant_(self.current_head.bias, 0.1)''')
replace("models", '        tokens = self.relay_embed.weight.unsqueeze(0) * relay_rate.unsqueeze(-1)  # (B, n_relay, d)', '''        rates = ((relay_rate - self.relay_mean) / self.relay_scale).clamp(-10, 10)
        tokens = self.token_norm(self.relay_embed.weight.unsqueeze(0) + self.rate_embed(rates.unsqueeze(-1)))''')
replace("models", '        latents = self.latents.unsqueeze(0).expand(B, -1, -1)', '''        summary = self.relay_summary(rates)
        latents = self.latents.unsqueeze(0) + self.latent_from_summary(summary).view(B, self.n_latents, self.d_model)''')
replace("models", 'self.encode_in(latents, tokens, tokens)[0]', 'self.encode_in(latents, tokens, tokens, need_weights=False)[0]')
replace("models", '''        for layer, norm in zip(self.self_layers, self.norms[1:]):
            latents = norm(latents + layer(latents, latents, latents)[0])''', '''        for layer, norm, ff, ff_norm in zip(self.self_layers, self.norms[1:], self.ff_layers, self.ff_norms):
            latents = norm(latents + layer(latents, latents, latents, need_weights=False)[0])
            latents = ff_norm(latents + ff(latents))''')
replace("models", '''        out, _ = self.decode_out(queries, latents, latents)
        return self.current_head(out).squeeze(-1)  # (B, n_premotor)''', '''        out, _ = self.decode_out(queries, latents, latents, need_weights=False)
        out = self.output_norm(queries + out)
        return torch.tanh(self.current_head(out).squeeze(-1) + self.premotor_from_summary(summary))''')
replace("models", '        self.policy = nn.Linear(n_motor, n_moves)', '''        self.register_buffer("motor_mean", torch.zeros(n_motor))
        self.register_buffer("motor_scale", torch.ones(n_motor))
        self.policy = nn.Linear(n_motor, n_moves)''')
replace("models", '        return self.policy(motor_rate), self.value(motor_rate).squeeze(-1)', '''        normalized = (motor_rate - self.motor_mean) / self.motor_scale
        return self.policy(normalized), self.value(normalized).squeeze(-1)''')
replace("models", '        tokens = self.token_norm(self.relay_embed.weight.unsqueeze(0) + self.rate_embed(rates.unsqueeze(-1)))', '''        if self.square_tokens:
            # These channels are pooled fly rates from the fixed sensory input groups.
            squares = rates[:, :768].reshape(B, 12, 64).transpose(1, 2)
            tokens = self.token_norm(self.relay_embed.weight.unsqueeze(0) + self.piece_embed(squares))
        else:
            tokens = self.token_norm(self.relay_embed.weight.unsqueeze(0) + self.rate_embed(rates.unsqueeze(-1)))''')
replace("models", '        latents = self.latents.unsqueeze(0) + self.latent_from_summary(summary).view(B, self.n_latents, self.d_model)', '''        latents = self.latents.unsqueeze(0) + self.latent_from_summary(summary).view(B, self.n_latents, self.d_model)
        if self.square_tokens:
            latents = latents + self.side_embed(rates[:, 768:]).unsqueeze(1)''')
replace("models", 'nn.Linear(2*d_model, d_model)) for _ in range(n_layers)',
        'nn.Linear(2*d_model, d_model), nn.Dropout(0.1)) for _ in range(n_layers)')
replace("models", 'normalized = (motor_rate - self.motor_mean) / self.motor_scale',
        'normalized = self.dropout((motor_rate - self.motor_mean) / self.motor_scale)')
replace("models", 'current_amplitude=2.0', 'current_amplitude=0.5')
replace("models", 'current_amplitude=0.5, **graft_kw):', 'current_amplitude=0.5, relay_steps=3, **graft_kw):')
replace("models", '        self.T_p, self.T_a, self.current_amplitude = T_p, T_a, current_amplitude',
        '        self.T_p, self.T_a, self.current_amplitude = T_p, T_a, current_amplitude\n        self.relay_steps = min(T_p, relay_steps)')
models_source = "".join(cell("models")["source"])
models_source = models_source[:models_source.index('    def forward(self, features, lesion_relay=False')]+'''
    def perceive(self, features):
        with torch.no_grad():
            current = sensory_current(features)
            state = self.brain.run(torch.zeros(self.brain.n, len(features), device=features.device),
                                   current, self.relay_steps)
            relay = state[self.relay_idx].t()
            state = self.brain.run(state, current, self.T_p - self.relay_steps)
            return (relay, state[self.active_idx].t(),
                    torch.sparse.mm(self.blocks[1], state).t())

    def forward_perceived(self, relay, initial, background, lesion_relay=False, intervention=None):
        if intervention == "no_senses":
            relay, initial, background = torch.zeros_like(relay), torch.zeros_like(initial), torch.zeros_like(background)
        if lesion_relay:
            relay = torch.zeros_like(relay)
        if intervention == "relay_permute":
            # Permute neuron identities; this also works on singleton inference batches.
            relay = relay.roll(1, dims=1)
        premotor = self.graft(relay) * self.current_amplitude
        if intervention == "no_graft":
            premotor = torch.zeros_like(premotor)
        current = torch.zeros(len(self.active_idx), len(relay), device=relay.device)
        current[self.premotor_active_idx] = premotor.t()
        final = self.brain.run_active_prepared(initial.t(), background.t(), current,
                                             self.blocks[0], self.T_a)
        return self.decoder(final[self.motor_active_idx].t())

    def forward(self, features, lesion_relay=False, intervention=None):
        return self.forward_perceived(*self.perceive(features), lesion_relay=lesion_relay, intervention=intervention)
'''
set_source("models", models_source)
replace("relay", 'RELAY_CANDIDATES = {', '''RELAY_STEPS = min(T_P, int(os.environ.get("FLY_CHESS_RELAY_STEPS", "3")))
assert RELAY_STEPS > 0
RELAY_CANDIDATES = {
    "early sensory activity": SENSORY_IDX,''')
replace("relay", 'probe_state = perceive_chunked(brain, probe_features, T_P,',
        'probe_state = perceive_chunked(brain, probe_features, RELAY_STEPS,')
replace("relay", '    idx = idx[~np.isin(idx, SENSORY_IDX)]\n', '')
replace("relay", '    model = Ridge(alpha=1.0).fit(Xtr, Ytr)\n    r2 = model.score(Xte, Yte)', '''    mean, scale = Xtr.mean(0), Xtr.std(0).clip(1e-4)
    model = Ridge(alpha=100.0).fit((Xtr-mean)/scale, Ytr)
    predictions = model.predict((Xte-mean)/scale)
    denominator = float(np.square(Yte-Ytr.mean(0)).sum())
    r2 = 1-float(np.square(predictions-Yte).sum())/denominator if denominator > 0 else -float("inf")''')
replace("controls", 'T_P, T_A, **GRAFT_KW)', 'T_P, T_A, relay_steps=RELAY_STEPS, **GRAFT_KW)')
set_source("control_flyonly", '''
class FlyOnlyModel(nn.Module):
    """Frozen perception -> normalized linear motor readout, without a graft."""
    def __init__(self, brain, motor_idx, T_p):
        super().__init__()
        self.brain, self.T_p = brain, T_p
        self.decoder = MoveDecoder(len(motor_idx))
        self.register_buffer("motor_idx", torch.tensor(motor_idx, dtype=torch.long), persistent=False)

    def perceive(self, features):
        with torch.no_grad():
            state = self.brain.run(torch.zeros(self.brain.n, len(features), device=features.device),
                                   sensory_current(features), self.T_p)
            return (state[self.motor_idx].t(),)

    def forward_perceived(self, motor):
        return self.decoder(motor)

    def forward(self, features, **kwargs):
        return self.forward_perceived(*self.perceive(features))
''')
replace("controls", 'BATCH_SIZE = 8 if MODE == "smoke" else 192', 'BATCH_SIZE = 8 if MODE == "smoke" else int(os.environ.get("FLY_CHESS_BATCH", "64"))')
replace("controls", '''shuffled_W, SWAP_COUNT = shuffled_graph(W, base_signs)
shuffled_Wt, _ = build_signed_transpose(shuffled_W, base_signs, EDGE_MIN_SYNAPSES)
shuffled_Wt = shuffled_Wt.to(DEVICE)''', '''TRAIN_TAGS = os.environ.get("FLY_CHESS_CONTROLS", "main,c0").split(",")
assert "main" in TRAIN_TAGS and "c0" in TRAIN_TAGS and set(TRAIN_TAGS) <= {"main", "c0", "c1", "c2"}
SWAP_COUNT = 0
shuffled_Wt = Wt
if "c2" in TRAIN_TAGS:
    shuffled_W, SWAP_COUNT = shuffled_graph(W, base_signs)
    shuffled_Wt, _ = build_signed_transpose(shuffled_W, base_signs, EDGE_MIN_SYNAPSES)
    shuffled_Wt = shuffled_Wt.to(DEVICE)''')
replace("controls", '("main", "c0", "c1", "c2")}', 'TRAIN_TAGS}')
replace("controls", 'assert n_trainable(models["main"]) == n_trainable(models["c1"])', 'if "c1" in models:\n    assert n_trainable(models["main"]) == n_trainable(models["c1"])')
replace("controls", 'current = self.graft(relay) * 2.0', 'current = self.graft(relay) * 0.5')
replace("controls", 'self.graft = CortexGraft(RELAY_IDX, PREMOTOR_IDX, **GRAFT_KW)',
        'self.graft = CortexGraft(RELAY_IDX, PREMOTOR_IDX, relay_pool=make_relay_pool(RELAY_IDX, INJECT_MAP), **GRAFT_KW)')

training = """def make_batch(indices):
    boards = [chess.Board(str(fens[i])) for i in indices]
    features = torch.tensor(np.stack([encode_board(b) for b in boards]), device=DEVICE)
    labels = torch.tensor([move_index(chess.Move.from_uci(str(labels_uci[i])), b.turn)
                           for i, b in zip(indices, boards)], device=DEVICE)
    targets = torch.tensor(values[indices], device=DEVICE)
    return boards, features, labels, targets


@contextlib.contextmanager
def inference_mode(model):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def move_match(model, indices, intervention=None, value_weight=4.0):
    correct, raw_correct, squared_error, policy_total, value_total = 0, 0, 0.0, 0.0, 0.0
    with inference_mode(model):
        for offset in range(0, len(indices), BATCH_SIZE):
            boards, features, labels, targets = make_batch(indices[offset:offset + BATCH_SIZE])
            logits, value = model(features, intervention=intervention) if intervention else model(features)
            raw_correct += int((logits.argmax(1) == labels).sum())
            correct += int((mask_logits(logits, boards).argmax(1) == labels).sum())
            squared_error += float(((value.sigmoid() - targets)**2).sum())
            policy_total += float(F.cross_entropy(mask_logits(logits, boards), labels, reduction="sum"))
            value_total += float(F.binary_cross_entropy_with_logits(value, targets, reduction="sum"))
    constant = float(np.mean(values[train_idx]))
    baseline = float(np.mean((values[indices] - constant)**2))
    mse = squared_error / len(indices)
    return {"legal_move_match": correct / len(indices), "unmasked_diagnostic": raw_correct / len(indices),
            "value_mse": mse, "constant_value_mse": baseline,
            "value_skill": 1 - mse / baseline if baseline > 1e-10 else None,
            "validation_loss": (policy_total + value_weight*value_total)/len(indices), "n": len(indices)}


def cache_perception(candidate, tag, pool, identity):
    # Full-precision CPU memmaps: exactly the same frozen states as live inference.
    directory = ROOT / "perception-cache" / RUN_ID / tag
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
    if progress and progress["identity"] != identity:
        raise ValueError("Perception cache identity changed")
    first = candidate.perceive(make_batch(pool[:1])[1])
    widths = [x.shape[1] for x in first]
    files = [directory / f"component-{i}.npy" for i in range(len(widths))]
    if progress and all(path.exists() for path in files):
        arrays = [np.lib.format.open_memmap(path, mode="r+") for path in files]
        completed = progress["completed"]
    else:
        arrays = [np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(pool), width))
                  for path, width in zip(files, widths)]
        completed = 0
    for start in range(completed, len(pool), BATCH_SIZE):
        check_stop()
        batch = pool[start:start + BATCH_SIZE]
        with torch.no_grad():
            components = candidate.perceive(make_batch(batch)[1])
        for array, component in zip(arrays, components):
            array[start:start + len(batch)] = component.cpu().numpy()
            array.flush()
        atomic_json(progress_path, {"identity": identity, "completed": start + len(batch), "widths": widths})
        if start % (BATCH_SIZE * 20) == 0:
            print("perception cache", tag, start + len(batch), "/", len(pool), flush=True)
    return {"pool": pool, "lookup": {int(i): j for j, i in enumerate(pool)}, "arrays": arrays}


def cached_forward(candidate, cache, indices):
    rows = [cache["lookup"][int(i)] for i in indices]
    components = []
    for i, array in enumerate(cache["arrays"]):
        selected = np.asarray(array[rows])
        if cache.get("select_columns"):
            selected = selected[:, cache["select_columns"][i]]
        components.append(torch.tensor(selected, device=DEVICE))
    return candidate.forward_perceived(*components)


def initialize_signal_stats(candidate, indices):
    with torch.no_grad():
        features = make_batch(indices)[1]
        components = candidate.perceive(features) if hasattr(candidate, "perceive") else None
        if isinstance(candidate, FlyChessModel):
            relay, initial, background = components
            relay = candidate.graft.pool_relay(relay)
            candidate.graft.relay_mean.copy_(relay.mean(0))
            floor = 0.1 if candidate.graft.relay_pool.numel() else 0.0001
            candidate.graft.relay_scale.copy_(relay.std(0, unbiased=False).clamp_min(floor))
            # Motor normalization is a fixed affine transform of the zero-graft act state.
            zero = torch.zeros_like(initial.t())
            motor = candidate.brain.run_active_prepared(initial.t(), background.t(), zero,
                candidate.blocks[0], candidate.T_a)[candidate.motor_active_idx].t()
        elif isinstance(candidate, FlyOnlyModel):
            motor = components[0]
        else:
            relay, motor = candidate.interface.perceive(features)
            relay = candidate.graft.pool_relay(relay)
            candidate.graft.relay_mean.copy_(relay.mean(0))
            floor = 0.1 if candidate.graft.relay_pool.numel() else 0.0001
            candidate.graft.relay_scale.copy_(relay.std(0, unbiased=False).clamp_min(floor))
        candidate.decoder.motor_mean.copy_(motor.mean(0))
        candidate.decoder.motor_scale.copy_(motor.std(0, unbiased=False).clamp_min(0.03))


def training_update(model, optimizer, sampler, cache=None, indices=None, value_weight=4.0):
    source = cache["pool"] if cache else train_idx
    if indices is None:
        indices = sampler.choice(source, min(BATCH_SIZE, len(source)), replace=False)
    boards, features, labels, targets = make_batch(indices)
    model.train()
    logits, value = cached_forward(model, cache, indices) if cache else model(features)
    policy_loss = F.cross_entropy(mask_logits(logits, boards), labels)
    value_loss = F.binary_cross_entropy_with_logits(value, targets)
    loss = policy_loss + value_weight * value_loss
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss; checkpoint retained")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()), "grad_norm": float(grad_norm)}


def learning_preflight(candidate, cache):
    indices = cache["pool"][:min(32, len(cache["pool"]))]
    boards, _, labels, targets = make_batch(indices)
    original = {key: value.detach().clone() for key, value in candidate.state_dict().items()}
    sampler = np.random.RandomState(SEED)
    saved_rng = rng_state(sampler)
    def metrics():
        with inference_mode(candidate):
            logits, value = cached_forward(candidate, cache, indices)
            return {"accuracy": float((mask_logits(logits, boards).argmax(1) == labels).float().mean()),
                    "value_mse": float(((value.sigmoid() - targets)**2).mean())}
    try:
        before = metrics()
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
        for _ in range(160 if MODE == "full" else 40):
            check_stop()
            training_update(candidate, optimizer, sampler, cache, indices)
        after = metrics()
        passed = after["accuracy"] >= max(0.5, before["accuracy"] + 0.15) and after["value_mse"] < before["value_mse"] * 0.8
        result = {"before": before, "after": after, "passed": passed, "n": len(indices)}
        atomic_json(RUN_DIR / "learning-preflight.json", result)
        print("learning preflight", result, flush=True)
        if MODE == "full" and not passed:
            raise RuntimeError("Learning preflight failed; long training skipped. Inspect learning-preflight.json")
        return result
    finally:
        candidate.load_state_dict(original)
        restore_rng(saved_rng, sampler)


STEPS_TARGET = 4 if MODE == "smoke" else int(os.environ.get("FLY_CHESS_STEPS", "4000"))
VALUE_WEIGHT = float(os.environ.get("FLY_CHESS_VALUE_WEIGHT", "4"))
assert math.isfinite(VALUE_WEIGHT) and VALUE_WEIGHT > 0
POOL_SIZE = min(len(train_idx), 64 if MODE == "smoke" else int(os.environ.get("FLY_CHESS_TRAIN_POOL", "32768")))
assert STEPS_TARGET > 0 and POOL_SIZE >= 32 and BATCH_SIZE > 0
TRAIN_POOL = np.random.RandomState(SEED).choice(train_idx, POOL_SIZE, replace=False)
VALIDATION_SAMPLE = np.random.RandomState(SEED+1).choice(validation_idx,min(512,len(validation_idx)),replace=False)
manifest = {"version": 3, "mode": MODE, "seed": SEED, "real_graph": USE_REAL_GRAPH,
    "dataset_sha256": sha256(positions_path), "graph_sha256": sha256(GRAPH_PATH) if USE_REAL_GRAPH else "synthetic-384-seed-0",
    "nodes_sha256": sha256(NODES_PATH) if USE_REAL_GRAPH else "synthetic",
    "sensory_ids": ids[SENSORY_IDX].tolist(), "relay_ids": ids[RELAY_IDX].tolist(),
    "premotor_ids": ids[PREMOTOR_IDX].tolist(), "motor_ids": ids[MOTOR_IDX].tolist(),
    "threshold": EDGE_MIN_SYNAPSES, "gain": GAIN, "alpha": ALPHA, "perception_steps": T_P,
    "motor_steps": T_A, "relay_steps": RELAY_STEPS, "relay_name": RELAY_NAME,
    "graft": GRAFT_KW, "current_amplitude": 0.5,
    "injection_seed": 0, "shuffle_seed": 12, "swaps": SWAP_COUNT, "train_tags": TRAIN_TAGS,
    "batch_size": BATCH_SIZE, "steps_target": STEPS_TARGET, "pool_size": POOL_SIZE,
    "value_weight": VALUE_WEIGHT,
    "train_pool_sha256": hashlib.sha256(TRAIN_POOL.tobytes()).hexdigest(),
    "architecture": "early-sensory-square-tokens-regularized-v3",
    "source_sha256": MODEL_SOURCE_SHA256,
    "engine_sha256": STOCKFISH_BINARY_SHA, "engine": ENGINE_ID}
manifest_path = RUN_DIR / "manifest.json"
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest, "Configuration changed; start a new run"
else:
    atomic_json(manifest_path, manifest)
IDENTITY = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
event({"stage": "runtime", "torch": torch.__version__, "device": str(DEVICE), "python": sys.version})
perception_caches = {}
stats_sample = TRAIN_POOL[:min(64, len(TRAIN_POOL))]
all_training_complete = all((RUN_DIR / f"{tag}.pt").exists() and
    load_checkpoint(RUN_DIR / f"{tag}.pt", IDENTITY)["step"] >= STEPS_TARGET for tag in models)
closed_status_path = RUN_DIR / "training-status.json"
if closed_status_path.exists():
    closed_status = json.loads(closed_status_path.read_text())
    all_training_complete = all_training_complete or (closed_status.get("status") == "complete" and
        closed_status.get("early_stopped") and all((RUN_DIR / f"{tag}.pt").exists() for tag in models))
for tag, candidate in models.items():
    checkpoint = load_checkpoint(RUN_DIR / f"{tag}.pt", IDENTITY)
    if checkpoint:
        candidate.load_state_dict(checkpoint["model"])
    else:
        initialize_signal_stats(candidate, stats_sample)
    if not all_training_complete and hasattr(candidate, "perceive"):
        if tag == "c0" and "main" in perception_caches:
            main_cache = perception_caches["main"]
            perception_caches[tag] = {"pool": TRAIN_POOL, "lookup": main_cache["lookup"],
                "arrays": [main_cache["arrays"][1]], "select_columns": [models["main"].motor_active_idx]}
        else:
            perception_caches[tag] = cache_perception(candidate, tag, TRAIN_POOL, IDENTITY)
if not (RUN_DIR / "main.pt").exists():
    learning_preflight(models["main"], perception_caches["main"])
print("training steps", STEPS_TARGET, "pool", POOL_SIZE, "cached", list(perception_caches))
"""
set_source("training_definitions", training)

train_source = "".join(cell("training")["source"])
train_source = train_source.replace('        stage_status = "complete"', '''        stage_status = "complete"
        old_status_path = RUN_DIR / "training-status.json"
        old_status = json.loads(old_status_path.read_text()) if old_status_path.exists() else {}
        early_stopped = old_status.get("status") == "complete" and old_status.get("early_stopped", False)
        effective_target = min(s["step"] for s in states.values()) if early_stopped else STEPS_TARGET
        patience = int(os.environ.get("FLY_CHESS_PATIENCE", "8"))
        session_started = time.monotonic()
        session_budget = float(os.environ.get("FLY_CHESS_TRAIN_MINUTES", "120")) * 60 if MODE == "full" else 120''')
train_source = train_source.replace('while min(s["step"] for s in states.values()) < STEPS_TARGET:',
                                   'while min(s["step"] for s in states.values()) < effective_target:')
train_source = train_source.replace('metrics = move_match(state["model"], validation_idx[:min(64, len(validation_idx))])', '''metrics = move_match(state["model"], VALIDATION_SAMPLE, value_weight=VALUE_WEIGHT)
                        best_path = RUN_DIR / f"{tag}.best.pt"
                        best = load_checkpoint(best_path, IDENTITY)
                        if best is None or metrics["validation_loss"] < best["validation_loss"]:
                            save_checkpoint(best_path, {"identity": IDENTITY, "model": state["model"].state_dict(),
                                "step": state["step"], "validation_loss": metrics["validation_loss"]})''')
needle = '        except (RunPaused, KeyboardInterrupt) as exc:'
train_source = train_source.replace(needle, '''                    best = load_checkpoint(RUN_DIR / "main.best.pt", IDENTITY)
                    if patience > 0 and best and states["main"]["step"]-best["step"] >= patience*100:
                        early_stopped = True
                        print("Training stopped after validation loss stopped improving; best weights retained")
                        break
''' + needle)
train_source = train_source.replace('"active_s": sum(s["active_s"] for s in states.values())})',
    '"active_s": sum(s["active_s"] for s in states.values()), "early_stopped": early_stopped})')
train_source = train_source.replace('        return states', '''        selection = {}
        for tag, state in states.items():
            best = load_checkpoint(RUN_DIR / f"{tag}.best.pt", IDENTITY)
            if best:
                state["model"].load_state_dict(best["model"])
                selection[tag] = {"checkpoint": f"{tag}.best.pt", "step": best["step"],
                                  "validation_loss": best["validation_loss"]}
        atomic_json(RUN_DIR / "model-selection.json", selection)
        return states''')
train_source = train_source.replace('sum(s["active_s"] for s in states.values()) >= (120 * 60 if MODE == "full" else 120)', 'time.monotonic() - session_started >= session_budget')
train_source = train_source.replace('training_update(state["model"], state["optimizer"], state["sampler"])', 'training_update(state["model"], state["optimizer"], state["sampler"], perception_caches.get(tag), value_weight=VALUE_WEIGHT)')
train_source = train_source.replace('>= 120:', '>= 60:')
train_source = train_source.replace('model, model_c0, model_c1, model_c2 = (models[tag] for tag in ("main", "c0", "c1", "c2"))', 'model, model_c0 = models["main"], models["c0"]')
set_source("training", train_source)

# Reuse the existing root prediction and transfer ranked logits to CPU once.
replace("search", 'def search(model, root_board, n_simulations=64, batch=16, deadline=None):',
        'def search(model, root_board, n_simulations=64, batch=16, deadline=None, root_priors=None):')
replace("search", '    priors, _ = evaluate_leaves(model, [root.board])\n    expand(root, priors[0])',
        '    if root_priors is None:\n        priors, _ = evaluate_leaves(model, [root.board])\n        root_priors = priors[0]\n    expand(root, root_priors)')
replace("search", 'def best_move_by_search(model, board, n_simulations=None, deadline=None):',
        'def best_move_by_search(model, board, n_simulations=None, deadline=None, root_priors=None):')
replace("search", 'root = search(model, board, n_simulations or (8 if MODE == "smoke" else 64), deadline=deadline)',
        'root = search(model, board, n_simulations or (8 if MODE == "smoke" else 64), deadline=deadline, root_priors=root_priors)')
replace("search", '        move, tree = best_move_by_search(model, board, deadline=deadline)',
        '        probabilities = F.softmax(logits[0, legal_indices], dim=0).cpu().numpy()\n        move, tree = best_move_by_search(model, board, deadline=deadline, root_priors=dict(zip(legal_indices, probabilities)))')
replace("search", 'neural_evaluations=tree.neural_evaluations + 1', 'neural_evaluations=tree.neural_evaluations')
replace("search", 'ranked = sorted(legal_indices, key=lambda i: (-float(logits[0, i]), i))', 'row = logits[0].detach().cpu().numpy()\n    ranked = sorted(legal_indices, key=lambda i: (-float(row[i]), i))')

# Inline the pure strength utility: the Colab notebook needs no repository imports.
strength = (ROOT / "flychess/strength.py").read_text()
matches = "".join(cell("matches")["source"])
# A terminal result reached exactly on the final allowed ply is still resolved.
matches = matches.replace('    state = {"status": "complete", "score": score, "reason": reason,', '''    if score is None:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            score = 0.5 if outcome.winner is None else float(outcome.winner == chess.WHITE)
            reason = outcome.termination.name
    state = {"status": "complete", "score": score, "reason": reason,''')
matches += '\n\n' + strength
matches += '''

def make_opening_pairs(count, seed=41):
    generator = np.random.RandomState(seed)
    openings, seen = [], set()
    while len(openings) < count:
        board = chess.Board()
        moves = OPENINGS[int(generator.randint(len(OPENINGS)))].split()
        for move in moves:
            board.push_uci(move)
        for _ in range(2):
            if terminal_value(board) is not None:
                break
            move = list(board.legal_moves)[int(generator.randint(board.legal_moves.count()))]
            moves.append(move.uci())
            board.push(move)
        key = " ".join(moves)
        if terminal_value(board) is None and key not in seen:
            seen.add(key)
            openings.append(moves)
    return openings


def evaluate_strength():
    path = RUN_DIR / "strength.json"
    pairs = 1 if MODE == "smoke" else int(os.environ.get("FLY_CHESS_MATCH_PAIRS", "20"))
    assert 1 <= pairs <= 1000
    openings = make_opening_pairs(pairs)
    report = json.loads(path.read_text()) if path.exists() else {"identity": IDENTITY, "matches": {}, "summaries": {}}
    assert report["identity"] == IDENTITY
    started = time.monotonic()
    budget = float(os.environ.get("FLY_CHESS_EVAL_MINUTES", "75"))*60 if MODE == "full" else 180
    clock, increment = (120, 1) if MODE == "full" else (2, .01)
    with open_stockfish() as engine:
        minimum, maximum = engine.options["UCI_Elo"].min, engine.options["UCI_Elo"].max
        ratings = sorted(set([minimum, min(maximum, minimum + 200)]))
        opponents = [("c0", model_opponent(model_c0), None), ("greedy", greedy_material, None)]
        for rating in ratings:
            def opponent(board, remaining, inc, rating=rating):
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": rating, "Skill Level": 20})
                return engine.play(board, chess.engine.Limit(white_clock=remaining if board.turn else clock,
                    black_clock=remaining if not board.turn else clock, white_inc=inc, black_inc=inc)).move
            opponents.append((f"stockfish-uci-{rating}", opponent, rating))
        configuration = {"pairs": pairs, "clock": clock, "increment": increment,
                         "ratings": ratings, "opening_seed": 41, "assist_mode": "none"}
        if report.get("configuration") and report["configuration"] != configuration:
            raise ValueError("Strength match configuration changed; use a new run")
        report["configuration"] = configuration
        selection_path = RUN_DIR / "model-selection.json"
        selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
        fingerprints = {}
        for tag in ("main","c0"):
            checkpoint_path = RUN_DIR / selection.get(tag,{}).get("checkpoint",f"{tag}.pt")
            fingerprints[tag] = sha256(checkpoint_path) if checkpoint_path.exists() else None
        if report.get("checkpoint_fingerprints") is not None and report["checkpoint_fingerprints"] != fingerprints:
            raise ValueError("Evaluated weights changed; use a separate run for a new benchmark")
        report["checkpoint_fingerprints"] = fingerprints
        report["model_selection"] = selection
        try:
            # Interleave opponents and modes so a pause leaves evidence for each condition.
            for pair, opening in enumerate(openings):
                for mode_name, use_search in (("policy", False), ("search", True)):
                    for name, opponent, rating in opponents:
                        key = mode_name + ":" + name
                        records = report["matches"].setdefault(key, [])
                        for color in ("white", "black"):
                            check_stop()
                            if time.monotonic() - started > budget:
                                report["status"] = "paused"
                                report["pause_reason"] = "Session evaluation allocation exhausted; rerun to continue"
                                return report
                            game_id = f"strength-{mode_name}-{name}-{pair}-{color}"
                            if any(r["id"] == game_id for r in records):
                                continue
                            player = model_opponent(model, use_search)
                            white, black = (player, opponent) if color == "white" else (opponent, player)
                            game = play_game(white, black, opening, game_id, initial_clock=clock,
                                             increment=increment, max_plies=600 if MODE == "full" else 12)
                            score = game["score"]
                            if score is not None and color == "black":
                                score = 1-score
                            records.append({"id": game_id, "pair": pair, "color": color, "score": score, "reason": game["reason"]})
                            report["summaries"][key] = strength_summary(records, rating,
                                f"Stockfish UCI_Elo at {clock}+{increment}" if rating is not None else None)
                            atomic_json(path, report)
                            print(key, pair, color, score, game["reason"], flush=True)
            report["status"] = "complete"
        except (RunPaused, KeyboardInterrupt) as exc:
            report["status"] = "paused"
            report["pause_reason"] = str(exc)
        finally:
            report["time_control"] = {"initial_seconds": clock, "increment_seconds": increment}
            report["engine_sha256"] = STOCKFISH_BINARY_SHA
            atomic_json(path, report)
    return report
'''
set_source("matches", matches)

evaluation = "".join(cell("evaluation")["source"])
evaluation = evaluation.replace('"version": 2', '"version": 3')
evaluation = evaluation.replace('prior_active + time.monotonic() - started > (75 * 60 if MODE == "full" else 180)', 'time.monotonic() - started > (float(os.environ.get("FLY_CHESS_EVAL_MINUTES", "75")) * 60 if MODE == "full" else 180)')
# Existing detailed evaluation is optional; strength matches are now their own first stage.
begin = evaluation.index('            with open_stockfish() as engine:\n                opponents =')
end = evaluation.index('            report["status"] = "complete"', begin)
evaluation = evaluation[:begin] + evaluation[end:]
evaluation = evaluation[:evaluation.index('results = evaluate_all()')] + '''strength_results = evaluate_strength()
print("strength report", RUN_DIR / "strength.json", strength_results.get("status"))
for condition, summary in strength_results["summaries"].items():
    print(condition, json.dumps(summary))
if os.environ.get("FLY_CHESS_DETAILED_EVAL", "0") == "1":
    results = evaluate_all()
    print("position/puzzle report", RUN_DIR / "results.json", results["status"])
else:
    print("Set FLY_CHESS_DETAILED_EVAL=1 to run the optional position/puzzle audit")
'''
set_source("evaluation", evaluation)

# Package only experiment artifacts, not gigabytes of already downloadable raw data.
notebook["cells"][37]["source"] = '''import tarfile
archive_path = ROOT / f"{RUN_ID}-checkpoints.tar.gz"
with tarfile.open(archive_path, "w:gz") as archive:
    archive.add(RUN_DIR, arcname=RUN_ID)
print("Checkpoint archive", archive_path)
if USE_DRIVE:
    print("Live checkpoints already saved to Drive:", RUN_DIR)
# In Colab, manually download the archive if desired:
# from google.colab import files
# files.download(str(archive_path))
'''.splitlines(keepends=True)
notebook["cells"][-1]["source"] = [
    "## Reading the results\n", "\n",
    "`learning-preflight.json` checks that the model can learn a fixed training batch,\n",
    "then restores its initial weights. `events.jsonl` records validation move accuracy\n",
    "and value skill against a constant predictor. A passing preflight is not evidence\n",
    "of generalization. `strength.json` records paired matches and score-derived Elo\n",
    "differences. For Stockfish references, the absolute-looking number is conditional\n",
    "on Stockfish's UCI calibration, engine build, hardware and time control.\n",
    "A null interval endpoint means unbounded; an all-loss sample has an upper bound\n",
    "and no finite point estimate. More independent opening pairs narrow sampling\n",
    "uncertainty. Safety-capped games do not count as draws.\n",
    "\n", "Training and evaluation allowances apply to each invocation. Rerun to continue\n",
    "the saved active run. Cache files are disposable and can be rebuilt after a runtime\n",
    "reset; model weights, optimizer and sampler state are persisted to Drive.\n",
]
# Update the historical section prose to describe the actual V3 controls.
for c in notebook["cells"]:
    if c["cell_type"] == "markdown" and "## Controls and matched training" in "".join(c["source"]):
        c["source"] = ["## Controls and matched training\n", "\n",
            "The default trains main and the fly-only C0 for the same number of batches.\n",
            "Add c1 or c2 to FLY_CHESS_CONTROLS to train the synthetic interface and shuffled\n",
            "wiring controls. All readouts remain linear in motor activity after a fixed affine\n",
            "normalization. A small learning preflight runs before long training.\n"]
for c in notebook["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
    c["metadata"].pop("execution", None)
    c["metadata"].pop("colab", None)
core = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and
        c["metadata"].get("tags") and not set(c["metadata"]["tags"]) & {"setup", "recovery", "play"}]
source_hash = hashlib.sha256(json.dumps(core).encode()).hexdigest()
cell("setup")["source"].append(f'\nMODEL_SOURCE_SHA256 = "{source_hash}"\n')
notebook["metadata"]["fly_chess"] = {"version": 3,
    "architecture": "early-sensory-square-tokens-regularized-v3", "source_sha256": source_hash}
notebook["metadata"]["fly_chess_version"] = 3
target = ROOT / "fly_chess_colab_V3.ipynb"
target.write_text(json.dumps(notebook, indent=2) + "\n")
print(target)
