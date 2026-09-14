"""Generate standalone V4, V5, paired V4/V5, and V5.5 comparison notebooks."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / 'fly_chess_colab_V3.ipynb').read_text())

def source(tag):
    return ''.join(next(c for c in BASE['cells'] if tag in c['metadata'].get('tags',[]))['source'])

def code(tag, text):
    return {'cell_type':'code','execution_count':None,'metadata':{'tags':[tag]},'outputs':[],
            'source':text.strip().splitlines(keepends=True)}

def markdown(text):
    return {'cell_type':'markdown','metadata':{},'source':text.strip().splitlines(keepends=True)}

def build(name, branches, ranked=False):
    description = ('Frozen fly + sensory lookahead + linear motor readout; no cortex graft.' if ranked else
                   'V4 and V5 share the same fly, graft, supervised labels, curriculum and initialization. '
                   'V4 plays its raw policy. V5 adds fixed-budget PUCT and gated search-guided self-play.')
    intro=f'''# Fly Chess {name}: resumable training and measured comparisons

{description}

Default **smoke** checks a synthetic graph. Set MODE_SETTING to **full** for the real connectome
on a T4. Change RUN_HOURS to allocate a fresh training allowance each invocation. Resume by
re-running; Drive stores datasets, prepared graph, teacher labels, optimizer/RNG state, complete
self-play replay, and checkpoints. Remove a run's STOP file to resume a deliberate pause.

Training rotates balanced shards through foundation, general play and consolidation; pinned
Stockfish MultiPV labels progressively improve the cached dataset and supply WDL targets.
Supervised source CP labels remain identified as proxies until relabelled. V5 self-play starts
only after search improves a training-only engine probe; unresolved capped games never become draws.
Validation and test position families never enter self-play replay. Plateau returns control after
10 non-improving rounds in consolidation; this is an experimental route towards 2000, not a promise.

The evaluation uses paired openings/colors at **120+1**, pinned Stockfish UCI_Elo references
(minimum supported rating, 1600, 2000), immutable checkpoint fingerprints and uncertainty bounds.
This conditional benchmark is not a FIDE/Chess.com/Lichess rating. All-loss results report a bound.
Search simulations and any assistance are saved with every result. A single T4 alternates the
paired notebook's training branches; separate notebooks can run concurrently on separate GPUs.
'''
    if ranked:
        intro+='''
V5.5 receives the top four legal moves, their child boards and material scores from fixed
**two-ply material minimax** through sensory injection. The fly can choose any legal move.
The ranker contains no Stockfish or network. Evaluation includes **ranker top-one alone**,
**no candidate input**, **reversed candidate order** and **zero sensory input** controls.
Only the motor readout trains; every chess feature passes through the frozen fly.
Its larger sensory vector requires a separate architecture and checkpoint from V4/V5.
'''
    config=f'''import os
# Edit the quoted fallback or assign MODE_SETTING = "full" explicitly, then Run all.
MODE_SETTING = os.environ.get("FLY_CHESS_MODE", "smoke")
RUN_HOURS = float(os.environ.get("FLY_CHESS_TRAIN_HOURS", "2"))
EVALUATE = os.environ.get("FLY_CHESS_EVALUATE", "1") == "1"
USE_DRIVE_SETTING = os.environ.get("FLY_CHESS_USE_DRIVE", "1")
os.environ["FLY_CHESS_MODE"] = MODE_SETTING
os.environ["FLY_CHESS_USE_DRIVE"] = USE_DRIVE_SETTING
os.environ.setdefault("FLY_CHESS_TRAIN_POOL", "{'4096' if ranked else '32768'}")
os.environ.setdefault("FLY_CHESS_BATCH", "128")
os.environ.setdefault("FLY_CHESS_TEACHER_NODES", "50000")
os.environ.setdefault("FLY_CHESS_RELABEL_PER_ROUND", "256")
os.environ.setdefault("FLY_CHESS_SIMULATIONS", "64")
os.environ.setdefault("FLY_CHESS_MATCH_PAIRS", "20")
# Optional: FLY_CHESS_SOURCE_DATASET points to your existing pinned V2 .npz + .json.
# Otherwise source MultiPV is streamed and saved once. Set FLY_CHESS_NEW_RUN=1 once
# for a fresh run, then remove it before the next invocation. Defaults resume active run.
'''
    setup=source('setup').replace('fly-chess-v3','fly-chess-v4-v5')
    setup=setup.replace('str(ROOT / "data")', f'str(PERSISTENT_ROOT / "data" / {name.lower()!r})')
    setup=setup.replace('RUNS = PERSISTENT_ROOT / "runs" / "fly-chess-v4-v5"',f'RUNS = PERSISTENT_ROOT / "runs" / "{name.lower()}"')
    setup=setup.split('\nMODEL_SOURCE_SHA256')[0]
    recovery=source('recovery').replace('EXPERIMENT_VERSION = 3',f'EXPERIMENT_VERSION = {5.5 if ranked else 5 if branches==["v5"] else 4}')
    recovery=recovery.replace('def check_stop():\n    if (RUN_DIR / "STOP").exists():','STOP_ROOT = RUN_DIR\n\ndef check_stop():\n    if (STOP_ROOT / "STOP").exists() or (RUN_DIR / "STOP").exists():')
    encoding=source('encoding').replace('N_FEATURES = 780','N_FEATURES = 788')
    encoding=encoding.replace('def encode_board(board):','def encode_base_board(board):').replace('x = np.zeros(N_FEATURES, dtype=np.float32)','x = np.zeros(780, dtype=np.float32)')
    encoding=encoding.replace('def move_index(move, us):','''def encode_history_board(board):
    x = np.zeros(788, np.float32)
    x[:780] = encode_base_board(board)
    known = bool(board.move_stack)
    last = board.peek() if known else None
    x[780:] = [min(board.halfmove_clock / 150, 1), board.is_repetition(2),
               board.is_repetition(3), known, min(board.fullmove_number / 200, 1),
               perspective_square(last.from_square,board.turn)/63 if known else 0,
               perspective_square(last.to_square,board.turn)/63 if known else 0,
               bool(last.promotion) if known else 0]
    return x

encode_board = encode_history_board


def move_index(move, us):''')
    encoding=encoding.replace('hashlib.sha256(encode_board(board)', 'hashlib.sha256(encode_base_board(board)')
    rankeddefs=(ROOT/'flychess/ranked.py').read_text()
    if ranked:
        rankeddefs+='''
N_FEATURES = RANKED_FEATURES

def encode_board(board):
    return encode_ranked_board(board, encode_history_board, encode_base_board)
'''
    advanced=(ROOT/'flychess/advanced.py').read_text()
    data='prepare_advanced_data(globals())'
    graph=source('graph').replace('for name, info in MALECNS_SOURCES.items():\n        path',
        'for name, info in MALECNS_SOURCES.items():\n        if name != "annotations.feather" and GRAPH_PATH.exists() and NODES_PATH.exists():\n            continue\n        path')
    graph=graph.replace('if not NODES_PATH.exists():','if not NODES_PATH.exists() or not GRAPH_PATH.exists():')
    brain=source('brain')
    # Frozen sparse dynamics retain FP32 under AMP; dense trainable blocks use FP16.
    brain=brain.replace('    def run_active_prepared(', '    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)\n    def run_active_prepared(')
    calibration='''GAIN, ALPHA = 1.5, .6
T_P = 15 if USE_REAL_GRAPH else 5
brain = FlyBrain(Wt.to(DEVICE), GAIN, ALPHA)
print("frozen perception", T_P, "gain", GAIN, "alpha", ALPHA)
'''
    relay='''RELAY_IDX = SENSORY_IDX.copy()
RELAY_STEPS = min(3, T_P)
print("relay",len(RELAY_IDX),"early steps",RELAY_STEPS)
'''
    models=source('models').replace('n_relay == 780','n_relay >= 780').replace('self.side_embed = nn.Linear(12, d_model)','self.side_embed = nn.Linear(n_relay - 768, d_model)')
    models=models.replace('    def pool_relay(self, rates):', '    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)\n    def pool_relay(self, rates):')
    models=models.replace('    def perceive(self, features):' ,'    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)\n    def perceive(self, features):')
    models=models.replace('= premotor.t()','= premotor.float().t()')
    flyonly=source('control_flyonly').replace('    def perceive(self, features):','    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)\n    def perceive(self, features):')
    if ranked:
        flyonly=flyonly.replace('return self.forward_perceived(*self.perceive(features))','''intervention = kwargs.get("intervention")
        if intervention in ("no_senses","no_candidates","reverse_candidates"):
            features = features.clone()
            if intervention == "no_senses": features.zero_()
            elif intervention == "no_candidates": features[:,788:] = 0
            else:
                candidates = features[:,788:].reshape(len(features),4,910)
                features[:,788:] = candidates.flip(1).reshape(len(features),-1)
        return self.forward_perceived(*self.perceive(features))''')
    trainingdefs=source('training_definitions').split('\nSTEPS_TARGET')[0]
    if ranked:
        # A readout-only branch is intentionally limited; retain diagnostics without
        # demanding graft-level fixed-batch memorization before the experiment.
        trainingdefs=trainingdefs.replace('if MODE == "full" and not passed:', 'if MODE == "full" and not passed and not RANKED_EXPERIMENT:')
    engine_setup=source('engine_setup').replace('engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_BIN))', 'import shutil\n# Drive stores the pin; execute from the VM, where executable permissions work.\nruntime_engine = ROOT / "engine-runtime" / STOCKFISH_BINARY_SHA / "stockfish"\nruntime_engine.parent.mkdir(parents=True,exist_ok=True)\nif not runtime_engine.exists() or sha256(runtime_engine) != STOCKFISH_BINARY_SHA:\n    shutil.copyfile(STOCKFISH_BIN,runtime_engine)\nruntime_engine.chmod(0o755)\nSTOCKFISH_BIN = runtime_engine\nassert sha256(STOCKFISH_BIN) == STOCKFISH_BINARY_SHA\nengine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_BIN))')
    controls=f'''BRANCHES = {branches!r}
RANKED_EXPERIMENT = {ranked!r}
GRAFT_KW = {{"d_model":16,"n_latents":8,"n_layers":1,"n_heads":2}} if MODE == "smoke" else {{"d_model":96,"n_latents":64,"n_layers":3,"n_heads":4}}
BATCH_SIZE = 8 if MODE == "smoke" else int(os.environ["FLY_CHESS_BATCH"])

def make_advanced_model():
    if RANKED_EXPERIMENT:
        return FlyOnlyModel(brain,MOTOR_IDX,T_P).to(DEVICE)
    return FlyChessModel(brain,RELAY_IDX,PREMOTOR_IDX,MOTOR_IDX,ACTIVE_MASK,
                         T_P,T_A,relay_steps=RELAY_STEPS,**GRAFT_KW).to(DEVICE)

torch.manual_seed(SEED)
model = make_advanced_model()
assert not any(p.requires_grad for p in brain.parameters())
if RANKED_EXPERIMENT:
    assert not hasattr(model,"graft")
print("branches",BRANCHES,"trainable parameters",sum(p.numel() for p in model.parameters() if p.requires_grad))
CONFIG = advanced_config(BRANCHES,MODE == "smoke")
'''
    ordered=[('config',config),('colab_pip',source('colab_pip')),('setup',setup),('recovery',recovery),
        ('encoding',encoding)]
    if ranked:ordered.append(('ranked_definitions',rankeddefs))
    ordered += [('advanced_definitions',advanced),('streaming',source('streaming')),('data',data),
        ('engine_setup',engine_setup),('graph',graph),('brain',brain),('injection',source('injection')),
        ('calibration',calibration),('relay',relay),('premotor',source('premotor')),
        ('models',models),('control_flyonly',flyonly),('training_definitions',trainingdefs),
        ('search',source('search')),('scoring',source('scoring')),('matches',source('matches')),
        ('controls',controls)]
    definition_tags={'encoding','ranked_definitions','brain','injection','models','control_flyonly',
                     'training_definitions','search','scoring','matches','advanced_definitions'}
    pinned={'cells':[code(tag,text) for tag,text in ordered if tag in definition_tags],
            'metadata':{},'nbformat':4,'nbformat_minor':5}
    source_hash=hashlib.sha256(json.dumps(pinned,sort_keys=True).encode()).hexdigest()
    ordered[-1]=('controls',controls+f'\nMODEL_SOURCE_SHA256 = {source_hash!r}\nNOTEBOOK_SOURCE = json.loads({json.dumps(pinned)!r})\n')
    ordered += [('training','''trainer = AdvancedTrainer(globals(),CONFIG)
with writer_lock():
    trainer.run(RUN_HOURS * 60 if MODE == "full" else 3)
'''),('evaluation','''if EVALUATE:
    with writer_lock():
        benchmark_advanced(trainer,minutes=float(os.environ.get("FLY_CHESS_EVAL_MINUTES","1" if MODE == "smoke" else "15")),
                           pairs=int(os.environ["FLY_CHESS_MATCH_PAIRS"]))
else:
    print("Evaluation skipped. Set EVALUATE=True and re-run this cell for paired matches.")
'''),('archive','''# Optional: set FLY_CHESS_ARCHIVE=1 to package this run after training/evaluation.
if os.environ.get("FLY_CHESS_ARCHIVE") == "1":
    import tarfile
    archive_path = ROOT / f"{RUN_ID}-checkpoints.tar.gz"
    with tarfile.open(archive_path,"w:gz") as archive:
        archive.add(STOP_ROOT,arcname=RUN_ID)
    print("archive",archive_path)
''')]
    cells=[markdown(intro)]
    for tag,text in ordered:
        if tag=='data':cells.append(markdown('Dataset is pinned once. Cached single-move labels are upgraded honestly; strong teacher relabelling runs incrementally and survives disconnects.'))
        if tag=='training':cells.append(markdown('Training saves on pause and every minute. Each re-run gets a fresh allowance. Paired branches stay at the same curriculum round, sharing labels and frozen caches.'))
        if tag=='evaluation':cells.append(markdown('Games capped before a result are unresolved and excluded from rating. More completed opening pairs narrow uncertainty. Resume evaluation of the same immutable snapshot; a new checkpoint gets a separate result folder.'))
        cells.append(code(tag,text))
    result={'cells':cells,'metadata':copy.deepcopy(BASE['metadata']),'nbformat':4,'nbformat_minor':5}
    result['metadata']['fly_chess_version']=5.5 if ranked else 5 if branches==['v5'] else 4
    result['metadata']['fly_chess']={'version':5.5 if ranked else name,'branches':branches,'source_sha256':source_hash}
    path=ROOT / f'fly_chess_colab_{name}.ipynb'
    path.write_text(json.dumps(result,indent=1)+'\n')
    print(path.name)

if __name__=='__main__':
    build('V4',['v4'])
    build('V5',['v5'])
    build('V4_V5',['v4','v5'])
    build('V5_5',['v55'],ranked=True)
