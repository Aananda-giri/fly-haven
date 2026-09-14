# Fly Heaven

One notebook, [`fly-heaven.ipynb`](fly-heaven.ipynb), that reruns two MaleCNS v1.0 connectome experiments on this machine and checks them against their published results:

- **Fruitless** ([`experiments/fruitless`](experiments/fruitless)): does blocking mAL output unmask a P1 courtship-neuron response to candidate male cues?
- **Fly / Wirehead** ([`experiments/fly-wirehead`](experiments/fly-wirehead)): a full connectome watching insect Shorts, with an artificial drive on its PAM11 dopamine neurons.

Both use the same three MaleCNS source files, so the notebook downloads them once (~1.1 GB, checksum-verified) and links them into each project's layout. It calls each project's own simulation code without modifying it.

A second notebook, [`fly-chess.ipynb`](fly-chess.ipynb), grafts a small trainable "cortex" onto the
same frozen connectome and has it play chess, as a comparison against [flychess-hq](https://flychess-hq.vercel.app/)'s
fly-plus-linear-readout design. It reuses the fruitless project's prepared graph, and downloads its
own chess data and a Stockfish binary into `data/fly-chess/`.

The new comparison notebooks share one generated implementation:

| Notebook | Training | Move selection |
| --- | --- | --- |
| [V4](fly_chess_colab_V4.ipynb) | Balanced supervised curriculum and progressive strong-engine relabelling | Raw network, no search |
| [V5](fly_chess_colab_V5.ipynb) | Same supervised start, then gated search-guided self-play | Network + 64-simulation PUCT; raw-policy control included |
| [V5.5](fly_chess_colab_V5_5.ipynb) | Same supervised curriculum; only the motor readout trains | Two-ply material lookahead sends its ranked top four moves and child boards into fly sensory neurons; **no cortex graft** |
| [Paired V4/V5](fly_chess_colab_V4_V5.ipynb) | Alternates branches on one T4; shares teacher labels and frozen perception caches | Both V4 and V5, with separate checkpoints |

Run the default smoke check first, then assign `MODE_SETTING = "full"` in the first cell
and choose a T4 GPU. `RUN_HOURS` gives each invocation a fresh allowance; run again to
continue. Separate V4/V5 notebooks can run concurrently on separate GPU runtimes.
The paired notebook makes progress on both using one T4. V5.5 runs in its own notebook.

Each notebook has its own persistent data/download directory to avoid concurrent writers.
For exactly the same corpus across separate runtimes, point both at the same pinned
`FLY_CHESS_SOURCE_DATASET` (V2 or V4); the paired notebook shares its corpus automatically.

Full mode uses one million deduplicated, legal, depth-filtered Lichess evaluation positions.
Available MultiPV moves receive soft probability targets; existing pinned V2 datasets can
be reused through `FLY_CHESS_SOURCE_DATASET`, with their original single-move labels
explicitly identified. Foundation training emphasizes short mates; later shards balance
tactics, endgames and general positions while retaining earlier skills. Every round rotates
the shard and relabels up to 256 training positions using pinned Stockfish, 50,000 nodes,
MultiPV 3 and WDL expectation targets. Source CP-based values remain proxies until
relabelled. All validation labels are refreshed once and fixed before the first update.
The same 788-feature board/history encoding, 13.8M-parameter graft/readout, initialization
and supervised updates serve V4 and V5. The frozen fly uses FP32; trainable dense blocks
use FP16 with loss scaling on CUDA. V5.5 instead has a 4,428-feature sensory vector and
readout-only architecture; its smaller default shard (4,096 positions) limits CPU lookahead work.

V5 starts self-play after six supervised rounds only when a training-only probe shows search
improving move quality. It learns root visit distributions and completed, mover-relative
game outcomes, mixes 35% replay batches with supervised data, and sometimes plays a historical
champion. Validation/test position families are excluded from replay. Capped games remain
unresolved. Champion matches give both players equal search budgets; their promotion threshold
is an engineering gate. Validation patience stops consolidation after ten non-improving rounds;
a promoted champion resets patience. This training design does **not** establish or guarantee 2000 Elo.

Drive saves datasets, prepared connectome, pinned source code, teacher snapshots, model,
optimizer, scaler/RNG state, self-play histories and compressed replay. Compact checkpoints
omit the frozen matrix and reconstruct it from the pinned graph. Frozen perception
caches stay on the VM and rebuild after a reset. Checkpoints save each round, every minute,
and on a deliberate pause. An abrupt runtime loss can lose updates since the last checkpoint;
self-play saves each completed ply. Create a run-root `STOP` file to pause; remove it to resume.
Change architecture/data/teacher settings only for a new run (`FLY_CHESS_NEW_RUN=1` once,
then remove it). If the learning preflight fails, V4/V5 skip long training; V5.5 records the
same diagnostic but continues its deliberately limited readout experiment.

Evaluation uses identical paired openings/colors and 120+1 clocks against the pinned
Stockfish UCI_Elo minimum, 1600 and 2000. It interleaves conditions, preserves checkpoint
fingerprints and reports score intervals plus conditional Elo on that reference scale.
There is no invented rating after all losses and no human-platform Elo claim. V5 includes
raw-policy evaluation. V5.5 includes ranker-top-one alone, no candidate input, reversed ranking,
and zero-sensory interventions; these diagnose whether the fly improves the external helper.
Only real completed results count; smoke safety caps commonly produce no rating estimate.
More opening pairs improve precision. Set `EVALUATE=False` to train without matches, then
run the evaluation cell when ready. Compare downloaded reports with:

```bash
.venv/bin/python scripts/summarize_fly_chess.py runs/fly-chess-v4 runs/fly-chess-v5 runs/fly-chess-v55
```

Local validation ran all four synthetic GPU smoke pipelines and reconstructed a saved
real-connectome checkpoint. The full 13.8M-parameter V4 architecture completed 355 updates
in a bounded 90-second check, at batch 128 with 1.62 GiB peak allocated VRAM on an RTX 4050.
That check used a small corpus and cheaper teacher labels to verify execution and memory;
it is not an Elo measurement. T4 performance and Colab Drive recovery still need a remote run.

The notebooks are standalone in Colab. Regenerate all four after shared-code edits with
`.venv/bin/python scripts/build_fly_chess_v4_v5.py`; their embedded implementation is pinned
beside every checkpoint. Historical V1/V2/V3 files are preserved.

[`fly_chess_colab_V3.ipynb`](fly_chess_colab_V3.ipynb) addresses the weak saved V2 run.
It captures relay activity early and includes sensory-neuron activity among the candidates, because
the late central relay in the saved run lost much of the material signal. Selection uses board
reconstruction error against a constant baseline. For a sensory relay, the graft averages neuronal
rates within the fixed injection groups; it receives 780 group signals instead of learning the
same input from 8,000 redundant neurons. It normalizes relay and motor signals using
training data, adds a learned relay summary, residual attention and feed-forward
blocks to the graft, bounds injected current, and caches full-precision frozen perception. The
default trains the main model and fly-only C0; the other controls are optional. A 32-position
memorization check must pass before full training, and validation includes value MSE against a
constant predictor. This check establishes learnability, not playing strength.
For sensory relays, the graft uses 64 square tokens with shared piece embeddings, plus the
castling/en-passant channels read from neuronal groups. Dropout limits memorization, and value
loss receives weight 4 by default. Validation chooses `main.best.pt` and `c0.best.pt`; training
stops after eight rounds without improvement in the main model's validation loss. Latest
optimizer checkpoints stay available for recovery. Set `FLY_CHESS_PATIENCE=0` to disable stopping.

Upload V3 to Colab and run its default `smoke` mode first. Edit the first configuration cell to
use `full` and choose a GPU runtime for real training. Google Drive storage is enabled by default;
authorize the mount when Colab asks. Model/optimizer/RNG checkpoints save at least every minute
and each training round. The active run pointer automatically selects the same run after a reset.
Each invocation gets a fresh training/evaluation allowance, so an exhausted stage can resume.
Data and frozen-perception caches stay on the VM and can be rebuilt. Change the configuration
only for a new run (`FLY_CHESS_NEW_RUN=1` once, then remove it).

V3 writes `strength.json` before the optional expensive position/puzzle audit. It reports actual
W/D/L, score-based Elo differences, and conservative intervals over complete opening pairs.
Stockfish's supported `UCI_Elo` settings are explicitly identified as the reference scale.
These are conditional Stockfish-benchmark estimates, not FIDE/Chess.com/Lichess ratings;
an all-loss sample produces an upper bound with no finite point estimate. Unfinished games
remain unresolved. The default uses 20 varied opening pairs per opponent/mode; more pairs
reduce sampling uncertainty and take longer. Set `FLY_CHESS_DETAILED_EVAL=1` for the separate audit.
Stockfish documents how its [UCI strength settings choose weaker moves](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html#how-do-skill-level-and-uci-elo-work).

Saved V2 checkpoints can be evaluated directly, without rerunning training or calibration:

```bash
.venv/bin/python scripts/evaluate_fly_chess.py \
  --run runs/fly-chess-v2/runs/fly-chess-v2/full-20260913-125139-56836f24 \
  --pairs 20 --minutes 30 --search
```

This workspace's matching graph and engine already exist in `data/fruitless/` and
`data/fly-chess/stockfish/`; the command verifies their hashes against the run manifest.
Use `--data`, `--graph`, and `--engine` for other locations. It persists games to a separate
`strength-benchmark/` folder. Repeat the same command to resume. V3 changes the architecture,
so its graft cannot resume a V2 optimizer checkpoint. For a bounded local V3 validation pilot
that reuses V2's graph, dataset and selected neurons:

```bash
.venv/bin/python scripts/train_fly_chess_pilot.py \
  --source-run runs/fly-chess-v2/runs/fly-chess-v2/full-20260913-125139-56836f24 \
  --out runs/fly-chess-v3/regularized-pilot --steps 900 --pool 32768 --batch 128 --minutes 15
```

The pilot writes a new checkpoint and validation report; it does not assign Elo. Its default uses
the early sensory relay. Pass `--relay-kind central` to compare with early central activity.
Evaluate its output by passing that new run directory to `evaluate_fly_chess.py`.
For the full notebook's validation-selected checkpoint, also pass `--checkpoint main.best.pt`.
`scripts/build_fly_chess_v3.py` rebuilds the standalone notebook; its tagged definitions
are exercised by `tests/test_fly_chess_v3.py`.

The completed local pilot used the real graph, 32,768 training positions and 900 updates.
On the same 200 saved test positions used for V2:

| Metric | Saved V2 | V3 pilot |
| --- | ---: | ---: |
| Teacher move match | 16.5% | 19.5% |
| Value MSE | 0.06042 | 0.03478 |
| Mean centipawn loss (177 scored positions) | 325.84 | 284.54 |

Value error fell 42.4%. This is one seed and a small position-level sample; these metrics
do not establish Elo or a statistically reliable playing-strength improvement.
Small 120+1 game checks recorded policy W/D/L of 1/3/0 against random, 0/4/0 against
greedy, and 0/0/2 against Stockfish UCI_Elo 1320. With 64-simulation search, the pilot
scored 1/3/0 against greedy. These samples are too small to assign an Elo rating;
the pilot still needs stronger training and a larger benchmark.
The checkpoint and `heldout-comparison.json` are saved under
`runs/fly-chess-v3/regularized-pilot/`. Reproduce the comparison with:

```bash
.venv/bin/python scripts/compare_fly_chess.py \
  --run runs/fly-chess-v3/regularized-pilot \
  --baseline runs/fly-chess-v2/runs/fly-chess-v2/full-20260913-125139-56836f24
```

[`fly-chess-v2.ipynb`](fly-chess-v2.ipynb) is a correctness pass over that notebook, prompted by
review findings against the original (a `stockfish_cp_loss` perspective bug, a PUCT selection/backup
sign error, an unsupported absolute-Elo estimate, a lesion control that couldn't prove its own claim,
and several architecture/spec drifts — see `runs/fly-chess-v2/*/manifest.json` for the corrected
design's exact identity). It keeps the original `fly-chess.ipynb` and its already-running full-mode
job untouched, and adds: a 4,184-class move vocabulary with distinct underpromotions, side-to-move
value labels, a linear (not MLP) motor decoder, mover-perspective Stockfish scoring, pending-leaf-safe
PUCT with `1 − child.q()` opponent-perspective backup, matched-step controls (C0/C1/C2) plus
lesion/no-graft/no-senses/relay-permute interventions, W/D/L + paired-opening match results with
unresolved games (no invented Elo), atomic checkpoints with full RNG state for exact resume, and an
explicit `baseline_unavailable` result instead of assuming `EF-Code/flychess` or `Noeljarillo/chessfly`
is the deployed site. `tests/test_fly_chess_v2.py` exercises the notebook's own tagged cell
definitions (move round-trips, label perspective, search backup/mate-in-one, checkpoint resume,
shuffle invariants, active-subgraph gradient equivalence, …) without downloading data or training. Run
it with `FLY_CHESS_DEVICE=cpu FLY_CHESS_MODE=smoke` for a synthetic-graph pipeline check in minutes,
add `FLY_CHESS_REAL_GRAPH=1` to exercise the real MaleCNS graph (166,606 neurons, 25.57M edges)
without a GPU, or `FLY_CHESS_MODE=full` for the real four-hour pilot once the GPU is free.

[`fly-chess-v2-colab.ipynb`](fly-chess-v2-colab.ipynb) is a standalone Colab port of the corrected
notebook above — upload it to Colab (or open from Drive/GitHub), pick a GPU runtime, and
`Runtime -> Run all`. It downloads everything itself (no local repo needed): a `pip install` cell for
`chess`/`zstandard`/`scikit-learn`/`pyarrow`, and, in `MODE="full"`, the real MaleCNS graph built
directly from the public HHMI Janelia / Google Research source (CC BY 4.0, checksum-verified,
~1.1 GB) instead of reusing the local `experiments/fruitless` build. Verified locally end-to-end in
smoke mode (synthetic connectome, CPU); the real-graph download path is adapted from this repo's
existing (uncorrected) `fly-chess-colab.ipynb` and is worth watching on its first real run. This is
not the same file as `fly-chess-colab.ipynb`, which ports the original, still-buggy notebook.

## Fly Heaven: a connectome-driven film

[`heaven/`](heaven/) presents the full 166,700-neuron MaleCNS v1.0 brain's
behavior in a continuous macro forest-floor film using the rigged BlenderKit
housefly. Feeding, grooming, escaping, courtship and mating are gated by real
annotated cell-type readouts; the environment, navigation, female agent and
visible poses are presentation or scripted world mechanics. See the continuous
film instructions below for the current asset and rendering pipeline.

The simulation can still be reproduced with `uv run python -m heaven.calibrate`
and `uv run python -m heaven.simulate --seconds 220 --out runs/fly-heaven/day1`.
Export a timeline for Blender using `uv run python -m heaven.blender.export_timeline`
followed by its `.parquet` path. Lesion comparisons (`--lesion MN9`, `P1`, `GF`,
or `DNg12`) remain available to verify the brain-to-behavior routes.

## Run the notebooks

Requires [uv](https://docs.astral.sh/uv/), a C++17 compiler, and FFmpeg (plus yt-dlp if the fly-wirehead videos aren't downloaded yet). `fly-chess.ipynb` additionally wants a CUDA GPU (it will run on CPU, just far slower) and needs `FLY_CHESS_MODE=full` set before starting Jupyter to run its multi-hour training budget instead of the default fast smoke-test slice.

```sh
uv sync
uv run jupyter lab fly-heaven.ipynb
```

Start Jupyter from this directory. Downloads and prepared graphs go to `data/`, and run outputs go to `runs/`; both are git-ignored.

## Continuous macro film

The Blender pipeline now uses Joachim Bornemann's **Housefly-Exotic Rigged**
(BlenderKit `1fd6b0b2-d5e4-457e-ad87-62f5002eef40`); see
[`assets/blenderkit-housefly/README.md`](assets/blenderkit-housefly/README.md) for
provenance and native rig mapping. It builds one connected forest floor and
replays the existing `final4` simulation chronologically, including travel and
quiet intervals. The default director no longer reorders or time-lapses events;
`--highlights` explicitly selects the previous edit.

```sh
PYTHONHOME=/usr PATH=/usr/bin:/bin blender --factory-startup -b --python heaven/blender/build_world.py
PYTHONHOME=/usr PATH=/usr/bin:/bin blender --factory-startup -b runs/fly-heaven/continuous/world.blend --python heaven/blender/animate.py -- runs/fly-heaven/final4/timeline.npz runs/fly-heaven/continuous/scene.blend
uv run python -m heaven.director runs/fly-heaven/final4/timeline.parquet runs/fly-heaven/continuous/edl.json
PYTHONHOME=/usr PATH=/usr/bin:/bin blender --factory-startup -b runs/fly-heaven/continuous/scene.blend --python heaven/blender/render.py -- continuous runs/fly-heaven/continuous/frames 1920 1080 64 CYCLES
uv run python -m heaven.overlay runs/fly-heaven/continuous/edl.json runs/fly-heaven/continuous/frames runs/fly-heaven/continuous/fly-heaven.mp4
```

For previewing, pass explicit frame ranges in place of `continuous`, lower the
resolution, and choose `BLENDER_EEVEE`. Always use a separate frames directory
when changing the scene or render settings: a scene checksum guards resumable
renders against stale images. The Cycles final uses OptiX on the local NVIDIA
GPU, adaptive sampling, denoising and a 2K render texture limit; original
source textures are preserved.

`heaven/blender/motion.py` handles interpolation and interrupted transitions
without importing Blender. State labels are held until their recorded onset;
continuous positions interpolate between samples. Gait phase follows travel
rather than neuronal firing rate; IK targets preserve stance contacts. Visual
courtship spacing prevents the simulation's coincident point agents from
interpenetrating, and mounting adds a blended height offset. These remain
illustrative motions, not biomechanical outputs of the connectome. The chosen
housefly is a visual proxy for the MaleCNS fruit-fly brain.

The current recording contains no GROOM or BASK events. Their poses can be
checked with diagnostic timelines but are not inserted into the continuous
film. Ground dressing uses a fixed random seed and clears recorded paths.

For a background final export with automatic encoding and frame-count validation:

```sh
nohup .venv/bin/python -m heaven.blender.finish_film > runs/fly-heaven/continuous/job.log 2>&1 &
```

`job.json` records rendering/encoding/completion or the failure reason;
`render.log` records progress. Restart the same command to resume an interrupted
render. A complete movie is written only after all 5,281 frames exist.
