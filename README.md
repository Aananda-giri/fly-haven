# Fly Heaven

One notebook, [`fly-heaven.ipynb`](fly-heaven.ipynb), that reruns two MaleCNS v1.0 connectome experiments on this machine and checks them against their published results:

- **Fruitless** ([`experiments/fruitless`](experiments/fruitless)): does blocking mAL output unmask a P1 courtship-neuron response to candidate male cues?
- **Fly / Wirehead** ([`experiments/fly-wirehead`](experiments/fly-wirehead)): a full connectome watching insect Shorts, with an artificial drive on its PAM11 dopamine neurons.

Both use the same three MaleCNS source files, so the notebook downloads them once (~1.1 GB, checksum-verified) and links them into each project's layout. It calls each project's own simulation code without modifying it.

A second notebook, [`fly-chess.ipynb`](fly-chess.ipynb), grafts a small trainable "cortex" onto the
same frozen connectome and has it play chess, as a comparison against [flychess-hq](https://flychess-hq.vercel.app/)'s
fly-plus-linear-readout design. It reuses the fruitless project's prepared graph, and downloads its
own chess data and a Stockfish binary into `data/fly-chess/`.

**This `version-2` worktree/branch is a correctness pass over that notebook**, prompted by review
findings against the original (a `stockfish_cp_loss` perspective bug, a PUCT selection/backup sign
error, an unsupported absolute-Elo estimate, a lesion control that couldn't prove its own claim, and
several architecture/spec drifts — see `runs/fly-chess-v2/*/manifest.json` for the corrected design's
exact identity). It keeps the original notebook and its already-running full-mode job on `master`
untouched, and adds: a 4,184-class move vocabulary with distinct underpromotions, side-to-move value
labels, a linear (not MLP) motor decoder, mover-perspective Stockfish scoring, pending-leaf-safe PUCT
with `1 − child.q()` opponent-perspective backup, matched-step controls (C0/C1/C2) plus lesion/no-graft/
no-senses/relay-permute interventions, W/D/L + paired-opening match results with unresolved games (no
invented Elo), atomic checkpoints with full RNG state for exact resume, and an explicit
`baseline_unavailable` result instead of assuming `EF-Code/flychess` or `Noeljarillo/chessfly` is the
deployed site. `tests/test_fly_chess_v2.py` exercises the notebook's own tagged cell definitions
(move round-trips, label perspective, search backup/mate-in-one, checkpoint resume, shuffle invariants,
active-subgraph gradient equivalence, …) without downloading data or training. Run it with
`FLY_CHESS_DEVICE=cpu FLY_CHESS_MODE=smoke` for a synthetic-graph pipeline check in minutes, add
`FLY_CHESS_REAL_GRAPH=1` to exercise the real MaleCNS graph (166,606 neurons, 25.57M edges) without a
GPU, or `FLY_CHESS_MODE=full` for the real four-hour pilot once the GPU is free.

## Fly Heaven: a connectome-driven film

[`heaven/`](heaven/) puts the rigged green bottle fly ([`assets/iridescent-green-bottle-fly-3d-model-free`](assets/iridescent-green-bottle-fly-3d-model-free)) into a fly-scale forest clearing built from [`assets/stylized-hand-painted-scene`](assets/stylized-hand-painted-scene), and renders a short film of it living there. **The full 166,700-neuron MaleCNS v1.0 brain (the same one `fly-wirehead` uses) decides what the fly does and when** -- feeding, grooming, escaping a falling leaf, courting and mating are all gated on calibrated firing-rate thresholds of real annotated cell types (`MN9`, `DNg12`, `DNp01`, P1, `pIP10`, `MDN`), not a scripted state machine. Only the world around that brain is scripted: physiology (hunger, temperature, dust), the female fly (no female connectome exists locally), navigation toward a sensed goal, and cosmetic poses. `heaven/brain.py`'s `describe()` and `heaven/calibrate.py`'s report say exactly which cells drive which behaviour.

```sh
uv run python -m heaven.calibrate                                        # verifies each brain->behaviour route, writes runs/fly-heaven/calibration.json
uv run python -m heaven.simulate --seconds 220 --out runs/fly-heaven/day1 # the closed loop: brain <-> behaviour <-> world
uv run python -m heaven.director runs/fly-heaven/day1/timeline.parquet runs/fly-heaven/day1/edl.json
uv run python -m heaven.blender.export_timeline runs/fly-heaven/day1/timeline.parquet   # Blender's Python has numpy but not pandas

PYTHONHOME=/usr PATH=/usr/bin:/bin blender -b --factory-startup --python heaven/blender/build_world.py
PYTHONHOME=/usr PATH=/usr/bin:/bin blender -b runs/fly-heaven/assets/world.blend --python heaven/blender/animate.py -- \
    runs/fly-heaven/day1/timeline.npz runs/fly-heaven/assets/scene.blend
PYTHONHOME=/usr PATH=/usr/bin:/bin blender -b runs/fly-heaven/assets/scene.blend --python heaven/blender/render.py -- \
    runs/fly-heaven/day1/edl.json runs/fly-heaven/day1/frames
uv run python -m heaven.overlay runs/fly-heaven/day1/edl.json runs/fly-heaven/day1/frames runs/fly-heaven/fly-heaven.mp4
```

The `PYTHONHOME=/usr PATH=/usr/bin:/bin` prefix avoids the local Python installation conflicting with system Blender, same as the fly rig's own build script. To check that a behaviour is really brain-gated and not an artifact, rerun `heaven.simulate` with `--lesion MN9` (or `P1`, `GF`, `DNg12`) and compare its `summary.json` against the intact run -- the lesioned behaviour should disappear.

## Run the notebooks

Requires [uv](https://docs.astral.sh/uv/), a C++17 compiler, and FFmpeg (plus yt-dlp if the fly-wirehead videos aren't downloaded yet). `fly-chess.ipynb` additionally wants a CUDA GPU (it will run on CPU, just far slower) and needs `FLY_CHESS_MODE=full` set before starting Jupyter to run its multi-hour training budget instead of the default fast smoke-test slice.

```sh
uv sync
uv run jupyter lab fly-heaven.ipynb
```

Start Jupyter from this directory. Downloads and prepared graphs go to `data/`, and run outputs go to `runs/`; both are git-ignored.
