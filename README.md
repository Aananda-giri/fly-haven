# Fly Heaven

One notebook, [`fly-heaven.ipynb`](fly-heaven.ipynb), that reruns two MaleCNS v1.0 connectome experiments on this machine and checks them against their published results:

- **Fruitless** ([`experiments/fruitless`](experiments/fruitless)): does blocking mAL output unmask a P1 courtship-neuron response to candidate male cues?
- **Fly / Wirehead** ([`experiments/fly-wirehead`](experiments/fly-wirehead)): a full connectome watching insect Shorts, with an artificial drive on its PAM11 dopamine neurons.

Both use the same three MaleCNS source files, so the notebook downloads them once (~1.1 GB, checksum-verified) and links them into each project's layout. It calls each project's own simulation code without modifying it.

A second notebook, [`fly-chess.ipynb`](fly-chess.ipynb), grafts a small trainable "cortex" onto the
same frozen connectome and has it play chess, as a comparison against [flychess-hq](https://flychess-hq.vercel.app/)'s
fly-plus-linear-readout design. It reuses the fruitless project's prepared graph, and downloads its
own chess data and a Stockfish binary into `data/fly-chess/`.

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
