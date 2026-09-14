# Fly Heaven

One notebook, [`fly-heaven.ipynb`](fly-heaven.ipynb), that reruns two MaleCNS v1.0 connectome experiments on this machine and checks them against their published results:

- **Fruitless** ([`experiments/fruitless`](experiments/fruitless)): does blocking mAL output unmask a P1 courtship-neuron response to candidate male cues?
- **Fly / Wirehead** ([`experiments/fly-wirehead`](experiments/fly-wirehead)): a full connectome watching insect Shorts, with an artificial drive on its PAM11 dopamine neurons.

Both use the same three MaleCNS source files, so the notebook downloads them once (~1.1 GB, checksum-verified) and links them into each project's layout. It calls each project's own simulation code without modifying it.

The [`fly-chess`](https://github.com/Aananda-giri/fly-chess) project — a small trainable "cortex"
grafted onto this same frozen connectome, playing chess — used to live in this repo and has since
moved to its own. Its earliest notebooks were built reusing this repo's `experiments/fruitless`
prepared graph; from V3 onward it downloads the connectome itself and needs no checkout of this
repo at all.

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

Requires [uv](https://docs.astral.sh/uv/), a C++17 compiler, and FFmpeg (plus yt-dlp if the fly-wirehead videos aren't downloaded yet).

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
