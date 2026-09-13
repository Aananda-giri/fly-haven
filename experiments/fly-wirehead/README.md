# Fly / Wirehead

**Born to fly. Forced to scroll.**

![A wired fly watching insect Shorts on a phone in a bright garden, with live dopamine-neuron activity and fly spike counts.](docs/assets/fly-wirehead.png)

A fly-connectome simulation watching an endless feed of insect videos. **166,700 neurons. 25.6 million connections. A new Short every three seconds.**

The phone plays real footage. Its pixels stimulate the reconstructed fly network, and the network's measured activity drives the fly's movements and the live neural overlay. The brain runs locally in Python and C++; Three.js renders the observation chamber in your browser.

Inspired by [Stonkfly](https://github.com/nftechie/stonkfly). This fly has been given a phone.

## Run it

You'll need **Python 3.11+**, a **C++17 compiler**, [uv](https://docs.astral.sh/uv/), **yt-dlp**, and **FFmpeg**. Allow several GB of disk space; **16 GB RAM** is recommended. Use a browser with WebGL 2 and H.264 playback.

On macOS, install the compiler with `xcode-select --install` if needed, and the tools with `brew install uv yt-dlp ffmpeg`. Then:

```sh
git clone https://github.com/mattyhempstead/fly-wirehead.git
cd fly-wirehead
uv sync
uv run flywirehead prepare
uv run python scripts/download_videos.py
uv run flywirehead run
```

The first preparation downloads about **1.1 GB** of neural data and verifies its checksums. The video script downloads the five selected YouTube Shorts and prepares portrait MP4s. The native neural kernel builds on first launch.

The last command opens [localhost:4173](http://127.0.0.1:4173/). Keep the Python process running. **Ctrl-C** saves the brain and stops the server; running it again resumes the saved state. On macOS, you can also use `run.command` after setup.

<details>
<summary>Using pip instead of uv</summary>

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m flywirehead prepare
python scripts/download_videos.py
python -m flywirehead run
```

</details>

## What the fly receives

The browser captures the phone's image at **90×160 pixels**. Luminance and color stimulate **3,335 R1–R6 inputs and 811 R8 inputs** in the full retained MaleCNS v1.0 graph. A compiled spiking-network kernel advances **50 ms of neural time** per accepted frame, using **0.1 ms integration steps**.

The overlay displays the measured **PAM11 dopamine-neuron firing rate** and whole-network spike counts. A full-width chart shows the latest 120 dopamine readings with labeled axis bounds and an automatic detail scale, making real rises and falls visible. The values are firing rates in Hz, not dopamine concentrations. Wing, body, leg, and head motion are amplified readouts of measured motor activity. Playback pauses when the window is hidden; stalled or missing video supplies no new observations.

The front right leg also performs a choreographed swipe: it reaches forward, sweeps upward with the video, and returns to the platform. The gesture and phone transition share one animation clock.

**The wiring is reconstructed; the physiology and movement mapping are approximations.** The feed advances on a timer, so the fly does not choose videos. While videos play, an artificial current boosts the 15 PAM11 dopamine neurons; an experimental synaptic plasticity rule can change existing connections. Learned preference, pleasure, and addiction have not been established. No living fly is involved.

[How the model works →](docs/model.md) · [Validation and measured results →](docs/validation.md)

## Artificial dopamine drive

Every accepted video frame supplies a **20 mV-equivalent current** to the model's **15 PAM11 dopamine cells** for that observation's neural time. The displayed neural measurements and electrode glow reflect the resulting spikes. Paused, buffering, hidden, or failed playback supplies no new observations or current; an observation already computing may finish.

The manual **P** pulse uses the same current and does not stack with the automatic drive. To run an unstimulated comparison in its own run directory:

```sh
uv run flywirehead run --no-video-reward --run-dir runs/control
```

## The feed

Five fly and insect Shorts play in a loop, swiping every **three seconds of playback**. Every prepared clip is **360×640 (9:16)**; the final source is trimmed to its first four seconds. Playback works offline once the videos are prepared.

Edit [video-sources.json](video-sources.json) and rerun the downloader to change the playlist. Titles, creators, source links, and file hashes are recorded in the [video credits](dist/media/playlist.json). Videos, datasets, and brain checkpoints stay local and are excluded from Git.

## Controls

The chamber has no visible controls. It starts playing automatically with audio muted.

| Input | Action |
| --- | --- |
| Drag | Orbit the camera |
| Scroll, vertical swipe, or up/down arrows | Next Short |
| Space | Pause / resume video and neural input |
| C / F | Change camera / toggle fullscreen |
| M | Toggle original video audio |
| P | Apply a 200 ms current pulse to the 15 PAM11 dopamine cells |
| S | Save a brain checkpoint |

Checkpoints also save every two active minutes. Run directories, API telemetry, independent experiments, and optional WebMCP controls are covered in the [model and operation notes](docs/model.md#controls-and-persistence).

## Check it

```sh
uv sync --extra test
uv run pytest -q
node --test tests/*.test.mjs
```

After preparing the dataset, run the full-network assays with:

```sh
FLYWIREHEAD_FULL_TEST=1 uv run pytest -q -s tests/test_full_connectome.py
```

## Credits

The neural backend is adapted from [nftechie/stonkfly](https://github.com/nftechie/stonkfly), with its MIT notice preserved. Wiring data comes from the **MaleCNS v1.0** dataset under **CC BY 4.0**. The observation chamber uses **Three.js**. Video creators retain the rights to their footage and audio.

See [sources and licenses](THIRD_PARTY.md) and [the pinned upstream revision](flywirehead/upstream.json).
