# How the experiment works

[← Back to the README](../README.md)

## What actually runs

1. An HTML video element plays the downloaded insect playlist. Its decoded frames fill the portrait phone canvas edge to edge, which Three.js displays. Clip changes slide vertically; there is no blurred filler or letterboxing added by the player.
2. The browser reads that composited screen into a **90×160 RGBA** frame. Python flips WebGL's rows and removes alpha. The most recent accepted screen image is saved as `runs/local/latest-input.png`.
3. The upstream inferred visual projection stimulates **3,335 R1–R6 inputs and 811 R8 inputs** from the image's luminance and color.
4. Python calls the compiled C++17 kernel through `ctypes`. It integrates the full retained spiking graph in **0.1 ms** steps.
5. Each accepted frame advances **50 ms of neural time** by default. The overlay shows wall-clock exposure; API telemetry and local logs retain simulated brain time and compute time. It samples the video; it does not claim frame-for-frame biological real-time playback.
6. Actual network spike counts, mean PAM11/KC firing rates, and 10 ms bins for 96 fixed identified cells return to the observation window. Missing/disconnected engines display no measurements; there is no fake fallback.

The playlist automatically advances after three seconds of actual media playback and wraps after the final clip. Pausing, buffering, or hiding the window holds that timer. An early video end also advances to the next clip. Its titles and creator links come from the downloaded metadata. The subject is compelled to receive whichever clip is on screen; the feed's order and playback timing are presentation controls, not learned behavior. Loading, failed, paused, or stalled playback supplies no new observations; the most recent valid measurements remain visible.

## Reward, learning, and movement

**Video-linked PAM11 stimulation is enabled by default.** Each accepted video observation applies a **20 mV-equivalent current** to the **15 annotated PAM11 cells** for the entire neural interval (50 ms by default). The C++ kernel computes the resulting spikes and propagates them through the existing network. This is a fixed artificial input, independent of the clip's identity or a swipe.

The browser withholds frames while playback is loading, paused, buffering, seeking, ended, stale, hidden, or failed. Without an accepted frame the neural worker waits, so no new current or simulated time is delivered. An observation already computing can finish; residual neural activity is not forcibly erased.

**Manual stimulation** (keyboard **P** or WebMCP) schedules a 200 ms pulse at the same amplitude. Manual and automatic drive overlap at 20 mV-equivalent, rather than adding together. Repeated manual requests replace the pending pulse. `stimulus_ms` records the total stimulated interval; `video_stimulus_ms` and `manual_stimulus_ms` record its possibly overlapping sources. `stimulus_current_mv` records the delivered amplitude. Model metadata and provenance record whether video stimulation is enabled.

Use `--no-video-reward` with a separate `--run-dir` for an unstimulated comparison; manual pulses remain available. The low-level `FlyEngine.observe()` API requires explicit `video_reward=True`, so numerical control assays do not receive automatic reward accidentally.

The upstream experimental plasticity rule can modify the **7,835 existing KC→MBON07/11 connections**. The API and saved telemetry report how many differ from baseline. A changing weight alone is not evidence of useful learning, pleasure, attention, or addiction. PAM11 telemetry uses **spikes per neuron per neural second (Hz)**, not a fabricated dopamine concentration or percentage.

The 3D movement is an amplified artistic readout: wing flutter and body/leg motion map MN9/DNp09 firing, head turning maps DNa02 right-minus-left firing, and electrode glow maps PAM11 firing. Motor rates use a saturating response tuned for the observed 5–30 Hz range, with a 100 ms attack and 700 ms release so brief bursts remain visible. Turning is smoothed over 200 ms. This changes only the animation; it adds no neural spikes or stimulation. Ambient breathing, swaying garden foliage, drifting pollen, and screen motion are visual effects. This is not a validated biomechanical fly model.

The display puts a **Dopamine activity** reading and one full-width **PAM11 firing rate** chart over the chamber, with no current-video captions or visible controls. It plots the server's latest 120 measurements of mean PAM11 spikes per neuron per neural second (Hz). The automatic detail scale shows its actual lower and upper bounds, which may start above zero, to make small changes legible. Samples are neither smoothed nor supplemented with artificial fluctuations; a constant signal remains flat. The horizontal position follows simulated timestamps, and the plot holds when no new samples arrive. **Fly spikes** remains the actual count across the full graph in the most recent 50 ms sample. The old network plot and 96-cell raster are removed from the display; their underlying telemetry remains available through the API. PAM11 firing is not a dopamine concentration or a measure of whole-network activity.

The front right leg has an additional choreographed swipe gesture, separate from its measured motor response. Each video transition uses a shared 900 ms timeline: the leg reaches forward, its upward stroke follows the exact same easing as the outgoing video, and it returns to its resting pose. Joint positions preserve the original segment lengths. Automatic advances and manual skips use the same gesture; loading, pause, and buffering hold the shared transition. Reduced-motion mode omits both the leg gesture and the phone slide. This presentation animation does not choose videos, stimulate the neural model, or represent a learned action.

## Controls and persistence

- Playback runs automatically with no visible controls. Drag to orbit; **C** cycles three camera positions and **F** toggles fullscreen.
- Scroll, swipe vertically, or press an arrow key to skip a short.
- **Space** pauses the actual video and stops new neural observations after any already-running step finishes.
- **P** applies PAM11 stimulation to the numerical model.
- **S** requests a checkpoint. Checkpoints are also saved every two active minutes and on **Ctrl-C**.
- Restarting restores neural state and plastic weights from `runs/local/brain.npz`. Queued stimulation is not replayed on restart.
- **M** toggles the video's original audio, which starts muted. Reduced-motion preferences start the experiment paused. Hidden or closed observation windows pause playback and supply no new frames, so the brain waits.
- One observation window at a time supplies the sensory stream. A second can take over after four seconds without input from the first.

`runs/local/events.jsonl` contains actual measurements and input/spike hashes; `latest.json` is the last observation; `provenance.json` records the data/model/source configuration. Neuron IDs are serialized as strings to preserve integer precision.

Useful options:

```sh
uv run flywirehead verify
uv run flywirehead run --no-browser
uv run flywirehead run --neural-ms 100
uv run flywirehead run --run-dir runs/control --no-video-reward --frozen
uv run flywirehead run --run-dir runs/new-experiment --fresh
uv run flywirehead --data /path/to/data run
```

Use a separate `--run-dir` for independent experiments. `--fresh` explicitly starts over and will replace that run's checkpoint when saved. Only one worker can own a run directory. The server binds to loopback, validates local origins, and uses an ephemeral session token for controls. No cloud service or trading credentials are used.

## Validation

```sh
uv sync --extra test
uv run pytest -q
FLYWIREHEAD_FULL_TEST=1 uv run pytest -q -s tests/test_full_connectome.py
node --test tests/*.test.mjs
```

The full-graph assay compares black/white visual input from the same checkpoint, stimulated/control trials, frozen plasticity, and exact replay after restore. It checks actual mechanism behavior, not biological validity or whether the fly has learned to prefer shorts.

## Sources

The numerical backend is adapted from [nftechie/stonkfly](https://github.com/nftechie/stonkfly), commit `78ef3e05ab0fa086032098558d893667068944a0`, under MIT. The local copy includes the source, provenance, lockfiles, and license; it does not depend on the reference clone or import its trading stack. See [THIRD_PARTY.md](../THIRD_PARTY.md) for MaleCNS CC BY 4.0 attribution and [flywirehead/upstream.json](../flywirehead/upstream.json) for source hashes.

The model combines real reconstructed wiring with approximate physiology and an unvalidated experimental memory rule. It does not reproduce a complete living fly or establish consciousness. No real animals are involved.

The optional WebMCP controls share the normal interface actions; status, pause, resume, and next-short actions were verified in the local browser. WebGL 2 and H.264 video playback are needed for the 3D window. Three.js is vendored locally; the Google Fonts stylesheet is optional and falls back to system fonts offline.
