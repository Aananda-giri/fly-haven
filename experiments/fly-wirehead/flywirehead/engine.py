"""RGB observations and measured telemetry for the retained MaleCNS network."""

import hashlib
import time

import numpy as np

from .neural.common import annotations
from .neural.visual import VisualMemoryBrain

FRAME_WIDTH, FRAME_HEIGHT = 90, 160
PAM11_CURRENT_MV = 20.0


def decode_frame(body: bytes) -> np.ndarray:
    """The browser sends WebGL's bottom-up RGBA pixels, with a fixed shape."""
    if len(body) != FRAME_WIDTH * FRAME_HEIGHT * 4:
        raise ValueError(f"Expected {FRAME_WIDTH}×{FRAME_HEIGHT} RGBA pixels")
    return np.ascontiguousarray(
        np.frombuffer(body, np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, 4)[::-1, :, :3]
    )


class FlyEngine:
    def __init__(self, *, frozen=False):
        self.brain = VisualMemoryBrain()
        self.brain.weights_frozen = frozen
        a = annotations(self.brain.ids)
        self.types = a.type.fillna("")
        self.left = np.flatnonzero(self.types.eq("DNa02") & a.somaSide.eq("L"))
        self.right = np.flatnonzero(self.types.eq("DNa02") & a.somaSide.eq("R"))
        self.motor = np.flatnonzero(self.types.isin(["MN9", "DNp09"]))
        b = self.brain
        chosen = list(dict.fromkeys(map(int, np.r_[b.circuit["dan"], b.retina[:24], b.r8[:8], b.circuit["mb"], self.motor, b.circuit["kc"][:32]])))
        if len(chosen) < 96:
            chosen = list(dict.fromkeys(chosen + list(map(int, np.linspace(0, b.n - 1, 96)))))
        self.sample = np.asarray(chosen[:96], dtype=np.int32)
        self.sample_cells = [{"id": str(b.ids[i]), "type": str(self.types.iloc[i])} for i in self.sample]
        self.pending_pulse_ms = 0.0

    def stimulate(self):
        # Repeated button presses replace the pending pulse, never accumulate it.
        self.pending_pulse_ms = 200.0

    def observe(self, frame, duration_ms=50.0, *, video_reward=False):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError("A nonempty RGB uint8 frame is required")
        if not np.isfinite(duration_ms) or not .1 <= duration_ms <= 500 or abs(duration_ms * 10 - round(duration_ms * 10)) > 1e-7:
            raise ValueError("Duration must be 0.1–500 ms in 0.1 ms increments")
        if not isinstance(video_reward, bool):
            raise ValueError("Video reward must be a boolean")
        b = self.brain
        started = time.perf_counter()
        counts = np.zeros(b.n, np.int64)
        bins = []
        ticks = round(duration_ms / b.dt)
        delivered = manual_delivered = 0.0
        while ticks:
            n = min(100, ticks)
            if self.pending_pulse_ms > 0:
                n = min(n, round(self.pending_pulse_ms / b.dt))
            interval = n * b.dt
            manual = self.pending_pulse_ms > 0
            # Watching supplies a bounded current to the actual annotated cells.
            # A manual pulse overlaps that drive; it never doubles the amplitude.
            stimulus = (b.circuit["reward"], PAM11_CURRENT_MV) if video_reward or manual else None
            spikes, _ = b.rgb_step(frame, interval, learning=not b.weights_frozen, stimulation=stimulus)
            counts += spikes
            bins.append({"end_ms": round(b.sim_ms, 3), "duration_ms": interval, "counts": spikes[self.sample].tolist()})
            if stimulus is not None:
                delivered += interval
            if manual:
                manual_delivered += interval
                self.pending_pulse_ms = max(0.0, round(self.pending_pulse_ms - interval, 6))
            ticks -= n
        b.counts[:] = counts
        seconds = duration_ms / 1000

        def mean_rate(indices):
            return float(counts[indices].sum() / (max(1, len(indices)) * seconds))

        return {
            "sim_ms": round(b.sim_ms, 3),
            "interval_ms": duration_ms,
            "compute_seconds": time.perf_counter() - started,
            "total_spikes": int(counts.sum()),
            "network_spikes_per_second": float(counts.sum() / seconds),
            "cumulative_spikes": b.total_spikes,
            "pam11_hz": mean_rate(b.circuit["reward"]),
            "pam11_spikes": int(counts[b.circuit["reward"]].sum()),
            "ppl101_hz": mean_rate(b.circuit["aversive"]),
            "kc_hz": mean_rate(b.circuit["kc"]),
            "kc_spikes": int(counts[b.circuit["kc"]].sum()),
            "motor_hz": mean_rate(self.motor),
            "turn_hz": mean_rate(self.right) - mean_rate(self.left),
            "mean_voltage_mv": float(b.v.mean()),
            "stimulus_ms": round(delivered, 3),
            "stimulus_current_mv": PAM11_CURRENT_MV if delivered else 0.0,
            "video_stimulus_ms": round(duration_ms, 3) if video_reward else 0.0,
            "manual_stimulus_ms": round(manual_delivered, 3),
            "pending_stimulus_ms": self.pending_pulse_ms,
            "input_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "spike_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
            "memory": b.memory(),
            "bins": bins,
        }
