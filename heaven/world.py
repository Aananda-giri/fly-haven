"""The scripted world: physiology, objects, a female agent, and a display-proxy retina.

Nothing here is neural. This module turns the fly's situation into sensory
currents for `heaven.brain.ConnectomeBrain`, and turns the brain's motor
readouts (via `heaven.behaviour`) back into positions or poses for the Blender
pipeline. Distances are metres, times are seconds; the clearing is fly-scale
(about 1.2 x 0.8 m), matching the scattered forest in `heaven/blender/build_world.py`.
"""

from dataclasses import dataclass, field

import numpy as np

CLEARING = (1.2, 0.8)
SUNBEAM_POS = np.array([0.30, 0.55])
SUNBEAM_RADIUS = 0.18
# The fruit sits just inside the sunbeam's warm halo (not stacked in the exact
# center, which is reserved for basking) so a hungry fly doesn't have to choose
# between thermal comfort and foraging across the whole clearing -- a real
# fallen fruit warming in a sunny spot is exactly this kind of coincidence.
FRUIT_POS = np.array([0.42, 0.51])
FRUIT_RADIUS = 0.05
SHADE_TEMP_C = 17.0
SUN_TEMP_C = 34.0
COMFORT_TEMP_C = (21.0, 27.0)
TAP_RANGE = 0.006  # foreleg tapping distance
PULSE_ON_S = 0.3    # a sense is delivered at full strength for this long per period...
PULSE_PERIOD_S = 1.0  # ...out of this long, whenever its underlying need is present.
# Found this session: holding a current on this full connectome continuously for
# 10+ seconds pushes the recurrent network into a persistently elevated firing
# regime that outlasts the stimulus by tens of seconds (weights are frozen, so
# this is intrinsic membrane/recurrent dynamics, not learning). Calibration's own
# 0.3-0.6 s probes never triggered it. So every held sense (not the brief, self-
# limiting sugar/female contacts) is delivered as a real duty-cycled pulse train
# within that validated envelope, rather than a sustained analog current -- both
# a stability fix and closer to how phasic sensory afferents actually behave.
LEAF_FALL_START_T = 85.0
LEAF_FALL_DURATION = 2.2
LEAF_START_HEIGHT = 0.5
FEMALE_ARRIVE_T = 125.0


def _ambient_temp(pos):
    d = np.linalg.norm(pos - SUNBEAM_POS)
    warm = max(0.0, 1 - d / SUNBEAM_RADIUS)
    return SHADE_TEMP_C + warm * (SUN_TEMP_C - SHADE_TEMP_C)


@dataclass
class Physiology:
    satiety: float = 0.35  # 0 starved, 1 full
    temperature_c: float = SHADE_TEMP_C
    dust: float = 0.15  # 0 clean, 1 filthy

    def step(self, dt, pos, feeding, grooming, flying):
        self.satiety = max(0.0, self.satiety - dt / 90.0)  # empties over ~90 s
        if feeding:
            self.satiety = min(1.0, self.satiety + dt / 3.0)  # a fruit fills it in ~3 s
        target = _ambient_temp(pos)
        tau = 4.0
        self.temperature_c += (target - self.temperature_c) * (1 - np.exp(-dt / tau))
        gain = (0.06 if flying else 0.015) * dt
        self.dust = min(1.0, self.dust + gain)
        if grooming:
            self.dust = max(0.0, self.dust - dt / 2.0)


@dataclass
class Agent:
    pos: np.ndarray
    heading: float = 0.0
    altitude: float = 0.0  # 0 = walking on the ground, >0 = airborne

    def step_toward(self, target, dt, speed):
        delta = target - self.pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return dist
        direction = delta / dist
        self.heading = float(np.arctan2(direction[1], direction[0]))
        move = min(dist, speed * dt)
        self.pos = self.pos + direction * move
        return dist - move


@dataclass
class Leaf:
    pos: np.ndarray = field(default_factory=lambda: np.array([0.75, 0.20]))
    height: float = LEAF_START_HEIGHT
    falling: bool = False
    landed: bool = False
    prev_angular_size: float = 0.0

    def step(self, t, dt, fly_pos):
        if not self.falling and not self.landed and t >= LEAF_FALL_START_T:
            self.falling = True
            # Seed the reference size from the resting leaf, not 0, so the first
            # falling tick reports a real (small) rate instead of a fake spike.
            ground_dist = max(0.02, float(np.linalg.norm(fly_pos - self.pos)))
            self.prev_angular_size = 0.05 / max(self.height + ground_dist, 0.02)
        loom_rate = 0.0
        if self.falling:
            self.height = max(0.0, LEAF_START_HEIGHT * (1 - (t - LEAF_FALL_START_T) / LEAF_FALL_DURATION))
            ground_dist = max(0.02, float(np.linalg.norm(fly_pos - self.pos)))
            apparent_size = 0.05 / max(self.height + ground_dist, 0.02)
            loom_rate = max(0.0, (apparent_size - self.prev_angular_size) / max(dt, 1e-6))
            self.prev_angular_size = apparent_size
            if self.height <= 0.0:
                self.falling = False
                self.landed = True
        return loom_rate


class World:
    """Owns the male fly's physiology plus every scripted actor and prop."""

    def __init__(self, seed=0, hunger_start=0.35):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self._on_time = {"heat": 0.0, "cold": 0.0, "antenna": 0.0, "sugar": 0.0}
        self.fly = Agent(pos=np.array([0.15, 0.15]))
        self.physiology = Physiology(satiety=hunger_start, temperature_c=SHADE_TEMP_C)
        self.leaf = Leaf()
        self.female = Agent(pos=np.array([-1.0, -1.0]))  # off-scene until she arrives
        self.female_present = False
        self.song_time = 0.0
        self.female_receptive = False

    def _female_wander(self, dt, holding_still):
        if not self.female_present:
            if self.t >= FEMALE_ARRIVE_T:
                self.female_present = True
                self.female.pos = np.array([1.05, 0.65])
            return
        if holding_still:
            return  # she holds once courtship engages closely; otherwise she forages
        wander_target = FRUIT_POS + self.rng.normal(0, 0.03, size=2)
        self.female.step_toward(wander_target, dt, speed=0.03)

    def _pulse(self, name, raw, dt):
        if raw <= 0.5:
            self._on_time[name] = 0.0
            return 0.0
        self._on_time[name] = (self._on_time[name] + dt) % PULSE_PERIOD_S
        return raw if self._on_time[name] < PULSE_ON_S else 0.0

    def _currents(self, dt):
        """Sensory currents for `ConnectomeBrain.step`, in mV-equivalent, from the
        fly's *current* situation (called before this tick's movement)."""
        near_fruit = np.linalg.norm(self.fly.pos - FRUIT_POS) < FRUIT_RADIUS
        hunger_gate = max(0.0, 1 - self.physiology.satiety / 0.6)
        sugar_raw = 20.0 * hunger_gate if near_fruit else 0.0
        # Pulsed for the same reason as heat/cold/antenna below: a fly sitting
        # right at the fruit radius boundary can hold sugar current on for
        # tens of seconds, which this session found drives the network into a
        # persistently distorted state (here, MN9 stuck silent) that a normal
        # few-second feeding bout never triggers.
        sugar = self._pulse("sugar", sugar_raw, dt)
        antenna = self._pulse("antenna", 22.0 * self.physiology.dust, dt)
        heat_raw = max(0.0, self.physiology.temperature_c - COMFORT_TEMP_C[1]) / 6.0 * 20.0
        cold_raw = max(0.0, COMFORT_TEMP_C[0] - self.physiology.temperature_c) / 6.0 * 20.0
        heat = self._pulse("heat", heat_raw, dt)
        cold = self._pulse("cold", cold_raw, dt)
        female = 0.0
        female_distance = None
        if self.female_present:
            female_distance = float(np.linalg.norm(self.female.pos - self.fly.pos))
            if female_distance < TAP_RANGE:
                female = 20.0
        return {
            "sugar": float(np.clip(sugar, 0, 20)),
            "antenna": float(np.clip(antenna, 0, 20)),
            "heat": float(np.clip(heat, 0, 20)),
            "cold": float(np.clip(cold, 0, 20)),
            "female": female,
            "near_fruit": bool(near_fruit),
            "female_distance": female_distance,
        }

    def retina_frame(self):
        """A cheap 90x160 panorama: sky, dappled canopy, ground, and any silhouette
        near the fly. This is a display proxy, exactly like fly-wirehead's own
        retinal input (`experiments/fly-wirehead/flywirehead/neural/sensory.py`);
        it is not a calibrated compound-eye render."""
        frame = np.empty((160, 90, 3), np.uint8)
        frame[:60] = (205, 222, 236)
        frame[60:] = (70, 92, 48)
        sun_col = int(np.clip(45 * (1 - np.linalg.norm(self.fly.pos - SUNBEAM_POS) / SUNBEAM_RADIUS), 0, 45))
        frame[:60, :, 0] = np.clip(frame[:60, :, 0].astype(int) + sun_col, 0, 255)
        if self.leaf.falling:
            col = int(30 * self.leaf.height / LEAF_START_HEIGHT)
            frame[70:100, 40:55] = np.clip(frame[70:100, 40:55].astype(int) - col, 0, 255)
        return frame

    def sense(self, dt):
        """Advance the leaf (the only thing that changes independently of the
        fly's chosen action) and return this tick's sensory currents."""
        self.t += dt
        loom_rate = self.leaf.step(self.t, dt, self.fly.pos)
        cur = self._currents(dt)
        # LOOM_GAIN is tuned (this session) so the leaf's actual fall reaches
        # roughly calibration's tested 20-30 mV range at its fastest expansion,
        # not a biomechanically calibrated angular-velocity-to-current transfer.
        LOOM_GAIN = 450.0
        cur["loom"] = float(np.clip(loom_rate * LOOM_GAIN, 0, 30))
        self._last_currents = cur
        return cur

    def act(self, dt, action):
        """Apply the behaviour-chosen motor program: move the fly (and the
        scripted female), and update physiology accordingly."""
        pos = self.fly.pos.copy()
        if action.get("target_pos") is not None and not action.get("flying"):
            self.fly.step_toward(action["target_pos"], dt, max(action.get("speed", 0.0), 0.015))
        elif action.get("flying") and action.get("target_pos") is not None:
            self.fly.altitude = min(0.15, self.fly.altitude + dt * 0.3)
            self.fly.step_toward(action["target_pos"], dt, 0.35)
        if not action.get("flying"):
            self.fly.altitude = max(0.0, self.fly.altitude - dt * 0.3)
        holding_still = action["state"] == "COURT" and action["subphase"] in ("TAP", "SING", "ATTEMPT", "MATE")
        self._female_wander(dt, holding_still)
        self.physiology.step(dt, pos, feeding=action.get("feeding", False), grooming=action.get("grooming", False), flying=action.get("flying", False))
        if action["state"] == "COURT" and action["subphase"] == "SING":
            self.song_time += dt
        elif action["state"] != "COURT":
            self.song_time = max(0.0, self.song_time - dt * 0.5)

    def snapshot(self):
        cur = getattr(self, "_last_currents", None) or self._currents(0.0)
        return {
            "t": self.t,
            "fly_x": float(self.fly.pos[0]),
            "fly_y": float(self.fly.pos[1]),
            "fly_heading": self.fly.heading,
            "fly_altitude": self.fly.altitude,
            "female_x": float(self.female.pos[0]),
            "female_y": float(self.female.pos[1]),
            "female_present": self.female_present,
            "female_distance": cur["female_distance"],
            "leaf_x": float(self.leaf.pos[0]),
            "leaf_y": float(self.leaf.pos[1]),
            "leaf_height": self.leaf.height,
            "leaf_falling": self.leaf.falling,
            "satiety": self.physiology.satiety,
            "temperature_c": self.physiology.temperature_c,
            "dust": self.physiology.dust,
            "song_time": self.song_time,
            "near_fruit": cur["near_fruit"],
        }
