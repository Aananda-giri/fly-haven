"""Turn smoothed connectome readouts into a fly's motor program.

The brain decides *whether and when* each program runs, by clearing (or
falling below) calibrated firing-rate thresholds. This module owns only the
priority order, the hysteresis and timing needed to turn a spike rate into a
held behaviour, and the kinematic/pose target each behaviour implies. That
split mirrors fly-wirehead's own animation code
(`experiments/fly-wirehead/dist/motion.js`): neural measurement in, an
"amplified artistic readout" out.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import world as W
from .brain import RUNS

ATTACK_S = 0.10
RELEASE_S = 0.70
FLOOR_TAU_S = 4.0  # tracks the ambient/tonic firing rate so decisions gate on the phasic rise above it
QUIET_MOTOR_HZ = 8.0  # experiments/fly-wirehead/docs/validation.md: median measured motor firing is ~5-8 Hz at rest
EXIT_FRACTION = 0.6  # a state holds until its readout drops below this fraction of threshold
# Some readouts are only a handful of cells (MDN: 4, DNa02/GF/pIP10: 2), so a
# single 50 ms bin's spike count is Poisson-noisy relative to their calibrated
# thresholds. Requiring the delta to clear threshold continuously for this long
# before *entering* a new state (not while holding one already) filters that
# counting noise without slowing a real transition down noticeably.
DEBOUNCE_S = 0.25
ESCAPE_HOP_S = 1.1
MAX_FEED_S = 8.0  # satiety normally saturates in ~3 s (heaven.world.Physiology); this is only a safety cap
MAX_GROOM_S = 20.0
MAX_THERMO_S = 40.0  # generous: crossing the whole clearing at the amplified walking speed takes well under this
MAX_FRUIT_WAIT_S = 6.0  # give up waiting for MN9 and retry shortly after, rather than holding forever
TAP_S = 0.3
SING_TIMEOUT_S = 45.0
ATTEMPT_S = 1.0
MATE_S = 14.0
DISMOUNT_S = 1.5
FOLLOW_RANGE = 0.03
CENTER = np.array([W.CLEARING[0] / 2, W.CLEARING[1] / 2])


def load_thresholds(path=None):
    path = Path(path) if path else RUNS / "calibration.json"
    return json.loads(path.read_text())["readout_thresholds_hz"]


class Smoother:
    """One EMA per channel, with a configurable attack and release tau."""

    def __init__(self, channels, attack=ATTACK_S, release=RELEASE_S):
        self.value = {c: 0.0 for c in channels}
        self.attack, self.release = attack, release

    def step(self, raw, dt):
        for c, target in raw.items():
            if c not in self.value:
                continue
            tau = self.attack if target > self.value[c] else self.release
            self.value[c] += (target - self.value[c]) * (1 - np.exp(-dt / tau))
        return self.value


@dataclass
class FlyBehavior:
    thresholds: dict
    seed: int = 0
    state: str = "EXPLORE"
    subphase: str = None
    timer: float = 0.0
    has_mated: bool = False
    rng: np.random.Generator = field(default=None)
    wander_target: np.ndarray = field(default_factory=lambda: CENTER.copy())
    wander_heading: float = 0.0

    def __post_init__(self):
        channels = ["MN9", "DNg12", "GF", "P1", "pIP10", "MDN", "motor", "DNa02_L", "DNa02_R"]
        self.smoother = Smoother(channels)
        # Calibration found that ambient visual drive alone moves these readouts'
        # *baseline* (e.g. a bright frame lifted resting MN9 to 25 Hz and MDN to
        # 23 Hz with no sugar or thermal cue at all -- see heaven/calibrate.py).
        # So decisions gate on the phasic rise above a slow-tracked floor, exactly
        # how heaven.calibrate derived each threshold (evoked-minus-baseline),
        # not on the raw rate.
        self.floor = Smoother(channels, attack=FLOOR_TAU_S, release=FLOOR_TAU_S)
        self._above = {c: 0.0 for c in channels}
        self._fruit_wait = 0.0
        self._fruit_cooldown = 0.0
        self.rng = np.random.default_rng(self.seed)

    def _enter(self, readout, delta, dt, debounce=DEBOUNCE_S):
        above = delta[readout] > self.thresholds[readout]
        self._above[readout] = self._above[readout] + dt if above else 0.0
        return above and self._above[readout] >= debounce

    def _hold(self, readout, delta):
        return delta[readout] > self.thresholds[readout] * EXIT_FRACTION

    def tick(self, raw_rates, dt, snapshot):
        fast = self.smoother.step(raw_rates, dt)
        floor = self.floor.step(raw_rates, dt)
        r = {k: fast[k] - floor[k] for k in fast}  # the phasic signal decisions gate on
        turn_bias = np.clip((fast["DNa02_R"] - fast["DNa02_L"]) / 15.0, -1.0, 1.0)
        speed = float(np.clip(fast["motor"] / 25.0, 0.0, 1.0)) * 0.05  # m/s, an amplified readout, not calibrated gait speed
        self.timer += dt

        action = {
            "state": self.state,
            "subphase": self.subphase,
            "target_pos": None,
            "speed": speed,
            "turn_bias": turn_bias,
            "flying": False,
            "feeding": False,
            "grooming": False,
            "proboscis": 0.0,
            "wing_song": False,
            "pose": "walk",
            "motor_rate_hz": fast["motor"],
            "readouts": dict(fast),
        }

        # Priority 1: escape. A momentary flight hop; nothing else can interrupt it
        # until the hop timer runs out (checked here, not just re-triggered below,
        # so the reported state never lags one tick behind the internal transition).
        if self.state == "ESCAPE":
            if self.timer < ESCAPE_HOP_S:
                action.update(flying=True, pose="flight", target_pos=self._away_from(snapshot))
                return action
            self._enter_state("EXPLORE")
            action["state"], action["subphase"] = self.state, self.subphase
        if self._enter("GF", r, dt, debounce=0.0):  # a startle must be immediate
            self._enter_state("ESCAPE")
            action.update(state="ESCAPE", subphase=None, flying=True, pose="flight", target_pos=self._away_from(snapshot))
            return action

        # Priority 2: courtship, once a female is present.
        if self.state == "COURT":
            result = self._court_tick(action, r, snapshot, dt)
            if result is not None:
                return result
        if snapshot["female_present"] and self._enter("P1", r, dt) and not self.has_mated:
            self._enter_state("COURT", subphase="FOLLOW")
            return self._court_tick(action, r, snapshot, dt)

        # Priorities 3-5 all follow the same "bout" pattern as ESCAPE above: the
        # brain's calibrated threshold-crossing decides whether to *start* the
        # behaviour, but once started it runs to a physical completion
        # condition (full, clean, comfortable) or a safety-cap duration,
        # instead of re-checking the readout every tick. That's a deliberate
        # choice, not just style: this session found that holding any of these
        # currents on for more than a few seconds (even duty-cycled, see
        # PULSE_ON_S in heaven/world.py) can push the full recurrent network
        # into a persistently elevated firing state that outlasts the
        # stimulus by tens of seconds -- so a *continuous* readout is not a
        # reliable way to ask "should I still be doing this?" in this model,
        # only a calibrated onset is. MN9 also isn't a clean single-purpose
        # channel here -- calibration's own exploratory probe found a thermal
        # current alone raises it too -- so feeding additionally requires the
        # fly to actually be at the fruit, the way real proboscis extension
        # needs gustatory contact, not just an efferent copy of the command.
        not_full = snapshot["satiety"] < 0.95
        can_feed = not_full and snapshot["near_fruit"]
        if self.state == "FEED":
            if can_feed and self.timer < MAX_FEED_S:
                action.update(feeding=True, proboscis=1.0, pose="feed", target_pos=W.FRUIT_POS)
                return action
            self._enter_state("EXPLORE")
        if self._enter("MN9", r, dt) and can_feed:
            self._enter_state("FEED")
            action.update(state="FEED", feeding=True, proboscis=1.0, pose="feed", target_pos=W.FRUIT_POS)
            return action

        # Priority 4: groom.
        dusty = snapshot["dust"] > 0.03
        if self.state == "GROOM":
            if dusty and self.timer < MAX_GROOM_S:
                action.update(grooming=True, pose="groom")
                return action
            self._enter_state("EXPLORE")
        if self._enter("DNg12", r, dt) and dusty:
            self._enter_state("GROOM")
            action.update(state="GROOM", grooming=True, pose="groom")
            return action

        # Priority 5: thermoregulate (Moonwalker backward-walking readout covers
        # both the heat and cold thermosensory routes -- see heaven/brain.py).
        # Which way to move is decided from the actual temperature, not the
        # brain: MDN signals "move away from here", and only the world knows
        # which direction that is away from discomfort. Entry additionally
        # requires the temperature to actually be out of the comfortable band,
        # for the same reason feeding requires fruit contact.
        not_comfortable = not (W.COMFORT_TEMP_C[0] <= snapshot["temperature_c"] <= W.COMFORT_TEMP_C[1])
        if self.state == "THERMOREGULATE":
            if not_comfortable and self.timer < MAX_THERMO_S:
                action.update(pose="retreat", target_pos=self._thermal_target(snapshot))
                return action
            self._enter_state("EXPLORE")
        if self._enter("MDN", r, dt, debounce=0.4) and not_comfortable:  # MDN is only 4 cells: a longer debounce
            self._enter_state("THERMOREGULATE")
            action.update(state="THERMOREGULATE", pose="retreat", target_pos=self._thermal_target(snapshot))
            return action

        # Priority 6/7: bask when quiet and warm enough, else explore/wander.
        comfortable = W.COMFORT_TEMP_C[0] <= snapshot["temperature_c"] <= W.COMFORT_TEMP_C[1] + 4
        quiet = fast["motor"] < QUIET_MOTOR_HZ
        if comfortable and quiet and self.timer > 1.5:
            self._enter_state("BASK")
            belly_up = self.has_mated or snapshot["satiety"] > 0.7
            action.update(state="BASK", pose="bask_belly_up" if belly_up else "bask", target_pos=W.SUNBEAM_POS)
            return action
        self._enter_state("EXPLORE")
        action.update(state="EXPLORE", pose="walk", target_pos=self._wander(snapshot, dt))
        return action

    # -- helpers -----------------------------------------------------------
    def _enter_state(self, state, subphase=None):
        # A subphase change resets the timer too, not just a state change --
        # courtship stays in state "COURT" across FOLLOW/TAP/SING/ATTEMPT/
        # MATE/DISMOUNT, so without this each subphase's own duration check
        # (TAP_S, SING_TIMEOUT_S, ATTEMPT_S, MATE_S) would read a timer still
        # counting from when COURT was first entered, not from this subphase.
        if state != self.state or subphase != self.subphase:
            self.timer = 0.0
        self.state = state
        self.subphase = subphase

    def _wander(self, snapshot, dt):
        # Once a female has arrived, pursuing her takes priority over the
        # fruit outright (checked before the hunger logic below), rather than
        # interleaving the two: this session found a fly that happened to be
        # hovering near the fruit boundary could get stuck cycling its wait
        # timer there for the rest of the film instead of ever reaching her.
        # A real fly that's found a mate can skip a snack.
        if snapshot["female_present"] and not self.has_mated:
            # Always her *current* position, not a stale waypoint: this closes
            # the distance continuously, all the way down to real tactile
            # contact (heaven.world.TAP_RANGE), rather than stopping short and
            # relying on her own small random drift to finish the approach.
            # Long-range approach toward a visible female is scripted for the
            # same reason as foraging below: there's no long-range visual/
            # olfactory courtship channel in `heaven.brain`'s sensory set,
            # only the tactile foreleg contact P1 actually gates on.
            return np.array([snapshot["female_x"], snapshot["female_y"]])

        # Long-range foraging toward the fruit is scripted navigation, not a
        # connectome readout -- LB3c is a contact gustatory sense, with no
        # long-range food-odor channel in `heaven.brain`'s sensory set. Once
        # actually at the fruit, MN9 (real, calibrated) decides whether to feed.
        self._fruit_cooldown = max(0.0, self._fruit_cooldown - dt)
        hungry = snapshot["satiety"] < 0.5
        if hungry and snapshot["near_fruit"]:
            # Hold still right at the fruit instead of peeling off toward some
            # older random wander target the instant near_fruit flips true --
            # otherwise the fly never dwells long enough for MN9's entry
            # debounce (heaven.behaviour.DEBOUNCE_S) to actually register.
            # But only up to MAX_FRUIT_WAIT_S: MN9 is only 2 cells, and this
            # session found it can stay silent under nominal 20 mV drive for
            # an unpredictable stretch depending on the rest of the network's
            # state -- an unconditional hold could freeze the fly indefinitely.
            self._fruit_wait += dt
            if self._fruit_wait < MAX_FRUIT_WAIT_S:
                return None
            # Gave up this attempt: a short cooldown (and a real move-away
            # target, below) so hunger doesn't just walk it straight back into
            # the same hold on the very next tick.
            self._fruit_wait = 0.0
            self._fruit_cooldown = 5.0
        elif hungry and not snapshot["near_fruit"] and self._fruit_cooldown <= 0:
            return W.FRUIT_POS
        else:
            self._fruit_wait = 0.0
        pos = np.array([snapshot["fly_x"], snapshot["fly_y"]])
        if np.linalg.norm(pos - self.wander_target) < 0.03 or self.timer == 0.0:
            margin = 0.05
            self.wander_target = self.rng.uniform([margin, margin], [W.CLEARING[0] - margin, W.CLEARING[1] - margin])
        return self.wander_target

    def _away_from(self, snapshot):
        pos = np.array([snapshot["fly_x"], snapshot["fly_y"]])
        leaf = np.array([snapshot["leaf_x"], snapshot["leaf_y"]])
        direction = pos - leaf
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.array([0.0, 1.0])
        return np.clip(pos + direction * 0.3, [0.02, 0.02], [W.CLEARING[0] - 0.02, W.CLEARING[1] - 0.02])

    def _thermal_target(self, snapshot):
        """Too hot: step away from the sunbeam, toward shade. Too cold: step
        into it. The MDN readout only says "move"; the direction is read off
        the actual temperature, since MDN doesn't carry that sign itself."""
        pos = np.array([snapshot["fly_x"], snapshot["fly_y"]])
        too_hot = snapshot["temperature_c"] > (W.COMFORT_TEMP_C[0] + W.COMFORT_TEMP_C[1]) / 2
        if too_hot:
            direction = pos - W.SUNBEAM_POS
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
            return np.clip(pos + direction * 0.15, [0.02, 0.02], [W.CLEARING[0] - 0.02, W.CLEARING[1] - 0.02])
        return W.SUNBEAM_POS.copy()

    def _court_tick(self, action, r, snapshot, dt):
        if not snapshot["female_present"]:
            self._enter_state("EXPLORE")
            return None
        female = np.array([snapshot["female_x"], snapshot["female_y"]])
        pos = np.array([snapshot["fly_x"], snapshot["fly_y"]])
        dist = float(np.linalg.norm(female - pos))
        action["state"] = "COURT"

        if self.subphase == "FOLLOW":
            action.update(subphase="FOLLOW", pose="follow", target_pos=female)
            if dist < FOLLOW_RANGE:
                self._enter_state("COURT", "TAP")
            return action
        if self.subphase == "TAP":
            action.update(subphase="TAP", pose="tap", target_pos=female)
            if self.timer >= TAP_S:
                self._enter_state("COURT", "SING")
            return action
        if self.subphase == "SING":
            action.update(subphase="SING", pose="sing", wing_song=True, target_pos=None)
            # Acceptance hazard rises the longer he sings, so mating reliably
            # happens well before the timeout without being scripted to a fixed time.
            hazard_per_s = min(1.0, snapshot["song_time"] / 10.0)
            if snapshot["song_time"] > 1.0 and self.rng.random() < hazard_per_s * dt:
                self._enter_state("COURT", "ATTEMPT")
            if self.timer >= SING_TIMEOUT_S:
                self._enter_state("EXPLORE")
                return None
            return action
        if self.subphase == "ATTEMPT":
            action.update(subphase="ATTEMPT", pose="mount_approach", target_pos=female)
            if self.timer >= ATTEMPT_S:
                self._enter_state("COURT", "MATE")
            return action
        if self.subphase == "MATE":
            action.update(subphase="MATE", pose="mate", target_pos=None)
            if self.timer >= MATE_S:
                self._enter_state("COURT", "DISMOUNT")
                self.has_mated = True
            return action
        if self.subphase == "DISMOUNT":
            action.update(subphase="DISMOUNT", pose="dismount", target_pos=None)
            if self.timer >= DISMOUNT_S:
                self._enter_state("EXPLORE")
                return None
            return action
        self._enter_state("COURT", "FOLLOW")
        return self._court_tick(action, r, snapshot, dt)
