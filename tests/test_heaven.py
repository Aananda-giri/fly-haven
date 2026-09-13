"""Fast unit tests using a fake brain: no connectome load, no C++ kernel.

Real end-to-end validation against the actual MaleCNS brain is
`heaven.calibrate` (route thresholds) and the intact-vs-lesion comparison in
`heaven.simulate` (see the plan's verification section). These tests check
only the scripted parts: physiology, priority/hysteresis and reproducibility.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heaven import world as W
from heaven.behaviour import FlyBehavior
from heaven.simulate import simulate

THRESHOLDS = {"MN9": 5.0, "DNg12": 5.0, "GF": 20.0, "MDN": 5.0, "P1": 3.0, "pIP10": 3.0}


class FakeBrain:
    """Currents drive readouts by a fixed, deterministic linear map -- no spikes."""

    class _Stub:
        build = {"model": "fake-linear-brain"}

    def __init__(self, silence=()):
        self.brain = self._Stub()
        self.silenced = set(silence)

    def step(self, frame, currents, ms=50.0):
        def g(name, key, gain):
            return 0.0 if name in self.silenced else currents.get(key, 0.0) * gain

        return {
            "MN9": g("MN9", "sugar", 3.0),
            "DNg12": g("DNg12", "antenna", 2.0),
            "GF": g("GF", "loom", 8.0),
            "P1": g("P1", "female", 0.4),
            "pIP10": g("pIP10", "female", 0.5),
            "vPR6": 0.0,
            "DNa02_L": 2.0,
            "DNa02_R": 2.0,
            "MDN": g("MDN", "heat", 1.0) + g("MDN", "cold", 1.0),
            "motor": 6.0,
            "PAM11": 0.0,
            "network_spikes": 100,
        }

    def describe(self):
        return {"fake": True, "silenced": sorted(self.silenced)}


class PhysiologyTests(unittest.TestCase):
    def test_satiety_ends_feeding_by_itself(self):
        # The MN9 route stays hot the whole time; only the satiety gate should
        # end feeding, exactly as `heaven.behaviour.FlyBehavior` requires.
        world = W.World(seed=1, hunger_start=0.0)
        world.fly.pos = W.FRUIT_POS.copy()
        behavior = FlyBehavior(thresholds=THRESHOLDS, seed=1)
        fed_ticks, stopped_while_hot = 0, False
        for _ in range(2000):
            currents = world.sense(0.05)
            rates = dict.fromkeys(["MN9", "DNg12", "GF", "P1", "pIP10", "MDN", "motor", "DNa02_L", "DNa02_R"], 0.0)
            rates["MN9"] = 20.0  # sugar route pinned on for the whole test
            snapshot = world.snapshot()
            action = behavior.tick(rates, 0.05, snapshot)
            world.act(0.05, action)
            fed_ticks += action["feeding"]
            if snapshot["satiety"] > 0.95 and not action["feeding"]:
                stopped_while_hot = True
                break
        self.assertGreater(fed_ticks, 10, "should have fed for a while first")
        self.assertTrue(stopped_while_hot, "feeding must stop once satiety saturates, even with MN9 still hot")

    def test_temperature_settles_toward_local_ambient(self):
        world = W.World(seed=0)
        world.fly.pos = W.SUNBEAM_POS.copy()
        for _ in range(400):  # 20 s, several times the 4 s time constant
            world.physiology.step(0.05, world.fly.pos, feeding=False, grooming=False, flying=False)
        self.assertAlmostEqual(world.physiology.temperature_c, W.SUN_TEMP_C, delta=0.5)

        world2 = W.World(seed=0)
        world2.fly.pos = np.array([0.0, 0.0])  # far outside the sunbeam radius: full shade
        for _ in range(400):
            world2.physiology.step(0.05, world2.fly.pos, feeding=False, grooming=False, flying=False)
        self.assertAlmostEqual(world2.physiology.temperature_c, W.SHADE_TEMP_C, delta=0.5)


class PriorityHysteresisTests(unittest.TestCase):
    def test_escape_preempts_and_cannot_be_interrupted_mid_hop(self):
        world = W.World(seed=0)
        behavior = FlyBehavior(thresholds=THRESHOLDS, seed=0)
        snapshot = world.snapshot()
        rates = dict.fromkeys(["MN9", "DNg12", "GF", "P1", "pIP10", "MDN", "motor", "DNa02_L", "DNa02_R"], 0.0)
        rates["GF"] = 60.0  # a loom spike, gone after this one tick (attack tau is fast enough to cross threshold in one 50 ms step)
        action = behavior.tick(rates, 0.05, snapshot)
        self.assertEqual(action["state"], "ESCAPE")

        rates["GF"] = 0.0
        action = behavior.tick(rates, 0.05, snapshot)
        self.assertEqual(action["state"], "ESCAPE", "a brief GF burst must still hold the flight hop")

        for _ in range(30):  # past ESCAPE_HOP_S, with the smoothed GF value long since decayed below threshold
            action = behavior.tick(rates, 0.05, snapshot)
        self.assertEqual(action["state"], "EXPLORE")

    def test_feed_beats_groom_when_both_are_hot(self):
        world = W.World(seed=0)
        world.fly.pos = W.FRUIT_POS.copy()  # feeding also requires actually being at the fruit
        behavior = FlyBehavior(thresholds=THRESHOLDS, seed=0)
        snapshot = world.snapshot()
        rates = dict.fromkeys(["MN9", "DNg12", "GF", "P1", "pIP10", "MDN", "motor", "DNa02_L", "DNa02_R"], 0.0)
        rates["MN9"] = 20.0
        rates["DNg12"] = 20.0
        for _ in range(10):  # past the entry debounce (heaven.behaviour.DEBOUNCE_S)
            action = behavior.tick(rates, 0.05, snapshot)
        self.assertEqual(action["state"], "FEED")


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_gives_identical_timeline(self, tmp_dir=None):
        import tempfile

        kwargs = dict(seconds=10.0, seed=7, hunger_start=0.4, verbose=False, thresholds=THRESHOLDS, brain_factory=FakeBrain)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            t1, _ = simulate(out=a, **kwargs)
            t2, _ = simulate(out=b, **kwargs)
        self.assertTrue(t1.equals(t2))


if __name__ == "__main__":
    unittest.main()
