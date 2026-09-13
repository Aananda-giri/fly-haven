import test from 'node:test';
import assert from 'node:assert/strict';
import { createMotionResponse } from '../dist/motion.js';

const advance = (motion, state, seconds) => {
  let result;
  for (let i = 0; i < seconds * 60; i++) result = motion.step(state, 1 / 60);
  return { ...result };
};
test('low measured motor rates produce a visible, bounded response without changing measurements', () => {
  const state = { motorHz: 5, turnHz: 0, pam11Hz: 0, paused: false };
  const original = { ...state }, motion = createMotionResponse();
  const low = advance(motion, state, .5);
  assert.ok(low.motor > .3 && low.motor < 1);
  const high = advance(motion, { ...state, motorHz: 30 }, .5);
  assert.ok(high.motor > low.motor && high.motor <= 1);
  assert.equal(low.reward, 0); assert.deepEqual(state, original);
});
test('motor bursts decay smoothly, and pause freezes the visual response', () => {
  const motion = createMotionResponse(), state = { motorHz: 20, turnHz: 40, pam11Hz: 0, paused: false };
  const active = advance(motion, state, .5);
  const release = { ...motion.step({ ...state, motorHz: 0 }, .05) };
  assert.ok(release.motor > 0 && release.motor < active.motor);
  const paused = advance(motion, { ...state, paused: true }, 1);
  assert.deepEqual(paused, release);
  const quiet = advance(motion, { motorHz: 0, turnHz: 0, pam11Hz: 0, paused: false }, 5);
  assert.ok(quiet.motor < .001); assert.ok(Math.abs(quiet.turn) < .001);
});
test('turn direction follows signed measured activity and invalid values stay finite', () => {
  const motion = createMotionResponse();
  const right = advance(motion, { motorHz: 0, turnHz: 100, pam11Hz: 0 }, 1);
  const left = advance(motion, { motorHz: 0, turnHz: -100, pam11Hz: 0 }, 1);
  assert.ok(right.turn > 0 && right.turn <= 1); assert.ok(left.turn < 0 && left.turn >= -1);
  const invalid = advance(motion, { motorHz: NaN, turnHz: Infinity, pam11Hz: undefined }, 1);
  assert.ok(Object.values(invalid).every(Number.isFinite));
});
