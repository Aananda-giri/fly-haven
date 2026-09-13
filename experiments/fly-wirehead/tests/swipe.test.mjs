import test from 'node:test';
import assert from 'node:assert/strict';
import { sampleSwipe, frontRightLegPose, FORELEG_REST } from '../dist/swipe.js';

const distance = (a, b) => Math.hypot(...a.map((value, i) => value - b[i]));

test('the foreleg reaches first, sweeps upward with the screen, and returns to its resting pose', () => {
  assert.deepEqual(frontRightLegPose(0), FORELEG_REST);
  assert.deepEqual(frontRightLegPose(1), FORELEG_REST);
  const ready = frontRightLegPose(.22), middle = frontRightLegPose(.47), end = frontRightLegPose(.72);
  assert.equal(sampleSwipe(.22).screen, 0);
  assert.equal(sampleSwipe(.72).screen, 1);
  assert.ok(ready[3][0] > FORELEG_REST[3][0], 'foot reaches toward the phone');
  assert.ok(middle[3][1] > ready[3][1]); assert.ok(end[3][1] > middle[3][1]);
  const fraction = (middle[2][1] - ready[2][1]) / (end[2][1] - ready[2][1]);
  assert.ok(Math.abs(fraction - sampleSwipe(.47).screen) < 1e-12, 'leg sweep and screen slide use identical easing');
});

test('the swipe keeps the shoulder fixed and preserves every leg segment length throughout the gesture', () => {
  const lengths = FORELEG_REST.slice(1).map((point, i) => distance(point, FORELEG_REST[i]));
  let previous = frontRightLegPose(0);
  for (let i = 0; i <= 1000; i++) {
    const pose = frontRightLegPose(i / 1000);
    assert.deepEqual(pose[0], FORELEG_REST[0]);
    pose.flat().forEach(value => assert.ok(Number.isFinite(value)));
    lengths.forEach((length, j) => assert.ok(Math.abs(distance(pose[j], pose[j + 1]) - length) < 1e-10));
    pose.forEach((point, j) => assert.ok(distance(point, previous[j]) < .03, 'pose remains continuous'));
    previous = pose;
  }
  assert.deepEqual(frontRightLegPose(NaN), FORELEG_REST);
});
