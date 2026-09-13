// One presentation timeline drives both the phone and the right foreleg.
// This choreography does not generate neural activity or choose the next clip.
export const SWIPE_SECONDS = .9;
const clamp = value => Math.max(0, Math.min(1, value));
const smooth = value => { const p = clamp(value); return p * p * (3 - 2 * p); };
const mix = (a, b, p) => a.map((value, i) => value + (b[i] - value) * p);
const subtract = (a, b) => a.map((value, i) => value - b[i]);
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const length = a => Math.sqrt(dot(a, a));

export function sampleSwipe(progress = 1) {
  const p = Number.isFinite(progress) ? clamp(progress) : 1;
  const stroke = clamp((p - .22) / .5);
  return {
    reach: smooth(p / .22) * (1 - smooth((p - .72) / .28)),
    screen: 1 - (1 - stroke) ** 3,
  };
}

export const FORELEG_REST = [[.33, -.17, .3], [.85, -.37, .75], [.92, -.94, .99], [1.16, -.96, 1.07]];
const upperLength = length(subtract(FORELEG_REST[1], FORELEG_REST[0]));
const lowerLength = length(subtract(FORELEG_REST[2], FORELEG_REST[1]));

export function frontRightLegPose(progress) {
  const { reach, screen } = sampleSwipe(progress), [shoulder, restJoint, restAnkle, restTip] = FORELEG_REST;
  if (reach === 0) return FORELEG_REST.map(point => [...point]);
  // Lift forward, sweep upward with the outgoing video, then plant the foot again.
  const ankle = mix(restAnkle, mix([1.42, -.22, .66], [1.05, .77, .6], screen), reach);
  const offset = subtract(ankle, shoulder), distance = length(offset), direction = offset.map(value => value / distance);
  const pole = mix(subtract(restJoint, shoulder), [0, 0, 1], reach), alongPole = dot(pole, direction);
  const bend = pole.map((value, i) => value - direction[i] * alongPole), bendLength = length(bend);
  const along = (upperLength ** 2 - lowerLength ** 2 + distance ** 2) / (2 * distance);
  const height = Math.sqrt(Math.max(0, upperLength ** 2 - along ** 2));
  const joint = shoulder.map((value, i) => value + direction[i] * along + bend[i] / bendLength * height);
  const restFoot = subtract(restTip, restAnkle);
  const foot = mix(restFoot, [.24, .035, -.08], reach), footScale = length(restFoot) / length(foot);
  const tip = ankle.map((value, i) => value + foot[i] * footScale);
  return [[...shoulder], joint, ankle, tip];
}
