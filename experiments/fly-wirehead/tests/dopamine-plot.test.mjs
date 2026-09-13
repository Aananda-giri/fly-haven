import test from 'node:test';
import assert from 'node:assert/strict';
import { dopaminePlot } from '../dist/dopamine-plot.js';

test('dopamine plot preserves measured rises and falls without smoothing or fabricated points', () => {
  const rates = [90.6666666667, 89.3333333333, 96, 86.6666666667, 88, 93.3333333333];
  const history = rates.map((pam11_hz, i) => ({ sim_ms: 5000 + i * 50, pam11_hz }));
  const original = structuredClone(history), plot = dopaminePlot(history);
  assert.deepEqual(plot.points.map(point => point.rate), rates);
  assert.deepEqual(plot.points.map(point => point.simMs), history.map(point => point.sim_ms));
  assert.deepEqual(history, original);
  assert.ok(plot.minimum < Math.min(...rates) && plot.maximum > Math.max(...rates));
  assert.ok(plot.maximum - plot.minimum <= 30, 'detail scale makes the real variation visible');
});

test('zero, flat signals, and missing measurements remain honest', () => {
  assert.deepEqual(dopaminePlot(null).points, []);
  const zero = dopaminePlot([{ sim_ms: 50, pam11_hz: 0 }]);
  assert.equal(zero.points[0].rate, 0); assert.equal(zero.minimum, 0);
  const flat = dopaminePlot(Array.from({ length: 20 }, (_, i) => ({ sim_ms: i * 50, pam11_hz: 88 })));
  assert.ok(flat.points.every(point => point.rate === 88));
  assert.ok(flat.maximum > flat.minimum);
  const invalid = dopaminePlot([{ sim_ms: 1, pam11_hz: NaN }, { sim_ms: 2 }, { sim_ms: 3, pam11_hz: -1 }, null]);
  assert.deepEqual(invalid.points, []);
});

test('repeated samples do not move the plot and a restored clock discards the old timeline', () => {
  const history = [{ sim_ms: 100, pam11_hz: 87 }, { sim_ms: 150, pam11_hz: 91 }];
  assert.deepEqual(dopaminePlot([...history, history[1]]), dopaminePlot(history));
  const reset = dopaminePlot([...history, { sim_ms: 50, pam11_hz: 5 }]);
  assert.deepEqual(reset.points, [{ simMs: 50, rate: 5 }]);
});

test('the rolling window includes every retained extreme without clipping', () => {
  const history = Array.from({ length: 125 }, (_, i) => ({ sim_ms: i * 50, pam11_hz: i === 124 ? 240 : i === 122 ? 0 : 90 }));
  const plot = dopaminePlot(history);
  assert.equal(plot.points.length, 120); assert.equal(plot.points[0].simMs, 250);
  assert.equal(plot.minimum, 0); assert.ok(plot.maximum > 240);
});
