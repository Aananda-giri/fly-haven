import test from 'node:test';
import assert from 'node:assert/strict';
import { VideoFeed, nextClipIndex } from '../dist/video-feed.js';
import { SWIPE_SECONDS, sampleSwipe } from '../dist/swipe.js';

function fixture(t) {
  const draws = [];
  const context = new Proxy({}, { get: (_, name) => (...args) => { if (name === 'drawImage') draws.push(args); } });
  const original = globalThis.document;
  globalThis.document = { createElement: () => ({ width: 0, height: 0, getContext: () => context }) };
  t.after(() => { globalThis.document = original; });
  const video = new EventTarget();
  Object.assign(video, { currentTime: 0, readyState: 0, paused: true, ended: false, seeking: false,
    videoWidth: 360, videoHeight: 640, load() { this.readyState = 0; this.currentTime = 0; },
    pause() { this.paused = true; }, async play() { this.paused = false; }, removeAttribute() {} });
  let now = 0;
  const feed = new VideoFeed({ video, now: () => now });
  feed.clips = ['abcdefghijk', '12345678901'].map(id => ({ id, file: `${id}.mp4`, title: id }));
  const loaded = () => { video.readyState = 4; video.dispatchEvent(new Event('loadeddata')); };
  return { feed, video, loaded, draws, setTime: value => now = value };
}

test('the final local video wraps to the first', () => {
  assert.equal(nextClipIndex(9, 10), 0); assert.equal(nextClipIndex(0, 10), 1);
});
test('each short swipes at three seconds of playback and resets for the next clip', async t => {
  const { feed, video, loaded } = fixture(t);
  feed.select(0); feed.setPaused(false); loaded(); await Promise.resolve();
  video.currentTime = 2.99; feed.render(.5); assert.equal(feed.index, 0);
  video.currentTime = 3; feed.render(.02); assert.equal(feed.index, 1);
  assert.equal(feed.observing, false); assert.equal(video.currentTime, 0);
  loaded(); await Promise.resolve(); video.currentTime = 2.99; feed.render(.5); assert.equal(feed.index, 1);
  video.currentTime = 3.02; feed.render(.02); assert.equal(feed.index, 0);
});
test('pause holds a short at its swipe boundary until playback resumes', async t => {
  const { feed, video, loaded } = fixture(t);
  feed.select(0); feed.setPaused(false); loaded(); await Promise.resolve(); feed.render(.5);
  feed.setPaused(true); video.currentTime = 3; feed.render(.5);
  assert.equal(feed.index, 0);
  feed.setPaused(false); await Promise.resolve(); feed.render(.02); assert.equal(feed.index, 1);
});
test('only fresh decoded playing video can supply neural input', async t => {
  const { feed, video, loaded, setTime } = fixture(t);
  feed.select(0); feed.setPaused(false); assert.equal(feed.observing, false);
  loaded(); await Promise.resolve(); video.currentTime = .1; feed.render(.5);
  assert.equal(feed.observing, true); assert.equal(feed.consumed, 1);
  setTime(2000); assert.equal(feed.observing, false);
  video.currentTime = .2; feed.render(.02); assert.equal(feed.observing, true);
  feed.setPaused(true); assert.equal(video.paused, true); assert.equal(feed.observing, false);
});
test('buffering and seeking immediately stop video observations and automatic reward', async t => {
  const { feed, video, loaded } = fixture(t);
  feed.select(0); feed.setPaused(false); loaded(); await Promise.resolve(); video.currentTime = .1; feed.render(.02);
  assert.equal(feed.observing, true);
  video.dispatchEvent(new Event('waiting')); assert.equal(feed.observing, false);
  video.dispatchEvent(new Event('playing')); video.currentTime = .2; feed.render(.02); assert.equal(feed.observing, true);
  video.seeking = true; assert.equal(feed.observing, false);
  video.seeking = false; video.ended = true; assert.equal(feed.observing, false);
});
test('ended advances the actual media source and manual next preserves pause', async t => {
  const { feed, video, loaded } = fixture(t);
  feed.select(0); feed.setPaused(false); loaded(); await Promise.resolve(); feed.render(.5);
  video.dispatchEvent(new Event('ended')); assert.equal(feed.index, 1); assert.equal(video.src, './media/12345678901.mp4');
  loaded(); await Promise.resolve(); feed.render(SWIPE_SECONDS); feed.setPaused(true);
  assert.equal(feed.next(), true); assert.equal(feed.index, 0); assert.equal(feed.wantsPaused, true);
  assert.equal(feed.next(), false);
});
test('a failed clip stops visual input and can be skipped', async t => {
  const { feed, video, loaded } = fixture(t);
  feed.select(0); feed.setPaused(false); loaded(); await Promise.resolve(); feed.render(.5);
  video.dispatchEvent(new Event('error'));
  assert.equal(feed.observing, false); assert.match(feed.error, /could not be loaded/);
  assert.equal(feed.next(), true); assert.equal(feed.error, '');
});

test('automatic and manual skips share the exact screen and foreleg timeline', async t => {
  const { feed, video, loaded, draws } = fixture(t);
  feed.select(0); feed.setPaused(false); loaded(); await Promise.resolve(); feed.render(.02);
  assert.equal(feed.swipeProgress, 1, 'initial load has no swipe gesture');
  video.currentTime = 3; feed.render(.02);
  assert.equal(feed.index, 1); assert.equal(feed.swipeProgress, 0);
  feed.render(1); assert.equal(feed.swipeProgress, 0, 'loading holds the gesture');
  loaded(); await Promise.resolve(); feed.render(SWIPE_SECONDS * .2);
  assert.equal(sampleSwipe(feed.swipeProgress).screen, 0, 'leg lifts before the phone moves');
  feed.render(SWIPE_SECONDS * .3);
  assert.ok(Math.abs(feed.swipeProgress - .5) < 1e-12);
  const oldFrame = draws.findLast(args => args[0] === feed.previous);
  assert.equal(oldFrame[2], -sampleSwipe(feed.swipeProgress).screen * 640);
  feed.render(SWIPE_SECONDS); assert.equal(feed.swipeProgress, 1);
  assert.equal(feed.next(), true); assert.equal(feed.swipeProgress, 0, 'manual skips use the same gesture');
});

test('pausing, buffering, and failed loading hold both the leg and phone in place', async t => {
  const { feed, video, loaded } = fixture(t);
  feed.select(0); feed.setPaused(false); loaded(); await Promise.resolve(); feed.render(.02);
  feed.next(); loaded(); await Promise.resolve(); feed.render(.3);
  const progress = feed.swipeProgress;
  feed.setPaused(true); feed.render(1); assert.equal(feed.swipeProgress, progress);
  feed.setPaused(false); await Promise.resolve();
  video.dispatchEvent(new Event('waiting')); feed.render(1); assert.equal(feed.swipeProgress, progress);
  video.dispatchEvent(new Event('playing')); feed.render(.05); assert.ok(feed.swipeProgress > progress);
  const resumed = feed.swipeProgress;
  video.dispatchEvent(new Event('error')); feed.render(1); assert.equal(feed.swipeProgress, resumed);
});

test('reduced motion replaces the clip without a phone slide or leg gesture', async t => {
  const { feed, video, loaded } = fixture(t); feed.reducedMotion = true;
  feed.select(0); feed.setPaused(false); loaded(); await Promise.resolve(); feed.render(.02);
  video.currentTime = 3; feed.render(.02); loaded(); await Promise.resolve(); feed.render(.02);
  assert.equal(feed.index, 1); assert.equal(feed.swipeProgress, 1);
  assert.deepEqual(sampleSwipe(feed.swipeProgress), { reach: 0, screen: 1 });
});
