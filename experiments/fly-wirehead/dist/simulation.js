export function createFrameClock() {
  let previous = null;
  return now => {
    // The first rAF timestamp can precede performance.now() during startup.
    // Use only frame timestamps, and bound gaps after a suspended tab resumes.
    const dt = previous === null ? 0 : Math.max(0, Math.min((now - previous) / 1000, .05));
    previous = now;
    return dt;
  };
}

// Exposure clock only. Video playback belongs to VideoFeed; neural data comes from Python.
export function createPlayback() {
  const state = { time: 0, paused: true, pam11Hz: 0, motorHz: 0, turnHz: 0 };
  function tick(dt) {
    if (!Number.isFinite(dt) || dt < 0) throw new TypeError('Time step must be a finite nonnegative number.');
    if (!state.paused) state.time += Math.min(dt, .1);
    return state;
  }
  function setPaused(value) {
    if (typeof value !== 'boolean') throw new TypeError('paused must be a boolean');
    state.paused = value;
    return value;
  }
  return { state, tick, setPaused };
}
