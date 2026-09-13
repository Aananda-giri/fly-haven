// A visual response to measured motor rates. This never changes the neural model.
export function createMotionResponse() {
  const response = { motor: 0, turn: 0, reward: 0, phase: 0 };
  const positive = n => Number.isFinite(n) ? Math.max(0, n) : 0;
  return {
    step(state, dt) {
      if (state.paused || !Number.isFinite(dt) || dt <= 0) return response;
      dt = Math.min(dt, .05);
      const motor = 1 - Math.exp(-positive(state.motorHz) / 10);
      const turn = Number.isFinite(state.turnHz) ? Math.tanh(state.turnHz / 45) : 0;
      const reward = 1 - Math.exp(-positive(state.pam11Hz) / 20);
      // Retain brief motor bursts long enough to see between measured samples.
      response.motor += (motor - response.motor) * (1 - Math.exp(-dt / (motor > response.motor ? .1 : .7)));
      response.turn += (turn - response.turn) * (1 - Math.exp(-dt / .2));
      response.reward += (reward - response.reward) * (1 - Math.exp(-dt / .18));
      response.phase += dt * (11 + response.motor * 22);
      return response;
    }
  };
}
