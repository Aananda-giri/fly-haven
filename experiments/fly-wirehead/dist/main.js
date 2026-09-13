import { createLab } from './scene.js';
import { createPlayback, createFrameClock } from './simulation.js';
import { VideoFeed } from './video-feed.js';
import { BrainClient } from './backend.js';
import { dopaminePlot } from './dopamine-plot.js';

const $ = selector => document.querySelector(selector);
const canvas = $('#scene'), reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
let lab, view = 0, messageTimer, status = null, connection = 'CONNECTING', commandPending = false;
let soundOn = false, sceneFailed = false, sceneRendered = false, connectionError = '';
const playback = createPlayback(), state = playback.state;
const feed = new VideoFeed({ reducedMotion, onChange: updateLabels });
try { lab = createLab(canvas, feed.canvas); }
catch (error) { sceneFailed = true; console.error(error); $('#scene-error').hidden = false; }
const client = new BrainClient({
  reducedMotion,
  captureFrame: () => sceneRendered && feed.observing ? lab?.captureFrame() : null,
  onStatus(data) {
    status = data; connectionError = '';
    connection = data.phase === 'ready' ? (data.paused ? 'PAUSED' : 'CONNECTED') : data.phase.toUpperCase();
    playback.setPaused(sceneFailed || data.phase !== 'ready' || data.paused);
    feed.setPaused(sceneFailed || data.phase !== 'ready' || data.paused || document.hidden);
    if (data.telemetry) {
      const t = data.telemetry;
      state.pam11Hz = t.pam11_hz; state.motorHz = t.motor_hz; state.turnHz = t.turn_hz;
    }
    updateLabels(); drawTelemetry(); syncAudio();
  },
  onError(error) {
    status = null; connection = 'DISCONNECTED'; connectionError = error.message;
    playback.setPaused(true); feed.setPaused(true);
    state.pam11Hz = state.motorHz = state.turnHz = 0;
    updateLabels(); drawTelemetry(); syncAudio();
  }
});
function showMessage(text) {
  clearTimeout(messageTimer); $('#scene-message').textContent = text; $('#scene-message').classList.add('visible');
  messageTimer = setTimeout(() => $('#scene-message').classList.remove('visible'), 2600);
}
function nextShort() { const changed = feed.next(); updateLabels(); return changed; }
async function command(action) {
  if (commandPending) return false;
  commandPending = true;
  try {
    await client.action(action);
    if (action === 'stimulate') showMessage('PAM11 pulse queued · 200 ms');
    if (action === 'save') showMessage('Brain checkpoint requested');
    return true;
  } catch (error) { showMessage(error.message); return false; }
  finally { commandPending = false; }
}
function updateLabels() {
  const t = status?.telemetry, ready = status?.phase === 'ready';
  $('#dopamine-value').textContent = Number.isFinite(t?.pam11_hz) ? t.pam11_hz.toFixed(1) : '—';
  $('#spike-value').textContent = t ? t.total_spikes.toLocaleString() : '—';
  $('#sample-label').textContent = t ? `in ${t.interval_ms} ms of neural time` : 'Waiting for a sample';
  const seconds = Math.floor(state.time), minutes = Math.floor(seconds / 60);
  $('#session-time').textContent = `${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  $('#top-state').textContent = `BRAIN ${connection}`;
  $('#model-detail').textContent = ready ? `${status.model.neurons.toLocaleString()} neurons · ${status.model.retinal_inputs.toLocaleString()} visual inputs` : connectionError || status?.message || 'Loading the local connectome…';
  $('#engine-message').textContent = sceneFailed ? 'Visual input suspended' : feed.error || (status?.paused ? 'Experiment paused' : ready ? '' : connectionError || status?.message || '');
  if (!sceneFailed) { $('#scene-error').hidden = !feed.error; if (feed.error) $('#scene-error').textContent = feed.error; }
  document.body.classList.toggle('paused', Boolean(status?.paused));
  document.body.classList.toggle('disconnected', !ready);
}
function togglePause() {
  if (feed.error && status?.phase === 'ready' && !status.paused) { feed.error = ''; void feed.play(); }
  else void command(status?.paused ? 'resume' : 'pause');
}
function syncAudio() { feed.setMuted(!soundOn || state.paused || document.hidden); }
async function fullscreen() {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('#scene-wrap').requestFullscreen(); }
  catch { showMessage('Fullscreen unavailable in this view'); }
}
// The display is unattended. Keyboard and WebMCP actions remain available without on-screen controls.
document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || /INPUT|TEXTAREA|SELECT|BUTTON|A/.test(document.activeElement?.tagName)) return;
  if (['Space', 'ArrowDown', 'ArrowUp', 'KeyC', 'KeyM', 'KeyF', 'KeyP', 'KeyS'].includes(event.code)) event.preventDefault();
  if (event.repeat) return;
  if (event.code === 'Space') togglePause();
  if (event.code === 'ArrowDown' || event.code === 'ArrowUp') nextShort();
  if (event.code === 'KeyC') { view = (view + 1) % 3; lab?.setView(view); }
  if (event.code === 'KeyM') { soundOn = !soundOn; syncAudio(); showMessage(soundOn ? 'Original audio on' : 'Audio muted'); }
  if (event.code === 'KeyF') void fullscreen();
  if (event.code === 'KeyP') void command('stimulate');
  if (event.code === 'KeyS') void command('save');
});
let pointer = null, wheelAt = 0;
canvas.addEventListener('pointerdown', e => { if (e.button !== 0) return; pointer = { id: e.pointerId, x: e.clientX, y: e.clientY, lastX: e.clientX, lastY: e.clientY, type: e.pointerType }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => { if (!pointer || pointer.id !== e.pointerId) return; const dy = e.clientY - pointer.y, dx = e.clientX - pointer.x; if (pointer.type !== 'touch' || Math.abs(dx) > Math.abs(dy) * .8) lab?.orbit(e.clientX - pointer.lastX, e.clientY - pointer.lastY); pointer.lastX = e.clientX; pointer.lastY = e.clientY; });
canvas.addEventListener('pointerup', e => { if (!pointer || pointer.id !== e.pointerId) return; const dx = e.clientX - pointer.x, dy = e.clientY - pointer.y; if (pointer.type === 'touch' && Math.abs(dy) > 42 && Math.abs(dy) > Math.abs(dx) * 1.2) nextShort(); pointer = null; });
canvas.addEventListener('pointercancel', () => pointer = null);
canvas.addEventListener('lostpointercapture', () => pointer = null);
canvas.addEventListener('wheel', e => { e.preventDefault(); if (Math.abs(e.deltaY) > 8 && performance.now() - wheelAt > 800) { nextShort(); wheelAt = performance.now(); } }, { passive: false });
document.addEventListener('visibilitychange', () => {
  feed.setPaused(sceneFailed || status?.phase !== 'ready' || status?.paused || document.hidden);
  syncAudio();
});
function contextFor(element) {
  const rect = element.getBoundingClientRect(), dpr = Math.min(devicePixelRatio, 2);
  const w = Math.round(rect.width * dpr), h = Math.round(rect.height * dpr);
  if (element.width !== w || element.height !== h) { element.width = w; element.height = h; }
  const ctx = element.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}
function drawTelemetry() {
  const { ctx: c, w, h } = contextFor($('#dopamine-chart'));
  const { points, minimum, maximum } = dopaminePlot(status?.history);
  c.clearRect(0, 0, w, h);
  $('#dopamine-range').textContent = points.length ? `${minimum}–${maximum} Hz · AUTO SCALE` : 'WAITING FOR DATA';
  const left = 32, right = w - 5, top = 7, bottom = h - 6;
  const y = rate => bottom - (rate - minimum) / (maximum - minimum) * (bottom - top);
  c.font = '11px monospace'; c.textAlign = 'left'; c.textBaseline = 'middle';
  for (let i = 0; i <= 2; i++) {
    const rate = minimum + (maximum - minimum) * i / 2;
    c.strokeStyle = '#698f6840'; c.lineWidth = .7;
    c.beginPath(); c.moveTo(left, y(rate)); c.lineTo(right, y(rate)); c.stroke();
    c.fillStyle = '#a3baa3'; c.fillText(`${Number(rate.toFixed(1))}`, 0, y(rate));
  }
  if (!points.length) return;
  const start = points[0].simMs, span = Math.max(1, points.at(-1).simMs - start);
  const x = point => points.length === 1 ? right : left + (point.simMs - start) / span * (right - left);
  c.beginPath(); points.forEach((point, i) => i ? c.lineTo(x(point), y(point.rate)) : c.moveTo(x(point), y(point.rate)));
  c.strokeStyle = '#c4f86a'; c.lineWidth = 2; c.lineJoin = 'round'; c.stroke();
  c.lineTo(x(points.at(-1)), bottom); c.lineTo(x(points[0]), bottom); c.closePath();
  const fill = c.createLinearGradient(0, top, 0, bottom); fill.addColorStop(0, '#c4f86a30'); fill.addColorStop(1, '#c4f86a04');
  c.fillStyle = fill; c.fill();
  c.beginPath(); c.arc(x(points.at(-1)), y(points.at(-1).rate), 3, 0, Math.PI * 2); c.fillStyle = '#e4ffaa'; c.fill();
}
window.addEventListener('resize', drawTelemetry);
const frameClock = createFrameClock();
let hudClock = 0;
function frame(now) {
  if (sceneFailed) return;
  const dt = frameClock(now);
  if (!document.hidden) {
    sceneRendered = false;
    try {
      feed.render(dt);
      playback.setPaused(sceneFailed || status?.phase !== 'ready' || status?.paused || !feed.observing);
      playback.tick(dt); lab.render(state.time, dt, state, feed.swipeProgress);
      sceneRendered = true;
      hudClock += dt; if (hudClock > .2) { hudClock = 0; updateLabels(); }
    } catch (error) {
      console.error(error); sceneFailed = true; sceneRendered = false; playback.setPaused(true); feed.setPaused(true);
      $('#scene-error').textContent = 'The 3D scene stopped. Visual input is suspended. Reload to try again.';
      $('#scene-error').hidden = false;
      updateLabels(); syncAudio(); return;
    }
  }
  requestAnimationFrame(frame);
}
updateLabels(); drawTelemetry(); requestAnimationFrame(frame); client.start(); void feed.load();
window.addEventListener('pagehide', () => { client.stop(); feed.stop(); }, { once: true });

const context = document.modelContext;
if (context?.registerTool) {
  const lifecycle = new AbortController();
  try { Promise.resolve(context.registerTool({ name: 'control_fly_experiment', title: 'Control local fly brain', description: 'Read the measured neural state, pause or resume, change the short, stimulate PAM11, or save the local brain.', inputSchema: { type: 'object', properties: { action: { type: 'string', enum: ['status', 'pause', 'resume', 'next_short', 'stimulate', 'save'] } }, required: ['action'], additionalProperties: false }, annotations: { readOnlyHint: false, untrustedContentHint: false }, async execute(input) {
    if (!input || Object.keys(input).length !== 1 || !['status', 'pause', 'resume', 'next_short', 'stimulate', 'save'].includes(input.action)) throw new TypeError('Invalid action');
    let applied = true; if (input.action === 'next_short') applied = nextShort(); else if (input.action !== 'status') await client.action(input.action);
    return { applied, connection, paused: state.paused, clip: feed.current?.title ?? null, telemetry: status?.telemetry ?? null };
  } }, { signal: lifecycle.signal })).catch(console.warn); } catch (error) { console.warn(error); }
  window.addEventListener('pagehide', () => lifecycle.abort(), { once: true });
}
