// Actual local media. The same composited canvas supplies the phone and retina.
import { SWIPE_SECONDS, sampleSwipe } from './swipe.js';

export const SHORT_SECONDS = 3;
export function nextClipIndex(index, length) { return length ? (index + 1) % length : 0; }

export class VideoFeed {
  constructor({ reducedMotion = false, onChange = () => {}, video = document.createElement('video'), now = () => performance.now() } = {}) {
    Object.assign(this, { reducedMotion, onChange, video, now });
    this.canvas = document.createElement('canvas'); this.canvas.width = 360; this.canvas.height = 640;
    this.previous = document.createElement('canvas'); this.previous.width = 360; this.previous.height = 640;
    this.ctx = this.canvas.getContext('2d');
    this.clips = []; this.index = 0; this.consumed = 0; this.wantsPaused = true;
    this.error = ''; this.phase = 'loading'; this.lastTime = -1; this.lastFrameAt = -Infinity;
    this.transition = 1; this.hasPrevious = false; this.loading = true; this.counted = false;
    this.advancing = false; this.generation = 0; this.buffering = false;
    video.preload = 'auto'; video.muted = true; video.playsInline = true; video.loop = false;
    video.addEventListener('loadeddata', () => { this.loading = false; this.buffering = false; this.phase = 'ready'; this.onChange(); if (!this.wantsPaused) void this.play(); });
    video.addEventListener('waiting', () => { this.buffering = true; });
    video.addEventListener('playing', () => { this.buffering = false; });
    video.addEventListener('ended', () => { if (!this.wantsPaused) this.next(true); });
    video.addEventListener('error', () => { this.error = 'This local video could not be loaded. Press an arrow key to skip it.'; this.phase = 'error'; this.lastFrameAt = -Infinity; this.onChange(); });
    this.placeholder('INSECT TV', 'Loading local videos…');
  }
  get current() { return this.clips[this.index]; }
  get swipeProgress() { return this.reducedMotion || !this.hasPrevious ? 1 : this.transition; }
  get ready() { return !this.loading && !this.error && this.video.readyState >= 2 && !this.video.seeking; }
  get observing() { return this.ready && !this.buffering && !this.wantsPaused && !this.video.paused && !this.video.ended && this.now() - this.lastFrameAt < 1500; }
  async load() {
    try {
      const response = await fetch('./media/playlist.json', { cache: 'no-store' });
      if (!response.ok) throw new Error('Download the insect playlist first.');
      const data = await response.json();
      this.clips = data.clips.filter(c => /^[\w-]{11}$/.test(c.id) && c.file === `${c.id}.mp4` && typeof c.title === 'string' && c.width === 360 && c.height === 640);
      if (!this.clips.length) throw new Error('No local insect videos are available.');
      this.select(0);
    } catch (error) {
      this.error = error.message; this.phase = 'error'; this.placeholder('NO LOCAL VIDEOS', 'Run the video download script'); this.onChange();
    }
  }
  select(index) {
    this.index = index; this.generation++; this.error = ''; this.loading = true; this.phase = 'loading';
    this.lastTime = -1; this.lastFrameAt = -Infinity; this.counted = false; this.transition = 0; this.buffering = false;
    this.video.src = `./media/${this.current.file}`;
    this.video.load(); this.onChange();
  }
  next(automatic = false) {
    if (!this.clips.length || this.advancing || (!automatic && this.transition < .99 && !this.error)) return false;
    this.advancing = true;
    this.previous.getContext('2d').drawImage(this.canvas, 0, 0); this.hasPrevious = true;
    this.select(nextClipIndex(this.index, this.clips.length)); this.advancing = false;
    return true;
  }
  async play() {
    const generation = this.generation;
    try { await this.video.play(); if (generation === this.generation) { this.error = ''; this.onChange(); } }
    catch (error) { if (generation === this.generation && error.name !== 'AbortError') { this.error = 'Press Space to start the video.'; this.onChange(); } }
  }
  setPaused(value) {
    if (this.wantsPaused === value) return;
    this.wantsPaused = value;
    if (value) this.video.pause(); else { this.error = ''; void this.play(); }
  }
  setMuted(value) { this.video.muted = value; }
  placeholder(title, subtitle) {
    const c = this.ctx; c.fillStyle = '#080d10'; c.fillRect(0, 0, 360, 640);
    c.textAlign = 'center'; c.fillStyle = '#c4f86a'; c.font = '22px monospace'; c.fillText(title, 180, 305);
    c.fillStyle = '#81978b'; c.font = '13px monospace'; c.fillText(subtitle, 180, 336);
  }
  render(dt) {
    if (!this.ready) return;
    const v = this.video, c = this.ctx;
    if (v.currentTime !== this.lastTime) { this.lastTime = v.currentTime; this.lastFrameAt = this.now(); }
    // Use played media time so pause, buffering, and hidden tabs never consume a turn.
    if (this.observing && v.currentTime >= SHORT_SECONDS) { this.next(true); return; }
    if (!this.counted) { this.counted = true; this.consumed++; this.onChange(); }
    if (!this.hasPrevious || this.reducedMotion) this.transition = 1;
    else if (this.observing && !this.buffering) this.transition = Math.min(1, this.transition + dt / SWIPE_SECONDS);
    const p = sampleSwipe(this.swipeProgress).screen;
    c.fillStyle = '#000'; c.fillRect(0, 0, 360, 640);
    if (p < 1) c.drawImage(this.previous, 0, -p * 640);
    c.save(); c.translate(0, (1 - p) * 640);
    c.beginPath(); c.rect(0, 0, 360, 640); c.clip();
    c.drawImage(v, 0, 0, 360, 640); c.restore();
  }
  stop() { this.video.pause(); this.video.removeAttribute('src'); this.video.load(); }
}
