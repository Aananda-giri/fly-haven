// Display measured PAM11 firing in Hz; never synthesize a dopamine concentration.
export function dopaminePlot(history) {
  const points = [];
  for (const entry of Array.isArray(history) ? history : []) {
    if (!Number.isFinite(entry?.sim_ms) || entry.sim_ms < 0 || !Number.isFinite(entry?.pam11_hz) || entry.pam11_hz < 0) continue;
    if (points.length && entry.sim_ms < points.at(-1).simMs) points.length = 0;
    if (entry.sim_ms === points.at(-1)?.simMs) continue;
    points.push({ simMs: entry.sim_ms, rate: entry.pam11_hz });
    if (points.length > 120) points.shift();
  }
  if (!points.length) return { points, minimum: 0, maximum: 100 };
  const rates = points.map(point => point.rate);
  // A labelled detail scale keeps genuine small changes visible without noise.
  const minimum = Math.max(0, Math.floor((Math.min(...rates) - 5) / 5) * 5);
  const maximum = Math.max(minimum + 20, Math.ceil((Math.max(...rates) + 5) / 5) * 5);
  return { points, minimum, maximum };
}
