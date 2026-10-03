import type { Bin } from '../lib/calibrate';

// Reliability diagram: x = predicted p(supported), y = observed fraction
// supported, per 10%-bin. Perfect calibration is the diagonal. Static SVG so
// it drops into the README and the deck; <title> gives native hover values.
const SERIES = ['#2a78d6', '#eb6834', '#1baf7a']; // reference palette slots 1-3 (all-pairs validated)

export function renderReliabilitySvg(results: { name: string; ece: number; bins: Bin[] }[], n: number): string {
  const W = 720, H = 520, m = { l: 64, r: 24, t: 56, b: 120 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const X = (v: number) => m.l + v * pw;
  const Yp = (v: number) => m.t + (1 - v) * ph;
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="Inter, system-ui, sans-serif">`;
  s += `<rect width="${W}" height="${H}" fill="#fcfcfb"/>`;
  s += `<text x="${m.l}" y="26" font-size="17" font-weight="600" fill="#0b0b0b">Reliability of the ANSWER/ABSTAIN confidence</text>`;
  s += `<text x="${m.l}" y="45" font-size="12.5" fill="#52514e">FinanceBench, n=${n} pairs, out-of-fold, 5-fold grouped by company. Dashed diagonal = perfect calibration.</text>`;
  for (const t of ticks) {
    s += `<line x1="${X(0)}" x2="${X(1)}" y1="${Yp(t)}" y2="${Yp(t)}" stroke="#e7e6e1" stroke-width="1"/>`;
    s += `<text x="${X(0) - 8}" y="${Yp(t) + 4}" font-size="11.5" text-anchor="end" fill="#52514e">${t.toFixed(1)}</text>`;
    s += `<text x="${X(t)}" y="${Yp(0) + 18}" font-size="11.5" text-anchor="middle" fill="#52514e">${t.toFixed(1)}</text>`;
  }
  s += `<line x1="${X(0)}" y1="${Yp(0)}" x2="${X(1)}" y2="${Yp(1)}" stroke="#8a8984" stroke-width="1.5" stroke-dasharray="5 4"/>`;
  s += `<text x="${X(0.5)}" y="${Yp(0) + 38}" font-size="12.5" text-anchor="middle" fill="#0b0b0b">Predicted confidence p(supported)</text>`;
  s += `<text transform="translate(18 ${Yp(0.5)}) rotate(-90)" font-size="12.5" text-anchor="middle" fill="#0b0b0b">Observed fraction supported</text>`;
  results.forEach((r, k) => {
    const pts = r.bins.filter((b) => b.count > 0);
    const c = SERIES[k];
    s += `<polyline fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round" points="${pts.map((b) => `${X(b.meanConf).toFixed(1)},${Yp(b.fracPos).toFixed(1)}`).join(' ')}"/>`;
    for (const b of pts) {
      const rad = 4 + Math.min(6, Math.sqrt(b.count) / 2);
      s += `<circle cx="${X(b.meanConf).toFixed(1)}" cy="${Yp(b.fracPos).toFixed(1)}" r="${rad.toFixed(1)}" fill="${c}" stroke="#fcfcfb" stroke-width="2"><title>${r.name}\nbin ${b.lo.toFixed(1)}–${b.hi.toFixed(1)}: mean conf ${b.meanConf.toFixed(2)}, observed ${b.fracPos.toFixed(2)}, n=${b.count}</title></circle>`;
    }
  });
  // legend (identity never colour-alone: name + ECE text in ink)
  results.forEach((r, k) => {
    const y = H - 62 + k * 20;
    s += `<line x1="${m.l}" x2="${m.l + 22}" y1="${y - 4}" y2="${y - 4}" stroke="${SERIES[k]}" stroke-width="2"/><circle cx="${m.l + 11}" cy="${y - 4}" r="4.5" fill="${SERIES[k]}"/>`;
    s += `<text x="${m.l + 30}" y="${y}" font-size="12.5" fill="#0b0b0b">${r.name} — ECE ${r.ece.toFixed(3)}</text>`;
  });
  s += `<text x="${W - m.r}" y="${H - 8}" font-size="10.5" text-anchor="end" fill="#8a8984">Marker area ∝ bin count</text>`;
  return s + '</svg>';
}
