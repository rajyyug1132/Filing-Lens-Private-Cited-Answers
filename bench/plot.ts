import type { Bin } from '../lib/calibrate';

// Static SVGs for the README and the deck; <title> gives native hover values.
// Colours: reference palette slots 1-2 (all-pairs validated); text in ink tokens.
const SERIES = ['#2a78d6', '#eb6834'];
const INK = '#0b0b0b', INK2 = '#52514e', MUTED = '#8a8984', GRID = '#e7e6e1', SURFACE = '#fcfcfb';

function frame(W: number, H: number, title: string, subtitle: string) {
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="Inter, system-ui, sans-serif">` +
    `<rect width="${W}" height="${H}" fill="${SURFACE}"/>` +
    `<text x="64" y="26" font-size="17" font-weight="600" fill="${INK}">${title}</text>` +
    `<text x="64" y="45" font-size="12.5" fill="${INK2}">${subtitle}</text>`
  );
}

// Reliability diagram: x = predicted p(supported), y = observed fraction supported, per 10% bin.
export function renderReliabilitySvg(series: { name: string; ece: number; bins: Bin[] }[], nLabel: string): string {
  const W = 720, H = 520, m = { l: 64, r: 24, t: 60, b: 110 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const X = (v: number) => m.l + v * pw;
  const Y = (v: number) => m.t + (1 - v) * ph;
  let s = frame(W, H, 'Reliability of the ANSWER/ABSTAIN confidence', `FinanceBench, ${nLabel}, out-of-fold (5-fold grouped by company). Dashed diagonal = perfect calibration.`);
  for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
    s += `<line x1="${X(0)}" x2="${X(1)}" y1="${Y(t)}" y2="${Y(t)}" stroke="${GRID}"/>`;
    s += `<text x="${X(0) - 8}" y="${Y(t) + 4}" font-size="11.5" text-anchor="end" fill="${INK2}">${t.toFixed(1)}</text>`;
    s += `<text x="${X(t)}" y="${Y(0) + 18}" font-size="11.5" text-anchor="middle" fill="${INK2}">${t.toFixed(1)}</text>`;
  }
  s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="${MUTED}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
  s += `<text x="${X(0.5)}" y="${Y(0) + 38}" font-size="12.5" text-anchor="middle" fill="${INK}">Predicted confidence p(supported)</text>`;
  s += `<text transform="translate(18 ${Y(0.5)}) rotate(-90)" font-size="12.5" text-anchor="middle" fill="${INK}">Observed fraction supported</text>`;
  series.forEach((r, k) => {
    const pts = r.bins.filter((b) => b.count > 0);
    s += `<polyline fill="none" stroke="${SERIES[k]}" stroke-width="2" stroke-linejoin="round" points="${pts.map((b) => `${X(b.meanConf).toFixed(1)},${Y(b.fracPos).toFixed(1)}`).join(' ')}"/>`;
    for (const b of pts)
      s += `<circle cx="${X(b.meanConf).toFixed(1)}" cy="${Y(b.fracPos).toFixed(1)}" r="${(4 + Math.min(6, Math.sqrt(b.count) / 2)).toFixed(1)}" fill="${SERIES[k]}" stroke="${SURFACE}" stroke-width="2"><title>${r.name}\nbin ${b.lo.toFixed(1)}–${b.hi.toFixed(1)}: mean conf ${b.meanConf.toFixed(2)}, observed ${b.fracPos.toFixed(2)}, n=${b.count}</title></circle>`;
  });
  series.forEach((r, k) => {
    const y = H - 44 + k * 20;
    s += `<line x1="${m.l}" x2="${m.l + 22}" y1="${y - 4}" y2="${y - 4}" stroke="${SERIES[k]}" stroke-width="2"/><circle cx="${m.l + 11}" cy="${y - 4}" r="4.5" fill="${SERIES[k]}"/>`;
    s += `<text x="${m.l + 30}" y="${y}" font-size="12.5" fill="${INK}">${r.name}: ECE ${r.ece.toFixed(3)}</text>`;
  });
  return s + '</svg>';
}

// Recall of the gold page vs retrieval depth k, with the recall target and the shipped k.
export function renderRecallSvg(
  recall: Record<number, number>,
  chosenK: number,
  target: number,
  tokens: Record<number, { p95: number }>,
  cappedRecall: Record<number, number> = {},
): string {
  const W = 720, H = 440, m = { l: 64, r: 24, t: 60, b: 90 };
  const ks = Object.keys(recall).map(Number).sort((a, b) => a - b);
  const kMax = ks.at(-1)!;
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const X = (k: number) => m.l + (k / kMax) * pw;
  const Y = (v: number) => m.t + (1 - v) * ph;
  let s = frame(W, H, 'Gold-page recall vs retrieval depth k', `150 FinanceBench answer cases. Shipped k = ${chosenK}; prompt capped at 3k tokens.`);
  for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
    s += `<line x1="${X(0)}" x2="${X(kMax)}" y1="${Y(t)}" y2="${Y(t)}" stroke="${GRID}"/>`;
    s += `<text x="${X(0) - 8}" y="${Y(t) + 4}" font-size="11.5" text-anchor="end" fill="${INK2}">${t.toFixed(1)}</text>`;
  }
  for (const k of ks) s += `<text x="${X(k)}" y="${Y(0) + 18}" font-size="11.5" text-anchor="middle" fill="${INK2}">${k}</text>`;
  s += `<text x="${X(kMax / 2)}" y="${Y(0) + 38}" font-size="12.5" text-anchor="middle" fill="${INK}">k (chunks given to the 3B model)</text>`;
  s += `<text transform="translate(18 ${Y(0.5)}) rotate(-90)" font-size="12.5" text-anchor="middle" fill="${INK}">Recall of gold page</text>`;
  s += `<line x1="${X(0)}" x2="${X(kMax)}" y1="${Y(target)}" y2="${Y(target)}" stroke="${MUTED}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
  s += `<text x="${X(kMax)}" y="${Y(target) - 6}" font-size="11.5" text-anchor="end" fill="${INK2}">target ${target}</text>`;
  s += `<line x1="${X(chosenK)}" x2="${X(chosenK)}" y1="${Y(1)}" y2="${Y(0)}" stroke="${INK}" stroke-width="1"/>`;
  s += `<text x="${X(chosenK) + 6}" y="${Y(1) + 12}" font-size="12" font-weight="600" fill="${INK}">shipped k = ${chosenK}</text>`;
  s += `<polyline fill="none" stroke="${SERIES[0]}" stroke-width="2" stroke-linejoin="round" points="${ks.map((k) => `${X(k).toFixed(1)},${Y(recall[k]).toFixed(1)}`).join(' ')}"/>`;
  const ck = Object.keys(cappedRecall).map(Number).sort((a, b) => a - b);
  if (ck.length) {
    s += `<polyline fill="none" stroke="${SERIES[1]}" stroke-width="2" stroke-dasharray="1 0" points="${ck.map((k) => `${X(k).toFixed(1)},${Y(cappedRecall[k]).toFixed(1)}`).join(' ')}"/>`;
    for (const k of ck) s += `<rect x="${(X(k) - 4.5).toFixed(1)}" y="${(Y(cappedRecall[k]) - 4.5).toFixed(1)}" width="9" height="9" rx="2" fill="${SERIES[1]}" stroke="${SURFACE}" stroke-width="2"><title>k=${k}, capped at 3k tokens: recall ${cappedRecall[k].toFixed(3)}</title></rect>`;
    const ly = H - 14;
    s += `<circle cx="${m.l + 6}" cy="${ly - 4}" r="4.5" fill="${SERIES[0]}"/><text x="${m.l + 16}" y="${ly}" font-size="12" fill="${INK}">recall@k</text>`;
    s += `<rect x="${m.l + 96}" y="${ly - 8.5}" width="9" height="9" rx="2" fill="${SERIES[1]}"/><text x="${m.l + 110}" y="${ly}" font-size="12" fill="${INK}">recall@k with the 3k-token cap (what the 3B sees)</text>`;
  }
  for (const k of ks) {
    const tip = tokens[k] ? `, p95 prompt ${tokens[k].p95} tokens` : '';
    s += `<circle cx="${X(k).toFixed(1)}" cy="${Y(recall[k]).toFixed(1)}" r="5" fill="${SERIES[0]}" stroke="${SURFACE}" stroke-width="2"><title>k=${k}: recall ${recall[k].toFixed(3)}${tip}</title></circle>`;
    if (tokens[k] || k === 1) s += `<text x="${X(k)}" y="${Y(recall[k]) - 12}" font-size="11.5" text-anchor="middle" fill="${INK}">${recall[k].toFixed(2)}</text>`;
    if (tokens[k]) s += `<text x="${X(k)}" y="${Y(0) - 8}" font-size="10.5" text-anchor="middle" fill="${INK2}">p95 ${tokens[k].p95} tok</text>`;
  }
  return s + '</svg>';
}
