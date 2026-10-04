// Tunes the GENERAL (document-agnostic) gate on the synthetic syllabus fixture: 20 answerable + 20 negative questions.
// Writes lib/general-gate.json and bench/results/general_gate.json. Accuracy is reported in-sample AND 5-fold CV,
// because a threshold tuned on 40 questions and scored on the same 40 overstates how well it generalises.
import { writeFileSync } from 'node:fs';
import { chunkPages } from '../lib/chunk';
import { nodeEmbedder } from '../lib/embed-node';
import { prepare } from '../lib/gate';
import { generalFeatures, generalProbability, type GeneralGateModel } from '../lib/general-gate';
import { hydrate } from '../lib/index-doc';
import { SYLLABUS_NEGATIVES, SYLLABUS_PAGES, SYLLABUS_POSITIVES } from '../tests/fixtures/syllabus';

const embed = await nodeEmbedder();
const chunks = chunkPages(SYLLABUS_PAGES.map((text, i) => ({ page: i + 1, text })));
const index = hydrate(chunks, await embed(chunks.map((c) => c.text)));
const qs = [...SYLLABUS_POSITIVES.map((q) => ({ q, y: 1 })), ...SYLLABUS_NEGATIVES.map((q) => ({ q, y: 0 }))];
const extra = ['Details', 'Introduction'];
const vecs = await embed([...qs.map((x) => x.q), ...extra]);
const X = qs.map((x, i) => generalFeatures(index, prepare(index, x.q, vecs[i])));
const Xextra = extra.map((q, i) => generalFeatures(index, prepare(index, q, vecs[qs.length + i])));
const y = qs.map((x) => x.y);

function fit(X: number[][], y: number[], l2 = 0.05, iters = 4000, lr = 0.3): Omit<GeneralGateModel, 'threshold' | 'trainedOn' | 'features'> {
  const d = X[0].length, n = X.length;
  const mean = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const std = Array.from({ length: d }, (_, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - mean[j]) ** 2, 0) / n) || 1);
  const Z = X.map((r) => r.map((v, j) => (v - mean[j]) / std[j]));
  const pos = y.filter(Boolean).length, sw = y.map((v) => (v ? n / (2 * pos) : n / (2 * (n - pos))));
  const w = new Array(d).fill(0); let b = 0;
  for (let it = 0; it < iters; it++) {
    const gw = new Array(d).fill(0); let gb = 0;
    Z.forEach((z, i) => { const p = 1 / (1 + Math.exp(-(z.reduce((s, v, j) => s + v * w[j], b)))); const e = (p - y[i]) * sw[i]; z.forEach((v, j) => (gw[j] += e * v)); gb += e; });
    for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] / n + l2 * w[j]);
    b -= lr * (gb / n);
  }
  return { mean, std, weights: w, bias: b };
}
const acc = (p: number[], y: number[], t: number) => p.filter((v, i) => (v >= t ? 1 : 0) === y[i]).length / y.length;

const base = fit(X, y);
const pIn0 = X.map((x) => generalProbability(x, { ...base, threshold: 0.5, trainedOn: '', features: [] }));
let threshold = 0.5, best = -1;
for (let t = 0.2; t <= 0.801; t += 0.02) { const a = acc(pIn0, y, t); if (a > best + 1e-9 || (Math.abs(a - best) < 1e-9 && Math.abs(t - 0.5) < Math.abs(threshold - 0.5))) { best = a; threshold = +t.toFixed(2); } }
const model: GeneralGateModel = { features: ['dense_top1', 'bm25_per_term', 'rrf_top', 'qcov_top_page'], ...base, threshold, trainedOn: 'synthetic OS syllabus fixture (tests/fixtures/syllabus.ts): 20 answerable + 20 negative questions' };

// 5-fold CV (stratified): refit and re-pick the threshold inside each fold, score the held-out fifth.
const cvPred: number[] = new Array(y.length);
for (let f = 0; f < 5; f++) {
  const tr = y.map((_, i) => i).filter((i) => i % 5 !== f), te = y.map((_, i) => i).filter((i) => i % 5 === f);
  const m = fit(tr.map((i) => X[i]), tr.map((i) => y[i]));
  const ptr = tr.map((i) => generalProbability(X[i], { ...m, threshold: 0.5, trainedOn: '', features: [] }));
  let ft = 0.5, fb = -1;
  for (let t = 0.2; t <= 0.801; t += 0.02) { const a = acc(ptr, tr.map((i) => y[i]), t); if (a > fb + 1e-9) { fb = a; ft = t; } }
  for (const i of te) cvPred[i] = generalProbability(X[i], { ...m, threshold: ft, trainedOn: '', features: [] }) >= ft ? 1 : 0;
}
const cvAcc = cvPred.filter((v, i) => v === y[i]).length / y.length;
const pIn = X.map((x) => generalProbability(x, model));
const rows = qs.map((x, i) => ({ q: x.q, label: x.y ? 'answerable' : 'negative', p: +pIn[i].toFixed(3), pass: pIn[i] >= threshold }));
const out = {
  threshold, accuracy_in_sample: +acc(pIn, y, threshold).toFixed(3), accuracy_cv5: +cvAcc.toFixed(3),
  answerable_passed: rows.filter((r) => r.label === 'answerable' && r.pass).length, negatives_blocked: rows.filter((r) => r.label === 'negative' && !r.pass).length,
  n: y.length, unlabelled: extra.map((q, i) => ({ q, p: +generalProbability(Xextra[i], model).toFixed(3) })), rows,
};
writeFileSync('lib/general-gate.json', JSON.stringify(model, null, 1) + '\n');
writeFileSync('bench/results/general_gate.json', JSON.stringify(out, null, 1) + '\n');
console.log({ threshold, in_sample: out.accuracy_in_sample, cv5: out.accuracy_cv5, passed: out.answerable_passed, blocked: out.negatives_blocked, unlabelled: out.unlabelled });
console.log('misses:', rows.filter((r) => (r.label === 'answerable') !== r.pass).map((r) => `${r.label}:${r.p} ${r.q}`));
