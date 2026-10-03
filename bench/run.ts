// Decision-layer bench on FinanceBench (open-source 150, real 10-K/10-Q/8-K PDFs).
//
// Task: given a question and the pages retrieved from ONE uploaded filing,
// decide ANSWER or ABSTAIN.
//   gold "answer"  (y=1): the FinanceBench question asked against its own filing
//                          (gold answer + gold evidence page exist)
//   gold "abstain" (y=0): the same question asked against a DIFFERENT company's 10-K
// 150 + 150 = 1:1 split. Out-of-fold predictions come from 5-fold CV grouped by
// company, so no company is ever in both train and test.
// Also reported: retrieval recall@k of the gold page on the answer cases.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import {
  auroc,
  fitLogistic,
  fitTemperature,
  logit,
  reliability,
  sigmoid,
  type DecisionModel,
} from '../lib/calibrate';
import { chunkPages } from '../lib/chunk';
import { nodeEmbedder } from '../lib/embed-node';
import { hydrate } from '../lib/index-doc';
import { FEATURE_NAMES, features, retrieve, type DocIndex } from '../lib/retrieve';
import { CACHE, FB_DIR, pagesFor } from './extract';
import { renderReliabilitySvg } from './plot';

const K = 4;
const RECALL_KS = [1, 2, 4, 8, 16];
const FOLDS = 5;

type Row = { financebench_id: string; company: string; doc_name: string; question: string; answer: string; question_type: string; evidence: { evidence_page_num: number; doc_name: string }[] };
type Inst = { qid: string; company: string; askedDoc: string; kind: 'own_doc' | 'off_doc'; y: number; goldInContext: boolean; firstGoldRank: number; x: number[]; latDecisionMs: number; latRetrieveMs: number; latQueryEmbedMs: number };

const rows: Row[] = readFileSync(join(FB_DIR, 'data/financebench_open_source.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const embed = await nodeEmbedder();

async function docIndex(doc: string): Promise<DocIndex> {
  const pages = await pagesFor(doc);
  const chunks = chunkPages(pages);
  const vecPath = join(CACHE, 'vec', doc + '.f32');
  mkdirSync(join(CACHE, 'vec'), { recursive: true });
  let vectors: Float32Array[];
  if (existsSync(vecPath)) {
    const buf = new Float32Array(readFileSync(vecPath).buffer.slice(0));
    vectors = chunks.map((_, i) => buf.slice(i * 384, (i + 1) * 384));
  } else {
    vectors = [];
    for (let i = 0; i < chunks.length; i += 64) vectors.push(...(await embed(chunks.slice(i, i + 64).map((c) => c.text))));
    const flat = new Float32Array(vectors.length * 384);
    vectors.forEach((v, i) => flat.set(v, i * 384));
    writeFileSync(vecPath, Buffer.from(flat.buffer));
  }
  return hydrate(chunks, vectors);
}

// Off-document pairing: deterministic, always a different company.
const docsByCompany = new Map<string, string[]>();
for (const r of rows)
  if (r.doc_name.endsWith('_10K')) docsByCompany.set(r.company, Array.from(new Set([...(docsByCompany.get(r.company) ?? []), r.doc_name])).sort());
const companies = Array.from(new Set(rows.map((r) => r.company))).sort();
const tenKCompanies = Array.from(docsByCompany.keys()).sort();
function offDoc(r: Row, i: number): string {
  const others = tenKCompanies.filter((c) => c !== r.company);
  const ds = docsByCompany.get(others[(i * 7) % others.length])!;
  return ds[i % ds.length];
}

const cacheIdx = new Map<string, DocIndex>();
async function getIdx(d: string) {
  if (!cacheIdx.has(d)) {
    if (cacheIdx.size > 6) cacheIdx.delete(cacheIdx.keys().next().value!);
    cacheIdx.set(d, await docIndex(d));
  }
  return cacheIdx.get(d)!;
}

const insts: Inst[] = [];
const genInputs: Record<string, unknown>[] = [];
const t0 = performance.now();
for (const [i, r] of rows.entries()) {
  const gold = new Set(r.evidence.filter((e) => e.doc_name === r.doc_name).map((e) => e.evidence_page_num + 1));
  for (const kind of ['own_doc', 'off_doc'] as const) {
    const doc = kind === 'own_doc' ? r.doc_name : offDoc(r, i);
    const idx = await getIdx(doc);
    const a = performance.now();
    const [qv] = await embed([r.question]);
    const b = performance.now();
    const ret = retrieve(idx, r.question, qv, K);
    const c = performance.now();
    const x = features(idx, ret);
    const d = performance.now();
    const wide = retrieve(idx, r.question, qv, Math.max(...RECALL_KS));
    const firstGoldRank = kind === 'own_doc' ? wide.hits.findIndex((h) => gold.has(h.chunk.page)) : -1;
    const y = kind === 'own_doc' ? 1 : 0;
    insts.push({ qid: r.financebench_id, company: r.company, askedDoc: doc, kind, y, goldInContext: firstGoldRank >= 0 && firstGoldRank < K, firstGoldRank, x, latQueryEmbedMs: b - a, latRetrieveMs: c - b, latDecisionMs: d - c });
    genInputs.push({
      id: `${r.financebench_id}:${kind}`, qid: r.financebench_id, kind, question: r.question, question_type: r.question_type,
      gold_answer: kind === 'own_doc' ? r.answer : null, gold_pages: kind === 'own_doc' ? [...gold] : [], doc,
      hits: ret.hits.map((h) => ({ page: h.chunk.page, text: h.chunk.text })),
    });
  }
  if (i % 25 === 0) console.log(`q${i} ${((performance.now() - t0) / 1000).toFixed(0)}s`);
}

// ---- cross-validated predictions -------------------------------------------
const foldOf = new Map(companies.map((c, i) => [c, i % FOLDS]));
const P = { always: [] as number[], cosine: [] as number[], lrRaw: [] as number[], lrTemp: [] as number[] };
const decide = { always: [] as number[], cosine: [] as number[], lrRaw: [] as number[], lrTemp: [] as number[] };
const order: number[] = [];
const predLat: number[] = [];

function trainModel(train: Inst[]): DecisionModel {
  // hold out ~25% of training companies for the temperature fit
  const cs = Array.from(new Set(train.map((t) => t.company))).sort();
  const calibCs = new Set(cs.filter((_, i) => i % 4 === 3));
  const fit = train.filter((t) => !calibCs.has(t.company));
  const cal = train.filter((t) => calibCs.has(t.company));
  const lr = fitLogistic(fit.map((t) => t.x), fit.map((t) => t.y));
  const T = fitTemperature(cal.map((t) => logit(lr, t.x)), cal.map((t) => t.y));
  return { featureNames: [...FEATURE_NAMES], ...lr, temperature: T, threshold: 0.5, trainedOn: '' };
}

for (let f = 0; f < FOLDS; f++) {
  const test = insts.map((t, i) => [t, i] as const).filter(([t]) => foldOf.get(t.company) === f);
  const train = insts.filter((t) => foldOf.get(t.company) !== f);
  const m = trainModel(train);
  // heuristic baseline: raw top-1 cosine, threshold tuned for train accuracy
  let bestTau = 0.5, bestAcc = -1;
  for (let tau = 0; tau <= 1; tau += 0.005) {
    const acc = train.filter((t) => (t.x[0] >= tau ? 1 : 0) === t.y).length / train.length;
    if (acc > bestAcc) [bestAcc, bestTau] = [acc, tau];
  }
  for (const [t, i] of test) {
    order.push(i);
    P.always.push(1);
    decide.always.push(1);
    P.cosine.push(Math.min(1, Math.max(0, t.x[0])));
    decide.cosine.push(t.x[0] >= bestTau ? 1 : 0);
    const z = logit(m, t.x);
    P.lrRaw.push(sigmoid(z));
    decide.lrRaw.push(sigmoid(z) >= 0.5 ? 1 : 0);
    const s = performance.now();
    const p = sigmoid(logit(m, t.x) / m.temperature);
    predLat.push(performance.now() - s);
    P.lrTemp.push(p);
    decide.lrTemp.push(p >= m.threshold ? 1 : 0);
  }
}
const Y = order.map((i) => insts[i].y);
const kinds = order.map((i) => insts[i].kind);

const pct = (v: number[], q: number) => [...v].sort((a, b) => a - b)[Math.min(v.length - 1, Math.floor(q * v.length))];
const methods = [
  { key: 'always', name: 'No gate (always answer)' },
  { key: 'cosine', name: 'Retrieval score threshold (uncalibrated)' },
  { key: 'lrRaw', name: 'LR gate, no temperature' },
  { key: 'lrTemp', name: 'Calibrated LR gate (LR + temperature, shipped)' },
] as const;

const featLat = insts.map((t) => t.latDecisionMs);
const retLat = insts.map((t) => t.latRetrieveMs + t.latQueryEmbedMs);
const results = methods.map(({ key, name }) => {
  const p = P[key];
  const d = decide[key];
  const acc = d.filter((v, i) => v === Y[i]).length / Y.length;
  const { ece, bins } = reliability(p, Y);
  const answered = d.filter((v) => v === 1).length;
  const unsupported = d.filter((v, i) => v === 1 && Y[i] === 0).length;
  const abst = d.map((v, i) => (v === 0 ? i : -1)).filter((i) => i >= 0);
  const negs = Y.filter((v) => v === 0).length;
  const abstainPrecision = abst.length ? abst.filter((i) => Y[i] === 0).length / abst.length : NaN;
  const abstainRecall = negs ? abst.filter((i) => Y[i] === 0).length / negs : NaN;
  const slice = (k: string) => {
    const idx = kinds.map((kk, i) => (kk === k ? i : -1)).filter((i) => i >= 0);
    return idx.filter((i) => d[i] === Y[i]).length / idx.length;
  };
  return {
    key,
    name,
    accuracy: acc,
    ece,
    auroc: key === 'always' ? NaN : auroc(p, Y),
    answerRate: answered / Y.length,
    abstainPrecision,
    abstainRecall,
    unsupportedAnswerRate: unsupported / Y.length,
    accOwnDoc: slice('own_doc'),
    accOffDoc: slice('off_doc'),
    bins,
  };
});

const n = Y.length;
const pos = Y.filter((v) => v === 1).length;
const summary = {
  dataset: 'FinanceBench open-source (patronus-ai/financebench), 150 questions, 84 filings',
  instances: n,
  positives: pos,
  recallAtK: Object.fromEntries(RECALL_KS.map((k) => {
    const own = insts.filter((t) => t.kind === 'own_doc');
    return [k, own.filter((t) => t.firstGoldRank >= 0 && t.firstGoldRank < k).length / own.length];
  })),
  goldInContextAt4OnAnswerCases: insts.filter((t) => t.kind === 'own_doc' && t.goldInContext).length,
  k: K,
  folds: FOLDS,
  latencyMs: {
    queryEmbedPlusRetrieve: { p50: pct(retLat, 0.5), p95: pct(retLat, 0.95) },
    featureExtraction: { p50: pct(featLat, 0.5), p95: pct(featLat, 0.95) },
    classifierPredict: { p50: pct(predLat, 0.5), p95: pct(predLat, 0.95) },
  },
  machine: `Node ${process.version}, ${(await import('node:os')).cpus().length} vCPU container, no GPU`,
  results: results.map(({ bins, ...r }) => r),
};
mkdirSync('bench/results', { recursive: true });
writeFileSync('bench/results/results.json', JSON.stringify({ ...summary, reliabilityBins: Object.fromEntries(results.map((r) => [r.key, r.bins])) }, null, 2));
writeFileSync('bench/results/gen_inputs.jsonl', genInputs.map((g, i) => JSON.stringify({ ...g, p_gate_oof: P.lrTemp[order.indexOf(i)] })).join('\n') + '\n');
writeFileSync('bench/results/instances.jsonl', insts.map((t) => JSON.stringify({ ...t, x: t.x.map((v) => +v.toFixed(4)) })).join('\n'));
writeFileSync('bench/results/reliability.svg', renderReliabilitySvg(results.filter((r) => r.key !== 'always'), Y.length));

// ---- final shipped model: train on everything ------------------------------
const finalModel = trainModel(insts);
finalModel.trainedOn = `FinanceBench open-source: ${pos} own-filing + ${n - pos} other-company-10-K questions, k=${K}`;
writeFileSync('lib/decision-model.json', JSON.stringify(finalModel, null, 2));

const f = (v: number) => (Number.isNaN(v) ? '—' : v.toFixed(3));
const ms = (v: number) => v.toFixed(2);
const L = summary.latencyMs;
let md = `| Gate | Accuracy | ECE ↓ | AUROC | Abstain precision | Abstain recall | Answer rate | Decision latency p50 | Marginal cost / query |\n|---|---|---|---|---|---|---|---|---|\n`;
for (const r of results) {
  const lat = r.key === 'always' ? '0 ms' : r.key === 'cosine' ? '<0.01 ms' : `${ms(L.featureExtraction.p50 + L.classifierPredict.p50)} ms`;
  md += `| ${r.name} | ${f(r.accuracy)} | ${f(r.ece)} | ${f(r.auroc)} | ${f(r.abstainPrecision)} | ${f(r.abstainRecall)} | ${f(r.answerRate)} | ${lat} | $0 (on-device) |\n`;
}
md += `| LLM-router (3B self-check) | pending (Kaggle) | | | | | | | |\n| Jev | [ASK: what is Jev + can it run locally] | | | | | | | |\n`;
md += `\nRetrieval recall of the gold page on the ${pos} answer cases: ` + RECALL_KS.map((k) => `@${k} ${(summary.recallAtK[k] as number).toFixed(3)}`).join(' · ') + '\n';
md += `\nn=${n} instances (${pos} gold-answer, ${n - pos} gold-abstain), ${FOLDS}-fold CV grouped by company, top-k=${K}. ` +
  `Shared retrieval cost per question (query embed + hybrid search): p50 ${ms(L.queryEmbedPlusRetrieve.p50)} ms, p95 ${ms(L.queryEmbedPlusRetrieve.p95)} ms on ${summary.machine}.\n`;
writeFileSync('bench/results/table.md', md);
console.log(md);
console.log(JSON.stringify({ ...summary, results: undefined }, null, 1));
