// Decision-layer bench on FinanceBench (open-source 150, real 10-K/10-Q/8-K PDFs).
//
// Task: given a question and the pages retrieved from ONE uploaded filing,
// decide ANSWER or ABSTAIN. Three instance sets, never merged in reporting:
//   gold "answer"        (y=1): the question asked against its own filing
//   Easy gold "abstain"  (y=0): the same question asked against a DIFFERENT company's 10-K
//   Hard gold "abstain"  (y=0): the question asked against its OWN filing with the gold
//                                evidence pages removed from the index
// One gate is trained on all three with 5-fold CV grouped by company (no company in both
// train and test). It is scored separately on Easy (answer + easy) and Hard (answer + hard).
//
// Retrieval: recall of the gold page at k, and the prompt size (Qwen tokenizer) at k. The
// prompt is capped at 3k tokens: lowest-ranked chunks are dropped until it fits (same rule as
// the app, lib/prompt.ts fitToBudget). The shipped k is the smallest k in K_CANDIDATES whose
// recall AFTER the cap is >= 0.6. Training uses class weights so the gate's prior is 50/50.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { getTokenizer } from '@cyberlangke/tokkit-qwen';
import { auroc, balancedWeights, fitLogistic, fitTemperature, logit, reliability, sigmoid, type DecisionModel } from '../lib/calibrate';
import { chunkPages } from '../lib/chunk';
import { nodeEmbedder } from '../lib/embed-node';
import { hydrate } from '../lib/index-doc';
import { formatMessages, type PredictorSpec } from '../lib/dspy-chat';
import { buildUserPrompt, fitToBudget, SYSTEM_PROMPT } from '../lib/prompt';
import { selectSnippets } from '../lib/snippets';
import { BASE_FEATURES, FEATURE_NAMES, features, retrieve, type DocIndex } from '../lib/retrieve';
import { CACHE, FB_DIR, pagesFor } from './extract';
import { renderRecallSvg, renderReliabilitySvg } from './plot';

const RECALL_KS = [1, 2, 4, 8, 12, 16];
const K_CANDIDATES = [4, 8, 12];
const RECALL_TARGET = 0.6;
const TOKEN_BUDGET = 3000;
const HARD_TARGET = 0.75;
const FOLDS = 5;

type Row = { financebench_id: string; company: string; doc_name: string; question: string; answer: string; question_type: string; evidence: { evidence_page_num: number; doc_name: string }[] };
type Kind = 'own_doc' | 'off_doc' | 'gold_removed';
type Inst = { qid: string; company: string; askedDoc: string; kind: Kind; y: number; x: number[]; latDecisionMs: number; latRetrieveMs: number; latQueryEmbedMs: number };

const rows: Row[] = readFileSync(join(FB_DIR, 'data/financebench_open_source.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const embed = await nodeEmbedder();
const qwen = await getTokenizer('qwen3.5');
const promptTokens = (q: string, hits: ReturnType<typeof retrieve>['hits']) => qwen.encode(SYSTEM_PROMPT).length + qwen.encode(buildUserPrompt(q, hits)).length;

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

// The same filing with the gold pages taken out (Set B / Hard).
function withoutPages(idx: DocIndex, drop: Set<number>): DocIndex {
  const keep = idx.chunks.map((c, i) => [c, i] as const).filter(([c]) => !drop.has(c.page));
  return hydrate(keep.map(([c], j) => ({ ...c, id: j })), keep.map(([, i]) => idx.vectors[i]));
}

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

// Company-grouped split for the DSPy compile on Kaggle: ~40% train, ~20% dev, ~40% test.
const splitOf = new Map(companies.map((c, i) => [c, i % 5 < 2 ? 'train' : i % 5 === 2 ? 'dev' : 'test']));

const cacheIdx = new Map<string, DocIndex>();
async function getIdx(d: string) {
  if (!cacheIdx.has(d)) {
    if (cacheIdx.size > 6) cacheIdx.delete(cacheIdx.keys().next().value!);
    cacheIdx.set(d, await docIndex(d));
  }
  return cacheIdx.get(d)!;
}
const goldOf = (r: Row) => new Set(r.evidence.filter((e) => e.doc_name === r.doc_name).map((e) => e.evidence_page_num + 1));

// ---- 1. retrieval recall@k and prompt size at k -----------------------------
const pct = (v: number[], q: number) => [...v].sort((a, b) => a - b)[Math.min(v.length - 1, Math.floor(q * v.length))];
const wides: { r: Row; hits: ReturnType<typeof retrieve>['hits'] }[] = [];
const firstGoldRank: number[] = [];
const tokensAtK: Record<number, number[]> = Object.fromEntries(K_CANDIDATES.map((k) => [k, []]));
const ratios: number[] = [];
for (const r of rows) {
  const [qv] = await embed([r.question]);
  const idx = await getIdx(r.doc_name);
  const wide = retrieve(idx, r.question, qv, Math.max(...RECALL_KS));
  const gold = goldOf(r);
  wides.push({ r, hits: wide.hits });
  firstGoldRank.push(wide.hits.findIndex((h) => gold.has(h.chunk.page)));
  for (const k of K_CANDIDATES) {
    const hits = wide.hits.slice(0, k);
    const t = promptTokens(r.question, hits);
    tokensAtK[k].push(t);
    ratios.push(t / (SYSTEM_PROMPT.length + buildUserPrompt(r.question, hits).length));
  }
}
// Conservative chars->tokens ratio for the browser (no Qwen tokenizer there): the max seen.
const TOK_PER_CHAR = Number((Math.max(...ratios) + 0.005).toFixed(3));
const recallAtK = Object.fromEntries(RECALL_KS.map((k) => [k, firstGoldRank.filter((g) => g >= 0 && g < k).length / rows.length]));
const tokenStats = Object.fromEntries(K_CANDIDATES.map((k) => [k, { p50: pct(tokensAtK[k], 0.5), p95: pct(tokensAtK[k], 0.95), max: Math.max(...tokensAtK[k]) }]));
const capped = Object.fromEntries(
  K_CANDIDATES.map((k) => {
    const kept = wides.map(({ r, hits }) => fitToBudget(r.question, hits.slice(0, k), TOKEN_BUDGET, TOK_PER_CHAR));
    const toks = kept.map((h, i) => promptTokens(wides[i].r.question, h));
    const rec = kept.filter((h, i) => h.some((x) => goldOf(wides[i].r).has(x.chunk.page))).length / rows.length;
    return [k, { recall: rec, meanChunks: kept.reduce((a, h) => a + h.length, 0) / kept.length, p95: pct(toks, 0.95), max: Math.max(...toks) }];
  }),
);
const fits = K_CANDIDATES.filter((k) => capped[k].recall >= RECALL_TARGET && capped[k].max <= TOKEN_BUDGET);
const K = fits[0] ?? K_CANDIDATES.reduce((a, k) => (capped[k].recall > capped[a].recall ? k : a), K_CANDIDATES[0]);
const kReason = fits.length
  ? `smallest k whose recall after the ${TOKEN_BUDGET}-token cap is >= ${RECALL_TARGET}`
  : `no k reached recall ${RECALL_TARGET} under the ${TOKEN_BUDGET}-token cap; k with the highest capped recall`;
console.log('recall', recallAtK, 'tokens', tokenStats, 'capped', capped, 'tokPerChar', TOK_PER_CHAR, '-> k =', K, `(${kReason})`);

// ---- 2. instances at the chosen k --------------------------------------------
const insts: Inst[] = [];
const genInputs: Record<string, unknown>[] = [];
for (const [i, r] of rows.entries()) {
  const gold = goldOf(r);
  const own = await getIdx(r.doc_name);
  for (const kind of ['own_doc', 'off_doc', 'gold_removed'] as const) {
    const doc = kind === 'off_doc' ? offDoc(r, i) : r.doc_name;
    const idx = kind === 'own_doc' ? own : kind === 'gold_removed' ? withoutPages(own, gold) : await getIdx(doc);
    const a = performance.now();
    const [qv] = await embed([r.question]);
    const b = performance.now();
    const full = retrieve(idx, r.question, qv, K);
    const ret = { ...full, hits: fitToBudget(r.question, full.hits, TOKEN_BUDGET, TOK_PER_CHAR) };
    const c = performance.now();
    const x = features(idx, ret);
    const d = performance.now();
    insts.push({ qid: r.financebench_id, company: r.company, askedDoc: doc, kind, y: kind === 'own_doc' ? 1 : 0, x, latQueryEmbedMs: b - a, latRetrieveMs: c - b, latDecisionMs: d - c });
    const snip = selectSnippets(r.question, ret.hits);
    genInputs.push({
      id: `${r.financebench_id}:${kind}`, qid: r.financebench_id, kind, split: splitOf.get(r.company), company: r.company,
      question: r.question, question_type: r.question_type,
      gold_answer: kind === 'own_doc' ? r.answer : null, gold_pages: kind === 'own_doc' ? [...gold] : [], doc,
      hits: ret.hits.map((h) => ({ page: h.chunk.page, text: h.chunk.text })),
      snippets: snip.text, snippet_pages: Array.from(new Set(snip.items.map((x) => x.page))),
      top1_page: ret.hits[0]?.chunk.page ?? null,
    });
  }
  if (i % 25 === 0) console.log(`q${i}`);
}

// ---- 3. cross-validated gates -------------------------------------------------
const foldOf = new Map(companies.map((c, i) => [c, i % FOLDS]));

function trainModel(train: Inst[], nFeat: number): DecisionModel {
  const cs = Array.from(new Set(train.map((t) => t.company))).sort();
  const calibCs = new Set(cs.filter((_, i) => i % 4 === 3)); // ~25% of training companies fit T
  const fit = train.filter((t) => !calibCs.has(t.company));
  const cal = train.filter((t) => calibCs.has(t.company));
  const fy = fit.map((t) => t.y);
  const cy = cal.map((t) => t.y);
  const lr = fitLogistic(fit.map((t) => t.x.slice(0, nFeat)), fy, 1e-2, 3000, 0.1, balancedWeights(fy));
  const T = fitTemperature(cal.map((t) => logit(lr, t.x.slice(0, nFeat))), cy, balancedWeights(cy));
  return { featureNames: FEATURE_NAMES.slice(0, nFeat), k: K, tokenBudget: TOKEN_BUDGET, tokPerChar: TOK_PER_CHAR, ...lr, temperature: T, threshold: 0.5, trainedOn: '' };
}

type Preds = { always: number[]; cosine: number[]; lrRaw: number[]; lrTemp: number[] };
function crossValidate(nFeat: number) {
  const P: Preds = { always: [], cosine: [], lrRaw: [], lrTemp: [] };
  const D: Preds = { always: [], cosine: [], lrRaw: [], lrTemp: [] };
  const order: number[] = [];
  const predLat: number[] = [];
  for (let f = 0; f < FOLDS; f++) {
    const test = insts.map((t, i) => [t, i] as const).filter(([t]) => foldOf.get(t.company) === f);
    const train = insts.filter((t) => foldOf.get(t.company) !== f);
    const m = trainModel(train, nFeat);
    let bestTau = 0.5, bestAcc = -1;
    for (let tau = 0; tau <= 1; tau += 0.005) {
      const acc = train.filter((t) => (t.x[0] >= tau ? 1 : 0) === t.y).length / train.length;
      if (acc > bestAcc) [bestAcc, bestTau] = [acc, tau];
    }
    for (const [t, i] of test) {
      order.push(i);
      P.always.push(1);
      D.always.push(1);
      P.cosine.push(Math.min(1, Math.max(0, t.x[0])));
      D.cosine.push(t.x[0] >= bestTau ? 1 : 0);
      const z = logit(m, t.x.slice(0, nFeat));
      P.lrRaw.push(sigmoid(z));
      D.lrRaw.push(sigmoid(z) >= 0.5 ? 1 : 0);
      const s = performance.now();
      const p = sigmoid(logit(m, t.x.slice(0, nFeat)) / m.temperature);
      predLat.push(performance.now() - s);
      P.lrTemp.push(p);
      D.lrTemp.push(p >= m.threshold ? 1 : 0);
    }
  }
  return { P, D, order, predLat };
}

function score(cv: ReturnType<typeof crossValidate>, key: keyof Preds, negKind: Kind) {
  const idx = cv.order.map((inst, j) => [insts[inst], j] as const).filter(([t]) => t.kind === 'own_doc' || t.kind === negKind);
  const y = idx.map(([t]) => t.y);
  const p = idx.map(([, j]) => cv.P[key][j]);
  const d = idx.map(([, j]) => cv.D[key][j]);
  const abst = d.map((v, i) => (v === 0 ? i : -1)).filter((i) => i >= 0);
  const negs = y.filter((v) => v === 0).length;
  const { ece, bins } = reliability(p, y);
  return {
    n: y.length,
    accuracy: d.filter((v, i) => v === y[i]).length / y.length,
    ece,
    auroc: key === 'always' ? NaN : auroc(p, y),
    abstainPrecision: abst.length ? abst.filter((i) => y[i] === 0).length / abst.length : NaN,
    abstainRecall: negs ? abst.filter((i) => y[i] === 0).length / negs : NaN,
    answerRate: d.filter((v) => v === 1).length / y.length,
    bins,
  };
}

const METHODS = [
  { key: 'always', name: 'No gate (always answer)' },
  { key: 'cosine', name: 'Retrieval score threshold (uncalibrated)' },
  { key: 'lrRaw', name: 'LR gate, no temperature' },
  { key: 'lrTemp', name: 'Calibrated LR gate (LR + temperature)' },
] as const;

const cvV1 = crossValidate(BASE_FEATURES);
const hardV1 = score(cvV1, 'lrTemp', 'gold_removed').accuracy;
const useV2 = hardV1 < HARD_TARGET;
const cvV2 = useV2 ? crossValidate(FEATURE_NAMES.length) : null;
const shippedCv = cvV2 ?? cvV1;
const shippedFeat = useV2 ? FEATURE_NAMES.length : BASE_FEATURES;

const results = METHODS.map(({ key, name }) => ({
  key,
  name: key === 'lrTemp' ? `${name}, shipped${useV2 ? ' (v2 features)' : ''}` : name,
  easy: score(shippedCv, key, 'off_doc'),
  hard: score(shippedCv, key, 'gold_removed'),
}));
const v1Row = { easy: score(cvV1, 'lrTemp', 'off_doc'), hard: score(cvV1, 'lrTemp', 'gold_removed') };

// Snippet recall (gold page among the sentence snippets) and verifier prompt size with 2 demos.
type G = { kind: string; split: string; question: string; snippets: string; snippet_pages: number[]; gold_pages: number[] };
const G = genInputs as unknown as G[];
const ownG = G.filter((g) => g.kind === 'own_doc');
const snippetRecall = ownG.filter((g) => g.snippet_pages.some((p) => g.gold_pages.includes(p))).length / ownG.length;
let verifierTokens: { p95: number; max: number; demos: string } | null = null;
if (existsSync('app/prompts/verifier.json')) {
  const check = JSON.parse(readFileSync('app/prompts/verifier.json', 'utf8')).predictors.check as PredictorSpec;
  const worst = G.filter((g) => g.split === 'train').sort((a, b) => b.snippets.length - a.snippets.length).slice(0, 2);
  const spec = { ...check, demos: worst.map((g) => ({ question: g.question, snippets: g.snippets, verdict: 'ANSWER', evidence_page: 'p.1' })) };
  const toks = G.filter((g) => g.split === 'test').map((g) =>
    formatMessages(spec, { question: g.question, snippets: g.snippets }).reduce((a, m) => a + qwen.encode(m.content).length + 5, 0),
  );
  verifierTokens = { p95: pct(toks, 0.95), max: Math.max(...toks), demos: '2 longest train snippets (worst case)' };
}
console.log('snippet recall', snippetRecall, 'verifier tokens', verifierTokens);

const featLat = insts.map((t) => t.latDecisionMs);
const retLat = insts.map((t) => t.latRetrieveMs + t.latQueryEmbedMs);
const strip = ({ bins, ...r }: ReturnType<typeof score>) => r;
const summary = {
  dataset: 'FinanceBench open-source (patronus-ai/financebench), 150 questions, 84 filings',
  sets: { answer: 150, easyAbstain: 150, hardAbstain: 150 },
  recallAtK,
  promptTokensAtK: tokenStats,
  cappedAtK: capped,
  snippetRecall,
  verifierPromptTokens: verifierTokens,
  tokenBudget: TOKEN_BUDGET,
  tokPerChar: TOK_PER_CHAR,
  tokenizer: 'Qwen3.5 tokenizer (@cyberlangke/tokkit-qwen) as a proxy for Qwen2.5; exact counts come from the Kaggle run',
  k: K,
  kReason,
  folds: FOLDS,
  hardTarget: HARD_TARGET,
  featureSet: useV2 ? 'v2' : 'v1',
  beforeAfter: useV2
    ? { v1: { easy: strip(v1Row.easy), hard: strip(v1Row.hard) }, v2: { easy: strip(score(cvV2!, 'lrTemp', 'off_doc')), hard: strip(score(cvV2!, 'lrTemp', 'gold_removed')) } }
    : null,
  latencyMs: {
    queryEmbedPlusRetrieve: { p50: pct(retLat, 0.5), p95: pct(retLat, 0.95) },
    featureExtraction: { p50: pct(featLat, 0.5), p95: pct(featLat, 0.95) },
    classifierPredict: { p50: pct(shippedCv.predLat, 0.5), p95: pct(shippedCv.predLat, 0.95) },
  },
  machine: `Node ${process.version}, ${cpus().length} vCPU container, no GPU`,
  results: results.map((r) => ({ key: r.key, name: r.name, easy: strip(r.easy), hard: strip(r.hard) })),
};

mkdirSync('bench/results', { recursive: true });
writeFileSync('bench/results/results.json', JSON.stringify({ ...summary, reliabilityBins: Object.fromEntries(results.map((r) => [r.key, { easy: r.easy.bins, hard: r.hard.bins }])) }, null, 2));
const pGate = new Map(shippedCv.order.map((inst, j) => [inst, shippedCv.P.lrTemp[j]]));
writeFileSync('bench/results/gen_inputs.jsonl', genInputs.map((g, i) => JSON.stringify({ ...g, p_gate_oof: pGate.get(i) })).join('\n') + '\n');
writeFileSync('bench/results/instances.jsonl', insts.map((t) => JSON.stringify({ ...t, x: t.x.map((v) => +v.toFixed(4)) })).join('\n') + '\n');
const shipped = results.find((r) => r.key === 'lrTemp')!;
writeFileSync(
  'bench/results/reliability.svg',
  renderReliabilitySvg(
    [
      { name: 'Calibrated LR gate: Easy set (answer + wrong company)', ece: shipped.easy.ece, bins: shipped.easy.bins },
      { name: 'Calibrated LR gate: Hard set (answer + gold pages removed)', ece: shipped.hard.ece, bins: shipped.hard.bins },
    ],
    `n=${shipped.easy.n} per set`,
  ),
);
writeFileSync('bench/results/recall.svg', renderRecallSvg(recallAtK, K, RECALL_TARGET, tokenStats, Object.fromEntries(K_CANDIDATES.map((k) => [k, capped[k].recall]))));

// ---- 4. shipped model: train on everything ----------------------------------
const finalModel = trainModel(insts, shippedFeat);
finalModel.trainedOn = `FinanceBench open-source: 150 answer + 150 wrong-company + 150 gold-removed, k=${K}, ${useV2 ? 'v2' : 'v1'} features`;
writeFileSync('lib/decision-model.json', JSON.stringify(finalModel, null, 2));

// ---- 5. table -----------------------------------------------------------------
const f = (v: number) => (Number.isNaN(v) ? '—' : v.toFixed(3));
const L = summary.latencyMs;
let md = `| Gate | Easy acc. | Easy ECE ↓ | Easy abstain P / R | Hard acc. | Hard ECE ↓ | Hard abstain P / R | Decision latency p50 |\n|---|---|---|---|---|---|---|---|\n`;
for (const r of results) {
  const lat = r.key === 'always' ? '0 ms' : r.key === 'cosine' ? '<0.01 ms' : `${(L.featureExtraction.p50 + L.classifierPredict.p50).toFixed(2)} ms`;
  md += `| ${r.name} | ${f(r.easy.accuracy)} | ${f(r.easy.ece)} | ${f(r.easy.abstainPrecision)} / ${f(r.easy.abstainRecall)} | ${f(r.hard.accuracy)} | ${f(r.hard.ece)} | ${f(r.hard.abstainPrecision)} / ${f(r.hard.abstainRecall)} | ${lat} |\n`;
}
if (useV2) {
  const ba = summary.beforeAfter!;
  md += `\nHard accuracy with v1 features was ${f(ba.v1.hard.accuracy)} (< ${HARD_TARGET}), so the gate was retrained once with two "does the top passage answer this kind of question" features. ` +
    `Before → after: Easy acc ${f(ba.v1.easy.accuracy)} → ${f(ba.v2.easy.accuracy)}, Easy ECE ${f(ba.v1.easy.ece)} → ${f(ba.v2.easy.ece)}; Hard acc ${f(ba.v1.hard.accuracy)} → ${f(ba.v2.hard.accuracy)}, Hard ECE ${f(ba.v1.hard.ece)} → ${f(ba.v2.hard.ece)}.\n`;
}
md += `\nRecall of the gold page (150 answer cases): ` + RECALL_KS.map((k) => `@${k} ${recallAtK[k].toFixed(3)}`).join(' · ') + '\n';
md += `\nPrompt tokens before the cap (system + question + k chunks, Qwen tokenizer proxy): ` + K_CANDIDATES.map((k) => `k=${k}: p50 ${tokenStats[k].p50}, p95 ${tokenStats[k].p95}, max ${tokenStats[k].max}`).join(' · ') + '\n';
md += `\nWith the ${TOKEN_BUDGET}-token cap (drop lowest-ranked chunks; ${TOK_PER_CHAR} tokens/char estimate): ` + K_CANDIDATES.map((k) => `k=${k}: recall ${capped[k].recall.toFixed(3)}, mean ${capped[k].meanChunks.toFixed(1)} chunks, max ${capped[k].max} tokens`).join(' · ') + '\n';
md += `\n**Shipped k = ${K}** (${kReason}).\n`;
md += `\nSentence snippets (what the verifier reads, ≤2,200 chars): gold page present in ${(snippetRecall * 100).toFixed(1)}% of answer cases` +
  (verifierTokens ? `; verifier prompt with 2 worst-case demos: p95 ${verifierTokens.p95}, max ${verifierTokens.max} tokens.\n` : '.\n');
md += `\nEach set: 150 answer + 150 abstain instances; one gate trained on all 450 with ${FOLDS}-fold CV grouped by company; Easy and Hard scored separately. ` +
  `Retrieval (query embed + hybrid search) p50 ${L.queryEmbedPlusRetrieve.p50.toFixed(2)} ms, p95 ${L.queryEmbedPlusRetrieve.p95.toFixed(2)} ms on ${summary.machine}.\n`;
writeFileSync('bench/results/table.md', md);
console.log(md);
