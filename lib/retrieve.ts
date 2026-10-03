import { bm25Scores, idf, type Bm25Index } from './bm25';
import type { Chunk } from './chunk';
import { contentTerms, properNouns, years } from './text';

export interface DocIndex {
  chunks: Chunk[];
  vectors: Float32Array[]; // L2-normalised, one per chunk
  bm25: Bm25Index;
  vocab: Set<string>; // every token in the document, for coverage features
}

export interface Hit {
  chunk: Chunk;
  dense: number;
  bm25: number;
  rrf: number;
}

export interface Retrieval {
  hits: Hit[]; // fused top-k, best first
  denseSorted: number[]; // dense cosine of the top chunks, desc
  bm25Top: number;
  bm25TopPage: number;
  denseTopPage: number;
  queryTerms: string[];
  query: string;
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function rankOrder(scores: ArrayLike<number>): number[] {
  return Array.from({ length: scores.length }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
}

// Hybrid retrieval: dense cosine + BM25, fused with reciprocal rank fusion.
export function retrieve(index: DocIndex, query: string, qvec: Float32Array, k = 4, rrfK = 60): Retrieval {
  const queryTerms = contentTerms(query);
  const dense = new Float64Array(index.chunks.length);
  for (let i = 0; i < index.chunks.length; i++) dense[i] = dot(qvec, index.vectors[i]);
  const sparse = bm25Scores(index.bm25, queryTerms);
  const dOrder = rankOrder(dense);
  const sOrder = rankOrder(sparse);
  const rrf = new Float64Array(index.chunks.length);
  dOrder.forEach((ci, r) => (rrf[ci] += 1 / (rrfK + r + 1)));
  sOrder.forEach((ci, r) => {
    if (sparse[ci] > 0) rrf[ci] += 1 / (rrfK + r + 1);
  });
  const fused = rankOrder(rrf).slice(0, k);
  return {
    hits: fused.map((ci) => ({ chunk: index.chunks[ci], dense: dense[ci], bm25: sparse[ci], rrf: rrf[ci] })),
    denseSorted: dOrder.slice(0, 10).map((ci) => dense[ci]),
    bm25Top: sparse[sOrder[0]] ?? 0,
    bm25TopPage: index.chunks[sOrder[0]]?.page ?? -1,
    denseTopPage: index.chunks[dOrder[0]]?.page ?? -1,
    queryTerms,
    query,
  };
}

export const FEATURE_NAMES = [
  'dense_top1',
  'dense_margin',
  'dense_mean5',
  'bm25_top_norm',
  'qcov_context',
  'qcov_doc_idf',
  'rank_agree',
  'year_missing_ctx',
  'n_terms_log',
  'entity_missing_doc',
  // v2: "does the top passage answer this kind of question?"
  'numeric_q_top_numbers',
  'year_in_top1',
] as const;

export const BASE_FEATURES = 10; // v1 gate uses the first 10

const NUMERIC_Q = /\b(how much|how many|amount|revenue|sales|income|earnings|eps|margin|ratio|cash|capex|capital expenditure|expense|cost|debt|liabilit|assets|dividend|growth|percent|%|usd|\$|millions?|billions?|turnover|days)\b/i;

// Decision features. Each one is something the reader of a filing would also
// check: does the best passage match, do the question's words appear in the
// retrieved pages at all, does the document mention them anywhere, and does
// a year asked about show up in the evidence.
export function features(index: DocIndex, r: Retrieval): number[] {
  const terms = Array.from(new Set(r.queryTerms));
  const ctx = new Set(r.hits.flatMap((h) => contentTerms(h.chunk.text)));
  const ctxText = r.hits.map((h) => h.chunk.text).join(' ');
  const d = r.denseSorted;
  const top1 = d[0] ?? 0;
  const mean5 = d.slice(0, 5).reduce((a, b) => a + b, 0) / Math.max(1, Math.min(5, d.length));
  const margin = top1 - (d[4] ?? top1);
  let idfSum = 0;
  let idfInDoc = 0;
  for (const t of terms) {
    const w = idf(index.bm25, t) || Math.log(1 + index.bm25.n);
    idfSum += w;
    if (index.vocab.has(t)) idfInDoc += w;
  }
  const qcovCtx = terms.length ? terms.filter((t) => ctx.has(t)).length / terms.length : 0;
  const qcovDoc = idfSum > 0 ? idfInDoc / idfSum : 0;
  const bm25Norm = idfSum > 0 ? r.bm25Top / (idfSum * 2.2) : 0;
  const ys = years(terms.join(' '));
  const yearMissing = ys.length ? ys.filter((y) => !ctxText.includes(y)).length / ys.length : 0;
  const top = r.hits[0]?.chunk.text ?? '';
  const numericQ = NUMERIC_Q.test(r.query) ? 1 : 0;
  const topWords = Math.max(1, top.split(/\s+/).length);
  const topNumbers = (top.match(/\$?\(?\d[\d,]*\.?\d*\)?%?/g) ?? []).filter((t) => t.replace(/\D/g, '').length >= 2).length;
  const qYears = years(r.query);
  const yearInTop1 = qYears.length ? qYears.filter((y) => top.includes(y)).length / qYears.length : 0.5;
  const ents = properNouns(r.query);
  const entityMissing = ents.length ? ents.filter((e) => !index.vocab.has(e)).length / ents.length : 0;
  return [
    top1,
    margin,
    mean5,
    bm25Norm,
    qcovCtx,
    qcovDoc,
    r.bm25TopPage === r.denseTopPage ? 1 : 0,
    yearMissing,
    Math.log(1 + terms.length),
    entityMissing,
    numericQ * Math.min(1, topNumbers / (0.15 * topWords)),
    yearInTop1,
  ];
}
