import gateJson from './general-gate.json';
import { contentTerms } from './text';
import type { DocIndex, Retrieval } from './retrieve';

// Document-agnostic gate for non-filing documents. No company-name or year features: only how strongly the
// best page matches (dense cosine, BM25 per query term, fused score) and how many query terms that page contains.
export interface GeneralGateModel { features: string[]; mean: number[]; std: number[]; weights: number[]; bias: number; threshold: number; trainedOn: string }
export const generalGate = gateJson as GeneralGateModel;

export function generalFeatures(index: DocIndex, r: Retrieval): number[] {
  const top = r.hits[0];
  if (!top) return [0, 0, 0, 0];
  const terms = Array.from(new Set(r.queryTerms));
  const pageTerms = new Set(contentTerms(index.chunks.filter((c) => c.page === top.chunk.page).map((c) => c.text).join(' ')));
  const qcovTop = terms.length ? terms.filter((t) => pageTerms.has(t)).length / terms.length : 0;
  return [r.denseSorted[0] ?? 0, r.bm25Top / Math.max(1, terms.length), top.rrf, qcovTop];
}

export function generalProbability(x: number[], m: GeneralGateModel = generalGate): number {
  const z = x.reduce((s, v, i) => s + ((v - m.mean[i]) / (m.std[i] || 1)) * m.weights[i], m.bias);
  return 1 / (1 + Math.exp(-z));
}
