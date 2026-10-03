import { tokenize } from './text';

export interface Bm25Index {
  docFreq: Record<string, number>;
  docLens: number[];
  avgLen: number;
  termFreqs: Record<string, number>[];
  n: number;
}

export function buildBm25(texts: string[]): Bm25Index {
  const docFreq: Record<string, number> = {};
  const termFreqs: Record<string, number>[] = [];
  const docLens: number[] = [];
  for (const t of texts) {
    const toks = tokenize(t);
    const tf: Record<string, number> = {};
    for (const w of toks) tf[w] = (tf[w] ?? 0) + 1;
    for (const w of Object.keys(tf)) docFreq[w] = (docFreq[w] ?? 0) + 1;
    termFreqs.push(tf);
    docLens.push(toks.length);
  }
  const avgLen = docLens.reduce((a, b) => a + b, 0) / Math.max(1, docLens.length);
  return { docFreq, docLens, avgLen, termFreqs, n: texts.length };
}

export function idf(index: Bm25Index, term: string): number {
  const df = index.docFreq[term] ?? 0;
  return Math.log(1 + (index.n - df + 0.5) / (df + 0.5));
}

export function bm25Scores(index: Bm25Index, queryTerms: string[], k1 = 1.2, b = 0.75): Float64Array {
  const scores = new Float64Array(index.n);
  const uniq = Array.from(new Set(queryTerms));
  for (const term of uniq) {
    if (!index.docFreq[term]) continue;
    const w = idf(index, term);
    for (let i = 0; i < index.n; i++) {
      const f = index.termFreqs[i][term];
      if (!f) continue;
      const norm = 1 - b + (b * index.docLens[i]) / index.avgLen;
      scores[i] += (w * f * (k1 + 1)) / (f + k1 * norm);
    }
  }
  return scores;
}
