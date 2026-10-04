import verifierJson from '../app/prompts/verifier.json';
import { checkCitations, type CitationCheck } from './cite';
import { formatMessages, parseOutput, type PredictorSpec } from './dspy-chat';
import { extractiveAnswer, type Engine } from './llm';
import type { DocType } from './doc-type';
import { decisionModel, prepare, routeGate } from './gate';
import type { DocIndex, Hit } from './retrieve';
import { selectSnippets, type Snippet } from './snippets';

export { decisionModel };
export const verifier = verifierJson as unknown as {
  compiled: boolean;
  model: string;
  optimizer: string;
  predictors: { check: PredictorSpec; answer: PredictorSpec };
};

export interface Timings {
  embedMs: number;
  retrieveMs: number;
  decideMs: number;
  verifyMs: number;
  generateMs: number;
}

export type AbstainReason = 'low_confidence' | 'verifier_abstained' | 'empty_answer' | 'citation_check_failed';

export type Outcome =
  | { kind: 'answer'; text: string; confidence: number; hits: Hit[]; snippets: Snippet[]; citations: CitationCheck; evidencePage: string | null; timings: Timings; engine: string; verified: boolean; threshold: number; forced: boolean; unverified?: boolean }
  | { kind: 'abstain'; reason: AbstainReason; confidence: number; hits: Hit[]; snippets: Snippet[]; draft?: string; timings: Timings; engine?: string; threshold: number; forced: boolean };

// Cascade: retrieve -> feature gate (calibrated LR) -> DSPy verifier on the
// 3B (ANSWER / ABSTAIN + evidence page) -> DSPy cited answerer -> [p.N] check.
// Prompts are rebuilt from app/prompts/verifier.json exactly as DSPy renders them.
export async function ask(
  index: DocIndex,
  question: string,
  embed: (t: string[]) => Promise<Float32Array[]>,
  engine: Engine,
  opts: { threshold?: number; strict?: boolean; docType?: DocType; force?: boolean; onText?: (t: string) => void } = {},
): Promise<Outcome> {
  const t0 = performance.now();
  const [qv] = await embed([question]);
  const t1 = performance.now();
  const r = prepare(index, question, qv);
  const t2 = performance.now();
  const gate = routeGate(index, r, opts.docType ?? 'FILING');
  const confidence = gate.confidence;
  const threshold = opts.threshold ?? gate.threshold;
  const forced = !!opts.force; // "Answer anyway": skip the gate (and the verifier), label the result unverified
  const snip = selectSnippets(question, r.hits);
  const t3 = performance.now();
  const timings: Timings = { embedMs: t1 - t0, retrieveMs: t2 - t1, decideMs: t3 - t2, verifyMs: 0, generateMs: 0 };
  const base = { confidence, hits: r.hits, snippets: snip.items, timings, threshold, forced };
  if (!forced && confidence < threshold) return { kind: 'abstain', reason: 'low_confidence', ...base };

  const pages = snip.items.map((s) => s.page);
  if (!engine.chat) {
    // Forced with no model: show the closest passages themselves (extractive preview), tagged with their pages.
    const text = extractiveAnswer(question, snip.items) || (forced ? snip.items.slice(0, 2).map((s) => `${s.text} [p.${s.page}]`).join(' ') : '');
    if (!text) return { kind: 'abstain', reason: 'empty_answer', ...base, engine: engine.label };
    return { kind: 'answer', text, ...base, citations: checkCitations(text, pages), evidencePage: null, engine: engine.label, verified: false, unverified: forced };
  }

  const inputs = { question, snippets: snip.text };
  // Strict mode adds the DSPy verifier step. Off by default: on the FinanceBench test split it
  // said ANSWER on 2-4 of 55 answerable questions (bench/results/cascade_metrics.json).
  let evidencePage: string | null = null;
  if (opts.strict && !forced) {
    const v = parseOutput(verifier.predictors.check, await engine.chat(formatMessages(verifier.predictors.check, inputs), 40));
    timings.verifyMs = performance.now() - t3;
    if ((v.verdict ?? '').trim().toUpperCase() !== 'ANSWER') return { kind: 'abstain', reason: 'verifier_abstained', ...base, engine: engine.label };
    evidencePage = v.evidence_page ?? null;
  }

  const t5 = performance.now();
  const raw = await engine.chat(formatMessages(verifier.predictors.answer, inputs), 200, opts.onText);
  timings.generateMs = performance.now() - t5;
  const text = (parseOutput(verifier.predictors.answer, raw).answer ?? '').trim();
  if (!text) return { kind: 'abstain', reason: 'empty_answer', ...base, draft: raw, engine: engine.label };
  const citations = checkCitations(text, pages);
  if (!citations.ok) return { kind: 'abstain', reason: 'citation_check_failed', ...base, draft: text, engine: engine.label };
  return { kind: 'answer', text, ...base, citations, evidencePage, engine: engine.label, verified: !!opts.strict && !forced, unverified: forced };
}
