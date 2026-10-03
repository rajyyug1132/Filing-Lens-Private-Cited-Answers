import { predict, type DecisionModel } from './calibrate';
import { checkCitations, type CitationCheck } from './cite';
import modelJson from './decision-model.json';
import type { Engine } from './llm';
import { buildUserPrompt, REFUSAL, SYSTEM_PROMPT } from './prompt';
import { features, retrieve, type DocIndex, type Hit } from './retrieve';

export const decisionModel = modelJson as DecisionModel;

export type Outcome =
  | { kind: 'answer'; text: string; confidence: number; hits: Hit[]; citations: CitationCheck; timings: Timings; engine: string }
  | { kind: 'abstain'; reason: 'low_confidence' | 'model_refused' | 'citation_check_failed'; confidence: number; hits: Hit[]; draft?: string; timings: Timings; engine?: string };

export interface Timings {
  embedMs: number;
  retrieveMs: number;
  decideMs: number;
  generateMs: number;
}

// ask -> retrieve -> decide -> (3B answer -> cite check) | ABSTAIN
export async function ask(
  index: DocIndex,
  question: string,
  embed: (t: string[]) => Promise<Float32Array[]>,
  engine: Engine,
  opts: { threshold?: number; onText?: (t: string) => void } = {},
): Promise<Outcome> {
  const t0 = performance.now();
  const [qv] = await embed([question]);
  const t1 = performance.now();
  const r = retrieve(index, question, qv, 4);
  const t2 = performance.now();
  const confidence = predict(decisionModel, features(index, r));
  const t3 = performance.now();
  const threshold = opts.threshold ?? decisionModel.threshold;
  const timings: Timings = { embedMs: t1 - t0, retrieveMs: t2 - t1, decideMs: t3 - t2, generateMs: 0 };
  if (confidence < threshold) return { kind: 'abstain', reason: 'low_confidence', confidence, hits: r.hits, timings };

  const text = (await engine.generate(SYSTEM_PROMPT, buildUserPrompt(question, r.hits), r.hits, question, opts.onText)).trim();
  timings.generateMs = performance.now() - t3;
  if (!text || text.toUpperCase().includes(REFUSAL))
    return { kind: 'abstain', reason: 'model_refused', confidence, hits: r.hits, draft: text, timings, engine: engine.label };
  const citations = checkCitations(text, r.hits.map((h) => h.chunk.page));
  if (!citations.ok)
    return { kind: 'abstain', reason: 'citation_check_failed', confidence, hits: r.hits, draft: text, timings, engine: engine.label };
  return { kind: 'answer', text, confidence, hits: r.hits, citations, timings, engine: engine.label };
}
