import type { Hit } from './retrieve';

export const SYSTEM_PROMPT =
  'You answer questions about a company filing using ONLY the excerpts provided. ' +
  'Every sentence must end with the page citation of the excerpt it came from, written exactly like [p.12]. ' +
  'Copy numbers exactly as written in the excerpt, with units. ' +
  'If the excerpts do not contain the answer, reply exactly: NOT IN DOCUMENT. ' +
  'Answer in at most 3 short sentences.';

export function buildUserPrompt(question: string, hits: Hit[]): string {
  const ctx = hits.map((h) => `[p.${h.chunk.page}] ${h.chunk.text}`).join('\n\n');
  return `Excerpts:\n${ctx}\n\nQuestion: ${question}\nAnswer with [p.N] citations:`;
}

export const REFUSAL = 'NOT IN DOCUMENT';

// Keep the highest-ranked hits whose prompt fits the token budget, estimated
// from characters with a conservative tokens-per-character ratio measured in
// the bench (the browser has no Qwen tokenizer). Always keeps at least one hit.
export function fitToBudget(question: string, hits: Hit[], budget: number, tokPerChar: number): Hit[] {
  const kept: Hit[] = [];
  for (const h of hits) {
    const next = [...kept, h];
    const chars = SYSTEM_PROMPT.length + buildUserPrompt(question, next).length;
    if (kept.length && chars * tokPerChar > budget) break;
    kept.push(h);
  }
  return kept;
}
