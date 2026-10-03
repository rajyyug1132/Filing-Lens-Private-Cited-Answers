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
