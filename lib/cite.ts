// "Cite or abstain": parse [p.N] markers out of a generated answer and check
// that each one points at a page that was actually in the model's context.

export interface CitationCheck {
  cited: number[]; // unique valid pages, in order of first use
  invalid: number[]; // pages cited that were NOT in the context
  sentences: number;
  citedSentences: number;
  coverage: number; // citedSentences / sentences
  ok: boolean;
}

const CITE_RE = /\[\s*(?:p(?:age|g)?\.?\s*)(\d+(?:\s*[,;–-]\s*(?:p(?:age|g)?\.?\s*)?\d+)*)\s*\]/gi;

export function parseCitations(text: string): number[] {
  const out: number[] = [];
  for (const m of text.matchAll(CITE_RE)) {
    for (const n of m[1].split(/[,;–-]/)) {
      const v = parseInt(n.replace(/\D/g, ''), 10);
      if (Number.isFinite(v)) out.push(v);
    }
  }
  return out;
}

export function splitSentences(text: string): string[] {
  return text
    .replace(/\s+/g, ' ')
    .split(/(?<=[.!?])\s+(?=[A-Z$(\d])/)
    .map((s) => s.trim())
    .filter((s) => s.replace(CITE_RE, '').trim().length > 3);
}

export function checkCitations(answer: string, contextPages: number[], minCoverage = 0.5): CitationCheck {
  const allowed = new Set(contextPages);
  const sents = splitSentences(answer);
  let citedSentences = 0;
  const cited: number[] = [];
  const invalid: number[] = [];
  for (const s of sents) {
    const ps = parseCitations(s);
    const good = ps.filter((p) => allowed.has(p));
    ps.filter((p) => !allowed.has(p)).forEach((p) => invalid.includes(p) || invalid.push(p));
    good.forEach((p) => cited.includes(p) || cited.push(p));
    if (good.length) citedSentences++;
  }
  const coverage = sents.length ? citedSentences / sents.length : 0;
  return { cited, invalid, sentences: sents.length, citedSentences, coverage, ok: cited.length > 0 && invalid.length === 0 && coverage >= minCoverage };
}
