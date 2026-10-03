import { splitSentences } from './cite';
import type { Hit } from './retrieve';
import { contentTerms } from './text';

export interface Snippet {
  page: number;
  text: string;
}

export const SNIPPET_CHAR_BUDGET = 2200; // bench sweep: verifier prompt with 2 worst-case demos max 2,674 Qwen tokens (< 3k)
const MAX_SNIPPET_CHARS = 300;
const NUMERIC_Q = /\b(how much|how many|amount|revenue|income|margin|ratio|cash|capex|expenditure|expense|debt|eps|dividend|%|usd|\$|millions?|billions?)\b/i;

// Long "sentences" (flattened tables have no full stops) are cut into
// word-boundary windows so a snippet stays a sentence-sized unit.
function pieces(text: string): string[] {
  const out: string[] = [];
  for (const s of splitSentences(text)) {
    if (s.length <= MAX_SNIPPET_CHARS) {
      out.push(s);
      continue;
    }
    let cur = '';
    for (const w of s.split(' ')) {
      if (cur && cur.length + 1 + w.length > MAX_SNIPPET_CHARS) {
        out.push(cur);
        cur = w;
      } else cur = cur ? `${cur} ${w}` : w;
    }
    if (cur) out.push(cur);
  }
  return out;
}

// Pick the sentence snippets the verifier and the answerer read. Deterministic:
// the bench exports these exact strings for the Kaggle notebook, and the browser
// rebuilds them with this same function.
export function selectSnippets(question: string, hits: Hit[], charBudget = SNIPPET_CHAR_BUDGET): { items: Snippet[]; text: string } {
  const q = new Set(contentTerms(question));
  const numeric = NUMERIC_Q.test(question);
  const cands = hits.flatMap((h, rank) =>
    pieces(h.chunk.text).map((text, pos) => {
      const terms = contentTerms(text);
      const overlap = terms.filter((t) => q.has(t)).length;
      const hasNum = /\d[\d,]*\.?\d*/.test(text) ? 1 : 0;
      return { page: h.chunk.page, text, rank, pos, score: overlap / Math.sqrt(1 + terms.length) + 0.3 / (1 + rank) + (numeric ? 0.15 * hasNum : 0) };
    }),
  );
  const chosen: typeof cands = [];
  let used = 0;
  for (const c of [...cands].sort((a, b) => b.score - a.score || a.rank - b.rank || a.pos - b.pos)) {
    const cost = c.text.length + `[p.${c.page}] `.length + 1;
    if (used + cost > charBudget) continue;
    chosen.push(c);
    used += cost;
  }
  chosen.sort((a, b) => a.rank - b.rank || a.pos - b.pos);
  const items = chosen.map(({ page, text }) => ({ page, text }));
  return { items, text: items.map((s) => `[p.${s.page}] ${s.text}`).join('\n') };
}
