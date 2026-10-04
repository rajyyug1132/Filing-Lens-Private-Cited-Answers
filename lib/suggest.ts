import type { DocType } from './doc-type';
import { prepare, routeGate } from './gate';
import { contentTerms } from './text';
import type { DocIndex } from './retrieve';

export interface Candidate { q: string; page: number; kind: 'list' | 'heading' | 'number' | 'term' }
export interface Suggestion extends Candidate { confidence: number }

const flat = (s: string) => s.replace(/\s+/g, ' ').trim();
const count = (re: RegExp, s: string) => (s.match(new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g')) ?? []).length;

const LISTS: [RegExp, string][] = [
  [/\b(?:text\s?books?|prescribed books?|prescribed text)\b/i, 'Which textbooks are listed?'],
  [/\b(?:reference books?|recommended reading|further reading)\b/i, 'Which reference books are listed?'],
  [/\bcourse outcomes?\b|\blearning outcomes?\b/i, 'What are the course outcomes?'],
  [/\bcourse objectives?\b|\blearning objectives?\b/i, 'What are the course objectives?'],
  [/\bprerequisites?\b/i, 'What are the prerequisites?'],
  [/\b(?:evaluation scheme|scheme of (?:evaluation|examination))\b/i, 'How is the course assessed?'],
];
const ITEMS = ['Risk Factors', 'Legal Proceedings', 'Properties', 'Business'];
const METRICS = ['revenue', 'net earnings', 'net income', 'operating income', 'gross profit', 'total assets', 'cash and cash equivalents', 'capital expenditures', 'diluted earnings per share', 'free cash flow'];
const STOP = new Set(['document', 'company', 'page', 'table', 'including', 'included', 'also', 'such', 'other', 'which', 'would', 'could', 'their', 'there', 'these', 'those', 'based', 'related', 'respect', 'following', 'section', 'during', 'under', 'within']);

export function deriveCandidates(pages: { page: number; text: string }[], docType: DocType): Candidate[] {
  const P = pages.map((p) => ({ page: p.page, text: flat(p.text) }));
  const lists: Candidate[] = [], heads: Candidate[] = [], nums: Candidate[] = [], terms: Candidate[] = [];
  const seen = new Set<string>();
  const add = (arr: Candidate[], q: string, page: number, kind: Candidate['kind']) => { if (!seen.has(q.toLowerCase())) { seen.add(q.toLowerCase()); arr.push({ q, page, kind }); } };

  if (docType === 'GENERAL') {
    for (const [re, q] of LISTS) {
      const best = P.map((p) => ({ page: p.page, n: count(re, p.text) })).sort((a, b) => b.n - a.n)[0];
      if (best && best.n > 0) add(lists, q, best.page, 'list');
    }
  }
  // headings: "Module 3: Deadlocks and Memory Management — 9 Hours" / "Item 1A. Risk Factors"
  const unit = /\b(Module|Unit|Chapter)\s+(\d+|[IVX]+)\s*[:.\-–—]\s*([A-Z][A-Za-z&,'/ -]{3,70}?)(?=\s*[—–]\s*\d|\s+\d+\s*(?:hours?|hrs?)\b|\s+(?:[A-Z][a-z]+\s+){0,2}(?:Topics|Contents)\b|[.;]|$)/g;
  for (const p of P) for (const m of p.text.matchAll(unit)) { const t = m[3].trim().split(' ').slice(0, 8).join(' '); if (t.split(' ').length >= 2) add(heads, `What does ${t} cover?`, p.page, 'heading'); }
  if (docType === 'FILING') {
    const toc = new Set(P.filter((p) => count(/\bitem\s+\d+[ab]?\b/i, p.text) >= 6).map((p) => p.page));
    for (const name of ITEMS) {
      const re = new RegExp(`\\bitem\\s+\\d+[ab]?\\.?\\s*${name}\\b`, 'i');
      const p = P.find((x) => !toc.has(x.page) && re.test(x.text));
      if (p) add(heads, `What does ${name} cover?`, p.page, 'heading');
    }
  }
  // labelled numbers: hours, marks, credits, dates, currency
  for (const p of P) {
    for (const m of p.text.matchAll(/\b(Module|Unit|Chapter)\s+(\d+)\b[^.]{0,100}?\b\d+\s*(?:hours?|hrs?)\b/gi)) add(nums, `How many hours is ${m[1]} ${m[2]}?`, p.page, 'number');
    for (const m of p.text.matchAll(/\b([A-Z]{2,6})\b[^.\n]{0,12}?\b\d+\s*marks\b/g)) add(nums, `How many marks for ${m[1]}?`, p.page, 'number');
    if (/\bcredits?\s*[:\-]?\s*\d|\b\d+\s*credits?\b/i.test(p.text)) add(nums, 'How many credits is the course?', p.page, 'number');
    for (const m of p.text.matchAll(/\b((?:Internal |Mid[- ]?term |Final )?(?:test|exam|examination|assignment|quiz|deadline)(?: \d)?)\s*[:\-–—]\s*\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*/gi)) add(nums, `When is ${m[1].replace(/^./, (c) => c.toUpperCase())}?`, p.page, 'number');
  }
  // filings: labelled money figures ("Revenue $ 46,298") -> "What was revenue in fiscal 2023?"
  if (docType === 'FILING') {
    const fy = (P.slice(0, 5).map((p) => p.text).join(' ').match(/\b(?:fiscal(?: year)?\s+)(20\d{2})\b/i) ?? [])[1];
    for (const metric of METRICS) {
      const re = new RegExp(`\\b${metric}\\b[^.$]{0,40}\\$\\s?\\d[\\d,]{2,}`, 'i');
      const p = P.find((x) => re.test(x.text));
      if (p) add(nums, `What was ${metric} in ${fy ? `fiscal ${fy}` : 'the latest year'}?`, p.page, 'number');
    }
  }
  // repeated key terms
  // repeated two-word terms ("memory management", "domestic segment"), counted where both words are adjacent content terms
  const tf = new Map<string, { n: number; pages: Map<number, number> }>();
  for (const p of P) {
    const ts = contentTerms(p.text);
    for (let i = 0; i + 1 < ts.length; i++) {
      const [a, b] = [ts[i], ts[i + 1]];
      if (a.length < 4 || b.length < 4 || STOP.has(a) || STOP.has(b) || /\d/.test(a + b) || a === b) continue;
      const t = `${a} ${b}`, e = tf.get(t) ?? { n: 0, pages: new Map() };
      e.n++; e.pages.set(p.page, (e.pages.get(p.page) ?? 0) + 1); tf.set(t, e);
    }
  }
  const topTerm = [...tf.entries()].filter(([, e]) => e.n >= 4).sort((a, b) => b[1].n - a[1].n)[0];
  if (topTerm) add(terms, `What does the document say about ${topTerm[0]}?`, [...topTerm[1].pages.entries()].sort((a, b) => b[1] - a[1])[0][0], 'term');

  return [...lists.slice(0, 3), ...heads.slice(0, 2), ...nums.slice(0, docType === 'FILING' ? 3 : 2), ...terms.slice(0, 1)].slice(0, 6);
}

// A suggestion is shown only if it passes the same routed gate the app uses for questions AND its source page is in the top-8.
export async function validateSuggestions(
  index: DocIndex, docType: DocType, cands: Candidate[], embed: (t: string[]) => Promise<Float32Array[]>,
): Promise<{ shown: Suggestion[]; closest: number[] }> {
  if (!cands.length) return { shown: [], closest: [] };
  const qv = await embed(cands.map((c) => c.q));
  const scored = cands.map((c, i) => {
    const r = prepare(index, c.q, qv[i]);
    const g = routeGate(index, r, docType);
    const pages = r.hits.map((h) => h.chunk.page);
    return { ...c, confidence: g.confidence, ok: g.confidence >= g.threshold && pages.slice(0, 8).includes(c.page), pages };
  });
  const shown = scored.filter((s) => s.ok).map(({ ok: _o, pages: _p, ...s }) => s);
  if (shown.length >= 2) return { shown, closest: [] };
  const best = [...scored].sort((a, b) => b.confidence - a.confidence)[0];
  return { shown: [], closest: Array.from(new Set(best.pages)).slice(0, 5) };
}

// "Suggest more" (3B, opt-in): one question per line.
export function parseQuestions(text: string): string[] {
  return Array.from(new Set(text.split('\n').map((l) => l.replace(/^\s*(?:[-*•]|\d+[.)])\s*/, '').trim()).filter((l) => l.length >= 12 && l.length <= 140 && l.endsWith('?')))).slice(0, 4);
}
