import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { FB_DIR, pagesFor } from './extract';
const rows = readFileSync(join(FB_DIR, 'data/financebench_open_source.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const docs = Array.from(new Set(rows.map((r) => r.doc_name))).sort();
const t0 = Date.now();
for (const [i, d] of docs.entries()) { const p = await pagesFor(d); if (i % 10 === 0) console.log(i, d, p.length, ((Date.now() - t0) / 1000).toFixed(0) + 's'); }
// page-index check: where does the evidence text actually live?
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
let same = 0, plus1 = 0, other = 0;
for (const r of rows) for (const ev of r.evidence) {
  const pages = await pagesFor(ev.doc_name);
  const probe = norm(ev.evidence_text).split(' ').slice(0, 12).join(' ');
  const hit = pages.findIndex((p: any) => norm(p.text).includes(probe));
  if (hit === ev.evidence_page_num) same++; else if (hit === ev.evidence_page_num - 1) plus1++; else other++;
}
console.log({ zeroIndexedMatch: same, oneIndexedMatch: plus1, unmatched: other });
