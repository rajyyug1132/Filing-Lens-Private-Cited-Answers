// Step 1 of the bench: extract every FinanceBench PDF to per-page text with
// the same pdf.js text pass the browser uses. Cached under bench/.cache.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';
import { extractPages } from '../lib/pdf-text';

export const FB_DIR = process.env.FINANCEBENCH_DIR ?? join(process.cwd(), 'bench/.data/financebench');
export const CACHE = join(process.cwd(), 'bench/.cache');

export async function pagesFor(docName: string) {
  mkdirSync(join(CACHE, 'pages'), { recursive: true });
  const out = join(CACHE, 'pages', docName + '.json');
  if (existsSync(out)) return JSON.parse(readFileSync(out, 'utf8'));
  const data = new Uint8Array(readFileSync(join(FB_DIR, 'pdfs', docName + '.pdf')));
  const task = getDocument({ data, useSystemFonts: true, verbosity: 0 });
  const pages = await extractPages((await task.promise) as never);
  await task.destroy();
  writeFileSync(out, JSON.stringify(pages));
  return pages;
}
