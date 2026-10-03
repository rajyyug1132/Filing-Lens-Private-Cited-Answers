import { normalizeWhitespace } from './text';

export interface Page {
  page: number; // 1-indexed, as printed on the citation chip
  text: string;
}

export interface Chunk {
  id: number;
  page: number;
  text: string;
}

// Chunk strictly within page boundaries so every chunk maps to exactly one
// page, which is what a [p.N] citation points at. Long pages are split into
// overlapping word windows; short pages stay whole.
export function chunkPages(pages: Page[], maxWords = 160, overlap = 40): Chunk[] {
  const chunks: Chunk[] = [];
  for (const p of pages) {
    const words = normalizeWhitespace(p.text).split(' ').filter(Boolean);
    if (words.length === 0) continue;
    const step = Math.max(1, maxWords - overlap);
    for (let start = 0; start < words.length; start += step) {
      const slice = words.slice(start, start + maxWords);
      chunks.push({ id: chunks.length, page: p.page, text: slice.join(' ') });
      if (start + maxWords >= words.length) break;
    }
  }
  return chunks;
}
