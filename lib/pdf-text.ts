import type { Page } from './chunk';

// Minimal structural type for the pdf.js document object, so this file works
// with both the browser build and the Node legacy build.
interface PdfDoc {
  numPages: number;
  getPage(n: number): Promise<{
    getTextContent(): Promise<{ items: Array<{ str?: string; hasEOL?: boolean }> }>;
    cleanup?: () => void;
  }>;
}

export async function extractPages(doc: PdfDoc, onPage?: (n: number, total: number) => void): Promise<Page[]> {
  const pages: Page[] = [];
  for (let n = 1; n <= doc.numPages; n++) {
    const page = await doc.getPage(n);
    const tc = await page.getTextContent();
    let text = '';
    for (const it of tc.items) {
      if (typeof it.str !== 'string') continue;
      text += it.str + (it.hasEOL ? '\n' : ' ');
    }
    pages.push({ page: n, text });
    page.cleanup?.();
    onPage?.(n, doc.numPages);
  }
  return pages;
}
