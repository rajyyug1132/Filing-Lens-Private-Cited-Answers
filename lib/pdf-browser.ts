import type { Page } from './chunk';
import { BASE, importUrl } from './env';
import { extractPages } from './pdf-text';

type PdfJs = {
  GlobalWorkerOptions: { workerSrc: string };
  getDocument(src: { data: Uint8Array }): { promise: Promise<any> };
  Util: { transform(a: number[], b: number[]): number[] };
};

export type Rect = { left: number; top: number; width: number; height: number };
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '');

let lib: Promise<PdfJs> | null = null;
export function pdfjs(): Promise<PdfJs> {
  return (lib ??= importUrl<PdfJs>(`${location.origin}${BASE}/vendor/pdfjs/pdf.min.mjs`).then((m) => {
    m.GlobalWorkerOptions.workerSrc = `${location.origin}${BASE}/vendor/pdfjs/pdf.worker.min.mjs`;
    return m;
  }));
}

export async function readPdf(bytes: ArrayBuffer, onPage?: (n: number, total: number) => void): Promise<Page[]> {
  const doc = await (await pdfjs()).getDocument({ data: new Uint8Array(bytes.slice(0)) }).promise;
  try {
    return await extractPages(doc, onPage);
  } finally {
    doc.destroy?.();
  }
}

// Draws the page and returns CSS-pixel boxes for the text items that make up the cited sentences (`spans`).
export async function renderPage(bytes: ArrayBuffer, pageNo: number, canvas: HTMLCanvasElement, cssWidth: number, spans: string[] = []): Promise<Rect[]> {
  const lib = await pdfjs();
  const doc = await lib.getDocument({ data: new Uint8Array(bytes.slice(0)) }).promise;
  try {
    const page = await doc.getPage(pageNo);
    const base = page.getViewport({ scale: 1 });
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const vp = page.getViewport({ scale: (cssWidth / base.width) * dpr });
    canvas.width = vp.width;
    canvas.height = vp.height;
    canvas.style.width = `${cssWidth}px`;
    await page.render({ canvasContext: canvas.getContext('2d')!, viewport: vp, canvas }).promise;
    if (!spans.length) return [];
    // Match the sentences against the concatenated text items (letters and digits only), then box the items they cover.
    const items = (await page.getTextContent()).items.filter((i: any) => typeof i.str === 'string' && norm(i.str));
    let flat = '';
    const owner: number[] = [];
    items.forEach((it: any, idx: number) => { const n = norm(it.str); flat += n; for (let k = 0; k < n.length; k++) owner.push(idx); });
    const hit = new Set<number>();
    for (const sp of spans) {
      const n = norm(sp);
      if (n.length < 12) continue;
      let at = flat.indexOf(n), len = n.length;
      if (at < 0) { const head = n.slice(0, 60); at = flat.indexOf(head); len = head.length; }
      for (let k = at; at >= 0 && k < at + len; k++) hit.add(owner[k]);
    }
    return [...hit].map((idx) => {
      const it: any = items[idx];
      const t = lib.Util.transform(vp.transform, it.transform);
      const h = Math.hypot(t[2], t[3]);
      return { left: t[4] / dpr, top: (t[5] - h) / dpr, width: (it.width * vp.scale) / dpr, height: (h * 1.2) / dpr };
    });
  } finally {
    doc.destroy?.();
  }
}
