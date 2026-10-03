import type { Page } from './chunk';
import { BASE, importUrl } from './env';
import { extractPages } from './pdf-text';

type PdfJs = {
  GlobalWorkerOptions: { workerSrc: string };
  getDocument(src: { data: Uint8Array }): { promise: Promise<any> };
};

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

export async function renderPage(bytes: ArrayBuffer, pageNo: number, canvas: HTMLCanvasElement, cssWidth: number) {
  const doc = await (await pdfjs()).getDocument({ data: new Uint8Array(bytes.slice(0)) }).promise;
  const page = await doc.getPage(pageNo);
  const base = page.getViewport({ scale: 1 });
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const vp = page.getViewport({ scale: (cssWidth / base.width) * dpr });
  canvas.width = vp.width;
  canvas.height = vp.height;
  canvas.style.width = `${cssWidth}px`;
  await page.render({ canvasContext: canvas.getContext('2d')!, viewport: vp, canvas }).promise;
  doc.destroy?.();
}
