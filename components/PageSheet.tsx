'use client';

import { Fragment, useEffect, useRef, useState } from 'react';
import { renderPage, type Rect } from '@/lib/pdf-browser';
import type { DocIndex } from '@/lib/retrieve';

// Wrap the cited sentences in <mark> inside the extracted page text.
function marked(text: string, spans: string[]) {
  const ranges: [number, number][] = [];
  for (const s of spans) {
    const k = s.trim();
    const at = k ? text.indexOf(k) : -1;
    if (at >= 0) ranges.push([at, at + k.length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const out: [number, number][] = [];
  for (const r of ranges) (out.length && r[0] <= out[out.length - 1][1] ? (out[out.length - 1][1] = Math.max(out[out.length - 1][1], r[1])) : out.push([...r]));
  const parts: React.ReactNode[] = [];
  let at = 0;
  out.forEach(([a, b], i) => { parts.push(<Fragment key={`t${i}`}>{text.slice(at, a)}</Fragment>, <mark key={`m${i}`}>{text.slice(a, b)}</mark>); at = b; });
  parts.push(<Fragment key="end">{text.slice(at)}</Fragment>);
  return parts;
}

export function PageSheet({ bytes, page, total, index, spans, onClose, onPage }: {
  bytes: ArrayBuffer; page: number; total: number; index: DocIndex; spans: Record<number, string[]>; onClose: () => void; onPage: (p: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [err, setErr] = useState('');
  const [view, setView] = useState<'page' | 'text'>('page');
  const [rects, setRects] = useState<Rect[]>([]);
  const here = spans[page] ?? [];

  useEffect(() => {
    if (view !== 'page' || !canvas.current || !wrap.current) return;
    setErr('');
    setRects([]);
    renderPage(bytes, page, canvas.current, wrap.current.clientWidth - 34, here).then(setRects).catch((e) => setErr(String(e?.message ?? e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bytes, page, view]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [onClose]);

  const text = index.chunks.filter((c) => c.page === page).map((c) => c.text).join(' … ');
  return (
    <div className="sheet-backdrop" onClick={onClose}>
      <div className="sheet" role="dialog" aria-modal aria-label={`Page ${page}`} onClick={(e) => e.stopPropagation()} data-testid="page-sheet">
        <div className="sheet-head">
          <div className="strong" data-testid="sheet-page">Page {page} <span className="muted">of {total}</span></div>
          <button className="ghost txt" onClick={onClose} aria-label="Close">Close</button>
          <div className="sheet-nav">
            <button className="ghost" onClick={() => onPage(Math.max(1, page - 1))} disabled={page <= 1} aria-label="Previous page">‹</button>
            <div className="seg">
              <button className={view === 'page' ? 'on' : ''} onClick={() => setView('page')}>Page</button>
              <button className={view === 'text' ? 'on' : ''} onClick={() => setView('text')}>Text</button>
            </div>
            <button className="ghost" onClick={() => onPage(Math.min(total, page + 1))} disabled={page >= total} aria-label="Next page">›</button>
          </div>
        </div>
        <div className="sheet-body" ref={wrap}>
          <p className="label hl-note" data-testid="hl-note">
            {here.length === 0 ? 'Not one of the pages the answer was built from' : view === 'page' ? (rects.length ? 'Evidence highlighted' : 'Evidence page · exact lines not matched') : 'Evidence marked in the text'}
          </p>
          {view === 'page' ? (
            <div className="canvas-wrap">
              <canvas ref={canvas} />
              {rects.map((r, i) => <span key={i} className="hl-span" style={{ left: r.left, top: r.top, width: r.width, height: r.height }} />)}
            </div>
          ) : (
            <p className="pagetext">{text ? marked(text, here) : 'No text on this page.'}</p>
          )}
          {err && <p className="error small">{err}</p>}
        </div>
      </div>
    </div>
  );
}
