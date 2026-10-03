'use client';

import { useEffect, useRef, useState } from 'react';
import { renderPage } from '@/lib/pdf-browser';
import type { DocIndex } from '@/lib/retrieve';

export function PageSheet({ bytes, page, total, index, onClose, onPage }: {
  bytes: ArrayBuffer; page: number; total: number; index: DocIndex; onClose: () => void; onPage: (p: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [err, setErr] = useState('');
  const [view, setView] = useState<'page' | 'text'>('page');

  useEffect(() => {
    if (view !== 'page' || !canvas.current || !wrap.current) return;
    setErr('');
    renderPage(bytes, page, canvas.current, wrap.current.clientWidth).catch((e) => setErr(String(e?.message ?? e)));
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
          <button className="ghost small" onClick={() => onPage(Math.max(1, page - 1))} disabled={page <= 1} aria-label="Previous page">‹</button>
          <div className="strong" data-testid="sheet-page">Page {page} <span className="muted">of {total}</span></div>
          <button className="ghost small" onClick={() => onPage(Math.min(total, page + 1))} disabled={page >= total} aria-label="Next page">›</button>
          <div className="seg">
            <button className={view === 'page' ? 'on' : ''} onClick={() => setView('page')}>Page</button>
            <button className={view === 'text' ? 'on' : ''} onClick={() => setView('text')}>Text</button>
          </div>
          <button className="ghost small" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="sheet-body" ref={wrap}>
          {view === 'page' ? <canvas ref={canvas} /> : <p className="pagetext">{text || 'No text on this page.'}</p>}
          {err && <p className="error small">{err}</p>}
        </div>
      </div>
    </div>
  );
}
