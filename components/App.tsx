'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { embed } from '@/lib/embed';
import { BASE } from '@/lib/env';
import { hydrate, indexPages } from '@/lib/index-doc';
import { checkWebGPU, extractiveEngine, loadFixtureEngine, loadWebLLM, loadWllama, type Engine, type GpuCheck } from '@/lib/llm';
import { readPdf } from '@/lib/pdf-browser';
import { ask, decisionModel, type Outcome } from '@/lib/pipeline';
import type { DocIndex } from '@/lib/retrieve';
import { deleteDoc, listDocs, loadDoc, packVectors, saveDoc, unpackVectors, type StoredDoc } from '@/lib/store';
import { AnswerCard } from './AnswerCard';
import { PageSheet } from './PageSheet';
import { PrivacySheet, useNetworkLog } from './PrivacySheet';

type DocMeta = Awaited<ReturnType<typeof listDocs>>[number];
type Turn = { id: number; question: string; outcome?: Outcome; streaming?: string; error?: string };

const SUGGESTIONS = ['How much cash did operating activities provide in fiscal 2023?', 'What are the main risk factors?', 'What did Tesla say about Cybertruck production?'];

export default function App() {
  const [docs, setDocs] = useState<DocMeta[]>([]);
  const [active, setActive] = useState<{ meta: DocMeta; index: DocIndex; bytes: ArrayBuffer } | null>(null);
  const [status, setStatus] = useState<string>('');
  const [engine, setEngine] = useState<Engine>(extractiveEngine);
  const [modelProgress, setModelProgress] = useState<{ progress: number; text: string } | null>(null);
  const [gpu, setGpu] = useState<GpuCheck | null>(null);
  const [modelError, setModelError] = useState<string>('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);
  const [sheetPage, setSheetPage] = useState<number | null>(null);
  const [showPrivacy, setShowPrivacy] = useState(false);
  const net = useNetworkLog();
  const fileRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listDocs().then(setDocs).catch(() => {});
    checkWebGPU().then(setGpu);
    if (new URLSearchParams(location.search).get('engine') === 'fixture') loadFixtureEngine().then(setEngine);
  }, []);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }), [turns]);

  const open = useCallback(async (id: string) => {
    const d = await loadDoc(id);
    if (!d) return;
    setActive({ meta: d, index: hydrate(d.chunks, unpackVectors(d.vectors, d.dim)), bytes: d.bytes });
    setTurns([]);
  }, []);

  async function onFile(f: File) {
    setBusy(true);
    try {
      const bytes = await f.arrayBuffer();
      setStatus('Reading pages…');
      const pages = await readPdf(bytes, (n, t) => setStatus(`Reading page ${n} of ${t}`));
      if (!pages.some((p) => p.text.trim())) throw new Error('No selectable text in this PDF (scanned image?). OCR is not supported yet.');
      const index = await indexPages(pages, embed, (d, t) => setStatus(`Embedding on-device ${d}/${t} chunks`));
      const { buffer, dim } = packVectors(index.vectors);
      const doc: StoredDoc = { id: crypto.randomUUID(), name: f.name, pages: pages.length, chunks: index.chunks, vectors: buffer, dim, bytes, createdAt: Date.now() };
      await saveDoc(doc);
      setDocs(await listDocs());
      setActive({ meta: doc, index, bytes });
      setTurns([]);
      setStatus(`Indexed ${pages.length} pages · ${index.chunks.length} chunks · stored in this browser only`);
    } catch (e) {
      setStatus(`Could not index: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  async function loadSample() {
    setBusy(true);
    setStatus('Loading the pre-indexed sample filing…');
    try {
      const dir = `${BASE}/samples/bestbuy-2023`;
      const [meta, vec, pdf] = await Promise.all([
        fetch(`${dir}/index.json`).then((r) => r.json()),
        fetch(`${dir}/vectors.f32`).then((r) => r.arrayBuffer()),
        fetch(`${dir}/filing.pdf`).then((r) => r.arrayBuffer()),
      ]);
      const doc: StoredDoc = { id: 'sample-bestbuy-2023', name: meta.name, pages: meta.pages, chunks: meta.chunks, vectors: vec, dim: meta.dim, bytes: pdf, createdAt: Date.now() };
      await saveDoc(doc);
      setDocs(await listDocs());
      setActive({ meta: doc, index: hydrate(doc.chunks, unpackVectors(vec, meta.dim)), bytes: pdf });
      setTurns([]);
      setStatus(`Sample loaded: ${meta.pages} pages · ${meta.chunks.length} chunks · pre-indexed`);
    } catch (e) {
      setStatus(`Could not load sample: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  async function loadModel() {
    setModelError('');
    setModelProgress({ progress: 0, text: 'Starting…' });
    try {
      const g = gpu ?? (await checkWebGPU());
      const e = g.webgpu ? await loadWebLLM(g, setModelProgress) : await loadWllama(setModelProgress);
      setEngine(e);
    } catch (e) {
      setModelError((e as Error).message || String(e));
    } finally {
      setModelProgress(null);
    }
  }

  async function submit(question: string) {
    if (!active || !question.trim() || busy) return;
    const id = Date.now();
    setTurns((t) => [...t, { id, question }]);
    setQ('');
    setBusy(true);
    try {
      const outcome = await ask(active.index, question, embed, engine, {
        onText: (s) => setTurns((t) => t.map((x) => (x.id === id ? { ...x, streaming: s } : x))),
      });
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, outcome, streaming: undefined } : x)));
    } catch (e) {
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, error: (e as Error).message } : x)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden>◎</span>
          <div>
            <h1>Filing Lens</h1>
            <p className="sub">Local 3B · cited · private</p>
          </div>
        </div>
        <button className="privacy-pill" onClick={() => setShowPrivacy(true)} data-testid="privacy-pill">
          <span className="dot" /> {net.docBytesSent} doc bytes sent
        </button>
      </header>

      <main>
        {!active && (
          <section className="card hero">
            <h2>Ask your annual report anything.</h2>
            <p>
              The PDF is read, indexed and answered <b>on this device</b>. Every answer cites its page, and when the filing doesn’t support an
              answer, Filing Lens says so instead of guessing.
            </p>
            <div className="row">
              <button className="primary" onClick={() => fileRef.current?.click()} disabled={busy} data-testid="upload-btn">
                {busy ? status || 'Working…' : 'Open a filing PDF'}
              </button>
              <button className="secondary" onClick={loadSample} disabled={busy} data-testid="sample-btn">Try a sample 10-K</button>
            </div>
            {status && !busy && <p className="muted small" data-testid="status">{status}</p>}
            {docs.length > 0 && (
              <ul className="doclist">
                {docs.map((d) => (
                  <li key={d.id}>
                    <button className="link" onClick={() => open(d.id)}>
                      {d.name} <span className="muted">· {d.pages} pages</span>
                    </button>
                    <button
                      className="ghost small"
                      aria-label={`Delete ${d.name}`}
                      onClick={async () => {
                        await deleteDoc(d.id);
                        setDocs(await listDocs());
                      }}
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf"
          hidden
          data-testid="file-input"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
            e.target.value = '';
          }}
        />

        {active && (
          <>
            <section className="card docbar">
              <div>
                <div className="docname" data-testid="doc-name">{active.meta.name}</div>
                <div className="muted small" data-testid="status">{status || `${active.meta.pages} pages · stored locally`}</div>
              </div>
              <button className="ghost small" onClick={() => { setActive(null); setStatus(''); }}>Switch</button>
            </section>

            <section className="card engine">
              <div>
                <div className="small strong" data-testid="engine-label">{engine.label}</div>
                <div className="muted small">
                  {engine.id === 'extractive'
                    ? 'Quotes matching sentences verbatim until a model is downloaded.'
                    : engine.id === 'fixture'
                      ? 'Test mode: replays recorded responses.'
                      : 'Running fully in this browser.'}
                </div>
                {engine.id === 'extractive' && gpu && (
                  <div className="xsmall muted" data-testid="gpu-check">
                    {gpu.webgpu ? '✓' : '✕'} {gpu.detail} → {gpu.webgpu ? 'Qwen2.5-3B (≈2 GB)' : 'Qwen2.5-1.5B on CPU (≈1 GB)'}
                  </div>
                )}
                {modelProgress && <div className="xsmall muted">{modelProgress.text}</div>}
                {modelError && <div className="error small">Model failed to load: {modelError}</div>}
              </div>
              {engine.id === 'extractive' &&
                (modelProgress === null ? (
                  <button className="secondary small" onClick={loadModel} data-testid="load-model">Download model</button>
                ) : (
                  <div className="progress" aria-label="Model download progress"><div style={{ width: `${Math.round(modelProgress.progress * 100)}%` }} /><span>{Math.round(modelProgress.progress * 100)}%</span></div>
                ))}
            </section>

            <div className="thread">
              {turns.length === 0 && (
                <div className="suggest">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} className="chip-q" onClick={() => submit(s)} disabled={busy}>{s}</button>
                  ))}
                </div>
              )}
              {turns.map((t) => (
                <div key={t.id} className="turn">
                  <div className="bubble-q">{t.question}</div>
                  {t.error && <div className="card error">{t.error}</div>}
                  {!t.outcome && !t.error && (
                    <div className="card pending" data-testid="pending">{t.streaming ? t.streaming : 'Retrieving pages and scoring confidence…'}</div>
                  )}
                  {t.outcome && <AnswerCard outcome={t.outcome} threshold={decisionModel.threshold} onPage={setSheetPage} />}
                </div>
              ))}
              <div ref={endRef} />
            </div>
          </>
        )}
      </main>

      {active && (
        <form className="askbar" onSubmit={(e) => { e.preventDefault(); submit(q); }}>
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about this filing…" aria-label="Question" data-testid="question" enterKeyHint="send" />
          <button className="primary" disabled={busy || !q.trim()} data-testid="ask-btn">Ask</button>
        </form>
      )}

      {sheetPage !== null && active && (
        <PageSheet bytes={active.bytes} page={sheetPage} total={active.meta.pages} index={active.index} onClose={() => setSheetPage(null)} onPage={setSheetPage} />
      )}
      {showPrivacy && <PrivacySheet log={net} onClose={() => setShowPrivacy(false)} />}
    </div>
  );
}
