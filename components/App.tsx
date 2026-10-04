'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { embed } from '@/lib/embed';
import { BASE } from '@/lib/env';
import { hydrate, indexPages } from '@/lib/index-doc';
import { checkWebGPU, extractiveEngine, loadFixtureEngine, loadWebLLM, loadWllama, type Engine, type GpuCheck } from '@/lib/llm';
import { readPdf } from '@/lib/pdf-browser';
import { ask, type Outcome } from '@/lib/pipeline';
import type { DocIndex } from '@/lib/retrieve';
import { deleteDoc, listDocs, loadDoc, packVectors, saveDoc, unpackVectors, type StoredDoc } from '@/lib/store';
import { caseLine } from '@/lib/case-line';
import { classifyDoc, type DocType } from '@/lib/doc-type';
import { deriveCandidates, parseQuestions, validateSuggestions, type Suggestion } from '@/lib/suggest';
import { AnswerCard } from './AnswerCard';
import { PageSheet } from './PageSheet';
import { PrivacySheet, useNetworkLog } from './PrivacySheet';
import { Ruler } from './Ruler';

type DocMeta = Awaited<ReturnType<typeof listDocs>>[number];
type Turn = { id: number; question: string; outcome?: Outcome; streaming?: string; error?: string };

function useOnline() {
  const [on, setOn] = useState(true);
  useEffect(() => {
    const u = () => setOn(navigator.onLine);
    u();
    window.addEventListener('online', u);
    window.addEventListener('offline', u);
    return () => { window.removeEventListener('online', u); window.removeEventListener('offline', u); };
  }, []);
  return on;
}

// "Reading page 12 of 75" / "Embedding on-device 40/388 chunks" -> a 0..1 ruler value.
function progressOf(s: string) {
  const m = /(\d+)\s*(?:of|\/)\s*(\d+)/.exec(s);
  return m && +m[2] ? { v: +m[1] / +m[2], right: `${m[1]} / ${m[2]}` } : null;
}


export default function App() {
  const [docs, setDocs] = useState<DocMeta[]>([]);
  const [active, setActive] = useState<{ meta: DocMeta; index: DocIndex; bytes: ArrayBuffer; docType: DocType } | null>(null);
  const [sug, setSug] = useState<{ status: 'loading' | 'ready'; items: Suggestion[]; closest: number[] }>({ status: 'loading', items: [], closest: [] });
  const [status, setStatus] = useState<string>('');
  const [engine, setEngine] = useState<Engine>(extractiveEngine);
  const [modelProgress, setModelProgress] = useState<{ progress: number; text: string } | null>(null);
  const [gpu, setGpu] = useState<GpuCheck | null>(null);
  const [modelError, setModelError] = useState<string>('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);
  const [showPrivacy, setShowPrivacy] = useState(false);
  const [confirmDel, setConfirmDel] = useState(false);
  const [sheet, setSheet] = useState<{ page: number; spans: Record<number, string[]> } | null>(null);
  const online = useOnline();
  const [strict, setStrict] = useState(false);
  const net = useNetworkLog();
  const fileRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listDocs().then(setDocs).catch(() => {});
    checkWebGPU().then(setGpu);
    if (new URLSearchParams(location.search).get('engine') === 'fixture') loadFixtureEngine().then(setEngine);
  }, []);
  useEffect(() => { if (turns.length) endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [turns]);

  const open = useCallback(async (id: string) => {
    const d = await loadDoc(id);
    if (!d) return;
    setActive({ meta: d, index: hydrate(d.chunks, unpackVectors(d.vectors, d.dim)), bytes: d.bytes, docType: d.docType ?? classifyDoc(d.chunks) });
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
      const doc: StoredDoc = { id: crypto.randomUUID(), name: f.name, pages: pages.length, chunks: index.chunks, vectors: buffer, dim, bytes, createdAt: Date.now(), docType: classifyDoc(pages) };
      await saveDoc(doc);
      setDocs(await listDocs());
      setActive({ meta: doc, index, bytes, docType: doc.docType! });
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
      const doc: StoredDoc = { id: 'sample-bestbuy-2023', name: meta.name, pages: meta.pages, chunks: meta.chunks, vectors: vec, dim: meta.dim, bytes: pdf, createdAt: Date.now(), docType: classifyDoc(meta.chunks) };
      await saveDoc(doc);
      setDocs(await listDocs());
      setActive({ meta: doc, index: hydrate(doc.chunks, unpackVectors(vec, meta.dim)), bytes: pdf, docType: doc.docType! });
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

  // Questions derived from the document's own text, shown only if they pass the routed gate.
  useEffect(() => {
    if (!active) return;
    let live = true;
    setSug({ status: 'loading', items: [], closest: [] });
    const pages = new Map<number, string>();
    for (const c of active.index.chunks) pages.set(c.page, (pages.get(c.page) ?? '') + ' ' + c.text);
    const cands = deriveCandidates([...pages].map(([page, text]) => ({ page, text })), active.docType);
    validateSuggestions(active.index, active.docType, cands, embed)
      .then((v) => live && setSug({ status: 'ready', items: v.shown, closest: v.closest }))
      .catch(() => live && setSug({ status: 'ready', items: [], closest: [] }));
    return () => { live = false; };
  }, [active]);

  async function suggestMore() {
    if (!active || !engine.chat || busy) return;
    setBusy(true);
    try {
      const src = Array.from(new Set([...sug.items.map((s) => s.page), ...sug.closest])).slice(0, 3);
      const text = active.index.chunks.filter((c) => src.includes(c.page)).map((c) => `[p.${c.page}] ${c.text}`).join('\n').slice(0, 2400);
      const raw = await engine.chat([{ role: 'system', content: 'Write short questions that the given text answers. One question per line. No numbering.' }, { role: 'user', content: `Text:\n${text}\n\nWrite 4 questions that this text answers.` }], 160);
      const cands = parseQuestions(raw).map((q) => ({ q, page: src[0] ?? 1, kind: 'term' as const }));
      const v = await validateSuggestions(active.index, active.docType, cands, embed);
      setSug((s) => ({ ...s, items: [...s.items, ...v.shown.filter((x) => !s.items.some((y) => y.q === x.q))] }));
    } finally {
      setBusy(false);
    }
  }

  const openSheet = (page: number, o: Outcome) => {
    const spans: Record<number, string[]> = {};
    for (const sn of o.snippets) (spans[sn.page] ??= []).push(sn.text);
    setSheet({ page, spans });
  };

  async function removeActive() {
    if (!active) return;
    if (!confirmDel) { setConfirmDel(true); return; }
    await deleteDoc(active.meta.id);
    setDocs(await listDocs());
    setActive(null);
    setStatus('');
    setTurns([]);
    setConfirmDel(false);
  }

  async function submit(question: string, force = false, id = Date.now()) {
    if (!active || !question.trim() || busy) return;
    if (force) setTurns((t) => t.map((x) => (x.id === id ? { id, question } : x)));
    else setTurns((t) => [...t, { id, question }]);
    setQ('');
    setBusy(true);
    try {
      const outcome = await ask(active.index, question, embed, engine, {
        strict,
        force,
        docType: active.docType,
        onText: (s) => setTurns((t) => t.map((x) => (x.id === id ? { ...x, streaming: s } : x))),
      });
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, outcome, streaming: undefined } : x)));
    } catch (e) {
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, error: (e as Error).message } : x)));
    } finally {
      setBusy(false);
    }
  }

  const prog = busy && !active ? progressOf(status) : null;
  const pct = modelProgress ? Math.round(modelProgress.progress * 100) : 0;
  return (
    <div className="shell">
      <header className="top">
        <div className="top-in">
          <div className="brand">
            <h1 className="wordmark">Filing Lens</h1>
            <p className="tag">Case files / local</p>
          </div>
          <button className="ledger" onClick={() => setShowPrivacy(true)} data-testid="privacy-pill" title="Where your data goes">
            <span className={`mark${online ? '' : ' off'}`} aria-hidden />
            <span className="led-t">{net.docBytesSent} B SENT · {online ? 'ON-DEVICE' : 'OFFLINE'}</span>
          </button>
        </div>
        <div className="ticks" aria-hidden />
      </header>

      <main className={`grid${active ? ' has-doc' : ''}`}>
        {active && (
          <section className="case" aria-label="Filing">
              <div className="casebar">
                <p className="caseline">{caseLine(active.meta.name, active.meta.pages)}</p>
                {active.docType === 'GENERAL' && <p className="caseline" data-testid="doc-type">GENERAL DOCUMENT · gate tuned for annual reports</p>}
                <div className="casebar-row">
                  <h2 className="casename" data-testid="doc-name">{active.meta.name}</h2>
                  <div className="acts">
                    <button className="textbtn label" style={{ color: 'var(--ink)' }} onClick={() => { setActive(null); setStatus(''); setConfirmDel(false); }}>Switch</button>
                    <button className="textbtn label" style={{ color: 'var(--ink)' }} onClick={removeActive} aria-label={confirmDel ? 'Confirm delete' : 'Delete this filing from the device'}>{confirmDel ? 'Confirm' : 'Delete'}</button>
                  </div>
                </div>
                <p className="status" data-testid="status">{status || `${active.meta.pages} pages · stored locally`}</p>
              </div>
          </section>
        )}
        <aside className="rail" aria-label="Model and mode">
          {!online && (
            <div className="rail-block">
              <span className="label">Offline</span>
              <p className="small">No connection. Indexed filings and a cached model still work.</p>
            </div>
          )}
          <div className="rail-block">
            <span className="label">Model</span>
            <div className="eng" data-testid="engine-label">{engine.label}</div>
            <p className="rail-long">
              {engine.id === 'extractive'
                ? 'Quotes matching sentences verbatim until a model is downloaded.'
                : engine.id === 'fixture'
                  ? 'Test mode: replays recorded responses.'
                  : 'Running fully in this browser.'}
            </p>
            {engine.id === 'extractive' && gpu && (
              <div className="gpu" data-testid="gpu-check">
                {gpu.webgpu ? '✓' : '✕'} {gpu.detail} → {gpu.webgpu ? 'Qwen2.5-3B (≈2 GB)' : 'Qwen2.5-1.5B on CPU (≈1 GB)'}
              </div>
            )}
            {engine.id === 'extractive' && modelProgress !== null && (
              <div className="dl" aria-label="Model download progress">
                <Ruler value={modelProgress.progress} compact ariaLabel="Model download progress" />
                <p className="cap"><span>{modelProgress.text}</span><span>{pct}%</span></p>
              </div>
            )}
            {modelError && (
              <div className="errbox"><span className="label">Error</span><p className="small">Model failed to load: {modelError}</p></div>
            )}
            <div className="rail-act">
              {engine.id === 'extractive' && modelProgress === null && (
                <button className="btn small" onClick={loadModel} data-testid="load-model">Download model</button>
              )}
              <p className="cap" style={{ marginTop: 8 }}>{engine.id === 'extractive' ? 'Works offline once cached' : 'Runs in this browser. Works offline.'}</p>
            </div>
          </div>
          <div className="rail-block">
            <span className="label">Mode</span>
            <label className="switch">
              <input type="checkbox" checked={strict} disabled={!engine.chat} onChange={(e) => setStrict(e.target.checked)} data-testid="strict-toggle" aria-label="Strict mode" />
              <span className="sw-track" aria-hidden><span className="sw-knob" /></span>
              <span className="sw-text">STRICT · {strict ? 'ON' : 'OFF'}</span>
            </label>
            <p className="rail-long">{engine.chat ? 'The model checks the snippets before it answers. It abstains much more often.' : 'Needs the model.'}</p>
          </div>
        </aside>

        <section className="stage">
          {!active && (
            <>
              <h2 className="display">Ask the filing.</h2>
              <p className="lede">Every answer cites its page. When the filing is silent, so is Filing Lens.</p>
              <div className="actions">
                <button className="btn solid" onClick={() => fileRef.current?.click()} disabled={busy} data-testid="upload-btn">
                  {busy ? 'Indexing…' : 'Open a filing PDF'}
                </button>
                <button className="btn" onClick={loadSample} disabled={busy} data-testid="sample-btn">Try a sample 10-K</button>
              </div>
              {busy && (
                <div className="dl" style={{ maxWidth: 480, marginTop: 24 }} role="status">
                  {prog && <Ruler value={prog.v} compact ariaLabel="Indexing progress" />}
                  <p className="cap"><span>{status || 'Working…'}</span>{prog && <span>{prog.right}</span>}</p>
                </div>
              )}
              {status && !busy && <p className="status" data-testid="status">{status}</p>}
              {docs.length > 0 && (
                <ul className="doclist">
                  <li><span className="label">On this device</span></li>
                  {docs.map((d) => (
                    <li key={d.id}>
                      <button className="link" onClick={() => open(d.id)}>
                        {d.name} <span>· {d.pages} pages</span>
                      </button>
                      <button
                        className="textbtn"
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
            </>
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
              <div className="thread">
                {turns.length === 0 && (
                  <div className="suggest" data-testid="suggest">
                    <span className="label">{sug.status === 'loading' ? 'Reading the document for questions…' : 'Questions from this document'}</span>
                    {sug.items.map((s) => (
                      <button key={s.q} onClick={() => submit(s.q)} disabled={busy} data-testid="suggestion" data-page={s.page} data-q={s.q}>{s.q}<span aria-hidden>p.{s.page}</span></button>
                    ))}
                    {sug.status === 'ready' && sug.items.length === 0 && (
                      <div className="own" data-testid="suggest-fallback">
                        <p className="reason">Ask your own question below.</p>
                        {sug.closest.length > 0 && (
                          <div className="pages">
                            <span className="label">Closest pages</span>
                            <div className="tabs">{sug.closest.map((p) => <button key={p} className="tabbtn" onClick={() => setSheet({ page: p, spans: {} })} aria-label={`Open page ${p}`}><span className="tab">p.{p}</span></button>)}</div>
                          </div>
                        )}
                      </div>
                    )}
                    {engine.chat && sug.status === 'ready' && <button className="textbtn more" onClick={suggestMore} disabled={busy} data-testid="suggest-more">Suggest more</button>}
                  </div>
                )}
                {turns.map((t, i) => (
                  <article key={t.id} className="turn">
                    <div className="idx">{String(i + 1).padStart(2, '0')}</div>
                    <div className="body">
                      <p className="q"><span className="label">Question</span>{t.question}</p>
                      {t.error && <div className="errbox"><span className="label">Error</span><p>{t.error}</p></div>}
                      {!t.outcome && !t.error && (
                        <div className="pending" data-testid="pending" role="status">
                          <span className="label">Searching this filing</span>
                          <p>{t.streaming ? t.streaming : 'Retrieving pages and scoring confidence…'}</p>
                          <div className="sweep" aria-hidden />
                        </div>
                      )}
                      {t.outcome && <AnswerCard outcome={t.outcome} threshold={t.outcome.threshold} onPage={(p) => openSheet(p, t.outcome!)} onForce={() => submit(t.question, true, t.id)} />}
                    </div>
                  </article>
                ))}
                <div ref={endRef} className="end" />
              </div>

              <form className="askbar" onSubmit={(e) => { e.preventDefault(); submit(q); }}>
                <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about this filing…" aria-label="Question" data-testid="question" enterKeyHint="send" />
                <button className="btn solid" disabled={busy || !q.trim()} data-testid="ask-btn">Ask</button>
              </form>
            </>
          )}
        </section>
      </main>

      {sheet && active && (
        <PageSheet bytes={active.bytes} page={sheet.page} total={active.meta.pages} index={active.index} spans={sheet.spans} onClose={() => setSheet(null)} onPage={(p) => setSheet({ ...sheet, page: p })} />
      )}
      {showPrivacy && <PrivacySheet log={net} onClose={() => setShowPrivacy(false)} />}
    </div>
  );
}
