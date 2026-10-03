'use client';

import { useEffect, useState } from 'react';

export interface NetLog {
  docBytesSent: number; // request-body bytes sent by any fetch/XHR/beacon from this page
  bodyRequests: { method: string; url: string; bytes: number }[];
  hosts: { host: string; requests: number; kb: number }[];
}

function bodySize(b: unknown): number {
  if (b == null) return 0;
  if (typeof b === 'string') return new TextEncoder().encode(b).length;
  if (b instanceof Blob) return b.size;
  if (b instanceof ArrayBuffer) return b.byteLength;
  if (ArrayBuffer.isView(b)) return b.byteLength;
  if (b instanceof URLSearchParams) return b.toString().length;
  return 1; // FormData / streams: unknown size, but non-zero = flagged
}

const sent: NetLog['bodyRequests'] = [];
let patched = false;
function patch() {
  if (patched || typeof window === 'undefined') return;
  patched = true;
  const f = window.fetch.bind(window);
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const method = (init?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase();
    const url = String(input instanceof Request ? input.url : input);
    if (init?.body != null || !['GET', 'HEAD'].includes(method)) {
      if (init?.body == null && input instanceof Request)
        input.clone().arrayBuffer().then((b) => sent.push({ method, url, bytes: b.byteLength || 1 }), () => sent.push({ method, url, bytes: 1 }));
      else sent.push({ method, url, bytes: Math.max(1, bodySize(init?.body)) });
    }
    return f(input, init);
  };
  const xo = XMLHttpRequest.prototype.open;
  const xs = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (this: XMLHttpRequest & { _m?: string; _u?: string }, m: string, u: string | URL, ...rest: unknown[]) {
    this._m = m;
    this._u = String(u);
    return (xo as (...a: unknown[]) => void).call(this, m, u, ...rest);
  } as typeof xo;
  XMLHttpRequest.prototype.send = function (this: XMLHttpRequest & { _m?: string; _u?: string }, body?: Document | XMLHttpRequestBodyInit | null) {
    const bytes = bodySize(body);
    if (bytes > 0) sent.push({ method: this._m ?? 'POST', url: this._u ?? '', bytes });
    return xs.call(this, body);
  };
  // Requests from workers bypass the wrappers above; the service worker reports those.
  navigator.serviceWorker?.addEventListener('message', (e: MessageEvent) => {
    if (e.data?.type === 'filing-lens:body') sent.push({ method: `${e.data.method} (sw)`, url: e.data.url, bytes: Math.max(1, e.data.bytes) });
  });
  if (navigator.sendBeacon) {
    const sb = navigator.sendBeacon.bind(navigator);
    navigator.sendBeacon = (url, data) => {
      sent.push({ method: 'BEACON', url: String(url), bytes: bodySize(data) });
      return sb(url, data);
    };
  }
}

export function useNetworkLog(): NetLog {
  const [log, setLog] = useState<NetLog>({ docBytesSent: 0, bodyRequests: [], hosts: [] });
  useEffect(() => {
    patch();
    const tick = () => {
      const hosts = new Map<string, { requests: number; kb: number }>();
      for (const e of performance.getEntriesByType('resource') as PerformanceResourceTiming[]) {
        const h = new URL(e.name).host || 'local';
        const v = hosts.get(h) ?? { requests: 0, kb: 0 };
        v.requests++;
        v.kb += (e.transferSize || 0) / 1024;
        hosts.set(h, v);
      }
      setLog({
        docBytesSent: sent.reduce((a, b) => a + b.bytes, 0),
        bodyRequests: [...sent],
        hosts: Array.from(hosts, ([host, v]) => ({ host, ...v })).sort((a, b) => b.requests - a.requests),
      });
    };
    tick();
    const id = setInterval(tick, 1500);
    return () => clearInterval(id);
  }, []);
  return log;
}

export function PrivacySheet({ log, onClose }: { log: NetLog; onClose: () => void }) {
  const here = typeof location !== 'undefined' ? location.host : '';
  return (
    <div className="sheet-backdrop" onClick={onClose}>
      <div className="sheet" role="dialog" aria-modal aria-label="Privacy" onClick={(e) => e.stopPropagation()} data-testid="privacy-sheet">
        <div className="sheet-head">
          <div className="strong">Where your data goes</div>
          <button className="ghost small" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="sheet-body pad">
          <div className="bigstat" data-testid="bytes-sent">{log.docBytesSent} B</div>
          <div className="muted small">
            request-body bytes sent this session: fetch, XHR and beacon from this page, plus every request the service worker sees from the page and its
            workers (the service worker is active from the second visit on)
          </div>
          <ul className="facts small">
            <li>The PDF is read from your file picker with pdf.js, in this tab.</li>
            <li>Embeddings (MiniLM, 23 MB) and the ONNX runtime are served from this app’s own origin.</li>
            <li>The index and the PDF live in IndexedDB on this device. Delete them from the home screen.</li>
            <li>The only third-party download is the 3B model weights (GET only, once, then cached).</li>
          </ul>
          <div className="strong small">Hosts contacted</div>
          <table className="nettable">
            <thead><tr><th>Host</th><th>Requests</th><th>KB in</th></tr></thead>
            <tbody>
              {log.hosts.map((h) => (
                <tr key={h.host}><td>{h.host === here ? `${h.host} (this app)` : h.host}</td><td>{h.requests}</td><td>{Math.round(h.kb)}</td></tr>
              ))}
            </tbody>
          </table>
          {log.bodyRequests.length > 0 && (
            <>
              <div className="strong small error">Requests with a body</div>
              <ul className="small">{log.bodyRequests.map((r, i) => <li key={i}>{r.method} {r.url} ({r.bytes} B)</li>)}</ul>
            </>
          )}
          <p className="xsmall muted">Verify it yourself: open DevTools → Network while you upload and ask.</p>
        </div>
      </div>
    </div>
  );
}
