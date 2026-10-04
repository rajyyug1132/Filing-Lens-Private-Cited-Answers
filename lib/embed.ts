import { BASE } from './env';

let worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, { resolve: (v: Float32Array[]) => void; reject: (e: Error) => void }>();

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(`${BASE}/workers/embed.js`, { type: 'module' });
  worker.onmessage = (e: MessageEvent) => {
    const { id, error, dim, buffer } = e.data;
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    if (error) return p.reject(new Error(error));
    const flat = new Float32Array(buffer);
    const out: Float32Array[] = [];
    for (let i = 0; i < flat.length / dim; i++) out.push(flat.slice(i * dim, (i + 1) * dim));
    p.resolve(out);
  };
  const fail = () => {
    const err = new Error('The on-device search worker could not start. Reload the page and try again.');
    for (const p of pending.values()) p.reject(err);
    pending.clear();
    worker?.terminate();
    worker = null; // the next question starts a fresh worker
  };
  worker.onerror = fail;
  worker.onmessageerror = fail;
  return worker;
}

export function embed(texts: string[]): Promise<Float32Array[]> {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, base: location.origin + BASE, texts });
  });
}
