// IndexedDB persistence: the PDF bytes, its chunks and their vectors stay in
// this browser profile. Nothing here is ever sent anywhere.
import { openDB, type IDBPDatabase } from 'idb';
import type { Chunk } from './chunk';

export interface StoredDoc {
  id: string;
  name: string;
  pages: number;
  chunks: Chunk[];
  vectors: ArrayBuffer; // chunks.length * dim float32
  dim: number;
  bytes: ArrayBuffer; // original PDF, for rendering cited pages
  createdAt: number;
}

let dbp: Promise<IDBPDatabase> | null = null;
const db = () => (dbp ??= openDB('filing-lens', 1, { upgrade: (d) => d.createObjectStore('docs', { keyPath: 'id' }) }));

export async function saveDoc(doc: StoredDoc) {
  await (await db()).put('docs', doc);
}
export async function loadDoc(id: string): Promise<StoredDoc | undefined> {
  return (await db()).get('docs', id);
}
export async function listDocs(): Promise<Pick<StoredDoc, 'id' | 'name' | 'pages' | 'createdAt'>[]> {
  const all = (await (await db()).getAll('docs')) as StoredDoc[];
  return all.map(({ id, name, pages, createdAt }) => ({ id, name, pages, createdAt })).sort((a, b) => b.createdAt - a.createdAt);
}
export async function deleteDoc(id: string) {
  await (await db()).delete('docs', id);
}

export function packVectors(vs: Float32Array[]): { buffer: ArrayBuffer; dim: number } {
  const dim = vs[0]?.length ?? 384;
  const flat = new Float32Array(vs.length * dim);
  vs.forEach((v, i) => flat.set(v, i * dim));
  return { buffer: flat.buffer, dim };
}
export function unpackVectors(buffer: ArrayBuffer, dim: number): Float32Array[] {
  const flat = new Float32Array(buffer);
  return Array.from({ length: flat.length / dim }, (_, i) => flat.subarray(i * dim, (i + 1) * dim));
}
