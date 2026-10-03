import { buildBm25 } from './bm25';
import { chunkPages, type Page } from './chunk';
import type { DocIndex } from './retrieve';
import { tokenize } from './text';

export type Embedder = (texts: string[]) => Promise<Float32Array[]>;

export async function indexPages(
  pages: Page[],
  embed: Embedder,
  onProgress?: (done: number, total: number) => void,
  batch = 32,
): Promise<DocIndex> {
  const chunks = chunkPages(pages);
  const vectors: Float32Array[] = [];
  for (let i = 0; i < chunks.length; i += batch) {
    vectors.push(...(await embed(chunks.slice(i, i + batch).map((c) => c.text))));
    onProgress?.(Math.min(i + batch, chunks.length), chunks.length);
  }
  return hydrate(chunks, vectors);
}

// Rebuild the non-persisted parts (BM25, vocab) from chunks + vectors.
export function hydrate(chunks: DocIndex['chunks'], vectors: Float32Array[]): DocIndex {
  const texts = chunks.map((c) => c.text);
  return { chunks, vectors, bm25: buildBm25(texts), vocab: new Set(texts.flatMap(tokenize)) };
}
