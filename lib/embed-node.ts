// Node-side embedder for the bench. Same model files and same pooling as the
// browser (lib/embed.ts), so bench features match what the app computes.
import { join } from 'node:path';
import { env, pipeline } from '@xenova/transformers';

export const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';

export async function nodeEmbedder() {
  env.allowRemoteModels = false;
  env.localModelPath = join(process.cwd(), 'public/models/');
  const fe = await pipeline('feature-extraction', EMBED_MODEL, { quantized: true });
  return async (texts: string[]): Promise<Float32Array[]> => {
    const out = await fe(texts, { pooling: 'mean', normalize: true });
    const dim = out.dims[1];
    return texts.map((_, i) => (out.data as Float32Array).slice(i * dim, (i + 1) * dim));
  };
}
