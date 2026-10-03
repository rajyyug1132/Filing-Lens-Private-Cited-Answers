import { splitSentences } from './cite';
import { BASE, importUrl } from './env';
import type { Hit } from './retrieve';
import { contentTerms } from './text';

export interface Engine {
  id: 'webllm' | 'wllama' | 'fixture' | 'extractive';
  label: string;
  generate(system: string, user: string, hits: Hit[], question: string, onText?: (t: string) => void): Promise<string>;
}

export interface LoadProgress {
  progress: number; // 0..1
  text: string;
}

// Primary: Qwen2.5-3B-Instruct, 4-bit, on WebGPU via WebLLM (bundled from npm).
// Fallback when the browser has no WebGPU: Qwen2.5-1.5B-Instruct Q4_K_M GGUF on
// WASM CPU via wllama. Only the weights are fetched at runtime (GET, once,
// cached by the browser); prompts and documents never leave the device.
export const MODELS = {
  webgpuF16: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  webgpuF32: 'Qwen2.5-3B-Instruct-q4f32_1-MLC',
  wasmGguf:
    process.env.NEXT_PUBLIC_WASM_MODEL_URL ??
    'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
};

export interface GpuCheck {
  webgpu: boolean;
  f16: boolean;
  detail: string;
}

export async function checkWebGPU(): Promise<GpuCheck> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<any> } }).gpu;
  if (!gpu) return { webgpu: false, f16: false, detail: 'WebGPU not available in this browser' };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { webgpu: false, f16: false, detail: 'No WebGPU adapter (GPU blocklisted or disabled)' };
    const f16 = adapter.features?.has?.('shader-f16') ?? false;
    return { webgpu: true, f16, detail: f16 ? 'WebGPU with f16 shaders' : 'WebGPU (no f16, using f32 weights)' };
  } catch (e) {
    return { webgpu: false, f16: false, detail: `WebGPU check failed: ${(e as Error).message}` };
  }
}

const GEN = { max_tokens: 200, temperature: 0.1, top_p: 0.9 };

export async function loadWebLLM(gpuCheck: GpuCheck, onProgress?: (p: LoadProgress) => void): Promise<Engine> {
  const { CreateMLCEngine } = await import('@mlc-ai/web-llm');
  const modelId = gpuCheck.f16 ? MODELS.webgpuF16 : MODELS.webgpuF32;
  const engine = await CreateMLCEngine(modelId, {
    initProgressCallback: (r) => onProgress?.({ progress: r.progress, text: r.text }),
  });
  return {
    id: 'webllm',
    label: 'Qwen2.5-3B on-device (WebGPU)',
    async generate(system, user, _hits, _q, onText) {
      const stream = await engine.chat.completions.create({
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: user },
        ],
        stream: true,
        ...GEN,
      });
      let text = '';
      for await (const chunk of stream) {
        text += chunk.choices[0]?.delta?.content ?? '';
        onText?.(text);
      }
      return text;
    },
  };
}

export async function loadWllama(onProgress?: (p: LoadProgress) => void): Promise<Engine> {
  type WllamaMod = { Wllama: new (paths: Record<string, string>, cfg?: Record<string, unknown>) => any };
  const { Wllama } = await importUrl<WllamaMod>(`${location.origin}${BASE}/vendor/wllama/index.js`);
  const w = new Wllama({ default: `${location.origin}${BASE}/vendor/wllama/wllama.wasm` }, { suppressNativeLog: true, parallelDownloads: 3 });
  await w.loadModelFromUrl(MODELS.wasmGguf, {
    n_ctx: 4096,
    n_gpu_layers: 0,
    progressCallback: ({ loaded, total }: { loaded: number; total: number }) =>
      onProgress?.({ progress: total ? loaded / total : 0, text: `Downloading ${Math.round(loaded / 1e6)} / ${Math.round(total / 1e6)} MB` }),
  });
  return {
    id: 'wllama',
    label: 'Qwen2.5-1.5B on-device (WASM CPU)',
    async generate(system, user, _hits, _q, onText) {
      let text = '';
      const res = await w.createChatCompletion({
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: user },
        ],
        stream: true,
        onChunk: (c: any) => {
          text += c?.choices?.[0]?.delta?.content ?? '';
          onText?.(text);
        },
        ...GEN,
      });
      return text || res?.choices?.[0]?.message?.content || '';
    },
  };
}

// Test stub: replays responses from a fixture file keyed by question. Used by
// the Playwright suite (?engine=fixture), so tests exercise the real
// citation check and abstain paths without a 2 GB download.
export async function loadFixtureEngine(): Promise<Engine> {
  const res = await fetch(`${BASE}/fixtures/recorded-responses.json`);
  const data = (await res.json()) as { source: string; responses: Record<string, string> };
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const table = new Map(Object.entries(data.responses).map(([q, a]) => [norm(q), a]));
  return {
    id: 'fixture',
    label: `Recorded responses (${data.source})`,
    async generate(_s, _u, _h, question, onText) {
      const a = table.get(norm(question)) ?? 'NOT IN DOCUMENT';
      onText?.(a);
      return a;
    },
  };
}

// No-model fallback: quotes the best-matching sentences verbatim, each tagged
// with its page. Labelled as such in the UI.
export const extractiveEngine: Engine = {
  id: 'extractive',
  label: 'Extractive preview (no model downloaded)',
  async generate(_s, _u, hits, question) {
    const q = new Set(contentTerms(question));
    const scored = hits.flatMap((h, rank) =>
      splitSentences(h.chunk.text).map((s) => {
        const terms = contentTerms(s);
        const overlap = terms.filter((t) => q.has(t)).length;
        return { s, page: h.chunk.page, score: overlap / Math.sqrt(1 + terms.length) - rank * 0.01 };
      }),
    );
    scored.sort((a, b) => b.score - a.score);
    const best = scored.filter((x) => x.score > 0).slice(0, 2);
    if (!best.length) return 'NOT IN DOCUMENT';
    return best.map((b) => `${b.s.length > 320 ? b.s.slice(0, 317) + '…' : b.s} [p.${b.page}]`).join(' ');
  },
};
