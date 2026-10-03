import type { ChatMessage } from './dspy-chat';
import { BASE, importUrl } from './env';
import type { Snippet } from './snippets';
import { contentTerms } from './text';

// An LLM engine takes the exact ChatAdapter messages (lib/dspy-chat.ts) and
// returns the raw completion. The extractive engine has no model: the pipeline
// skips the verifier and quotes snippets instead.
export interface Engine {
  id: 'webllm' | 'wllama' | 'fixture' | 'extractive';
  label: string;
  chat?(messages: ChatMessage[], maxTokens: number, onText?: (t: string) => void): Promise<string>;
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

type Chunk = { choices?: { delta?: { content?: string | null } }[] };

// Concatenate the delta text of an OpenAI-style chunk stream (WebLLM and wllama both yield these).
export async function drainStream(stream: AsyncIterable<Chunk>, onText?: (t: string) => void): Promise<string> {
  let text = '';
  for await (const chunk of stream) {
    text += chunk?.choices?.[0]?.delta?.content ?? '';
    onText?.(text);
  }
  return text;
}

const GEN = { temperature: 0, top_p: 1 }; // greedy, as in the Kaggle eval

export async function loadWebLLM(gpuCheck: GpuCheck, onProgress?: (p: LoadProgress) => void): Promise<Engine> {
  const { CreateMLCEngine } = await import('@mlc-ai/web-llm');
  const modelId = gpuCheck.f16 ? MODELS.webgpuF16 : MODELS.webgpuF32;
  const engine = await CreateMLCEngine(modelId, {
    initProgressCallback: (r) => onProgress?.({ progress: r.progress, text: r.text }),
  });
  return {
    id: 'webllm',
    label: 'Qwen2.5-3B on-device (WebGPU)',
    async chat(messages, maxTokens, onText) {
      const stream = await engine.chat.completions.create({ messages, stream: true, max_tokens: maxTokens, ...GEN });
      return drainStream(stream, onText);
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
    async chat(messages, maxTokens, onText) {
      // stream:true without onData returns an async iterator of OpenAI-style chunks
      const stream = await w.createChatCompletion({ messages, max_tokens: maxTokens, stream: true, ...GEN });
      return drainStream(stream, onText);
    },
  };
}

// Test stub: replays recorded ChatAdapter completions keyed by question and
// predictor (check = verifier, answer = cited answerer). Used by the Playwright
// suite (?engine=fixture), so tests run the real gate, prompt builder, parser,
// citation check and abstain paths without a 2 GB download.
export async function loadFixtureEngine(): Promise<Engine> {
  const res = await fetch(`${BASE}/fixtures/recorded-responses.json`);
  const data = (await res.json()) as { source: string; responses: Record<string, { check: string; answer?: string }> };
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const table = new Map(Object.entries(data.responses).map(([q, a]) => [norm(q), a]));
  const ABSTAIN = '[[ ## verdict ## ]]\nABSTAIN\n\n[[ ## evidence_page ## ]]\nNONE\n\n[[ ## completed ## ]]';
  return {
    id: 'fixture',
    label: `Recorded responses (${data.source})`,
    async chat(messages, _max, onText) {
      const last = messages[messages.length - 1].content;
      const q = last.match(/\[\[ ## question ## \]\]\n([^\n]*)/)?.[1] ?? '';
      const isCheck = messages[0].content.includes('`evidence_page`');
      const rec = table.get(norm(q));
      const out = isCheck ? (rec?.check ?? ABSTAIN) : (rec?.answer ?? '[[ ## answer ## ]]\n\n[[ ## completed ## ]]');
      onText?.(out);
      return out;
    },
  };
}

// No-model fallback (gate only, no verifier): quotes the two snippets that
// overlap the question most, each tagged with its page. Labelled in the UI.
export const extractiveEngine: Engine = { id: 'extractive', label: 'Extractive preview (no model downloaded)' };

export function extractiveAnswer(question: string, snippets: Snippet[]): string {
  const q = new Set(contentTerms(question));
  const best = snippets
    .map((s, i) => {
      const terms = contentTerms(s.text);
      return { s, score: terms.filter((t) => q.has(t)).length / Math.sqrt(1 + terms.length) - i * 0.001 };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 2);
  return best.map(({ s }) => `${s.text} [p.${s.page}]`).join(' ');
}
