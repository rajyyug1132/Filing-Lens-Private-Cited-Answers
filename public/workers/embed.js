// Embedding worker. Model weights, tokenizer and ONNX runtime are all served
// from this app's own origin (see scripts/vendor-assets.mjs); remote model
// fetching is switched off, so no text ever leaves the device from here.
let extractor = null;

async function load(base) {
  const T = await import(`${base}/vendor/transformers/transformers.min.js`);
  T.env.allowRemoteModels = false;
  T.env.allowLocalModels = true;
  T.env.localModelPath = `${base}/models/`;
  T.env.backends.onnx.wasm.wasmPaths = `${base}/vendor/transformers/`;
  T.env.backends.onnx.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  extractor = await T.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { quantized: true });
}

self.onmessage = async (e) => {
  const { id, base, texts } = e.data;
  try {
    if (!extractor) await load(base);
    const out = await extractor(texts, { pooling: 'mean', normalize: true });
    const flat = new Float32Array(out.data);
    self.postMessage({ id, dim: out.dims[1], buffer: flat.buffer }, [flat.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message ? err.message : err) });
  }
};
