// Copy every runtime asset the browser needs into public/, so the deployed app
// serves them from its own origin. Nothing at runtime comes from a CDN except
// the 3B model weights (one-time download, cached by the browser).
import { cpSync, existsSync, mkdirSync } from 'node:fs';

const copies = [
  ['node_modules/pdfjs-dist/legacy/build/pdf.min.mjs', 'public/vendor/pdfjs/pdf.min.mjs'],
  ['node_modules/pdfjs-dist/legacy/build/pdf.worker.min.mjs', 'public/vendor/pdfjs/pdf.worker.min.mjs'],
  ['node_modules/@xenova/transformers/dist/transformers.min.js', 'public/vendor/transformers/transformers.min.js'],
  ['node_modules/@xenova/transformers/dist/ort-wasm-simd.wasm', 'public/vendor/transformers/ort-wasm-simd.wasm'],
  ['node_modules/@xenova/transformers/dist/ort-wasm-simd-threaded.wasm', 'public/vendor/transformers/ort-wasm-simd-threaded.wasm'],
  ['node_modules/@wllama/wllama/esm/index.js', 'public/vendor/wllama/index.js'],
  ['node_modules/@wllama/wllama/esm/wasm/wllama.wasm', 'public/vendor/wllama/wllama.wasm'],
  ['node_modules/@ryanstark24/sfgraph-models/data/Xenova', 'public/models/Xenova'],
];

for (const [from, to] of copies) {
  if (!existsSync(from)) {
    console.warn(`[vendor-assets] missing ${from}`);
    continue;
  }
  mkdirSync(to.split('/').slice(0, -1).join('/'), { recursive: true });
  cpSync(from, to, { recursive: true });
}
console.log('[vendor-assets] copied', copies.length, 'assets into public/');
