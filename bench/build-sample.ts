// Pre-index the bundled sample filing (Best Buy FY2023 10-K, from FinanceBench)
// with the same chunker and embedding model the browser uses, so judges can
// try the app without uploading anything or waiting for on-device indexing.
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';
import { nodeEmbedder } from '../lib/embed-node';
import { indexPages } from '../lib/index-doc';
import { extractPages } from '../lib/pdf-text';
import { packVectorsNode } from './pack';

const SRC = 'tests/fixtures/BESTBUY_2023_10K.pdf';
const OUT = 'public/samples/bestbuy-2023';
mkdirSync(OUT, { recursive: true });
const task = getDocument({ data: new Uint8Array(readFileSync(SRC)), useSystemFonts: true, verbosity: 0 });
const pages = await extractPages((await task.promise) as never);
await task.destroy();
const idx = await indexPages(pages, await nodeEmbedder());
const { buffer, dim } = packVectorsNode(idx.vectors);
writeFileSync(`${OUT}/vectors.f32`, buffer);
writeFileSync(`${OUT}/index.json`, JSON.stringify({ name: 'Best Buy FY2023 10-K (sample)', pages: pages.length, dim, chunks: idx.chunks }));
copyFileSync(SRC, `${OUT}/filing.pdf`);
console.log(`sample: ${pages.length} pages, ${idx.chunks.length} chunks`);
