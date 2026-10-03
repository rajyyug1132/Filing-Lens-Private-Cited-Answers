import assert from 'node:assert/strict';
import { test } from 'node:test';
import { chunkPages } from '../../lib/chunk';
import { hydrate } from '../../lib/index-doc';
import { FEATURE_NAMES, features, retrieve } from '../../lib/retrieve';

const pages = [
  { page: 1, text: 'Risk factors include competition and supply chain disruption.' },
  { page: 2, text: 'Total cash provided by operating activities was 1,824 in fiscal 2023.' },
];

test('chunks never cross page boundaries', () => {
  const long = { page: 3, text: Array.from({ length: 400 }, (_, i) => `w${i}`).join(' ') };
  const cs = chunkPages([...pages, long]);
  assert.ok(cs.filter((c) => c.page === 3).length >= 3);
  for (const c of cs) assert.ok([1, 2, 3].includes(c.page));
});

test('BM25 side of hybrid retrieval finds the right page; features have fixed arity', () => {
  const chunks = chunkPages(pages);
  const vecs = chunks.map(() => new Float32Array([1, 0]));
  const idx = hydrate(chunks, vecs);
  const r = retrieve(idx, 'cash provided by operating activities 2023', new Float32Array([1, 0]), 2);
  assert.equal(r.hits[0].chunk.page, 2);
  const f = features(idx, r);
  assert.equal(f.length, FEATURE_NAMES.length);
  assert.equal(f[11], 1, "the year asked about is in the top passage");
  assert.equal(f[7], 0, 'year 2023 is present in context');
});

test('properNouns finds the company a question names, not boilerplate', async () => {
  const { properNouns } = await import('../../lib/text');
  assert.deepEqual(properNouns("What was Walmart's capital expenditure in FY2023?"), ['walmart']);
  assert.deepEqual(properNouns('What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response.'), ['3m']);
  assert.deepEqual(properNouns('How much cash did operating activities provide?'), []);
});
