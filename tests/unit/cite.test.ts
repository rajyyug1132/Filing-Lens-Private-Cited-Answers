import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkCitations, parseCitations } from '../../lib/cite';

test('parses common citation spellings', () => {
  assert.deepEqual(parseCitations('A [p.12]. B [p. 3, p.4]. C [page 7]. D [p12]'), [12, 3, 4, 7, 12]);
});

test('accepts an answer whose citations are all in context', () => {
  const r = checkCitations('Operating cash flow was $1,824 million [p.42]. It fell from $3,252 million [p.42].', [42, 51]);
  assert.equal(r.ok, true);
  assert.deepEqual(r.cited, [42]);
  assert.equal(r.coverage, 1);
});

test('rejects citations to pages that were not retrieved', () => {
  const r = checkCitations('Revenue was $46 billion [p.99].', [42, 51]);
  assert.equal(r.ok, false);
  assert.deepEqual(r.invalid, [99]);
});

test('rejects uncited answers', () => {
  assert.equal(checkCitations('Revenue was $46 billion.', [42]).ok, false);
});
