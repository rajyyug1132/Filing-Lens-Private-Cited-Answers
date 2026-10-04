import assert from 'node:assert/strict';
import { test } from 'node:test';
import { caseLine } from '../../lib/case-line';

test('caseLine reads form, fiscal year and company from the file name', () => {
  assert.equal(caseLine('Best Buy FY2023 10-K (sample)', 75), 'BEST BUY · 10-K · FY2023 · 75 PP · INDEXED');
  assert.equal(caseLine('BESTBUY_2023_10K.pdf', 75), 'BESTBUY · 10-K · FY2023 · 75 PP · INDEXED');
  assert.equal(caseLine('notes.pdf', 3), 'NOTES · 3 PP · INDEXED');
});
