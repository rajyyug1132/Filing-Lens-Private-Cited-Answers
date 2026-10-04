import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { classifyDoc } from '../../lib/doc-type';
import { deriveCandidates, parseQuestions } from '../../lib/suggest';
import { SYLLABUS_PAGES } from '../fixtures/syllabus';

const syllabus = SYLLABUS_PAGES.map((text, i) => ({ page: i + 1, text }));
const sample = JSON.parse(readFileSync('public/samples/bestbuy-2023/index.json', 'utf8')).chunks as { page: number; text: string }[];

test('doc type: the Best Buy 10-K is FILING, the syllabus is GENERAL', () => {
  assert.equal(classifyDoc(sample), 'FILING');
  assert.equal(classifyDoc(syllabus), 'GENERAL');
});

test('syllabus candidates come from its own text and keep their source page', () => {
  const c = deriveCandidates(syllabus, 'GENERAL');
  assert.ok(c.length >= 4 && c.length <= 6, `got ${c.length}`);
  const tb = c.find((x) => x.q === 'Which textbooks are listed?');
  assert.ok(tb, 'textbooks question derived');
  assert.equal(tb!.page, 9);
  assert.ok(c.every((x) => !/tesla|cybertruck|operating activities/i.test(x.q)));
});

test('filing candidates exist and no list-title templates leak into filings', () => {
  const pages = new Map<number, string>();
  for (const ch of sample) pages.set(ch.page, (pages.get(ch.page) ?? '') + ' ' + ch.text);
  const c = deriveCandidates([...pages].map(([page, text]) => ({ page, text })), 'FILING');
  assert.ok(c.length >= 2, `got ${c.length}`);
  assert.ok(!c.some((x) => x.kind === 'list'));
});

test('no hardcoded suggestion chips remain in the app', () => {
  const app = readFileSync('components/App.tsx', 'utf8');
  assert.ok(!/SUGGESTIONS\s*=|Cybertruck|What are the main risk factors\?/.test(app));
});

test('parseQuestions keeps question lines only', () => {
  assert.deepEqual(parseQuestions('1. What is CIE weightage?\nnoise\n- Which modules cover paging?'), ['What is CIE weightage?', 'Which modules cover paging?']);
});

test('general gate: tuned threshold blocks the wrong-subject question; accuracy is reported, not assumed', () => {
  const r = JSON.parse(readFileSync('bench/results/general_gate.json', 'utf8'));
  const dbms = r.rows.find((x: { q: string }) => x.q === 'What is the DBMS syllabus?');
  assert.equal(dbms.pass, false);
  assert.ok(r.rows.find((x: { q: string }) => x.q === 'Which textbooks are listed?').pass);
  assert.ok(typeof r.accuracy_cv5 === 'number');
});
