import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { formatMessages, parseOutput, type PredictorSpec } from '../../lib/dspy-chat';

type Ref = { inputs: Record<string, string>; messages: { role: string; content: string }[] };
type Exported = { predictors: Record<string, PredictorSpec & { references: Ref[] }> };
const load = (p: string) => JSON.parse(readFileSync(p, 'utf8'));

// Every reference was rendered by DSPy's own ChatAdapter (bench/export_verifier.py).
function assertSame(spec: PredictorSpec & { references: Ref[] }, label: string) {
  assert.ok(spec.references.length > 0, `${label}: no references`);
  for (const ref of spec.references) assert.deepEqual(formatMessages(spec, ref.inputs), ref.messages, label);
}

test('browser prompt equals the DSPy-compiled prompt (shipped app/prompts/verifier.json)', () => {
  const v: Exported = load('app/prompts/verifier.json');
  for (const [name, spec] of Object.entries(v.predictors)) assertSame(spec, `shipped:${name}`);
});

test('export path: notebook export with demos (dry run) round-trips exactly', () => {
  const v: Exported = load('tests/fixtures/verifier-dryrun-export.json');
  for (const [name, spec] of Object.entries(v.predictors)) {
    assert.ok(spec.demos.length > 0, `${name} should carry demos`);
    assertSame(spec, `dryrun:${name}`);
  }
});

test('DSPy edge cases: multi-line instructions, incomplete demo, field without description', () => {
  for (const c of load('tests/fixtures/dspy-edge-cases.json').cases) assertSame(c, 'edge');
});

test('parseOutput reads ChatAdapter-style completions', () => {
  const spec = (load('app/prompts/verifier.json') as Exported).predictors.check;
  assert.deepEqual(parseOutput(spec, '[[ ## verdict ## ]]\nANSWER\n\n[[ ## evidence_page ## ]]\np.42\n\n[[ ## completed ## ]]'), { verdict: 'ANSWER', evidence_page: 'p.42' });
  assert.deepEqual(parseOutput(spec, 'garbage'), { verdict: null, evidence_page: null });
});
