import assert from 'node:assert/strict';
import { test } from 'node:test';
import { drainStream } from '../../lib/llm';

test('drainStream concatenates streamed deltas (WebLLM / wllama async iterators)', async () => {
  async function* chunks() {
    yield { choices: [{ delta: { content: '[[ ## verdict ## ]]\n' } }] };
    yield { choices: [{ delta: {} }] };
    yield { choices: [{ delta: { content: 'ANSWER' } }] };
  }
  const seen: string[] = [];
  assert.equal(await drainStream(chunks(), (t) => seen.push(t)), '[[ ## verdict ## ]]\nANSWER');
  assert.equal(seen.at(-1), '[[ ## verdict ## ]]\nANSWER');
});
