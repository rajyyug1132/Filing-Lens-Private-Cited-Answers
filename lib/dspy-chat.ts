// A TypeScript port of DSPy 3.x ChatAdapter.format / parse for string-typed
// signatures, so the browser sends the 3B model exactly the prompt the DSPy
// program was compiled with (instructions + demos from app/prompts/verifier.json).
// Equality with DSPy's own rendering is checked by tests/unit/dspy-chat.test.ts
// against references rendered by real DSPy (bench/export_verifier.py).

export interface FieldSpec {
  name: string;
  desc: string; // '' when DSPy's default "${name}" placeholder is used
}

export interface PredictorSpec {
  instructions: string;
  inputs: FieldSpec[];
  outputs: FieldSpec[];
  demos: Record<string, string | null>[];
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

const pyStrip = (s: string) => s.replace(/^\s+|\s+$/g, '');

function dedent(text: string): string {
  const lines = text.split('\n');
  const indents = lines.filter((l) => l.trim()).map((l) => l.match(/^[ \t]*/)![0].length);
  const n = indents.length ? Math.min(...indents) : 0;
  return lines.map((l) => (l.trim() ? l.slice(n) : l.replace(/^[ \t]+$/, ''))).join('\n');
}

// Python str.splitlines() on the separators that matter here.
const splitLines = (s: string) => (s === '' ? [] : s.replace(/\r\n?/g, '\n').replace(/\n$/, '').split('\n'));

function fieldDescription(fields: FieldSpec[]): string {
  return pyStrip(fields.map((f, i) => `${i + 1}. \`${f.name}\` (str): ${f.desc}`).join('\n'));
}

function withValues(pairs: [string, string][]): string {
  return pyStrip(pairs.map(([k, v]) => `[[ ## ${k} ## ]]\n${v}`).join('\n\n'));
}

export function systemMessage(sig: PredictorSpec): string {
  const desc = `Your input fields are:\n${fieldDescription(sig.inputs)}\nYour output fields are:\n${fieldDescription(sig.outputs)}`;
  const structure = pyStrip(
    [
      'All interactions will be structured in the following way, with the appropriate values filled in.',
      withValues(sig.inputs.map((f) => [f.name, `{${f.name}}`])),
      withValues(sig.outputs.map((f) => [f.name, `{${f.name}}`])),
      '[[ ## completed ## ]]\n',
    ].join('\n\n'),
  );
  const objective = ['', ...splitLines(dedent(sig.instructions))].join('\n' + ' '.repeat(8));
  return [desc, structure, `In adhering to this structure, your objective is: ${objective}`].join('\n');
}

function userContent(sig: PredictorSpec, values: Record<string, string | null>, prefix = '', mainRequest = false): string {
  const parts = [prefix];
  for (const f of sig.inputs) if (f.name in values) parts.push(`[[ ## ${f.name} ## ]]\n${values[f.name]}`);
  if (mainRequest)
    parts.push(
      'Respond with the corresponding output fields, starting with the field ' +
        sig.outputs.map((f) => `\`[[ ## ${f.name} ## ]]\``).join(', then ') +
        ', and then ending with the marker for `[[ ## completed ## ]]`.',
    );
  parts.push('');
  return pyStrip(parts.join('\n\n'));
}

function assistantContent(sig: PredictorSpec, values: Record<string, string | null>, missing: string): string {
  return withValues(sig.outputs.map((f) => [f.name, values[f.name] ?? missing])) + '\n\n[[ ## completed ## ]]\n';
}

export function formatMessages(sig: PredictorSpec, inputs: Record<string, string>): ChatMessage[] {
  const all = [...sig.inputs, ...sig.outputs].map((f) => f.name);
  const complete = sig.demos.filter((d) => all.every((k) => k in d && d[k] != null));
  const incomplete = sig.demos.filter(
    (d) => !complete.includes(d) && sig.inputs.some((f) => f.name in d) && sig.outputs.some((f) => f.name in d),
  );
  const msgs: ChatMessage[] = [{ role: 'system', content: systemMessage(sig) }];
  for (const d of incomplete) {
    msgs.push({ role: 'user', content: userContent(sig, d, 'This is an example of the task, though some input or output fields are not supplied.') });
    msgs.push({ role: 'assistant', content: assistantContent(sig, d, 'Not supplied for this particular example. ') });
  }
  for (const d of complete) {
    msgs.push({ role: 'user', content: userContent(sig, d) });
    msgs.push({ role: 'assistant', content: assistantContent(sig, d, 'Not supplied for this conversation history message. ') });
  }
  msgs.push({ role: 'user', content: userContent(sig, inputs, '', true) });
  return msgs;
}

// ChatAdapter.parse: split on [[ ## field ## ]] headers; missing fields -> null.
export function parseOutput(sig: PredictorSpec, completion: string): Record<string, string | null> {
  const out: Record<string, string | null> = Object.fromEntries(sig.outputs.map((f) => [f.name, null]));
  let cur: string | null = null;
  let buf: string[] = [];
  const flush = () => {
    if (cur && cur in out && out[cur] === null) out[cur] = pyStrip(buf.join('\n'));
  };
  for (const line of completion.split('\n')) {
    const m = line.trim().match(/^\[\[ ## (\w+) ## \]\](.*)$/);
    if (m) {
      flush();
      cur = m[1];
      buf = m[2].trim() ? [m[2].trim()] : [];
    } else buf.push(line);
  }
  flush();
  return out;
}
