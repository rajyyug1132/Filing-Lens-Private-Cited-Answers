// Export the exact prompt the browser uses, so the Kaggle generation run
// (bench/kaggle_gen.ipynb) prompts Qwen identically.
import { writeFileSync, mkdirSync } from 'node:fs';
import { SYSTEM_PROMPT, REFUSAL, buildUserPrompt } from '../lib/prompt';
mkdirSync('bench/results', { recursive: true });
const template = buildUserPrompt('{question}', [{ chunk: { id: 0, page: 0, text: '{text}' }, dense: 0, bm25: 0, rrf: 0 }]);
writeFileSync('bench/results/prompt.json', JSON.stringify({ system: SYSTEM_PROMPT, refusal: REFUSAL, user_template: template, excerpt_format: '[p.{page}] {text}', excerpt_joiner: '\n\n' }, null, 2));
console.log(template);
