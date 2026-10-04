// Builds deck/FilingLens.pdf (8 slides, 16:9) from live artifacts:
// bench/results/{results.json, cascade_metrics.json, recall.svg}, demo
// screenshots and the e2e privacy log. No number on a slide is typed by hand.
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';

const R = JSON.parse(readFileSync('bench/results/results.json', 'utf8'));
const C = JSON.parse(readFileSync('bench/results/cascade_metrics.json', 'utf8'));
const recallSvg = readFileSync('bench/results/recall.svg', 'utf8');
const priv = existsSync('demo/privacy-log.json') ? JSON.parse(readFileSync('demo/privacy-log.json', 'utf8')) : null;
const img = (p) => (existsSync(p) ? `data:image/png;base64,${readFileSync(p).toString('base64')}` : '');
const f3 = (v) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(3));
const ask = (s) => `<span class="ask">[ASK: ${s}]</span>`;
const shipped = R.results.find((r) => r.key === 'lrTemp');
const gate = C.rows['Feature gate only'];
const ba = R.beforeAfter;
const H = JSON.parse(readFileSync('bench/results/history.json', 'utf8')).walmart_fix;
const nFeat = JSON.parse(readFileSync('lib/decision-model.json', 'utf8')).weights.length;
const V = JSON.parse(readFileSync('app/prompts/verifier.json', 'utf8'));

const cascadeRows = Object.entries(C.rows)
  .map(([name, m]) => {
    const lat = name.startsWith('Feature') ? `${m.latency_p50_ms.toFixed(2)} ms` : `${Math.round(m.latency_p50_ms)} ms`;
    return `<tr class="${name.startsWith('Cascade') ? 'hl' : ''}"><td>${name}</td><td>${f3(m.easy_acc)}</td><td>${f3(m.hard_acc)}</td><td>${f3(m.easy_ece)} / ${f3(m.hard_ece)}</td><td>${f3(m.citation_page_match)}${m.page_source ? '*' : ''}</td><td>${lat}</td><td>${Math.round(m.verifier_calls_pct)}%</td></tr>`;
  })
  .join('');
const pendingRows =
  C.status === 'pending'
    ? `<tr><td>Verifier only (DSPy-compiled 3B)</td><td colspan="5" class="pending">pending (Kaggle)</td><td>100%</td></tr>` +
      `<tr class="hl"><td>Cascade: gate → verifier</td><td colspan="5" class="pending">pending (Kaggle)</td><td>${Math.round(C.cascade_verifier_calls_pct)}%</td></tr>`
    : '';

const slides = [
  `<h1>Filing Lens</h1><h2>Cited answers from company filings, from a local 3B model (Qwen 2.5, runs on the phone)</h2>
   <div class="cols"><div><p class="lead">Retail investors are handed 100-page 10-Ks. Chatbots answer anyway, with no page reference, and the filing gets uploaded to someone else's server.</p>
   <p>FinanceBench (Islam et al., 2023, arXiv:2311.11944): GPT-4-Turbo with a retrieval system <b>incorrectly answered or refused 81%</b> of 150 open-book filing questions.</p>
   <p class="kicker">Problem: wrong answers look the same as right ones, and privacy is the price of asking.</p></div></div>`,
  `<h1>Demo</h1><div class="shots">${['demo/answer.png', 'demo/page.png', 'demo/abstain.png'].map((p, i) => (img(p) ? `<figure><img src="${img(p)}"/><figcaption>${['Cited answer + calibrated confidence', 'Citation chip opens the page', 'Wrong-company question → abstain'][i]}</figcaption></figure>` : '')).join('')}</div>
   <p class="small">Recorded in headless Chromium at Pixel-7 size with recorded model responses (the build machine can't download weights). Phone video: ${ask('phone recording link')}.</p>`,
  `<h1>Privacy proof</h1><div class="cols"><div>
   <p class="lead">The document never leaves the device. The build's e2e test enforces this; it isn't just claimed.</p>
   ${priv ? `<table class="kv"><tr><td>Requests during upload → index → ask → abstain</td><td><b>${priv.total}</b></td></tr><tr><td>to any host other than the app's own origin</td><td><b>${priv.external}</b></td></tr><tr><td>carrying a request body (POST/PUT/beacon)</td><td><b>${priv.withBody}</b></td></tr></table>` : ask('privacy log missing')}
   <ul><li>pdf.js, MiniLM embeddings, ONNX runtime, wllama: npm packages served from the app origin, no external scripts</li><li>Index + PDF stored in IndexedDB</li><li>Only third-party fetch: the Qwen2.5 weights (GET, once, then cached). Airplane mode after that: ${ask('confirm on phone')}</li></ul></div>
   ${img('demo/privacy.png') ? `<img class="phone" src="${img('demo/privacy.png')}"/>` : ''}</div>`,
  `<h1>Architecture: a two-stage cascade</h1><div class="arch">
   <div class="flow"><span>PDF</span>→<span>pdf.js page text</span>→<span>page-bounded chunks</span>→<span>MiniLM embeddings</span>→<span>IndexedDB</span></div>
   <div class="flow"><span>Question</span>→<span>hybrid retrieval<br/>dense + BM25, k=${R.k}, ≤${R.tokenBudget} tok</span>→<span class="key">① feature gate<br/>${nFeat} features → LR ÷ T<br/>${gate.latency_p50_ms.toFixed(2)} ms</span>→<span class="key2">② DSPy verifier on the 3B<br/>ANSWER / ABSTAIN + evidence page</span>→<span>DSPy cited answer<br/>Qwen2.5-3B, WebGPU</span>→<span>[p.N] check</span></div>
   <div class="stops"><span>p &lt; 0.5 → abstain, 3B never runs</span><span>verdict ABSTAIN → abstain</span><span>citation outside snippets → abstain</span></div>
   <p class="small">① is cheap and catches wrong-company questions (it scores retrieval and word coverage, not content). ② reads the actual sentences: it is a DSPy program, ${V.compiled ? `compiled with ${V.optimizer} on ${V.model}` : 'shipped uncompiled for now (0 demos; the Kaggle compile with BootstrapFewShot, 2 demos, is pending)'}, exported to <code>app/prompts/verifier.json</code> and rebuilt in the browser byte-for-byte. A unit test checks this against DSPy's own rendering. ${C.status === 'pending' ? `In the cascade, ${Math.round(C.cascade_verifier_calls_pct)}% of test questions reach ②.` : ''}</p></div>`,
  `<h1>Bench: gate vs verifier vs cascade</h1>
   <p class="small">${R.dataset}. Gold answer = question on its own filing; <b>Easy</b> abstain = another company's 10-K; <b>Hard</b> abstain = own filing with the gold pages (and any page still holding the gold numbers) removed; ${R.hardSet.excludedAnswerStillRetrieved} questions excluded where the answer was still retrieved. Test split (company-grouped), ${C.instances} instances.</p>
   <div class="cols bench"><div class="tcol"><table class="t"><thead><tr><th>System</th><th>Easy acc.</th><th>Hard acc.</th><th>ECE E / H</th><th>Cite page</th><th>p50</th><th>3B calls</th></tr></thead><tbody>${cascadeRows}${pendingRows}</tbody></table>
   <p class="small">* gate proposes no page: top-ranked retrieved page. Gate p50 measured in Node on ${R.machine}; verifier p50: one verifier call on a Kaggle T4 (llama.cpp; vLLM failed to start).</p>
   <p class="small">Gate, all ${R.sets.answer + R.sets.easyAbstain + R.sets.hardAbstain} out-of-fold: Easy acc ${f3(shipped.easy.accuracy)}, abstain P/R ${f3(shipped.easy.abstainPrecision)} / ${f3(shipped.easy.abstainRecall)}; Hard acc ${f3(shipped.hard.accuracy)}.${ba ? ` Two "does the top passage answer this?" features: Hard ${f3(ba.v1.hard.accuracy)} → ${f3(ba.v2.hard.accuracy)}.` : ''}</p></div>
   <div class="plot">${recallSvg}</div></div>`,
  `<h1>Limits (measured)</h1><ul class="big">
   <li><b>The feature gate can't see content: Hard ${f3(shipped.hard.accuracy)}</b> (always answering scores ${f3(R.results[0].hard.accuracy)} on this set). It scores retrieval and word overlap, so "the gold page" and "a nearby page about the same metric" look alike. The verifier stage was meant for that case. ${C.status === 'real' ? `It didn't help: the compiled 3B verifier abstains on almost everything (Hard ${f3(C.rows['Verifier only (gguf_q4)'].hard_acc)}, Easy ${f3(C.rows['Verifier only (gguf_q4)'].easy_acc)} on Q4; auto-graded answer accuracy ${f3(C.generation.gguf_q4.answer_accuracy_auto_graded_gold_answer_cases)}). Behind the gate it cuts Easy accuracy from ${f3(gate.easy_acc)} to ${f3(C.rows['Cascade: gate → verifier (gguf_q4)'].easy_acc)}, so it ships off by default, as an opt-in Strict mode.` : 'Its numbers are pending the Kaggle run.'}</li>
   <li><b>Retrieval is the bottleneck:</b> gold-page recall@4 ${f3(R.recallAtK[4])}, @8 ${f3(R.recallAtK[8])}, @12 ${f3(R.recallAtK[12])}. With the 3k-token cap the 3B sees the gold page ${(R.cappedAtK[R.k].recall * 100).toFixed(0)}% of the time; the sentence snippets keep it ${(R.snippetRecall * 100).toFixed(0)}%.</li>
   <li><b>A failure I shipped and fixed:</b> Best Buy's 10-K never mentions Walmart, yet "Walmart capex?" passed the first gate. Adding a feature for whether the companies a question names appear in the filing at all fixed it (gate accuracy ${H.calibrated_lr_accuracy_before.toFixed(2)} → ${H.calibrated_lr_accuracy_after.toFixed(2)} on the earlier own-vs-wrong-company bench).</li>
   <li>Answers are auto-graded (number within 1%, else token-F1), an approximation of FinanceBench's human grading. Scanned PDFs (no text layer) are rejected.</li></ul>`,
  `<h1>Roadmap</h1><ul class="big"><li>Verifier: 2 bootstrapped demos on a 3B model learned to abstain. Next: more labelled demos, a threshold on P(ANSWER) tuned on dev rather than the argmax verdict, and a 7B check on a laptop GPU</li>
   <li>Retrieval: table-aware chunking for financial statements; a reranker only if one ships as an npm package (none found) and adds &lt;150 ms</li>
   <li>Raise the context budget if the phone allows (uncapped k=${R.k} prompts reach ${R.promptTokensAtK[R.k].max} tokens)</li>
   <li>Indian filings: BSE/NSE annual reports, Hindi questions</li></ul>`,
  `<h1>The ask</h1><p class="lead">A Finale slot, one iQOO test phone to benchmark on-device inference, and mentor feedback on the Snapdragon NPU path.</p>
   <p>Repo: github.com/rajyyug1132/Filing-Lens-Private-Cited-Answers</p><p class="kicker">Cited answers, or an honest "I can't tell". Your filing stays yours.</p>`,
];

const html = `<!doctype html><html><head><meta charset="utf-8"><style>
@page { size: 1280px 720px; margin: 0 }
* { box-sizing: border-box } body { margin: 0; font-family: Inter, system-ui, sans-serif; color: #141413 }
section { width: 1280px; height: 720px; padding: 52px 64px; page-break-after: always; background: #f7f6f2; position: relative; overflow: hidden }
section::after { content: attr(data-n); position: absolute; right: 40px; bottom: 24px; color: #807f7a; font-size: 14px }
h1 { font-size: 38px; margin: 0 0 16px; letter-spacing: -0.02em } h2 { font-size: 23px; color: #1d5fb8; margin: 0 0 26px; font-weight: 600 }
p, li { font-size: 20px; line-height: 1.42 } .lead { font-size: 25px; line-height: 1.4 } .small { font-size: 14px; color: #52514e; line-height: 1.4 } .kicker { font-weight: 700; color: #1d5fb8; font-size: 22px }
code { font-size: 0.92em; background: #ecebe5; padding: 0 4px; border-radius: 4px }
.cols { display: flex; gap: 36px; align-items: flex-start } .cols > div { flex: 1 }
.shots { display: flex; gap: 28px; justify-content: center } .shots figure { margin: 0; text-align: center } .shots img { height: 470px; border-radius: 18px; border: 1px solid #e2e0d8 } figcaption { font-size: 16px; margin-top: 8px; color: #52514e }
.phone { height: 560px; border-radius: 18px; border: 1px solid #e2e0d8 }
.kv td { font-size: 19px; padding: 6px 14px 6px 0 } .kv td:last-child { font-size: 26px; color: #0f7b3e }
.arch { border: 1px solid #e2e0d8; background: #fff; border-radius: 16px; padding: 18px 20px }
.flow { display: flex; align-items: center; gap: 8px; margin: 14px 0; font-size: 18px; color: #807f7a; flex-wrap: wrap }
.flow span { color: #141413; background: #f0efe9; border-radius: 10px; padding: 9px 12px; font-size: 15px; text-align: center; line-height: 1.3 }
.flow span.key { background: #1d5fb8; color: #fff; font-weight: 600 } .flow span.key2 { background: #0f7b3e; color: #fff; font-weight: 600 }
.stops { display: flex; gap: 10px; margin: 4px 0 10px; flex-wrap: wrap } .stops span { font-size: 14px; color: #b3261e; border: 1px solid #f1d3b3; background: #fff6ed; border-radius: 999px; padding: 4px 10px }
.bench { gap: 22px } .tcol { flex: 1.25 } .t { border-collapse: collapse; font-size: 13.5px; width: 100% } .t th, .t td { border-bottom: 1px solid #e2e0d8; padding: 6px 5px; text-align: right } .t th:first-child, .t td:first-child { text-align: left }
.t tr.hl td { background: #e6eefa; font-weight: 600 } .t td.pending { text-align: center; color: #9a6200; font-weight: 600 } .plot { flex: 1 } .plot svg { width: 100%; height: auto }
.big li { margin: 9px 0; font-size: 19px } .ask { background: #fff1c2; padding: 0 6px; border-radius: 6px; font-weight: 600 }
</style></head><body>${slides.map((s, i) => `<section data-n="${i + 1} / ${slides.length}">${s}</section>`).join('')}</body></html>`;

writeFileSync('deck/deck.html', html);
const b = await chromium.launch();
const p = await b.newPage();
await p.setContent(html, { waitUntil: 'load' });
await p.pdf({ path: 'deck/FilingLens.pdf', width: '1280px', height: '720px', printBackground: true });
await b.close();
console.log('deck/FilingLens.pdf written');
