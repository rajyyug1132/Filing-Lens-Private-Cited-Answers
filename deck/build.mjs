// Builds deck/FilingLens.pdf (8 slides, 16:9) from live artifacts:
// bench/results/results.json, bench/results/reliability.svg, demo screenshots
// and the e2e privacy log. No number on a slide is typed by hand.
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';

const R = JSON.parse(readFileSync('bench/results/results.json', 'utf8'));
const svg = readFileSync('bench/results/reliability.svg', 'utf8');
const G = existsSync('bench/results/gen_metrics.json') ? JSON.parse(readFileSync('bench/results/gen_metrics.json', 'utf8')) : { status: 'pending' };
const gv = (k, d = 3) => (G.status === 'real' && G[k] != null ? G[k].toFixed(d) : 'pending');
const priv = existsSync('demo/privacy-log.json') ? JSON.parse(readFileSync('demo/privacy-log.json', 'utf8')) : null;
const img = (p) => (existsSync(p) ? `data:image/png;base64,${readFileSync(p).toString('base64')}` : '');
const by = Object.fromEntries(R.results.map((r) => [r.key, r]));
const f3 = (v) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(3));
const pc = (v) => `${Math.round(v * 100)}%`;
const lat = R.latencyMs;
const ask = (s) => `<span class="ask">[ASK: ${s}]</span>`;

const rows = R.results
  .map(
    (r) => `<tr class="${r.key === 'lrTemp' ? 'hl' : ''}"><td>${r.name}</td><td>${f3(r.accuracy)}</td><td>${f3(r.ece)}</td><td>${f3(r.abstainPrecision)}</td><td>${f3(r.abstainRecall)}</td><td>${r.key === 'always' ? '0' : r.key === 'cosine' ? '&lt;0.01' : (lat.featureExtraction.p50 + lat.classifierPredict.p50).toFixed(2)} ms</td><td>$0</td></tr>`,
  )
  .join('');

const slides = [
  `<h1>Filing Lens</h1><h2>Cited answers from company filings, from a local 3B model (Qwen 2.5, runs on the phone)</h2>
   <div class="cols"><div><p class="lead">Retail investors are handed 100-page 10-Ks. Chatbots answer anyway, with no page reference, and the filing gets uploaded to someone else's server.</p>
   <p>FinanceBench (Islam et al., 2023, arXiv:2311.11944): GPT-4-Turbo with a retrieval system <b>incorrectly answered or refused 81%</b> of 150 open-book filing questions.</p>
   <p class="kicker">Problem: wrong answers look the same as right ones, and privacy is the price of asking.</p></div></div>`,
  `<h1>Demo</h1><div class="shots">${['demo/answer.png', 'demo/page.png', 'demo/abstain.png'].map((p, i) => (img(p) ? `<figure><img src="${img(p)}"/><figcaption>${['Cited answer + calibrated confidence', 'Citation chip opens the page', 'Unsupported question → abstain'][i]}</figcaption></figure>` : '')).join('')}</div>
   <p class="small">Recorded in headless Chromium at Pixel-7 size (no phone on the build machine). The phone video is ${ask('phone recording link')}.</p>`,
  `<h1>Privacy proof</h1><div class="cols"><div>
   <p class="lead">The document never leaves the device. This is enforced by the build's e2e test, not just claimed.</p>
   ${priv ? `<table class="kv"><tr><td>Requests during upload → index → ask → abstain</td><td><b>${priv.total}</b></td></tr><tr><td>to any host other than the app's own origin</td><td><b>${priv.external}</b></td></tr><tr><td>carrying a request body (POST/PUT/beacon)</td><td><b>${priv.withBody}</b></td></tr></table>` : ask('privacy log missing')}
   <ul><li>pdf.js, MiniLM embeddings (23 MB), ONNX runtime: served from the app origin</li><li>Index + PDF stored in IndexedDB</li><li>Only 3rd-party fetch: the Qwen2.5 weights, GET, once, then cached. After that it works in airplane mode ${ask('confirm on phone')}</li><li>In-app counter wraps fetch/XHR/beacon and shows request-body bytes: 0</li></ul></div>
   ${img('demo/privacy.png') ? `<img class="phone" src="${img('demo/privacy.png')}"/>` : ''}</div>`,
  `<h1>Architecture</h1><div class="arch">
   <div class="lane"><b>On the phone (browser, PWA)</b>
   <div class="flow"><span>PDF file</span>→<span>pdf.js<br/>page text</span>→<span>page-bounded chunks<br/>160 words</span>→<span>MiniLM-L6 embed<br/>(ONNX, WASM)</span>→<span>IndexedDB</span></div>
   <div class="flow"><span>Question</span>→<span>hybrid retrieval<br/>dense + BM25, RRF, top-4</span>→<span class="key">decision layer<br/>9 features → LR → ÷T</span>→<span>p ≥ 0.5 ? Qwen2.5-3B (WebLLM, WebGPU)<br/>no WebGPU → 1.5B (wllama, WASM)<br/>: ABSTAIN</span>→<span>cite check<br/>[p.N] ∈ context</span></div></div>
   <p class="small">Answer only if (1) calibrated p(supported) ≥ 0.5, (2) the model doesn't say NOT IN DOCUMENT, (3) every citation points at a retrieved page and ≥50% of sentences are cited. Otherwise show the abstain card with the closest pages.</p></div>`,
  `<h1>Bench: decision layer</h1><p class="small">${R.dataset}. ${R.instances} question–filing pairs, 1:1: ${R.positives} gold-answer (own filing), ${R.instances - R.positives} gold-abstain (another company's 10-K). ${R.folds}-fold CV grouped by company. Gold-page recall: ${Object.entries(R.recallAtK).map(([k, v]) => `@${k} ${v.toFixed(2)}`).join(' · ')}.</p>
   <div class="cols bench"><table class="t"><thead><tr><th>Gate</th><th>Acc.</th><th>ECE ↓</th><th>Abstain prec.</th><th>Abstain rec.</th><th>Latency</th><th>Cost/q</th></tr></thead><tbody>${rows}<tr><td>LLM-router (3B YES/NO)</td><td colspan="6">${G.status === 'real' ? 'end-to-end ' + (G.end_to_end['LLM-router (3B YES/NO)']?.accuracy ?? 0).toFixed(3) : 'pending (Kaggle)'}</td></tr><tr><td>Jev</td><td colspan="6">${ask('what is Jev + can it run locally')}</td></tr></tbody></table>
   <div class="plot">${svg}</div></div>
   <p class="small">3B generation (Kaggle T4): citation accuracy ${gv('citation_accuracy_on_answered')} · answer accuracy ${gv('answer_accuracy_gold_answer_cases')} · tokens/s ${gv('tokens_per_s_median', 1)}. Retrieval p50 ${lat.queryEmbedPlusRetrieve.p50.toFixed(1)} ms on ${R.machine}. Phone latency ${ask('measure on iQOO')}.</p>`,
  `<h1>Limits (honest)</h1><ul class="big">
   <li>The 3B was <b>not run</b> on the build machine (model hosts blocked). Generation metrics come from the Kaggle T4 run: ${G.status === 'real' ? 'see table' : 'pending'}.</li>
   <li>Abstain cases are another company's 10-K: the gate mostly has to notice the question's company isn't in the filing. A near-miss question on the right filing is harder and not in this bench.</li>
   <li>Answer grading is automatic (number within 1%, else token-F1), an approximation of FinanceBench's human grading.</li>
   <li>Scanned PDFs (no text layer) are rejected; no OCR yet. Tables are read as flattened text.</li>
   <li>~2 GB one-time model download; WebGPU needs ~2.5 GB of GPU memory (WebLLM's figure for Qwen2.5-3B q4f16).</li></ul>`,
  `<h1>Roadmap</h1><ul class="big"><li>QA fine-tune of the local 3B on filing questions, if the Kaggle numbers show the gap is generation rather than retrieval</li>
   <li>Train the decision layer on hard negatives: same filing, a question the filing doesn't answer</li><li>Table-aware chunking for financial statements (row/column headers kept with numbers)</li>
   <li>Indian filings: BSE/NSE annual reports, Hindi questions</li><li>Multi-document compare: "this year vs last year" across two 10-Ks, still on-device</li></ul>`,
  `<h1>The ask</h1><p class="lead">${ask('what you want from the judges / iQOO: e.g. device lab time on iQOO phones, mentorship, pilot partner')}</p>
   <p>Repo: github.com/rajyyug1132/Filing-Lens-Private-Cited-Answers</p><p class="kicker">Cited answers, or an honest "I can't tell". Your filing stays yours.</p>`,
];

const html = `<!doctype html><html><head><meta charset="utf-8"><style>
@page { size: 1280px 720px; margin: 0 }
* { box-sizing: border-box } body { margin: 0; font-family: Inter, system-ui, sans-serif; color: #141413 }
section { width: 1280px; height: 720px; padding: 56px 72px; page-break-after: always; background: #f7f6f2; position: relative; overflow: hidden }
section::after { content: attr(data-n); position: absolute; right: 40px; bottom: 28px; color: #807f7a; font-size: 14px }
h1 { font-size: 40px; margin: 0 0 18px; letter-spacing: -0.02em } h2 { font-size: 24px; color: #1d5fb8; margin: 0 0 28px; font-weight: 600 }
p, li { font-size: 21px; line-height: 1.45 } .lead { font-size: 26px; line-height: 1.4 } .small { font-size: 15px; color: #52514e } .kicker { font-weight: 700; color: #1d5fb8; font-size: 22px }
.cols { display: flex; gap: 40px; align-items: flex-start } .cols > div { flex: 1 }
.shots { display: flex; gap: 28px; justify-content: center } .shots figure { margin: 0; text-align: center } .shots img { height: 470px; border-radius: 18px; border: 1px solid #e2e0d8 } figcaption { font-size: 16px; margin-top: 8px; color: #52514e }
.phone { height: 560px; border-radius: 18px; border: 1px solid #e2e0d8 }
.kv td { font-size: 20px; padding: 6px 14px 6px 0 } .kv td:last-child { font-size: 26px; color: #0f7b3e }
.arch .lane { border: 1px solid #e2e0d8; background: #fff; border-radius: 16px; padding: 20px }
.flow { display: flex; align-items: center; gap: 10px; margin: 18px 0; font-size: 18px; color: #807f7a; flex-wrap: wrap }
.flow span { color: #141413; background: #f0efe9; border-radius: 10px; padding: 10px 14px; font-size: 16px; text-align: center } .flow span.key { background: #1d5fb8; color: #fff; font-weight: 600 }
.bench { gap: 24px } .t { border-collapse: collapse; font-size: 14.5px; flex: 1.15 } .t th, .t td { border-bottom: 1px solid #e2e0d8; padding: 7px 6px; text-align: right } .t th:first-child, .t td:first-child { text-align: left }
.t tr.hl td { background: #e6eefa; font-weight: 700 } .plot { flex: 1 } .plot svg { width: 100%; height: auto }
.big li { margin: 10px 0 } .ask { background: #fff1c2; padding: 0 6px; border-radius: 6px; font-weight: 600 }
</style></head><body>${slides.map((s, i) => `<section data-n="${i + 1} / ${slides.length}">${s}</section>`).join('')}</body></html>`;

writeFileSync('deck/deck.html', html);
const b = await chromium.launch();
const p = await b.newPage();
await p.setContent(html, { waitUntil: 'load' });
await p.pdf({ path: 'deck/FilingLens.pdf', width: '1280px', height: '720px', printBackground: true });
await b.close();
console.log('deck/FilingLens.pdf written');
