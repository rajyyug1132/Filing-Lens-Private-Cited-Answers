// UI QA harness: screenshots every app state at 390 and 1440 px, light and dark, plus a geometry audit.
// usage: node scripts/ui-shots.mjs <outDir> [state,state,...]   (needs `npm run build && npm run serve` on :4173)
import { chromium } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'fs';

const OUT = process.argv[2];
mkdirSync(OUT, { recursive: true });
const BASE = 'http://127.0.0.1:4173';
const PDF = 'tests/fixtures/BESTBUY_2023_10K.pdf';
const CASH = 'How much cash did operating activities provide in fiscal 2023?';
const WALMART = "What was Walmart's capital expenditure in fiscal 2023?";
const REVENUE = "What was Best Buy's total revenue in fiscal 2023?";
const VPS = { 390: { width: 390, height: 844, dsf: 2, mobile: true }, 1440: { width: 1440, height: 900, dsf: 1, mobile: false } };

// Slow model host, in-page: a fetch stub that streams a 1.1 GB "GGUF" at ~8 MB/s so the real download-progress UI runs.
const SLOW_MODEL = () => {
  const real = window.fetch.bind(window), total = 1_117_000_000;
  window.fetch = (input, init) => {
    const url = String(input instanceof Request ? input.url : input);
    if (!/huggingface\.co/.test(url)) return real(input, init);
    const h = { 'Content-Length': String(total), 'Accept-Ranges': 'bytes' };
    if ((init?.method || 'GET') === 'HEAD') return Promise.resolve(new Response(null, { status: 200, headers: h }));
    let sent = 0;
    return Promise.resolve(new Response(new ReadableStream({ async pull(c) { await new Promise((r) => setTimeout(r, 250)); c.enqueue(new Uint8Array(2_000_000)); sent += 2_000_000; if (sent > 8e7) c.close(); } }), { status: 200, headers: h }));
  };
};

async function newPage(browser, vp, theme, fixture, extraCtx = {}) {
  const v = VPS[vp];
  const ctx = await browser.newContext({ viewport: { width: v.width, height: v.height }, deviceScaleFactor: v.dsf, isMobile: v.mobile, hasTouch: v.mobile, colorScheme: theme, serviceWorkers: 'allow', ...extraCtx });
  const page = await ctx.newPage();
  await page.goto(`${BASE}/${fixture ? '?engine=fixture' : ''}`);
  return { ctx, page };
}
const sample = async (page) => { await page.getByTestId('sample-btn').click(); await page.getByTestId('doc-name').waitFor({ timeout: 60000 }); };
const askQ = async (page, q) => { await page.getByTestId('question').fill(q); await page.getByTestId('ask-btn').click(); };

const STATES = {
  async empty({ page }) { await page.waitForTimeout(1200); },
  async indexing({ page }) {
    await page.getByTestId('file-input').setInputFiles(PDF);
    await page.waitForFunction(() => /Reading page \d+ of|Embedding on-device/.test(document.body.innerText), null, { timeout: 60000 });
    await page.waitForTimeout(1500);
  },
  async 'model-idle'({ page }) { await sample(page); await page.getByTestId('gpu-check').waitFor(); await page.waitForTimeout(400); },
  async downloading({ page }) {
    await page.addInitScript(SLOW_MODEL);
    await page.reload();
    await sample(page);
    await page.getByTestId('load-model').click();
    await page.waitForFunction(() => /Downloading \d+/.test(document.body.innerText) || /Model failed/.test(document.body.innerText), null, { timeout: 25000 });
    await page.waitForTimeout(2500);
  },
  async 'model-error'({ page }) {
    await page.route(/huggingface\.co|hf\.co/, (r) => r.abort());
    await sample(page);
    await page.getByTestId('load-model').click();
    await page.getByText(/Model failed to load/).waitFor({ timeout: 30000 });
  },
  async asking({ page }) { await sample(page); await askQ(page, CASH); await page.getByTestId('pending').waitFor(); },
  async answer({ page }) { await sample(page); await askQ(page, CASH); await page.getByTestId('answer-card').waitFor({ timeout: 60000 }); await page.waitForTimeout(500); },
  async abstain({ page }) { await sample(page); await askQ(page, WALMART); await page.getByTestId('abstain-card').waitFor({ timeout: 60000 }); await page.waitForTimeout(500); },
  async 'strict-abstain'({ page }) {
    await sample(page); await page.getByTestId('strict-toggle').check(); await askQ(page, REVENUE);
    await page.getByTestId('abstain-card').waitFor({ timeout: 60000 }); await page.waitForTimeout(500);
  },
  async 'page-sheet'({ page }) {
    await sample(page); await askQ(page, CASH); await page.getByTestId('answer-card').waitFor({ timeout: 60000 });
    await page.getByTestId('cite-chip').first().click(); await page.getByTestId('sheet-page').waitFor(); await page.waitForTimeout(3500);
  },
  async privacy({ page }) { await sample(page); await page.getByTestId('privacy-pill').click(); await page.getByTestId('privacy-sheet').waitFor(); await page.waitForTimeout(400); },
  async offline({ page, ctx }) {
    await sample(page); await page.waitForTimeout(2500);
    await page.reload(); await page.waitForTimeout(3500);
    await ctx.setOffline(true);
    await page.reload({ waitUntil: 'domcontentloaded' }); await page.waitForTimeout(2500);
    const b = page.getByRole('button', { name: /Best Buy/ }).first();
    if (await b.count()) { await b.click(); await page.waitForTimeout(800); }
    await page.evaluate(() => scrollTo(0, 0));
  },
  async 'turn-error'({ page }) {
    await page.route(/workers\/embed\.js/, (r) => r.abort());
    await sample(page); await askQ(page, CASH);
    await page.getByText(/^Error$/).first().waitFor({ timeout: 20000 });
  },
};
const FIXTURE = new Set(['asking', 'answer', 'abstain', 'strict-abstain', 'page-sheet']);

async function geometry(page, vp) {
  return page.evaluate((vp) => {
    const bad = [];
    for (const el of document.querySelectorAll('button, a[href], input, [role=button], select, textarea')) {
      const r = el.getBoundingClientRect(); const cs = getComputedStyle(el);
      if (!r.width || !r.height || cs.visibility === 'hidden') continue;
      if (r.width < 44 || r.height < 44) bad.push({ el: (el.getAttribute('data-testid') || el.className || el.tagName) + ':' + (el.textContent || '').trim().slice(0, 18), w: Math.round(r.width), h: Math.round(r.height) });
    }
    return { overflowX: document.documentElement.scrollWidth > innerWidth, small: bad };
  }, vp);
}

const only = process.argv[3] ? process.argv[3].split(',') : Object.keys(STATES);
const browser = await chromium.launch();
const audit = {};
await Promise.all(Object.keys(VPS).flatMap((vp) => ['light', 'dark'].map(async (theme) => {
  for (const name of only) {
    let c;
    try {
      c = await newPage(browser, vp, theme, FIXTURE.has(name), name === 'turn-error' ? { serviceWorkers: 'block' } : {});
      await STATES[name](c);
      await c.page.screenshot({ path: `${OUT}/${name}-${vp}-${theme}.png` });
      if (['answer', 'abstain', 'model-idle', 'empty'].includes(name)) audit[`${name}-${vp}-${theme}`] = await geometry(c.page, vp);
    } catch (e) { console.log(`FAIL ${name} ${vp} ${theme}: ${String(e.message).split('\n')[0]}`); try { await c?.page.screenshot({ path: `${OUT}/${name}-${vp}-${theme}.FAIL.png` }); } catch {} }
    finally { await c?.ctx.close(); }
  }
})));
writeFileSync(`${OUT}/audit.json`, JSON.stringify(audit, null, 1));
await browser.close();
console.log('done', OUT);
