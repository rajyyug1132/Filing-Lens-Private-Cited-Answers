import { writeFileSync } from 'node:fs';
import { expect, test, type Request } from '@playwright/test';

// The generator is stubbed with recorded responses (?engine=fixture); the
// retrieval, decision layer, citation check and UI are the real ones.

const PDF = 'tests/fixtures/BESTBUY_2023_10K.pdf'; // FinanceBench copy of Best Buy's FY2023 10-K

test('upload → cited answer → abstain, with zero outbound document requests', async ({ page, baseURL }) => {
  const requests: Request[] = [];
  page.on('request', (r) => requests.push(r));
  const workerReqs: string[] = [];
  page.on('worker', (w) => workerReqs.push(w.url()));

  await page.goto('/?engine=fixture');
  await expect(page.getByTestId('engine-label').or(page.getByTestId('upload-btn'))).toBeVisible();
  await page.getByTestId('file-input').setInputFiles(PDF);
  await expect(page.getByTestId('doc-name')).toHaveText('BESTBUY_2023_10K.pdf', { timeout: 200_000 });
  await expect(page.getByTestId('engine-label')).toContainText('Recorded responses');

  // answerable: FinanceBench gold evidence is the cash-flow statement, p.42
  await page.getByTestId('question').fill('How much total cash was provided by operating activities in fiscal 2023?');
  await page.getByTestId('ask-btn').click();
  const answer = page.getByTestId('answer-card');
  await expect(answer).toBeVisible();
  const chips = answer.getByTestId('cite-chip');
  expect(await chips.count()).toBeGreaterThan(0);
  const conf = Number(await answer.getByTestId('confidence').getAttribute('data-value'));
  expect(conf).toBeGreaterThanOrEqual(0.5);

  // citation chip jumps to that page
  const first = await chips.first().innerText();
  await chips.first().click();
  await expect(page.getByTestId('sheet-page')).toContainText(`Page ${first.replace('p.', '')}`);
  await page.getByLabel('Close').click();

  // unanswerable from this filing → abstain card
  await page.getByTestId('question').fill('What did Tesla say about Cybertruck production?');
  await page.getByTestId('ask-btn').click();
  await expect(page.getByTestId('abstain-card')).toBeVisible();

  // privacy proof
  const origin = new URL(baseURL!).origin;
  const external = requests.filter((r) => new URL(r.url()).origin !== origin && !r.url().startsWith('data:') && !r.url().startsWith('blob:'));
  const withBody = requests.filter((r) => r.method() !== 'GET' || (r.postDataBuffer()?.length ?? 0) > 0);
  console.log(`requests: ${requests.length} total, ${external.length} external, ${withBody.length} with a body`);
  writeFileSync('demo/privacy-log.json', JSON.stringify({ total: requests.length, external: external.length, withBody: withBody.length, browser: 'headless Chromium (Playwright), Pixel 7 emulation' }, null, 2));
  expect(external.map((r) => r.url())).toEqual([]);
  expect(withBody.map((r) => `${r.method()} ${r.url()}`)).toEqual([]);
  await expect(page.getByTestId('privacy-pill')).toContainText('0 doc bytes sent');
});

test('pre-indexed sample: cited answer, page jump, wrong-company abstain; persists across reload', async ({ page }) => {
  await page.goto('/?engine=fixture');
  await page.getByTestId('sample-btn').click();
  await expect(page.getByTestId('doc-name')).toContainText('Best Buy FY2023 10-K');
  await page.getByTestId('question').fill('How much cash did operating activities provide in fiscal 2023?');
  await page.getByTestId('ask-btn').click();
  await expect(page.getByTestId('answer-card').getByTestId('cite-chip').first()).toHaveText('p.42');
  await page.getByTestId('question').fill('What was Walmart\'s capital expenditure in fiscal 2023?');
  await page.getByTestId('ask-btn').click();
  await expect(page.getByTestId('abstain-card')).toBeVisible();
  await page.reload();
  await expect(page.getByRole('button', { name: /^Best Buy FY2023 10-K/ })).toBeVisible();
});
