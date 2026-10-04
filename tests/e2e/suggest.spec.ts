import { expect, test, type Browser, type Page, type Request } from '@playwright/test';
import { syllabusHtml } from '../fixtures/syllabus';

const OLD_CHIPS = /Tesla|Cybertruck|How much cash did operating activities provide in fiscal 2023\?|What are the main risk factors\?/;

async function syllabusPdf(browser: Browser) {
  const p = await browser.newPage();
  await p.setContent(syllabusHtml());
  const pdf = await p.pdf({ format: 'A4' });
  await p.close();
  return pdf;
}
const cards = (page: Page) => page.locator('[data-testid=answer-card],[data-testid=abstain-card]');
async function ask(page: Page, q: string) {
  const n = await cards(page).count();
  await page.getByTestId('question').fill(q);
  await page.getByTestId('ask-btn').click();
  await expect(cards(page)).toHaveCount(n + 1, { timeout: 60_000 });
  return cards(page).nth(n);
}
async function suggestions(page: Page) {
  await expect(page.getByTestId('suggestion').first()).toBeVisible({ timeout: 60_000 });
  return page.getByTestId('suggestion').evaluateAll((els) => els.map((e) => ({ q: e.getAttribute('data-q')!, page: e.getAttribute('data-page')! })));
}

test('syllabus: GENERAL gate, derived suggestions pass it, textbooks cited, wrong subject abstains, answer anyway is unverified', async ({ page, browser, baseURL }) => {
  test.setTimeout(300_000);
  const requests: Request[] = [];
  page.on('request', (r) => requests.push(r));
  await page.goto('/');
  await expect(page.getByTestId('upload-btn')).toBeVisible();
  await page.getByTestId('file-input').setInputFiles({ name: 'CS3201_Operating_Systems_Syllabus.pdf', mimeType: 'application/pdf', buffer: await syllabusPdf(browser) });
  await expect(page.getByTestId('doc-name')).toHaveText('CS3201_Operating_Systems_Syllabus.pdf', { timeout: 200_000 });
  await expect(page.getByTestId('doc-type')).toHaveText('GENERAL DOCUMENT · gate tuned for annual reports');
  const sugs = await suggestions(page);
  console.log('syllabus suggestions:', JSON.stringify(sugs));
  expect(sugs.length).toBeGreaterThanOrEqual(2);
  expect(sugs.map((s) => s.q).join(' ')).not.toMatch(OLD_CHIPS);
  for (const s of sugs) await expect(await ask(page, s.q), `suggestion passes its gate: ${s.q}`).toHaveAttribute('data-testid', 'answer-card');

  const tb = await ask(page, 'Which textbooks are listed?');
  await expect(tb).toHaveAttribute('data-testid', 'answer-card');
  await expect(tb.getByTestId('cite-chip').filter({ hasText: 'p.9' }).first()).toBeVisible();

  const dbms = await ask(page, 'What is the DBMS syllabus?');
  await expect(dbms).toHaveAttribute('data-testid', 'abstain-card');
  await dbms.getByTestId('answer-anyway').click();
  const forced = page.getByTestId('answer-card').filter({ has: page.getByTestId('unverified') });
  await expect(forced).toHaveCount(1, { timeout: 60_000 });
  await expect(forced.getByTestId('unverified')).toHaveText('Below the gate: unverified');
  await expect(forced.getByTestId('confidence')).toHaveCount(0);

  const origin = new URL(baseURL!).origin;
  expect(requests.filter((r) => new URL(r.url()).origin !== origin && !r.url().startsWith('data:') && !r.url().startsWith('blob:')).map((r) => r.url())).toEqual([]);
  expect(requests.filter((r) => r.method() !== 'GET' || (r.postDataBuffer()?.length ?? 0) > 0).map((r) => r.url())).toEqual([]);
});

test('sample 10-K: FILING gate, derived suggestions only, each passes its gate', async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto('/');
  await page.getByTestId('sample-btn').click();
  await expect(page.getByTestId('doc-name')).toContainText('Best Buy FY2023 10-K');
  await expect(page.getByTestId('doc-type')).toHaveCount(0);
  const sugs = await suggestions(page);
  console.log('sample suggestions:', JSON.stringify(sugs));
  expect(sugs.length).toBeGreaterThanOrEqual(2);
  expect(sugs.map((s) => s.q).join(' ')).not.toMatch(OLD_CHIPS);
  for (const s of sugs) await expect(await ask(page, s.q), `suggestion passes its gate: ${s.q}`).toHaveAttribute('data-testid', 'answer-card');
});
