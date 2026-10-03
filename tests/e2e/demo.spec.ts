import { test, expect } from '@playwright/test';

// Scripted demo recorded as video at phone size (headless Chromium, Pixel 7
// emulation; there is no phone on the build machine). The generator is the
// recorded-response stub (?engine=fixture) because the build machine cannot
// download model weights. On the phone, tap "Download model" first.
// Voice-over for each beat: demo/SCRIPT.md.
test.use({ video: { mode: 'on', size: { width: 412, height: 915 } } });

const beat = (ms: number) => new Promise((r) => setTimeout(r, ms));

test('demo recording', async ({ page }) => {
  test.setTimeout(420_000);
  await page.goto('/?engine=fixture');
  await beat(2500);
  await page.getByTestId('file-input').setInputFiles('tests/fixtures/BESTBUY_2023_10K.pdf');
  await expect(page.getByTestId('doc-name')).toBeVisible({ timeout: 300_000 });
  await beat(2000);

  await page.getByTestId('question').pressSequentially('How much cash did operating activities provide in fiscal 2023?', { delay: 35 });
  await page.getByTestId('ask-btn').click();
  await expect(page.getByTestId('answer-card')).toBeVisible();
  await beat(3000);
  await page.screenshot({ path: 'demo/answer.png' });
  await page.getByTestId('cite-chip').first().click();
  await expect(page.getByTestId('page-sheet')).toBeVisible();
  await beat(3500);
  await page.screenshot({ path: 'demo/page.png' });
  await page.getByLabel('Close').click();

  await page.getByTestId('question').pressSequentially("What was Walmart's capital expenditure in fiscal 2023?", { delay: 35 });
  await page.getByTestId('ask-btn').click();
  await expect(page.getByTestId('abstain-card')).toBeVisible();
  await page.getByTestId('abstain-card').scrollIntoViewIfNeeded();
  await beat(3500);
  await page.screenshot({ path: 'demo/abstain.png' });

  await page.getByTestId('privacy-pill').click();
  await expect(page.getByTestId('privacy-sheet')).toBeVisible();
  await beat(4000);
  await page.screenshot({ path: 'demo/privacy.png' });
});
