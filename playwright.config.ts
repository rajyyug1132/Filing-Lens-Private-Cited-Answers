import { defineConfig, devices } from '@playwright/test';

// Headless Chromium on the build machine. There is no phone here, so the
// mobile layout is emulated with a Pixel 7 profile (412×915, touch, DPR 2.6).
export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 240_000,
  expect: { timeout: 120_000 },
  reporter: [['list']],
  use: { baseURL: 'http://127.0.0.1:4173', ...devices['Pixel 7'], browserName: 'chromium' },
  webServer: { command: 'npx http-server out -p 4173 -a 127.0.0.1 -c-1 --silent', url: 'http://127.0.0.1:4173', reuseExistingServer: true },
});
