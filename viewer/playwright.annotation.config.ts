import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  testMatch: 'annotation.spec.ts',
  workers: 1,
  timeout: 60_000,
  use: {
    baseURL: 'http://127.0.0.1:4173',
    channel: 'chrome',
    viewport: { width: 1586, height: 992 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: 'npm run dev',
      url: 'http://127.0.0.1:4173/annotate.html',
      reuseExistingServer: true,
      timeout: 120_000,
    },
    {
      command: 'FACE3D_ANNOTATION_EXECUTION_MODE=manual FACE3D_ANNOTATION_JOB_ROOT=../.tmp/annotation-jobs node bridge/annotation-bridge.mjs',
      url: 'http://127.0.0.1:4174/api/annotation/health',
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
})
