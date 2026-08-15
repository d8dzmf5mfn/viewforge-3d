import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'

const demo = path.resolve('../.tmp/face-001.face3d')
const demoV2 = path.resolve('../.tmp/face-v2-contract.face3d')
const demoV3 = path.resolve('../.tmp/face-v3-contract.face3d')
const viewerContract = (JSON.parse(
  fs.readFileSync(path.resolve('../quality/template-head-v0-contract.json'), 'utf8'),
) as {
  viewer: {
    minimumInteractionFps: number
    maximumModelLoadMs: number
    repeatLoadCount: number
    maximumJsHeapGrowthRatio: number
    maximumExternalRequestCount: number
    maximumWebglCanvasCount: number
  }
}).viewer

const v1FixtureTests = new Set([
  'loads locally, keeps one WebGL canvas, and switches modes',
  'sustains the desktop interaction frame-rate gate',
  'keeps synchronized comparison views with a draggable divider',
])
const v2FixtureTests = new Set(['keeps Face v2 package compatibility'])

test.beforeEach(async ({ page }, testInfo) => {
  const external: string[] = []
  const browserErrors: string[] = []
  page.on('request', (request) => {
    const url = new URL(request.url())
    if (url.hostname !== '127.0.0.1' && url.protocol !== 'blob:' && url.protocol !== 'data:') external.push(request.url())
  })
  page.on('console', (message) => {
    if (message.type() === 'error' || message.type() === 'warning') browserErrors.push(message.text())
  })
  page.on('pageerror', (error) => browserErrors.push(error.message))
  await page.goto('/')
  const useV1 = v1FixtureTests.has(testInfo.title)
  const useV2 = v2FixtureTests.has(testInfo.title)
  const fixture = useV1 ? demo : useV2 ? demoV2 : demoV3
  const subject = useV1 ? 'face-001' : useV2 ? 'face-v2-contract' : 'face-v3-contract'
  const started = Date.now()
  await page.locator('input[type=file]').setInputFiles(fixture)
  await expect(page.getByText(subject)).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(100)
  // Quality-first mode may carry a denser Pixel surface. Keep a broad safety
  // ceiling, while reporting the measured load time separately in the UI.
  expect(Date.now() - started).toBeLessThan(10_000)
  expect(external).toHaveLength(viewerContract.maximumExternalRequestCount)
  expect(browserErrors).toEqual([])
})

test.afterEach(async ({ page }) => {
  // Navigate before Playwright destroys the context so the application's
  // pagehide cleanup can release WebGL contexts and ImageBitmap allocations.
  if (page.isClosed()) return
  try {
    await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide')))
    if (page.isClosed()) return
    await page.waitForTimeout(100)
    if (!page.isClosed()) await page.goto('about:blank')
  } catch (error) {
    if (!page.isClosed()) throw error
  }
})

test('loads locally, keeps one WebGL canvas, and switches modes', async ({ page }) => {
  await expect(page.locator('canvas')).toHaveCount(viewerContract.maximumWebglCanvasCount)
  await page.getByRole('button', { name: '3D Pixel' }).click()
  await expect(page.getByRole('button', { name: '3D Pixel' })).toHaveClass(/selected/)
  await page.getByRole('button', { name: '平滑网格' }).click()
  await expect(page.getByRole('button', { name: '平滑网格' })).toHaveClass(/selected/)
  await page.getByRole('button', { name: '人皮' }).click()
  await expect(page.getByRole('button', { name: '人皮' })).toHaveClass(/selected/)
  await expect(page.getByAltText('完整人头皮肤 UV 展开图')).toBeVisible()
  await expect(page.locator('.viewport-host')).toHaveAttribute('data-skin-ready', 'true')
  await page.getByRole('button', { name: '正面' }).click()
  await page.screenshot({ path: '../.tmp/viewer-skin-front.png', fullPage: true })
  await page.getByRole('button', { name: '三分之四' }).click()
  await page.screenshot({ path: '../.tmp/viewer-skin-three-quarter.png', fullPage: true })
  await page.getByRole('button', { name: '侧面' }).click()
  await page.screenshot({ path: '../.tmp/viewer-skin-side.png', fullPage: true })
  await page.getByRole('button', { name: '三分之四' }).click()
  await expect(page.getByText('区域置信度')).toBeVisible()
})

test('sustains the desktop interaction frame-rate gate', async ({ page }) => {
  const fps = await page.evaluate(() => new Promise<number>((resolve) => {
    let frames = 0
    const started = performance.now()
    const sample = (now: number) => {
      frames += 1
      if (now - started >= 1_000) resolve(frames * 1_000 / (now - started))
      else requestAnimationFrame(sample)
    }
    requestAnimationFrame(sample)
  }))
  expect(fps).toBeGreaterThanOrEqual(viewerContract.minimumInteractionFps)
})

test('keeps synchronized comparison views with a draggable divider', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 })
  const divider = page.getByRole('button', { name: '调整对照分隔线' })
  const viewport = page.locator('.viewport-host')
  const bounds = await viewport.boundingBox()
  if (!bounds) throw new Error('viewport bounds unavailable')
  await page.screenshot({ path: '../.tmp/viewer-pixel-direct.png', fullPage: true })
  await page.getByRole('button', { name: '侧面' }).click()
  await expect(page.getByRole('button', { name: '侧面' })).toHaveClass(/selected/)
  await page.screenshot({ path: '../.tmp/viewer-side.png', fullPage: true })
  await divider.dragTo(viewport, {
    targetPosition: { x: Math.round(bounds.width * 0.64), y: Math.round(bounds.height / 2) },
  })
  await expect(viewport).toHaveCSS('--viewport-split', /6[0-9](\.\d+)?%/)
})

test('exports a local acceptance report', async ({ page }) => {
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: '导出验收报告' }).click()
  expect((await download).suggestedFilename()).toContain('-acceptance.zip')
})

test('uses a single viewport below 900px', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await expect(page.locator('.reference-rail')).toBeHidden()
  await expect(page.locator('.quality-panel')).toBeHidden()
  await expect(page.locator('canvas')).toHaveCount(viewerContract.maximumWebglCanvasCount)
})

test('loads Face v3 once, shares geometry, exposes diagnostics, and releases repeated loads', async ({ page, context }) => {
  const input = page.locator('input[type=file]')
  const viewport = page.locator('.viewport-host')
  const waitForGeneration = async (expected: number) => {
    await expect(viewport).toHaveAttribute('data-model-generation', String(expected), { timeout: 15_000 })
    await expect(input).toHaveValue('')
    const modelLoadMs = Number(await page.locator('html').getAttribute('data-model-load-ms'))
    expect(Number.isFinite(modelLoadMs)).toBe(true)
    expect(modelLoadMs).toBeLessThanOrEqual(viewerContract.maximumModelLoadMs)
  }
  await expect(page.getByRole('button', { name: '3D Pixel' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: '对照', exact: true })).toHaveClass(/selected/)
  await expect(viewport).toHaveAttribute('data-head-parse-count', '1')
  await expect(viewport).toHaveAttribute('data-shared-geometry', 'true')
  await expect(page.getByText('21,400 面')).toBeVisible()
  await page.screenshot({ path: '../.tmp/viewer-v3-same-geometry-comparison.png', fullPage: true })
  await page.getByRole('button', { name: '眼球接触' }).click()
  await expect(page.getByRole('button', { name: '眼球接触' })).toHaveClass(/selected/)
  await page.getByRole('button', { name: '耳根连续' }).click()
  await expect(page.getByRole('button', { name: '耳根连续' })).toHaveClass(/selected/)
  await page.getByRole('button', { name: '皮肤投影' }).click()
  await expect(page.getByRole('button', { name: '皮肤投影' })).toHaveClass(/selected/)
  // Warm the GLTF parser, shaders, and V8 optimized code before measuring
  // retained memory. Those one-time allocations are not package resources.
  let generation = Number(await viewport.getAttribute('data-model-generation'))
  for (let index = 0; index < 3; index += 1) {
    await input.setInputFiles(demoV3)
    generation += 1
    await waitForGeneration(generation)
  }
  const baseline = await viewport.evaluate((element) => ({
    geometries: element.getAttribute('data-resource-geometries'),
    textures: element.getAttribute('data-resource-textures'),
    generation: Number(element.getAttribute('data-model-generation')),
  }))
  const collectGarbage = async () => {
    await page.evaluate(() => {
      const gc = (globalThis as typeof globalThis & { gc?: () => void }).gc
      if (!gc) throw new Error('window.gc is unavailable')
      gc()
    })
  }
  const cdp = await context.newCDPSession(page)
  await collectGarbage()
  const baselineHeap = (await cdp.send('Runtime.getHeapUsage')).usedSize
  for (let index = 0; index < viewerContract.repeatLoadCount; index += 1) {
    await input.setInputFiles(demoV3)
    await waitForGeneration(baseline.generation + index + 1)
  }
  await collectGarbage()
  await page.waitForTimeout(100)
  await expect(viewport).toHaveAttribute('data-resource-geometries', baseline.geometries ?? '')
  await expect(viewport).toHaveAttribute('data-resource-textures', baseline.textures ?? '')
  const finalHeap = (await cdp.send('Runtime.getHeapUsage')).usedSize
  expect(finalHeap).toBeLessThanOrEqual(
    baselineHeap * (1 + viewerContract.maximumJsHeapGrowthRatio),
  )
  await expect(page.locator('canvas')).toHaveCount(viewerContract.maximumWebglCanvasCount)
})

test('keeps Face v2 package compatibility', async ({ page }) => {
  await expect(page.getByRole('button', { name: '3D Pixel' })).toBeVisible()
  await expect(page.locator('.viewport-host')).toHaveAttribute('data-head-parse-count', '1')
  await expect(page.getByText('统一头模')).toBeVisible()
})
