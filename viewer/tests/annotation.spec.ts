import { expect, test } from '@playwright/test'

test('draws a surface region and submits it for manual takeover', async ({ page }) => {
  const browserErrors: string[] = []
  const externalRequests: string[] = []
  const failedResponses: string[] = []
  page.on('console', (message) => {
    if (message.type() === 'error' || message.type() === 'warning') browserErrors.push(message.text())
  })
  page.on('pageerror', (error) => browserErrors.push(error.message))
  page.on('response', (response) => {
    if (response.status() >= 400) failedResponses.push(`${response.status()} ${response.url()}`)
  })
  page.on('request', (request) => {
    const url = new URL(request.url())
    if (!['127.0.0.1'].includes(url.hostname) && !['blob:', 'data:'].includes(url.protocol)) externalRequests.push(request.url())
  })

  await page.goto('/annotate.html')
  await expect(page.getByText(/人工接手模式/)).toBeVisible()
  await expect(page.getByText('OriginalAnime V29 已锁定')).toBeVisible()
  await expect(page.getByRole('button', { name: '重新载入 OriginalAnime V29' })).toBeVisible()
  await expect(page.getByRole('button', { name: /压低/ })).toBeVisible()
  await expect(page.getByText('OriginalAnime V29 已载入；标注模式下拖动画闭合区域')).toBeVisible({ timeout: 30_000 })

  await page.getByRole('button', { name: /压低/ }).click()

  const viewport = page.getByTestId('annotation-viewport')
  const bounds = await viewport.boundingBox()
  if (!bounds) throw new Error('viewport bounds unavailable')
  const cx = bounds.x + bounds.width * 0.47
  const cy = bounds.y + bounds.height * 0.57
  const points = [
    [cx - 52, cy - 44], [cx + 8, cy - 58], [cx + 62, cy - 18],
    [cx + 52, cy + 42], [cx, cy + 68], [cx - 58, cy + 34], [cx - 52, cy - 44],
  ]
  await page.mouse.move(points[0][0], points[0][1])
  await page.mouse.down()
  for (const [x, y] of points.slice(1)) await page.mouse.move(x, y, { steps: 9 })
  await page.mouse.up()
  await expect(page.getByText(/表面命中 [1-9]/)).toBeVisible()

  await page.getByLabel('给当前任务的说明').fill('只压低圈选区域，保持鼻子、眼眶、嘴线和下颚轮廓不动。')
  await page.getByRole('button', { name: '完成这个区域' }).click()
  await expect(page.getByText('脸部局部修整', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: '提交圈选区域' })).toBeEnabled()
  await page.getByRole('button', { name: '提交圈选区域' }).click()
  await expect(page.locator('.cli-job-heading strong')).toHaveText('待接手', { timeout: 10_000 })
  await expect(page.getByRole('progressbar', { name: '标注任务进度' })).toHaveAttribute('aria-valuenow', '0')
  await expect(page.getByText('标注已保存，等待当前任务接手')).toBeVisible()
  await expect(page.getByText(/告诉我“已提交”后我会接手执行/)).toBeVisible()

  await page.getByRole('button', { name: '取消' }).click()
  await expect(page.locator('.cli-job-heading strong')).toHaveText('已取消')

  await page.getByRole('button', { name: '重新载入 OriginalAnime V29' }).click()
  await expect(page.getByText('OriginalAnime V29 已载入；标注模式下拖动画闭合区域')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText('暂无区域')).toBeVisible()
  await expect(page.getByText('提交后等待当前任务接手执行')).toBeVisible()
  await page.locator('.inspector-scroll').evaluate((element) => { element.scrollTop = 0 })
  await page.screenshot({ path: '../.tmp/annotation-platform-implementation.png', fullPage: true })
  expect(externalRequests).toEqual([])
  expect(failedResponses).toEqual([])
  expect(browserErrors).toEqual([])
})
