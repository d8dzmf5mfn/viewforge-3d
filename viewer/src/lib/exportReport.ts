import { strToU8, zipSync } from 'fflate'
import type { FacePackage } from './types'

function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  anchor.style.display = 'none'
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  setTimeout(() => URL.revokeObjectURL(url), 5_000)
}

export function exportAcceptance(facePackage: FacePackage, screenshot: string): void {
  const json = JSON.stringify(facePackage.report, null, 2)
  const html = `<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>${facePackage.name} 验收报告</title>
<style>body{font:14px/1.6 -apple-system,BlinkMacSystemFont,sans-serif;max-width:1080px;margin:40px auto;color:#111827}img{width:100%;background:#11151a}pre{white-space:pre-wrap;background:#f5f7fa;padding:20px;border-radius:10px}</style>
<h1>${facePackage.name} 验收报告</h1><p>本地生成；未上传人脸数据。</p><img src="${screenshot}" alt="固定视角验收截图"><pre>${json.replaceAll('&', '&amp;').replaceAll('<', '&lt;')}</pre></html>`
  const bytes = Uint8Array.from(atob(screenshot.split(',')[1] ?? ''), (value) => value.charCodeAt(0))
  const archive = zipSync({
    'report.json': strToU8(json),
    'report.html': strToU8(html),
    'fixed-view.png': bytes,
  }, { level: 6 })
  download(
    new Blob([archive], { type: 'application/zip' }),
    `${facePackage.name}-acceptance.zip`,
  )
}
