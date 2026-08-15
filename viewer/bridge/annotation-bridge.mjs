import { createServer } from 'node:http'
import { spawn, spawnSync } from 'node:child_process'
import { createHash, randomBytes } from 'node:crypto'
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  writeFileSync,
} from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildCodexExecArgs, classifyCodexFinalOutcome } from './codex-command.mjs'
import {
  applyEventProgress,
  applyProgress,
  finishProgress,
  initialProgress,
  progressPublicFields,
  submittedProgress,
} from './job-progress.mjs'

const bridgeDirectory = dirname(fileURLToPath(import.meta.url))
const projectRoot = resolve(bridgeDirectory, '../..')
const host = '127.0.0.1'
const port = Number(process.env.FACE3D_ANNOTATION_BRIDGE_PORT ?? 4174)
const dryRun = process.env.FACE3D_ANNOTATION_DRY_RUN === '1'
const executionMode = process.env.FACE3D_ANNOTATION_EXECUTION_MODE === 'auto' ? 'auto' : 'manual'
const jobRoot = resolve(
  process.env.FACE3D_ANNOTATION_JOB_ROOT
    ?? join(projectRoot, 'runs/annotation-jobs'),
)
const sourceModelInput = process.env.FACE3D_ANNOTATION_SOURCE_MODEL?.trim() ?? ''
const sourceModel = sourceModelInput ? resolve(projectRoot, sourceModelInput) : null
const sourceModelSha256 = process.env.FACE3D_ANNOTATION_SOURCE_SHA256?.trim().toLowerCase() ?? ''
const sourceVersion = process.env.FACE3D_ANNOTATION_SOURCE_VERSION?.trim() || 'source-v1'
const sourceRoute = process.env.FACE3D_ANNOTATION_ROUTE?.trim() || 'profile-loft-preview'
const subjectProfile = process.env.FACE3D_ANNOTATION_SUBJECT_PROFILE?.trim() || 'generic-object'
const realPerson = process.env.FACE3D_ANNOTATION_REAL_PERSON === 'true'
const sourceConfigured = Boolean(sourceModel && /^[a-f0-9]{64}$/.test(sourceModelSha256))
const jobs = new Map()
const queue = []
let activeJobId = null
let activeChild = null

mkdirSync(jobRoot, { recursive: true })

function atomicJson(path, value) {
  const temporary = `${path}.${process.pid}.tmp`
  writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 })
  renameSync(temporary, path)
}

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

function currentCodex() {
  if (dryRun) return { available: true, version: 'codex-cli dry-run' }
  const result = spawnSync('codex', ['--version'], {
    cwd: projectRoot,
    encoding: 'utf8',
    timeout: 5_000,
    shell: false,
  })
  const text = `${result.stdout ?? ''}\n${result.stderr ?? ''}`
  const version = text.match(/codex-cli[^\n]*/)?.[0]?.trim() ?? null
  const codeModeHost = spawnSync('codex-code-mode-host', ['--help'], {
    cwd: projectRoot,
    encoding: 'utf8',
    timeout: 5_000,
    shell: false,
  })
  return {
    available: result.status === 0 && Boolean(version) && codeModeHost.status === 0,
    version,
  }
}

function publicJob(job) {
  return {
    id: job.id,
    state: job.state,
    createdAt: job.createdAt,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt,
    annotationCount: job.annotationCount,
    summary: job.summary,
    logs: job.logs,
    finalMessage: job.finalMessage,
    outputDirectory: job.outputDirectory,
    error: job.error,
    ...progressPublicFields(job),
  }
}

function persistStatus(job) {
  atomicJson(join(job.directory, 'status.json'), publicJob(job))
}

const persistedStates = new Set(['submitted', 'queued', 'running', 'completed', 'failed', 'cancelled'])
const restorableStates = new Set(['submitted', 'completed', 'failed', 'cancelled'])
const persistedJobFields = [
  'state',
  'createdAt',
  'startedAt',
  'finishedAt',
  'annotationCount',
  'summary',
  'logs',
  'finalMessage',
  'outputDirectory',
  'error',
  'stage',
  'stageLabel',
  'progress',
  'progressDetail',
  'lastActivityAt',
  'plannedSeconds',
]

function readPersistedJob(directory, expectedId) {
  try {
    const value = JSON.parse(readFileSync(join(directory, 'status.json'), 'utf8'))
    if (value?.id !== expectedId || !persistedStates.has(value.state)) return null
    const restored = { id: expectedId, directory }
    for (const field of persistedJobFields) {
      if (value[field] !== undefined) restored[field] = value[field]
    }
    if (!Array.isArray(restored.logs)) restored.logs = []
    return restored
  } catch {
    return null
  }
}

function refreshPersistedJob(job) {
  const persisted = readPersistedJob(job.directory, job.id)
  if (persisted) Object.assign(job, persisted)
}

function restorePersistedJobs() {
  for (const entry of readdirSync(jobRoot, { withFileTypes: true })) {
    if (!entry.isDirectory() || !/^[A-Za-z0-9-]+$/.test(entry.name)) continue
    const persisted = readPersistedJob(join(jobRoot, entry.name), entry.name)
    if (persisted && restorableStates.has(persisted.state)) jobs.set(entry.name, persisted)
  }
}

function appendLog(job, line) {
  const safe = String(line).replace(/[\r\n]+/g, ' ').trim().slice(0, 480)
  if (!safe) return
  job.logs.push(safe)
  if (job.logs.length > 120) job.logs.splice(0, job.logs.length - 120)
  persistStatus(job)
}

function isFiniteTuple(value, length) {
  return Array.isArray(value) && value.length === length && value.every(Number.isFinite)
}

export function validateAnnotationPackage(value) {
  const errors = []
  if (!value || typeof value !== 'object') return ['annotationPackage must be an object']
  if (!sourceConfigured) errors.push('bridge source model and SHA-256 are not configured')
  if (value.schemaVersion !== 1) errors.push('unsupported schemaVersion')
  if (value.route !== sourceRoute) errors.push(`route must be ${sourceRoute}`)
  if (value.outputState !== 'annotation-input') errors.push('outputState must be annotation-input')
  if (sourceModel && resolve(String(value.model?.modelPath ?? '')) !== sourceModel) errors.push(`modelPath is not the locked ${sourceVersion} source`)
  if (value.model?.modelSha256 !== sourceModelSha256) errors.push(`modelSha256 does not match locked ${sourceVersion}`)
  if (value.model?.realPerson !== realPerson || value.model?.subjectProfile !== subjectProfile) errors.push(`subject profile must remain ${subjectProfile}`)
  if (value.pickingProxy?.role !== 'qa-only-ray-picking-proxy' || value.pickingProxy?.surfaceGenerated !== false) errors.push('picking proxy provenance is invalid')
  if (sourceModel && resolve(String(value.pickingProxy?.source?.model ?? '')) !== sourceModel) errors.push(`picking proxy source is not locked ${sourceVersion}`)
  if (value.pickingProxy?.source?.modelSha256 !== sourceModelSha256) errors.push(`picking proxy source SHA-256 does not match locked ${sourceVersion}`)
  const annotations = value.annotations
  if (!Array.isArray(annotations) || annotations.length < 1 || annotations.length > 64) errors.push('annotations must contain 1 to 64 regions')
  if (Array.isArray(annotations)) annotations.forEach((annotation, index) => {
    if (!['polish', 'smooth', 'lower', 'raise', 'protect'].includes(annotation?.intent)) errors.push(`annotation ${index} intent is invalid`)
    if (typeof annotation?.note !== 'string' || annotation.note.trim().length < 2 || annotation.note.length > 2_000) errors.push(`annotation ${index} note is invalid`)
    if (!Array.isArray(annotation?.screenPath) || annotation.screenPath.length < 3 || annotation.screenPath.length > 20_000) errors.push(`annotation ${index} screenPath is invalid`)
    if (!Array.isArray(annotation?.surfacePath) || annotation.surfacePath.length < 3 || annotation.surfacePath.length > 20_000) errors.push(`annotation ${index} surfacePath is invalid`)
    if (Array.isArray(annotation?.surfacePath)) annotation.surfacePath.forEach((sample, sampleIndex) => {
      if (!isFiniteTuple(sample?.position, 3) || !isFiniteTuple(sample?.normal, 3)) errors.push(`annotation ${index} surface sample ${sampleIndex} is invalid`)
    })
    if (!Number.isFinite(annotation?.radius) || annotation.radius < 1 || annotation.radius > 256) errors.push(`annotation ${index} radius is invalid`)
    if (!Number.isFinite(annotation?.strength) || annotation.strength <= 0 || annotation.strength > 1) errors.push(`annotation ${index} strength is invalid`)
  })
  if (typeof value.globalNotes !== 'string' || value.globalNotes.length > 8_000) errors.push('globalNotes is invalid')
  return errors.slice(0, 30)
}

function parseScreenshot(dataUrl) {
  if (!dataUrl) return null
  const match = /^data:image\/png;base64,([A-Za-z0-9+/=]+)$/.exec(dataUrl)
  if (!match) throw new Error('screenshotDataUrl must be a PNG data URL')
  const buffer = Buffer.from(match[1], 'base64')
  if (buffer.length > 8 * 1024 * 1024) throw new Error('screenshot exceeds 8 MB')
  if (buffer.length < 8 || buffer.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a') throw new Error('screenshot PNG signature is invalid')
  return buffer
}

function jobPrompt(job) {
  const annotationPath = join(job.directory, 'annotation.json')
  const screenshotPath = join(job.directory, 'viewport.png')
  return `You are the local geometry-modification worker for Face3D Modeling Toolkit.

Read the immutable user annotation package:
${annotationPath}
${existsSync(screenshotPath) ? `Also inspect its viewport evidence image:\n${screenshotPath}\n` : ''}
Locked source model:
${sourceModel}
SHA-256: ${sourceModelSha256}

Execute the requested visible-region edits under the installed reconstruct-3d-from-multiview skill and the repository's existing topology-preserving refinement methods.

Progress reporting:
- Before each phase, send one concise agent message in this exact form:
  [PROGRESS stage=<inspect|plan|edit|validate|render|package> percent=<0-98> detail=<short Chinese status>]
- Percent must be monotonic and evidence-based. Never announce 100; the bridge owns completion.

Hard boundaries:
- This is route=${sourceRoute}, subjectProfile=${subjectProfile}, realPerson=${realPerson}, and output remains preview.
- Never overwrite or mutate ${sourceVersion} or an existing run. Create a new immutable, versioned output directory.
- Surface positions in annotation.json are ${sourceVersion} model coordinates. The decimated picking proxy is QA-only; never edit, render, validate, or deliver it as the source surface.
- Manual regions authorize local visible-surface changes only. They do not authorize invented posterior Z or unseen geometry.
- Preserve the accepted silhouette, authored features, topology, and protected regions unless a region explicitly targets them. Keep exact feature-lock masks where relevant.
- Preserve topology, face/vertex order, UV arrays, material partitions, textures, and watertightness unless the request explicitly and unavoidably conflicts; if it conflicts, stop and report blocked instead of guessing.
- Execute only the requested region-local operation. Do not add smoothing unless the annotation explicitly requests it.
- intent=polish means topology-preserving removal of small terraces/ripples only. Preserve the local silhouette, low-frequency facial volume, feature relief, Y coordinates, UVs, material partitions, and vertex/face order. It is stricter than general smooth.
- For immutable-source local edits, avoid repeating an exhaustive all-vs-all self-intersection run. Cache any source-global result by ${sourceVersion} SHA-256, then prove the candidate with exact moved/swept-triangle BVH tests against the full-resolution source plus the normal topology/watertightness gates. The decimated picking proxy is never valid final geometry evidence.
- Produce fixed-view QA at front, left/right 45, left/right 90, and rear, plus metrics proving source unchanged and invariants.
- Do not edit viewer/, this bridge, skills, or unrelated files. Do not use Sentry. Do not use network access.

Finish with a concise report containing outcome state, new output/model paths, tests/gates, and residual uncertainty. If the annotation is insufficient, make no geometry change and explain exactly what additional annotation is needed.`
}

function eventSummary(event) {
  if (!event || typeof event !== 'object') return null
  if (event.type === 'thread.started') return '> Codex CLI 已接收标注包'
  if (event.type === 'turn.started') return '> 开始分析区域与约束'
  if (event.type === 'turn.completed') return '> CLI 修改流程结束'
  if (event.type === 'turn.failed') return `> CLI 失败：${event.error?.message ?? 'unknown error'}`
  const item = event.item
  if (event.type === 'item.started' && item?.type === 'command_execution') return `> 执行：${String(item.command ?? '').slice(0, 160)}`
  if (event.type === 'item.completed' && item?.type === 'command_execution') return `> 命令${item.status === 'completed' ? '完成' : item.status ?? '结束'}`
  if (event.type === 'item.completed' && item?.type === 'agent_message') return `> ${String(item.text ?? '').slice(0, 220)}`
  return null
}

function completeJob(job, state, fields = {}) {
  job.state = state
  job.finishedAt = new Date().toISOString()
  Object.assign(job, fields)
  finishProgress(job, state)
  persistStatus(job)
  activeJobId = null
  activeChild = null
  queueMicrotask(processQueue)
}

function runDryJob(job) {
  applyProgress(job, { stage: 'intake', progress: 8, detail: '标注包已校验' })
  appendLog(job, '> Codex CLI 已接收标注包')
  setTimeout(() => {
    applyProgress(job, { stage: 'edit', progress: 52, detail: '正在验证任务连接' })
    appendLog(job, '> 分析标注区域与保护约束')
  }, 120)
  setTimeout(() => {
    applyProgress(job, { stage: 'package', progress: 95, detail: '正在整理验证结果' })
    appendLog(job, '> DRY RUN：未修改任何模型文件')
  }, 240)
  setTimeout(() => completeJob(job, 'completed', {
    finalMessage: `DRY RUN 通过：标注包、截图和 CLI 队列连接有效；${sourceVersion} 未修改。`,
  }), 360)
}

function runCodexJob(job) {
  const finalPath = join(job.directory, 'final-message.md')
  const eventPath = join(job.directory, 'codex-events.jsonl')
  const eventStream = createWriteStream(eventPath, { flags: 'a', mode: 0o600 })
  const args = buildCodexExecArgs({ projectRoot, finalPath })
  const child = spawn('codex', args, { cwd: projectRoot, stdio: ['pipe', 'pipe', 'pipe'], shell: false })
  activeChild = child
  child.stdin.end(jobPrompt(job))
  let stdoutBuffer = ''
  let stderrBuffer = ''
  child.stdout.on('data', (chunk) => {
    eventStream.write(chunk)
    stdoutBuffer += chunk.toString('utf8')
    const lines = stdoutBuffer.split('\n')
    stdoutBuffer = lines.pop() ?? ''
    lines.forEach((line) => {
      try {
        const event = JSON.parse(line)
        applyEventProgress(job, event)
        const summary = eventSummary(event)
        if (summary) appendLog(job, summary)
        else persistStatus(job)
      } catch {
        // Raw output remains in codex-events.jsonl; malformed partial lines are ignored here.
      }
    })
  })
  child.stderr.on('data', (chunk) => {
    stderrBuffer += chunk.toString('utf8')
    if (stderrBuffer.length > 16_000) stderrBuffer = stderrBuffer.slice(-16_000)
  })
  child.on('error', (error) => {
    eventStream.end()
    completeJob(job, 'failed', { error: error.message })
  })
  child.on('close', (code, signal) => {
    eventStream.end()
    if (job.state === 'cancelled') {
      activeJobId = null
      activeChild = null
      queueMicrotask(processQueue)
      return
    }
    if (code === 0 && existsSync(finalPath)) {
      const finalMessage = readFileSync(finalPath, 'utf8').trim().slice(0, 12_000)
      const state = classifyCodexFinalOutcome(finalMessage)
      completeJob(job, state, {
        finalMessage,
        ...(state === 'failed'
          ? { error: 'Codex CLI reported a blocked or failed outcome; no successful model output was produced.' }
          : {}),
      })
    } else {
      completeJob(job, 'failed', {
        error: `codex exited with code ${code ?? 'null'}${signal ? ` (${signal})` : ''}: ${stderrBuffer.trim().slice(-2_000)}`,
      })
    }
  })
}

function processQueue() {
  if (activeJobId) return
  const nextId = queue.shift()
  if (!nextId) return
  const job = jobs.get(nextId)
  if (!job || job.state !== 'queued') return queueMicrotask(processQueue)
  activeJobId = job.id
  job.state = 'running'
  job.startedAt = new Date().toISOString()
  applyProgress(job, { stage: 'intake', progress: 4, detail: '正在校验锁定模型与标注包' })
  persistStatus(job)
  if (dryRun) runDryJob(job)
  else runCodexJob(job)
}

function sendJson(response, status, value) {
  const body = `${JSON.stringify(value)}\n`
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  })
  response.end(body)
}

function readBody(request, maximumBytes = 12 * 1024 * 1024) {
  return new Promise((resolveBody, rejectBody) => {
    let size = 0
    const chunks = []
    request.on('data', (chunk) => {
      size += chunk.length
      if (size > maximumBytes) {
        rejectBody(new Error('request body exceeds 12 MB'))
        request.destroy()
        return
      }
      chunks.push(chunk)
    })
    request.on('end', () => {
      try { resolveBody(JSON.parse(Buffer.concat(chunks).toString('utf8'))) }
      catch { rejectBody(new Error('invalid JSON body')) }
    })
    request.on('error', rejectBody)
  })
}

restorePersistedJobs()

const server = createServer(async (request, response) => {
  const url = new URL(request.url ?? '/', `http://${host}:${port}`)
  if (request.method === 'GET' && url.pathname === '/api/annotation/health') {
    const codex = currentCodex()
    sendJson(response, 200, {
      ok: true,
      service: 'face3d-annotation-bridge',
      host,
      codexAvailable: codex.available,
      codexVersion: codex.version,
      activeJobId,
      queueLength: queue.length,
      dryRun,
      executionMode,
      sourceConfigured,
    })
    return
  }
  if (request.method === 'POST' && url.pathname === '/api/annotation/jobs') {
    try {
      if (executionMode === 'auto' && !currentCodex().available) return sendJson(response, 503, { error: 'Codex CLI or codex-code-mode-host is unavailable' })
      const body = await readBody(request)
      const errors = validateAnnotationPackage(body.annotationPackage)
      if (errors.length > 0) return sendJson(response, 422, { error: errors.join('; '), errors })
      if (!sourceModel || !existsSync(sourceModel) || sha256(sourceModel) !== sourceModelSha256) return sendJson(response, 409, { error: `locked ${sourceVersion} source is missing or its SHA-256 changed` })
      const screenshot = parseScreenshot(body.screenshotDataUrl)
      const now = new Date()
      const id = `${now.toISOString().replace(/[:.]/g, '-')}-${randomBytes(4).toString('hex')}`
      const directory = join(jobRoot, id)
      mkdirSync(directory, { recursive: false, mode: 0o700 })
      atomicJson(join(directory, 'annotation.json'), body.annotationPackage)
      if (screenshot) writeFileSync(join(directory, 'viewport.png'), screenshot, { mode: 0o600 })
      const job = {
        id,
        directory,
        state: executionMode === 'manual' ? 'submitted' : 'queued',
        createdAt: now.toISOString(),
        annotationCount: body.annotationPackage.annotations.length,
        summary: body.annotationPackage.annotations.map((annotation) => annotation.label).join('、').slice(0, 180),
        logs: executionMode === 'manual'
          ? ['> 标注包已校验并持久化', '> 等待当前任务人工接手']
          : ['> 标注包已校验并持久化', '> 等待本机 Codex CLI'],
        outputDirectory: directory,
        ...(executionMode === 'manual'
          ? submittedProgress(body.annotationPackage.annotations)
          : initialProgress(body.annotationPackage.annotations)),
      }
      jobs.set(id, job)
      persistStatus(job)
      if (executionMode === 'auto') {
        queue.push(id)
        processQueue()
      }
      sendJson(response, 202, publicJob(job))
    } catch (error) {
      sendJson(response, 400, { error: error instanceof Error ? error.message : String(error) })
    }
    return
  }
  const jobMatch = /^\/api\/annotation\/jobs\/([A-Za-z0-9-]+)$/.exec(url.pathname)
  if (request.method === 'GET' && jobMatch) {
    const job = jobs.get(jobMatch[1])
    if (!job) return sendJson(response, 404, { error: 'job not found' })
    refreshPersistedJob(job)
    return sendJson(response, 200, publicJob(job))
  }
  const cancelMatch = /^\/api\/annotation\/jobs\/([A-Za-z0-9-]+)\/cancel$/.exec(url.pathname)
  if (request.method === 'POST' && cancelMatch) {
    const job = jobs.get(cancelMatch[1])
    if (!job) return sendJson(response, 404, { error: 'job not found' })
    if (job.state === 'submitted') {
      job.state = 'cancelled'
      job.finishedAt = new Date().toISOString()
      finishProgress(job, 'cancelled')
      appendLog(job, '> 用户取消待接手标注')
      persistStatus(job)
    } else if (job.state === 'queued') {
      const index = queue.indexOf(job.id)
      if (index >= 0) queue.splice(index, 1)
      job.state = 'cancelled'
      job.finishedAt = new Date().toISOString()
      finishProgress(job, 'cancelled')
      persistStatus(job)
    } else if (job.state === 'running' && activeJobId === job.id) {
      job.state = 'cancelled'
      job.finishedAt = new Date().toISOString()
      finishProgress(job, 'cancelled')
      appendLog(job, '> 用户取消任务')
      activeChild?.kill('SIGTERM')
      persistStatus(job)
    }
    return sendJson(response, 200, publicJob(job))
  }
  sendJson(response, 404, { error: 'not found' })
})

server.on('clientError', (_error, socket) => socket.end('HTTP/1.1 400 Bad Request\r\n\r\n'))
server.listen(port, host, () => {
  console.log(`[face3d-annotation-bridge] http://${host}:${port} dryRun=${dryRun} jobs=${jobRoot}`)
})

function shutdown() {
  activeChild?.kill('SIGTERM')
  server.close(() => process.exit(0))
  setTimeout(() => process.exit(1), 3_000).unref()
}
process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)
