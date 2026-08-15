const STAGES = {
  queued: { label: '等待执行', floor: 0 },
  intake: { label: '校验输入', floor: 4 },
  inspect: { label: '分析区域', floor: 12 },
  plan: { label: '制定修改方案', floor: 26 },
  edit: { label: '生成候选模型', floor: 42 },
  validate: { label: '几何质量检查', floor: 66 },
  render: { label: '生成六视图', floor: 82 },
  package: { label: '整理结果', floor: 94 },
  completed: { label: '已完成', floor: 100 },
  failed: { label: '执行失败', floor: 0 },
  cancelled: { label: '已取消', floor: 0 },
}

const STAGE_ORDER = Object.keys(STAGES)

function stageRank(stage) {
  return STAGE_ORDER.indexOf(stage)
}

function isInspectionCommand(command) {
  return /^\s*!?\s*(?:[a-z_][a-z0-9_]*=\S+\s+)*(?:\S*\/)?(?:cat|cmp|find|head|jq|ls|rg|sed|shasum|sha256sum|stat|tail|wc)\b/i.test(String(command ?? ''))
}

function invokesRenderer(command) {
  const segments = String(command ?? '').split(/(?:&&|\|\||;|\n)/)
  return segments.some((segment) => {
    const text = segment.trim().replace(/^(?:[a-z_][a-z0-9_]*=\S+\s+)*/i, '')
    if (/^(?:\S*\/)?(?:blender|montage)\b/i.test(text)) return true
    return /^(?:\S*\/)?python(?:3(?:\.\d+)?)?\s+(?:-[^\s]+\s+)*\S*(?:render|fixed.?view|six.?view|contact.?sheet)\S*\.py\b/i.test(text)
  })
}

function commandProgress(command) {
  const text = String(command ?? '').toLowerCase()
  if (isInspectionCommand(command)) {
    return { stage: 'inspect', progress: 14, detail: '正在读取标注、模型与既有方法' }
  }
  if (invokesRenderer(command)) {
    return { stage: 'render', progress: 84, detail: '正在生成固定六视图' }
  }
  if (/\b(package|manifest|atomic|rename|publish|deliver)\b|export.*\.glb/.test(text)) {
    return { stage: 'package', progress: 94, detail: '正在校验并整理输出文件' }
  }
  if (/self.?intersection|watertight|non.?manifold|normal.?change|topology|quality.?gate|\bqa\b|metrics/.test(text)) {
    return { stage: 'validate', progress: 68, detail: '正在运行局部几何与不变量检查' }
  }
  if (/refine_|polish_|fairing|candidate|py_compile|ruff check/.test(text)) {
    return { stage: 'edit', progress: 44, detail: '正在生成并校验局部修改候选' }
  }
  if (/probe_environment/.test(text)) {
    return { stage: 'inspect', progress: 14, detail: '正在读取标注、模型与既有方法' }
  }
  return null
}

function messageProgress(message) {
  const text = String(message ?? '')
  const marker = text.match(/\[PROGRESS\s+stage=(queued|intake|inspect|plan|edit|validate|render|package)\s+percent=(\d{1,3})\s+detail=([^\]]+)\]/i)
  if (marker) {
    const stage = marker[1].toLowerCase()
    const progress = Math.max(STAGES[stage].floor, Math.min(98, Number(marker[2])))
    return { stage, progress, detail: marker[3].trim().slice(0, 160) }
  }

  const lower = text.toLowerCase()
  if (/six.?view|fixed.?view|render/.test(lower)) return { stage: 'render', progress: 82, detail: '正在生成固定六视图' }
  if (/publish|packag|atomic|output directory|directory.*visible/.test(lower)) return { stage: 'package', progress: 94, detail: '正在整理不可变输出' }
  if (/gate|self.?intersection|normal.?change|watertight|quality|exact test/.test(lower)) return { stage: 'validate', progress: 67, detail: '正在执行几何质量检查' }
  if (/candidate|actual displacement|refinement will|local edit|polish/.test(lower)) return { stage: 'edit', progress: 43, detail: '正在生成局部修改候选' }
  if (/sufficient|mapped the annotation|plan|strategy/.test(lower)) return { stage: 'plan', progress: 28, detail: '标注已解析，正在制定修改方案' }
  return null
}

export function plannedSeconds(annotations = []) {
  const intentCost = { polish: 180, smooth: 240, lower: 360, raise: 360, protect: 60 }
  const regionSeconds = annotations.reduce((sum, annotation) => sum + (intentCost[annotation?.intent] ?? 240), 0)
  return Math.min(4_800, 780 + regionSeconds + Math.max(0, annotations.length - 1) * 120)
}

export function initialProgress(annotations = []) {
  return {
    stage: 'queued',
    stageLabel: STAGES.queued.label,
    progress: 0,
    progressDetail: '等待本机 Codex CLI',
    plannedSeconds: plannedSeconds(annotations),
    lastActivityAt: new Date().toISOString(),
  }
}

export function submittedProgress(annotations = []) {
  return {
    ...initialProgress(annotations),
    stageLabel: '等待接手',
    progressDetail: '标注已保存，等待当前任务接手',
  }
}

export function applyProgress(job, candidate) {
  if (!candidate || !STAGES[candidate.stage]) return
  const currentRank = stageRank(job.stage ?? 'queued')
  const nextRank = stageRank(candidate.stage)
  if (nextRank < currentRank && !['failed', 'cancelled'].includes(candidate.stage)) return
  job.stage = candidate.stage
  job.stageLabel = STAGES[candidate.stage].label
  job.progress = Math.max(job.progress ?? 0, STAGES[candidate.stage].floor, candidate.progress ?? 0)
  if (candidate.detail) job.progressDetail = candidate.detail
  job.lastActivityAt = new Date().toISOString()
}

export function applyEventProgress(job, event) {
  if (!event || typeof event !== 'object') return
  if (event.type === 'thread.started') applyProgress(job, { stage: 'intake', progress: 5, detail: 'CLI 已接收标注包' })
  if (event.type === 'turn.started') applyProgress(job, { stage: 'inspect', progress: 10, detail: '正在分析区域与保护约束' })
  if (event.type === 'turn.completed') applyProgress(job, { stage: 'package', progress: 97, detail: 'CLI 已完成，正在读取最终结果' })
  if (event.type === 'turn.failed') applyProgress(job, { stage: 'failed', progress: job.progress ?? 0, detail: 'CLI 执行失败' })

  const item = event.item
  if (event.type === 'item.started' && item?.type === 'command_execution') applyProgress(job, commandProgress(item.command))
  if (event.type === 'item.completed' && item?.type === 'agent_message') applyProgress(job, messageProgress(item.text))
}

export function finishProgress(job, state) {
  const stage = state === 'completed' ? 'completed' : state === 'cancelled' ? 'cancelled' : 'failed'
  applyProgress(job, {
    stage,
    progress: state === 'completed' ? 100 : job.progress ?? 0,
    detail: STAGES[stage].label,
  })
}

export function timingFor(job, nowMs = Date.now()) {
  const startedMs = job.startedAt ? Date.parse(job.startedAt) : null
  const finishedMs = job.finishedAt ? Date.parse(job.finishedAt) : null
  const elapsedSeconds = startedMs
    ? Math.max(0, Math.round(((finishedMs ?? nowMs) - startedMs) / 1_000))
    : 0

  if (job.state === 'completed') return { elapsedSeconds, estimatedSecondsRemaining: 0 }
  if (job.state === 'submitted') return { elapsedSeconds, estimatedSecondsRemaining: null }
  if (['failed', 'cancelled'].includes(job.state)) return { elapsedSeconds, estimatedSecondsRemaining: null }

  const progressRatio = Math.max(0.04, Math.min(0.98, (job.progress ?? 0) / 100))
  const dynamicTotal = elapsedSeconds > 30 ? elapsedSeconds / progressRatio : 0
  const expectedTotal = Math.max(job.plannedSeconds ?? 1_200, dynamicTotal)
  const remaining = Math.max(30, Math.min(7_200, expectedTotal - elapsedSeconds))
  return {
    elapsedSeconds,
    estimatedSecondsRemaining: Math.round(remaining / 30) * 30,
  }
}

export function progressPublicFields(job) {
  return {
    stage: job.stage ?? 'queued',
    stageLabel: job.stageLabel ?? STAGES.queued.label,
    progress: Math.max(0, Math.min(100, Math.round(job.progress ?? 0))),
    progressDetail: job.progressDetail ?? '',
    lastActivityAt: job.lastActivityAt,
    ...timingFor(job),
  }
}
