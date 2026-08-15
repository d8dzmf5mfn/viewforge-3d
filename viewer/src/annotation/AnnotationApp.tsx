import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Check,
  CircleDot,
  Clock3,
  Download,
  Eye,
  EyeOff,
  LoaderCircle,
  LockKeyhole,
  MousePointer2,
  OctagonX,
  Orbit,
  RotateCcw,
  RefreshCw,
  Send,
  Shield,
  Sparkles,
  Trash2,
  Undo2,
} from 'lucide-react'
import { AnnotationViewport, type AnnotationViewportHandle } from './AnnotationViewport'
import { cancelCliJob, createCliJob, fetchCliHealth, fetchCliJob } from './api'
import { createAnnotationPackage, downloadAnnotationPackage, validateAnnotationPackage } from './package'
import type {
  AnnotationIntent,
  CliHealth,
  CliJob,
  DraftStats,
  EditorMode,
  ModelMetadata,
  ProxyMetadata,
  SurfaceAnnotation,
  ViewPreset,
} from './types'

const intentOptions: Array<{
  value: AnnotationIntent
  label: string
  description: string
}> = [
  { value: 'polish', label: '抛光', description: '只去除细小波纹与台阶，不改变轮廓' },
  { value: 'smooth', label: '打磨', description: '平顺局部体块与过渡，可轻微调整弧度' },
  { value: 'lower', label: '压低', description: '沿表面法线小幅向内调整' },
  { value: 'raise', label: '抬高', description: '沿表面法线小幅向外调整' },
  { value: 'protect', label: '保护', description: '作为硬锁区，禁止周边操作带动' },
]

const presetOptions: Array<{ value: ViewPreset; label: string }> = [
  { value: 'front', label: '正面' },
  { value: 'left45', label: '左45°' },
  { value: 'right45', label: '右45°' },
  { value: 'left90', label: '左侧' },
  { value: 'right90', label: '右侧' },
  { value: 'rear', label: '背面' },
]

const colorByIntent: Record<AnnotationIntent, string> = {
  polish: '#e8c86f',
  smooth: '#f36c5c',
  lower: '#f0a24d',
  raise: '#65b8ff',
  protect: '#8ce0b4',
}

const intentLabel = (intent: AnnotationIntent) => intentOptions.find((option) => option.value === intent)?.label ?? intent
const defaultGlobalNotes = '保持动漫美术素材的脸部弧度，不要磨平鼻子、眼眶、嘴线和耳朵。'
const storageRevision = 'v2-polish-progress'

function storageKey(modelSha256: string): string {
  return `viewforge3d-annotations:${storageRevision}:${modelSha256}`
}

function reloadUrl(url: string, revision: number): string {
  return `${url}${url.includes('?') ? '&' : '?'}reload=${revision}`
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—'
  if (seconds === 0) return '0 分钟'
  if (seconds < 60) return '不足 1 分钟'
  const minutes = Math.ceil(seconds / 60)
  if (minutes < 60) return `约 ${minutes} 分钟`
  const hours = Math.floor(minutes / 60)
  const remainder = minutes % 60
  return remainder > 0 ? `约 ${hours} 小时 ${remainder} 分钟` : `约 ${hours} 小时`
}

function statusLabel(state: CliJob['state']): string {
  return {
    submitted: '待接手',
    queued: '排队中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }[state]
}

function useCliHealth(): CliHealth | null {
  const [health, setHealth] = useState<CliHealth | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    const refresh = () => {
      fetchCliHealth(controller.signal).then(setHealth).catch(() => setHealth(null))
    }
    refresh()
    const interval = window.setInterval(refresh, 4_000)
    return () => {
      controller.abort()
      window.clearInterval(interval)
    }
  }, [])
  return health
}

export function AnnotationApp() {
  const viewportRef = useRef<AnnotationViewportHandle>(null)
  const [model, setModel] = useState<ModelMetadata | null>(null)
  const [proxy, setProxy] = useState<ProxyMetadata | null>(null)
  const [modelReady, setModelReady] = useState(false)
  const [modelRevision, setModelRevision] = useState(0)
  const [mode, setMode] = useState<EditorMode>('annotate')
  const [preset, setPreset] = useState<ViewPreset>('left45')
  const [intent, setIntent] = useState<AnnotationIntent>('smooth')
  const [radius, setRadius] = useState(32)
  const [strength, setStrength] = useState(0.45)
  const [mirror, setMirror] = useState(true)
  const [showHits, setShowHits] = useState(true)
  const [label, setLabel] = useState('脸部局部修整')
  const [note, setNote] = useState('')
  const [globalNotes, setGlobalNotes] = useState(defaultGlobalNotes)
  const [annotations, setAnnotations] = useState<SurfaceAnnotation[]>([])
  const [draftStats, setDraftStats] = useState<DraftStats>({ drawing: false, screenPoints: 0, surfaceHits: 0 })
  const [pointerWorld, setPointerWorld] = useState<[number, number, number] | null>(null)
  const [message, setMessage] = useState('在模型表面圈出一个闭合区域')
  const [job, setJob] = useState<CliJob | null>(null)
  const health = useCliHealth()

  useEffect(() => {
    fetch('/annotation/model.json')
      .then((response) => response.json() as Promise<ModelMetadata>)
      .then(async (modelMetadata) => {
        const proxyMetadata = await fetch(modelMetadata.pickProxyReportUrl)
          .then((response) => response.json() as Promise<ProxyMetadata>)
        return [modelMetadata, proxyMetadata] as const
      })
      .then(([modelMetadata, proxyMetadata]) => {
      setModel(modelMetadata)
      setProxy(proxyMetadata)
      const saved = localStorage.getItem(storageKey(modelMetadata.modelSha256))
      if (saved) {
        try {
          const parsed = JSON.parse(saved) as { annotations: SurfaceAnnotation[]; globalNotes: string }
          setAnnotations(parsed.annotations ?? [])
          if (parsed.globalNotes) setGlobalNotes(parsed.globalNotes)
        } catch {
          localStorage.removeItem(storageKey(modelMetadata.modelSha256))
        }
      }
      }).catch((error: unknown) => setMessage(error instanceof Error ? error.message : String(error)))
  }, [])

  useEffect(() => {
    if (!model) return
    localStorage.setItem(
      storageKey(model.modelSha256),
      JSON.stringify({ annotations, globalNotes }),
    )
  }, [annotations, globalNotes, model])

  const activeJobId = job && ['submitted', 'queued', 'running'].includes(job.state) ? job.id : null

  useEffect(() => {
    if (!activeJobId) return
    const controller = new AbortController()
    const interval = window.setInterval(() => {
      fetchCliJob(activeJobId, controller.signal).then(setJob).catch(() => undefined)
    }, 1_200)
    return () => {
      controller.abort()
      window.clearInterval(interval)
    }
  }, [activeJobId])

  const activeColor = colorByIntent[intent]
  const packageValue = useMemo(() => {
    if (!model || !proxy) return null
    return createAnnotationPackage(model, proxy, annotations, globalNotes)
  }, [annotations, globalNotes, model, proxy])
  const packageErrors = packageValue ? validateAnnotationPackage(packageValue) : ['模型尚未载入']

  const handleReady = useCallback(() => {
    setModelReady(true)
    setMessage(`${model?.displayName ?? '当前模型'} 已载入；标注模式下拖动画闭合区域`)
  }, [model])
  const handleDraftStats = useCallback((stats: DraftStats) => setDraftStats(stats), [])
  const handlePointerWorld = useCallback((position: [number, number, number] | null) => setPointerWorld(position), [])

  const finishAnnotation = () => {
    if (!note.trim()) {
      setMessage('先填写给 Codex 的具体修改说明')
      return
    }
    const annotation = viewportRef.current?.finishDraft({
      label: label.trim() || `${intentLabel(intent)}区域`,
      intent,
      color: activeColor,
      radius,
      strength,
      mirror,
      note: note.trim(),
    })
    if (!annotation) {
      setMessage('区域至少需要 3 个表面命中点；请重新圈选')
      return
    }
    setAnnotations((current) => [...current, annotation])
    setNote('')
    setMessage(`已保存“${annotation.label}”，可继续添加或提交`)
  }

  const savePackage = () => {
    if (!packageValue) return
    if (packageErrors.length > 0) {
      setMessage(packageErrors[0] ?? '标注包不完整')
      return
    }
    downloadAnnotationPackage(packageValue)
    setMessage('标注包已下载；模型未被修改')
  }

  const submitForExecution = async () => {
    if (!packageValue) return
    if (packageErrors.length > 0) {
      setMessage(packageErrors[0] ?? '标注包不完整')
      return
    }
    try {
      const created = await createCliJob(packageValue, viewportRef.current?.capture() ?? '')
      setJob(created)
      setMessage(`标注 ${created.id} 已提交；告诉我“已提交”后我会接手执行`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error))
    }
  }

  const reloadLockedModel = () => {
    if (!model) return
    if (job && ['submitted', 'queued', 'running'].includes(job.state)) {
      setMessage('请先取消当前标注任务')
      return
    }
    localStorage.removeItem(storageKey(model.modelSha256))
    viewportRef.current?.clearDraft()
    setAnnotations([])
    setDraftStats({ drawing: false, screenPoints: 0, surfaceHits: 0 })
    setPointerWorld(null)
    setNote('')
    setGlobalNotes(defaultGlobalNotes)
    setJob(null)
    setPreset('left45')
    setMode('annotate')
    setModelReady(false)
    setModelRevision((current) => current + 1)
    setMessage(`正在从锁定源重新载入 ${model.displayName}…`)
  }

  if (!model || !proxy) {
    return (
      <main className="annotation-loading">
        <LoaderCircle className="spin" />
        <span>读取当前标注环境…</span>
      </main>
    )
  }

  return (
    <main className="annotation-app">
      <header className="annotation-header">
        <div className="brand-block">
          <strong>ViewForge 3D</strong>
          <span className="brand-divider" />
          <span>区域标注与修改</span>
        </div>
        <div className={`connection-state ${health && (health.executionMode === 'manual' || health.codexAvailable) ? 'is-online' : ''}`}>
          <span className="connection-dot" />
          {health?.executionMode === 'manual'
            ? '人工接手模式 · 圈选后由当前任务执行'
            : health?.codexAvailable
              ? `本机 CLI 已连接 · ${health.codexVersion}`
              : '本机 CLI 未连接'}
        </div>
      </header>

      <aside className="annotation-left-rail">
        <section>
          <h2>当前模型</h2>
          <div className="model-card">
            <div className="model-orb"><LockKeyhole size={18} /></div>
            <div>
              <strong>{model.displayName}</strong>
              <span>{model.displayName} 已锁定</span>
            </div>
          </div>
          <dl className="model-facts">
            <div><dt>路线</dt><dd>Profile Loft Preview</dd></div>
            <div><dt>状态</dt><dd>{model.outputState}</dd></div>
            <div><dt>SHA</dt><dd>{model.modelSha256.slice(0, 9)}…</dd></div>
          </dl>
          <button
            className="model-reload-button"
            onClick={reloadLockedModel}
            disabled={Boolean(job && ['submitted', 'queued', 'running'].includes(job.state))}
          ><RefreshCw size={14} />重新载入 {model.displayName}</button>
        </section>

        <section>
          <h2>视角</h2>
          <div className="preset-grid">
            {presetOptions.map((option) => (
              <button
                key={option.value}
                className={preset === option.value ? 'selected' : ''}
                onClick={() => setPreset(option.value)}
              >{option.label}</button>
            ))}
          </div>
        </section>

        <section className="annotation-list-section">
          <div className="section-heading-row">
            <h2>标注</h2>
            <span>{annotations.length}</span>
          </div>
          <div className="annotation-list">
            {annotations.length === 0 ? (
              <div className="annotation-empty">暂无区域<br />在中间模型上拖动画圈</div>
            ) : annotations.map((annotation, index) => (
              <div className="annotation-row" key={annotation.id}>
                <button
                  className="visibility-button"
                  aria-label={annotation.visible ? '隐藏标注' : '显示标注'}
                  onClick={() => setAnnotations((current) => current.map((item) => item.id === annotation.id ? { ...item, visible: !item.visible } : item))}
                >{annotation.visible ? <Eye size={15} /> : <EyeOff size={15} />}</button>
                <span className="annotation-swatch" style={{ background: annotation.color }} />
                <div><strong>{annotation.label}</strong><span>{intentLabel(annotation.intent)} · {annotation.surfacePath.length} 点</span></div>
                <button
                  className="delete-button"
                  aria-label={`删除 ${annotation.label}`}
                  onClick={() => setAnnotations((current) => current.filter((item) => item.id !== annotation.id))}
                ><Trash2 size={14} /></button>
                <span className="annotation-index">{String(index + 1).padStart(2, '0')}</span>
              </div>
            ))}
          </div>
        </section>

        <button
          className="clear-all-button"
          disabled={annotations.length === 0}
          onClick={() => {
            setAnnotations([])
            viewportRef.current?.clearDraft()
            setMessage(`全部标注已清空；${model.displayName} 未改变`)
          }}
        ><RotateCcw size={15} /> 清空全部标注</button>
      </aside>

      <section className="annotation-canvas-shell">
        <div className="canvas-toolbar">
          <div className="mode-switch" aria-label="编辑模式">
            <button className={mode === 'orbit' ? 'selected' : ''} onClick={() => setMode('orbit')}>
              <Orbit size={16} />观察
            </button>
            <button className={mode === 'annotate' ? 'selected' : ''} onClick={() => setMode('annotate')}>
              <MousePointer2 size={16} />标注
            </button>
          </div>
          <div className="canvas-actions">
            <button onClick={() => viewportRef.current?.undoDraft()} disabled={draftStats.screenPoints === 0}>
              <Undo2 size={16} />撤销路径
            </button>
            <label><input type="checkbox" checked={showHits} onChange={(event) => setShowHits(event.target.checked)} />显示命中点</label>
          </div>
        </div>
        <AnnotationViewport
          key={modelRevision}
          ref={viewportRef}
          modelUrl={reloadUrl(model.modelUrl, modelRevision)}
          pickProxyUrl={reloadUrl(model.pickProxyUrl, modelRevision)}
          mode={mode}
          preset={preset}
          annotations={annotations}
          activeColor={activeColor}
          radius={radius}
          showHits={showHits}
          onReady={handleReady}
          onDraftStats={handleDraftStats}
          onPointerWorld={handlePointerWorld}
        />
        {!modelReady && <div className="model-loading"><LoaderCircle className="spin" />载入 47 MB {model.displayName} 高模…</div>}
        <div className="canvas-status">
          <span>{mode === 'annotate' ? '拖动圈选 · 视角已锁定' : '拖动旋转 · 滚轮缩放'}</span>
          <span>屏幕点 {draftStats.screenPoints} · 表面命中 {draftStats.surfaceHits}</span>
          <span>{pointerWorld ? `X ${pointerWorld[0].toFixed(3)}  Y ${pointerWorld[1].toFixed(3)}  Z ${pointerWorld[2].toFixed(3)}` : '等待表面命中'}</span>
        </div>
      </section>

      <aside className="annotation-inspector">
        <div className="inspector-scroll">
          <section>
            <h2>修改意图</h2>
            <div className="intent-list">
              {intentOptions.map((option) => (
                <button
                  key={option.value}
                  className={intent === option.value ? 'selected' : ''}
                  onClick={() => setIntent(option.value)}
                >
                  <span className="intent-radio"><span /></span>
                  <span><strong>{option.label}</strong><small>{option.description}</small></span>
                </button>
              ))}
            </div>
          </section>

          <section className="control-section">
            <label className="control-label"><span>区域名称</span></label>
            <input value={label} onChange={(event) => setLabel(event.target.value)} maxLength={40} />
            <label className="control-label"><span>笔刷半径</span><output>{radius} px</output></label>
            <input type="range" min="12" max="96" value={radius} onChange={(event) => setRadius(Number(event.target.value))} />
            <label className="control-label"><span>影响强度</span><output>{strength.toFixed(2)}</output></label>
            <input type="range" min="0.05" max="1" step="0.05" value={strength} onChange={(event) => setStrength(Number(event.target.value))} />
            <label className="toggle-row">
              <span><CircleDot size={15} />镜像到另一侧</span>
              <input type="checkbox" checked={mirror} onChange={(event) => setMirror(event.target.checked)} />
            </label>
          </section>

          <section>
            <label className="textarea-label" htmlFor="annotation-note">给当前任务的说明</label>
            <textarea
              id="annotation-note"
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder={intent === 'polish'
                ? '例如：只抛光圈选区域的细小横纹，保持原有脸部弧度与五官不动。'
                : '例如：降低脸颊正面弧度，保持鼻子、嘴线和下颚轮廓不动。'}
              rows={4}
            />
            <div className="draft-summary">
              <span>当前路径</span>
              <strong>{draftStats.surfaceHits} 个表面点</strong>
            </div>
            <button className="finish-region-button" onClick={finishAnnotation} disabled={draftStats.surfaceHits < 3}>
              <Check size={17} />完成这个区域
            </button>
          </section>

          <details className="global-limits">
            <summary><span>全局限制</span><small>已设置</small></summary>
            <textarea id="global-note" aria-label="全局限制" value={globalNotes} onChange={(event) => setGlobalNotes(event.target.value)} rows={3} />
            <div className="authority-note"><Shield size={15} />标注只授权可见区域；不会自动推断后脑深度。</div>
          </details>

          <section className="send-section">
            <button className="save-button" onClick={savePackage} disabled={packageErrors.length > 0}>
              <Download size={17} />保存标注包
            </button>
            <button className="send-button" onClick={submitForExecution} disabled={packageErrors.length > 0 || Boolean(job && ['submitted', 'queued', 'running'].includes(job.state))}>
              <Send size={17} />提交圈选区域
            </button>
            <p className="action-message" role="status">{message}</p>
          </section>
        </div>

        <section className={`cli-job-panel ${job ? `state-${job.state}` : ''}`}>
          <div className="cli-job-heading">
            <span>处理进度</span>
            {job && <strong>{statusLabel(job.state)}</strong>}
          </div>
          {!job ? (
            <div className="cli-empty"><Sparkles size={16} />提交后等待当前任务接手执行</div>
          ) : (
            <>
              <div className="job-summary">
                <span className="job-state-dot" />
                <strong>{job.summary}</strong>
                {['submitted', 'queued', 'running'].includes(job.state) && (
                  <button onClick={() => cancelCliJob(job.id).then(setJob)}><OctagonX size={14} />取消</button>
                )}
              </div>
              <div className="job-progress" aria-live="polite">
                <div className="job-progress-label">
                  <span>{job.stageLabel || statusLabel(job.state)}</span>
                  <strong>{Math.round(job.progress ?? 0)}%</strong>
                </div>
                <div
                  className="job-progress-track"
                  role="progressbar"
                  aria-label="标注任务进度"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(job.progress ?? 0)}
                >
                  <span style={{ width: `${Math.max(0, Math.min(100, job.progress ?? 0))}%` }} />
                </div>
                <p className="job-progress-detail">{job.progressDetail || '等待阶段更新'}</p>
                <div className="job-timing">
                  <span><Clock3 size={13} />剩余 {formatDuration(job.estimatedSecondsRemaining)}</span>
                  <span>已用 {formatDuration(job.elapsedSeconds)}</span>
                </div>
                {job.error && <div className="cli-error">{job.error}</div>}
                {job.finalMessage && <div className="cli-final">{job.finalMessage}</div>}
              </div>
              {job.logs.length > 0 && (
                <details className="job-technical-details">
                  <summary>技术日志</summary>
                  <div className="cli-log">
                    {job.logs.slice(-12).map((line, index) => <div key={`${index}-${line}`}>{line}</div>)}
                  </div>
                </details>
              )}
            </>
          )}
        </section>
      </aside>
    </main>
  )
}
