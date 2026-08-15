import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { ChevronDown, FileText, FolderOpen, ListChecks } from 'lucide-react'
import { QualityPanel } from './components/QualityPanel'
import { ReferenceRail } from './components/ReferenceRail'
import { ViewToolbar } from './components/ViewToolbar'
import { loadFacePackage } from './lib/archive'
import { exportAcceptance } from './lib/exportReport'
import type { CameraPreset, DisplayMode, FacePackage } from './lib/types'
import type { ViewportHandle } from './three/DualViewport'

const DualViewport = lazy(() => import('./three/DualViewport'))

export default function App() {
  const [facePackage, setFacePackage] = useState<FacePackage | null>(null)
  const [mode, setMode] = useState<DisplayMode>('comparison')
  const [preset, setPreset] = useState<CameraPreset>('three-quarter')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const viewportRef = useRef<ViewportHandle>(null)
  const demoRequestedRef = useRef(false)
  const loadStartedRef = useRef(0)

  useEffect(() => {
    let disposed = false
    const dispose = () => {
      if (disposed) return
      disposed = true
      facePackage?.dispose()
    }
    window.addEventListener('pagehide', dispose)
    return () => {
      window.removeEventListener('pagehide', dispose)
      dispose()
    }
  }, [facePackage])

  useEffect(() => {
    if (facePackage?.manifest.schemaVersion === '3.0.0' && mode === 'voxel') {
      setMode('comparison')
    }
  }, [facePackage, mode])

  useEffect(() => {
    if (demoRequestedRef.current || new URLSearchParams(window.location.search).get('demo') !== '1') return
    demoRequestedRef.current = true
    void fetch('/demo.face3d', { cache: 'no-store' })
      .then((response) => {
        if (!response.ok) throw new Error(`演示结果包不可用 (${response.status})`)
        return response.blob()
      })
      .then((blob) => load(new File([blob], 'face-001.face3d', { type: 'application/zip' })))
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : '无法载入演示结果包')
      })
  }, [])

  async function load(file?: File) {
    if (!file) return
    loadStartedRef.current = performance.now()
    delete document.documentElement.dataset.modelLoadMs
    setLoading(true)
    setError(null)
    try {
      const loaded = await loadFacePackage(file)
      document.documentElement.dataset.packageLoadMs = loaded.loadDurationMs.toFixed(1)
      setFacePackage(loaded)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '无法读取结果包')
    } finally {
      setLoading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  const recordModelsReady = useCallback(() => {
    document.documentElement.dataset.modelLoadMs = (
      performance.now() - loadStartedRef.current
    ).toFixed(1)
  }, [])

  function exportReport() {
    if (!facePackage) return
    exportAcceptance(facePackage, viewportRef.current?.capture() ?? '')
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">3D Face Lab</div>
        <span className="header-rule" />
        <div className="subject-name">{facePackage?.name ?? '未加载'}<ChevronDown size={16} /></div>
        <div className="header-actions">
          <input ref={inputRef} type="file" accept=".face3d,.zip" hidden onChange={(event) => void load(event.target.files?.[0])} />
          <button className="button secondary" onClick={() => inputRef.current?.click()} disabled={loading}><FolderOpen />{loading ? '解析中…' : '加载结果包'}</button>
          <button className="button primary" onClick={exportReport} disabled={!facePackage}><FileText />导出验收报告</button>
        </div>
      </header>
      {error && <div className="error-banner" role="alert">{error}</div>}
      <main className="workspace">
        <ReferenceRail facePackage={facePackage} />
        <section className="model-workspace">
          <ViewToolbar mode={mode} preset={preset} showVoxel={facePackage?.manifest.schemaVersion !== '3.0.0'} onMode={setMode} onPreset={setPreset} />
          <Suspense fallback={<div className="viewport-loading">初始化本地 3D 查看器…</div>}>
            <DualViewport ref={viewportRef} mode={mode} preset={preset} voxelGlb={facePackage?.voxelGlb} smoothGlb={facePackage?.smoothGlb} skinGlb={facePackage?.skinGlb} headGlb={facePackage?.headGlb} sourceMapUrl={facePackage?.sourceMapUrl} onModelsReady={recordModelsReady} />
          </Suspense>
        </section>
        <QualityPanel facePackage={facePackage} />
      </main>
      <footer className="privacy-bar"><span><i />本地处理&nbsp;&nbsp;·&nbsp;&nbsp;人脸数据未上传</span><ListChecks size={19} /></footer>
    </div>
  )
}
