import type { FacePackage } from '../lib/types'

interface Metric {
  label: string
  value: string
  status: 'pass' | 'warn' | 'fail'
  progress?: number
}

function emptyMetrics(): Metric[] {
  return [
    { label: '统一头模', value: '—', status: 'warn' },
    { label: '像素重投影', value: '—', status: 'warn' },
    { label: '3D Pixel', value: '—', status: 'warn' },
    { label: '眼球接触', value: '—', status: 'warn' },
    { label: '耳根连续', value: '—', status: 'warn' },
    { label: '皮肤匹配', value: '—', status: 'warn' },
    { label: '区域置信度', value: '—', status: 'warn', progress: 0 },
    { label: '视觉签收', value: '待加载', status: 'warn' },
  ]
}

function v2Metrics(facePackage: FacePackage): Metric[] {
  const { manifest, report } = facePackage
  const anatomy = manifest.anatomy!
  const views = Object.values(manifest.fit.perView)
  const reprojection = views.reduce((sum, view) => sum + (view.landmarkErrorPx ?? 0), 0) / views.length
  const surfaceDistance = manifest.voxel!.maximumSurfaceDistanceVoxels ?? Number.POSITIVE_INFINITY
  const geometryIdentity = manifest.mesh.geometryHash === manifest.skin.geometryHash
    && manifest.skin.maximumVertexDifference === 0
  const confidence = manifest.confidence.mean
  return [
    { label: '统一头模', value: anatomy.unifiedHead!.topCurvatureSpikeRatio.toFixed(2), status: anatomy.unifiedHead!.connectedComponents === 1 && anatomy.unifiedHead!.topCurvatureSpikeRatio <= 4 ? 'pass' : 'fail' },
    { label: '像素重投影', value: `${reprojection.toFixed(2)} px`, status: views.every((view) => view.passed) ? 'pass' : 'fail' },
    { label: '3D Pixel 贴面', value: `${surfaceDistance.toFixed(2)} voxel`, status: surfaceDistance <= 0.75 ? 'pass' : 'fail' },
    { label: '眼球接触', value: `${((anatomy.eyes.contactGapP99R ?? 0) * 100).toFixed(2)} % R`, status: anatomy.eyes.completeEyeballNodes === 2 && anatomy.eyes.penetrationCount === 0 && (anatomy.eyes.contactGapP99R ?? 1) <= 0.03 ? 'pass' : 'fail' },
    { label: '耳根连续', value: anatomy.ears.rootSharedWithScalp && !anatomy.ears.carrierPresent ? '同一拓扑' : '失败', status: anatomy.ears.rootSharedWithScalp && !anatomy.ears.carrierPresent ? 'pass' : 'fail' },
    { label: '皮肤匹配', value: geometryIdentity ? '0 顶点差' : '几何漂移', status: geometryIdentity ? 'pass' : 'fail' },
    { label: '纹理接缝', value: `ΔE ${manifest.skin.seamDeltaE00Median?.toFixed(2) ?? '—'}`, status: (manifest.skin.seamDeltaE00Median ?? 99) <= 3 ? 'pass' : 'fail' },
    { label: '区域置信度', value: confidence.toFixed(2), status: confidence >= 0.75 ? 'pass' : 'warn', progress: confidence },
    { label: '视觉签收', value: report.summary.finalAcceptance ? '已通过' : '待确认', status: report.summary.finalAcceptance ? 'pass' : 'warn' },
  ]
}

function v3Metrics(facePackage: FacePackage): Metric[] {
  const { manifest, report } = facePackage
  const views = Object.values(manifest.fit.perView)
  const anatomy = manifest.anatomy!
  const geometry = anatomy.geometry!
  const eyeGaps = [anatomy.eyes.left?.contactGapP99R, anatomy.eyes.right?.contactGapP99R]
    .filter((value): value is number => value !== null && value !== undefined)
  const maximumEyeGap = eyeGaps.length ? Math.max(...eyeGaps) : Number.POSITIVE_INFINITY
  const geometryIdentity = manifest.mesh.geometryHash === manifest.skin.geometryHash
    && manifest.skin.maximumVertexDifference === 0
  const confidence = manifest.confidence.mean
  return [
    { label: '连续头颈模板', value: `${manifest.mesh.triangles.toLocaleString()} 面`, status: geometry.passed && geometry.componentCount === 1 ? 'pass' : 'fail' },
    { label: '三视图拟合', value: views.every((view) => view.passed) ? '全部通过' : '未通过', status: views.every((view) => view.passed) ? 'pass' : 'fail' },
    { label: '头顶曲率', value: geometry.topCurvatureSpikeRatio.toFixed(2), status: geometry.topCurvatureSpikeRatio <= 4 ? 'pass' : 'fail' },
    { label: 'SDF 用途', value: manifest.sdf?.role ?? '缺失', status: manifest.sdf?.role === 'qa-only' && !manifest.sdf.surfaceGenerated && manifest.sdf.passed ? 'pass' : 'fail' },
    { label: '眼球接触', value: Number.isFinite(maximumEyeGap) ? `${(maximumEyeGap * 100).toFixed(2)} % R` : '缺失', status: anatomy.eyes.completeEyeballNodes === 2 && anatomy.eyes.intersectionCount === 0 && maximumEyeGap <= 0.03 ? 'pass' : 'fail' },
    { label: '耳根连续', value: anatomy.ears.rootSharedWithScalp && !anatomy.ears.carrierPresent ? '同一拓扑' : '失败', status: anatomy.ears.rootSharedWithScalp && !anatomy.ears.carrierPresent ? 'pass' : 'fail' },
    { label: '人皮/基座匹配', value: geometryIdentity ? '0 顶点差' : '几何漂移', status: geometryIdentity ? 'pass' : 'fail' },
    { label: '纹理接缝', value: `ΔE ${manifest.skin.seamDeltaE00Median?.toFixed(2) ?? '—'}`, status: (manifest.skin.seamDeltaE00Median ?? 99) <= 3 ? 'pass' : 'fail' },
    { label: '区域置信度', value: confidence.toFixed(2), status: confidence >= 0.75 ? 'pass' : 'warn', progress: confidence },
    { label: '质量基准签收', value: report.summary.finalAcceptance ? '已通过' : '待确认', status: report.summary.finalAcceptance ? 'pass' : 'warn' },
  ]
}

function v1Metrics(facePackage: FacePackage): Metric[] {
  const manifest = facePackage.manifest
  const views = Object.values(manifest.fit.perView)
  const landmark = views.reduce((sum, view) => sum + view.landmarkNME, 0) / views.length
  const smoothViews = manifest.mesh.silhouette
  const smoothValues = smoothViews ? Object.values(smoothViews) : []
  const smoothSilhouette = smoothValues.length
    ? smoothValues.reduce((sum, view) => sum + view.smoothIoU, 0) / smoothValues.length
    : null
  const pixelCoverage = manifest.voxel!.surfaceCellCoverage
  const confidence = manifest.confidence.mean
  return [
    { label: '平滑网格轮廓', value: smoothSilhouette === null ? '未测量' : `${(smoothSilhouette * 100).toFixed(1)} %`, status: smoothSilhouette !== null && smoothSilhouette >= 0.90 ? 'pass' : 'fail' },
    { label: '像素重投影', value: `${(landmark * 100).toFixed(2)} %`, status: landmark <= 0.025 ? 'pass' : 'fail' },
    { label: 'Pixel 曲面覆盖', value: `${(pixelCoverage * 100).toFixed(1)} %`, status: pixelCoverage >= 0.95 ? 'pass' : 'fail' },
    { label: '人皮 UV 可观测区', value: `${(manifest.skin.atlasObservedFraction * 100).toFixed(1)} %`, status: manifest.skin.atlasObservedFraction >= 0.35 ? 'pass' : 'warn' },
    { label: '孔洞', value: String(manifest.mesh.boundaryEdges), status: manifest.mesh.boundaryEdges === 0 ? 'pass' : 'fail' },
    { label: '区域置信度', value: confidence.toFixed(2), status: confidence >= 0.75 ? 'pass' : 'warn', progress: confidence },
    { label: '视觉签收', value: facePackage.report.summary.finalAcceptance ? '已通过' : '待确认', status: facePackage.report.summary.finalAcceptance ? 'pass' : 'warn' },
  ]
}

function metrics(facePackage: FacePackage | null): Metric[] {
  if (!facePackage) return emptyMetrics()
  if (facePackage.manifest.schemaVersion === '3.0.0') return v3Metrics(facePackage)
  return facePackage.manifest.schemaVersion === '2.0.0'
    ? v2Metrics(facePackage)
    : v1Metrics(facePackage)
}

export function QualityPanel({ facePackage }: { facePackage: FacePackage | null }) {
  const rows = metrics(facePackage)
  return (
    <aside className="quality-panel" aria-label="重建质量">
      <h2>重建质量</h2>
      <div className="quality-rule" />
      <div className="quality-list">
        {rows.map((metric) => (
          <div className="quality-row" key={metric.label}>
            <div className="quality-line">
              <span>{metric.label}</span>
              <span className="quality-value">{metric.value}</span>
              <i className={`status-dot ${metric.status}`} aria-label={metric.status} />
            </div>
            {metric.progress !== undefined && <div className="progress-track"><span style={{ width: `${metric.progress * 100}%` }} /></div>}
          </div>
        ))}
      </div>
      {facePackage && (!facePackage.report.summary.automatedGatesPassed || !facePackage.report.summary.finalAcceptance) && <p className="quality-warning">尚未完成视觉签收</p>}
    </aside>
  )
}
