import { describe, expect, it } from 'vitest'
import { createAnnotationPackage, validateAnnotationPackage } from './package'
import type { ModelMetadata, ProxyMetadata, SurfaceAnnotation } from './types'

const model: ModelMetadata = {
  schemaVersion: 1,
  displayName: 'OriginalAnime V24',
  sourceVersion: 'v24',
  route: 'profile-loft-preview',
  outputState: 'automated-gates-passed',
  subjectProfile: 'anime-character',
  realPerson: false,
  modelUrl: '/model.glb',
  modelPath: '/project/model.glb',
  modelSha256: 'a'.repeat(64),
  pickProxyUrl: '/proxy.glb',
  pickProxyReportUrl: '/proxy.json',
  coordinateSystem: { frontAxis: '-Z', upAxis: '+Y', bilateralAxis: 'X' },
  invariants: [],
}

const proxy: ProxyMetadata = {
  schemaVersion: 1,
  role: 'qa-only-ray-picking-proxy',
  surfaceGenerated: false,
  coordinateSystem: 'source',
  source: { npz: '/source.npz', npzSha256: 'b'.repeat(64), model: '/model.glb', modelSha256: 'a'.repeat(64), outerVertices: 3, outerTriangles: 1 },
  proxy: { path: '/proxy.glb', sha256: 'c'.repeat(64), vertices: 3, triangles: 1, finite: true, maximumBoundsDelta: 0 },
  warning: 'qa only',
}

const annotation: SurfaceAnnotation = {
  id: 'one', label: '脸颊', intent: 'smooth', color: '#f36c5c', radius: 32, strength: 0.4,
  mirror: true, note: '打磨脸颊，保护鼻子。', visible: true, closed: true, createdAt: new Date().toISOString(),
  screenPath: [{ x: 1, y: 1 }, { x: 2, y: 2 }, { x: 3, y: 1 }],
  surfacePath: [
    { screenIndex: 0, position: [0, 0, 0], normal: [0, 0, -1], proxyTriangleIndex: 1, primitiveName: 'proxy' },
    { screenIndex: 1, position: [1, 0, 0], normal: [0, 0, -1], proxyTriangleIndex: 2, primitiveName: 'proxy' },
    { screenIndex: 2, position: [0, 1, 0], normal: [0, 0, -1], proxyTriangleIndex: 3, primitiveName: 'proxy' },
  ],
  camera: {
    fov: 30, near: 0.01, far: 100, aspect: 1, positionWorld: [0, 0, -5], targetWorld: [0, 0, 0],
    positionModel: [0, 0, -5], targetModel: [0, 0, 0], quaternion: [0, 0, 0, 1],
    projectionMatrix: Array(16).fill(0), viewMatrix: Array(16).fill(0),
    viewport: { width: 100, height: 100, devicePixelRatio: 1 }, normalization: { scale: 1, center: [0, 0, 0] },
  },
}

describe('annotation package', () => {
  it('records manual authority boundaries and fixed six-view QA', () => {
    const value = createAnnotationPackage(model, proxy, [annotation], '保护感官')
    expect(validateAnnotationPackage(value)).toEqual([])
    expect(value.authority.z).toContain('not-hidden-depth-authority')
    expect(value.requestedExecution.fixedQaViews).toHaveLength(6)
    expect(value.pickingProxy.surfaceGenerated).toBe(false)
  })

  it('rejects incomplete regions', () => {
    const value = createAnnotationPackage(model, proxy, [{ ...annotation, note: '', surfacePath: [] }], '')
    expect(validateAnnotationPackage(value)).toEqual(expect.arrayContaining([
      expect.stringContaining('表面命中'),
      expect.stringContaining('缺少给 Codex 的说明'),
    ]))
  })

  it('preserves the dedicated polish intent in the immutable package', () => {
    const value = createAnnotationPackage(model, proxy, [{ ...annotation, intent: 'polish' }], '保护感官与轮廓')
    expect(validateAnnotationPackage(value)).toEqual([])
    expect(value.annotations[0]?.intent).toBe('polish')
    expect(value.requestedExecution.preserveTopology).toBe(true)
  })
})
