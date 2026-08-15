export type ViewRole = 'front' | 'left45' | 'right45'
export type DisplayMode = 'comparison' | 'voxel' | 'smooth' | 'skin' | 'eye-contact' | 'ear-continuity' | 'skin-projection'
export type CameraPreset = 'front' | 'side' | 'three-quarter'

export interface CameraRecord {
  role: ViewRole
  width: number
  height: number
  focal_length_px: number
  principal_point_px: [number, number]
  rotation_vector: [number, number, number]
  translation: [number, number, number]
}

export interface FaceManifest {
  schemaVersion: '1.0.0' | '2.0.0' | '3.0.0'
  subjectProfile: 'face-v1' | 'face-v2' | 'face-v3'
  mode: 'pixel-direct' | 'pixel-flame-hybrid' | 'template-head-v0'
  provenance: {
    inputs: Record<ViewRole, { sourceSha256: string; normalizedSha256: string; file: string }>
    models: Record<string, string>
    configSha256: string
    codeSha256: string
    viewforge3dVersion: string
  }
  cameras: CameraRecord[]
  pixel?: {
    mapping: string
    gridSize: number | [number, number] | [number, number, number]
    instanceCount: number
    complexPixelCount: number
    simpleInterpolatedPixelCount: number
    binary: string
    binarySha256: string
    schema: string
    schemaSha256: string
  }
  voxel?: {
    representation: string
    resolution: [number, number] | [number, number, number]
    voxelSize: number
    instanceCount: number
    surfaceCellCoverage: number
    maximumInstances: number
    maximumSurfaceDistanceVoxels?: number
    model: string
  }
  mesh: {
    vertices: number
    triangles: number
    watertight: boolean
    edgeManifold: boolean
    boundaryEdges: number
    degenerateTriangles: number
    normalVarianceReduction: number
    completeness?: number
    smoothness?: number
    hausdorffVoxels: number
    silhouette?: Record<ViewRole, { rawIoU: number; smoothIoU: number; drop: number }>
    model: string
    geometryHash?: string
    nodes?: string[]
    componentCount?: number
    boundaryEdgeCount?: number
    nonManifoldEdgeCount?: number
    degenerateFaceCount?: number
    selfIntersectionPairCount?: number
    topCurvatureSpikeRatio?: number
    geometryHashesMatch?: boolean
    surfaceGeneratedBySdf?: boolean
    passed?: boolean
  }
  skin: {
    representation: string
    model: string
    atlas?: string
    confidenceMap: string
    sourceMap?: string
    atlasResolution: [number, number]
    observedVertexFraction: number
    atlasObservedFraction: number
    frontAlignmentMethod?: string
    frontAlignmentInliers?: number
    projectionMethod?: string
    seamDeltaE00Median?: number
    seamDeltaE00P95?: number
    geometryHash?: string
    neutralGeometryHash?: string
    skinGeometryHash?: string
    maximumVertexDifference?: number
    modelSha256?: string
    inferredRegions: string[]
  }
  fit: {
    perView: Record<ViewRole, {
      landmarkNME: number
      landmarkErrorPx?: number
      reprojectionErrorPx?: number
      silhouetteIoU: number
      silhouetteErrorMm?: number
      passed: boolean
    }>
  }
  projection?: {
    mapping: string
    recordCount: number
    binary: string
    binarySha256: string
    schema: string
    schemaSha256: string
    sourceViewOrder: ViewRole[]
  }
  sdf?: {
    role: string
    surfaceGenerated: boolean
    finite: boolean
    outsideSignConsistency: number
    insideSignConsistency: number
    passed: boolean
  }
  anatomy?: {
    unifiedHead?: {
      connectedComponents: number
      boundaryEdges: number
      nonManifoldEdges: number
      topCurvatureSpikeRatio: number
      geometryHash: string
    }
    ears: {
      source?: string
      carrierPresent: boolean
      rootSharedWithScalp: boolean
    }
    eyes: {
      completeEyeballNodes: number
      contactGapP99R?: number
      radiusDifferenceRatio?: number
      irisReprojectionErrorPx?: number
      penetrationCount?: number
      intersectionCount?: number
      left?: { contactGapP99R: number | null; passed: boolean }
      right?: { contactGapP99R: number | null; passed: boolean }
    }
    geometry?: {
      componentCount: number
      boundaryEdgeCount: number
      nonManifoldEdgeCount: number
      selfIntersectionPairCount: number
      topCurvatureSpikeRatio: number
      geometryHash: string
      geometryHashesMatch: boolean
      passed: boolean
    }
    passed?: boolean
  }
  confidence: {
    mean: number
    lowConfidenceRegions: string[]
    templateInferredRegions: string[]
  }
  runtime: Record<string, unknown>
}

export interface GateCheck {
  metric?: string
  role?: string
  measured?: unknown
  threshold?: unknown
  status: 'pass' | 'fail' | 'recorded' | 'notEvaluated' | 'pendingUserSignoff'
  [key: string]: unknown
}

export interface QAReport {
  schemaVersion: '1.0.0' | '2.0.0' | '3.0.0'
  gates: Array<{
    gate: string
    status: 'pass' | 'fail' | 'partial' | 'pendingUserSignoff'
    checks: GateCheck[]
  }>
  summary: {
    automatedGatesPassed: boolean
    userSignoffRequired: boolean
    browserAuditRequired: boolean
    visualReviewStatus?: 'pendingUserSignoff' | 'accepted' | 'rejected'
    finalAcceptance?: boolean
    visualBaselineReviewed?: boolean
  }
}

export interface FacePackage {
  name: string
  manifest: FaceManifest
  report: QAReport
  references: Record<ViewRole, string>
  voxelGlb?: Uint8Array
  smoothGlb?: Uint8Array
  skinGlb?: Uint8Array
  headGlb?: Uint8Array
  skinAtlasUrl?: string
  sourceMapUrl?: string
  confidenceMapUrl?: string
  loadDurationMs: number
  dispose: () => void
}
