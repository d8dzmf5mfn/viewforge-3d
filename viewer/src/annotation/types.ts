export type AnnotationIntent = 'polish' | 'smooth' | 'lower' | 'raise' | 'protect'
export type EditorMode = 'orbit' | 'annotate'
export type ViewPreset = 'front' | 'left45' | 'right45' | 'left90' | 'right90' | 'rear'

export interface ModelMetadata {
  schemaVersion: 1
  displayName: string
  sourceVersion: string
  route: string
  outputState: string
  subjectProfile: string
  realPerson: boolean
  modelUrl: string
  modelPath: string
  modelSha256: string
  pickProxyUrl: string
  pickProxyReportUrl: string
  coordinateSystem: {
    frontAxis: '-Z'
    upAxis: '+Y'
    bilateralAxis: 'X'
  }
  invariants: string[]
}

export interface ProxyMetadata {
  schemaVersion: 1
  role: 'qa-only-ray-picking-proxy'
  surfaceGenerated: false
  coordinateSystem: string
  source: {
    npz: string
    npzSha256: string
    model: string
    modelSha256: string
    outerVertices: number
    outerTriangles: number
  }
  proxy: {
    path: string
    sha256: string
    vertices: number
    triangles: number
    finite: boolean
    maximumBoundsDelta: number
  }
  warning: string
}

export interface ScreenPoint {
  x: number
  y: number
}

export interface SurfaceSample {
  screenIndex: number
  position: [number, number, number]
  normal: [number, number, number]
  proxyTriangleIndex: number | null
  primitiveName: string
}

export interface CameraSnapshot {
  fov: number
  near: number
  far: number
  aspect: number
  positionWorld: [number, number, number]
  targetWorld: [number, number, number]
  positionModel: [number, number, number]
  targetModel: [number, number, number]
  quaternion: [number, number, number, number]
  projectionMatrix: number[]
  viewMatrix: number[]
  viewport: { width: number; height: number; devicePixelRatio: number }
  normalization: {
    scale: number
    center: [number, number, number]
  }
}

export interface SurfaceAnnotation {
  id: string
  label: string
  intent: AnnotationIntent
  color: string
  radius: number
  strength: number
  mirror: boolean
  note: string
  visible: boolean
  closed: true
  screenPath: ScreenPoint[]
  surfacePath: SurfaceSample[]
  camera: CameraSnapshot
  createdAt: string
}

export interface AnnotationPackage {
  schemaVersion: 1
  createdAt: string
  route: string
  outputState: 'annotation-input'
  authority: {
    xy: 'user-annotated-region-and-intent'
    z: 'surface-samples-only-not-hidden-depth-authority'
    posterior: 'not-authorized-without-multiview-evidence'
  }
  model: ModelMetadata
  pickingProxy: ProxyMetadata
  annotations: SurfaceAnnotation[]
  globalNotes: string
  requestedExecution: {
    immutableOutput: true
    preserveTopology: true
    preserveUvs: true
    preserveMaterials: true
    fixedQaViews: ViewPreset[]
  }
}

export type JobState = 'submitted' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
export type JobStage = 'queued' | 'intake' | 'inspect' | 'plan' | 'edit' | 'validate' | 'render' | 'package' | 'completed' | 'failed' | 'cancelled'

export interface CliHealth {
  ok: boolean
  service: string
  host: '127.0.0.1'
  codexAvailable: boolean
  codexVersion: string | null
  activeJobId: string | null
  queueLength: number
  executionMode: 'manual' | 'auto'
  sourceConfigured: boolean
}

export interface CliJob {
  id: string
  state: JobState
  createdAt: string
  startedAt?: string
  finishedAt?: string
  annotationCount: number
  summary: string
  logs: string[]
  finalMessage?: string
  outputDirectory?: string
  error?: string
  stage: JobStage
  stageLabel: string
  progress: number
  progressDetail: string
  elapsedSeconds: number
  estimatedSecondsRemaining: number | null
  lastActivityAt?: string
}

export interface DraftStats {
  screenPoints: number
  surfaceHits: number
  drawing: boolean
}
