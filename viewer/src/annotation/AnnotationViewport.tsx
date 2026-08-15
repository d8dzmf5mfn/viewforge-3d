import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import type {
  AnnotationIntent,
  CameraSnapshot,
  DraftStats,
  EditorMode,
  ScreenPoint,
  SurfaceAnnotation,
  SurfaceSample,
  ViewPreset,
} from './types'

export interface AnnotationViewportHandle {
  finishDraft: (options: {
    label: string
    intent: AnnotationIntent
    color: string
    radius: number
    strength: number
    mirror: boolean
    note: string
  }) => SurfaceAnnotation | null
  undoDraft: () => void
  clearDraft: () => void
  capture: () => string
}

interface Props {
  modelUrl: string
  pickProxyUrl: string
  mode: EditorMode
  preset: ViewPreset
  annotations: SurfaceAnnotation[]
  activeColor: string
  radius: number
  showHits: boolean
  onReady: () => void
  onDraftStats: (stats: DraftStats) => void
  onPointerWorld: (position: [number, number, number] | null) => void
}

interface Draft {
  drawing: boolean
  screenPath: ScreenPoint[]
  surfacePath: SurfaceSample[]
  camera: CameraSnapshot | null
}

const emptyDraft = (): Draft => ({ drawing: false, screenPath: [], surfacePath: [], camera: null })

function disposeObject(root: THREE.Object3D | null): void {
  const geometries = new Set<THREE.BufferGeometry>()
  const materials = new Set<THREE.Material>()
  const textures = new Set<THREE.Texture>()
  root?.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return
    geometries.add(object.geometry)
    const entries = Array.isArray(object.material) ? object.material : [object.material]
    entries.forEach((material) => {
      materials.add(material)
      Object.values(material).forEach((value) => {
        if (value instanceof THREE.Texture) textures.add(value)
      })
    })
  })
  textures.forEach((texture) => texture.dispose())
  materials.forEach((material) => material.dispose())
  geometries.forEach((geometry) => geometry.dispose())
}

async function loadGlb(url: string): Promise<THREE.Object3D> {
  const gltf = await new GLTFLoader().loadAsync(url)
  return gltf.scene
}

function modelBounds(root: THREE.Object3D): { center: THREE.Vector3; scale: number } {
  const box = new THREE.Box3().setFromObject(root)
  const center = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3())
  return { center, scale: 2.62 / Math.max(size.x, size.y, size.z, 1e-6) }
}

function applyNormalization(root: THREE.Object3D, center: THREE.Vector3, scale: number): void {
  root.position.copy(center).multiplyScalar(-scale)
  root.scale.setScalar(scale)
  root.updateMatrixWorld(true)
}

function addStudio(scene: THREE.Scene): void {
  scene.background = new THREE.Color(0x111416)
  scene.add(new THREE.HemisphereLight(0xf8eee9, 0x262d31, 1.7))
  const key = new THREE.DirectionalLight(0xfff2e9, 2.5)
  key.position.set(-4, 5, -5)
  scene.add(key)
  const fill = new THREE.DirectionalLight(0xc7d8e7, 0.75)
  fill.position.set(4, 1, -4)
  scene.add(fill)
  const rim = new THREE.DirectionalLight(0xffa78f, 0.5)
  rim.position.set(-4, 1, 4)
  scene.add(rim)
}

function vectorTuple(vector: THREE.Vector3): [number, number, number] {
  return [vector.x, vector.y, vector.z]
}

function presetPosition(preset: ViewPreset, distance: number): THREE.Vector3 {
  const degrees: Record<ViewPreset, number> = {
    front: 0,
    left45: -45,
    right45: 45,
    left90: -90,
    right90: 90,
    rear: 180,
  }
  const radians = THREE.MathUtils.degToRad(degrees[preset])
  return new THREE.Vector3(Math.sin(radians) * distance, 0.03, -Math.cos(radians) * distance)
}

function buildAnnotationObject(annotation: SurfaceAnnotation): THREE.Group {
  const group = new THREE.Group()
  group.name = annotation.id
  group.visible = annotation.visible
  if (annotation.surfacePath.length < 2) return group
  const points = annotation.surfacePath.map((sample) => (
    new THREE.Vector3(...sample.position).addScaledVector(new THREE.Vector3(...sample.normal), 0.014)
  ))
  const firstPoint = points[0]
  if (!firstPoint) return group
  points.push(firstPoint.clone())
  const curve = new THREE.CatmullRomCurve3(points, true, 'centripetal', 0.35)
  const geometry = new THREE.TubeGeometry(curve, Math.max(24, points.length * 2), 0.012, 6, true)
  const material = new THREE.MeshBasicMaterial({
    color: annotation.color,
    depthTest: true,
    depthWrite: false,
    transparent: true,
    opacity: 0.98,
  })
  const line = new THREE.Mesh(geometry, material)
  line.renderOrder = 20
  group.add(line)

  const screenContour = annotation.surfacePath.map((sample) => {
    const screen = annotation.screenPath[sample.screenIndex] ?? annotation.screenPath[0]
    return new THREE.Vector2(screen?.x ?? 0, screen?.y ?? 0)
  })
  const triangles = THREE.ShapeUtils.triangulateShape(screenContour, [])
  if (triangles.length > 0) {
    const fillGeometry = new THREE.BufferGeometry().setFromPoints(points.slice(0, -1))
    fillGeometry.setIndex(triangles.flat())
    fillGeometry.computeVertexNormals()
    const fill = new THREE.Mesh(fillGeometry, new THREE.MeshBasicMaterial({
      color: annotation.color,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      opacity: 0.19,
    }))
    fill.name = 'surface-fill'
    fill.renderOrder = 18
    group.add(fill)
  }

  const pointGeometry = new THREE.BufferGeometry().setFromPoints(
    annotation.surfacePath.filter((_, index) => index % 3 === 0).map((sample) => (
      new THREE.Vector3(...sample.position).addScaledVector(new THREE.Vector3(...sample.normal), 0.016)
    )),
  )
  const pointMaterial = new THREE.PointsMaterial({
    color: 0xffe3dc,
    size: 0.025,
    sizeAttenuation: true,
      depthTest: true,
    transparent: true,
    opacity: 0.9,
  })
  const hits = new THREE.Points(pointGeometry, pointMaterial)
  hits.name = 'surface-hits'
  hits.renderOrder = 21
  group.add(hits)
  const storedDirection = new THREE.Vector3(...annotation.camera.targetWorld)
    .sub(new THREE.Vector3(...annotation.camera.positionWorld))
    .normalize()
  group.userData.annotationViewDirection = storedDirection
  return group
}

export const AnnotationViewport = forwardRef<AnnotationViewportHandle, Props>(
  function AnnotationViewport(
    {
      modelUrl,
      pickProxyUrl,
      mode,
      preset,
      annotations,
      activeColor,
      radius,
      showHits,
      onReady,
      onDraftStats,
      onPointerWorld,
    },
    ref,
  ) {
    const hostRef = useRef<HTMLDivElement>(null)
    const overlayRef = useRef<SVGSVGElement>(null)
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
    const sceneRef = useRef<THREE.Scene | null>(null)
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
    const controlsRef = useRef<OrbitControls | null>(null)
    const modelRef = useRef<THREE.Object3D | null>(null)
    const proxyRef = useRef<THREE.Object3D | null>(null)
    const annotationRootRef = useRef<THREE.Group | null>(null)
    const normalizationRef = useRef({ center: new THREE.Vector3(), scale: 1 })
    const draftRef = useRef<Draft>(emptyDraft())
    const modeRef = useRef(mode)
    const showHitsRef = useRef(showHits)
    modeRef.current = mode
    showHitsRef.current = showHits

    const syncOverlay = () => {
      const overlay = overlayRef.current
      if (!overlay) return
      const draft = draftRef.current
      const polyline = overlay.querySelector('[data-draft-path]')
      polyline?.setAttribute('points', draft.screenPath.map(({ x, y }) => `${x},${y}`).join(' '))
      overlay.dataset.drawing = String(draft.drawing)
      onDraftStats({
        drawing: draft.drawing,
        screenPoints: draft.screenPath.length,
        surfaceHits: draft.surfacePath.length,
      })
    }

    const clearDraft = () => {
      draftRef.current = emptyDraft()
      controlsRef.current && (controlsRef.current.enabled = modeRef.current === 'orbit')
      syncOverlay()
    }

    useImperativeHandle(ref, () => ({
      finishDraft: (options) => {
        const draft = draftRef.current
        if (!draft.camera || draft.screenPath.length < 3 || draft.surfacePath.length < 3) return null
        const annotation: SurfaceAnnotation = {
          id: crypto.randomUUID(),
          label: options.label,
          intent: options.intent,
          color: options.color,
          radius: options.radius,
          strength: options.strength,
          mirror: options.mirror,
          note: options.note,
          visible: true,
          closed: true,
          screenPath: [...draft.screenPath],
          surfacePath: [...draft.surfacePath],
          camera: draft.camera,
          createdAt: new Date().toISOString(),
        }
        clearDraft()
        return annotation
      },
      undoDraft: () => {
        const draft = draftRef.current
        draft.screenPath.splice(Math.max(0, draft.screenPath.length - 8))
        const remaining = new Set(draft.screenPath.map((_, index) => index))
        draft.surfacePath = draft.surfacePath.filter((sample) => remaining.has(sample.screenIndex))
        syncOverlay()
      },
      clearDraft,
      capture: () => canvasRef.current?.toDataURL('image/png') ?? '',
    }), [])

    useEffect(() => {
      const host = hostRef.current
      if (!host) return
      const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      renderer.outputColorSpace = THREE.SRGBColorSpace
      renderer.toneMapping = THREE.ACESFilmicToneMapping
      renderer.toneMappingExposure = 1.0
      rendererRef.current = renderer
      canvasRef.current = renderer.domElement
      host.prepend(renderer.domElement)

      const scene = new THREE.Scene()
      addStudio(scene)
      sceneRef.current = scene
      const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100)
      camera.position.copy(presetPosition('left45', 5.8))
      cameraRef.current = camera
      const controls = new OrbitControls(camera, renderer.domElement)
      controls.target.set(0, -0.03, 0)
      controls.enableDamping = true
      controls.dampingFactor = 0.08
      controls.minDistance = 3.2
      controls.maxDistance = 9
      controls.enabled = modeRef.current === 'orbit'
      controlsRef.current = controls
      const annotationRoot = new THREE.Group()
      annotationRoot.name = 'annotation-root'
      scene.add(annotationRoot)
      annotationRootRef.current = annotationRoot

      const resize = new ResizeObserver(() => {
        const { width, height } = host.getBoundingClientRect()
        renderer.setSize(width, height, false)
      })
      resize.observe(host)
      let frame = 0
      let disposed = false
      const render = () => {
        if (disposed) return
        frame = requestAnimationFrame(render)
        controls.update()
        const width = renderer.domElement.clientWidth
        const height = renderer.domElement.clientHeight
        if (!width || !height) return
        camera.aspect = width / height
        camera.updateProjectionMatrix()
        const currentDirection = camera.getWorldDirection(new THREE.Vector3())
        annotationRoot.children.forEach((annotationGroup) => {
          const storedDirection = annotationGroup.userData.annotationViewDirection as THREE.Vector3 | undefined
          annotationGroup.traverse((object) => {
            if (object.name === 'surface-hits') object.visible = showHitsRef.current
            if (object.name === 'surface-fill') object.visible = !storedDirection || storedDirection.dot(currentDirection) > 0.94
          })
        })
        renderer.render(scene, camera)
      }
      render()

      let cancelled = false
      Promise.all([loadGlb(modelUrl), loadGlb(pickProxyUrl)]).then(([model, proxy]) => {
        if (cancelled) {
          disposeObject(model)
          disposeObject(proxy)
          return
        }
        model.traverse((object) => {
          if (!(object instanceof THREE.Mesh)) return
          if (!object.geometry.getAttribute('normal')) object.geometry.computeVertexNormals()
          const materials = Array.isArray(object.material) ? object.material : [object.material]
          materials.forEach((material) => {
            if (!(material instanceof THREE.MeshStandardMaterial)) return
            material.roughness = 0.84
            material.metalness = 0
            material.envMapIntensity = 0.25
            material.needsUpdate = true
          })
        })
        proxy.traverse((object) => {
          object.visible = false
          if (object instanceof THREE.Mesh) object.material.side = THREE.DoubleSide
        })
        const { center, scale } = modelBounds(model)
        applyNormalization(model, center, scale)
        applyNormalization(proxy, center, scale)
        normalizationRef.current = { center, scale }
        modelRef.current = model
        proxyRef.current = proxy
        scene.add(model, proxy)
        onReady()
      }).catch((error: unknown) => {
        host.dataset.error = error instanceof Error ? error.message : String(error)
      })

      return () => {
        cancelled = true
        disposed = true
        cancelAnimationFrame(frame)
        resize.disconnect()
        controls.dispose()
        disposeObject(modelRef.current)
        disposeObject(proxyRef.current)
        disposeObject(annotationRoot)
        renderer.renderLists.dispose()
        renderer.dispose()
        renderer.forceContextLoss()
        renderer.domElement.remove()
        modelRef.current = null
        proxyRef.current = null
      }
    }, [modelUrl, pickProxyUrl, onReady])

    useEffect(() => {
      if (controlsRef.current) controlsRef.current.enabled = mode === 'orbit' && !draftRef.current.drawing
      if (mode === 'orbit' && draftRef.current.drawing) clearDraft()
    }, [mode])

    useEffect(() => {
      const camera = cameraRef.current
      const controls = controlsRef.current
      if (!camera || !controls) return
      const distance = camera.position.distanceTo(controls.target)
      camera.position.copy(presetPosition(preset, distance))
      controls.target.set(0, -0.03, 0)
      controls.update()
    }, [preset])

    useEffect(() => {
      const root = annotationRootRef.current
      if (!root) return
      disposeObject(root)
      root.clear()
      annotations.forEach((annotation) => {
        const object = buildAnnotationObject(annotation)
        const { center, scale } = normalizationRef.current
        applyNormalization(object, center, scale)
        root.add(object)
      })
    }, [annotations])

    const raycast = (event: ReactPointerEvent<HTMLDivElement>): SurfaceSample | null => {
      const camera = cameraRef.current
      const proxy = proxyRef.current
      const host = hostRef.current
      if (!camera || !proxy || !host) return null
      const rect = host.getBoundingClientRect()
      const pointer = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      )
      const raycaster = new THREE.Raycaster()
      raycaster.setFromCamera(pointer, camera)
      const hit = raycaster.intersectObject(proxy, true)[0]
      if (!hit) return null
      const { center, scale } = normalizationRef.current
      const modelPoint = hit.point.clone().divideScalar(scale).add(center)
      const worldNormal = hit.face?.normal.clone() ?? new THREE.Vector3(0, 0, 1)
      const normalMatrix = new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld)
      worldNormal.applyMatrix3(normalMatrix).normalize()
      return {
        screenIndex: draftRef.current.screenPath.length - 1,
        position: vectorTuple(modelPoint),
        normal: vectorTuple(worldNormal),
        proxyTriangleIndex: hit.faceIndex ?? null,
        primitiveName: hit.object.name || 'AnnotationPickProxy',
      }
    }

    const cameraSnapshot = (): CameraSnapshot | null => {
      const camera = cameraRef.current
      const controls = controlsRef.current
      const host = hostRef.current
      if (!camera || !controls || !host) return null
      camera.updateMatrixWorld(true)
      const { center, scale } = normalizationRef.current
      const modelPosition = camera.position.clone().divideScalar(scale).add(center)
      const modelTarget = controls.target.clone().divideScalar(scale).add(center)
      const rect = host.getBoundingClientRect()
      return {
        fov: camera.fov,
        near: camera.near,
        far: camera.far,
        aspect: camera.aspect,
        positionWorld: vectorTuple(camera.position),
        targetWorld: vectorTuple(controls.target),
        positionModel: vectorTuple(modelPosition),
        targetModel: vectorTuple(modelTarget),
        quaternion: [camera.quaternion.x, camera.quaternion.y, camera.quaternion.z, camera.quaternion.w],
        projectionMatrix: camera.projectionMatrix.toArray(),
        viewMatrix: camera.matrixWorldInverse.toArray(),
        viewport: { width: rect.width, height: rect.height, devicePixelRatio: window.devicePixelRatio },
        normalization: { scale, center: vectorTuple(center) },
      }
    }

    const recordPoint = (event: ReactPointerEvent<HTMLDivElement>) => {
      const host = hostRef.current
      if (!host) return
      const rect = host.getBoundingClientRect()
      const point = { x: event.clientX - rect.left, y: event.clientY - rect.top }
      const previous = draftRef.current.screenPath.at(-1)
      if (previous && Math.hypot(point.x - previous.x, point.y - previous.y) < 3) return
      draftRef.current.screenPath.push(point)
      const hit = raycast(event)
      if (hit) {
        draftRef.current.surfacePath.push(hit)
        onPointerWorld(hit.position)
      } else {
        onPointerWorld(null)
      }
      syncOverlay()
    }

    const pointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
      if (modeRef.current !== 'annotate' || event.button !== 0) return
      event.currentTarget.setPointerCapture(event.pointerId)
      draftRef.current = emptyDraft()
      draftRef.current.drawing = true
      draftRef.current.camera = cameraSnapshot()
      if (controlsRef.current) controlsRef.current.enabled = false
      recordPoint(event)
    }

    const pointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!draftRef.current.drawing) return
      recordPoint(event)
    }

    const pointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!draftRef.current.drawing) return
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
      draftRef.current.drawing = false
      syncOverlay()
    }

    return (
      <div
        className={`annotation-viewport ${mode === 'annotate' ? 'is-annotating' : ''}`}
        ref={hostRef}
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect()
          event.currentTarget.style.setProperty('--cursor-x', `${event.clientX - rect.left}px`)
          event.currentTarget.style.setProperty('--cursor-y', `${event.clientY - rect.top}px`)
        }}
        onPointerUp={pointerUp}
        onPointerCancel={pointerUp}
        style={{ '--brush-size': `${Math.max(18, radius * 1.6)}px`, '--annotation-color': activeColor } as React.CSSProperties}
        data-testid="annotation-viewport"
      >
        <svg ref={overlayRef} className="annotation-overlay" aria-hidden="true">
          <polyline data-draft-path fill="none" stroke={activeColor} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
        </svg>
        <div className="brush-cursor" />
      </div>
    )
  },
)
