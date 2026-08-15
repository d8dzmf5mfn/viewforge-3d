import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import type { CameraPreset, DisplayMode } from '../lib/types'

export interface ViewportHandle {
  capture: () => string
}

interface Props {
  voxelGlb?: Uint8Array
  smoothGlb?: Uint8Array
  skinGlb?: Uint8Array
  headGlb?: Uint8Array
  sourceMapUrl?: string
  mode: DisplayMode
  preset: CameraPreset
  onModelsReady?: () => void
}

interface DiagnosticObjects {
  eye: THREE.Object3D
  ear: THREE.Object3D
  projection: THREE.Object3D
}

function disposeObjects(roots: Array<THREE.Object3D | null>): void {
  const geometries = new Set<THREE.BufferGeometry>()
  const materials = new Set<THREE.Material>()
  const textures = new Set<THREE.Texture>()
  const imageBitmaps = new Set<ImageBitmap>()
  for (const root of roots) {
    root?.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return
      geometries.add(object.geometry)
      const meshMaterials = Array.isArray(object.material) ? object.material : [object.material]
      for (const material of meshMaterials) {
        materials.add(material)
        for (const value of Object.values(material)) if (value instanceof THREE.Texture) textures.add(value)
      }
    })
  }
  for (const texture of textures) {
    const image = texture.image
    if (typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) imageBitmaps.add(image)
    texture.dispose()
  }
  for (const image of imageBitmaps) image.close()
  for (const material of materials) material.dispose()
  for (const geometry of geometries) geometry.dispose()
}

function resourceCounts(roots: Array<THREE.Object3D | null>): { geometries: number; textures: number } {
  const geometries = new Set<THREE.BufferGeometry>()
  const textures = new Set<THREE.Texture>()
  for (const root of roots) root?.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return
    geometries.add(object.geometry)
    const materials = Array.isArray(object.material) ? object.material : [object.material]
    for (const material of materials) {
      for (const value of Object.values(material)) if (value instanceof THREE.Texture) textures.add(value)
    }
  })
  return { geometries: geometries.size, textures: textures.size }
}

function sharesGeometry(first: THREE.Object3D, second: THREE.Object3D): boolean {
  const firstGeometry: THREE.BufferGeometry[] = []
  const secondGeometry: THREE.BufferGeometry[] = []
  first.traverse((object) => { if (object instanceof THREE.Mesh) firstGeometry.push(object.geometry) })
  second.traverse((object) => { if (object instanceof THREE.Mesh) secondGeometry.push(object.geometry) })
  return firstGeometry.length === secondGeometry.length
    && firstGeometry.every((geometry, index) => geometry === secondGeometry[index])
}

async function parseGlb(bytes: Uint8Array): Promise<THREE.Object3D> {
  const copy = new Uint8Array(bytes).buffer
  return new Promise((resolve, reject) => {
    new GLTFLoader().parse(copy, '', (gltf) => {
      gltf.scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return
        if (!object.geometry.getAttribute('normal')) object.geometry.computeVertexNormals()
        const materials = Array.isArray(object.material) ? object.material : [object.material]
        for (const material of materials) {
          if (!(material instanceof THREE.MeshStandardMaterial)) continue
          material.flatShading = false
          material.metalness = 0
          material.envMapIntensity = 0.72
          material.needsUpdate = true
        }
      })
      resolve(gltf.scene)
    }, reject)
  })
}

function neutralClone(root: THREE.Object3D): THREE.Object3D {
  const clone = root.clone(true)
  clone.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return
    const isEye = object.name.includes('Eyeball') || object.parent?.name.includes('Eyeball')
    object.material = new THREE.MeshStandardMaterial({
      color: isEye ? 0xd9e7ef : 0x858b94,
      roughness: isEye ? 0.35 : 0.84,
      metalness: 0,
    })
  })
  return clone
}

function diagnosticClones(root: THREE.Object3D, sourceTexture?: THREE.Texture): DiagnosticObjects {
  const eye = root.clone(true)
  eye.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return
    const isEye = object.name.includes('Eyeball') || object.parent?.name.includes('Eyeball')
    object.material = new THREE.MeshStandardMaterial({
      color: isEye ? 0x22d3ee : 0x6b7280,
      emissive: isEye ? 0x064e5a : 0x000000,
      transparent: !isEye,
      opacity: isEye ? 1 : 0.56,
      depthWrite: isEye,
      roughness: 0.65,
      metalness: 0,
    })
  })
  const ear = root.clone(true)
  ear.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return
    const isEye = object.name.includes('Eyeball') || object.parent?.name.includes('Eyeball')
    object.visible = !isEye
    object.material = new THREE.MeshStandardMaterial({
      color: 0x67e8f9,
      emissive: 0x083344,
      wireframe: true,
      roughness: 0.8,
      metalness: 0,
    })
  })
  const projection = root.clone(true)
  projection.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return
    const isEye = object.name.includes('Eyeball') || object.parent?.name.includes('Eyeball')
    if (!isEye && sourceTexture) {
      object.material = new THREE.MeshStandardMaterial({
        map: sourceTexture,
        roughness: 0.78,
        metalness: 0,
      })
    }
  })
  return { eye, ear, projection }
}

async function loadTexture(url?: string): Promise<THREE.Texture | undefined> {
  if (!url) return undefined
  const texture = await new THREE.TextureLoader().loadAsync(url)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.flipY = false
  texture.needsUpdate = true
  return texture
}

function addEnvironment(scene: THREE.Scene): void {
  scene.background = new THREE.Color(0x111518)
  scene.add(new THREE.AmbientLight(0xf3f6fa, 0.34))
  scene.add(new THREE.HemisphereLight(0xeaf2ff, 0x46505a, 1.10))
  const key = new THREE.DirectionalLight(0xffffff, 2.10)
  key.position.set(-3, 5, 5)
  scene.add(key)
  const fill = new THREE.DirectionalLight(0xd8e4f4, 0.55)
  fill.position.set(3, 1, 5)
  scene.add(fill)
  const rim = new THREE.DirectionalLight(0x9db9ec, 0.75)
  rim.position.set(4, 2, -3)
  scene.add(rim)
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(3.2, 64),
    new THREE.MeshStandardMaterial({ color: 0x14191d, roughness: 0.92 }),
  )
  floor.rotation.x = -Math.PI / 2
  floor.position.y = -1.56
  scene.add(floor)
}

function normalizeModels(objects: Array<THREE.Object3D | null>, reference: THREE.Object3D | null): void {
  if (!reference) return
  for (const object of objects) {
    object?.position.set(0, 0, 0)
    object?.scale.setScalar(1)
  }
  const box = new THREE.Box3().setFromObject(reference)
  const size = box.getSize(new THREE.Vector3())
  const center = box.getCenter(new THREE.Vector3())
  const scale = 2.8 / Math.max(size.x, size.y, size.z, 1e-6)
  for (const object of objects) {
    if (!object) continue
    object.position.copy(center).multiplyScalar(-scale)
    object.scale.setScalar(scale)
  }
}

export const DualViewport = forwardRef<ViewportHandle, Props>(function DualViewport(
  { voxelGlb, smoothGlb, skinGlb, headGlb, sourceMapUrl, mode, preset, onModelsReady },
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const scenesRef = useRef<Record<'voxel' | 'smooth' | 'skin' | 'diagnostic', THREE.Scene> | null>(null)
  const objectsRef = useRef<{
    voxel: THREE.Object3D | null
    smooth: THREE.Object3D | null
    skin: THREE.Object3D | null
    diagnostics: DiagnosticObjects | null
  }>({ voxel: null, smooth: null, skin: null, diagnostics: null })
  const modeRef = useRef(mode)
  const splitRef = useRef(0.5)
  const headOnlyRef = useRef(false)
  modeRef.current = mode

  useImperativeHandle(ref, () => ({
    capture: () => canvasRef.current?.toDataURL('image/png') ?? '',
  }), [])

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.05
    renderer.setScissorTest(true)
    canvasRef.current = renderer.domElement
    host.appendChild(renderer.domElement)
    const camera = new THREE.PerspectiveCamera(31, 1, 0.01, 100)
    camera.position.set(3.4, 0.10, 4.8)
    cameraRef.current = camera
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 2.4
    controls.maxDistance = 8
    controls.target.set(0, -0.10, 0)
    controlsRef.current = controls
    const scenes = {
      voxel: new THREE.Scene(),
      smooth: new THREE.Scene(),
      skin: new THREE.Scene(),
      diagnostic: new THREE.Scene(),
    }
    Object.values(scenes).forEach(addEnvironment)
    scenesRef.current = scenes
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
      const objects = objectsRef.current.diagnostics
      if (objects) {
        objects.eye.visible = modeRef.current === 'eye-contact'
        objects.ear.visible = modeRef.current === 'ear-continuity'
        objects.projection.visible = modeRef.current === 'skin-projection'
      }
      const comparison = modeRef.current === 'comparison' && window.innerWidth >= 900
      if (comparison) {
        const split = Math.floor(width * splitRef.current)
        camera.aspect = split / height
        camera.updateProjectionMatrix()
        renderer.setViewport(0, 0, split, height)
        renderer.setScissor(0, 0, split, height)
        renderer.render(headOnlyRef.current ? scenes.smooth : scenes.voxel, camera)
        camera.aspect = (width - split) / height
        camera.updateProjectionMatrix()
        renderer.setViewport(split, 0, width - split, height)
        renderer.setScissor(split, 0, width - split, height)
        renderer.render(headOnlyRef.current ? scenes.skin : scenes.smooth, camera)
      } else {
        camera.aspect = width / height
        camera.updateProjectionMatrix()
        renderer.setViewport(0, 0, width, height)
        renderer.setScissor(0, 0, width, height)
        const selected = modeRef.current === 'voxel'
          ? (headOnlyRef.current ? scenes.smooth : scenes.voxel)
          : modeRef.current === 'skin'
            ? scenes.skin
            : modeRef.current === 'smooth' || modeRef.current === 'comparison'
              ? scenes.smooth
              : scenes.diagnostic
        renderer.render(selected, camera)
      }
    }
    render()
    const teardown = () => {
      if (disposed) return
      disposed = true
      cancelAnimationFrame(frame)
      resize.disconnect()
      controls.dispose()
      disposeObjects(Object.values(scenes))
      renderer.renderLists.dispose()
      renderer.dispose()
      renderer.forceContextLoss()
      renderer.domElement.remove()
      canvasRef.current = null
      cameraRef.current = null
      controlsRef.current = null
      scenesRef.current = null
    }
    window.addEventListener('pagehide', teardown)
    return () => {
      window.removeEventListener('pagehide', teardown)
      teardown()
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    async function loadModels() {
      const scenes = scenesRef.current
      if (!scenes) return
      const current = objectsRef.current
      const oldObjects = [
        current.voxel,
        current.smooth,
        current.skin,
        current.diagnostics?.eye ?? null,
        current.diagnostics?.ear ?? null,
        current.diagnostics?.projection ?? null,
      ]
      for (const object of oldObjects) object?.removeFromParent()
      disposeObjects(oldObjects)
      objectsRef.current = { voxel: null, smooth: null, skin: null, diagnostics: null }
      headOnlyRef.current = false
      delete hostRef.current?.dataset.skinReady
      if (oldObjects.some(Boolean)) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
        if (cancelled) return
      }

      const voxelPromise = voxelGlb ? parseGlb(voxelGlb) : Promise.resolve(null)
      if (headGlb) {
        const [voxel, head, sourceTexture] = await Promise.all([
          voxelPromise,
          parseGlb(headGlb),
          loadTexture(sourceMapUrl),
        ])
        const smooth = neutralClone(head)
        const diagnostics = diagnosticClones(head, sourceTexture)
        if (cancelled) {
          disposeObjects([voxel, head, smooth, diagnostics.eye, diagnostics.ear, diagnostics.projection])
          return
        }
        objectsRef.current = { voxel, smooth, skin: head, diagnostics }
        headOnlyRef.current = !voxel
        if (voxel) scenes.voxel.add(voxel)
        scenes.smooth.add(smooth)
        scenes.skin.add(head)
        scenes.diagnostic.add(diagnostics.eye, diagnostics.ear, diagnostics.projection)
        normalizeModels(
          [voxel, smooth, head, diagnostics.eye, diagnostics.ear, diagnostics.projection],
          smooth,
        )
        if (hostRef.current) hostRef.current.dataset.skinReady = 'true'
        if (hostRef.current) {
          const counts = resourceCounts([
            voxel,
            smooth,
            head,
            diagnostics.eye,
            diagnostics.ear,
            diagnostics.projection,
          ])
          hostRef.current.dataset.headParseCount = '1'
          hostRef.current.dataset.sharedGeometry = String(sharesGeometry(head, smooth))
          hostRef.current.dataset.resourceGeometries = String(counts.geometries)
          hostRef.current.dataset.resourceTextures = String(counts.textures)
          hostRef.current.dataset.modelGeneration = String(
            Number(hostRef.current.dataset.modelGeneration ?? '0') + 1,
          )
        }
        onModelsReady?.()
        return
      }

      const [voxel, smooth, skin] = await Promise.all([
        voxelPromise,
        smoothGlb ? parseGlb(smoothGlb) : Promise.resolve(null),
        skinGlb ? parseGlb(skinGlb) : Promise.resolve(null),
      ])
      if (cancelled) {
        disposeObjects([voxel, smooth, skin])
        return
      }
      objectsRef.current = { voxel, smooth, skin, diagnostics: null }
      if (hostRef.current) {
        delete hostRef.current.dataset.headParseCount
        delete hostRef.current.dataset.sharedGeometry
        delete hostRef.current.dataset.resourceGeometries
        delete hostRef.current.dataset.resourceTextures
      }
      if (voxel) scenes.voxel.add(voxel)
      if (smooth) scenes.smooth.add(smooth)
      if (skin) scenes.skin.add(skin)
      normalizeModels([voxel, smooth, skin], smooth ?? skin ?? voxel)
      if (skin && hostRef.current) hostRef.current.dataset.skinReady = 'true'
      if (voxel || smooth || skin) onModelsReady?.()
    }
    void loadModels()
    return () => { cancelled = true }
  }, [voxelGlb, smoothGlb, skinGlb, headGlb, sourceMapUrl, onModelsReady])

  useEffect(() => {
    const camera = cameraRef.current
    const controls = controlsRef.current
    if (!camera || !controls) return
    const positions: Record<CameraPreset, [number, number, number]> = {
      front: [0, 0.05, 6.0],
      side: [6.0, 0.05, 0],
      'three-quarter': [3.4, 0.10, 4.8],
    }
    camera.position.set(...positions[preset])
    controls.target.set(0, -0.10, 0)
    controls.update()
  }, [preset])

  function updateSplit(clientX: number) {
    const host = hostRef.current
    if (!host) return
    const bounds = host.getBoundingClientRect()
    const split = Math.min(0.75, Math.max(0.25, (clientX - bounds.left) / bounds.width))
    splitRef.current = split
    host.style.setProperty('--viewport-split', `${split * 100}%`)
  }

  function startSplitDrag(event: ReactPointerEvent<HTMLButtonElement>) {
    event.currentTarget.setPointerCapture(event.pointerId)
    updateSplit(event.clientX)
  }

  function moveSplit(event: ReactPointerEvent<HTMLButtonElement>) {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) updateSplit(event.clientX)
  }

  return (
    <div className="viewport-host" ref={hostRef}>
      {mode === 'comparison' && <button className="viewport-split" aria-label="调整对照分隔线" onPointerDown={startSplitDrag} onPointerMove={moveSplit}><span aria-hidden="true">‹ ›</span></button>}
      {!voxelGlb && !smoothGlb && !headGlb && <div className="viewport-empty"><strong>加载 `.viewforge3d` 结果包</strong><span>模型、照片和报告只在当前浏览器标签页解析</span></div>}
    </div>
  )
})

export default DualViewport
