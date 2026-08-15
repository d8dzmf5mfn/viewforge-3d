import { unzip } from 'fflate'
import type { FaceManifest, FacePackage, QAReport, ViewRole } from './types'

const common = [
  'manifest.json',
  'qa/report.json',
  'references/front.png',
  'references/left45.png',
  'references/right45.png',
] as const
const pixelSurface = ['models/voxels.glb', 'pixels/pixels.bin', 'pixels/schema.json'] as const
const requiredV1 = [...common, ...pixelSurface, 'models/smooth.glb', 'models/skin.glb', 'textures/skin-atlas.jpg'] as const
const requiredV2 = [...common, ...pixelSurface, 'models/head.glb', 'textures/head-confidence.png', 'textures/head-source.png', 'qa/anatomy.json'] as const
const requiredV3 = [
  ...common,
  'models/head.glb',
  'textures/head-confidence.png',
  'textures/head-source.png',
  'projection/skin-projection.npz',
  'projection/schema.json',
  'qa/anatomy.json',
] as const
const optionalViewerEntries = [
  ...requiredV1,
  ...requiredV2,
  ...requiredV3,
  'textures/skin-confidence.png',
] as const
const MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
const MAX_ENTRIES = 100
const roles: ViewRole[] = ['front', 'left45', 'right45']
const decoder = new TextDecoder()
const viewerEntries = new Set<string>(optionalViewerEntries)

function parseJson<T>(bytes: Uint8Array, label: string): T {
  try {
    return JSON.parse(decoder.decode(bytes)) as T
  } catch {
    throw new Error(`${label} 不是有效 JSON`)
  }
}

function assertManifest(value: FaceManifest): void {
  const v1 = value.schemaVersion === '1.0.0' && value.subjectProfile === 'face-v1' && value.mode === 'pixel-direct'
  const v2 = value.schemaVersion === '2.0.0' && value.subjectProfile === 'face-v2' && value.mode === 'pixel-flame-hybrid'
  const v3 = value.schemaVersion === '3.0.0' && value.subjectProfile === 'face-v3' && value.mode === 'template-head-v0'
  if (!v1 && !v2 && !v3) {
    throw new Error(`不支持的结果包 schema/profile/mode: ${value.schemaVersion}/${value.subjectProfile}/${value.mode}`)
  }
  if (!value.mesh || !value.skin || !value.fit || !value.confidence) {
    throw new Error('manifest 缺少重建指标')
  }
  if (!v3 && (!value.pixel || !value.voxel)) throw new Error('manifest 缺少 3D Pixel 指标')
  if (v3 && (!value.projection || !value.sdf)) throw new Error('Face v3 缺少皮肤投影或 SDF QA 契约')
  if (v2 && (!value.anatomy || value.mesh.geometryHash !== value.skin.geometryHash || value.skin.maximumVertexDifference !== 0)) {
    throw new Error('Face v2 统一几何哈希或解剖报告无效')
  }
  if (v3 && (
    !value.anatomy
    || value.mesh.geometryHash !== value.skin.geometryHash
    || value.skin.maximumVertexDifference !== 0
    || value.sdf?.role !== 'qa-only'
    || value.sdf.surfaceGenerated
  )) {
    throw new Error('Face v3 同一几何或 SDF QA-only 契约无效')
  }
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new Uint8Array(bytes).buffer)
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('')
}

function assertPixelBinary(bytes: Uint8Array, manifest: FaceManifest): void {
  if (bytes.byteLength < 64) throw new Error('pixels.bin 小于文件头')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const magic = String.fromCharCode(...bytes.subarray(0, 4))
  const version = view.getUint16(4, true)
  const recordBytes = view.getUint16(6, true)
  const count = view.getUint32(8, true)
  const expectedVersion = manifest.schemaVersion === '2.0.0' ? 2 : 1
  const expectedRecordBytes = expectedVersion === 2 ? 48 : 36
  if (magic !== 'P2D3' || version !== expectedVersion || recordBytes !== expectedRecordBytes) {
    throw new Error('pixels.bin 格式与 manifest 不一致')
  }
  if (count !== manifest.pixel!.instanceCount || bytes.byteLength !== 64 + count * recordBytes) {
    throw new Error('pixels.bin 记录数量与 manifest 不一致')
  }
}

function objectUrl(bytes: Uint8Array, mime: string, urls: string[]): string {
  const url = URL.createObjectURL(new Blob([new Uint8Array(bytes)], { type: mime }))
  urls.push(url)
  return url
}

export async function loadFacePackage(file: File): Promise<FacePackage> {
  const started = performance.now()
  if (file.size > MAX_ARCHIVE_BYTES) throw new Error('结果包超过 80 MB 浏览器安全上限')
  const names: string[] = []
  let expandedBytes = 0
  const compressed = new Uint8Array(await file.arrayBuffer())
  const entries = await new Promise<Record<string, Uint8Array>>((resolve, reject) => {
    unzip(compressed, {
      filter: (entry) => {
        names.push(entry.name)
        expandedBytes += entry.originalSize
        return viewerEntries.has(entry.name)
      },
    }, (error, result) => error ? reject(error) : resolve(result))
  })
  if (names.length > MAX_ENTRIES || names.some((name) => name.includes('..') || name.startsWith('/'))) {
    throw new Error('结果包目录结构不安全')
  }
  if (expandedBytes > MAX_ARCHIVE_BYTES) throw new Error('结果包解压后超过 80 MB 安全上限')
  if (!entries['manifest.json']) throw new Error('结果包缺少 manifest.json')
  const manifest = parseJson<FaceManifest>(entries['manifest.json'], 'manifest.json')
  assertManifest(manifest)
  const required = manifest.schemaVersion === '3.0.0'
    ? requiredV3
    : manifest.schemaVersion === '2.0.0'
      ? requiredV2
      : requiredV1
  for (const entry of required) if (!entries[entry]) throw new Error(`结果包缺少 ${entry}`)
  if (manifest.schemaVersion === '3.0.0') {
    const projection = manifest.projection!
    const [binaryHash, schemaHash] = await Promise.all([
      sha256(entries['projection/skin-projection.npz']!),
      sha256(entries['projection/schema.json']!),
    ])
    if (binaryHash !== projection.binarySha256 || schemaHash !== projection.schemaSha256) {
      throw new Error('皮肤投影追溯文件哈希与 manifest 不一致')
    }
  } else {
    const pixelBinary = entries['pixels/pixels.bin']!
    const pixelSchema = entries['pixels/schema.json']!
    assertPixelBinary(pixelBinary, manifest)
    const [binaryHash, schemaHash] = await Promise.all([sha256(pixelBinary), sha256(pixelSchema)])
    if (binaryHash !== manifest.pixel!.binarySha256 || schemaHash !== manifest.pixel!.schemaSha256) {
      throw new Error('像素追溯文件哈希与 manifest 不一致')
    }
  }
  if (manifest.schemaVersion !== '1.0.0') {
    const headHash = await sha256(entries['models/head.glb']!)
    if (headHash !== manifest.skin.modelSha256) throw new Error('head.glb 哈希与 manifest 不一致')
  }
  const report = parseJson<QAReport>(entries['qa/report.json']!, 'qa/report.json')
  if (report.schemaVersion !== manifest.schemaVersion) throw new Error('报告与 manifest schemaVersion 不一致')
  const urls: string[] = []
  const references = Object.fromEntries(roles.map((role) => [
    role,
    objectUrl(entries[`references/${role}.png`]!, 'image/png', urls),
  ])) as Record<ViewRole, string>
  const unifiedHead = manifest.schemaVersion !== '1.0.0'
  const skinAtlasUrl = unifiedHead
    ? objectUrl(entries['textures/head-source.png']!, 'image/png', urls)
    : objectUrl(entries['textures/skin-atlas.jpg']!, 'image/jpeg', urls)
  const sourceMapUrl = unifiedHead ? skinAtlasUrl : undefined
  const confidenceBytes = unifiedHead
    ? entries['textures/head-confidence.png']
    : entries['textures/skin-confidence.png']
  const confidenceMapUrl = confidenceBytes ? objectUrl(confidenceBytes, 'image/png', urls) : undefined
  return {
    name: file.name.replace(/\.viewforge3d$/i, ''),
    manifest,
    report,
    references,
    voxelGlb: entries['models/voxels.glb'],
    smoothGlb: unifiedHead ? undefined : entries['models/smooth.glb'],
    skinGlb: unifiedHead ? undefined : entries['models/skin.glb'],
    headGlb: unifiedHead ? entries['models/head.glb'] : undefined,
    skinAtlasUrl,
    sourceMapUrl,
    confidenceMapUrl,
    loadDurationMs: performance.now() - started,
    dispose: () => urls.forEach((url) => URL.revokeObjectURL(url)),
  }
}
