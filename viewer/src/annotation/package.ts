import type {
  AnnotationPackage,
  ModelMetadata,
  ProxyMetadata,
  SurfaceAnnotation,
} from './types'

export function createAnnotationPackage(
  model: ModelMetadata,
  pickingProxy: ProxyMetadata,
  annotations: SurfaceAnnotation[],
  globalNotes: string,
): AnnotationPackage {
  return {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    route: 'profile-loft-preview',
    outputState: 'annotation-input',
    authority: {
      xy: 'user-annotated-region-and-intent',
      z: 'surface-samples-only-not-hidden-depth-authority',
      posterior: 'not-authorized-without-multiview-evidence',
    },
    model,
    pickingProxy,
    annotations: annotations.filter((annotation) => annotation.visible),
    globalNotes,
    requestedExecution: {
      immutableOutput: true,
      preserveTopology: true,
      preserveUvs: true,
      preserveMaterials: true,
      fixedQaViews: ['front', 'left45', 'right45', 'left90', 'right90', 'rear'],
    },
  }
}

export function validateAnnotationPackage(value: AnnotationPackage): string[] {
  const errors: string[] = []
  if (value.schemaVersion !== 1) errors.push('不支持的标注格式')
  if (value.model.modelSha256.length !== 64) errors.push('模型 SHA-256 无效')
  if (value.annotations.length === 0) errors.push('至少需要一个标注区域')
  value.annotations.forEach((annotation, index) => {
    if (annotation.screenPath.length < 3) errors.push(`标注 ${index + 1} 的屏幕路径不足 3 点`)
    if (annotation.surfacePath.length < 3) errors.push(`标注 ${index + 1} 的表面命中不足 3 点`)
    if (!annotation.note.trim()) errors.push(`标注 ${index + 1} 缺少给 Codex 的说明`)
  })
  return errors
}

export function annotationFilename(): string {
  return `viewforge3d-annotation-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
}

export function downloadAnnotationPackage(value: AnnotationPackage): void {
  const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = annotationFilename()
  link.click()
  URL.revokeObjectURL(url)
}
