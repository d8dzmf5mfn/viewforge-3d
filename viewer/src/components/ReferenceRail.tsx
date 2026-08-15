import type { FacePackage, ViewRole } from '../lib/types'

const items: Array<{ role: ViewRole; label: string }> = [
  { role: 'front', label: '正面' },
  { role: 'left45', label: '左侧' },
  { role: 'right45', label: '右侧' },
]

export function ReferenceRail({ facePackage }: { facePackage: FacePackage | null }) {
  return (
    <aside className="reference-rail" aria-label="参考视图">
      {items.map(({ role, label }) => (
        <figure className="reference-item" key={role}>
          <figcaption>{label}</figcaption>
          <div className="reference-frame">
            {facePackage ? <img src={facePackage.references[role]} alt={`${label}参考图`} /> : <div className="reference-empty" />}
          </div>
        </figure>
      ))}
      <figure className="reference-item skin-atlas-item">
        <figcaption>{facePackage && facePackage.manifest.schemaVersion !== '1.0.0' ? '皮肤来源图' : '人皮展开'}</figcaption>
        <div className="reference-frame skin-atlas-frame">
          {facePackage?.skinAtlasUrl ? <img src={facePackage.skinAtlasUrl} alt="完整人头皮肤 UV 展开图" /> : <div className="reference-empty" />}
        </div>
      </figure>
    </aside>
  )
}
