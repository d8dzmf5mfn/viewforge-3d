import type { CameraPreset, DisplayMode } from '../lib/types'
import { CompareIcon, HeadIcon, MeshIcon, PixelIcon } from './Icons'

interface Props {
  mode: DisplayMode
  preset: CameraPreset
  showVoxel?: boolean
  onMode: (mode: DisplayMode) => void
  onPreset: (preset: CameraPreset) => void
}

export function ViewToolbar({ mode, preset, showVoxel = true, onMode, onPreset }: Props) {
  return (
    <div className="view-toolbar" aria-label="查看控制">
      <div className="segmented mode-control">
        <button className={mode === 'comparison' ? 'selected' : ''} onClick={() => onMode('comparison')}><CompareIcon />对照</button>
        {showVoxel && <button className={mode === 'voxel' ? 'selected' : ''} onClick={() => onMode('voxel')}><PixelIcon />3D Pixel</button>}
        <button className={mode === 'smooth' ? 'selected' : ''} onClick={() => onMode('smooth')}><MeshIcon />平滑网格</button>
        <button className={mode === 'skin' ? 'selected' : ''} onClick={() => onMode('skin')}><HeadIcon direction="front" />人皮</button>
        <button className={mode === 'eye-contact' ? 'selected' : ''} onClick={() => onMode('eye-contact')}><HeadIcon direction="front" />眼球接触</button>
        <button className={mode === 'ear-continuity' ? 'selected' : ''} onClick={() => onMode('ear-continuity')}><MeshIcon />耳根连续</button>
        <button className={mode === 'skin-projection' ? 'selected' : ''} onClick={() => onMode('skin-projection')}><CompareIcon />皮肤投影</button>
      </div>
      <span className="toolbar-divider" />
      <div className="segmented preset-control">
        <button className={preset === 'front' ? 'selected' : ''} onClick={() => onPreset('front')}><HeadIcon direction="front" />正面</button>
        <button className={preset === 'side' ? 'selected' : ''} onClick={() => onPreset('side')}><HeadIcon direction="side" />侧面</button>
        <button className={preset === 'three-quarter' ? 'selected' : ''} onClick={() => onPreset('three-quarter')}><HeadIcon direction="three-quarter" />三分之四</button>
      </div>
    </div>
  )
}
