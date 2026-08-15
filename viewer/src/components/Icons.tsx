import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement>

const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.7,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
}

export function CompareIcon(props: IconProps) {
  return <svg {...base} {...props}><path d="M4 5.5h6.2v13H4zM13.8 5.5H20v13h-6.2z" /><path d="M10.2 8.2h3.6M10.2 15.8h3.6" /></svg>
}

export function PixelIcon(props: IconProps) {
  return <svg {...base} {...props}><path d="m12 2.8 7.5 4.3v8.7L12 20.2l-7.5-4.4V7.1z" /><path d="m4.8 7.3 7.2 4.2 7.2-4.2M12 11.5v8.3M8 5.1l7.6 4.4M16 5.1 8.4 9.5" /></svg>
}

export function MeshIcon(props: IconProps) {
  return <svg {...base} {...props}><circle cx="12" cy="12" r="8.4" /><path d="M3.9 12h16.2M12 3.6c2.2 2.2 3.3 5 3.3 8.4S14.2 18.2 12 20.4M12 3.6C9.8 5.8 8.7 8.6 8.7 12s1.1 6.2 3.3 8.4M5.7 7.2h12.6M5.7 16.8h12.6" /></svg>
}

export function HeadIcon({ direction = 'front', ...props }: IconProps & { direction?: 'front' | 'side' | 'three-quarter' }) {
  if (direction === 'side') return <svg {...base} {...props}><path d="M14.6 3.5c-4.4-.6-7.2 2.5-7.2 6.7 0 2.2.8 4.1 2.3 5.4v3.2h5.1v-2.2c1.5-.6 2.5-1.8 2.7-3.4l2-1.4-2.1-1.2c.1-3.5-.6-6.6-2.8-7.1Z" /><path d="M13.6 8.5h.1M13.2 12.4c.8.4 1.5.4 2.1 0" /></svg>
  return <svg {...base} {...props}><path d="M5.8 10.1C5.8 5.8 8.1 3 12 3s6.2 2.8 6.2 7.1c0 4.9-2.2 9-6.2 10.2-4-1.2-6.2-5.3-6.2-10.2Z" /><path d="M8.6 10.3h1.7M13.7 10.3h1.7M10.2 15.6c1.2.7 2.4.7 3.6 0M12 11v2.2" />{direction === 'three-quarter' && <path d="M12 3.2c2.6 1.7 3.8 4.1 3.8 7.2 0 4.3-1.4 7.5-3.8 9.6" />}</svg>
}
