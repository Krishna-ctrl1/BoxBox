import { useJson } from './lib'

/* Race identity: Grand Prix names, circuit maps and photography.
   Lives in /assets (separate from the engine's /data files). */

export type CircuitMapData = { path: string; w: number; h: number; start: [number, number]; toward: [number, number] }
export type RaceMeta = {
  round: number
  gp: string
  official: string | null
  country: string | null
  country_code: string | null
  location: string | null
  circuit_type: string | null
  map: CircuitMapData | null
}
export type Photo = {
  id: string
  race: string
  alt: string
  src1200: string
  src2400: string
  title: string
  author: string
  license: string
  license_url: string
  page: string
  changes: string
}

export function useIdentity() {
  const meta = useJson<{ year: number; races: Record<string, RaceMeta> }>('/assets/races-meta.json')
  const photos = useJson<Photo[]>('/assets/photos/photos.json')
  return { meta: meta.data?.races ?? null, photos: photos.data ?? null }
}

export function photosFor(photos: Photo[] | null, race: string): Photo[] {
  return (photos ?? []).filter((p) => p.race === race)
}

export function PhotoImg({ photo, className = '', eager = false, sizes = '(min-width: 1024px) 60vw, 100vw' }: { photo: Photo; className?: string; eager?: boolean; sizes?: string }) {
  return (
    <img
      src={photo.src1200}
      srcSet={`${photo.src1200} 1200w, ${photo.src2400} 2400w`}
      sizes={sizes}
      alt={photo.alt}
      loading={eager ? 'eager' : 'lazy'}
      decoding="async"
      width={1200}
      height={675}
      className={`photo ${className}`}
    />
  )
}

/** The circuit drawn from the race winner's own GPS trace, with the start/finish line and direction of travel. */
export function CircuitMap({ map, className = '', stroke = 'var(--color-ink)', width = 10, label }: { map: CircuitMapData; className?: string; stroke?: string; width?: number; label: string }) {
  const [sx, sy] = map.start
  const [tx, ty] = map.toward
  const ang = (Math.atan2(ty - sy, tx - sx) * 180) / Math.PI
  return (
    <svg viewBox={`0 0 ${map.w} ${map.h}`} className={className} role="img" aria-label={label} preserveAspectRatio="xMidYMid meet">
      <path d={map.path} fill="none" stroke="var(--color-line)" strokeWidth={width * 2.6} strokeLinejoin="round" strokeLinecap="round" />
      <path d={map.path} fill="none" stroke={stroke} strokeWidth={width} strokeLinejoin="round" strokeLinecap="round" />
      <g transform={`translate(${sx} ${sy}) rotate(${ang})`}>
        <rect x={-3} y={-width * 2.2} width={6} height={width * 4.4} fill="var(--color-accent)" />
        <path d={`M ${width * 2.2} ${-width * 1.3} L ${width * 4.4} 0 L ${width * 2.2} ${width * 1.3}`} fill="none" stroke="var(--color-accent)" strokeWidth={width * 0.55} strokeLinecap="round" strokeLinejoin="round" />
      </g>
    </svg>
  )
}

export function fmtDate(iso: string) {
  const d = new Date(`${iso}T12:00:00Z`)
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
}
