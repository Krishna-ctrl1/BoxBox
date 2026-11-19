import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'

export function useJson<T>(url: string | null): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (!url) return
    let alive = true
    setData(null)
    setError(null)
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} loading ${url}`))))
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e.message || e)))
    return () => {
      alive = false
    }
  }, [url])
  return { data, error }
}

export function useHash(): string {
  const [hash, setHash] = useState(() => window.location.hash || '#/')
  useEffect(() => {
    const on = () => {
      setHash(window.location.hash || '#/')
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return hash
}

export const TYRE: Record<string, { colour: string; letter: string; name: string; text: string }> = {
  SOFT: { colour: 'var(--color-soft)', letter: 'S', name: 'Soft', text: '#fff' },
  MEDIUM: { colour: 'var(--color-medium)', letter: 'M', name: 'Medium', text: '#111' },
  HARD: { colour: 'var(--color-hard)', letter: 'H', name: 'Hard', text: '#111' },
  INTERMEDIATE: { colour: 'var(--color-inter)', letter: 'I', name: 'Intermediate', text: '#111' },
  WET: { colour: 'var(--color-wet)', letter: 'W', name: 'Wet', text: '#fff' },
}
export const TYRE_BY_LETTER: Record<string, string> = { S: 'SOFT', M: 'MEDIUM', H: 'HARD', I: 'INTERMEDIATE', W: 'WET' }

export function Tyre({ compound, age, size = 22 }: { compound: string; age?: number | null; size?: number }) {
  const t = TYRE[compound] ?? TYRE[TYRE_BY_LETTER[compound] ?? ''] ?? { colour: '#666', letter: '?', name: compound, text: '#fff' }
  return (
    <span className="inline-flex items-center gap-1.5" title={`${t.name}${age != null ? `, ${age} laps old` : ''}`}>
      <span
        className="inline-grid place-items-center rounded-full font-bold num"
        style={{ width: size, height: size, border: `3px solid ${t.colour}`, color: 'var(--color-ink)', fontSize: size * 0.45 }}
      >
        {t.letter}
      </span>
      {age != null && <span className="num text-xs text-mute">{age}</span>}
    </span>
  )
}

export const pct = (x: number | null | undefined, d = 0) => (x == null ? '–' : `${(x * 100).toFixed(d)}%`)
export const fx = (x: number | null | undefined, d = 2) => (x == null ? '–' : x.toFixed(d))

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-xl border border-line bg-panel ${className}`}>{children}</div>
}

export function Label({ children }: { children: ReactNode }) {
  return <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-mute">{children}</div>
}

export function Stat({ value, label, accent = false }: { value: ReactNode; label: ReactNode; accent?: boolean }) {
  return (
    <div className="flex flex-col gap-1">
      <div className={`num text-2xl md:text-3xl font-medium ${accent ? 'text-amber' : 'text-ink'}`}>{value}</div>
      <div className="text-sm text-mute">{label}</div>
    </div>
  )
}

export function Loading({ what }: { what: string }) {
  return <div className="py-24 text-center text-mute">Loading {what}…</div>
}

export function Failed({ error }: { error: string }) {
  return (
    <div className="py-24 text-center">
      <div className="text-bad font-semibold">Couldn’t load the data.</div>
      <div className="mt-2 text-sm text-mute">{error}. Refresh the page to try again.</div>
    </div>
  )
}

export const axis = { stroke: 'var(--color-line)', tick: { fill: 'var(--color-mute)', fontSize: 11, fontFamily: 'var(--font-mono)' } }
export const tooltipStyle = {
  contentStyle: { background: '#0f1318', border: '1px solid var(--color-line)', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: 'var(--color-mute)' },
  itemStyle: { color: 'var(--color-ink)' },
}
