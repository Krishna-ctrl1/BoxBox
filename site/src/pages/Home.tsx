import { Failed, Loading, useJson } from '../lib'
import type { RaceEntry, Results } from '../types'
import { CircuitMap, PhotoImg, fmtDate, photosFor, useIdentity } from '../identity'
import type { Photo, RaceMeta } from '../identity'

const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN)
const show = (x: number, f: (x: number) => string) => (Number.isFinite(x) ? f(x) : '…')

function useSeasonStats() {
  const { data: res } = useJson<Results>('/data/results.json')
  const v = res?.validation ?? []
  const w = v.reduce((a, r) => a + r.n, 0)
  const sb = res?.sim_backtest ?? []
  const st = res?.strategy ?? []
  return {
    lapErr: w ? v.reduce((a, r) => a + r.model * r.n, 0) / w : NaN,
    lapBase: w ? v.reduce((a, r) => a + r.recent * r.n, 0) / w : NaN,
    fav: avg(sb.map((r) => Number(r.model_top_pick_won))),
    lead: avg(sb.map((r) => Number(r.leader_won))),
    gain: avg(st.map((r) => r.team_pos - r.rule_pos)),
    same: avg(st.filter((r) => r.bb_lap != null).map((r) => Number(r.bb_tyre === r.team_tyre))),
  }
}

function Hero({ photo, races }: { photo: Photo | undefined; races: RaceEntry[] | null }) {
  const s = useSeasonStats()
  return (
    <section className="relative isolate overflow-hidden border-b border-line">
      <div className="absolute inset-0 -z-10">
        {photo && <PhotoImg photo={photo} eager sizes="100vw" className="object-[60%_50%]" />}
        <div className="shade-left absolute inset-0" />
        <div className="shade-bottom absolute inset-0" />
        <div className="absolute inset-0 bg-bg/50 md:hidden" />
      </div>
      <div className="wrap flex min-h-[78vh] flex-col justify-end pb-10 pt-28 md:min-h-[86vh] md:pb-14">
        <div className="eyebrow">BoxBox · Formula 1 strategy intelligence · 2024 season</div>
        <h1 className="display mt-5 max-w-[14ch] text-[clamp(3rem,9vw,8.5rem)] font-semibold leading-[0.9]">
          Every lap.<br />Every decision.
        </h1>
        <p className="mt-6 max-w-[48ch] text-lg leading-relaxed text-ink/80 md:text-xl">
          The science behind every strategic move on the circuit. Real timing data, thousands of simulated races, and pit stops
          priced like financial options.
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-4">
          <a
            href="#season"
            onClick={(e) => { e.preventDefault(); document.getElementById('season')?.scrollIntoView({ behavior: 'smooth' }) }}
            className="bg-accent px-6 py-3.5 text-sm font-semibold uppercase tracking-[0.14em] text-ink transition hover:brightness-110"
          >
            Explore the 2024 season
          </a>
          <a href="#/methods" className="border border-ink/30 px-6 py-3.5 text-sm font-semibold uppercase tracking-[0.14em] text-ink transition hover:border-ink">
            The method
          </a>
        </div>
        <dl className="mt-14 grid grid-cols-2 gap-x-6 gap-y-6 border-t border-ink/15 pt-6 md:grid-cols-4">
          <div>
            <dt className="eyebrow">Races analysed</dt>
            <dd className="num mt-2 text-3xl">{races ? races.length : '…'}</dd>
          </div>
          <div>
            <dt className="eyebrow">Lap-time error</dt>
            <dd className="num mt-2 text-3xl">{show(s.lapErr, (x) => `${x.toFixed(2)}s`)}</dd>
            <dd className="mt-1 text-xs text-mute">baseline {show(s.lapBase, (x) => `${x.toFixed(2)}s`)}</dd>
          </div>
          <div>
            <dt className="eyebrow">Favourite won</dt>
            <dd className="num mt-2 text-3xl">{show(s.fav, (x) => `${Math.round(x * 100)}%`)}</dd>
            <dd className="mt-1 text-xs text-mute">race leader at that point: {show(s.lead, (x) => `${Math.round(x * 100)}%`)}</dd>
          </div>
          <div>
            <dt className="eyebrow">Vs the teams’ pit calls</dt>
            <dd className="num mt-2 text-3xl">{show(s.gain, (x) => `+${x.toFixed(2)}`)}</dd>
            <dd className="mt-1 text-xs text-mute">places per decision, in-model · same tyre {show(s.same, (x) => `${Math.round(x * 100)}%`)}</dd>
          </div>
        </dl>
      </div>
      {photo && (
        <a href="#/credits" className="absolute bottom-3 right-4 hidden text-[10px] text-ink/45 hover:text-ink/80 md:block">
          Photo: {photo.author} · {photo.license}
        </a>
      )}
    </section>
  )
}

function Story({ photo }: { photo: Photo | undefined }) {
  return (
    <section className="wrap grid gap-10 py-20 md:grid-cols-12 md:items-center md:py-28">
      <div className="md:col-span-5">
        <div className="eyebrow">The idea</div>
        <h2 className="display mt-4 text-4xl font-semibold leading-tight md:text-5xl">A pit stop is an option. BoxBox prices it.</h2>
        <p className="mt-6 leading-relaxed text-mute">
          Tyres lose pace every lap, like time decay. Safety cars arrive without warning, like a market jump. A stop costs around twenty
          seconds: the exercise price. BoxBox learns each circuit from real timing data, plays the rest of the race thousands of times,
          and finds the lap where boxing pays, with the same method banks use to value American options.
        </p>
        <div className="mt-8 grid grid-cols-3 gap-4 border-t border-line pt-6 text-sm">
          <div><div className="num text-2xl">2,000</div><div className="mt-1 text-mute">races simulated at once</div></div>
          <div><div className="num text-2xl">51k</div><div className="mt-1 text-mute">real laps learned from</div></div>
          <div><div className="num text-2xl">34</div><div className="mt-1 text-mute">pit duels solved as games</div></div>
        </div>
      </div>
      <figure className="relative aspect-[16/10] overflow-hidden md:col-span-7 md:col-start-6">
        {photo && <PhotoImg photo={photo} />}
        {photo && <figcaption className="absolute bottom-0 left-0 bg-bg/80 px-3 py-2 text-[11px] text-mute">{photo.alt}</figcaption>}
      </figure>
    </section>
  )
}

function RaceCard({ r, meta, photo, wide }: { r: RaceEntry; meta: RaceMeta | undefined; photo: Photo | undefined; wide: boolean }) {
  const title = meta?.gp ?? `${r.name} Grand Prix`
  return (
    <a
      href={`#/race/${r.slug}`}
      className={`group relative isolate flex flex-col overflow-hidden border border-line bg-panel transition hover:border-ink/40 ${wide ? 'sm:col-span-2' : ''}`}
    >
      <div className={`relative overflow-hidden ${wide ? 'aspect-[4/3] sm:aspect-[8/3]' : 'aspect-[4/3]'}`}>
        {photo ? (
          <>
            <PhotoImg photo={photo} className="transition duration-700 group-hover:scale-[1.03]" sizes="(min-width: 1024px) 50vw, 100vw" />
            <div className="shade-bottom absolute inset-0" />
          </>
        ) : (
          <div className="grid-lines absolute inset-0 grid place-items-center bg-panel-2 p-8">
            {meta?.map && <CircuitMap map={meta.map} label={`${r.circuit} circuit layout`} className="h-full w-full opacity-90 transition duration-500 group-hover:opacity-100" width={9} />}
          </div>
        )}
        <div className="absolute left-4 top-4 flex items-center gap-2">
          <span className="num bg-bg/85 px-2 py-1 text-xs">R{String(meta?.round ?? '').padStart(2, '0')}</span>
          {r.wet && <span className="bg-wet/85 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider">Wet</span>}
        </div>
        {photo && meta?.map && (
          <div className="absolute bottom-3 right-3 h-16 w-20 opacity-90">
            <CircuitMap map={meta.map} label={`${r.circuit} circuit layout`} className="h-full w-full" width={14} />
          </div>
        )}
      </div>
      <div className="flex flex-1 flex-col gap-1 p-4">
        <div className="eyebrow">{fmtDate(r.date)} · {meta?.country ?? r.name}</div>
        <div className="display text-xl font-semibold">{title}</div>
        <div className="text-sm text-mute">{r.circuit}</div>
        <div className="num mt-auto flex gap-4 pt-3 text-xs text-mute">
          <span>{r.laps} laps</span>
          <span>{r.sc} SC</span>
          <span>{r.vsc} VSC</span>
        </div>
      </div>
    </a>
  )
}

export default function Home() {
  const { data, error } = useJson<{ year: number; races: RaceEntry[] }>('/data/races.json')
  const { meta, photos } = useIdentity()
  const hero = (photos ?? []).find((p) => p.id === 'hamilton-silverstone')
  const story = (photos ?? []).find((p) => p.id === 'mclaren-pan-suzuka')
  const cardPhoto: Record<string, string> = { silverstone: 'gasly-silverstone', suzuka: 'verstappen-suzuka', singapore: 'bottas-singapore' }
  return (
    <>
      <Hero photo={hero} races={data?.races ?? null} />
      <Story photo={story} />
      <section id="season" className="wrap scroll-mt-20 pb-10">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-line pb-6">
          <div>
            <div className="eyebrow">Season dossier</div>
            <h2 className="display mt-3 text-4xl font-semibold md:text-5xl">2024 · Twenty-four races</h2>
          </div>
          <p className="max-w-[44ch] text-sm text-mute">Every circuit is drawn from the race winner’s own GPS trace. Open any race for the replay, odds, pit window and pit duels.</p>
        </div>
        {error && <Failed error={error} />}
        {!data && !error && <Loading what="races" />}
        {data && (
          <div className="mt-8 grid gap-4 sm:grid-flow-row-dense sm:grid-cols-2 lg:grid-flow-row lg:grid-cols-4">
            {data.races.map((r) => {
              const p = cardPhoto[r.slug] ? photosFor(photos, r.slug).find((x) => x.id === cardPhoto[r.slug]) : undefined
              return <RaceCard key={r.slug} r={r} meta={meta?.[r.slug]} photo={p} wide={!!p || r.slug === 'sakhir'} />
            })}
          </div>
        )}
      </section>
    </>
  )
}
