import { Failed, Label, Loading, Panel, Stat, useJson } from '../lib'
import type { RaceEntry, Results } from '../types'

const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN)

export default function Home() {
  const { data, error } = useJson<{ year: number; races: RaceEntry[] }>('/data/races.json')
  const { data: res } = useJson<Results>('/data/results.json')
  const v = res?.validation ?? []
  const w = v.reduce((a, r) => a + r.n, 0)
  const lapErr = w ? v.reduce((a, r) => a + r.model * r.n, 0) / w : NaN
  const lapBase = w ? v.reduce((a, r) => a + r.recent * r.n, 0) / w : NaN
  const sb = res?.sim_backtest ?? []
  const fav = avg(sb.map((r) => Number(r.model_top_pick_won)))
  const lead = avg(sb.map((r) => Number(r.leader_won)))
  const st = res?.strategy ?? []
  const gain = avg(st.map((r) => r.team_pos - r.rule_pos))
  const same = avg(st.filter((r) => r.bb_lap != null).map((r) => Number(r.bb_tyre === r.team_tyre)))
  const show = (x: number, f: (x: number) => string) => (Number.isFinite(x) ? f(x) : '…')
  return (
    <>
      <section className="border-b border-line">
        <div className="wrap grid gap-10 py-14 md:grid-cols-12 md:py-20">
          <div className="md:col-span-7">
            <Label>F1 strategy · quantitative finance</Label>
            <h1 className="mt-4 font-display text-5xl font-bold leading-[0.95] tracking-tight md:text-7xl">
              Pit stops, priced like <span className="text-amber">options.</span>
            </h1>
            <p className="mt-6 max-w-[60ch] text-lg leading-relaxed text-mute">
              Deciding when to pit is the same maths banks use for American options: tyres lose pace every lap, safety cars
              arrive at random, and a stop costs about 20 seconds. BoxBox learns tyre wear from real timing data, plays the rest
              of a race thousands of times, and finds the best lap to box. Pick any 2024 race below.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <a href="#/race/sakhir" className="rounded-lg bg-amber px-5 py-3 font-semibold text-bg hover:brightness-110">
                Open Bahrain 2024 →
              </a>
              <a href="#/methods" className="rounded-lg border border-line px-5 py-3 font-semibold hover:border-mute">
                How it works
              </a>
            </div>
          </div>
          <Panel className="p-6 md:col-span-5 md:self-end">
            <Label>Tested on 2024, trained only on the past</Label>
            <div className="mt-5 grid grid-cols-2 gap-6">
              <Stat value={show(lapErr, (x) => `${x.toFixed(2)} s`)} label={`lap-time error (baseline ${show(lapBase, (x) => `${x.toFixed(2)} s`)})`} accent />
              <Stat value={show(fav, (x) => `${Math.round(x * 100)}%`)} label={`simulator’s favourite won (race leader at that point: ${show(lead, (x) => `${Math.round(x * 100)}%`)})`} />
              <Stat value={show(gain, (x) => `+${x.toFixed(2)}`)} label="places per pit call vs the real teams*" accent />
              <Stat value={show(same, (x) => `${Math.round(x * 100)}%`)} label="same tyre choice as the teams" />
            </div>
            <p className="mt-5 text-xs text-dim">* inside BoxBox’s own simulator; see Results for what that means.</p>
          </Panel>
        </div>
      </section>

      <section className="wrap py-12">
        <div className="mb-6 flex items-end justify-between gap-4">
          <h2 className="font-display text-3xl font-bold">2024 season</h2>
          <div className="text-sm text-mute">Wet races have a replay only.</div>
        </div>
        {error && <Failed error={error} />}
        {!data && !error && <Loading what="races" />}
        {data && (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {data.races.map((r, i) => (
              <a key={r.slug} href={`#/race/${r.slug}`} className="group rounded-xl border border-line bg-panel p-4 transition hover:border-amber">
                <div className="flex items-center justify-between">
                  <span className="num text-xs text-dim">R{String(i + 1).padStart(2, '0')} · {r.date.slice(5)}</span>
                  {r.wet && <span className="rounded bg-wet/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-wet">wet</span>}
                </div>
                <div className="mt-2 font-display text-xl font-bold group-hover:text-amber">{r.name}</div>
                <div className="text-sm text-mute">{r.circuit}</div>
                <div className="num mt-3 flex gap-3 text-xs text-mute">
                  <span>{r.laps} laps</span>
                  <span>SC {r.sc}</span>
                  <span>VSC {r.vsc}</span>
                </div>
              </a>
            ))}
          </div>
        )}
      </section>
    </>
  )
}
