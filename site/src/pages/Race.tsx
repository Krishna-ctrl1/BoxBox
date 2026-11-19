import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Failed, Label, Loading, Panel, Stat, TYRE, TYRE_BY_LETTER, Tyre, axis, fx, pct, tooltipStyle, useJson } from '../lib'
import type { Duel, Odds, Pit, RaceDoc } from '../types'

type Tab = 'replay' | 'odds' | 'pit' | 'duels'

export default function Race({ slug }: { slug: string }) {
  const { data, error } = useJson<RaceDoc>(`/data/races/${slug}.json`)
  const [tab, setTab] = useState<Tab>('replay')
  useEffect(() => setTab('replay'), [slug])
  if (error) return <div className="wrap"><Failed error={error} /></div>
  if (!data) return <div className="wrap"><Loading what="race" /></div>
  const tabs: [Tab, string][] = [['replay', 'Race replay']]
  if (data.odds) tabs.push(['odds', 'Finishing odds'])
  if (data.pit?.length) tabs.push(['pit', 'Pit window'])
  if (data.duels?.length) tabs.push(['duels', 'Pit duels'])
  return (
    <div className="wrap py-10">
      <a href="#/" className="text-sm text-mute hover:text-ink">← All races</a>
      <div className="mt-4 flex flex-wrap items-end justify-between gap-6">
        <div>
          <Label>{data.date} · {data.circuit}</Label>
          <h1 className="mt-2 font-display text-4xl font-bold md:text-5xl">{data.name} {data.date.slice(0, 4)}</h1>
        </div>
        <div className="flex flex-wrap gap-6">
          <Stat value={data.laps} label="laps" />
          <Stat value={`${data.sc} / ${data.vsc}`} label="safety cars / VSCs" />
          {data.track && <Stat value={`${fx(data.track.pit_loss, 1)} s`} label="pit stop cost" />}
          {data.track && <Stat value={pct(data.track.pass_prob)} label="close battles that pass" />}
        </div>
      </div>
      {data.wet && (
        <p className="mt-4 rounded-lg border border-wet/40 bg-wet/10 px-4 py-3 text-sm text-ink">
          A wet race: BoxBox doesn’t model rain yet, so only the replay is shown.
        </p>
      )}
      <div className="mt-8 flex gap-1 overflow-x-auto border-b border-line" role="tablist">
        {tabs.map(([k, label]) => (
          <button
            key={k}
            role="tab"
            aria-selected={tab === k}
            onClick={() => setTab(k)}
            className={`-mb-px whitespace-nowrap border-b-2 px-4 py-2.5 text-sm font-semibold transition ${tab === k ? 'border-amber text-ink' : 'border-transparent text-mute hover:text-ink'}`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="mt-6">
        {tab === 'replay' && <Replay race={data} />}
        {tab === 'odds' && data.odds && <OddsView race={data} />}
        {tab === 'pit' && data.pit && <PitWindow race={data} />}
        {tab === 'duels' && data.duels && <Duels race={data} />}
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- Replay */

function neutralAt(race: RaceDoc, lap: number) {
  return race.replay.neutral.find((n) => lap >= n.start && lap <= n.end)
}

function Replay({ race }: { race: RaceDoc }) {
  const [lap, setLap] = useState(1)
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => setLap((l) => (l >= race.laps ? (setPlaying(false), l) : l + 1)), 450)
    return () => window.clearInterval(id)
  }, [playing, race.laps])
  const rows = race.replay.laps[String(lap)] ?? []
  const n = neutralAt(race, lap)
  const top = useMemo(() => (race.replay.laps[String(race.laps)] ?? []).slice(0, 10).map((r) => r[0]), [race])
  const gapData = useMemo(() => {
    const out = []
    for (let l = 1; l <= race.laps; l++) {
      const row: Record<string, number> = { lap: l }
      for (const r of race.replay.laps[String(l)] ?? []) if (r[1] != null && r[1] < 90) row[r[0]] = r[1]
      out.push(row)
    }
    return out
  }, [race])
  const colour = (c: string) => race.replay.colours[c] ?? '#888'
  return (
    <div className="grid gap-6 lg:grid-cols-12">
      <Panel className="p-4 lg:col-span-4">
        <div className="flex items-center gap-3">
          <button
            onClick={() => (lap >= race.laps ? (setLap(1), setPlaying(true)) : setPlaying((p) => !p))}
            className="rounded-md bg-amber px-3 py-1.5 text-sm font-semibold text-bg"
            aria-label={playing ? 'Pause replay' : 'Play replay'}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <label htmlFor="lap-slider" className="num text-sm">Lap {lap}/{race.laps}</label>
          {n && <span className="rounded bg-sc/15 px-2 py-0.5 text-xs font-bold text-sc">{n.kind}</span>}
        </div>
        <input id="lap-slider" type="range" min={1} max={race.laps} value={lap} onChange={(e) => { setPlaying(false); setLap(+e.target.value) }} className="mt-3 w-full" />
        <table className="mt-3 w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-dim">
              <th className="py-1 font-normal">Pos</th><th className="font-normal">Driver</th><th className="text-right font-normal">Gap</th><th className="pl-3 font-normal">Tyre</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r[0]} className="border-t border-line/60">
                <td className="num py-1 text-mute">{i + 1}</td>
                <td>
                  <span className="mr-2 inline-block h-3 w-1 rounded-sm align-middle" style={{ background: colour(r[0]) }} />
                  <span className="font-semibold">{r[0]}</span>
                </td>
                <td className="num text-right text-mute">{i === 0 ? 'Leader' : r[1] == null ? '' : r[1] > 90 ? 'lapped' : `+${r[1].toFixed(1)}`}</td>
                <td className="pl-3">{r[2] ? <Tyre compound={TYRE_BY_LETTER[r[2]] ?? r[2]} age={r[3]} size={20} /> : null}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
      <div className="flex flex-col gap-6 lg:col-span-8">
        <Panel className="p-4">
          <div className="mb-2 text-sm font-semibold">Gap to the leader, top 10 at the flag (seconds)</div>
          <div className="h-72">
            <ResponsiveContainer>
              <LineChart data={gapData} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="var(--color-line)" strokeDasharray="2 4" vertical={false} />
                {race.replay.neutral.map((nn, i) => (
                  <ReferenceArea key={i} x1={nn.start} x2={nn.end} fill="var(--color-sc)" fillOpacity={0.12} />
                ))}
                <XAxis dataKey="lap" {...axis} interval="preserveStartEnd" minTickGap={28} />
                <YAxis reversed {...axis} width={36} domain={[0, 'auto']} />
                <Tooltip {...tooltipStyle} formatter={(v) => `+${Number(v).toFixed(1)} s`} labelFormatter={(l) => `Lap ${l}`} />
                <ReferenceLine x={lap} stroke="var(--color-amber)" strokeDasharray="3 3" />
                {top.map((c) => <Line key={c} dataKey={c} stroke={colour(c)} dot={false} strokeWidth={2} isAnimationActive={false} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-mute">
            {top.map((c) => <span key={c}><i className="mr-1 inline-block h-0.5 w-3 align-middle" style={{ background: colour(c) }} />{c}</span>)}
            <span><i className="mr-1 inline-block h-2.5 w-3 align-middle bg-sc/30" />safety car / VSC</span>
          </div>
        </Panel>
        <Panel className="p-4">
          <div className="mb-3 text-sm font-semibold">Tyre strategy (finishing order)</div>
          <Stints race={race} />
        </Panel>
      </div>
    </div>
  )
}

function Stints({ race }: { race: RaceDoc }) {
  const order = race.replay.result.map((r) => r.code)
  const W = 100 / race.laps
  return (
    <div className="flex flex-col gap-1">
      {order.map((code) => (
        <div key={code} className="flex items-center gap-2">
          <span className="num w-10 shrink-0 text-xs text-mute">{code}</span>
          <div className="relative h-4 flex-1 rounded-sm bg-panel-2">
            {race.replay.neutral.map((n, i) => (
              <div key={i} className="absolute inset-y-0 bg-sc/15" style={{ left: `${(n.start - 1) * W}%`, width: `${(n.end - n.start + 1) * W}%` }} />
            ))}
            {race.replay.stints.filter((s) => s.code === code).map((s, i) => {
              const t = TYRE[s.compound]
              return (
                <div
                  key={i}
                  title={`${t?.name ?? s.compound}: laps ${s.start}–${s.end}`}
                  className="absolute inset-y-0.5 grid place-items-center rounded-sm text-[9px] font-bold"
                  style={{ left: `${(s.start - 1) * W}%`, width: `calc(${(s.end - s.start + 1) * W}% - 2px)`, background: t?.colour ?? '#666', color: t?.text ?? '#fff' }}
                >
                  {s.end - s.start >= 4 ? t?.letter : ''}
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

/* ---------------------------------------------------------------- Odds */

function OddsView({ race }: { race: RaceDoc }) {
  const laps = Object.keys(race.odds!).map(Number).sort((a, b) => a - b)
  const [lap, setLap] = useState(laps[1] ?? laps[0])
  const rows: Odds[] = race.odds![String(lap)]
  const actual = Object.fromEntries(race.replay.result.map((r) => [r.code, r.dnf ? 'DNF' : r.pos]))
  const Bar = ({ v, colour }: { v: number; colour: string }) => (
    <div className="flex items-center gap-2">
      <div className="h-2 w-20 rounded-full bg-panel-2"><div className="h-2 rounded-full" style={{ width: `${v * 100}%`, background: colour }} /></div>
      <span className="num w-10 text-right text-xs">{pct(v)}</span>
    </div>
  )
  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-mute">Simulate the rest of the race from the end of lap</span>
        {laps.map((l) => (
          <button key={l} onClick={() => setLap(l)} className={`num rounded-md border px-3 py-1 text-sm ${l === lap ? 'border-amber text-amber' : 'border-line text-mute hover:text-ink'}`}>{l}</button>
        ))}
      </div>
      <p className="mt-2 text-sm text-mute">1,000 simulated finishes from the real order, gaps and tyres at that lap, with each team’s real later pit stops, random safety cars, battles and retirements.</p>
      <Panel className="mt-4 scroll-x">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="text-left text-xs text-dim">
              <th className="px-4 py-2 font-normal">Driver</th><th className="font-normal">Win</th><th className="font-normal">Podium</th><th className="font-normal">Points</th>
              <th className="font-normal">Likely finish</th><th className="pr-4 text-right font-normal">Real result</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.code} className="border-t border-line/60">
                <td className="px-4 py-1.5"><span className="mr-2 inline-block h-3 w-1 rounded-sm align-middle" style={{ background: race.replay.colours[r.code] }} /><b>{r.code}</b></td>
                <td><Bar v={r.win} colour="var(--color-amber)" /></td>
                <td><Bar v={r.podium} colour="var(--color-ink)" /></td>
                <td><Bar v={r.points} colour="var(--color-mute)" /></td>
                <td className="num text-xs text-mute">P{r.p10}–P{r.p90}</td>
                <td className="num pr-4 text-right">{actual[r.code] === 'DNF' ? 'DNF' : actual[r.code] ? `P${actual[r.code]}` : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </>
  )
}

/* ---------------------------------------------------------------- Pit window */

function PitWindow({ race }: { race: RaceDoc }) {
  const pits = race.pit!
  const [code, setCode] = useState(pits[0].code)
  const p: Pit = pits.find((x) => x.code === code) ?? pits[0]
  const compounds = [...new Set(p.curve.map((c) => c.compound))]
  const [comp, setComp] = useState(compounds[0])
  useEffect(() => setComp(compounds.includes(p.real[0]?.compound) ? p.real[0].compound : compounds[0]), [code]) // eslint-disable-line
  const laps = p.curve.filter((c) => c.compound === comp).map((c) => c.lap).sort((a, b) => a - b)
  const [lap, setLap] = useState<number>(laps[0] ?? 0)
  useEffect(() => {
    const real = p.real[0]?.lap
    setLap(laps.includes(real) ? real : laps[Math.floor(laps.length / 2)] ?? 0)
  }, [code, comp]) // eslint-disable-line
  const point = p.curve.find((c) => c.compound === comp && c.lap === lap)
  const chart = useMemo(() => {
    const byLap: Record<number, Record<string, number>> = {}
    for (const c of p.curve) (byLap[c.lap] ??= { lap: c.lap })[c.compound] = c.exp
    return Object.values(byLap).sort((a, b) => a.lap - b.lap)
  }, [p])
  const start = race.model?.start ?? 0
  const real = p.real[0]
  const ruleLaps = p.rule.map((r) => r.lap)
  return (
    <div className="grid gap-6 lg:grid-cols-12">
      <div className="flex flex-col gap-4 lg:col-span-4">
        <Panel className="p-4">
          <Label>Decision point: end of lap {start}</Label>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {pits.map((x) => (
              <button key={x.code} onClick={() => setCode(x.code)} className={`num rounded-md border px-2.5 py-1 text-sm ${x.code === code ? 'border-amber text-amber' : 'border-line text-mute hover:text-ink'}`}>
                P{x.pos} {x.code}
              </button>
            ))}
          </div>
          <div className="mt-4 flex items-center gap-3 text-sm text-mute">
            On <Tyre compound={p.compound} age={p.age} /> now. Team really did:{' '}
            <span className="text-ink">{p.real.map((r) => `L${r.lap} ${TYRE[r.compound]?.letter ?? r.compound}`).join(', ') || 'no stop'}</span>
          </div>
        </Panel>
        <Panel className="p-4">
          <Label>BoxBox’s call</Label>
          <div className="mt-2 font-display text-2xl font-bold text-amber">{p.best.plan ? `Pit ${p.best.plan}` : 'Stay out'}</div>
          <div className="mt-1 text-sm text-mute">
            Expected finish <span className="num text-ink">P{fx(p.best.exp)}</span> · podium <span className="num text-ink">{pct(p.best.podium)}</span>
          </div>
          {ruleLaps.length > 0 ? (
            <div className="mt-3 text-sm">
              <span className="text-sc font-semibold">Safety-car rule:</span> if a SC or VSC comes out between laps{' '}
              <span className="num">{Math.min(...ruleLaps)}–{Math.max(...ruleLaps)}</span>, box at once. Worth{' '}
              <span className="num text-ink">{p.option_value >= 0 ? '+' : ''}{fx(p.option_value, 2)}</span> places on average (option value).
            </div>
          ) : (
            <div className="mt-3 text-sm text-mute">No safety-car rule beats sticking to the plan here.</div>
          )}
          {p.two.length > 0 && (
            <div className="mt-3 text-xs text-mute">Best two-stop: {p.two[0].plan} (P{fx(p.two[0].exp)})</div>
          )}
        </Panel>
        <Panel className="p-4">
          <Label>Try it yourself</Label>
          <div className="mt-3 flex gap-1.5">
            {compounds.map((c) => (
              <button key={c} onClick={() => setComp(c)} className={`rounded-md border px-2 py-1 ${c === comp ? 'border-amber' : 'border-line'}`} aria-pressed={c === comp}>
                <Tyre compound={c} />
              </button>
            ))}
          </div>
          <label htmlFor="pit-lap" className="mt-4 block text-sm">Pit at the end of lap <b className="num text-amber">{lap}</b></label>
          <input id="pit-lap" type="range" min={laps[0]} max={laps[laps.length - 1]} value={lap} onChange={(e) => setLap(+e.target.value)} className="mt-2 w-full" />
          {point && (
            <div className="mt-4 grid grid-cols-3 gap-3">
              <Stat value={`P${fx(point.exp)}`} label="expected" />
              <Stat value={pct(point.podium)} label="podium" />
              <Stat value={pct(point.points)} label="points" />
            </div>
          )}
        </Panel>
      </div>
      <Panel className="p-4 lg:col-span-8">
        <div className="mb-1 text-sm font-semibold">Expected finishing position by pit lap (one stop, lower is better)</div>
        <p className="mb-3 text-xs text-mute">Each point is 400 fresh simulated races. Rivals keep their real strategies and may cover an undercut.</p>
        <div className="h-80">
          <ResponsiveContainer>
            <LineChart data={chart} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--color-line)" strokeDasharray="2 4" vertical={false} />
              {ruleLaps.length > 0 && <ReferenceArea x1={Math.min(...ruleLaps)} x2={Math.max(...ruleLaps)} fill="var(--color-sc)" fillOpacity={0.1} />}
              <XAxis dataKey="lap" {...axis} interval="preserveStartEnd" minTickGap={28} />
              <YAxis reversed {...axis} width={36} domain={['auto', 'auto']} tickFormatter={(v) => `P${Number(v).toFixed(1)}`} />
              <Tooltip {...tooltipStyle} formatter={(v, name) => [`P${Number(v).toFixed(2)}`, TYRE[String(name)]?.name ?? String(name)]} labelFormatter={(l) => `Pit lap ${l}`} />
              {real && <ReferenceLine x={real.lap} stroke="var(--color-mute)" strokeDasharray="4 4" label={{ value: 'team', fill: 'var(--color-mute)', fontSize: 11, position: 'top' }} />}
              <ReferenceLine x={lap} stroke="var(--color-amber)" />
              {compounds.map((c) => (
                <Line key={c} dataKey={c} stroke={TYRE[c]?.colour ?? '#999'} strokeWidth={c === comp ? 3 : 1.5} dot={false} connectNulls isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex flex-wrap gap-4 text-xs text-mute">
          {compounds.map((c) => <span key={c} className="flex items-center gap-1.5"><Tyre compound={c} size={16} />{TYRE[c]?.name}</span>)}
          <span>┆ team’s real lap</span>
          {ruleLaps.length > 0 && <span><i className="mr-1 inline-block h-2.5 w-3 align-middle bg-sc/25" />box-if-safety-car window</span>}
        </div>
      </Panel>
    </div>
  )
}

/* ---------------------------------------------------------------- Duels */

function heat(v: number) {
  // 0 = defender keeps the place (blue), 1 = attacker gets ahead (amber); grey around 50%.
  const t = Math.max(0, Math.min(1, v))
  const a = t < 0.5 ? [59, 140, 255] : [255, 176, 32]
  const k = Math.abs(t - 0.5) * 2
  const g = [38, 44, 52]
  const c = g.map((x, i) => Math.round(x + (a[i] - x) * k))
  return `rgb(${c.join(',')})`
}

function Duels({ race }: { race: RaceDoc }) {
  const duels = race.duels!
  const [i, setI] = useState(0)
  const d: Duel = duels[i]
  const bars = (laps: number[], p: number[], real: number) => (
    <div className="flex flex-col gap-1">
      {laps.map((l, k) => (
        <div key={l} className="flex items-center gap-2 text-xs">
          <span className={`num w-8 ${l === real ? 'text-amber' : 'text-mute'}`}>L{l}</span>
          <div className="h-2 flex-1 rounded-full bg-panel-2"><div className="h-2 rounded-full bg-ink" style={{ width: `${p[k] * 100}%` }} /></div>
          <span className="num w-9 text-right">{p[k] >= 0.005 ? pct(p[k]) : ''}</span>
        </div>
      ))}
    </div>
  )
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap gap-1.5">
        {duels.map((x, k) => (
          <button key={k} onClick={() => setI(k)} className={`num rounded-md border px-2.5 py-1 text-sm ${k === i ? 'border-amber text-amber' : 'border-line text-mute hover:text-ink'}`}>
            {x.us} vs {x.them}
          </button>
        ))}
      </div>
      <p className="max-w-[75ch] text-sm text-mute">
        {d.us} is {fx(d.gap, 1)} s behind {d.them}, and both still have one stop to make. Each team picks a pit lap; each cell is the chance{' '}
        {d.us} finishes ahead. Solving the game (minimax, by linear programming) gives each team’s optimal strategy, which is often a mix
        of laps: being predictable can be exploited.
      </p>
      <div className="grid gap-6 lg:grid-cols-12">
        <Panel className="scroll-x p-4 lg:col-span-8">
          <div className="mb-2 text-sm font-semibold">Chance {d.us} finishes ahead of {d.them}</div>
          <table className="text-xs">
            <thead>
              <tr>
                <th className="pr-2 text-right font-normal text-dim">{d.us} ↓ · {d.them} →</th>
                {d.laps_them.map((l) => <th key={l} className={`num px-0.5 pb-1 font-normal ${l === d.real[1] ? 'text-amber' : 'text-mute'}`}>{l}</th>)}
              </tr>
            </thead>
            <tbody>
              {d.laps_us.map((lu, r) => (
                <tr key={lu}>
                  <th className={`num pr-2 text-right font-normal ${lu === d.real[0] ? 'text-amber' : 'text-mute'}`}>L{lu}</th>
                  {d.laps_them.map((lt, c) => {
                    const v = d.matrix[r][c]
                    const isReal = lu === d.real[0] && lt === d.real[1]
                    return (
                      <td key={lt} className="p-0.5">
                        <div
                          className="num grid h-8 w-10 place-items-center rounded text-[10px]"
                          style={{ background: heat(v), outline: isReal ? '2px solid var(--color-ink)' : 'none', color: Math.abs(v - 0.5) > 0.3 ? '#0b0d10' : 'var(--color-ink)' }}
                          title={`${d.us} lap ${lu}, ${d.them} lap ${lt}: ${pct(v, 1)}`}
                        >
                          {Math.round(v * 100)}
                        </div>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3 flex flex-wrap gap-4 text-xs text-mute">
            <span><i className="mr-1 inline-block h-2.5 w-3 rounded-sm align-middle" style={{ background: heat(1) }} />{d.us} gets ahead</span>
            <span><i className="mr-1 inline-block h-2.5 w-3 rounded-sm align-middle" style={{ background: heat(0) }} />{d.them} stays ahead</span>
            <span>▢ outlined: what the teams really did (amber laps)</span>
          </div>
        </Panel>
        <div className="flex flex-col gap-4 lg:col-span-4">
          <Panel className="p-4">
            <Label>Nash equilibrium</Label>
            <div className="mt-2 num text-3xl text-amber">{pct(d.value, 1)}</div>
            <div className="text-sm text-mute">chance {d.us} finishes ahead with both teams playing optimally</div>
            <div className="mt-3 text-sm">As really played: <span className="num">{pct(d.matrix[d.laps_us.indexOf(d.real[0])]?.[d.laps_them.indexOf(d.real[1])], 1)}</span></div>
            <div className="mt-1 text-sm text-mute">{d.saddle ? `Pure: ${d.us} lap ${d.saddle[0]}, ${d.them} lap ${d.saddle[1]}.` : 'Mixed: the best strategy randomises between laps.'}</div>
          </Panel>
          <Panel className="p-4">
            <div className="mb-2 text-sm font-semibold">{d.us}’s optimal mix</div>
            {bars(d.laps_us, d.x, d.real[0])}
            <div className="mb-2 mt-4 text-sm font-semibold">{d.them}’s optimal mix</div>
            {bars(d.laps_them, d.y, d.real[1])}
          </Panel>
        </div>
      </div>
    </div>
  )
}
