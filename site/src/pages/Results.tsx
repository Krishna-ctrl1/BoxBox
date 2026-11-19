import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ReferenceLine, ZAxis,
} from 'recharts'
import { Failed, Label, Loading, Panel, Stat, axis, fx, pct, tooltipStyle, useJson } from '../lib'
import type { Results as R } from '../types'

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1)

export default function Results() {
  const { data, error } = useJson<R>('/data/results.json')
  if (error) return <div className="wrap"><Failed error={error} /></div>
  if (!data) return <div className="wrap"><Loading what="results" /></div>
  const v = data.validation ?? []
  const w = v.reduce((a, r) => a + r.n, 0)
  const wavg = (k: 'model' | 'recent') => v.reduce((a, r) => a + r[k] * r.n, 0) / Math.max(w, 1)
  const st = data.strategy ?? []
  const gain = st.map((r) => r.team_pos - r.rule_pos)
  const sameTyre = st.filter((r) => r.bb_lap != null).map((r) => r.bb_tyre === r.team_tyre)
  const du = data.duels ?? []
  return (
    <div className="wrap py-12">
      <Label>Every test is walk-forward: models only ever see the past</Label>
      <h1 className="mt-3 font-display text-4xl font-bold md:text-5xl">Results on the 2024 season</h1>

      <section className="mt-12">
        <h2 className="font-display text-2xl font-bold">1 · Lap times: predicting laps it never saw</h2>
        <p className="mt-2 max-w-[75ch] text-mute">Fitted on the first half of each dry 2024 race, with priors from 2023, then asked to predict every clean lap of the second half.</p>
        <div className="mt-4 grid gap-4 lg:grid-cols-12">
          <Panel className="flex flex-col gap-5 p-5 lg:col-span-3">
            <Stat value={`${fx(wavg('model'))} s`} label="BoxBox error per lap" accent />
            <Stat value={`${fx(wavg('recent'))} s`} label="driver’s recent pace" />
            <Stat value={w.toLocaleString()} label={`laps in ${v.length} races`} />
          </Panel>
          <Panel className="p-4 lg:col-span-9">
            <div className="h-72">
              <ResponsiveContainer>
                <BarChart data={v} margin={{ top: 8, right: 8, bottom: 40, left: 0 }}>
                  <CartesianGrid stroke="var(--color-line)" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="circuit" {...axis} angle={-40} textAnchor="end" interval={0} height={60} />
                  <YAxis {...axis} width={36} unit="s" />
                  <Tooltip {...tooltipStyle} formatter={(x) => `${Number(x).toFixed(3)} s`} />
                  <Legend wrapperStyle={{ fontSize: 12, color: 'var(--color-mute)' }} />
                  <Bar dataKey="model" name="BoxBox" fill="var(--color-accent)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                  <Bar dataKey="recent" name="Recent pace" fill="var(--color-dim)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>
      </section>

      <section className="mt-16">
        <h2 className="font-display text-2xl font-bold">2 · The simulator’s odds are honest</h2>
        <p className="mt-2 max-w-[75ch] text-mute">From 1/3, 1/2 and 2/3 distance in every dry 2024 race. When BoxBox says “X% chance of a podium”, how often did it happen? Points on the diagonal mean the odds can be trusted.</p>
        <Panel className="mt-4 p-4">
          <div className="h-80">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid stroke="var(--color-line)" strokeDasharray="2 4" />
                <XAxis type="number" dataKey="predicted" name="BoxBox said" domain={[0, 1]} tickFormatter={(x) => pct(x)} {...axis} />
                <YAxis type="number" dataKey="happened" name="Happened" domain={[0, 1]} tickFormatter={(x) => pct(x)} {...axis} width={44} />
                <ZAxis type="number" dataKey="n" range={[60, 400]} name="cases" />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--color-dim)" strokeDasharray="4 4" />
                <Tooltip {...tooltipStyle} formatter={(x, name) => (name === 'cases' ? x : pct(Number(x), 1))} />
                <Scatter data={data.calibration ?? []} fill="var(--color-accent)" isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-1 text-xs text-mute">Bubble size = number of cases. Dashed line = perfect calibration.</div>
        </Panel>
      </section>

      <section className="mt-16">
        <h2 className="font-display text-2xl font-bold">3 · BoxBox vs the real teams’ pit calls</h2>
        <p className="mt-2 max-w-[75ch] text-mute">
          At 1/3 distance, each top-10 driver who made one more stop under green: the team’s real lap and tyre against BoxBox’s plan,
          on the same fresh simulated races, with rivals able to cover. Judged inside BoxBox’s simulator, so this measures “better under
          our model”, not proof a team was wrong.
        </p>
        <div className="mt-4 grid gap-4 md:grid-cols-4">
          <Panel className="p-5"><Stat value={st.length} label="real pit decisions" /></Panel>
          <Panel className="p-5"><Stat value={`P${fx(mean(st.map((r) => r.team_pos)))}`} label="teams’ calls, expected finish" /></Panel>
          <Panel className="p-5"><Stat value={`P${fx(mean(st.map((r) => r.rule_pos)))}`} label="BoxBox, expected finish" accent /></Panel>
          <Panel className="p-5"><Stat value={pct(mean(sameTyre.map(Number)))} label="same tyre as the team" /></Panel>
        </div>
        <p className="mt-3 text-sm text-mute">
          Average gain <span className="num text-ink">{gain.length ? `+${fx(mean(gain))}` : '–'}</span> places per decision; BoxBox at least as good in{' '}
          <span className="num text-ink">{pct(mean(gain.map((g) => Number(g >= -0.01))))}</span> of decisions.
        </p>
      </section>

      {du.length > 0 && (
        <section className="mt-16">
          <h2 className="font-display text-2xl font-bold">4 · Pit duels as games</h2>
          <p className="mt-2 max-w-[75ch] text-mute">Every close 2024 battle (within 3 s, both one stop to go) solved for its Nash equilibrium.</p>
          <div className="mt-4 grid gap-4 md:grid-cols-4">
            <Panel className="p-5"><Stat value={du.length} label="duels" /></Panel>
            <Panel className="p-5"><Stat value={pct(mean(du.map((d) => Number(d.real))), 1)} label="attacker ahead, as really played" /></Panel>
            <Panel className="p-5"><Stat value={pct(mean(du.map((d) => Number(d.value))), 1)} label="attacker ahead, both optimal" accent /></Panel>
            <Panel className="p-5"><Stat value={pct(mean(du.map((d) => Number(!d.pure))))} label="duels where the best play is a mix of laps" /></Panel>
          </div>
        </section>
      )}
    </div>
  )
}
