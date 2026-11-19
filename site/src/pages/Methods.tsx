import { Label, Panel } from '../lib'

const steps = [
  {
    title: 'Learn how fast each car is',
    body: 'From two seasons of real timing data (OpenF1), a regression learns every car’s pace, how much faster cars get as fuel burns, how fast each tyre compound is when new and how much time it loses with every lap of wear. Bayesian priors from the previous season stop thin data from producing nonsense (in Monaco 2024 everyone ran one tyre all race, so the data alone cannot separate car pace from tyre pace).',
    stat: '0.66 s',
    statLabel: 'average error predicting the second half of 2024 races (baseline 1.14 s)',
  },
  {
    title: 'Play the rest of the race thousands of times',
    body: 'From the real order, gaps and tyres at any lap, a Monte Carlo simulator plays the remaining laps 2,000 times at once. Each copy draws its own lap-time wobble, tyre-wear rate, safety cars (a Poisson process per track), battles (a faster car only passes with the track’s measured pass chance), pit stops and retirements.',
    stat: '93%',
    statLabel: 'of the time the simulator’s favourite won, from mid-race (the leader at that point: 79%)',
  },
  {
    title: 'Price the pit stop like an American option',
    body: 'Every possible pit lap and tyre is replayed on the same simulated races (common random numbers). The best fixed plan is chosen on one set of races; then, working backwards lap by lap (Least-Squares Monte Carlo, Longstaff–Schwartz 2001), BoxBox learns when a safety car should trigger an early stop. Everything is scored on a fresh set of races.',
    stat: '+0.10',
    statLabel: 'places from reacting to safety cars alone: the option value',
  },
  {
    title: 'Let the rivals fight back',
    body: 'Real teams cover an undercut: when a car pits, the car just ahead pits within two laps 31% of the time against 11% normally. BoxBox’s rivals do the same. For a head-to-head battle, both teams’ choices are solved as a zero-sum game, giving each side’s optimal (often mixed) strategy.',
    stat: '34',
    statLabel: 'close 2024 battles solved for their Nash equilibrium',
  },
]

const analogy = [
  ['An option loses value as time passes', 'A tyre loses pace every lap'],
  ['Exercise now, or keep waiting?', 'Pit now, or one more lap?'],
  ['A market jump', 'A safety car'],
  ['The exercise price', '~20 seconds lost in the pit lane'],
  ['Price = expected payoff over random paths', 'Best strategy = best expected finish over simulated races'],
]

const assumed = [
  ['Measured from data', 'tyre wear and pace, fuel effect, pit loss per track, safety-car and VSC rates, safety-car length and pace, pass chance per track, retirement rate, tyre life, how often rivals cover, the safety-car pit discount (0.58, from 4 clean stops)'],
  ['Assumed', 'VSC lap pace (1.35×), the VSC pit discount (0.75, derived from the safety-car one), the pass boost when the car behind is clearly faster (2×), following gaps (0.35 s held up, 0.8 s behind the safety car)'],
]

export default function Methods() {
  return (
    <div className="wrap py-12">
      <Label>How it works</Label>
      <h1 className="mt-3 max-w-[22ch] font-display text-4xl font-bold leading-tight md:text-5xl">Four steps from timing data to a pit call</h1>
      <div className="mt-10 grid gap-4">
        {steps.map((s, i) => (
          <Panel key={s.title} className="grid gap-6 p-6 md:grid-cols-12">
            <div className="md:col-span-8">
              <div className="num text-sm text-amber">Step {i + 1}</div>
              <h2 className="mt-1 font-display text-2xl font-bold">{s.title}</h2>
              <p className="mt-3 max-w-[70ch] leading-relaxed text-mute">{s.body}</p>
            </div>
            <div className="md:col-span-4 md:border-l md:border-line md:pl-6">
              <div className="num text-4xl text-ink">{s.stat}</div>
              <div className="mt-1 text-sm text-mute">{s.statLabel}</div>
            </div>
          </Panel>
        ))}
      </div>

      <h2 className="mt-16 font-display text-3xl font-bold">Why the finance comparison holds</h2>
      <Panel className="mt-4 scroll-x">
        <table className="w-full min-w-[520px] text-sm">
          <thead><tr className="text-left text-xs text-dim"><th className="px-4 py-2 font-normal">Finance</th><th className="px-4 font-normal">Formula 1</th></tr></thead>
          <tbody>
            {analogy.map(([a, b]) => (
              <tr key={a} className="border-t border-line/60"><td className="px-4 py-2 text-mute">{a}</td><td className="px-4">{b}</td></tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <h2 className="mt-16 font-display text-3xl font-bold">What’s measured and what’s assumed</h2>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        {assumed.map(([k, v]) => (
          <Panel key={k} className="p-5">
            <div className={`font-semibold ${k === 'Assumed' ? 'text-amber' : 'text-good'}`}>{k}</div>
            <p className="mt-2 text-sm leading-relaxed text-mute">{v}</p>
          </Panel>
        ))}
      </div>

      <h2 className="mt-16 font-display text-3xl font-bold">Limits</h2>
      <ul className="mt-4 grid max-w-[80ch] gap-2 text-mute">
        <li>• “Better than the team” is measured inside BoxBox’s own simulator. Teams see tyre temperatures, damage, fuel and driver feedback that public timing data doesn’t show.</li>
        <li>• Tyre wear is linear in age; tyre-life limits from real stints stand in for the “cliff”. Team-specific wear was tested and made predictions slightly worse, so it is off.</li>
        <li>• Rain isn’t modelled, and the data starts in 2023.</li>
      </ul>
      <p className="mt-8 text-sm text-mute">
        Full method, code and every number: <a className="text-ink underline" href="https://github.com/Krishna-ctrl1/BoxBox" target="_blank" rel="noreferrer">github.com/Krishna-ctrl1/BoxBox</a>
      </p>
    </div>
  )
}
