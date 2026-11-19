import { useHash } from './lib'
import Home from './pages/Home'
import Race from './pages/Race'
import Methods from './pages/Methods'
import Results from './pages/Results'
import Credits from './pages/Credits'

const GITHUB = 'https://github.com/Krishna-ctrl1/BoxBox'

function Nav({ hash }: { hash: string }) {
  const link = (href: string, label: string) => {
    const on = hash === href || (href !== '#/' && hash.startsWith(href))
    return (
      <a href={href} className={`whitespace-nowrap px-1.5 py-1.5 text-[12px] uppercase tracking-[0.08em] transition sm:px-2.5 sm:text-[13px] sm:tracking-[0.14em] ${on ? 'text-ink' : 'text-mute hover:text-ink'}`}>
        {label}
      </a>
    )
  }
  return (
    <header className="sticky top-0 z-30 border-b border-line/70 bg-bg/80 backdrop-blur-md" style={{ top: 'env(safe-area-inset-top, 0px)' }}>
      <div className="wrap flex h-14 items-center justify-between gap-4">
        <a href="#/" className="flex items-center gap-2.5" aria-label="BoxBox home">
          <span className="block h-4 w-1.5 bg-accent" aria-hidden="true" />
          <span className="display text-base font-bold tracking-[0.12em] sm:text-lg sm:tracking-[0.18em]">BOXBOX</span>
        </a>
        <nav className="flex items-center gap-0.5 overflow-x-auto">
          {link('#/', 'Season')}
          {link('#/methods', 'Method')}
          {link('#/results', 'Results')}
          <a href={GITHUB} target="_blank" rel="noreferrer" className="ml-1 hidden whitespace-nowrap px-2.5 py-1.5 text-[13px] uppercase tracking-[0.14em] text-mute hover:text-ink sm:inline">
            Code ↗
          </a>
        </nav>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="mt-24 border-t border-line">
      <div className="wrap grid gap-6 py-10 text-sm text-mute md:grid-cols-3">
        <div>
          <div className="display text-base font-bold tracking-[0.18em] text-ink">BOXBOX</div>
          <div className="mt-2">Formula 1 strategy intelligence by <span className="text-ink">Krishna Gupta</span>.</div>
        </div>
        <div className="flex flex-col gap-1.5">
          <a className="hover:text-ink" href="https://openf1.org" target="_blank" rel="noreferrer">Timing data: OpenF1 ↗</a>
          <a className="hover:text-ink" href="#/credits">Photography credits</a>
          <a className="hover:text-ink" href={GITHUB} target="_blank" rel="noreferrer">Source code (MIT) ↗</a>
        </div>
        <div className="text-dim">Research project. Not affiliated with Formula 1, the FIA or any team. Not for betting.</div>
      </div>
    </footer>
  )
}

export default function App() {
  const hash = useHash()
  let page
  if (hash.startsWith('#/race/')) page = <Race slug={hash.slice('#/race/'.length).split('/')[0]} />
  else if (hash.startsWith('#/methods')) page = <Methods />
  else if (hash.startsWith('#/results')) page = <Results />
  else if (hash.startsWith('#/credits')) page = <Credits />
  else page = <Home />
  return (
    <div className="min-h-screen">
      <Nav hash={hash} />
      <main>{page}</main>
      <Footer />
    </div>
  )
}
