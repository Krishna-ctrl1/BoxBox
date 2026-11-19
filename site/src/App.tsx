import { useHash } from './lib'
import Home from './pages/Home'
import Race from './pages/Race'
import Methods from './pages/Methods'
import Results from './pages/Results'

const GITHUB = 'https://github.com/Krishna-ctrl1/BoxBox'

function Nav({ hash }: { hash: string }) {
  const link = (href: string, label: string) => {
    const on = hash === href || (href !== '#/' && hash.startsWith(href))
    return (
      <a href={href} className={`rounded-md px-3 py-1.5 text-sm transition ${on ? 'bg-panel-2 text-ink' : 'text-mute hover:text-ink'}`}>
        {label}
      </a>
    )
  }
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/90 backdrop-blur" style={{ top: 'env(safe-area-inset-top, 0px)' }}>
      <div className="wrap flex h-14 items-center justify-between gap-4">
        <a href="#/" className="flex items-center gap-2.5">
          <svg width="26" height="26" viewBox="0 0 64 64" aria-hidden="true">
            <circle cx="32" cy="32" r="20" fill="none" stroke="var(--color-amber)" strokeWidth="7" />
            <circle cx="32" cy="32" r="8" fill="var(--color-soft)" />
          </svg>
          <span className="font-display text-xl font-bold tracking-wide">BOXBOX</span>
        </a>
        <nav className="flex items-center gap-1 overflow-x-auto">
          {link('#/', 'Races')}
          {link('#/methods', 'How it works')}
          {link('#/results', 'Results')}
          <a href={GITHUB} target="_blank" rel="noreferrer" className="ml-1 rounded-md px-3 py-1.5 text-sm text-mute hover:text-ink">
            GitHub ↗
          </a>
        </nav>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="mt-24 border-t border-line">
      <div className="wrap flex flex-col gap-3 py-8 text-sm text-mute md:flex-row md:items-center md:justify-between">
        <div>
          BoxBox by <span className="text-ink">Krishna Gupta</span> · data from{' '}
          <a className="underline hover:text-ink" href="https://openf1.org" target="_blank" rel="noreferrer">OpenF1</a> ·{' '}
          <a className="underline hover:text-ink" href={GITHUB} target="_blank" rel="noreferrer">source (MIT)</a>
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
  else page = <Home />
  return (
    <div className="min-h-screen">
      <Nav hash={hash} />
      <main>{page}</main>
      <Footer />
    </div>
  )
}
