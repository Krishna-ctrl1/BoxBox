import { useIdentity } from '../identity'

export default function Credits() {
  const { photos } = useIdentity()
  return (
    <div className="wrap py-14">
      <div className="eyebrow">Credits</div>
      <h1 className="display mt-3 text-4xl font-semibold md:text-5xl">Photography and data</h1>
      <p className="mt-5 max-w-[70ch] leading-relaxed text-mute">
        Every photograph on BoxBox is from Wikimedia Commons, used under the licence shown, and was cropped to 16:9 and resized for the
        web. Photos under CC BY-SA are shared under the same licence. Circuit maps are drawn by BoxBox from the race winner’s GPS
        position data. Timing data comes from OpenF1. Formula 1 and team names are trademarks of their owners; BoxBox is not affiliated
        with them.
      </p>
      <div className="mt-10 grid gap-px overflow-hidden border border-line bg-line">
        {(photos ?? []).map((p) => (
          <div key={p.id} className="grid gap-4 bg-bg p-4 sm:grid-cols-[200px_1fr] sm:items-center">
            <img src={p.src1200} alt={p.alt} loading="lazy" width={1200} height={675} className="aspect-video w-full object-cover" />
            <div className="text-sm">
              <div className="text-ink">{p.alt}</div>
              <div className="mt-1 text-mute">
                <a className="underline hover:text-ink" href={p.page} target="_blank" rel="noreferrer">“{p.title}”</a> by {p.author} ·{' '}
                {p.license_url ? <a className="underline hover:text-ink" href={p.license_url} target="_blank" rel="noreferrer">{p.license}</a> : p.license} · {p.changes}
              </div>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-6 text-sm text-mute">
        Timing data: <a className="underline hover:text-ink" href="https://openf1.org" target="_blank" rel="noreferrer">OpenF1</a> ·
        Fonts: Space Grotesk, Inter, IBM Plex Mono (SIL Open Font License)
      </p>
    </div>
  )
}
