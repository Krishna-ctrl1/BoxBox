"""Race identity for the website: Grand Prix names and real circuit maps.

Writes site/public/assets/races-meta.json, separate from the engine's data files
(nothing in site/public/data/ is touched). For each race of the season:

  - the official Grand Prix name, round number, country and circuit type (OpenF1 meetings)
  - a circuit outline drawn from the race winner's GPS position on a clean green-flag lap
    (OpenF1 location data, ~3.7 samples a second), as an SVG path in a 1000-unit box,
    with the start/finish point and the direction of travel

No track artwork is copied from anywhere: every map is drawn from the cars' own positions.

Usage:  python scripts/build_assets.py 2024
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boxbox import openf1  # noqa: E402
from boxbox.race import load_race  # noqa: E402


def outline(race) -> dict | None:
    """SVG path of one clean lap by the winner (or the next finisher with data)."""
    res = race.results.sort_values("position")
    for d in res["driver_number"].head(5):
        ok = race.laps["clean"]
        if not ok.any():  # wet race: no "clean" dry laps, but any normal racing lap traces the track
            ok = (race.laps["lap"] > 1) & ~race.laps["neutral"] & ~race.laps["is_pit_in_lap"] & ~race.laps["is_pit_out_lap"] & race.laps["lap_time"].notna()
        L = race.laps[(race.laps["driver"] == d) & ok].sort_values("lap_time")
        for row in L.head(3).itertuples():
            t0 = dt.datetime.fromisoformat(row.date_start)
            t1 = t0 + dt.timedelta(seconds=float(row.lap_time))
            pts = openf1.get("location", session_key=race.session_key, driver_number=int(d),
                             **{"date>": t0.isoformat(), "date<": t1.isoformat()})
            xy = np.array([(p["x"], p["y"]) for p in pts if p.get("x") is not None], dtype=float)
            if len(xy) < 150:
                continue
            # Drop duplicate points and the occasional zero glitch.
            keep = np.r_[True, np.any(np.diff(xy, axis=0) != 0, axis=1)] & ~np.all(xy == 0, axis=1)
            xy = xy[keep]
            xy[:, 1] *= -1  # screen y points down
            mn, mx = xy.min(axis=0), xy.max(axis=0)
            scale = 900 / max(mx - mn)
            p = (xy - mn) * scale + 50
            w, h = (mx - mn) * scale + 100
            path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in p) + " Z"
            ahead = p[min(6, len(p) - 1)]
            return dict(path=path, w=round(float(w)), h=round(float(h)),
                        start=[round(float(p[0][0]), 1), round(float(p[0][1]), 1)],
                        toward=[round(float(ahead[0]), 1), round(float(ahead[1]), 1)],
                        points=int(len(p)), driver=int(d), lap=int(row.lap))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    a = ap.parse_args()
    out = ROOT / "site" / "public" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for rnd, s in enumerate(openf1.season_races(a.year), start=1):
        m = openf1.get("meetings", meeting_key=s["meeting_key"])[0]
        race = load_race(a.year, s["circuit_short_name"])
        slug = race.circuit.lower().replace(" ", "-")
        meta[slug] = dict(
            round=rnd,
            gp=m["meeting_name"],
            official=m.get("meeting_official_name"),
            country=m.get("country_name"),
            country_code=m.get("country_code"),
            location=m.get("location"),
            circuit_type=m.get("circuit_type"),
            map=outline(race),
        )
        mp = meta[slug]["map"]
        print(f"R{rnd:02d} {m['meeting_name']:<28} {race.circuit:<20} map {'ok ' + str(mp['points']) + ' pts' if mp else 'MISSING'}", flush=True)
    (out / "races-meta.json").write_text(json.dumps(dict(year=a.year, races=meta), separators=(",", ":")))


if __name__ == "__main__":
    main()
