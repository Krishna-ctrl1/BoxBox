"""Pre-compute everything the website shows, as small JSON files.

For every dry race of a season:
  - replay: running order, gap to the leader and tyre for every driver on every lap,
    plus safety-car / VSC / red-flag periods and the official result
  - odds: the simulator's win / podium / points chances at 1/3, 1/2 and 2/3 distance
  - pit: for each top-10 driver at 1/3 distance, the full pit-window curve (every lap
    and tyre), BoxBox's best plan (one or two stops), the safety-car rule, and what
    the team really did
  - duels: close battles solved as a two-player game (payoff matrix + Nash strategies)

Plus the season-level results (validation, simulator backtest, team comparison).

Usage:  python scripts/export_site.py 2024 [--n 500] [--out site/public/data]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boxbox import degradation, game, hazards, openf1, season, timing  # noqa: E402
from boxbox.race import load_race  # noqa: E402
from boxbox.simulate import Car, simulate  # noqa: E402
from boxbox.strategy import optimal_pit  # noqa: E402


def r(x, d=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def build_cars(race, lap, codes):
    st = timing.state_at(race, lap)
    return [
        Car(int(x.driver), codes.get(int(x.driver), str(x.driver)), float(x.gap), x.compound, int(x.tyre_age),
            timing.strategy_after(race, int(x.driver), lap))
        for x in st.itertuples() if isinstance(x.compound, str)
    ]


def replay(race, codes, colours):
    C = timing.crossings(race)
    L = race.laps
    out = {}
    for lap in range(1, race.total_laps + 1):
        st = timing.state_at(race, lap)
        out[lap] = [[codes.get(int(x.driver), str(x.driver)), r(x.gap, 2), (x.compound or "")[:1], int(x.tyre_age) if pd.notna(x.tyre_age) else None]
                    for x in st.itertuples()]
    stints = []
    for (d, s), g in L.dropna(subset=["compound"]).groupby(["driver", "stint"]):
        stints.append(dict(code=codes.get(int(d), str(d)), start=int(g["lap"].min()), end=int(g["lap"].max()), compound=g["compound"].iloc[0]))
    res = timing.finishing_order(race)
    return dict(
        laps=out,
        stints=stints,
        neutral=[dict(kind=n.kind, start=n.start_lap, end=n.end_lap) for n in race.neutralisations],
        result=[dict(pos=int(x.position) if pd.notna(x.position) else None, code=codes.get(int(x.driver), str(x.driver)), dnf=bool(x.dnf))
                for x in res.itertuples()],
        colours={codes.get(int(d), str(d)): f"#{c}" for d, c in colours.items() if isinstance(c, str)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=str(ROOT / "site" / "public" / "data"))
    ap.add_argument("--only", default=None, help="one circuit, for a quick test")
    ap.add_argument("--jobs", type=int, default=6, help="races exported in parallel")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "races").mkdir(parents=True, exist_ok=True)
    sessions = [s for s in openf1.season_races(a.year) if not a.only or a.only.lower() in s["circuit_short_name"].lower()]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        index = list(pool.map(export_race, [(a.year, s, a.n, str(out)) for s in sessions]))
    if not a.only:
        (out / "races.json").write_text(json.dumps(dict(year=a.year, races=index), indent=1))
        export_results(out, a.year)


def export_race(job):
    year, s, n_sims, out = job
    out = Path(out)
    prior, params = season.load_priors(year - 1), hazards.load(str(year - 1))
    if True:
        race = load_race(year, s["circuit_short_name"])
        codes = dict(zip(race.drivers["driver"], race.drivers["code"]))
        teams = dict(zip(race.drivers["code"], race.drivers["team"]))
        colours = dict(zip(race.drivers["driver"], race.drivers["colour"]))
        slug = race.circuit.lower().replace(" ", "-")
        entry = dict(slug=slug, name=race.name, circuit=race.circuit, date=s["date_start"][:10], laps=race.total_laps,
                     wet=race.wet, sc=sum(n.kind == "SC" for n in race.neutralisations),
                     vsc=sum(n.kind == "VSC" for n in race.neutralisations))
        doc = dict(**entry, teams=teams, replay=replay(race, codes, colours))
        if not race.wet:
            track = hazards.for_track(params, race.circuit)
            doc["track"] = dict(pit_loss=r(track.pit_loss, 1), sc_rate=r(track.sc_rate, 4), pass_prob=r(track.pass_prob, 3),
                                tyre_life=track.tyre_life)
            # Simulator odds at three points.
            odds = {}
            for frac in (1 / 3, 1 / 2, 2 / 3):
                lap = int(race.total_laps * frac)
                m = degradation.fit(race.clean[race.clean["lap"] <= lap], prior=prior)
                cars = build_cars(race, lap, codes)
                t = simulate(cars, lap, race.total_laps, m, track, n=1000, seed=lap).table()
                odds[lap] = [dict(code=x.code, exp=r(x.exp_pos, 2), win=r(x.p_win), podium=r(x.p_podium),
                                  points=r(x.p_points), dnf=r(x.p_dnf), p10=r(x.pos_p10, 0), p90=r(x.pos_p90, 0))
                             for x in t.itertuples()]
            doc["odds"] = odds
            # Pit windows at 1/3 distance for the top 10.
            start = int(race.total_laps / 3)
            model = degradation.fit(race.clean[race.clean["lap"] <= start], prior=prior)
            cars = build_cars(race, start, codes)
            doc["model"] = dict(deg={c: r(v, 4) for c, v in model.deg.items()}, fuel=r(model.fuel_per_lap, 4),
                                sigma=r(model.sigma, 3), start=start)
            pits = []
            for i, me in enumerate(cars[:10]):
                used = set(race.laps[(race.laps["driver"] == me.driver) & (race.laps["lap"] <= start)]["compound"].dropna())
                try:
                    adv = optimal_pit(cars, i, start, race.total_laps, model, track, used_compounds=used,
                                      n_train=n_sims, n_test=n_sims, full_table=True)
                except ValueError:
                    continue
                f = adv.fixed
                one = f[f["stops"] == 1]
                pits.append(dict(
                    code=me.code, pos=i + 1, compound=me.compound, age=me.tyre_age,
                    real=[dict(lap=x.lap, compound=x.compound) for x in me.stops],
                    curve=[dict(lap=int(x.lap), compound=x.compound, exp=r(x.exp_pos, 3), win=r(x.p_win), podium=r(x.p_podium),
                                points=r(x.p_points)) for x in one.itertuples()],
                    two=[dict(plan=x.plan, exp=r(x.exp_pos, 3), podium=r(x.p_podium)) for x in f[f["stops"] == 2].head(5).itertuples()],
                    best=dict(plan=adv.best_fixed.get("plan"), stops=adv.best_fixed.get("stops"), exp=r(adv.best_fixed["exp_pos"], 3),
                              podium=r(adv.best_fixed["p_podium"]), points=r(adv.best_fixed["p_points"])),
                    rule=[dict(lap=int(x.lap), compound=x.compound) for x in adv.sc_rule.itertuples()],
                    adaptive=dict(exp=r(adv.adaptive["exp_pos"], 3), podium=r(adv.adaptive["p_podium"]), used=r(adv.adaptive["boxed_under_sc"])),
                    option_value=r(adv.option_value, 3),
                    no_stop=(dict(exp=r(adv.no_stop["exp_pos"], 3)) if adv.no_stop else None),
                ))
                print(f"  {race.circuit} {me.code}: {adv.best_fixed.get('plan')}", flush=True)
            doc["pit"] = pits
            # Duels: adjacent top-10 cars within 3 s, one stop each.
            duels = []
            neutral = {l for nn in race.neutralisations for l in range(nn.start_lap, nn.end_lap + 1)}
            for t_ in range(min(10, len(cars)) - 1):
                u = t_ + 1
                if cars[u].gap - cars[t_].gap > 3.0 or len(cars[u].stops) != 1 or len(cars[t_].stops) != 1:
                    continue
                if cars[u].stops[0].lap in neutral or cars[t_].stops[0].lap in neutral:
                    continue
                win = lambda c: [l for l in range(c.stops[0].lap - 5, c.stops[0].lap + 6) if start < l <= race.total_laps - 3]  # noqa: E731
                d = game.pit_duel(cars, u, t_, start, race.total_laps, model, track, win(cars[u]), win(cars[t_]),
                                  cars[u].stops[0].compound, cars[t_].stops[0].compound, n=400)
                duels.append(dict(us=d.us, them=d.them, gap=r(cars[u].gap - cars[t_].gap, 2), laps_us=d.laps_us, laps_them=d.laps_them,
                                  matrix=[[r(v) for v in row] for row in d.matrix], value=r(d.value), x=[r(v) for v in d.x],
                                  y=[r(v) for v in d.y], saddle=d.saddle, real=[cars[u].stops[0].lap, cars[t_].stops[0].lap]))
            doc["duels"] = duels
        (out / "races" / f"{slug}.json").write_text(json.dumps(doc, separators=(",", ":")))
        print(f"{race.circuit}: done", flush=True)
        return entry


def export_results(out: Path, year: int):
    res = ROOT / "results"
    data = {}
    v = res / f"validation_{year}.txt"
    if v.exists():
        rows = []
        for line in v.read_text().splitlines():
            p = line.split()
            if len(p) >= 6 and p[-5].isdigit():
                rows.append(dict(circuit=" ".join(p[:-5]), n=int(p[-5]), model=float(p[-4]), recent=float(p[-3]), stint=float(p[-2])))
        data["validation"] = rows
    cal = res / f"sim_calibration_{year}.json"
    if cal.exists():
        data["calibration"] = json.loads(cal.read_text())
    sb = res / f"sim_backtest_{year}.csv"
    if sb.exists():
        data["sim_backtest"] = json.loads(pd.read_csv(sb).round(3).to_json(orient="records"))
    st = res / f"strategy_backtest_{year}.csv"
    if st.exists():
        data["strategy"] = json.loads(pd.read_csv(st).round(3).to_json(orient="records"))
    du = res / f"duels_{year}.csv"
    if du.exists():
        data["duels"] = json.loads(pd.read_csv(du).round(4).to_json(orient="records"))
    (out / "results.json").write_text(json.dumps(data, separators=(",", ":")))


if __name__ == "__main__":
    main()
