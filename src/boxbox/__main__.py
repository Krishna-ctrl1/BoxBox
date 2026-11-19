"""Command line:  python -m boxbox fit 2024 Bahrain  |  python -m boxbox season 2023"""

from __future__ import annotations

import argparse

from . import degradation, pitloss, season
from .race import load_race


def main() -> None:
    ap = argparse.ArgumentParser(prog="boxbox")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="tyre wear and pit loss for one race")
    f.add_argument("year", type=int)
    f.add_argument("race", help="country, circuit or city, e.g. Bahrain, Monza, Suzuka")
    f.add_argument("--quadratic", action="store_true", help="let wear speed up with age")
    f.add_argument("--no-prior", action="store_true", help="do not shrink toward last season's priors")
    s = sub.add_parser("season", help="fit every race in a season and save priors")
    s.add_argument("year", type=int)
    m = sub.add_parser("simulate", help="simulate the rest of a race from a given lap")
    m.add_argument("year", type=int)
    m.add_argument("race")
    m.add_argument("--from-lap", type=int, required=True, help="start from the end of this lap")
    m.add_argument("--n", type=int, default=2000, help="number of simulated races")
    a = ap.parse_args()

    if a.cmd == "fit":
        race = load_race(a.year, a.race)
        print(race)
        prior = None
        if not a.no_prior:
            try:
                prior = season.load_priors(a.year - 1)
            except FileNotFoundError:
                pass
        m = degradation.fit(race.clean, quadratic=a.quadratic, prior=prior)
        codes = dict(zip(race.drivers["driver"], race.drivers["code"])) if not race.drivers.empty else None
        print(m.summary(codes))
        print(pitloss.estimate(race, m).summary())
    elif a.cmd == "simulate":
        from . import hazards, timing
        from .simulate import Car, simulate

        race = load_race(a.year, a.race)
        try:
            prior = season.load_priors(a.year - 1)
        except FileNotFoundError:
            prior = None
        m = degradation.fit(race.clean[race.clean["lap"] <= a.from_lap], prior=prior)
        try:
            params = hazards.load(str(a.year - 1))
        except FileNotFoundError:
            params = hazards.load("2023_2024")
        codes = dict(zip(race.drivers["driver"], race.drivers["code"]))
        st = timing.state_at(race, a.from_lap)
        cars = [
            Car(int(r.driver), codes.get(int(r.driver), str(r.driver)), float(r.gap), r.compound, int(r.tyre_age),
                timing.strategy_after(race, int(r.driver), a.from_lap))
            for r in st.itertuples() if isinstance(r.compound, str)
        ]
        res = simulate(cars, a.from_lap, race.total_laps, m, hazards.for_track(params, race.circuit), n=a.n)
        print(f"{race}\nFrom the end of lap {a.from_lap}, {a.n} simulated finishes (teams' real strategies):")
        t = res.table()
        actual = timing.finishing_order(race).set_index("driver")["position"]
        t["actual"] = t["driver"].map(actual)
        print(t.drop(columns=["driver"]).round(3).to_string(index=False))
        print(f"Safety cars per simulated race: {res.sc_count.mean():.2f}")
    elif a.cmd == "season":
        t = season.fit_season(a.year)
        path = season.save(a.year, t)
        print(f"Saved {len(t)} races and priors to {path}")
        for c, (mean, sd) in season.priors(t).items():
            print(f"  {c:<6} typical wear {mean:+.4f} s/lap, varies by ±{sd:.4f} across tracks")


if __name__ == "__main__":
    main()
