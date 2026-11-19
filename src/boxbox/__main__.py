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
    hz = sub.add_parser("hazards", help="learn safety-car, overtaking, DNF and tyre-life rates per track")
    hz.add_argument("years", type=int, nargs="+", help="e.g. 2023, or 2023 2024")
    du = sub.add_parser("duel", help="two drivers' pit timing as a game: Nash equilibrium")
    du.add_argument("year", type=int)
    du.add_argument("race")
    du.add_argument("us", help="three-letter code, e.g. LEC")
    du.add_argument("them", help="three-letter code, e.g. SAI")
    du.add_argument("--from-lap", type=int, required=True)
    du.add_argument("--window", type=int, default=5, help="laps either side of each team's real stop")
    du.add_argument("--n", type=int, default=600)
    pt = sub.add_parser("pit", help="best pit lap and tyre for one driver, plus a safety-car rule")
    pt.add_argument("year", type=int)
    pt.add_argument("race")
    pt.add_argument("driver", help="three-letter code, e.g. VER")
    pt.add_argument("--from-lap", type=int, required=True)
    pt.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()

    if a.cmd == "hazards":
        from . import hazards

        p = hazards.estimate(a.years)
        name = "_".join(str(y) for y in a.years)
        path = hazards.save(p, name)
        d = p["_default"]
        print(f"Saved {len(p) - 1} tracks to {path}")
        print(f"  season-wide: SC {d.sc_rate:.4f}/lap, VSC {d.vsc_rate:.4f}/lap, pass {d.pass_prob:.1%} of close battles, "
              f"DNF {d.dnf_rate:.4f}/car/lap, pit loss {d.pit_loss:.1f} s")
        return

    if a.cmd == "duel":
        from . import game, hazards, timing
        from .simulate import Car

        race = load_race(a.year, a.race)
        try:
            prior, params = season.load_priors(a.year - 1), hazards.load(str(a.year - 1))
        except FileNotFoundError:
            prior, params = None, hazards.load("2023_2024")
        m = degradation.fit(race.clean[race.clean["lap"] <= a.from_lap], prior=prior)
        codes = dict(zip(race.drivers["driver"], race.drivers["code"]))
        st = timing.state_at(race, a.from_lap)
        cars = [
            Car(int(r.driver), codes.get(int(r.driver), str(r.driver)), float(r.gap), r.compound, int(r.tyre_age),
                timing.strategy_after(race, int(r.driver), a.from_lap))
            for r in st.itertuples() if isinstance(r.compound, str)
        ]
        idx = {c.code: i for i, c in enumerate(cars)}
        for code in (a.us, a.them):
            if code.upper() not in idx:
                raise SystemExit(f"{code} is not running at lap {a.from_lap}.")
        u, t = idx[a.us.upper()], idx[a.them.upper()]
        for i in (u, t):
            if not cars[i].stops:
                raise SystemExit(f"{cars[i].code} made no more stops after lap {a.from_lap}; a duel needs one each.")

        def window(c):
            lap = c.stops[0].lap
            return [l for l in range(lap - a.window, lap + a.window + 1) if a.from_lap < l <= race.total_laps - 3]

        d = game.pit_duel(cars, u, t, a.from_lap, race.total_laps, m, hazards.for_track(params, race.circuit),
                          window(cars[u]), window(cars[t]), cars[u].stops[0].compound, cars[t].stops[0].compound, n=a.n)
        real_u, real_t = cars[u].stops[0].lap, cars[t].stops[0].lap
        print(race)
        print(f"Gap at lap {a.from_lap}: {cars[u].code} P{u + 1}, {cars[t].code} P{t + 1}. "
              f"Real stops: {cars[u].code} lap {real_u}, {cars[t].code} lap {real_t}.")
        print(d.summary())
        print(f"  What really happened (laps {real_u} vs {real_t}): {cars[u].code} ahead {d.at(real_u, real_t):.1%} in the simulator")
        br, p = d.best_reply(real_t)
        print(f"  {cars[u].code}'s best reply to {cars[t].code}'s real lap: lap {br} -> {p:.1%}")
        return

    if a.cmd == "pit":
        from . import hazards, timing
        from .simulate import Car
        from .strategy import optimal_pit

        race = load_race(a.year, a.race)
        try:
            prior = season.load_priors(a.year - 1)
            params = hazards.load(str(a.year - 1))
        except FileNotFoundError:
            prior, params = None, hazards.load("2023_2024")
        m = degradation.fit(race.clean[race.clean["lap"] <= a.from_lap], prior=prior)
        codes = dict(zip(race.drivers["driver"], race.drivers["code"]))
        st = timing.state_at(race, a.from_lap)
        cars = [
            Car(int(r.driver), codes.get(int(r.driver), str(r.driver)), float(r.gap), r.compound, int(r.tyre_age),
                timing.strategy_after(race, int(r.driver), a.from_lap))
            for r in st.itertuples() if isinstance(r.compound, str)
        ]
        idx = [i for i, c in enumerate(cars) if c.code == a.driver.upper()]
        if not idx:
            raise SystemExit(f"{a.driver} is not running at lap {a.from_lap}. Running: {', '.join(c.code for c in cars)}")
        i = idx[0]
        used = set(race.laps[(race.laps["driver"] == cars[i].driver) & (race.laps["lap"] <= a.from_lap)]["compound"].dropna())
        adv = optimal_pit(cars, i, a.from_lap, race.total_laps, m, hazards.for_track(params, race.circuit),
                          used_compounds=used, n_train=a.n, n_test=a.n)
        print(race)
        print(f"{cars[i].code} is P{i + 1}, on {cars[i].compound.lower()}s aged {cars[i].tyre_age} laps. "
              f"What the team really did next: {[(s.lap, s.compound) for s in cars[i].stops] or 'no more stops'}")
        print(adv.summary())
        print("\nTop fixed plans (fresh races):")
        print(adv.fixed.head(8)[["stops", "plan", "exp_pos", "p_win", "p_podium", "p_points"]].round(3).to_string(index=False))
        return

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
