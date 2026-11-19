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
    elif a.cmd == "season":
        t = season.fit_season(a.year)
        path = season.save(a.year, t)
        print(f"Saved {len(t)} races and priors to {path}")
        for c, (mean, sd) in season.priors(t).items():
            print(f"  {c:<6} typical wear {mean:+.4f} s/lap, varies by ±{sd:.4f} across tracks")


if __name__ == "__main__":
    main()
