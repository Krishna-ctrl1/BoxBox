"""Walk-forward test of the lap-time model.

For every dry race in a season: fit the model on laps up to the halfway point
only, then predict every clean lap after it. Nothing from the second half is
seen while fitting, and the wear priors come from the PREVIOUS season, so the
test never peeks at the future.

Compared against two simple baselines:
  - "recent pace": the driver's average of their last 5 clean laps before halfway
  - "stint pace":  the driver's average clean lap so far

Usage:  python scripts/validate.py 2024 [--cut 0.5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxbox import degradation, openf1, season  # noqa: E402
from boxbox.race import load_race  # noqa: E402


def evaluate(year: int, cut: float) -> pd.DataFrame:
    try:
        prior = season.load_priors(year - 1)
        print(f"Using {year - 1} wear priors: " + ", ".join(f"{c} {m:.3f}±{s:.3f}" for c, (m, s) in prior.items()))
    except FileNotFoundError:
        prior = None
        print(f"No {year - 1} priors found; fitting without them.")

    rows = []
    for s in openf1.season_races(year):
        race = load_race(year, s["circuit_short_name"])
        if race.wet:
            continue
        cutoff = int(race.total_laps * cut)
        clean = race.clean
        train, test = clean[clean["lap"] <= cutoff], clean[clean["lap"] > cutoff]
        try:
            model = degradation.fit(train, prior=prior)
        except ValueError:
            continue
        test = test[test["driver"].isin(model.driver_base) & test["compound"].isin(model.deg)]
        if len(test) < 50:
            continue
        pred = model.predict(test["driver"], test["lap"], test["compound"], test["tyre_age"])
        recent = train.sort_values("lap").groupby("driver")["lap_time"].apply(lambda x: x.tail(5).mean())
        overall = train.groupby("driver")["lap_time"].mean()
        y = test["lap_time"].to_numpy()
        rows.append(
            dict(
                circuit=race.circuit,
                laps_tested=len(test),
                model_mae=np.mean(np.abs(y - pred)),
                recent_mae=np.mean(np.abs(y - test["driver"].map(recent).to_numpy())),
                stint_mae=np.mean(np.abs(y - test["driver"].map(overall).to_numpy())),
                bias=np.mean(pred - y),
            )
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--cut", type=float, default=0.5, help="fraction of the race used for fitting")
    a = ap.parse_args()
    t = evaluate(a.year, a.cut)
    pd.set_option("display.width", 160)
    print(t.round(3).to_string(index=False))
    w = t["laps_tested"]
    avg = lambda col: np.average(t[col], weights=w)  # noqa: E731
    print(
        f"\n{len(t)} dry races, {int(w.sum())} future laps predicted.\n"
        f"Average error per lap: model {avg('model_mae'):.3f}s | recent pace {avg('recent_mae'):.3f}s | "
        f"stint pace {avg('stint_mae'):.3f}s\n"
        f"Model beats recent-pace baseline in {(t['model_mae'] < t['recent_mae']).sum()}/{len(t)} races."
    )
