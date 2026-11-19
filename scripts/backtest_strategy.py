"""BoxBox's pit calls vs the real teams' calls, 2024.

For every dry race, stop the clock at 1/3 distance. For each top-10 driver who
in reality made exactly one more stop, compare on the same 800 fresh
simulated races (common random numbers):

  - TEAM:    the lap and tyre the team actually chose (stops made under a real
             safety car are left out: the simulator would replay that lap green)
  - FIXED:   BoxBox's best fixed plan (chosen on separate training races)
  - RULE:    BoxBox's plan plus its safety-car rule

Lower expected finishing position is better. Also reported: how close
BoxBox's pit lap is to the team's, and whether it picked the same tyre.

Rivals respond in every arm: the car just ahead may cover a green-flag stop
(rates measured from 2023), so an undercut is no longer free.

Caveat, stated plainly: the comparison is judged inside BoxBox's own
simulator, so it measures "better under our model of the race", not proof
that a team was wrong. Teams also see things we can't (tyre temperatures,
damage, radio). Agreement with the teams is itself a sanity check.

Usage:  python scripts/backtest_strategy.py 2024 [--n 800] [--no-response]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxbox import degradation, hazards, openf1, season, timing  # noqa: E402
from boxbox.race import load_race  # noqa: E402
from boxbox.simulate import Car, simulate  # noqa: E402
from boxbox.strategy import optimal_pit  # noqa: E402

SEED = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--frac", type=float, default=1 / 3)
    ap.add_argument("--no-response", action="store_true", help="rivals never cover (the earlier, simpler model)")
    a = ap.parse_args()
    prior = season.load_priors(a.year - 1)
    params = hazards.load(str(a.year - 1))
    rows = []
    for s in openf1.season_races(a.year):
        race = load_race(a.year, s["circuit_short_name"])
        if race.wet:
            continue
        start = int(race.total_laps * a.frac)
        model = degradation.fit(race.clean[race.clean["lap"] <= start], prior=prior)
        track = hazards.for_track(params, race.circuit)
        codes = dict(zip(race.drivers["driver"], race.drivers["code"]))
        st = timing.state_at(race, start)
        cars = [
            Car(int(r.driver), codes.get(int(r.driver), str(r.driver)), float(r.gap), r.compound, int(r.tyre_age),
                timing.strategy_after(race, int(r.driver), start))
            for r in st.itertuples() if isinstance(r.compound, str)
        ]
        neutral = {l for nn in race.neutralisations for l in range(nn.start_lap, nn.end_lap + 1)}
        for i, me in enumerate(cars[:10]):
            if len(me.stops) != 1 or me.stops[0].compound not in ("SOFT", "MEDIUM", "HARD"):
                continue
            if me.stops[0].lap in neutral:
                continue  # the team stopped under a real safety car: the simulator would replay that lap green
            used = set(race.laps[(race.laps["driver"] == me.driver) & (race.laps["lap"] <= start)]["compound"].dropna())
            try:
                adv = optimal_pit(cars, i, start, race.total_laps, model, track, used_compounds=used,
                                  n_train=a.n, n_test=a.n, seed=SEED, rivals_respond=not a.no_response)
            except ValueError as e:
                print(f"  skip {race.circuit} {me.code}: {e}")
                continue
            # The team's real call, judged in the same fresh races with the same rival responses.
            team = simulate(cars, start, race.total_laps, model, track, n=a.n, seed=SEED + 1000,
                            respond_to=None if a.no_response else i).finish[:, i]
            b = adv.best_fixed
            rows.append(dict(
                circuit=race.circuit, code=me.code, pos_now=i + 1,
                team_lap=me.stops[0].lap, team_tyre=me.stops[0].compound,
                bb_lap=b["lap"], bb_tyre=b["compound"],
                team_pos=float(team.mean()), fixed_pos=b["exp_pos"], rule_pos=adv.adaptive["exp_pos"],
                rule_used=adv.adaptive["boxed_under_sc"],
            ))
            r = rows[-1]
            print(f"  {race.circuit:<18} {me.code} P{i+1}: team L{r['team_lap']} {r['team_tyre'][:1]} -> P{r['team_pos']:.2f} | "
                  f"BoxBox L{r['bb_lap']} {str(r['bb_tyre'])[:1]} -> P{r['fixed_pos']:.2f}, with SC rule P{r['rule_pos']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    suffix = "_no_response" if a.no_response else ""
    df.to_csv(out / f"strategy_backtest_{a.year}{suffix}.csv", index=False)

    stopped = df[df["bb_lap"].notna()]
    gain_fixed = df["team_pos"] - df["fixed_pos"]
    gain_rule = df["team_pos"] - df["rule_pos"]
    print(f"\n{len(df)} pit decisions across {df['circuit'].nunique()} races (from 1/3 distance, {a.n} fresh races each).")
    print(f"Expected finish: team {df['team_pos'].mean():.3f} | BoxBox fixed {df['fixed_pos'].mean():.3f} | BoxBox + SC rule {df['rule_pos'].mean():.3f}")
    print(f"Positions gained vs team: fixed {gain_fixed.mean():+.3f}, with SC rule {gain_rule.mean():+.3f} per decision")
    print(f"BoxBox at least as good as the team in {(gain_rule >= -0.01).mean():.0%} of decisions; clearly better (>0.1 place) in {(gain_rule > 0.1).mean():.0%}")
    print(f"Option value of the SC rule: {(df['fixed_pos'] - df['rule_pos']).mean():+.3f} positions on average")
    if len(stopped):
        d = (stopped["bb_lap"] - stopped["team_lap"]).abs()
        print(f"Pit lap vs team: median {d.median():.0f} laps apart, within 3 laps in {(d <= 3).mean():.0%} of decisions")
        print(f"Same tyre as the team: {(stopped['bb_tyre'] == stopped['team_tyre']).mean():.0%}")
    print(f"BoxBox said 'no more stops' in {df['bb_lap'].isna().mean():.0%} of decisions (team stopped in all of them)")


if __name__ == "__main__":
    main()
