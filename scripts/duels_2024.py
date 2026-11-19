"""Every close pit duel of a season, solved as a two-player game.

At 1/3 distance in every dry race, take each pair of adjacent top-10 cars
within 3 s of each other where both, in reality, made exactly one more stop
under green. The car behind is "us" (the attacker), the car ahead "them".
Each team chooses its pit lap within 5 laps of the lap it really used.

For each duel: the payoff matrix (chance the attacker finishes ahead), the
Nash equilibrium and its value, the cell the teams actually played, and each
side's best reply to the other's real choice.

Usage:  python scripts/duels_2024.py 2024 [--n 400] [--window 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxbox import degradation, game, hazards, openf1, season, timing  # noqa: E402
from boxbox.race import load_race  # noqa: E402
from boxbox.simulate import Car  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--window", type=int, default=5)
    a = ap.parse_args()
    prior, params = season.load_priors(a.year - 1), hazards.load(str(a.year - 1))
    rows = []
    for s in openf1.season_races(a.year):
        race = load_race(a.year, s["circuit_short_name"])
        if race.wet:
            continue
        start = int(race.total_laps / 3)
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
        for t in range(min(10, len(cars)) - 1):
            u = t + 1  # attacker = car behind
            if cars[u].gap - cars[t].gap > 3.0:
                continue
            if len(cars[u].stops) != 1 or len(cars[t].stops) != 1:
                continue
            if cars[u].stops[0].lap in neutral or cars[t].stops[0].lap in neutral:
                continue

            def window(c):
                lap = c.stops[0].lap
                return [l for l in range(lap - a.window, lap + a.window + 1) if start < l <= race.total_laps - 3]

            d = game.pit_duel(cars, u, t, start, race.total_laps, model, track, window(cars[u]), window(cars[t]),
                              cars[u].stops[0].compound, cars[t].stops[0].compound, n=a.n)
            ru, rt = cars[u].stops[0].lap, cars[t].stops[0].lap
            real = d.at(ru, rt)
            bru, pu = d.best_reply(rt)
            col = d.matrix[d.laps_us.index(ru), :]  # defender's best reply to the attacker's real lap
            brt_i = int(col.argmin())
            # Equilibrium mass within 1 lap of what each team really did.
            near_u = float(sum(p for l, p in zip(d.laps_us, d.x) if abs(l - ru) <= 1))
            near_t = float(sum(p for l, p in zip(d.laps_them, d.y) if abs(l - rt) <= 1))
            rows.append(dict(
                circuit=race.circuit, attacker=cars[u].code, defender=cars[t].code, gap=cars[u].gap - cars[t].gap,
                real_att=ru, real_def=rt, value=d.value, real=real,
                att_best_reply=bru, att_reply_gain=pu - real,
                def_best_reply=d.laps_them[brt_i], def_reply_gain=real - col[brt_i],
                pure=d.saddle is not None, near_att=near_u, near_def=near_t,
                eq_att=";".join(f"{l}:{p:.2f}" for l, p in d.strategy("us")),
                eq_def=";".join(f"{l}:{p:.2f}" for l, p in d.strategy("them")),
            ))
            r = rows[-1]
            print(f"  {race.circuit:<18} {r['attacker']} vs {r['defender']} ({r['gap']:.1f}s): real L{ru}/L{rt} -> {real:.1%} | "
                  f"Nash value {d.value:.1%} | attacker best reply L{bru} +{r['att_reply_gain']:.1%}, "
                  f"defender best reply L{d.laps_them[brt_i]} +{r['def_reply_gain']:.1%}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results"
    df.to_csv(out / f"duels_{a.year}.csv", index=False)
    print(f"\n{len(df)} close duels (within 3 s, top 10, one stop each) across {df['circuit'].nunique()} races, {a.n} races per cell.")
    print(f"Attacker finishes ahead: as really played {df['real'].mean():.1%} | Nash equilibrium {df['value'].mean():.1%}")
    print(f"Gap between real play and equilibrium: mean |real - value| {np.abs(df['real'] - df['value']).mean():.1%}")
    print(f"Best reply to the other team's real lap gains: attacker +{df['att_reply_gain'].mean():.1%}, defender +{df['def_reply_gain'].mean():.1%} on average")
    print(f"Equilibrium is pure (a single best lap for both) in {df['pure'].mean():.0%} of duels; mixed otherwise")
    print(f"Equilibrium weight within 1 lap of the real choice: attacker {df['near_att'].mean():.0%}, defender {df['near_def'].mean():.0%}")


if __name__ == "__main__":
    main()
