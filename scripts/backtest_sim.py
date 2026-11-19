"""Does the race simulator predict how races really finish?

For every dry race of a season, stop the clock at 1/3, 1/2 and 2/3 distance.
From the real running order, gaps and tyres at that moment, and the pit stops
each team actually made afterwards, simulate the rest of the race 1,000 times.
Then compare with the real result.

No future information leaks into the model: the lap-time model only sees laps
before the start point, and track parameters (safety cars, overtaking) come
from the PREVIOUS season. The only "future" input is each team's real pit
strategy, because this tests the simulator, not the strategy choice.

Baseline: "the order stays as it is now".

Usage:  python scripts/backtest_sim.py 2024 [--n 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxbox import degradation, hazards, openf1, season, timing  # noqa: E402
from boxbox.race import load_race  # noqa: E402
from boxbox.simulate import Car, simulate  # noqa: E402


def run_point(race, start, prior, params, n):
    clean = race.clean[race.clean["lap"] <= start]
    model = degradation.fit(clean, prior=prior)
    state = timing.state_at(race, start)
    state = state[state["laps_down"] <= 1]  # cars far behind add nothing to a top-order test
    codes = dict(zip(race.drivers["driver"], race.drivers["code"]))
    cars = [
        Car(int(r.driver), codes.get(int(r.driver), str(r.driver)), float(r.gap), r.compound, int(r.tyre_age),
            timing.strategy_after(race, int(r.driver), start))
        for r in state.itertuples()
        if isinstance(r.compound, str)
    ]
    res = simulate(cars, start, race.total_laps, model, hazards.for_track(params, race.circuit), n=n, seed=start)
    table = res.table()

    fin = timing.finishing_order(race).set_index("driver")
    k = len(cars)
    actual_pos = {}
    for c in cars:
        p = fin["position"].get(c.driver, np.nan)
        actual_pos[c.driver] = p if pd.notna(p) and not fin["dnf"].get(c.driver, False) else 99
    # Rank actual finishing among the cars we simulated.
    ranked = sorted(actual_pos, key=lambda d: actual_pos[d])
    actual_rank = {d: i + 1 for i, d in enumerate(ranked)}
    now_rank = {c.driver: i + 1 for i, c in enumerate(cars)}

    t = table.set_index("driver")
    model_rank = {d: i + 1 for i, d in enumerate(t.sort_values("exp_pos").index)}
    drivers = [c.driver for c in cars]
    a = np.array([actual_rank[d] for d in drivers])
    m = np.array([model_rank[d] for d in drivers])
    b = np.array([now_rank[d] for d in drivers])
    winner = ranked[0]
    podium = set(ranked[:3])
    p_pod = np.array([t.loc[d, "p_podium"] for d in drivers])
    y_pod = np.array([d in podium for d in drivers], dtype=float)
    p_win = np.array([t.loc[d, "p_win"] for d in drivers])
    y_win = np.array([d == winner for d in drivers], dtype=float)
    return dict(
        circuit=race.circuit,
        start=start,
        laps_left=race.total_laps - start,
        cars=k,
        model_mae=np.abs(m - a).mean(),
        stay_mae=np.abs(b - a).mean(),
        model_rho=spearmanr(m, a).statistic,
        stay_rho=spearmanr(b, a).statistic,
        p_actual_winner=float(t.loc[winner, "p_win"]),
        model_top_pick_won=bool(model_rank[winner] == 1),
        leader_won=bool(now_rank[winner] == 1),
        brier_win=float(((p_win - y_win) ** 2).sum()),
        brier_win_stay=float(((np.array([b_ == 1 for b_ in b], dtype=float) - y_win) ** 2).sum()),
        _pod=list(zip(p_pod.round(4).tolist(), y_pod.tolist())),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    prior = season.load_priors(a.year - 1)
    params = hazards.load(str(a.year - 1))
    rows = []
    for s in openf1.season_races(a.year):
        race = load_race(a.year, s["circuit_short_name"])
        if race.wet:
            continue
        for frac in (1 / 3, 1 / 2, 2 / 3):
            start = int(race.total_laps * frac)
            try:
                rows.append(run_point(race, start, prior, params, a.n))
            except Exception as e:  # noqa: BLE001
                print(f"  skip {race.circuit} L{start}: {e}")
    df = pd.DataFrame(rows)
    pod = [p for xs in df.pop("_pod") for p in xs]

    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print(f"\n{len(df)} race situations ({df['circuit'].nunique()} races x 3 start points), {a.n} sims each.")
    print(f"Position error (places): model {df['model_mae'].mean():.2f} | 'order stays' {df['stay_mae'].mean():.2f}")
    print(f"Rank correlation:        model {df['model_rho'].mean():.3f} | 'order stays' {df['stay_rho'].mean():.3f}")
    print(f"Winner: model's favourite won {df['model_top_pick_won'].mean():.0%} | leader at that point won {df['leader_won'].mean():.0%}")
    print(f"Brier score for 'who wins' (lower is better): model {df['brier_win'].mean():.3f} | 'leader wins' {df['brier_win_stay'].mean():.3f}")
    for frac_name, g in df.groupby(pd.cut(df["laps_left"] / (df["laps_left"] + df["start"]), [0, 0.4, 0.6, 1.0], labels=["2/3 done", "1/2 done", "1/3 done"])):
        if len(g):
            print(f"  {frac_name}: model err {g['model_mae'].mean():.2f} vs stay {g['stay_mae'].mean():.2f} places")

    # Calibration: when the model says "x% chance of a podium", how often did it happen?
    p, y = np.array(pod).T
    bins = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001]
    print("\nPodium calibration (predicted -> happened):")
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"  {lo:>4.0%}-{min(hi,1):<4.0%}  predicted {p[m].mean():5.1%}  happened {y[m].mean():5.1%}  ({m.sum()} cases)")
            cal.append(dict(lo=lo, hi=min(hi, 1), predicted=float(p[m].mean()), happened=float(y[m].mean()), n=int(m.sum())))

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"sim_backtest_{a.year}.csv", index=False)
    (out / f"sim_calibration_{a.year}.json").write_text(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
