# BoxBox 🏎️

**F1 pit strategy priced like a financial option.**

Deciding when to pit is an *optimal stopping* problem: the same maths banks use to price American options. Tyres lose pace every lap (time decay), safety cars arrive at random (jumps), and a pit stop has a fixed cost. BoxBox learns each track's tyre wear from real timing data, simulates the rest of the race thousands of times, and finds the best lap to pit.

> Research project by Krishna Gupta. Not affiliated with Formula 1, the FIA or any team. Data: [OpenF1](https://openf1.org) (free, no login).

## Status

| Phase | What | State |
|---|---|---|
| 1 | Data pipeline: laps, tyres, pit stops, safety cars for every race 2023+ | ✅ |
| 1 | Lap-time model: car pace + fuel + compound + tyre wear, robust fit, empirical-Bayes priors | ✅ |
| 1 | Pit loss per track (green flag vs safety car) | ✅ |
| 1 | Walk-forward validation vs baselines | ✅ |
| 2 | Monte Carlo race simulator (traffic, safety cars as a Poisson process) | ⏳ |
| 3 | Optimal pit engine: Least-Squares Monte Carlo (Longstaff–Schwartz) | ⏳ |
| 3 | Undercut / overcut as a two-player game | ⏳ |
| 4 | Season backtest: positions gained vs the teams' real calls | ⏳ |
| 5 | Website: race replay, pit windows, what-if sliders | ⏳ |

## The lap-time model

```
lap_time = driver_base + fuel_per_lap · lap + compound_offset[c] + wear[c] · tyre_age + noise
```

- **Fitted on clean laps only:** no lap 1, in-laps, out-laps, safety-car laps, wet laps, or laps slower than 107% of race pace.
- **Robust:** a second pass drops laps far from the fit (traffic, mistakes) using the median absolute deviation.
- **Shrinkage:** a wear rate measured from only a few laps is pulled toward the previous season's typical value for that compound, weighted by how precise each is (`degradation.shrink`). This stops impossible results such as tyres getting faster with age.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m boxbox fit 2024 Bahrain          # tyre wear + pit loss for one race
python -m boxbox season 2023               # fit every race, save priors to data/season_2023.json
python scripts/validate.py 2024            # walk-forward test on every dry 2024 race
pytest                                     # offline tests on synthetic races
```

## Layout

```
src/boxbox/
  openf1.py        cached OpenF1 client (each race downloads once)
  race.py          one tidy row per driver per lap; SC / VSC / red-flag ranges
  degradation.py   lap-time + tyre-wear model, shrinkage
  pitloss.py       time lost per pit stop at a track
  season.py        fit a whole season, learn priors
  __main__.py      command line
scripts/validate.py  walk-forward test
tests/               synthetic-race tests
```

## Why the finance comparison holds

| Finance | F1 |
|---|---|
| Option loses value as time passes | Tyre loses pace every lap |
| Exercise early or keep waiting? | Pit now or one more lap? |
| Market jump | Safety car |
| Exercise cost | ~20 s lost in the pit lane |
| Price = expected payoff over random paths | Best strategy = best expected finish over simulated races |
