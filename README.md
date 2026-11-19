# BoxBox 🏎️

**F1 pit strategy priced like a financial option.**

Deciding when to pit is an *optimal stopping* problem: the same maths banks use to price American options. Tyres lose pace every lap (time decay), safety cars arrive at random (jumps), and a pit stop has a fixed cost. BoxBox learns each track's tyre wear from real timing data, simulates the rest of the race thousands of times, and finds the best lap to pit.

> Research project by Krishna Gupta. Not affiliated with Formula 1, the FIA or any team. Data: [OpenF1](https://openf1.org) (free, no login).

## Status

| Phase | What | State |
|---|---|---|
| 1 | Data pipeline: laps, tyres, pit stops, safety cars for every race 2023+ | ✅ |
| 1 | Lap-time model: car pace + fuel + compound + tyre wear, robust Bayesian fit with season priors | ✅ |
| 1 | Pit loss per track (green flag vs safety car) | ✅ |
| 1 | Walk-forward validation vs baselines: 0.66 s/lap error vs 1.14 s baseline | ✅ |
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
- **Bayesian priors inside the regression:** each compound's wear rate gets a prior learned from the previous season (typical value and how much it varies by track), added as pseudo-observations so the least-squares answer is the posterior mean. Thin data (few laps on a compound) leans on the prior; rich data barely moves. Weak common-sense priors on fuel effect and compound pace keep races like Monaco 2024 (one stint per driver, so driver pace and tyre pace can't be separated) solvable.

### Validation (walk-forward, 2024)

Fit on the **first half** of each dry 2024 race only, with priors from **2023**, then predict every clean lap of the second half. Nothing from the future is seen.

| | Average error per lap |
|---|---|
| **BoxBox model** | **0.66 s** |
| Driver's recent pace (last 5 laps) | 1.14 s |
| Driver's average pace so far | 1.46 s |

8,564 future laps across 18 dry races; the model beats the best baseline in 17/18 races. Full table: [`results/validation_2024.txt`](results/validation_2024.txt).

Known gaps (next steps): the model leans slightly slow in the second half of races (drivers push harder late; track grip grows non-linearly), safety-car pit-loss estimates are rough, and team-specific tyre wear is not modelled yet.

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
