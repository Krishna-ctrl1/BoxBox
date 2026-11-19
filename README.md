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
| 2 | Monte Carlo race simulator: safety cars/VSCs (Gamma-Poisson), overtaking (Beta-Binomial), DNFs, tyre-wear uncertainty | ✅ |
| 2 | Simulator backtest: 1.57 places error vs 1.89, favourite wins 91% vs 79%, well calibrated | ✅ |
| 3 | Optimal pit engine: Least-Squares Monte Carlo (Longstaff–Schwartz), plan + safety-car rule, option value | ✅ |
| 3 | Rivals respond: the car ahead covers an undercut at the rate measured from 2023 (+22%) | ✅ |
| 3 | Backtest vs real 2024 team calls, rivals covering: +0.47 places per decision in our simulator; same tyre 89% | ✅ |
| 3 | Full two-player game: both teams optimise against each other (Nash equilibrium); today rivals react at historical rates | ⏳ |
| 4 | Season backtest: positions gained vs the teams' real calls (115 decisions, 18 races; see phase 3 results) | ✅ |
| 5 | Website: race replay, pit windows, what-if sliders | ⏳ |
| 6 | Research paper write-up | ⏳ |

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

## The race simulator (phase 2)

From the real state of a race at any lap (running order, gaps, tyres), it plays out the rest of the race thousands of times at once (NumPy arrays of simulations × cars). Each simulated race draws its own:

| Random part | How it's modelled | Learned from |
|---|---|---|
| Lap-time noise | Normal, with the model's residual spread | the lap-time model |
| Tyre wear | Drawn per race from the wear rate's uncertainty (parameter risk) | the lap-time model |
| Safety cars / VSCs | Poisson process per lap; per-track rate shrunk to the season rate (Gamma–Poisson) | previous season |
| Safety-car length and pace | Sampled from observed safety-car lengths; ~1.6× lap time | previous season |
| Overtaking | A faster car within 1 s gets past with the track's pass chance, else is held behind (Beta–Binomial per track: Monaco 2.6%, Bahrain 12%) | previous season |
| Retirements | Per-car, per-lap hazard | previous season |
| Pit stops | Track pit loss; under SC/VSC 55% of it (stated assumption, to be calibrated) | this race / assumption |

### Backtest (2024, leak-free)

Stop each dry 2024 race at 1/3, 1/2 and 2/3 distance, start the simulator from the real state with each team's real strategy, simulate 1,000 times, and compare with the real result. Track parameters come from 2023 only.

| 57 race situations | BoxBox | "Order stays as it is" |
|---|---|---|
| Finishing-position error | **1.57 places** | 1.89 |
| Rank correlation with result | **0.879** | 0.848 |
| Favourite actually won | **91%** | 79% |
| Brier score, "who wins" | **0.153** | 0.421 |

Calibration: when it says ~50% chance of a podium it happened 47% of the time; ~70% → 68%; ~89% → 92%. Full output: [`results/sim_backtest_2024.txt`](results/sim_backtest_2024.txt).

```bash
python -m boxbox simulate 2024 Bahrain --from-lap 20     # win / podium / points chances for every driver
python scripts/backtest_sim.py 2024                       # the full backtest (about 5 seconds)
```

## The pit engine (phase 3)

Pitting is an **American option**: on any lap the team may "exercise" (box) or wait, knowing only what it can see right now. BoxBox solves it the way banks price American options, with **Least-Squares Monte Carlo (Longstaff–Schwartz)**:

1. List every allowed stop: lap × tyre, respecting the two-compound rule and tyre life (95th-percentile stint length, per track where history exists).
2. Simulate 1,500 training races. Replay **the same races** (common random numbers) with every possible stop, so each race has a known result for every choice.
3. Pick the best fixed plan on the training races.
4. **Backward induction** from the planned lap: on each lap, among races where a safety car or VSC is out, compare "box now" with "wait and follow the rule" (whose value comes from later laps). Box only if the gain is 1.5 standard errors clear of zero.
5. Score the plan and the rule on **1,500 fresh races**, so they are not flattered by the luck they were fitted on.

**Option value** = expected positions gained by the reacting rule over the best fixed plan.

**A result worth knowing:** the textbook LSM, regressing on every visible feature on every lap, did *worse* than a fixed plan. On green-flag laps, pitting one lap earlier or later changes the result by ~0.02 places, below the regression noise, so the rule flipped at random. A safety car changes the cost of a stop by ~10 s: a signal strong enough to learn. So the decision state is "neutralised this lap or not".

### Rivals respond: covering the undercut

If a car pits first on fresh tyres, it can jump the car ahead (the **undercut**), unless that car **covers** by pitting the next lap. Measured from every 2023 dry race: when a car pits under green, the car directly ahead (within 3 s) pits within 2 laps **31%** of the time, against **11%** on an ordinary lap, an extra **22%** caused by the reaction; the car directly behind reacts ~11% extra. In the simulator, when our driver pits under green, those rivals bring a stop they had planned within the next 12 laps forward to the next lap with those probabilities, separately in every simulated race. In a hand-made undercut, the attacker wins 100% if nobody reacts, 90% at the real cover rate, 45% if the rival always covers. This is a leader–follower response model (rivals behave as real teams did), not a full Nash equilibrium.

### Backtest vs the real teams (2024)

At 1/3 distance in every dry 2024 race, each top-10 driver who really made one more stop under green: the team's real call vs BoxBox, on the same 800 fresh simulated races, **with rivals covering in every arm**. Team stops made under a real safety car are left out (the simulator would replay those laps as green and unfairly punish the team).

| 115 decisions, 18 races | Rivals never react | **Rivals cover** |
|---|---|---|
| Team's real call | 6.79 | **6.78** |
| BoxBox fixed plan | 6.44 | **6.44** |
| BoxBox plan + safety-car rule | 6.30 | **6.31** |
| BoxBox gain per decision | +0.49 | **+0.47** |

- With covering rivals BoxBox still gains **+0.47 places per decision** (+0.34 from the plan, +0.13 from the safety-car rule) and is at least as good as the team in 94% of decisions. The gain barely moved, so it was **not** coming from free undercuts; covering hurt the teams' calls as much as BoxBox's.
- The advice did adapt: BoxBox's pit lap changed in 59% of decisions (median one lap) once rivals could cover.
- Picks the **same tyre as the team 89%** of the time; median pit lap 4 laps from the team's.

**Read this carefully.** "Better" means *better inside BoxBox's simulator*. Teams see things we cannot (tyre temperatures, damage, fuel, radio), and rivals here respond with an average historical rate rather than optimising against us. Agreement on tyre choice and timing is the sanity check. Full output: [`results/strategy_backtest_2024.txt`](results/strategy_backtest_2024.txt) (rivals covering) and [`results/strategy_backtest_2024_no_response.txt`](results/strategy_backtest_2024_no_response.txt).

```bash
python -m boxbox pit 2024 "Yas Marina" VER --from-lap 12   # best plan, pit window, safety-car rule, option value
python scripts/backtest_strategy.py 2024                   # the full comparison with the teams (~10 min)
```

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
  timing.py        running order, gaps and real strategies at any lap
  hazards.py       safety-car, overtaking and DNF rates per track
  simulate.py      vectorised Monte Carlo race simulator
  strategy.py      optimal pit engine (Least-Squares Monte Carlo)
  __main__.py      command line
scripts/validate.py      walk-forward test of the lap-time model
scripts/backtest_sim.py  backtest of the race simulator
scripts/backtest_strategy.py  BoxBox pit calls vs the real teams
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
