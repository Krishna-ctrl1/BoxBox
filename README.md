# BoxBox 🏎️

**F1 pit strategy priced like a financial option.**

Deciding when to pit is an *optimal stopping* problem, the same maths banks use to price **American options**. Tyres lose pace every lap (time decay), safety cars arrive at random (jumps), and a pit stop has a fixed cost (the exercise price). BoxBox learns each track's tyre wear from real timing data, simulates the rest of a race thousands of times, and finds the best lap and tyre to pit on, plus when a safety car should change the plan.

> Research project by Krishna Gupta. Not affiliated with Formula 1, the FIA or any team. Data from [OpenF1](https://openf1.org), free and with no login. Nothing here is for betting.

## Headline results

All tests are walk-forward: models only see the past (earlier laps of the race, and the previous season), never the future.

| Part | Test | BoxBox | Simple baseline |
|---|---|---|---|
| Lap-time model | Predict 8,564 second-half laps of 18 dry 2024 races | **0.66 s** error per lap | 1.14 s (driver's recent pace) |
| Race simulator | Predict finishing order from 57 mid-race situations, 2024 | **1.55** places error; favourite won **93%** | 1.89; leader won 79% |
| Pit engine | 115 real 2024 pit decisions vs the teams' own calls | **+0.47 places** per decision (in-model); same tyre **89%** | the team's real call |

The pit-engine gain is measured inside BoxBox's own simulator; see [Limitations](#limitations).

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
  - [1. Data](#1-data)
  - [2. Lap-time and tyre-wear model](#2-lap-time-and-tyre-wear-model)
  - [3. Race simulator](#3-race-simulator)
  - [4. Pit engine (American option)](#4-pit-engine-american-option)
  - [5. Rivals respond](#5-rivals-respond)
- [Results in detail](#results-in-detail)
- [Measured vs assumed](#measured-vs-assumed)
- [Limitations](#limitations)
- [Reproduce everything](#reproduce-everything)
- [Repository layout](#repository-layout)
- [Methods at a glance](#methods-at-a-glance)
- [Status and roadmap](#status-and-roadmap)
- [References](#references)

## Quick start

Python 3.11 or newer.

```bash
git clone git@github.com:Krishna-ctrl1/BoxBox.git && cd BoxBox
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m boxbox fit 2024 Bahrain                              # tyre wear and pit loss for one race
python -m boxbox simulate 2024 Bahrain --from-lap 20           # win / podium / points chances for every driver
python -m boxbox pit 2024 "Yas Marina" VER --from-lap 12       # best pit lap and tyre, plus a safety-car rule
pytest                                                         # 15 offline tests on hand-made races
```

The first run of a race downloads it from OpenF1 into `data/raw/` (about 10 seconds); after that it works offline. Races are named by country, circuit or city: `Bahrain`, `Monza`, `Suzuka`, `"Yas Marina"`. Where a name matches several races (e.g. three in the United States), use the circuit: `Austin`, `Miami`, `"Las Vegas"`.

Example, Abu Dhabi 2024, Verstappen at the end of lap 12:

```
VER from the end of lap 12 of 58:
  Best fixed plan: pit at the end of lap 29 for hards -> expected finish P6.32 (podium 15%, points 88%)
  Safety-car rule: if a SC or VSC comes out between laps 17 and 28, box at once for hards
  With the rule: expected P6.07 (podium 22%, points 88%); rule triggered in 16% of races
  Value of flexibility (option value): +0.252 positions
```

Red Bull actually pitted on lap 29 for hards, the same lap and tyre.

## How it works

### 1. Data

- **Source:** [OpenF1](https://openf1.org). F1's own timing server (used by the FastF1 library) now requires an F1 account login, so BoxBox uses OpenF1, which is free and needs no login.
- **Coverage:** every race of 2023 (22; Imola was cancelled) and 2024 (24). Each race is downloaded once and cached as JSON in `data/raw/`.
- **Per race:** every lap time and lap start time, tyre stints (compound and age), pit stops, race-control messages (safety car, VSC, red flag), weather and the official result.
- **Clean-up** (`race.py`):
  - Safety-car, VSC and red-flag lap ranges come from race-control messages. A red flag also ends a running safety car.
  - Early-2023 races with no pit records get their stops rebuilt from tyre stints (a new stint means a stop).
  - **Clean laps** = true racing pace. That excludes lap 1, in-laps and out-laps, neutralised laps, wet-tyre laps, and laps slower than 107% of race pace.
- **Running order at any lap** (`timing.py`): crossing time = lap start time + lap duration, sorted. This reproduces the official Bahrain 2024 result exactly, with gaps within 0.01 s.
- **Wet races** are kept in the data but left out of model fitting (e.g. Brazil 2024).

### 2. Lap-time and tyre-wear model

```
lap_time = car_pace[driver] + fuel · lap + compound_pace[c] + wear[c] · tyre_age + noise
```

- **Fixed effects:** one pace per car and driver, with fuel, compound pace and wear shared across the field.
- **Robust fit:** repeated passes drop laps far from the fit (traffic, mistakes), using the median absolute deviation.
- **Bayesian priors inside the regression:** each compound's wear rate has a prior from the previous season (typical value and how much it varies by track), added as pseudo-observations so least squares returns the posterior mean. Thin data leans on the prior; rich data barely moves.
- **Why the priors matter:** in Monaco 2024 every car ran one compound all race after a lap-1 red flag, so the data alone cannot separate car pace from tyre pace. Without priors the predictions were off by 125 s per lap; with them, 0.88 s.
- **Pit loss per track** (`pitloss.py`): real in-lap + out-lap minus what those two laps would have taken without stopping. For example: Bahrain 24.3 s, Hungary 20.5 s.

### 3. Race simulator

`simulate.py` starts from the real state of a race (order, gaps, tyres) and plays out the rest thousands of times at once, as NumPy arrays of *simulated races × cars*. 2,000 races of 20 cars take under a second. Each simulated race has its own random draws:

| Random part | How it's modelled | Learned from |
|---|---|---|
| Lap-time noise | Normal, with the model's residual spread | lap-time model |
| Tyre wear | Drawn per race from the wear rate's uncertainty (parameter risk) | lap-time model |
| Safety cars / VSCs | Poisson process per green lap; each track's rate shrunk to the season rate (Gamma–Poisson) | previous season |
| Safety-car length and pace | Sampled from real safety-car lengths; ~1.6× normal lap time; the field bunches up | previous season |
| Overtaking | A faster car within 1 s gets past with the track's pass chance, else is held behind (Beta–Binomial per track, e.g. Monaco 2.6%, Bahrain 12% on 2023–24 data) | previous season |
| Retirements | Per-car, per-lap hazard (~2.6 per race) | previous season |
| Pit stops | Track pit loss; under SC/VSC 55% of it | this race / assumption |

Random numbers are drawn in the same order whatever the strategies, so two simulations with the same seed share every safety car, lap-time wobble and retirement (**common random numbers**). Differences between strategies are then due to the strategy, not to luck.

### 4. Pit engine (American option)

`strategy.py` treats pitting as an American option. On any lap the team may *exercise* (box) or wait, knowing only what it can see right now. It is solved with **Least-Squares Monte Carlo** (Longstaff & Schwartz, 2001):

1. **List every allowed stop:** lap × tyre, respecting the two-compound rule and tyre life (the 95th-percentile stint length: soft 24, medium 32, hard 47 laps; per track where that track has history).
2. **Simulate 1,500 training races,** then replay the same races with every possible stop, so each race has a known result for every choice.
3. **Pick the best fixed plan** on the training races.
4. **Backward induction** from the planned lap towards now. On each lap, among races where a safety car or VSC is out, compare "box now" with "wait and follow the rule" (whose value comes from laps already processed). Box only if the gain is 1.5 standard errors clear of zero.
5. **Score the plan and the rule on 1,500 fresh races,** so they are not flattered by the luck they were fitted on.

**Option value** = places gained by the reacting rule over the best fixed plan. It is the value of being able to react.

**Why the rule reacts only to safety cars:** the textbook version regressed on every visible feature (position, gaps, tyre age) on every lap, and it did *worse* than a fixed plan. On green-flag laps, pitting one lap earlier or later changes the result by about 0.02 places, which is below the regression noise, so the rule flipped at random. A safety car makes a stop roughly 10 s cheaper, a signal strong enough to learn. The decision state is therefore "neutralised this lap or not".

### 5. Rivals respond

If a car pits first on fresh tyres it can jump the car ahead (the **undercut**), unless that car **covers** by pitting the next lap. This was measured from every 2023 dry race:

| When a car pits under green… | Pits within 2 laps | Normal rate | Reaction |
|---|---|---|---|
| the car directly ahead (within 3 s) | 31% | 11% | **+22%** |
| the car directly behind (within 3 s) | 21% | 11% | **+11%** |

In the simulator, when our driver pits under green, those rivals bring forward a stop they had planned within the next 12 laps, with those probabilities, separately in every simulated race. In a hand-made undercut, the attacker wins 100% if nobody reacts, 90% at the real cover rate, and 45% if the rival always covers. This is a **leader–follower** model: rivals behave as real teams did on average, but they do not optimise against us (see [roadmap](#status-and-roadmap)).

## Results in detail

### Lap-time model (walk-forward, 2024)

Fit on the **first half** of each dry 2024 race with **2023** priors, then predict every clean lap of the second half.

| | Average error per lap |
|---|---|
| **BoxBox model** | **0.66 s** |
| Driver's recent pace (last 5 laps) | 1.14 s |
| Driver's average pace so far | 1.46 s |

8,564 laps, 18 dry races; the model beats the best baseline in 17 of 18 (the exception, Qatar, had only 95 test laps). Full table: [`results/validation_2024.txt`](results/validation_2024.txt).

### Race simulator (2024)

Each dry 2024 race was stopped at 1/3, 1/2 and 2/3 distance. The simulator started from the real state, using each team's real later pit stops, and simulated the rest 1,000 times. Track parameters come from 2023.

| 57 race situations | BoxBox | "Order stays as it is" |
|---|---|---|
| Finishing-position error | **1.55 places** | 1.89 |
| Rank correlation with result | **0.882** | 0.848 |
| Favourite actually won | **93%** | 79% |
| Brier score, "who wins" | **0.148** | 0.421 |

Error shrinks as the race goes on: 2.03 places from 1/3 distance, 1.49 from halfway, 1.12 from 2/3 (vs 2.32, 1.81, 1.53 for "order stays").

**Calibration** (does "X% chance of a podium" happen X% of the time?):

| BoxBox said | It happened | Cases |
|---|---|---|
| 0–5% | 0.4% | 764 |
| 5–20% (avg 11%) | 6.5% | 62 |
| 20–40% (avg 29%) | 27.5% | 40 |
| 40–60% (avg 48%) | 48.3% | 29 |
| 60–80% (avg 70%) | 64.5% | 31 |
| 80–95% (avg 89%) | 92.9% | 113 |
| 95–100% | 100% | 14 | Full output: [`results/sim_backtest_2024.txt`](results/sim_backtest_2024.txt).

### Pit engine vs the real teams (2024)

At 1/3 distance in every dry 2024 race, for each top-10 driver who really made one more stop under green: the team's real call vs BoxBox, on the same 800 fresh simulated races, with rivals covering in every arm. Team stops made under a real safety car are left out, because the simulator would replay that lap as green and unfairly punish the team.

| 115 decisions, 18 races | Rivals never react | **Rivals cover** |
|---|---|---|
| Team's real call | 6.77 | **6.78** |
| BoxBox fixed plan | 6.43 | **6.43** |
| BoxBox plan + safety-car rule | 6.31 | **6.31** |
| BoxBox gain per decision | +0.47 | **+0.47** |

- **Gain:** with covering rivals BoxBox gains **+0.47 places per decision**: +0.35 from the plan and +0.13 from the safety-car rule. It is at least as good as the team in 94% of decisions.
- **Not from free undercuts:** letting rivals cover changes the gain by less than 0.01 places and BoxBox's chosen pit lap in only 10% of decisions. At a one-in-five cover rate, covering costs both sides about equally.
- **Agreement with the teams:** same tyre as the team 89% of the time, and a median pit lap 4 laps from the team's.
- **By track:** the largest gains were at Spa (+1.37), Baku (+1.12) and Melbourne (+0.99). At Monaco the teams did better (−0.35), where track position is everything.

Full output: [`results/strategy_backtest_2024.txt`](results/strategy_backtest_2024.txt) (rivals covering) and [`results/strategy_backtest_2024_no_response.txt`](results/strategy_backtest_2024_no_response.txt).

## Measured vs assumed

Every number the simulator uses is listed here, so it is clear what comes from data and what is a modelling choice.

| Quantity | Value | Source |
|---|---|---|
| Tyre wear, compound pace, car pace, fuel effect | per race | **measured** (lap-time model) |
| Pit loss, green flag | per track (18–30 s) | **measured** |
| Safety car / VSC rate | ~0.5 each per race, per track | **measured** (previous season) |
| Safety-car length; SC lap pace | 1–6 laps; ~1.6× | **measured** |
| VSC lap pace | 1.35× | **assumed**: too few full VSC laps to measure |
| Pass chance in a close battle | 2.6–23% per track (2023–24 data; Monaco lowest) | **measured** |
| Pass chance when the car behind is genuinely faster | 2× the above | **assumed** (`PASS_BOOST`) |
| Retirement rate | ~0.0026 per car per lap | **measured** |
| Tyre life | 95th-percentile stint length | **measured** |
| Cover rates | +22% ahead, +11% behind | **measured** |
| Pit stop under SC/VSC | 55% of a green stop | **assumed**: the direct estimate is biased by cars closing up to the SC queue |
| Following gap when held up; gap in SC queue | 0.35 s; 0.8 s | **assumed** |
| Typical pace and wear for a compound a race hasn't used yet | soft −0.6 s, hard +0.45 s vs medium | **assumed** fallback |
| Safety-car rule confidence | 1.5 standard errors | modelling choice |

## Limitations

- **The pit-engine gain is in-model.** "Better than the team" means better inside BoxBox's simulator, not proof that a team was wrong. Teams see things we cannot: tyre temperatures, damage, fuel, driver feedback, radio.
- **Rivals react at an average historical rate.** They do not optimise against us; a full two-player (Nash) game is future work.
- **One more stop at most** per decision; two-stop plans are not optimised yet.
- **Tyre wear is the same for every car** within a race, and linear in tyre age (the tyre-life limits stand in for the "cliff").
- **Late-race pace bias:** the model leans slightly slow in the second half of races (drivers push harder late; track grip grows non-linearly).
- **Assumptions:** the safety-car pit discount (55%) and the pass boost are assumptions, not yet calibrated.
- **Coverage:** wet races are not modelled. Data starts in 2023 (OpenF1's coverage).

## Reproduce everything

Each step uses only earlier seasons for the season being tested.

```bash
python -m boxbox season 2023          # wear priors          -> data/season_2023.json
python -m boxbox season 2024          #                      -> data/season_2024.json
python -m boxbox hazards 2023         # track parameters     -> data/track_params_2023.json
python -m boxbox hazards 2023 2024    #                      -> data/track_params_2023_2024.json

python scripts/validate.py 2024                # lap-time model test       (~1 min)
python scripts/backtest_sim.py 2024            # race simulator test       (~5 s once data is cached)
python scripts/backtest_strategy.py 2024       # pit engine vs the teams   (~10 min)
python scripts/backtest_strategy.py 2024 --no-response   # same, rivals never cover (for comparison)
pytest                                         # 15 offline tests
```

The first run downloads both seasons from OpenF1 (a few minutes, rate-limited politely). After that everything runs offline from `data/raw/`.

## Repository layout

```
src/boxbox/
  openf1.py        cached OpenF1 client (each race downloads once; retries and rate limiting)
  race.py          one tidy row per driver per lap; SC / VSC / red-flag ranges; clean laps
  timing.py        running order and gaps at any lap; each team's real later stops
  degradation.py   lap-time and tyre-wear model (robust Bayesian regression)
  pitloss.py       time lost per pit stop at a track
  season.py        fit a whole season, learn wear priors
  hazards.py       safety-car, overtaking, retirement and tyre-life rates per track
  simulate.py      vectorised Monte Carlo race simulator, with rivals covering
  strategy.py      optimal pit engine (Least-Squares Monte Carlo)
  __main__.py      command line: fit, season, hazards, simulate, pit
scripts/
  validate.py            walk-forward test of the lap-time model
  backtest_sim.py        backtest of the race simulator
  backtest_strategy.py   BoxBox's pit calls vs the real teams
tests/                   15 tests on hand-made races where the right answer is known
data/                    season priors and track parameters (raw OpenF1 cache in data/raw/, not committed)
results/                 outputs of the three tests
```

## Methods at a glance

| Method | Where it is used |
|---|---|
| Linear regression with fixed effects | car pace vs tyre and fuel effects |
| Robust statistics (median absolute deviation) | dropping traffic and mistake laps |
| Bayesian linear regression (priors as pseudo-observations) | wear rates with thin data |
| Empirical Bayes | priors learned from the previous season |
| Poisson process, Gamma–Poisson | safety cars and VSCs |
| Beta–Binomial | overtaking chance per track |
| Monte Carlo simulation, vectorised | thousands of races at once |
| Parameter uncertainty | tyre wear drawn per simulated race |
| Common random numbers | fair comparison of strategies |
| Optimal stopping, Least-Squares Monte Carlo | when to pit (American option) |
| Train / test split | honest scoring of the pit rule |
| Leader–follower response model | rivals covering undercuts |
| Walk-forward validation, Brier score, calibration | every result above |

## Status and roadmap

| Phase | What | State |
|---|---|---|
| 1 | Data pipeline: laps, tyres, pit stops, safety cars for every race 2023+ | ✅ |
| 1 | Lap-time model with robust Bayesian fit and season priors; pit loss per track | ✅ |
| 1 | Walk-forward validation: 0.66 s/lap vs 1.14 s baseline | ✅ |
| 2 | Monte Carlo race simulator: safety cars, overtaking, retirements, tyre-wear uncertainty | ✅ |
| 2 | Simulator backtest: 1.55 vs 1.89 places error, favourite wins 93% vs 79%, calibrated | ✅ |
| 3 | Optimal pit engine (Longstaff–Schwartz): plan, safety-car rule, option value | ✅ |
| 3 | Rivals respond: covering the undercut at rates measured from 2023 | ✅ |
| 4 | Season backtest vs the real teams' calls: +0.47 places per decision (in-model), same tyre 89% | ✅ |
| 3+ | Full two-player game: both teams optimise against each other (Nash equilibrium) | ⏳ |
| 3+ | Two-stop plans; team-specific tyre wear; calibrate the safety-car pit discount | ⏳ |
| 5 | Website: race replay, pit windows, what-if sliders | ⏳ |
| 6 | Research paper write-up | ⏳ |

## References

- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: a simple least-squares approach. *The Review of Financial Studies*, 14(1), 113–147.
- OpenF1, an open API for Formula 1 timing data: <https://openf1.org>

## Why the finance comparison holds

| Finance | F1 |
|---|---|
| An option loses value as time passes | A tyre loses pace every lap |
| Exercise now or keep waiting? | Pit now or one more lap? |
| Market jump | Safety car |
| Exercise cost | ~20 s lost in the pit lane |
| Price = expected payoff over random paths | Best strategy = best expected finish over simulated races |
