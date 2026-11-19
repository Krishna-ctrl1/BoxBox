# BoxBox: the complete study guide

*What we built, why, how, every technique (what it is in general and how BoxBox uses it), every design decision, the results, the mistakes we caught, and what to say about it.*

Author: Krishna Gupta · Built November 2026
Live site: https://box-box-sand.vercel.app/ · Code: https://github.com/Krishna-ctrl1/BoxBox

---

## How to use this guide

- **Quick refresh (10 minutes):** read Part 1, the results table in Part 12, and the talking points in Part 15.
- **Before an interview or SOP writing:** read Parts 4–8 (the maths) and Part 13 (limits). Interviewers love the limits.
- **To reuse the code:** read Part 10 (engineering) and Part 17 (commands).

Each technique section follows the same pattern:
1. **What it is** (in general, any field)
2. **How BoxBox uses it**
3. **Why we chose it** (and what we rejected)

---

## Part 1 — The big picture

### 1.1 What BoxBox is

BoxBox is an F1 race-strategy engine and website. It:

1. learns how fast every car is on every lap, including how quickly each tyre wears, from real timing data;
2. simulates the rest of a race thousands of times, with random safety cars, battles and retirements;
3. finds the best lap (and tyre) to pit, treating the pit stop exactly like a financial **American option**;
4. models rivals reacting to your stop, and solves head-to-head pit battles as **two-player games** (Nash equilibrium);
5. shows all of it on a public website: replays, odds, a "what if he'd pitted on lap X?" slider, and duel heatmaps.

**One-line pitch:** *"I priced Formula 1 pit stops as American options, using a Bayesian tyre-wear model, a calibrated Monte Carlo race simulator and Least-Squares Monte Carlo, then tested it against every real 2024 team pit decision."*

### 1.2 Why this project (the goal)

- **Target:** master's in Quantitative Finance / Financial Mathematics (UNSW and similar), and Quant ML / financial-AI roles.
- **The gaps it fills** (from the master's roadmap): stochastic processes, Monte Carlo, optimal stopping, numerical methods, Bayesian statistics, game theory, and honest out-of-sample validation.
- **Why F1:** a personal interest that is full of probability, risk and timing decisions. It is memorable in an SOP and in interviews, and it has no overlap with fintech day-job work.
- **Why "option pricing" framing:** it maps exactly. That turns a sports project into a quant-finance project.

| Finance | F1 |
|---|---|
| An option loses value as time passes (time decay) | A tyre loses pace every lap (degradation) |
| Exercise now or keep waiting? | Pit now or one more lap? |
| Market jump | Safety car |
| Exercise price | ~20 s lost in the pit lane |
| Option price = expected payoff over random paths | Best strategy = best expected finish over simulated races |

### 1.3 The build, in order

| Phase | What | Key result |
|---|---|---|
| 1 | Data pipeline + Bayesian lap-time / tyre-wear model | 0.66 s per lap error vs 1.14 s baseline |
| 2 | Monte Carlo race simulator | favourite won 93% vs 79%; probabilities calibrated |
| 3 | Optimal pit engine (Least-Squares Monte Carlo) | +0.47 places per decision vs real teams (in-model) |
| 3+ | Rivals cover undercuts (measured rates) | gain unchanged → it wasn't free undercuts |
| 3+ | Two-stop plans; safety-car pit cost measured; team wear tested | team wear made things worse → switched off |
| 3+ | Nash pit duels | real play 40.3% vs equilibrium 37.9% |
| 5 | Website (React), then a premium automotive redesign | live on Vercel, all 24 races of 2024 |

---

## Part 2 — Data

### 2.1 Where the data comes from, and a decision we had to make

- **First choice: FastF1** (the popular Python library). It failed: F1's own timing server now returns **401 (login required)**.
- **Switched to OpenF1** (https://openf1.org): free, no login, and every session from 2023 on. It has laps, tyre stints, pit stops, race-control messages (safety car, VSC, red flag), weather, results, and even car GPS positions.
- **Design decision:** don't depend on someone's personal F1 account; use an open source that anyone can reproduce.

### 2.2 The client (`openf1.py`)

- **Disk cache:** every API response is saved as JSON in `data/raw/`. A race downloads once, then everything runs offline. That's fast, reproducible, and polite to a free API.
- **Rate limiting:** a minimum gap between requests. On HTTP 429 ("too many requests") it backs off harder each time, and respects the server's `Retry-After` header.
- **Lesson:** the first season download skipped 3 races because retries gave up too early. We increased retries and backoff, and lost no more races.

### 2.3 Turning raw data into a race table (`race.py`)

One row per driver per lap: lap time, compound, stint, tyre age, in-lap/out-lap flags, "neutralised" flag, and a `clean` flag.

- **Tyre age** = the stint's starting age (tyres can be pre-used from qualifying) + laps into the stint.
- **Neutralisations** (safety car, VSC, red flag) come from race-control messages: "DEPLOYED" opens, "IN THIS LAP"/"ENDING" closes.
  - **Bug we found:** in Australia 2023 a red flag ended a safety car without an "in this lap" message, so the SC stayed "open" for 50 laps. Fix: a red flag closes any running SC/VSC.
- **Missing data:** early-2023 races had no pit records in OpenF1. Every stop still shows as a new tyre stint, so we **rebuild stops from stints**.
- **Clean laps** = true racing pace. Excluded:
  - lap 1
  - in-laps and out-laps
  - safety-car, VSC and red-flag laps
  - wet-tyre laps
  - laps slower than 107% of race pace (traffic, damage)

  Only clean laps teach the model about pace and wear.

### 2.4 Running order and gaps at any lap (`timing.py`)

- **Crossing time** at the end of lap L = lap start time + lap duration. Sort by laps completed, then crossing time, to get the order. The gap to the leader is the difference.
- **Checked against the official result:** Bahrain 2024 order identical, and gaps within 0.01 s (Pérez +22.463 s vs official +22.457 s).
- **Each team's real future pit stops** (lap and tyre) are extracted too, for testing.
- **Edge case:** Monaco 2024 had a lap-1 red flag, so nobody completed lap 1. The function now returns an empty order instead of crashing.

### 2.5 Coverage numbers

- **46 races:** 22 in 2023 (Imola cancelled) and 24 in 2024.
- **Laps:** 51,108 laps, of which 40,678 are clean.
- **Pit stops:** 1,518.
- **Neutralisations:** 29 safety cars, 19 VSCs and 8 red flags across both seasons.
- **Wet races** (e.g. Brazil 2024) are kept for replays but excluded from model fitting.

---

## Part 3 — Validation philosophy (read this once, it applies everywhere)

**Walk-forward / leak-free testing.** A model is only tested on information that comes *after* everything it learned from:

- **Lap-time model:** fit on the first half of a race, predict the second half.
- **Priors and track parameters for 2024 tests:** learned from **2023 only**.
- **Simulator tests:** start from the real state at a lap, and compare with what really happened later.

**Why it matters:** in finance this is "no look-ahead bias". A model that peeks at the future looks brilliant and fails live. Every number in this guide is out-of-sample.

**Always against a baseline.** "0.66 s error" means nothing alone. It means something against "driver's recent pace: 1.14 s".

---

## Part 4 — Phase 1: the lap-time and tyre-wear model (`degradation.py`)

### 4.1 The model

```
lap_time = car_pace[driver] + fuel · lap + compound_pace[c] + wear[c] · tyre_age + noise
```

- **car_pace:** one number per car and driver (some cars are faster).
- **fuel:** cars get lighter and the track grips up, so laps get faster through the race (≈ −0.05 to −0.08 s per lap).
- **compound_pace:** a new soft is faster than a new hard.
- **wear:** each extra lap on a tyre costs time (e.g. Bahrain 2024: soft ≈ 0.12 s/lap, hard ≈ 0.11 s/lap).

### 4.2 Technique: linear regression with fixed effects

- **What it is:** linear regression fits y = Xβ + ε by least squares (minimise the squared errors). "Fixed effects" means a separate intercept for each group (here, each driver), so group differences don't contaminate the shared effects.
- **In BoxBox:** each driver gets their own pace intercept; fuel, compound pace and wear are shared across the field.
- **Why:** it separates "this car is fast" from "this tyre is fast". Otherwise a quick car on softs would make softs look faster than they are.

### 4.3 Technique: robust statistics (MAD outlier removal)

- **What it is:** the median absolute deviation (MAD) measures spread using medians, so a few extreme values can't distort it. Points more than ~3.5 robust standard deviations from the fit are treated as outliers.
- **In BoxBox:** fit, find laps far from the fit (traffic, mistakes, a lock-up), drop them, refit, and repeat until stable. In Bahrain about 33 of 1,024 clean laps were dropped.
- **Why not plain least squares:** one lap stuck behind a backmarker can be 3 s slow and would bias the wear estimate.

### 4.4 Technique: Bayesian linear regression (priors as pseudo-observations)

- **What it is:** combine data with prior knowledge. With a Gaussian prior N(m, s²) on a coefficient and Gaussian noise σ², the posterior mean equals ordinary least squares on an **augmented** dataset with one extra "fake" row per prior: the value σ/s in that coefficient's column, and σ/s·m on the right-hand side. Strong data outweighs the prior; thin data leans on it.
- **In BoxBox:** each compound's wear rate has a prior from the previous season (typical value and how much it varies across tracks). There are also weak "common sense" priors on fuel effect (≈ −0.06 ± 0.1) and compound pace (0 ± 1.5 s).
- **Why:**
  - **Hungary 2024:** softs came out getting *faster* with age, because very few soft laps were run. The prior fixes impossible answers like that.
  - **Mistake we made first:** we shrank the wear rate *after* fitting. When the data couldn't separate effects, this broke the other coefficients. **Monaco 2024 predictions were off by 125 s per lap.** Putting the prior *inside* the regression keeps every coefficient consistent: Monaco dropped to 0.88 s.

### 4.5 Technique: empirical Bayes

- **What it is:** learn the prior itself from data (from many similar units) instead of inventing it.
- **In BoxBox:** fit every race of a season, take the typical wear rate per compound (the median over well-measured races) and its spread across tracks. That becomes next season's prior (`season.py`). For 2023 the priors were roughly soft 0.08, medium 0.05 and hard 0.044 s/lap.
- **Why:** the prior is grounded in reality and honest about uncertainty: tracks really do differ a lot.

### 4.6 Concept: identifiability

- **What it is:** a parameter is identifiable if the data can pin it down. If two effects always move together, the data can't tell them apart.
- **In BoxBox:** Monaco 2024 had a red flag on lap 1, after which everyone changed tyres and nobody stopped again. Each driver ran one compound all race, so "fast car" and "fast tyre" were inseparable. The priors make the problem solvable.
- **Why it matters:** recognising identifiability problems is a core statistics skill, and a great interview story.

### 4.7 Pit loss per track (`pitloss.py`)

- **Method:** pit loss = the real in-lap + out-lap − what the model says those two laps would have taken without stopping. Take the median across green-flag stops.
- **Results:** Bahrain 24.3 s, Hungary 20.5 s (match commonly reported values); range 18–30 s across tracks.
- **Stops under a safety car** are measured against *the rest of the field's* laps (everyone is slow), not against green pace. The first version compared against green pace and claimed an 84-second "loss".

### 4.8 Results (walk-forward, 2024)

| Second half of 18 dry 2024 races (8,564 laps) | Error per lap |
|---|---|
| **BoxBox model** | **0.66 s** |
| Driver's recent pace (last 5 laps) | 1.14 s |
| Driver's average pace so far | 1.46 s |

- **Overall:** about 42% less error; better in 17 of 18 races (the exception, Qatar, had only 95 test laps).
- **Bahrain fit quality:** R² 0.93, noise ±0.32 s.

---

## Part 5 — Phase 2: the Monte Carlo race simulator (`simulate.py`, `hazards.py`)

### 5.1 Technique: Monte Carlo simulation

- **What it is:** estimate something uncertain by simulating many random scenarios and averaging. Used everywhere in finance (option pricing, risk/VaR).
- **In BoxBox:** from the real state at any lap, play out the rest of the race 2,000 times. Each run is a **complete** future (lap 21 → lap 57), where every lap follows from the previous one. Count outcomes: "Verstappen won 1,680 of 2,000 → 84%".
- **Important nuance:** it's 2,000 whole-race scenarios, not 2,000 tries of each lap. A safety car on lap 31 bunches the field, which changes who is close enough to fight on lap 35. Only whole-race paths capture that.

### 5.2 Technique: vectorisation (NumPy)

- **What it is:** do the same calculation on whole arrays at once instead of Python loops. 10–100× faster.
- **In BoxBox:** all state is arrays of shape (simulations × cars): race time, tyre age, compound, alive. Each lap is one array update for all 2,000 races. 2,000 races of 20 cars take under a second; the full 57-situation backtest takes ~5 s.

### 5.3 What each simulated lap does

1. **Random events:** maybe a safety car or VSC starts.
2. **Lap time** per car from the Phase 1 formula plus Normal noise (σ from the model).
3. **Under SC/VSC:** everyone runs a slow fixed pace; tyres wear at half rate.
4. **Pit stops** on planned laps: add pit loss (cheaper under SC/VSC), fit the new tyre, reset age.
5. **Rivals may cover** (Part 7).
6. **Battles:** in running order, a car that would pass the one ahead only succeeds with the track's pass chance; otherwise it is held 0.35 s behind.
7. **Behind a safety car** the field bunches into a train (0.8 s gaps).
8. **Retirements:** small chance per car per lap.

### 5.4 Technique: Poisson process (and Gamma–Poisson)

- **What it is:** a model for rare events arriving at random times at a constant rate λ: each small time step has probability ≈ λ·dt. Used for market jumps, defaults, insurance claims.
- **Gamma–Poisson:** if you're unsure of λ, put a Gamma prior on it. The posterior is (prior events + observed events) / (prior exposure + observed exposure).
- **In BoxBox:** every green lap has a small chance of a safety car (≈ 1% per lap → ≈ 0.5 per race; VSC similar). Each track's rate is shrunk toward the season rate with a prior worth about two races of laps. One lucky race can't make a track look perfectly safe.

### 5.5 Technique: Beta–Binomial

- **What it is:** for success/failure counts. A Beta prior on the success probability, updated with successes and failures: (prior successes + successes) / (prior total + trials).
- **In BoxBox:** overtaking difficulty per track = the share of close battles (< 1 s gap) that become a pass on the next lap, with a prior worth ~60 battles. Results: **Monaco 2.6%** (hardest), **Bahrain ~12%**, season average ~12.6%.

### 5.6 Technique: parameter uncertainty (not just noise)

- **What it is:** two kinds of randomness. **Noise:** random lap-to-lap wobble. **Parameter risk:** we don't know the true wear rate exactly.
- **In BoxBox:** each simulated race draws its own wear rate from N(estimate, standard error). Probabilities then reflect both kinds of uncertainty, not just luck.
- **Finance parallel:** model risk vs market risk.

### 5.7 Technique: common random numbers (CRN)

- **What it is:** when comparing two choices by simulation, use **the same random numbers** for both. The comparison isn't drowned in luck. It's like testing two umbrellas in the *same* storm.
- **In BoxBox:** random draws happen in the same order regardless of strategy, so two runs with the same seed share every safety car, lap-time wobble and retirement. This is essential for Phase 3.

### 5.8 Measured vs assumed simulator inputs

| Input | Value | Source |
|---|---|---|
| Safety car / VSC rate | ~0.5 each per race | measured (previous season) |
| SC lap pace | ~1.59× normal | measured |
| VSC lap pace | 1.35× | assumed (too few full VSC laps) |
| Pass chance | 2.6–23% per track | measured |
| Pass boost when clearly faster | 2× | assumed |
| Retirements | ~0.0023–0.0026 per car per lap (~2.6 per race) | measured |
| Pit stop under SC | 58% of green | measured (only 4 clean stops: small) |
| Pit stop under VSC | 75% of green | derived from the SC value |
| Follow gap / SC train gap | 0.35 s / 0.8 s | assumed |

**Design decision:** list assumptions openly. Honesty about what's measured vs assumed is what separates research from a demo.

### 5.9 Results (2024, leak-free)

57 situations (19 races × 1/3, 1/2, 2/3 distance), 1,000 sims each, using each team's real later pit stops:

| | BoxBox | "Order stays as it is" |
|---|---|---|
| Finishing-position error | **1.55 places** | 1.89 |
| Rank correlation | **0.881** | 0.848 |
| Favourite actually won | **93%** | 79% |
| Brier score "who wins" | **0.148** | 0.421 |

Error shrinks as the race goes on: 2.04 → 1.49 → 1.12 places (vs 2.32 → 1.81 → 1.53).

### 5.10 Technique: Brier score and calibration

- **Brier score:** the mean squared error of probability forecasts. 0 = perfect, lower is better. It rewards being confident *and* right.
- **Calibration:** when you say 70%, does it happen 70% of the time? Group forecasts into bins and compare predicted vs actual frequency.
- **BoxBox podium calibration:**

| Said | Happened |
|---|---|
| ~49% | 46% |
| ~70% | 65% |
| ~89% | 93% |

- **Spearman rank correlation:** how well the predicted order matches the real order (1 = identical).
- **Why it matters:** a quant must trust the probabilities, not just the ranking.

---

## Part 6 — Phase 3: the optimal pit engine (`strategy.py`)

### 6.1 Concept: optimal stopping and American options

- **Optimal stopping:** choose *when* to take an action, to maximise expected reward, using only information available at the time.
- **American option:** can be exercised any day until expiry, unlike a European option (expiry only). Its value = max over all exercise rules of the expected payoff. The hard part is the **continuation value**, meaning the value of waiting, which depends on optimal future behaviour.
- **In F1:** each lap, "box now" (exercise) or "stay out" (wait), knowing only this lap's situation.

### 6.2 Technique: dynamic programming / backward induction (Bellman)

- **What it is:** solve multi-step decisions from the end backwards. Value today = max(act now, expected value of acting optimally tomorrow). This is the Bellman equation.
- **In BoxBox:** start from the planned pit lap and step backwards towards now, deciding on each lap whether stopping early beats the rule already built for later laps.

### 6.3 Technique: Least-Squares Monte Carlo (Longstaff & Schwartz, 2001)

- **What it is:** the standard simulation method for American options.
  1. Simulate many price paths.
  2. At the last date, the value is the exercise payoff.
  3. Step backwards. At each date, regress the realised future value (from later dates) on variables observable *today*. The fitted value is the estimated continuation value.
  4. Exercise where the immediate payoff beats it.
  5. The option value is the average payoff of following that rule.
- **In BoxBox:**
  1. List every allowed plan (lap × tyre; and two-stop plans).
  2. Simulate 1,500 **training** races and replay *the same races* (CRN) with every plan, so each race has a known result for every choice.
  3. Choose the best **fixed** plan on the training races.
  4. Backward induction from the plan's first stop. On each lap, among races where a **safety car or VSC is out**, compare "box now" vs "wait and follow the rule" (that value comes from the laps already processed). Box only if the gain is **1.5 standard errors** clear of zero.
  5. Score the plan and the rule on **1,500 fresh test races**.

### 6.4 The most important lesson of the project: why the textbook version failed

- **First attempt:** regress on everything visible (position, gaps, tyre age) on every lap. It did **worse than a fixed plan**: negative option value; it boxed on lap 16 in 91% of races for no reason.
- **Why:** on green-flag laps, pitting one lap earlier or later changes the result by ~0.02 places. Regression noise was bigger than that signal, so the rule flipped at random.
- **Fix:** react only where the signal is strong. A safety car makes a stop ~10 s cheaper. The decision state became "neutralised this lap or not". The conditional expectation is then a group mean, which is exactly least squares on an indicator variable.
- **Takeaway:** a more complex model isn't better if the signal is weaker than the noise. Know where your model *can* learn.

### 6.5 Technique: train/test split (avoiding in-sample bias)

- **What it is:** fitting and scoring on the same data overstates performance (it "memorises the luck"). Score on fresh data.
- **In BoxBox:**
  - **Two separate sets of simulated races:** one chooses the plan and learns the rule, a different one scores them.
  - **A bias we caught:** at first we picked the "best fixed plan" on the test races, which flattered it. Now it's picked on training races.

### 6.6 Option value

- **What it is:** the value of flexibility, i.e. American minus European value.
- **In BoxBox:** expected places gained by the safety-car rule over the best fixed plan. Examples:
  - **Abu Dhabi 2024, Verstappen:** +0.25 places; the rule says "box if a SC/VSC comes out between laps 17 and 28".
  - **Season average:** +0.10 places.

### 6.7 Constraints (feasibility)

- **Two-compound rule:** in a dry race you must use two different dry compounds, so some plans are compulsory or forbidden.
- **Tyre life:** the 95th-percentile stint length from 2023 (soft 24, medium 32, hard 47 laps), per track where history exists (e.g. Bahrain hard 31). Clamped so a rarely used compound isn't given an absurd 3-lap life.
- **Why tyre-life limits exist:** the wear model is linear, with no "cliff". Without limits the engine proposed a 46-lap hard stint in Bahrain. The limits stand in for the cliff.
- **Two-stop plans:** a coarse 3-lap grid over (first stop, second stop, tyres), minimum stint 6 laps, then refined lap by lap around the best. Example: Austria 2024 Norris → "lap 21 for hards then lap 44 for hards" (McLaren really two-stopped, on laps 23 and 51).

### 6.8 Results vs the real teams (2024)

At 1/3 distance, each top-10 driver who really made one more stop under green: 115 decisions, 18 races, 800 fresh simulated races each, rivals able to cover.

| | Expected finish |
|---|---|
| Team's real call | 6.78 |
| BoxBox fixed plan | 6.41 |
| BoxBox plan + safety-car rule | 6.31 |

- **Gain:** **+0.47 places per decision**: +0.37 from the plan, +0.10 from the safety-car rule.
- **Win rate:** at least as good as the team in 95% of decisions.
- **Agreement:** same tyre as the team **85%**; median pit lap 4 laps from the team's. BoxBox *thinks like a strategist*.
- **Fairness choice:** team stops made under a real safety car are excluded (11 of 126). The simulator would replay that lap as green and unfairly punish the team.
- **Monaco** swung from −0.35 (one-stop engine) to +1.18 (two-stop engine) on only 2 decisions. Treat it as small-sample, not a finding.

---

## Part 7 — Rivals respond: covering the undercut

### 7.1 Concepts: undercut and cover

- **Undercut:** pit first, use fresher (faster) tyres while the rival stays out on worn ones, and come out ahead after they stop.
- **Cover:** the car ahead pits the very next lap to neutralise the undercut.
- **Problem in Phase 3 v1:** rivals never reacted, so undercuts always worked, and "+0.49" looked like an upper bound.

### 7.2 Technique: measuring a response rate (uplift)

- **What it is:** compare the rate of an action after a trigger with its normal (baseline) rate. The difference is the causal-ish "uplift".
- **In BoxBox (2023 data):**

| Car within 3 s after someone pits under green | Pitted within 2 laps |
|---|---|
| Directly ahead | 31% |
| Directly behind | 21% |
| Any car, normal lap | 11% |

- **Uplift:** **+22%** for the car ahead and **+11%** for the car behind (normalised).
- **In the simulator:** when our driver pits under green, the car just ahead/behind within 3 s brings a stop it had planned within the next 12 laps forward to the next lap, with that probability, independently in each simulated race.
- **Toy check:** attacker 2.5 s behind, undercut. Wins 100% if nobody reacts, **90%** at the real rate, 45% if the rival always covers.

### 7.3 Technique (model type): leader–follower (Stackelberg-style) response

- **What it is:** one player moves first; the others respond with a known rule. Here the response rule is *estimated from history* rather than optimised.
- **Result:** the gain vs teams barely changed (under 0.01 places), and BoxBox's pit lap changed in only 10% of decisions. So **the gain was not coming from free undercuts**: at a one-in-five rate, covering costs both sides about equally.
- **A correction we made (be honest about this):** an earlier comparison claimed "59% of pit laps changed". It was inflated because the two runs used slightly different prior files. Rerunning with identical inputs gave 10%. **Always compare like with like.**

---

## Part 8 — Pit duels as a game (Nash equilibrium) (`game.py`)

### 8.1 Concepts

- **Zero-sum (constant-sum) game:** one player's gain is the other's loss. Here: P(attacker finishes ahead) + P(defender stays ahead) = 1.
- **Payoff matrix:** rows are our choices, columns are theirs, and cells are our payoff.
- **Pure vs mixed strategy:** pure means always pick one action. Mixed means randomise with set probabilities; it's optimal when any fixed choice can be exploited.
- **Saddle point:** a cell that is the best for both sides simultaneously, so the equilibrium is pure.
- **Nash equilibrium:** neither player can improve by changing strategy alone.
- **Von Neumann's minimax theorem:** every finite zero-sum game has a value v and optimal mixed strategies; max-min = min-max.

### 8.2 Technique: solving with linear programming

- **What it is:** maximise v subject to M^T x ≥ v (for every opponent action), Σx = 1, x ≥ 0. Solve a second LP for the opponent. SciPy's `linprog` (HiGHS solver) does this in milliseconds.
- **Verified on textbook games:** matching pennies → 50/50; rock-paper-scissors → ⅓ each; a saddle-point game → pure strategy. These are automated tests.

### 8.3 In BoxBox

- **Setup:** two adjacent cars within 3 s, each with one stop to make. Each picks a pit lap within ±5 of the lap it really used (11×11 matrix). Each cell = 400 simulated races (CRN).
- **Example, Bahrain 2024 from lap 19, Leclerc 1.2 s behind Russell:**
  - **Equilibrium:** Leclerc mixes laps 31 (65%) / 32 (35%); Russell 31/32 at 50/50.
  - **Value:** Leclerc ahead **29.7%**. The teams' real laps (34 and 31) give **29.2%**, very close to optimal.

### 8.4 Study of all 2024 duels

34 close duels, 13 races:

| | Attacker finishes ahead |
|---|---|
| As teams really played | 40.3% |
| Both playing the Nash equilibrium | 37.9% |

- **Real play vs equilibrium:** on average within 3.9 points of the equilibrium value.
- **Best replies:** the defender's best reply to the attacker's real lap gains 5.1 points; the attacker's best reply to the defender's lap gains 3.1. **Defenders left more on the table.**
- **The equilibrium is mixed in 62% of duels.** Unpredictability is part of optimal strategy.

---

## Part 9 — Improvements and negative results

### 9.1 Calibrating the safety-car pit discount

- **Problem:** the original estimate (~0.8 of a green stop) was biased. On the deploy lap the field is still racing to catch the SC queue, so non-pitting cars' laps aren't steady.
- **Clean measurement:** only stops whose in-lap **and** out-lap were both fully behind the SC, excluding laps overlapping a VSC. Loss vs the rest of the field on those laps.
- **Result:** Suzuka 2023 0.58, Miami 2024 0.66, Shanghai 2024 0.52 → median **0.58** (only 4 clean stops; documented as a small sample).
- **VSC:** too few clean examples, so **derived**: the SC saving scaled by how much less a VSC slows the field (0.35 vs 0.59) → **0.75**.

### 9.2 Team-specific tyre wear (a negative result, kept honest)

- **Idea:** some cars are kinder to tyres. Add one wear term per team, with a prior centred on "same as the field".
- **In-sample:** it looked better (noise ±0.323 → ±0.314 s in Bahrain).
- **Walk-forward:** it predicted **worse**:
  - prior spread 0.02: 0.691 s vs 0.660 s
  - prior spread 0.01: 0.671 s
  - from a third distance: 1.101–1.103 s vs 1.096 s
- **Lesson:** better in-sample fit ≠ better prediction (overfitting). Team differences measured from a few stints are mostly noise. The feature stays in the code, **switched off**. Report negative results; they show scientific honesty.

---

## Part 10 — Engineering

### 10.1 Package layout

```
src/boxbox/
  openf1.py        cached OpenF1 client
  race.py          tidy lap table, neutralisations, clean laps
  timing.py        running order, gaps, real strategies
  degradation.py   Bayesian lap-time / wear model
  pitloss.py       pit loss per track
  season.py        season fits, empirical-Bayes priors
  hazards.py       SC/VSC rates, overtaking, DNFs, tyre life, SC pit discount
  simulate.py      vectorised Monte Carlo simulator (+ rival cover, feature recording)
  strategy.py      LSM pit engine (one or two stops)
  game.py          zero-sum duel solver (linear programming)
  __main__.py      CLI: fit, season, hazards, simulate, pit, duel
scripts/           validate, backtest_sim, backtest_strategy, duels_2024, export_site, build_assets
tests/             20 tests on hand-made races with known answers
results/           every test's output (committed, so numbers are auditable)
site/              the website
docs/              this guide
```

### 10.2 Design principles

- **Small modules with one job each.** Easy to test, easy to explain.
- **Everything reproducible from commands** (Part 17). Results files are committed so anyone can check the numbers.
- **Tests on synthetic races where the truth is known.**
  - Fit recovers known wear rates (within 3 standard errors; we checked over 40 seeds that the estimator is unbiased: mean 0.1201 vs true 0.12).
  - A faster car wins; Monaco-style tracks keep the slower car ahead; pit stops cost the right time; safety cars bunch the field; retirement rates are right.
  - The engine discovers "box under a safety car" by itself.
  - Covering blunts the undercut.
  - The game solver matches textbook answers.
- **Parallelism:** the site export and duel study are per-race independent, so they use a process pool (6 workers). That cut ~3 hours to ~30 minutes. Fixed seeds keep results identical.

### 10.3 Git and account hygiene

- **Personal accounts only:** BoxBox and the portfolio use the personal GitHub (Krishna-ctrl1) and a personal Vercel account. Work accounts are never used. Identity, remote and SSH user are checked before every push.
- **Deployment:** the site was imported on vercel.com under the personal account, rather than deployed from a command-line tool that might be signed in to a different account.
- **Licence:** MIT.

---

## Part 11 — The website (`site/`)

### 11.1 Architecture: static site + offline "backend"

```
OpenF1 → Python engine → scripts/export_site.py → site/public/data/*.json → React app → Vercel (static)
```

- **No runtime server, database or API.** The heavy maths (minutes per race) is pre-computed into small JSON files (~1.5 MB for the whole season).
- **Why:** instant loading, free hosting, nothing to overload or hack, and simple to deploy. Live races would be the reason to add a real backend later (e.g. FastAPI).

### 11.2 The data contract

- `races.json`: the index of 24 races.
- `races/<slug>.json`, one per race:
  - **replay:** order, gaps and tyres per lap; stints; neutral periods; result; team colours
  - **odds:** at 1/3, 1/2, 2/3 distance
  - **pit:** windows for the top 10 (full curve per lap × tyre, best plan, SC rule, team's real call)
  - **duels:** matrix and Nash strategies
- `results.json`: all season-level test results. The site's headline numbers are **computed from it**, so they always match the backtests.
- `assets/races-meta.json` (separate from the engine data): official Grand Prix names, rounds, and GPS circuit maps.
- `assets/photos/photos.json`: photo credits, licences and alt text.

### 11.3 Frontend stack and why

- **React 19 + TypeScript:** components and type safety; the types mirror the JSON files.
- **Vite:** fast builds.
- **Tailwind CSS v4:** a design-token based styling system (`@theme`).
- **Recharts:** line, bar and scatter charts. The duel heatmap is a plain table with colour-scaled cells.
- **Hash routing** (`#/race/sakhir`): works on any static host with no server rewrites.
- **A small `useJson` hook** for loading, with loading and error states.

### 11.4 What visitors can do

1. **Home:** the cinematic hero ("Every lap. Every decision."), live season stats, the idea in one section, and a 24-race editorial calendar.
2. **Race dossier:** the GP title, round, date and location, the GPS circuit map, and facts (laps, conditions, SC/VSC, pit cost, passes, winner).
3. **Race replay:** a lap slider and Play button, timing tower, gap chart with safety-car shading, and tyre-strategy bars.
4. **Finishing odds:** win, podium and points chances at three moments, next to the real result.
5. **Pit window:** pick a driver, see the expected finish for every pit lap and tyre, drag a slider, and see BoxBox's plan, the SC rule and the team's real choice.
6. **Pit duels:** the payoff heatmap and both teams' optimal mixes.
7. **Method, Results and Credits pages.**

### 11.5 Design: v1 → v2, and why

- **v1** was a dark analytics dashboard: functional but generic.
- **v2 (your brief):** "luxury automotive editorial × F1 cinematic photography × race intelligence".
  - **Fonts:** Space Grotesk (display), Inter (body), IBM Plex Mono (numbers).
  - **Palette:** carbon, graphite, off-white, a restrained racing red, metallic grey. **Amber is kept for safety cars**, because it means something in F1.
  - **Balance:** the home page is mostly visual storytelling; race headers are half visual; analytics tabs are almost all data.
  - **Tyre chips:** use F1 colours **plus S/M/H letters**, so colour is never the only cue (accessibility).
  - **Calendar layout:** 4 wide cards + 20 standard tile a 4-column grid in calendar order with no gaps; wide-card height is matched so rows are even.
  - **Phones:** an extra darkening layer behind text on photos, a nav that fits 375 px, no horizontal scroll.

### 11.6 Images: licensing decisions

- **Licensing:** real F1 race photos belong to F1, teams and agencies, so we **did not** take them from Google or formula1.com.
- **Sourcing:** Wikimedia Commons, under Creative Commons. 12 shortlisted, **approved by you**, downloaded, **9 kept** (2 rejected: a portrait and an empty grandstand). Cropped to 16:9, converted to WebP (36–99 KB card size), and credited on the Credits page with author, licence and changes. CC BY-SA images are shared under the same licence.
- **Circuit maps:** **drawn from the race winner's own GPS trace** (OpenF1 location data, ~300–400 points per lap, `scripts/build_assets.py`), with the start/finish line and direction of travel. Authentic, race-specific, no licensing questions. Wet races use any normal racing lap.
- **Where there's no matching photo** (21 of 24 races): the circuit map is the visual, rather than an irrelevant stock image.

### 11.7 Deployment

- **Vercel project root = `site`.** `npm run build` outputs `dist`. Every push to `main` redeploys.
- **Live verification after each deploy:** all 24 race pages load, data files return 200, the slider changes values, the duel heatmap renders, there are no console errors, and the phone layout fits.

### 11.8 Portfolio

BoxBox is the lead featured project on https://krishna-ctrl1.github.io/NewPortfolio/, with live and GitHub links and its headline numbers.

---

## Part 12 — Results at a glance

| Test (2024, trained only on the past) | BoxBox | Baseline |
|---|---|---|
| Lap-time error, second half of races | **0.66 s** | 1.14 s (recent pace) |
| Finishing-position error from mid-race | **1.55 places** | 1.89 (order stays) |
| Favourite won | **93%** | 79% (leader) |
| Brier score, who wins | **0.148** | 0.421 |
| Pit calls vs real teams (in-model) | **+0.47 places**/decision | team's call |
| Same tyre as the team | **85%** | — |
| Option value of SC rule | **+0.10** places | fixed plan |
| Duels: attacker ahead | **37.9%** at equilibrium | 40.3% as played |

---

## Part 13 — Honest limitations (know these cold)

1. **"Better than the teams" is in-model.** It's judged inside BoxBox's simulator, not proof a team was wrong. Teams see tyre temperatures, damage, fuel and driver feedback that public timing data doesn't show.
2. **Rival responses** use an average historical rate in the pit engine; the full game is solved one rival at a time with one stop each.
3. **Tyre wear is linear** in age and the same for every car; tyre-life limits stand in for the cliff.
4. **Up to two stops**; no three-stop races.
5. **Rain isn't modelled**; data starts in 2023.
6. **Small samples:** the SC pit discount rests on 4 clean stops; Monaco's gain on 2 decisions.
7. **Assumptions:** VSC pace, pass boost, following gaps.
8. **The late-race bias:** the lap model leans slightly slow in second halves (drivers push harder late; track grip grows non-linearly).

---

## Part 14 — Glossary

| Term | Meaning |
|---|---|
| Stint | laps on one set of tyres |
| Compound | soft / medium / hard (dry), intermediate / wet |
| Degradation (wear) | time lost per lap as a tyre ages |
| Pit loss | time lost by stopping vs staying out |
| Safety car (SC) / VSC | neutralisations; slow laps, cheap stops |
| Undercut / overcut | pit earlier / later than a rival to gain position |
| Cover | respond to a rival's stop by stopping next lap |
| Fixed effects | a separate intercept per group |
| MAD | median absolute deviation, a robust spread measure |
| Prior / posterior | belief before / after seeing data |
| Empirical Bayes | priors learned from many similar units |
| Identifiability | whether data can separate two effects |
| Monte Carlo | estimate by simulating random scenarios |
| Poisson process | random arrivals at a constant rate |
| Gamma–Poisson / Beta–Binomial | Bayesian models for rates / proportions |
| Common random numbers | compare choices on identical random scenarios |
| Optimal stopping | choosing when to act |
| American option | can be exercised at any time |
| Continuation value | the value of waiting |
| LSM (Longstaff–Schwartz) | regression-based backward induction for American options |
| Option value | the value of flexibility |
| Walk-forward validation | train on the past, test on the future |
| Brier score | mean squared error of probability forecasts |
| Calibration | stated probabilities match observed frequencies |
| Zero-sum game | one side's gain is the other's loss |
| Mixed strategy | randomising between actions |
| Nash equilibrium | no player gains by deviating alone |
| Minimax theorem | max-min = min-max in zero-sum games |

---

## Part 15 — Talking points (SOP, interviews)

**30-second version:**
> "BoxBox treats an F1 pit stop as an American option. I built a Bayesian tyre-wear model from real timing data, a vectorised Monte Carlo race simulator with Poisson safety cars and Beta-Binomial overtaking, and a Least-Squares Monte Carlo pit engine. Tested leak-free on 2024, the simulator's favourite won 93% of the time against 79% for the race leader, and the pit engine agreed with real teams' tyre choice 85% of the time while gaining about half a place per decision inside the model. I also solved pit duels as zero-sum games."

**Stories that show judgement:**
1. **Monaco and identifiability:** a 125 s error fixed by putting priors inside the regression.
2. **Textbook LSM failed:** signal smaller than noise; restricting the state to "safety car or not" fixed it.
3. **A negative result:** team-specific wear fit better but predicted worse; switched off.
4. **Catching my own bias:** the 59% claim was really 10% once the inputs matched.
5. **Fairness in testing:** excluded team stops under real safety cars.

**Likely questions, short answers:**
- *Why not just use machine learning?* Interpretability, small data per race, and the need for a structural model to simulate counterfactual strategies.
- *Is +0.47 real?* It's in-model. The robust evidence is agreement with teams and calibrated odds.
- *Why Bayesian?* Thin data per race, and some races where effects can't be separated.
- *What would you do next?* Live mode, rain, a non-linear tyre cliff, full multi-player equilibria, a published paper.

---

## Part 16 — Future work

- **Research paper:** ICAIF-style, or the MIT Sloan Sports Analytics conference.
- **Live race mode:** a small API (FastAPI) running the simulator on demand.
- **Non-linear tyre model:** a learned "cliff".
- **Multi-car games:** more than two cars; three-stop strategies.
- **Rain:** modelling wet races.
- **Better SC pit calibration:** more seasons of data.

---

## Part 17 — Command cheat sheet

```bash
# setup
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# explore one race
python -m boxbox fit 2024 Bahrain
python -m boxbox simulate 2024 Bahrain --from-lap 20
python -m boxbox pit 2024 "Yas Marina" VER --from-lap 12
python -m boxbox duel 2024 Bahrain LEC RUS --from-lap 19

# rebuild learned inputs
python -m boxbox season 2023 && python -m boxbox season 2024
python -m boxbox hazards 2023 && python -m boxbox hazards 2023 2024

# tests and backtests
pytest
python scripts/validate.py 2024
python scripts/backtest_sim.py 2024
python scripts/backtest_strategy.py 2024            # add --no-response to compare
python scripts/duels_2024.py 2024

# website data and dev
python scripts/export_site.py 2024 --jobs 6
python scripts/build_assets.py 2024
cd site && npm install && npm run dev
```

---

## Reference

Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: a simple least-squares approach. *The Review of Financial Studies*, 14(1), 113–147.
