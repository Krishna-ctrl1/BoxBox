"""Optimal pit stop as an American option, solved with Least-Squares Monte Carlo.

The question for one driver: from now, on which lap should we pit, and onto
which tyre? An American option can be exercised on any day; the holder must
decide each day "exercise now or wait?", knowing only today's information.
Pitting is the same: each lap the team must say "box" or "stay out".

Method (after Longstaff & Schwartz, 2001):

  1. Simulate many races ("training" races). For every possible pit lap and
     tyre, replay the same races (same random numbers) with that stop, so every
     simulated race has a known result for every choice.
  2. Pick the best fixed plan on the training races.
  3. Backward induction from the planned lap towards now: on each lap, among
     the races where a safety car or VSC is out on that lap, compare "box now"
     with "wait and follow the rule from here" (whose value comes from the laps
     already processed). Box only if the gain is clearly bigger than the noise.
  4. Judge the plan and the rule on fresh "test" races, so the scores are not
     flattered by the luck they were fitted on.

Why the rule only reacts to safety cars: we first tried the textbook version,
regressing on every visible feature (position, gaps, tyre age) on every lap.
On green-flag laps the difference between pitting on lap 21 or 22 is a few
hundredths of a position, smaller than the regression noise, so that rule
flipped almost at random and did worse than a fixed plan. A safety car makes a
stop roughly 10 s cheaper: a strong, reliable signal. The state used is
therefore "neutralised this lap or not", and the conditional expectation is
the plain sample mean over those races (the regression on an indicator).

Comparing the rule with the best fixed plan gives the value of being able to
react: the option value. Current scope: at most one more stop; rivals keep
their own strategies (responding rivals come in the game-theory step).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .degradation import DegModel
from .hazards import TrackParams
from .simulate import Car, simulate
from .timing import Stop

# 95th percentile of how long a stint lasted before a pit stop, 2023 dry races.
TYRE_LIFE = {"SOFT": 24, "MEDIUM": 32, "HARD": 47}
DRY = ("SOFT", "MEDIUM", "HARD")
SAFETY_MARGIN = 1.5  # box under SC only if the gain is 1.5 standard errors clear of zero
MIN_SC_RACES = 10  # need at least this many training races with a SC on a lap to judge it


@dataclass
class PitAdvice:
    code: str
    start_lap: int
    total_laps: int
    fixed: pd.DataFrame  # every fixed plan: lap, compound, expected position, chances (test races)
    no_stop: dict | None  # staying out to the flag, if the rules allow it
    adaptive: dict  # the plan + safety-car rule, on test races
    best_fixed: dict  # the plan chosen on training races, scored on test races
    option_value: float  # positions gained by being able to react
    sc_rule: pd.DataFrame  # laps where a safety car means "box now", and onto which tyre
    pit_laps: pd.Series  # where the car pitted under the rule (share of races)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        b, a = self.best_fixed, self.adaptive
        plan = "stay out to the flag" if b["lap"] is None else f"pit at the end of lap {b['lap']} for {b['compound'].lower()}s"
        lines = [
            f"{self.code} from the end of lap {self.start_lap} of {self.total_laps}:",
            f"  Best fixed plan: {plan} -> expected finish P{b['exp_pos']:.2f} "
            f"(podium {b['p_podium']:.0%}, points {b['p_points']:.0%})",
        ]
        if self.no_stop and b["lap"] is not None:
            lines.append(f"  No more stops instead: expected finish P{self.no_stop['exp_pos']:.2f}")
        if len(self.sc_rule):
            laps = self.sc_rule["lap"]
            comps = "/".join(sorted(set(self.sc_rule["compound"].str.lower())))
            lines.append(f"  Safety-car rule: if a SC or VSC comes out between laps {laps.min()} and {laps.max()}, box at once for {comps}s")
        else:
            lines.append("  Safety-car rule: nothing clearly better than sticking to the plan")
        lines.append(
            f"  With the rule: expected P{a['exp_pos']:.2f} (podium {a['p_podium']:.0%}, points {a['p_points']:.0%}); "
            f"rule triggered in {a['boxed_under_sc']:.0%} of races"
        )
        lines.append(f"  Value of flexibility (option value): {self.option_value:+.3f} positions")
        lines += [f"  Note: {n}" for n in self.notes]
        return "\n".join(lines)


def _stats(pos: np.ndarray) -> dict:
    pos = np.asarray(pos, dtype=float)
    return dict(exp_pos=float(pos.mean()), p_win=float((pos == 1).mean()), p_podium=float((pos <= 3).mean()),
                p_points=float((pos <= 10).mean()))


def feasible_plans(me: Car, used: set[str], start_lap: int, total_laps: int, life: dict, min_laps_after: int = 2):
    """(lap, compound) pairs a single stop could use, and whether staying out is allowed."""
    used = set(used) | {me.compound}
    must_change = len(used & set(DRY)) < 2
    plans = []
    for lap in range(start_lap + 1, total_laps - min_laps_after + 1):
        if me.tyre_age + (lap - start_lap) > life.get(me.compound, 40) + 3:
            break  # current tyre would be far past its usual life
        for c in DRY:
            if must_change and c in used:
                continue
            if total_laps - lap > life[c]:
                continue  # new tyre would not reach the flag
            plans.append((lap, c))
    can_stay_out = not must_change and me.tyre_age + (total_laps - start_lap) <= life.get(me.compound, 40)
    return plans, can_stay_out, must_change


def optimal_pit(
    cars: list[Car],
    focal: int,
    start_lap: int,
    total_laps: int,
    model: DegModel,
    track: TrackParams,
    used_compounds: set[str] | None = None,
    n_train: int = 1500,
    n_test: int = 1500,
    seed: int = 7,
    tyre_life: dict[str, int] | None = None,
) -> PitAdvice:
    """Best pit lap and tyre for cars[focal]: a fixed plan plus a safety-car rule."""
    # Tyre life: this track's own history where we have it, else the season-wide limit.
    life = dict(TYRE_LIFE)
    # (Clamped: a compound barely used at a track would otherwise look like it lasts 3 laps.)
    life.update({c: int(np.clip(v, TYRE_LIFE[c] // 2, TYRE_LIFE[c] + 5)) for c, v in (track.tyre_life or {}).items() if c in TYRE_LIFE})
    life.update(tyre_life or {})
    me = cars[focal]
    plans, can_stay_out, must_change = feasible_plans(me, used_compounds or set(), start_lap, total_laps, life)
    if not plans and not can_stay_out:
        raise ValueError(f"{me.code}: one stop cannot reach the flag within tyre life (two-stop plans not modelled yet).")
    notes = ["rules require a different compound, so one more stop is compulsory"] if must_change else []

    def with_stop(stops):
        out = list(cars)
        out[focal] = replace(me, stops=stops)
        return out

    sims = {}
    for name, sd, n in (("train", seed, n_train), ("test", seed + 1000, n_test)):
        base = simulate(with_stop([]), start_lap, total_laps, model, track, n=n, seed=sd, record=focal)
        Y = {p: simulate(with_stop([Stop(*p)]), start_lap, total_laps, model, track, n=n, seed=sd).finish[:, focal]
             for p in plans}
        f = base.features
        sims[name] = dict(Y=Y, stay=base.finish[:, focal], neutral=(f["sc"] + f["vsc"]) > 0)
    tr, te = sims["train"], sims["test"]

    # 1. Best fixed plan, chosen on the training races.
    train_means = {p: tr["Y"][p].mean() for p in plans}
    best_plan = min(train_means, key=train_means.get) if plans else None
    plan_is_stay = can_stay_out and (best_plan is None or tr["stay"].mean() <= train_means[best_plan])
    plan_lap = total_laps + 1 if plan_is_stay else best_plan[0]

    # 2. Backward induction for "box early if a safety car / VSC comes out".
    V = (tr["stay"] if plan_is_stay else tr["Y"][best_plan]).astype(float)
    rule = {}
    for lap in sorted({p[0] for p in plans if p[0] < plan_lap}, reverse=True):
        S = tr["neutral"][:, lap - start_lap - 1]
        if S.sum() < MIN_SC_RACES:
            continue
        options = {}
        for c in DRY:
            if (lap, c) in tr["Y"]:
                d = tr["Y"][(lap, c)][S] - V[S]  # negative = better than waiting
                options[c] = (d.mean(), d.std(ddof=1) / np.sqrt(S.sum()))
        c_best = min(options, key=lambda c: options[c][0])
        gain, se = options[c_best]
        if gain + SAFETY_MARGIN * se < 0:
            rule[lap] = c_best
            V[S] = tr["Y"][(lap, c_best)][S]

    # 3. Judge the plan and the rule on fresh races.
    n = len(te["stay"])
    plan_result = (te["stay"] if plan_is_stay else te["Y"][best_plan]).astype(float)
    result = plan_result.copy()
    decided = np.zeros(n, dtype=bool)
    pit_lap = np.full(n, np.nan if plan_is_stay else float(plan_lap))
    for lap in sorted(rule):
        take = te["neutral"][:, lap - start_lap - 1] & ~decided
        result = np.where(take, te["Y"][(lap, rule[lap])], result)
        pit_lap = np.where(take, lap, pit_lap)
        decided |= take

    fixed = pd.DataFrame(
        [dict(lap=l, compound=c, train_exp_pos=train_means[(l, c)], **_stats(te["Y"][(l, c)])) for (l, c) in plans]
    )
    if len(fixed):
        fixed = fixed.sort_values("exp_pos").reset_index(drop=True)
    if plan_is_stay:
        best = dict(lap=None, compound="stay out", **_stats(te["stay"]))
    else:
        best = dict(lap=best_plan[0], compound=best_plan[1], **_stats(te["Y"][best_plan]))
    adaptive = _stats(result)
    adaptive["boxed_under_sc"] = float(decided.mean())
    no_stop = _stats(te["stay"]) if can_stay_out else None
    sc_rule = pd.DataFrame([dict(lap=l, compound=c) for l, c in sorted(rule.items())], columns=["lap", "compound"])
    pl = pd.Series(pit_lap[~np.isnan(pit_lap)]).value_counts(normalize=True) * (~np.isnan(pit_lap)).mean()
    return PitAdvice(
        code=me.code, start_lap=start_lap, total_laps=total_laps, fixed=fixed, no_stop=no_stop, adaptive=adaptive,
        best_fixed=best, option_value=float(best["exp_pos"] - adaptive["exp_pos"]), sc_rule=sc_rule,
        pit_laps=pl, notes=notes,
    )
