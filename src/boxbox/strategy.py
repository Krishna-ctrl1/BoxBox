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
react: the option value. Rivals respond as real teams did in 2023: the car
just ahead covers a green-flag stop about one time in five (see simulate.COVER_*).
This is a leader-follower model, not a full Nash equilibrium. Scope: at most
one more stop.
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
        plan = b.get("plan") or ("stay out to the flag" if b["lap"] is None else f"lap {b['lap']} for {b['compound'].lower()}s")
        plan = plan if b["lap"] is None else f"{b.get('stops', 1)} stop{'s' if b.get('stops', 1) > 1 else ''}: pit {plan}"
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


def _stint_ok(me: Car, start_lap: int, total_laps: int, stops: tuple, life: dict, min_stint: int) -> bool:
    """Every stint of a plan fits its tyre's life; the current tyre is not run far past it."""
    if me.tyre_age + (stops[0][0] - start_lap) > life.get(me.compound, 40) + 3:
        return False
    for (lap, c), nxt in zip(stops, list(stops[1:]) + [(total_laps, None)]):
        length = nxt[0] - lap
        if length > life[c] or (nxt[1] is not None and length < min_stint):
            return False
    return True


def feasible_plans(me: Car, used: set[str], start_lap: int, total_laps: int, life: dict, min_laps_after: int = 2,
                   max_stops: int = 1, grid: int = 3, min_stint: int = 6):
    """Stop plans that fit the rules and tyre life, and whether staying out is allowed.

    A plan is a tuple of (lap, compound) stops. One-stop plans are listed lap by lap;
    two-stop plans on a coarse grid (every `grid` laps), refined later around the best.
    """
    used = set(used) | {me.compound}
    must_change = len(used & set(DRY)) < 2
    plans = []
    for lap in range(start_lap + 1, total_laps - min_laps_after + 1):
        for c in DRY:
            p = ((lap, c),)
            if must_change and c in used:
                continue
            if _stint_ok(me, start_lap, total_laps, p, life, min_stint):
                plans.append(p)
    if max_stops >= 2:
        for l1 in range(start_lap + 1, total_laps - min_stint - min_laps_after, grid):
            for l2 in range(l1 + min_stint, total_laps - min_laps_after + 1, grid):
                for c1 in DRY:
                    for c2 in DRY:
                        if must_change and {c1, c2} <= used:
                            continue
                        p = ((l1, c1), (l2, c2))
                        if _stint_ok(me, start_lap, total_laps, p, life, min_stint):
                            plans.append(p)
    can_stay_out = not must_change and me.tyre_age + (total_laps - start_lap) <= life.get(me.compound, 40)
    return plans, can_stay_out, must_change


def _describe(plan) -> str:
    return " then ".join(f"lap {l} for {c.lower()}s" for l, c in plan)


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
    rivals_respond: bool = True,
    max_stops: int = 2,
    full_table: bool = False,
) -> PitAdvice:
    """Best stop plan for cars[focal] (one or two stops): a fixed plan plus a safety-car rule."""
    # Tyre life: this track's own history where we have it, else the season-wide limit.
    life = dict(TYRE_LIFE)
    # (Clamped: a compound barely used at a track would otherwise look like it lasts 3 laps.)
    life.update({c: int(np.clip(v, TYRE_LIFE[c] // 2, TYRE_LIFE[c] + 5)) for c, v in (track.tyre_life or {}).items() if c in TYRE_LIFE})
    life.update(tyre_life or {})
    me = cars[focal]
    used = set(used_compounds or set()) | {me.compound}
    plans, can_stay_out, must_change = feasible_plans(me, used, start_lap, total_laps, life, max_stops=max_stops)
    if not plans and not can_stay_out:
        raise ValueError(f"{me.code}: no plan with up to {max_stops} stops reaches the flag within tyre life.")
    notes = ["rules require a different compound, so at least one more stop is compulsory"] if must_change else []
    resp = focal if rivals_respond else None
    seeds = {"train": (seed, n_train), "test": (seed + 1000, n_test)}

    def with_stops(stops):
        out = list(cars)
        out[focal] = replace(me, stops=[Stop(l, c) for l, c in stops])
        return out

    Y = {"train": {}, "test": {}}

    def run(plan, which=("train", "test")):
        for name in which:
            if plan not in Y[name]:
                sd, n = seeds[name]
                Y[name][plan] = simulate(with_stops(plan), start_lap, total_laps, model, track, n=n, seed=sd,
                                         respond_to=resp).finish[:, focal]
        return Y["train"][plan]

    base = {}
    for name, (sd, n) in seeds.items():
        b = simulate(with_stops(()), start_lap, total_laps, model, track, n=n, seed=sd, record=focal)
        base[name] = dict(stay=b.finish[:, focal], neutral=(b.features["sc"] + b.features["vsc"]) > 0)

    # 1. Best fixed plan, chosen on the training races. Two-stop plans were listed on a
    #    coarse grid, so refine around the best one lap at a time.
    for p in plans:
        run(p, ("train",))
    two = [p for p in plans if len(p) == 2]
    if two:
        best_two = min(two, key=lambda p: Y["train"][p].mean())
        (l1, c1), (l2, c2) = best_two
        for d1 in range(-2, 3):
            for d2 in range(-2, 3):
                p = ((l1 + d1, c1), (l2 + d2, c2))
                if p not in Y["train"] and start_lap < p[0][0] < p[1][0] <= total_laps - 2 and _stint_ok(me, start_lap, total_laps, p, life, 6):
                    plans.append(p)
                    run(p, ("train",))
    train_means = {p: Y["train"][p].mean() for p in plans}
    best_plan = min(train_means, key=train_means.get) if plans else None
    plan_is_stay = can_stay_out and (best_plan is None or base["train"]["stay"].mean() <= train_means[best_plan])
    first_lap = total_laps + 1 if plan_is_stay else best_plan[0][0]

    # 2. Backward induction for "box early if a safety car / VSC comes out": take the plan's
    #    first stop now (or, when staying out, make one stop now) and keep the rest of the plan.
    def early(lap, c):
        rest = () if plan_is_stay else best_plan[1:]
        p = ((lap, c),) + tuple(rest)
        if rest and rest[0][0] - lap < 6:
            return None
        if must_change and {c} | {x for _, x in rest} <= used:
            return None
        return p if _stint_ok(me, start_lap, total_laps, p, life, 6) else None

    V = (base["train"]["stay"] if plan_is_stay else Y["train"][best_plan]).astype(float)
    rule = {}
    for lap in range(first_lap - 1, start_lap, -1):
        S = base["train"]["neutral"][:, lap - start_lap - 1]
        if S.sum() < MIN_SC_RACES:
            continue
        options = {}
        for c in DRY:
            p = early(lap, c)
            if p is None:
                continue
            d = run(p, ("train",))[S] - V[S]  # negative = better than waiting
            options[p] = (d.mean(), d.std(ddof=1) / np.sqrt(S.sum()))
        if not options:
            continue
        p_best = min(options, key=lambda p: options[p][0])
        gain, se = options[p_best]
        if gain + SAFETY_MARGIN * se < 0:
            rule[lap] = p_best
            V[S] = Y["train"][p_best][S]

    # 3. Judge the plan and the rule on fresh races.
    te_stay = base["test"]["stay"]
    n = len(te_stay)
    if not plan_is_stay:
        run(best_plan, ("test",))
    result = (te_stay if plan_is_stay else Y["test"][best_plan]).astype(float).copy()
    decided = np.zeros(n, dtype=bool)
    pit_lap = np.full(n, np.nan if plan_is_stay else float(first_lap))
    for lap in sorted(rule):
        run(rule[lap], ("test",))
        take = base["test"]["neutral"][:, lap - start_lap - 1] & ~decided
        result = np.where(take, Y["test"][rule[lap]], result)
        pit_lap = np.where(take, lap, pit_lap)
        decided |= take

    # The best few plans of each kind, scored on fresh races (for the table and the backtest).
    top = sorted(plans, key=lambda p: train_means[p])
    one = [p for p in top if len(p) == 1]
    show = (one if full_table else one[:12]) + [p for p in top if len(p) == 2][:8]
    for p in show:
        run(p, ("test",))
    fixed = pd.DataFrame([
        dict(stops=len(p), plan=_describe(p), lap=p[0][0], compound=p[0][1], train_exp_pos=train_means[p],
             **_stats(Y["test"][p])) for p in show
    ])
    if len(fixed):
        fixed = fixed.sort_values("exp_pos").reset_index(drop=True)
    if plan_is_stay:
        best = dict(lap=None, compound="stay out", stops=0, plan="stay out to the flag", **_stats(te_stay))
    else:
        best = dict(lap=best_plan[0][0], compound=best_plan[0][1], stops=len(best_plan), plan=_describe(best_plan),
                    **_stats(Y["test"][best_plan]))
    adaptive = _stats(result)
    adaptive["boxed_under_sc"] = float(decided.mean())
    no_stop = _stats(te_stay) if can_stay_out else None
    sc_rule = pd.DataFrame([dict(lap=l, compound=p[0][1]) for l, p in sorted(rule.items())], columns=["lap", "compound"])
    pl = pd.Series(pit_lap[~np.isnan(pit_lap)]).value_counts(normalize=True) * (~np.isnan(pit_lap)).mean()
    return PitAdvice(
        code=me.code, start_lap=start_lap, total_laps=total_laps, fixed=fixed, no_stop=no_stop, adaptive=adaptive,
        best_fixed=best, option_value=float(best["exp_pos"] - adaptive["exp_pos"]), sc_rule=sc_rule,
        pit_laps=pl, notes=notes,
    )
