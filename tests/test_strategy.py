"""Pit engine checks on hand-made races."""

from boxbox.simulate import Car
from boxbox.strategy import TYRE_LIFE, feasible_plans, optimal_pit
from test_simulate import model, track


def test_rules_force_a_new_compound():
    me = Car(1, "AAA", 0.0, "MEDIUM", 10)
    plans, stay, must = feasible_plans(me, {"MEDIUM"}, 20, 50, TYRE_LIFE)
    assert must and not stay
    assert all(c != "MEDIUM" for _, c in plans)


def test_new_tyre_must_reach_the_flag():
    me = Car(1, "AAA", 0.0, "MEDIUM", 5)
    plans, _, _ = feasible_plans(me, {"MEDIUM"}, 10, 60, TYRE_LIFE)
    assert plans
    assert all(60 - lap <= TYRE_LIFE[c] for lap, c in plans)


def test_engine_discovers_box_under_safety_car():
    """AAA leads by 15 s and must stop once; passing is hard.

    A green-flag stop (22 s) hands the lead to BBB; a stop under a safety car
    (~12 s) keeps it. The engine should find "box if a safety car comes out"
    by itself, and that rule should beat any fixed pit lap on fresh races.
    """
    m = model({1: 90.0, 2: 90.0})
    cars = [Car(1, "AAA", 0.0, "MEDIUM", 8), Car(2, "BBB", 15.0, "HARD", 3)]
    t = track(sc_rate=0.04, vsc_rate=0.0, pass_prob=0.05, tyre_life={})
    adv = optimal_pit(cars, 0, 10, 50, m, t, used_compounds={"MEDIUM"}, n_train=1500, n_test=1500)
    assert adv.best_fixed["compound"] == "HARD"
    assert len(adv.sc_rule) > 0
    assert adv.option_value > 0.05


def test_covering_rivals_blunt_the_undercut(monkeypatch):
    """BBB leads AAA by 2.5 s, both must stop; passing is impossible.

    If AAA pits first (undercut) on fresher tyres it jumps BBB, unless BBB covers
    by pitting the next lap. With covering switched fully on, the undercut
    should win far less often.
    """
    import boxbox.simulate as S
    from boxbox.timing import Stop

    m = model({1: 90.0, 2: 90.0})
    m.deg = {"SOFT": 0.12, "MEDIUM": 0.08, "HARD": 0.04}  # a normal undercut: fresh hards gain ~1.5 s/lap
    cars = [Car(2, "BBB", 0.0, "MEDIUM", 20, [Stop(30, "HARD")]), Car(1, "AAA", 2.5, "MEDIUM", 20, [Stop(24, "HARD")])]
    t = track(pass_prob=0.0, tyre_life={})
    calm = S.simulate(cars, 20, 50, m, t, n=600, seed=3, events=False).table().set_index("code")
    monkeypatch.setattr(S, "COVER_AHEAD", 1.0)  # BBB is the car ahead of AAA
    covered = S.simulate(cars, 20, 50, m, t, n=600, seed=3, events=False, respond_to=1).table().set_index("code")
    assert calm.loc["AAA", "p_win"] > 0.8  # the undercut works when nobody reacts
    assert covered.loc["AAA", "p_win"] < calm.loc["AAA", "p_win"] - 0.3
