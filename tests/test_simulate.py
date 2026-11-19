"""Simulator sanity checks on hand-made races where the right answer is obvious."""

import numpy as np

from boxbox.degradation import DegModel
from boxbox.hazards import TrackParams
from boxbox.simulate import Car, simulate
from boxbox.timing import Stop


def model(bases):
    return DegModel(
        driver_base=bases, fuel_per_lap=-0.05,
        compound_offset={"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.4},
        deg={"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}, deg_se={"SOFT": 0.01, "MEDIUM": 0.01, "HARD": 0.01},
        deg_raw={}, deg2={}, sigma=0.3, r2=0.9, n_laps=1000, n_dropped=0, reference="MEDIUM", quadratic=False,
    )


def track(**kw):
    p = dict(circuit="Test", races=1, sc_rate=0.0, vsc_rate=0.0, pass_prob=0.5, pit_loss=22.0, sc_pit_factor=0.55,
             sc_pace_factor=1.4, vsc_pace_factor=1.35, sc_laps=[4], lap_time=90.0, dnf_rate=0.0)
    p.update(kw)
    return TrackParams(**p)


def test_much_faster_car_wins_when_overtaking_is_easy():
    m = model({1: 90.0, 2: 89.0})  # car 2 is a second a lap faster
    cars = [Car(1, "AAA", 0.0, "HARD", 5), Car(2, "BBB", 3.0, "HARD", 5)]
    r = simulate(cars, 10, 40, m, track(pass_prob=0.45), n=500)
    t = r.table().set_index("code")
    assert t.loc["BBB", "p_win"] > 0.95


def test_monaco_style_track_keeps_the_slower_car_ahead():
    m = model({1: 90.0, 2: 89.6})  # faster, but not by much
    cars = [Car(1, "AAA", 0.0, "HARD", 5), Car(2, "BBB", 0.8, "HARD", 5)]
    easy = simulate(cars, 30, 50, m, track(pass_prob=0.45), n=1000).table().set_index("code")
    hard = simulate(cars, 30, 50, m, track(pass_prob=0.01), n=1000).table().set_index("code")
    assert hard.loc["AAA", "p_win"] > easy.loc["AAA", "p_win"] + 0.3


def test_pit_stop_costs_pit_loss():
    m = model({1: 90.0, 2: 90.0})
    no_stop = [Car(1, "AAA", 0.0, "HARD", 5), Car(2, "BBB", 0.0, "HARD", 5)]
    one_stop = [Car(1, "AAA", 0.0, "HARD", 5, [Stop(20, "HARD")]), Car(2, "BBB", 0.0, "HARD", 5)]
    a = simulate(no_stop, 10, 30, m, track(pass_prob=0.0), n=200, events=False, param_uncertainty=False)
    b = simulate(one_stop, 10, 30, m, track(pass_prob=0.0), n=200, events=False, param_uncertainty=False)
    # Fresh tyres claw some time back, so the net cost is less than the full 22 s.
    diff = b.race_time[:, 0].mean() - a.race_time[:, 0].mean()
    assert 14 < diff < 22


def test_safety_car_bunches_the_field():
    m = model({1: 90.0, 2: 90.0})
    cars = [Car(1, "AAA", 0.0, "HARD", 5), Car(2, "BBB", 20.0, "HARD", 5)]
    calm = simulate(cars, 10, 20, m, track(pass_prob=0.0), n=300, events=False)
    sc = simulate(cars, 10, 20, m, track(pass_prob=0.0, sc_rate=1.0), n=300)
    gap_calm = (calm.race_time[:, 1] - calm.race_time[:, 0]).mean()
    gap_sc = (sc.race_time[:, 1] - sc.race_time[:, 0]).mean()
    assert gap_calm > 15 and gap_sc < 5


def test_dnf_and_results_shape():
    m = model({1: 90.0, 2: 90.5, 3: 91.0})
    cars = [Car(d, f"C{d}", g, "MEDIUM", 3) for d, g in [(1, 0.0), (2, 2.0), (3, 4.0)]]
    r = simulate(cars, 1, 50, m, track(dnf_rate=0.01), n=400)
    assert r.finish.shape == (400, 3)
    assert set(np.unique(r.finish)) <= {1, 2, 3}
    t = r.table()
    assert 0.2 < t["p_dnf"].mean() < 0.6  # ~1 - 0.99**49
    assert abs(t["p_win"].sum() - 1) < 1e-9
