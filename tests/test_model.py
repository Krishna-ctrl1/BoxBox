"""Tests that run offline, on synthetic data where the true answer is known."""

import numpy as np
import pandas as pd
import pytest

from boxbox import degradation
from boxbox.race import _neutralisations


def synthetic_race(seed=0, n_drivers=10, laps=50, noise=0.2):
    """A fake race with known pace, fuel effect, compound offsets and wear."""
    rng = np.random.default_rng(seed)
    truth = dict(fuel=-0.06, off={"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.4}, deg={"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04})
    rows = []
    for d in range(1, n_drivers + 1):
        base = 90 + rng.normal(0, 0.5)
        plan = [("SOFT", 1, 15), ("HARD", 16, 50)] if d % 2 else [("MEDIUM", 1, 22), ("SOFT", 23, 32), ("HARD", 33, 50)]
        for comp, a, b in plan:
            for lap in range(a, b + 1):
                age = lap - a + 1
                t = base + truth["fuel"] * lap + truth["off"][comp] + truth["deg"][comp] * age + rng.normal(0, noise)
                rows.append(dict(driver=d, lap=lap, compound=comp, tyre_age=age, lap_time=t))
    return pd.DataFrame(rows), truth


def test_fit_recovers_known_wear_rates():
    df, truth = synthetic_race()
    m = degradation.fit(df)
    for c, k in truth["deg"].items():
        assert m.deg[c] == pytest.approx(k, abs=3 * m.deg_se[c])  # within 3 standard errors
    assert m.fuel_per_lap == pytest.approx(truth["fuel"], abs=0.01)
    # A compound's starting pace and its wear rate trade off against each other
    # (a bit more wear can be offset by a faster new tyre), so this is looser.
    assert m.compound_offset["SOFT"] - m.compound_offset["HARD"] == pytest.approx(-1.0, abs=0.3)


def test_robust_fit_ignores_outlier_laps():
    df, truth = synthetic_race(seed=1)
    df.loc[df.sample(30, random_state=1).index, "lap_time"] += 8  # traffic / mistakes
    m = degradation.fit(df)
    assert m.n_dropped >= 25
    assert m.deg["MEDIUM"] == pytest.approx(truth["deg"]["MEDIUM"], abs=0.015)


def test_prediction_matches_truth():
    df, _ = synthetic_race(seed=2, noise=0.0)
    m = degradation.fit(df)
    pred = m.predict(df["driver"], df["lap"], df["compound"], df["tyre_age"])
    # The weak built-in priors pull estimates very slightly, so allow a few hundredths.
    assert np.abs(pred - df["lap_time"].to_numpy()).max() < 0.05


def test_shrink_moves_noisy_estimates_more():
    noisy, _ = degradation.shrink(-0.05, se=0.05, prior_mean=0.08, prior_sd=0.02)
    precise, _ = degradation.shrink(-0.05, se=0.002, prior_mean=0.08, prior_sd=0.02)
    assert noisy > 0.05  # pulled most of the way to the prior
    assert precise < -0.04  # barely moves


def test_safety_car_and_red_flag_parsing():
    msgs = [
        dict(date="1", lap_number=7, category="SafetyCar", flag=None, message="SAFETY CAR DEPLOYED"),
        dict(date="2", lap_number=8, category="Flag", flag="RED", message="RED FLAG"),
        dict(date="3", lap_number=18, category="SafetyCar", flag=None, message="VIRTUAL SAFETY CAR DEPLOYED"),
        dict(date="4", lap_number=19, category="SafetyCar", flag=None, message="VIRTUAL SAFETY CAR ENDING"),
        dict(date="5", lap_number=40, category="SafetyCar", flag=None, message="SAFETY CAR DEPLOYED"),
    ]
    n = [(x.kind, x.start_lap, x.end_lap) for x in _neutralisations(msgs, total_laps=50)]
    assert ("SC", 7, 8) in n  # closed by the red flag
    assert ("RED", 8, 9) in n
    assert ("VSC", 18, 20) in n
    assert ("SC", 40, 50) in n  # never ended: runs to the flag


def test_one_stint_per_driver_stays_sane():
    """Monaco 2024 case: every driver runs a single compound all race.

    Driver pace and tyre pace cannot be separated from the data alone; the
    priors must keep predictions sensible instead of exploding.
    """
    df, _ = synthetic_race(seed=3)
    df = df[df["lap"] <= 15]  # odd drivers on SOFT only, even drivers on MEDIUM only
    m = degradation.fit(df, prior={"SOFT": (0.12, 0.03), "MEDIUM": (0.07, 0.03)})
    pred = m.predict(df["driver"], df["lap"], df["compound"], df["tyre_age"])
    assert np.abs(pred - df["lap_time"].to_numpy()).mean() < 0.4
