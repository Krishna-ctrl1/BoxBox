"""Game-theory checks: the solver must reproduce textbook answers."""

import numpy as np
import pytest

from boxbox.game import pit_duel, solve_zero_sum
from boxbox.simulate import Car
from test_simulate import model, track


def test_matching_pennies_is_a_fifty_fifty_mix():
    v, x, y = solve_zero_sum(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert v == pytest.approx(0.5, abs=1e-6)
    assert x == pytest.approx([0.5, 0.5], abs=1e-6)
    assert y == pytest.approx([0.5, 0.5], abs=1e-6)


def test_saddle_point_gives_a_pure_strategy():
    M = np.array([[0.6, 0.7], [0.4, 0.5]])  # row 1 dominates; column 1 is the column player's best
    v, x, y = solve_zero_sum(M)
    assert v == pytest.approx(0.6, abs=1e-6)
    assert x == pytest.approx([1, 0], abs=1e-6) and y == pytest.approx([1, 0], abs=1e-6)


def test_rock_paper_scissors_value_and_mix():
    M = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
    v, x, _ = solve_zero_sum(M)
    assert v == pytest.approx(0.5, abs=1e-6)
    assert x == pytest.approx([1 / 3] * 3, abs=1e-6)


def test_duel_on_a_simulated_race():
    m = model({1: 90.0, 2: 90.0})
    cars = [Car(1, "AAA", 0.0, "MEDIUM", 15), Car(2, "BBB", 1.0, "MEDIUM", 15)]
    d = pit_duel(cars, 1, 0, 15, 45, m, track(pass_prob=0.05, tyre_life={}), [18, 20, 22, 24], [18, 20, 22, 24],
                 "HARD", "HARD", n=300)
    assert d.matrix.shape == (4, 4)
    assert 0 <= d.value <= 1
    assert d.x.sum() == pytest.approx(1, abs=1e-6) and d.y.sum() == pytest.approx(1, abs=1e-6)
    # Game value lies between the worst and best guaranteed outcomes.
    assert d.matrix.min(axis=1).max() - 1e-9 <= d.value <= d.matrix.max(axis=0).min() + 1e-9
