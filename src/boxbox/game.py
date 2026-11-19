"""Pit-stop duel as a two-player game, solved for its Nash equilibrium.

Two cars fight for track position; each still has one stop to make and each
team picks the lap. Our payoff is the chance we finish ahead of the rival; the
rival wants that as low as possible, so the game is zero-sum (constant-sum).

For every pair (our pit lap, their pit lap) in a window, the simulator plays
the same random races (common random numbers) and records how often we end up
ahead. That fills a payoff matrix M. Von Neumann's minimax theorem says the
game has a value v and optimal (possibly mixed) strategies for both sides,
found by linear programming:

    maximise v  subject to  M^T x >= v,  sum(x) = 1,  x >= 0

A mixed strategy ("lap 18 with 60%, lap 20 with 40%") means any fixed lap can
be exploited by a rival who knows it; being unpredictable is part of optimal
play. If a single cell is best for both sides (a saddle point), the
equilibrium is pure.

This replaces "rivals react at the historical rate" with "both teams play
optimally against each other" for the main rival. Other cars keep their own
strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .degradation import DegModel
from .hazards import TrackParams
from .simulate import Car, simulate
from .timing import Stop


@dataclass
class Duel:
    us: str
    them: str
    laps_us: list[int]
    laps_them: list[int]
    matrix: np.ndarray  # P(we finish ahead) for each (our lap, their lap)
    value: float  # game value: P(we finish ahead) when both play optimally
    x: np.ndarray  # our equilibrium strategy over laps_us
    y: np.ndarray  # their equilibrium strategy over laps_them
    saddle: tuple[int, int] | None  # (our lap, their lap) if the equilibrium is pure

    def strategy(self, who: str = "us", min_p: float = 0.02) -> list[tuple[int, float]]:
        laps, p = (self.laps_us, self.x) if who == "us" else (self.laps_them, self.y)
        return [(l, float(q)) for l, q in zip(laps, p) if q >= min_p]

    def at(self, lap_us: int, lap_them: int) -> float:
        return float(self.matrix[self.laps_us.index(lap_us), self.laps_them.index(lap_them)])

    def best_reply(self, lap_them: int) -> tuple[int, float]:
        """Our best lap if we knew the rival's lap."""
        col = self.matrix[:, self.laps_them.index(lap_them)]
        i = int(col.argmax())
        return self.laps_us[i], float(col[i])

    def summary(self) -> str:
        fmt = lambda s: ", ".join(f"L{l} {p:.0%}" for l, p in s)  # noqa: E731
        lines = [f"Pit duel {self.us} vs {self.them}: chance {self.us} finishes ahead"]
        if self.saddle:
            lines.append(f"  Pure equilibrium: {self.us} pits lap {self.saddle[0]}, {self.them} lap {self.saddle[1]}")
        else:
            lines.append(f"  Mixed equilibrium: {self.us} {fmt(self.strategy('us'))}")
            lines.append(f"                     {self.them} {fmt(self.strategy('them'))}")
        lines.append(f"  Game value: {self.us} ahead {self.value:.1%} with both sides playing optimally")
        return "\n".join(lines)


def solve_zero_sum(M: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Value and optimal mixed strategies of a zero-sum game (row player maximises)."""
    m, n = M.shape
    # Row player: variables (x_1..x_m, v); maximise v -> minimise -v.
    c = np.r_[np.zeros(m), -1.0]
    A_ub = np.c_[-M.T, np.ones(n)]  # v - (M^T x)_j <= 0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=np.r_[np.ones(m), 0.0][None, :], b_eq=[1.0],
                  bounds=[(0, None)] * m + [(None, None)], method="highs")
    x, v = res.x[:m], res.x[m]
    # Column player: variables (y_1..y_n, w); minimise w with (M y)_i <= w.
    c2 = np.r_[np.zeros(n), 1.0]
    A_ub2 = np.c_[M, -np.ones(m)]
    res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(m), A_eq=np.r_[np.ones(n), 0.0][None, :], b_eq=[1.0],
                   bounds=[(0, None)] * n + [(None, None)], method="highs")
    y = res2.x[:n]
    return float(v), np.clip(x, 0, 1), np.clip(y, 0, 1)


def pit_duel(
    cars: list[Car],
    us: int,
    them: int,
    start_lap: int,
    total_laps: int,
    model: DegModel,
    track: TrackParams,
    laps_us: list[int],
    laps_them: list[int],
    compound_us: str,
    compound_them: str,
    n: int = 800,
    seed: int = 11,
) -> Duel:
    """Payoff matrix and Nash equilibrium of a one-stop pit duel between cars[us] and cars[them]."""
    M = np.zeros((len(laps_us), len(laps_them)))
    for i, lu in enumerate(laps_us):
        for j, lt in enumerate(laps_them):
            field = list(cars)
            field[us] = replace(cars[us], stops=[Stop(lu, compound_us)])
            field[them] = replace(cars[them], stops=[Stop(lt, compound_them)])
            fin = simulate(field, start_lap, total_laps, model, track, n=n, seed=seed).finish
            M[i, j] = float((fin[:, us] < fin[:, them]).mean())
    v, x, y = solve_zero_sum(M)
    # Pure saddle point: a cell that is the minimum of its row and the maximum of its column.
    saddle = None
    for i in range(len(laps_us)):
        for j in range(len(laps_them)):
            if M[i, j] >= M[:, j].max() - 1e-9 and M[i, j] <= M[i, :].min() + 1e-9:
                saddle = (laps_us[i], laps_them[j])
    return Duel(cars[us].code, cars[them].code, list(laps_us), list(laps_them), M, v, x, y, saddle)


def table(duel: Duel) -> pd.DataFrame:
    return pd.DataFrame(duel.matrix, index=[f"{duel.us} L{l}" for l in duel.laps_us],
                        columns=[f"{duel.them} L{l}" for l in duel.laps_them])
