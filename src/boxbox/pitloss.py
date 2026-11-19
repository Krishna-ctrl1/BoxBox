"""How much time does a pit stop cost at this track?

For every stop we compare the real in-lap + out-lap with what the model says
those two laps would have taken without stopping. The difference is the pit
loss: pit-lane speed limit + the stop itself + slow tyre warm-up.

Stops made under a safety car are much cheaper (everyone else is slow too),
so green-flag and neutralised stops are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .degradation import DegModel
from .race import Race


@dataclass
class PitLoss:
    green: float  # median loss of a stop under green flag (s)
    green_spread: float  # interquartile range (s)
    neutral: float | None  # median loss when stopping under SC/VSC
    lane_time: float | None  # median time inside the pit lane, from OpenF1
    n_green: int
    n_neutral: int
    stops: pd.DataFrame

    def summary(self) -> str:
        if self.n_green:
            s = f"Pit loss: {self.green:.1f}s under green (IQR {self.green_spread:.1f}s, {self.n_green} stops)"
        else:
            s = "Pit loss: no green-flag stops in this race"
        if self.neutral is not None:
            s += f", {self.neutral:.1f}s under SC/VSC ({self.n_neutral} stops)"
        if self.lane_time is not None:
            s += f"; time in pit lane {self.lane_time:.1f}s"
        return s


def estimate(race: Race, model: DegModel) -> PitLoss:
    laps = race.laps.set_index(["driver", "lap"])
    rows = []
    for stop in race.pits.itertuples():
        d, in_lap = int(stop.driver), int(stop.lap)
        if (d, in_lap) not in laps.index or (d, in_lap + 1) not in laps.index:
            continue
        a, b = laps.loc[(d, in_lap)], laps.loc[(d, in_lap + 1)]
        if pd.isna(a.lap_time) or pd.isna(b.lap_time) or in_lap <= 1:
            continue
        if a.compound not in model.deg or b.compound not in model.deg:
            continue
        expected = model.predict([d, d], [in_lap, in_lap + 1], [a.compound, b.compound], [a.tyre_age, 1]).sum()
        if np.isnan(expected):
            continue
        rows.append(
            dict(
                driver=d,
                code=a.get("code"),
                lap=in_lap,
                actual=a.lap_time + b.lap_time,
                expected=expected,
                loss=a.lap_time + b.lap_time - expected,
                neutral=bool(a.neutral or b.neutral),
                lane=getattr(stop, "lane_duration", np.nan),
            )
        )
    stops = pd.DataFrame(rows, columns=["driver", "code", "lap", "actual", "expected", "loss", "neutral", "lane"])
    g = stops[~stops["neutral"]]["loss"]
    # Drop stops with problems (slow wheel, damage): far above the typical stop.
    g = g[g <= g.median() + 3 * (g.quantile(0.75) - g.quantile(0.25) + 0.5)]
    n = stops[stops["neutral"]]["loss"]
    lane = stops["lane"].dropna()
    return PitLoss(
        green=float(g.median()) if len(g) else float("nan"),  # nan: no green-flag stops (use a prior)
        green_spread=float(g.quantile(0.75) - g.quantile(0.25)) if len(g) else float("nan"),
        neutral=float(n.median()) if len(n) else None,
        lane_time=float(lane.median()) if len(lane) else None,
        n_green=int(len(g)),
        n_neutral=int(len(n)),
        stops=stops,
    )
