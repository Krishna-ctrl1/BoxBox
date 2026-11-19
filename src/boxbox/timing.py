"""Running order and gaps at the end of any lap, rebuilt from lap timestamps.

A car's crossing time at the end of lap L is when lap L started plus how long
it took. Sorting cars by laps completed, then by crossing time, gives the
running order; the difference to the leader's crossing time gives the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .race import Race


def crossings(race: Race) -> pd.DataFrame:
    """One row per driver per completed lap with the absolute crossing time (s)."""
    L = race.laps.dropna(subset=["lap_time"]).copy()
    t0 = pd.to_datetime(L["date_start"], utc=True, format="ISO8601")
    L["cross"] = (t0 - t0.min()).dt.total_seconds() + L["lap_time"]
    return L[["driver", "lap", "cross", "compound", "tyre_age", "stint"]]


def state_at(race: Race, lap: int) -> pd.DataFrame:
    """Running order at the end of `lap`: position, gap to leader, current tyre."""
    C = crossings(race)
    last = C[C["lap"] <= lap].sort_values("lap").groupby("driver").tail(1)
    last = last.sort_values(["lap", "cross"], ascending=[False, True]).reset_index(drop=True)
    if last.empty:  # e.g. Monaco 2024: red flag on lap 1, nobody completed it
        return pd.DataFrame(columns=["position", "driver", "lap", "gap", "compound", "tyre_age", "stint", "laps_down"])
    leader = last.iloc[0]
    # Gap to leader: same-lap cars by crossing time; lapped cars get a lap-time penalty per lap down.
    lap_len = race.clean["lap_time"].median() if len(race.clean) else 90.0
    last["laps_down"] = leader["lap"] - last["lap"]
    last["gap"] = np.where(
        last["laps_down"] == 0,
        last["cross"] - leader["cross"],
        last["cross"] - leader["cross"] + 0.0,  # crossing earlier lap later in time: still the right total gap
    )
    # A lapped car crossed lap L-k at time t; its gap to the leader is t + k*lap_len - leader time (approx).
    last.loc[last["laps_down"] > 0, "gap"] = (
        last["cross"] + last["laps_down"] * lap_len - leader["cross"]
    )
    last = last.sort_values("gap").reset_index(drop=True)
    last["position"] = np.arange(1, len(last) + 1)
    return last[["position", "driver", "lap", "gap", "compound", "tyre_age", "stint", "laps_down"]]


@dataclass
class Stop:
    lap: int  # in-lap: the car pits at the end of this lap
    compound: str  # tyre fitted


def strategy_after(race: Race, driver: int, lap: int) -> list[Stop]:
    """The pit stops this driver actually made after `lap` (what really happened)."""
    L = race.laps[race.laps["driver"] == driver].sort_values("lap")
    stops = []
    for row in L[L["is_pit_in_lap"] & (L["lap"] > lap)].itertuples():
        nxt = L[L["lap"] == row.lap + 1]
        if not nxt.empty and isinstance(nxt.iloc[0]["compound"], str):
            stops.append(Stop(int(row.lap), nxt.iloc[0]["compound"]))
    return stops


def finishing_order(race: Race) -> pd.DataFrame:
    """Official classification (position, driver, DNF) from OpenF1 results."""
    R = race.results.rename(columns={"driver_number": "driver"})
    return R[["position", "driver", "dnf", "dns", "dsq"]].copy()
