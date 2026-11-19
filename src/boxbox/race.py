"""Build one tidy row per driver per lap for a race, from OpenF1.

Each lap gets its tyre (compound, stint, age), whether it was an in-lap or
out-lap, whether the race was neutralised (safety car, VSC, red flag), and a
`clean` flag for laps that show true racing pace. Only clean laps are used to
learn tyre wear; the rest still matter for strategy (pit loss, safety cars).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import openf1

DRY = ("SOFT", "MEDIUM", "HARD")


@dataclass
class Neutralisation:
    kind: str  # "SC", "VSC" or "RED"
    start_lap: int
    end_lap: int


@dataclass
class Race:
    year: int
    name: str
    circuit: str
    session_key: int
    total_laps: int
    laps: pd.DataFrame
    pits: pd.DataFrame
    neutralisations: list[Neutralisation] = field(default_factory=list)
    drivers: pd.DataFrame = field(default_factory=pd.DataFrame)
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    wet: bool = False

    @property
    def clean(self) -> pd.DataFrame:
        return self.laps[self.laps["clean"]]

    def __repr__(self) -> str:  # short, readable
        n = ", ".join(f"{x.kind} L{x.start_lap}-{x.end_lap}" for x in self.neutralisations) or "none"
        return (
            f"Race({self.year} {self.name} @ {self.circuit}, {self.total_laps} laps, "
            f"{self.laps['driver'].nunique()} drivers, {int(self.laps['clean'].sum())} clean laps, "
            f"neutralised: {n}, wet: {self.wet})"
        )


def _neutralisations(race_control: list[dict], total_laps: int) -> list[Neutralisation]:
    """Turn race-control messages into SC / VSC / red-flag lap ranges."""
    out: list[Neutralisation] = []
    open_: dict[str, int] = {}
    for m in sorted(race_control, key=lambda m: m["date"]):
        msg = (m.get("message") or "").upper()
        lap = m.get("lap_number") or 1
        if m.get("flag") == "RED":
            # A red flag ends any running SC/VSC (the race restarts from the grid or behind the SC).
            for kind, start in list(open_.items()):
                out.append(Neutralisation(kind, start, lap))
                open_.pop(kind)
            if out and out[-1].kind == "RED" and out[-1].end_lap >= lap - 1:
                out[-1].end_lap = min(lap + 1, total_laps)  # repeated red-flag messages: merge
            else:
                out.append(Neutralisation("RED", lap, min(lap + 1, total_laps)))
            continue
        if m.get("category") != "SafetyCar":
            continue
        kind = "VSC" if "VIRTUAL" in msg else "SC"
        if "DEPLOYED" in msg and kind not in open_:
            open_[kind] = lap
        elif ("IN THIS LAP" in msg or "ENDING" in msg) and kind in open_:
            # The restart lap is not representative either, so include it.
            out.append(Neutralisation(kind, open_.pop(kind), min(lap + 1, total_laps)))
    for kind, start in open_.items():  # never ended (race finished behind it)
        out.append(Neutralisation(kind, start, total_laps))
    return sorted(out, key=lambda n: n.start_lap)


def load_race(year: int, query: str, refresh: bool = False) -> Race:
    session = openf1.find_race(year, query)
    key = session["session_key"]
    g = lambda ep: openf1.get(ep, refresh=refresh, session_key=key)  # noqa: E731

    laps = pd.DataFrame(g("laps"))
    stints = pd.DataFrame(g("stints"))
    pits = pd.DataFrame(g("pit"))
    drivers = pd.DataFrame(g("drivers"))
    results = pd.DataFrame(g("session_result"))
    weather = pd.DataFrame(g("weather"))
    rc = g("race_control")

    if laps.empty or stints.empty:
        raise RuntimeError(f"OpenF1 has no lap or tyre data for {year} {query} yet.")

    laps = laps.rename(columns={"driver_number": "driver", "lap_number": "lap", "lap_duration": "lap_time"})
    laps = laps[["driver", "lap", "lap_time", "is_pit_out_lap", "date_start"]].copy()
    laps["lap"] = laps["lap"].astype(int)
    total_laps = int(laps["lap"].max())

    # Tyre on every lap, from the stint table.
    stints = stints.dropna(subset=["lap_start"]).copy()
    stints["lap_end"] = stints["lap_end"].fillna(total_laps)
    rows = []
    for s in stints.itertuples():
        for lap in range(int(s.lap_start), int(s.lap_end) + 1):
            age0 = 0 if pd.isna(s.tyre_age_at_start) else int(s.tyre_age_at_start)
            rows.append((s.driver_number, lap, s.compound, int(s.stint_number), age0 + lap - int(s.lap_start) + 1))
    tyres = pd.DataFrame(rows, columns=["driver", "lap", "compound", "stint", "tyre_age"])
    tyres = tyres.drop_duplicates(["driver", "lap"], keep="last")
    laps = laps.merge(tyres, on=["driver", "lap"], how="left")

    # In-laps: OpenF1 records a pit stop on the lap the car entered the pit lane.
    laps["is_pit_in_lap"] = False
    if not pits.empty:
        in_laps = set(zip(pits["driver_number"], pits["lap_number"]))
        laps["is_pit_in_lap"] = [(d, l) in in_laps for d, l in zip(laps["driver"], laps["lap"])]
    laps["is_pit_out_lap"] = laps["is_pit_out_lap"].fillna(False).astype(bool)

    neutral = _neutralisations(rc, total_laps)
    neutral_laps = {l for n in neutral for l in range(n.start_lap, n.end_lap + 1)}
    laps["neutral"] = laps["lap"].isin(neutral_laps)

    wet = bool(not weather.empty and weather["rainfall"].fillna(0).gt(0).any()) or bool(
        laps["compound"].isin(["INTERMEDIATE", "WET"]).any()
    )

    # Clean = true racing pace on dry tyres.
    clean = (
        (laps["lap"] > 1)
        & ~laps["is_pit_in_lap"]
        & ~laps["is_pit_out_lap"]
        & ~laps["neutral"]
        & laps["lap_time"].notna()
        & laps["compound"].isin(DRY)
    )
    # Drop laps far off the pace (traffic, damage, spins): >107% of the race's typical clean lap.
    ref = laps.loc[clean, "lap_time"].quantile(0.2) if clean.any() else np.nan
    clean &= laps["lap_time"] <= 1.07 * ref
    laps["clean"] = clean

    if not drivers.empty:
        info = drivers[["driver_number", "name_acronym", "team_name", "team_colour"]].rename(
            columns={"driver_number": "driver", "name_acronym": "code", "team_name": "team", "team_colour": "colour"}
        )
        laps = laps.merge(info, on="driver", how="left")
        drivers = info

    pits_df = pd.DataFrame()
    if not pits.empty:
        pits_df = pits.rename(columns={"driver_number": "driver", "lap_number": "lap"})[
            ["driver", "lap", "lane_duration", "pit_duration"]
        ]

    laps = laps.sort_values(["driver", "lap"]).reset_index(drop=True)
    return Race(
        year=year,
        name=session["country_name"],
        circuit=session["circuit_short_name"],
        session_key=key,
        total_laps=total_laps,
        laps=laps,
        pits=pits_df,
        neutralisations=neutral,
        drivers=drivers,
        results=results,
        wet=wet,
    )
