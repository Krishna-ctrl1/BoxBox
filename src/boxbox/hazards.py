"""Random race events, learned per track from past races.

Safety cars and VSCs are modelled as a Poisson process: each green lap has a
small chance of one starting. Tracks with few races borrow strength from the
season-wide rate through a Gamma prior (the Gamma-Poisson model), so one lucky
or unlucky race cannot make a track look absurdly safe or dangerous.

Overtaking difficulty is the share of close battles (under 1 s) that turn into
a pass on the next lap, smoothed with a Beta prior (Beta-Binomial).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import degradation, openf1, pitloss
from .race import Race, load_race
from .timing import crossings

OUT = Path(__file__).resolve().parents[2] / "data"
CLOSE = 1.0  # seconds: a "battle"
PRIOR_LAPS = 120  # Gamma prior worth about two races of evidence
PRIOR_BATTLES = 60  # Beta prior worth about 60 battles
SC_PIT_FACTOR = 0.55


@dataclass
class TrackParams:
    circuit: str
    races: int
    sc_rate: float  # chance per green lap that a safety car starts
    vsc_rate: float
    pass_prob: float  # share of close battles that become a pass next lap
    pit_loss: float  # green-flag pit loss (s)
    sc_pit_factor: float  # pit loss under SC as a share of green pit loss
    sc_pace_factor: float  # SC lap time / green lap time
    vsc_pace_factor: float
    sc_laps: list[int] = field(default_factory=list)  # observed SC lengths (laps)
    lap_time: float = 90.0  # typical clean lap (s)
    dnf_rate: float = 0.0006  # chance per car per lap of retiring


def _battles(race: Race) -> tuple[int, int]:
    """(passes, opportunities): close pairs at the end of a green lap and whether they swapped."""
    C = crossings(race)
    L = race.laps
    bad = set(zip(L.loc[L["is_pit_in_lap"] | L["is_pit_out_lap"] | L["neutral"], "driver"],
                  L.loc[L["is_pit_in_lap"] | L["is_pit_out_lap"] | L["neutral"], "lap"]))
    by_lap = {lap: g.sort_values("cross") for lap, g in C.groupby("lap")}
    passes = opps = 0
    for lap in range(2, race.total_laps):
        if lap not in by_lap or lap + 1 not in by_lap:
            continue
        a, b = by_lap[lap], by_lap[lap + 1]
        order_next = {d: i for i, d in enumerate(b["driver"])}
        rows = a[["driver", "cross"]].to_numpy()
        for (d1, t1), (d2, t2) in zip(rows[:-1], rows[1:]):
            d1, d2 = int(d1), int(d2)
            if t2 - t1 > CLOSE or d1 not in order_next or d2 not in order_next:
                continue
            if any(x in bad for x in [(d1, lap), (d2, lap), (d1, lap + 1), (d2, lap + 1)]):
                continue
            opps += 1
            passes += order_next[d2] < order_next[d1]
    return passes, opps


def _race_stats(race: Race) -> dict:
    clean_med = race.clean["lap_time"].median()
    green_laps = race.total_laps - sum(n.end_lap - n.start_lap + 1 for n in race.neutralisations)
    neutral = race.laps[race.laps["neutral"] & ~race.laps["is_pit_in_lap"] & ~race.laps["is_pit_out_lap"]]
    pace = {}
    for kind in ("SC", "VSC"):
        # Only laps fully under SC/VSC: skip the deploy lap and the ending/restart laps.
        laps = {l for n in race.neutralisations if n.kind == kind for l in range(n.start_lap + 1, n.end_lap - 1)}
        t = neutral[neutral["lap"].isin(laps)]["lap_time"]
        pace[kind] = float(t.median() / clean_med) if len(t) >= 10 else np.nan
    p, o = _battles(race)
    m = degradation.fit(race.clean)
    pl = pitloss.estimate(race, m)
    dnf = int(race.results["dnf"].fillna(False).sum()) if not race.results.empty else 0
    return dict(
        circuit=race.circuit,
        laps=race.total_laps,
        green_laps=max(green_laps, 1),
        sc=sum(n.kind == "SC" and n.start_lap > 1 for n in race.neutralisations),  # lap-1 SCs are a different animal
        vsc=sum(n.kind == "VSC" for n in race.neutralisations),
        sc_lengths=[n.end_lap - n.start_lap for n in race.neutralisations if n.kind == "SC"],
        sc_pace=pace["SC"],
        vsc_pace=pace["VSC"],
        passes=p,
        battles=o,
        pit_loss=pl.green,
        sc_pit=pl.neutral,
        lap_time=float(clean_med),
        dnf=dnf,
        car_laps=int(race.laps["lap"].count()),
    )


def estimate(years: list[int], verbose: bool = True) -> dict[str, TrackParams]:
    rows = []
    for y in years:
        for s in openf1.season_races(y):
            try:
                race = load_race(y, s["circuit_short_name"])
                if race.wet:
                    continue
                rows.append(_race_stats(race))
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"  skip {y} {s['circuit_short_name']}: {e}")
    df = pd.DataFrame(rows)

    # Season-wide rates = the priors.
    sc_rate = df["sc"].sum() / df["green_laps"].sum()
    vsc_rate = df["vsc"].sum() / df["green_laps"].sum()
    pass_rate = df["passes"].sum() / max(df["battles"].sum(), 1)
    sc_pace = float(np.nanmedian(df["sc_pace"])) if df["sc_pace"].notna().any() else 1.4
    vsc_pace = float(np.nanmedian(df["vsc_pace"])) if df["vsc_pace"].notna().any() else np.nan
    if not (1.2 <= vsc_pace <= 1.6):  # too few full VSC laps to measure: use the usual ~35% slower
        vsc_pace = 1.35
    # ASSUMPTION: a stop under SC/VSC costs about 55% of a green-flag stop. Our direct
    # estimate from lap times is biased (cars are still closing up to the SC train),
    # so this is a stated assumption to calibrate in the strategy backtest.
    sc_pit_factor = SC_PIT_FACTOR
    dnf_rate = df["dnf"].sum() / df["car_laps"].sum()
    all_sc = [x for xs in df["sc_lengths"] for x in xs] or [4]

    out = {}
    for c, g in df.groupby("circuit"):
        green = g["green_laps"].sum()
        out[c] = TrackParams(
            circuit=c,
            races=len(g),
            sc_rate=float((sc_rate * PRIOR_LAPS + g["sc"].sum()) / (PRIOR_LAPS + green)),
            vsc_rate=float((vsc_rate * PRIOR_LAPS + g["vsc"].sum()) / (PRIOR_LAPS + green)),
            pass_prob=float((pass_rate * PRIOR_BATTLES + g["passes"].sum()) / (PRIOR_BATTLES + g["battles"].sum())),
            pit_loss=float(np.nanmedian(g["pit_loss"])) if g["pit_loss"].notna().any() else float(np.nanmedian(df["pit_loss"])),
            sc_pit_factor=sc_pit_factor,
            sc_pace_factor=float(np.nanmedian(g["sc_pace"])) if g["sc_pace"].notna().any() else sc_pace,
            vsc_pace_factor=vsc_pace,  # too few VSC laps per track to measure separately
            sc_laps=[int(x) for xs in g["sc_lengths"] for x in xs] or [int(x) for x in all_sc],
            lap_time=float(g["lap_time"].median()),
            dnf_rate=float(dnf_rate),
        )
    out["_default"] = TrackParams(
        circuit="_default", races=len(df), sc_rate=float(sc_rate), vsc_rate=float(vsc_rate), pass_prob=float(pass_rate),
        pit_loss=float(np.nanmedian(df["pit_loss"])), sc_pit_factor=sc_pit_factor, sc_pace_factor=sc_pace,
        vsc_pace_factor=vsc_pace, sc_laps=[int(x) for x in all_sc], lap_time=float(df["lap_time"].median()),
        dnf_rate=float(dnf_rate),
    )
    return out


def save(params: dict[str, TrackParams], name: str) -> Path:
    path = OUT / f"track_params_{name}.json"
    path.write_text(json.dumps({k: asdict(v) for k, v in params.items()}, indent=1))
    return path


def load(name: str) -> dict[str, TrackParams]:
    raw = json.loads((OUT / f"track_params_{name}.json").read_text())
    return {k: TrackParams(**v) for k, v in raw.items()}


def for_track(params: dict[str, TrackParams], circuit: str) -> TrackParams:
    return params.get(circuit, params["_default"])
