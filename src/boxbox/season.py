"""Fit every race in a season and learn priors from all of them.

The output is a track-by-track table (tyre wear, pit loss, safety cars) plus a
season prior for each compound's wear rate. The prior is what lets a single
race with thin data still give a sensible answer (see degradation.shrink).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import degradation, openf1, pitloss
from .race import load_race

OUT = Path(__file__).resolve().parents[2] / "data"


def fit_season(year: int, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for s in openf1.season_races(year):
        name = s["circuit_short_name"]
        try:
            race = load_race(year, name)
        except Exception as e:  # missing data for a race should not stop the season
            if verbose:
                print(f"  skip {name}: {e}")
            continue
        row = dict(
            year=year,
            circuit=race.circuit,
            country=race.name,
            session_key=race.session_key,
            total_laps=race.total_laps,
            wet=race.wet,
            sc=sum(n.kind == "SC" for n in race.neutralisations),
            vsc=sum(n.kind == "VSC" for n in race.neutralisations),
            red=sum(n.kind == "RED" for n in race.neutralisations),
            clean_laps=int(race.laps["clean"].sum()),
        )
        try:
            m = degradation.fit(race.clean)
            pl = pitloss.estimate(race, m)
            row.update(r2=m.r2, sigma=m.sigma, fuel=m.fuel_per_lap, pit_loss=pl.green, pit_loss_n=pl.n_green)
            for c in m.compounds:
                row[f"deg_{c}"] = m.deg[c]
                row[f"se_{c}"] = m.deg_se[c]
                row[f"off_{c}"] = m.compound_offset[c]
        except Exception as e:
            if verbose:
                print(f"  {name}: no model ({e})")
        rows.append(row)
        if verbose:
            print(f"  {race}")
    return pd.DataFrame(rows)


def priors(table: pd.DataFrame, max_se: float = 0.02) -> dict[str, tuple[float, float]]:
    """Season prior per compound: typical wear rate and how much it varies by track.

    Only well-measured races (small standard error) and dry races are used.
    """
    out = {}
    dry = table[~table["wet"]] if "wet" in table else table
    for c in degradation.COMPOUND_ORDER:
        if f"deg_{c}" not in dry:
            continue
        good = dry[(dry[f"se_{c}"] < max_se)][f"deg_{c}"].dropna()
        if len(good) >= 3:
            spread = 1.4826 * np.median(np.abs(good - good.median()))  # robust std across tracks
            out[c] = (float(good.median()), float(max(spread, 0.01)))
    return out


def save(year: int, table: pd.DataFrame) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"season_{year}.json"
    payload = {"year": year, "priors": priors(table), "races": json.loads(table.to_json(orient="records"))}
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_priors(year: int) -> dict[str, tuple[float, float]]:
    data = json.loads((OUT / f"season_{year}.json").read_text())
    return {k: tuple(v) for k, v in data["priors"].items()}
