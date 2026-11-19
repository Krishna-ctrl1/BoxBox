"""Small cached client for the OpenF1 API (https://openf1.org).

OpenF1 is free, needs no login, and covers every session from 2023 on.
Every response is cached as JSON on disk, so a race is downloaded once and
then works offline. Requests are spaced out to stay inside the free rate limit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

BASE = "https://api.openf1.org/v1"
CACHE = Path(__file__).resolve().parents[2] / "data" / "raw"
_MIN_GAP = 0.45  # seconds between live requests
_last = 0.0


def _cache_path(endpoint: str, params: dict[str, Any]) -> Path:
    key = "&".join(f"{k}={params[k]}" for k in sorted(params))
    safe = key.replace("/", "_").replace(" ", "_") or "all"
    return CACHE / endpoint / f"{safe}.json"


def get(endpoint: str, refresh: bool = False, **params: Any) -> list[dict]:
    """GET /v1/{endpoint}?params, cached on disk."""
    global _last
    path = _cache_path(endpoint, params)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    for attempt in range(6):
        wait = _MIN_GAP - (time.monotonic() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 404:
            data: list[dict] = []
            break
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):  # OpenF1 returns {"detail": ...} for "no results"
            data = []
        break
    else:
        raise RuntimeError(f"OpenF1 kept failing for {endpoint} {params}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def find_race(year: int, query: str) -> dict:
    """Find a Grand Prix race session by country, circuit or location name."""
    q = query.strip().lower()
    sessions = get("sessions", year=year, session_name="Race")
    hits = [
        s
        for s in sessions
        if q in (s.get("country_name") or "").lower()
        or q in (s.get("circuit_short_name") or "").lower()
        or q in (s.get("location") or "").lower()
    ]
    if not hits:
        names = sorted({f"{s['country_name']} ({s['circuit_short_name']})" for s in sessions})
        raise ValueError(f"No {year} race matches '{query}'. Options: {', '.join(names)}")
    if len(hits) > 1:
        # e.g. "united states" matches Austin, Miami and Las Vegas
        opts = ", ".join(f"{h['circuit_short_name']}" for h in hits)
        raise ValueError(f"'{query}' matches several {year} races: {opts}. Use the circuit name.")
    return hits[0]


def season_races(year: int) -> list[dict]:
    """All Grand Prix race sessions in a season (sprints excluded), in date order."""
    sessions = get("sessions", year=year, session_name="Race")
    races = [s for s in sessions if s.get("session_type") == "Race" and not s.get("is_cancelled")]
    return sorted(races, key=lambda s: s["date_start"])
