"""Monte Carlo race simulator.

Starting from the real state of a race at some lap, simulate the rest of the
race thousands of times at once. Each simulated race gets its own random draws:

  - lap-time noise for every car (the model's residual spread)
  - tyre-wear rates drawn from the model's uncertainty (parameter risk)
  - safety cars and VSCs as a Poisson process with the track's rates
  - close battles: the faster car behind only gets past with the track's pass chance
  - retirements (DNFs) at the field's per-lap rate

Everything is vectorised: arrays are (simulations x cars), so 2,000 races of
20 cars take well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .degradation import DegModel
from .hazards import TrackParams
from .timing import Stop

FOLLOW_GAP = 0.35  # s: how close a car can follow when it cannot pass
SC_TRAIN_GAP = 0.8  # s between cars after the field bunches up behind the SC
PASS_BOOST = 2.0  # pass chance when the car behind is genuinely faster, vs any close battle
# Covering an undercut, measured from 2023 dry races: when a car pits under green, the car
# directly ahead within 3 s pits in the next 2 laps 31% of the time vs 11% normally (a 22%
# uplift); the car directly behind 21% (11% uplift).
COVER_AHEAD = 0.22
COVER_BEHIND = 0.11
COVER_GAP = 3.0  # s
COVER_HORIZON = 12  # only a stop already planned within this many laps is brought forward
TYPICAL_OFFSET = {"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.45}  # used only for compounds a race hasn't shown yet
TYPICAL_DEG = {"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.045}


@dataclass
class Car:
    driver: int
    code: str
    gap: float  # seconds behind the leader at the start lap
    compound: str
    tyre_age: int
    stops: list[Stop] = field(default_factory=list)  # planned stops for the rest of the race


@dataclass
class SimResult:
    codes: list[str]
    drivers: list[int]
    finish: np.ndarray  # (n_sims, n_cars) finishing position, 1-based; DNF = n_cars
    race_time: np.ndarray  # (n_sims, n_cars) seconds from start lap; inf = DNF
    sc_count: np.ndarray  # (n_sims,) safety cars per simulated race
    features: dict | None = None  # per-lap state of one recorded car (see simulate(record=...))

    @property
    def n(self) -> int:
        return self.finish.shape[0]

    def table(self) -> pd.DataFrame:
        f = self.finish
        k = f.shape[1]
        dnf = np.isinf(self.race_time)
        return (
            pd.DataFrame(
                {
                    "code": self.codes,
                    "driver": self.drivers,
                    "exp_pos": f.mean(axis=0),
                    "p_win": (f == 1).mean(axis=0),
                    "p_podium": (f <= 3).mean(axis=0),
                    "p_points": (f <= 10).mean(axis=0),
                    "p_dnf": dnf.mean(axis=0),
                    "pos_p10": np.percentile(f, 10, axis=0),
                    "pos_p90": np.percentile(f, 90, axis=0),
                }
            )
            .sort_values("exp_pos")
            .reset_index(drop=True)
            .assign(rank=lambda d: np.arange(1, len(d) + 1))
            .loc[:, lambda d: ["rank"] + [c for c in d.columns if c != "rank"]]
            .head(k)
        )


def _compound_params(model: DegModel, compounds: list[str]):
    """Offset and wear for every compound, filling unseen ones with typical values."""
    ref_typ = TYPICAL_OFFSET.get(model.reference, 0.0)
    off, deg, se = {}, {}, {}
    for c in compounds:
        if c in model.deg:
            off[c], deg[c], se[c] = model.compound_offset[c], model.deg[c], model.deg_se[c]
        else:
            off[c] = TYPICAL_OFFSET.get(c, 0.0) - ref_typ
            deg[c], se[c] = TYPICAL_DEG.get(c, 0.06), 0.03
    return off, deg, se


def simulate(
    cars: list[Car],
    start_lap: int,
    total_laps: int,
    model: DegModel,
    track: TrackParams,
    n: int = 2000,
    seed: int | None = 0,
    events: bool = True,
    param_uncertainty: bool = True,
    record: int | None = None,
    respond_to: int | None = None,
) -> SimResult:
    """Simulate the rest of a race n times.

    record: index of one car whose state is saved at the end of every lap, before
    any pit stop that lap: what a strategist could see when deciding "box or stay
    out". Used by the pit engine (strategy.py).

    The random draws are made in the same order whatever the cars' strategies,
    so two calls with the same seed share every safety car, lap-time wobble and
    retirement (common random numbers): differences between strategies are then
    due to the strategy, not to luck.
    """
    rng = np.random.default_rng(seed)
    k = len(cars)
    # respond_to: index of the car whose green-flag stops rivals may cover (see COVER_*).
    compounds = sorted({c.compound for c in cars} | {s.compound for c in cars for s in c.stops} | {"SOFT", "MEDIUM", "HARD"})
    cidx = {c: i for i, c in enumerate(compounds)}
    off, deg, se = _compound_params(model, compounds)
    off_v = np.array([off[c] for c in compounds])

    # Tyre wear per simulation: drawn once per race from the model's uncertainty.
    deg_mean = np.array([deg[c] for c in compounds])
    deg_sd = np.array([se[c] for c in compounds]) if param_uncertainty else np.zeros(len(compounds))
    deg_sims = rng.normal(deg_mean, deg_sd, size=(n, len(compounds))).clip(min=0.0)

    median_base = np.median(list(model.driver_base.values()))
    base = np.array([model.driver_base.get(c.driver, median_base + 0.5) for c in cars])
    car_wear = np.array([model.car_wear(c.driver) for c in cars])  # team-specific extra wear (0 if off)
    t = np.tile(np.array([c.gap for c in cars], dtype=float), (n, 1))  # cumulative time
    comp = np.tile(np.array([cidx[c.compound] for c in cars]), (n, 1))
    age = np.tile(np.array([c.tyre_age for c in cars], dtype=float), (n, 1))
    alive = np.ones((n, k), dtype=bool)
    # Planned stops per simulated race (a rival covering us changes its own plan in that race only).
    max_s = max(1, max(len(c.stops) for c in cars))
    stop_lap = np.full((n, k, max_s), np.inf)
    stop_comp = np.zeros((n, k, max_s), dtype=int)
    for i, c in enumerate(cars):
        for j, st in enumerate(sorted(c.stops, key=lambda x: x.lap)):
            stop_lap[:, i, j] = st.lap
            stop_comp[:, i, j] = cidx[st.compound]

    sc_left = np.zeros(n, dtype=int)
    vsc_left = np.zeros(n, dtype=int)
    sc_count = np.zeros(n, dtype=int)
    sc_lengths = np.array(track.sc_laps or [4])
    rows = np.arange(n)[:, None]
    laps_range = list(range(start_lap + 1, total_laps + 1))
    if record is not None:
        L = len(laps_range)
        feats = {k_: np.zeros((n, L)) for k_ in ("sc", "vsc", "pos", "gap_ahead", "gap_behind", "tyre_age", "alive")}

    for li, lap in enumerate(laps_range):
        # 1. Random events start at the beginning of a green lap.
        if events:
            green = (sc_left == 0) & (vsc_left == 0)
            u = rng.random(n)
            new_sc = green & (u < track.sc_rate)
            new_vsc = green & ~new_sc & (u < track.sc_rate + track.vsc_rate)
            sc_left[new_sc] = rng.choice(sc_lengths, size=new_sc.sum()).clip(min=1)
            vsc_left[new_vsc] = rng.integers(1, 4, size=new_vsc.sum())
            sc_count += new_sc
        under_sc, under_vsc = sc_left > 0, vsc_left > 0
        neutral = under_sc | under_vsc

        # 2. Racing lap time for every car from the lap-time model, plus noise.
        age += 1
        lap_t = (
            base[None, :]
            + model.fuel_per_lap * lap
            + off_v[comp]
            + (deg_sims[rows, comp] + car_wear[None, :]) * age
            + rng.normal(0.0, model.sigma, size=(n, k))
        )
        # Under SC/VSC everyone runs a slow, fixed pace (tyres barely wear).
        slow = np.where(under_sc, track.lap_time * track.sc_pace_factor, track.lap_time * track.vsc_pace_factor)
        lap_t = np.where(neutral[:, None], slow[:, None] + rng.normal(0, 0.3, size=(n, k)), lap_t)
        age = np.where(neutral[:, None], age - 0.5, age)  # half wear behind the SC
        new_t = t + lap_t

        # 3. Pit stops at the end of this lap (cheaper under SC/VSC).
        pitting = np.zeros((n, k), dtype=bool)
        loss = np.where(under_sc, track.pit_loss * track.sc_pit_factor,
                        np.where(under_vsc, track.pit_loss * track.vsc_pit_factor, track.pit_loss))
        for j in range(max_s):
            m = stop_lap[:, :, j] == lap
            pitting |= m
            comp = np.where(m, stop_comp[:, :, j], comp)
        new_t += pitting * loss[:, None]
        age = np.where(pitting, 0, age)

        # 3b. Rivals cover: when the responding-to car pits under green, the car just
        #     ahead (and, less often, just behind) within COVER_GAP may bring its own next
        #     stop forward to the next lap, as real teams do (rates measured from 2023).
        u_ahead, u_behind = rng.random(n), rng.random(n)  # drawn every lap: keeps common random numbers
        if respond_to is not None:
            f = respond_to
            trig = pitting[:, f] & ~neutral & alive[:, f]
            if trig.any():
                prev = np.where(alive, t, np.inf)
                my = prev[:, f]
                others = prev.copy()
                others[:, f] = np.inf
                with np.errstate(invalid="ignore"):  # retired cars are inf
                    ahead_gap = np.where(others <= my[:, None], my[:, None] - others, np.inf)
                    behind_gap = np.where(others > my[:, None], others - my[:, None], np.inf)
                for gaps, u, p_cover in ((ahead_gap, u_ahead, COVER_AHEAD), (behind_gap, u_behind, COVER_BEHIND)):
                    rival = gaps.argmin(axis=1)
                    close = trig & (gaps[np.arange(n), rival] <= COVER_GAP) & (u < p_cover)
                    if not close.any():
                        continue
                    idx = np.nonzero(close)[0]
                    r = rival[idx]
                    laps_r = stop_lap[idx, r, :]  # (m, max_s)
                    pending = (laps_r > lap + 1) & np.isfinite(laps_r) & (laps_r <= lap + COVER_HORIZON)
                    has = pending.any(axis=1)
                    j = pending.argmax(axis=1)
                    stop_lap[idx[has], r[has], j[has]] = lap + 1

        # 4. Battles: in running order, a car that would pass the one ahead only does so
        #    with the track's pass chance; otherwise it is held just behind.
        if not np.all(neutral):
            order = np.argsort(np.where(alive, t, np.inf), axis=1)
            p_pass = min(0.95, track.pass_prob * PASS_BOOST)
            for p in range(1, k):
                front, back = order[:, p - 1], order[:, p]
                tf, tb = new_t[rows[:, 0], front], new_t[rows[:, 0], back]
                ok = alive[rows[:, 0], front] & alive[rows[:, 0], back] & ~neutral
                ok &= ~pitting[rows[:, 0], front] & ~pitting[rows[:, 0], back]
                catching = ok & (tb < tf + FOLLOW_GAP)
                passes = catching & (tb < tf) & (rng.random(n) < p_pass)
                held = catching & ~passes
                new_t[rows[held, 0], back[held]] = tf[held] + FOLLOW_GAP

        # 5. Behind the safety car the field bunches up into a train.
        if under_sc.any():
            order = np.argsort(np.where(alive, new_t, np.inf), axis=1)
            lead_t = new_t[rows[:, 0], order[:, 0]]
            for p in range(1, k):
                car = order[:, p]
                m = under_sc & alive[np.arange(n), car]
                # Everyone closes up to the car ahead: the gap becomes a fixed train gap.
                new_t[np.arange(n)[m], car[m]] = lead_t[m] + p * SC_TRAIN_GAP

        # Strategist's view at the end of this lap (before the recorded car's own stop is applied).
        if record is not None:
            own_t = new_t[:, record] - pitting[:, record] * loss
            others = np.where(alive, new_t, np.inf)
            others[:, record] = np.inf
            with np.errstate(invalid="ignore"):  # retired cars are inf: inf - inf is fine to drop
                ahead = np.where(others <= own_t[:, None], own_t[:, None] - others, np.inf).min(axis=1)
                behind = np.where(others > own_t[:, None], others - own_t[:, None], np.inf).min(axis=1)
            feats["sc"][:, li] = under_sc
            feats["vsc"][:, li] = under_vsc
            feats["pos"][:, li] = 1 + (others <= own_t[:, None]).sum(axis=1)
            feats["gap_ahead"][:, li] = np.minimum(ahead, 30.0)
            feats["gap_behind"][:, li] = np.minimum(behind, 30.0)
            feats["tyre_age"][:, li] = np.where(pitting[:, record], 0, age[:, record])
            feats["alive"][:, li] = alive[:, record]

        # 6. Retirements.
        if events:
            dnf = alive & (rng.random((n, k)) < track.dnf_rate)
            alive &= ~dnf

        t = np.where(alive, new_t, np.inf)
        sc_left = np.maximum(sc_left - 1, 0)
        vsc_left = np.maximum(vsc_left - 1, 0)

    # Finishing position: order by total time; retired cars share last places.
    order = np.argsort(t, axis=1, kind="stable")
    finish = np.empty_like(order)
    finish[rows, order] = np.arange(1, k + 1)[None, :]
    if record is not None:
        feats["laps"] = np.array(laps_range)
    return SimResult(
        codes=[c.code for c in cars], drivers=[c.driver for c in cars], finish=finish, race_time=t, sc_count=sc_count,
        features=feats if record is not None else None,
    )
