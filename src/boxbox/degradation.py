"""Lap-time model: how fast is a car on a given lap, on a given tyre?

    lap_time = driver_base                    (car + driver pace)
             + fuel_per_lap * lap             (car gets lighter, track grips up: usually negative)
             + compound_offset[c]             (soft is faster than hard when new)
             + deg[c] * tyre_age              (each extra lap on a tyre costs time)
             + deg2[c] * tyre_age**2          (optional: wear that speeds up near the "cliff")
             + noise                          (traffic, mistakes, wind)

Fitted by least squares on clean laps, with a robust second pass that drops
outlier laps. The standard errors tell us how sure we are about each wear rate,
which the race simulator will use later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD"]


@dataclass
class DegModel:
    driver_base: dict[int, float]
    fuel_per_lap: float
    compound_offset: dict[str, float]  # relative to the reference compound (0.0)
    deg: dict[str, float]  # seconds lost per lap of tyre age
    deg_se: dict[str, float]  # standard error of each wear rate
    deg_raw: dict[str, float]  # this race's estimate before shrinking toward the prior
    deg2: dict[str, float]
    sigma: float  # residual standard deviation (s)
    r2: float
    n_laps: int
    n_dropped: int
    reference: str
    quadratic: bool

    @property
    def compounds(self) -> list[str]:
        return [c for c in COMPOUND_ORDER if c in self.deg]

    def predict(self, driver, lap, compound, tyre_age) -> np.ndarray:
        """Predicted lap time; arguments may be scalars or arrays."""
        driver, lap, compound, tyre_age = (np.atleast_1d(x) for x in (driver, lap, compound, tyre_age))
        base = np.array([self.driver_base.get(int(d), np.nan) for d in driver], dtype=float)
        off = np.array([self.compound_offset.get(c, np.nan) for c in compound], dtype=float)
        k1 = np.array([self.deg.get(c, np.nan) for c in compound], dtype=float)
        k2 = np.array([self.deg2.get(c, 0.0) for c in compound], dtype=float)
        age = tyre_age.astype(float)
        return base + self.fuel_per_lap * lap.astype(float) + off + k1 * age + k2 * age**2

    def summary(self, codes: dict[int, str] | None = None) -> str:
        lines = [
            f"Lap-time model: {self.n_laps} clean laps ({self.n_dropped} outliers dropped), "
            f"R² {self.r2:.3f}, noise ±{self.sigma:.3f}s",
            f"  Fuel + track effect: {self.fuel_per_lap:+.4f} s per lap",
        ]
        for c in self.compounds:
            q = f", curve {self.deg2[c]:+.5f}·age²" if self.quadratic else ""
            raw = f" (race alone: {self.deg_raw[c]:+.4f})" if abs(self.deg_raw[c] - self.deg[c]) > 5e-4 else ""
            lines.append(
                f"  {c:<6} pace vs {self.reference.lower()}: {self.compound_offset[c]:+.3f}s | "
                f"wear {self.deg[c]:+.4f} ± {self.deg_se[c]:.4f} s/lap{raw}{q}"
            )
        if codes:
            best = sorted(self.driver_base.items(), key=lambda kv: kv[1])[:5]
            lines.append("  Fastest base pace: " + ", ".join(f"{codes.get(d, d)} {v:.2f}s" for d, v in best))
        return "\n".join(lines)


def _design(df: pd.DataFrame, compounds: list[str], reference: str, quadratic: bool) -> pd.DataFrame:
    X = pd.get_dummies(df["driver"].astype(int), prefix="drv", dtype=float)
    X["lap"] = df["lap"].astype(float)
    for c in compounds:
        is_c = (df["compound"] == c).astype(float)
        if c != reference:
            X[f"off_{c}"] = is_c
        X[f"deg_{c}"] = is_c * df["tyre_age"].astype(float)
        if quadratic:
            X[f"deg2_{c}"] = is_c * df["tyre_age"].astype(float) ** 2
    return X


def _solve(X: pd.DataFrame, y: np.ndarray, priors: dict[str, tuple[float, float]], sigma: float) -> dict:
    """Bayesian linear regression with Gaussian priors on some coefficients.

    Each prior N(mean, sd) on coefficient j is added as one pseudo-observation:
    a row that is (sigma/sd) in column j and (sigma/sd)*mean on the right-hand
    side. Least squares on the augmented system gives exactly the posterior
    mean, and every coefficient stays consistent with every other one.
    """
    cols = list(X.columns)
    A, b = [X.to_numpy(dtype=float)], [y.astype(float)]
    for name, (mean, sd) in priors.items():
        if name in cols:
            row = np.zeros(len(cols))
            row[cols.index(name)] = sigma / sd
            A.append(row[None, :])
            b.append(np.array([mean * sigma / sd]))
    A_aug, b_aug = np.vstack(A), np.concatenate(b)
    beta, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    fitted = X.to_numpy(dtype=float) @ beta
    resid = y - fitted
    dof = max(len(y) - len(cols), 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(A_aug.T @ A_aug)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(
        params=pd.Series(beta, index=cols),
        se=pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=cols),
        sigma=np.sqrt(s2),
        r2=1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0,
    )


def shrink(estimate: float, se: float, prior_mean: float, prior_sd: float) -> tuple[float, float]:
    """Empirical-Bayes shrinkage: blend a noisy race estimate with a season prior.

    Each is weighted by its precision (1 / variance). A wear rate measured from
    only a handful of laps has a big standard error, so it moves most of the way
    to the prior; a well-measured one barely moves.
    """
    w_data, w_prior = 1.0 / se**2, 1.0 / prior_sd**2
    mean = (w_data * estimate + w_prior * prior_mean) / (w_data + w_prior)
    return mean, (w_data + w_prior) ** -0.5


def fit(
    clean_laps: pd.DataFrame,
    quadratic: bool = False,
    min_laps_per_compound: int = 15,
    prior: dict[str, tuple[float, float]] | None = None,
) -> DegModel:
    """Fit the lap-time model on clean laps (see Race.clean).

    prior: optional {compound: (mean wear s/lap, spread)} learned across a
    season (see boxbox.season). When given, each compound's wear rate is
    shrunk toward it, which fixes impossible results from thin data.
    """
    df = clean_laps.dropna(subset=["lap_time", "compound", "tyre_age"]).copy()
    counts = df["compound"].value_counts()
    compounds = [c for c in COMPOUND_ORDER if counts.get(c, 0) >= min_laps_per_compound]
    if not compounds:
        raise ValueError("Not enough clean dry laps to fit a model.")
    df = df[df["compound"].isin(compounds)]
    reference = "MEDIUM" if "MEDIUM" in compounds else compounds[0]

    X_all = _design(df, compounds, reference, quadratic)
    y_all = df["lap_time"].to_numpy()

    # Weak "common sense" priors keep the fit solvable even when the data cannot
    # separate driver pace from tyre pace (e.g. Monaco 2024: one stint each).
    weak = {"lap": (-0.06, 0.1)}
    for c in compounds:
        weak[f"deg_{c}"] = (0.06, 0.1)
        if c != reference:
            weak[f"off_{c}"] = (0.0, 1.5)
    season = dict(weak)
    for c, (m, s) in (prior or {}).items():
        if f"deg_{c}" in season:
            season[f"deg_{c}"] = (m, s)

    keep = np.ones(len(df), dtype=bool)
    sigma = 0.5
    for _ in range(4):  # robust refit: drop laps far from the fit, then refit
        sol = _solve(X_all[keep], y_all[keep], season, sigma)
        resid = y_all - X_all.to_numpy() @ sol["params"].to_numpy()
        mad = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        sigma = max(float(np.std(resid[keep])), 0.05)
        new_keep = np.abs(resid) <= 3.5 * max(mad, 1e-3)
        if (new_keep == keep).all():
            break
        keep = new_keep
    sol = _solve(X_all[keep], y_all[keep], season, sigma)
    raw = _solve(X_all[keep], y_all[keep], weak, sigma) if prior else sol  # this race on its own

    p, se = sol["params"], sol["se"]
    drivers = {int(k[4:]): float(v) for k, v in p.items() if k.startswith("drv_")}
    deg = {c: float(p[f"deg_{c}"]) for c in compounds}
    deg_se = {c: float(se[f"deg_{c}"]) for c in compounds}
    deg_raw = {c: float(raw["params"][f"deg_{c}"]) for c in compounds}
    res = sol
    return DegModel(
        driver_base=drivers,
        fuel_per_lap=float(p["lap"]),
        compound_offset={c: (0.0 if c == reference else float(p[f"off_{c}"])) for c in compounds},
        deg=deg,
        deg_se=deg_se,
        deg_raw=deg_raw,
        deg2={c: (float(p[f"deg2_{c}"]) if quadratic else 0.0) for c in compounds},
        sigma=float(res["sigma"]),
        r2=float(res["r2"]),
        n_laps=int(keep.sum()),
        n_dropped=int((~keep).sum()),
        reference=reference,
        quadratic=quadratic,
    )
