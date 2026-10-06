"""Pairwise state LMP correlation with signed lag-aware (directional) metrics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

from experiments.lib.state_lmp_loader import (
    STATE_CENTROIDS,
    load_state_prices,
)
from experiments.lib.state_lmp_registry import DEFAULT_CACHE, load_registry

SeasonMode = Literal["winter", "summer", "combined", "annual"]

MetricName = Literal[
    "corr_lag0",
    "phase_lag",
    "anti_corr_lag",
    "corr_at_anti_lag",
]


@dataclass
class StatePriceSeries:
    state: str
    prices: np.ndarray
    lon: float
    lat: float
    iso: str
    hub: str


def cross_correlation_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 12) -> dict[int, float]:
    """Cross-correlation over integer hour lags (positive lag → b delayed vs a)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return {lag: 0.0 for lag in range(-max_lag, max_lag + 1)}
    out: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa, bb = a[lag:], b[: n - lag]
        else:
            aa, bb = a[: n + lag], b[-lag:]
        if len(aa) < 2:
            out[lag] = 0.0
            continue
        out[lag] = float(np.dot(aa, bb) / denom)
    return out


@dataclass(frozen=True)
class DirectedPairMetrics:
    """Price-only directed relationship from reference A to partner B."""

    corr_lag0: float
    phase_lag: float
    anti_corr_lag: int
    corr_at_anti_lag: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "corr_lag0": round(self.corr_lag0, 6),
            "phase_lag": round(self.phase_lag, 4),
            "anti_corr_lag": int(self.anti_corr_lag),
            "corr_at_anti_lag": round(self.corr_at_anti_lag, 6),
        }


def _load_all_state_prices(
    season: SeasonMode,
    *,
    cache_dir=None,
    conus_only: bool = False,
) -> list[StatePriceSeries]:
    cache_dir = cache_dir or DEFAULT_CACHE
    reg = load_registry()
    out: list[StatePriceSeries] = []

    for state, info in sorted(reg.get("states", {}).items()):
        if conus_only and state in ("AK", "HI"):
            continue
        if not info.get("iso") or not info.get("hub"):
            continue
        try:
            if season == "combined":
                w = load_state_prices(state, "winter", cache_dir=cache_dir)
                s = load_state_prices(state, "summer", cache_dir=cache_dir)
                prices = np.concatenate([w, s])
            else:
                prices = load_state_prices(state, season, cache_dir=cache_dir)
        except Exception as exc:
            print(f"[warn] skip {state}: {exc}")
            continue
        lon, lat = STATE_CENTROIDS.get(state, (-98.0, 39.0))
        out.append(
            StatePriceSeries(
                state=state,
                prices=prices,
                lon=lon,
                lat=lat,
                iso=str(info["iso"]),
                hub=str(info["hub"]),
            )
        )
    return out


def pearson_corr(a: np.ndarray, b: np.ndarray, *, lag: int = 0) -> float:
    """Pearson r with optional integer-hour shift applied to b (positive lag → b delayed)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a = a[:n]
    b = b[:n]
    if lag > 0:
        a, b = a[lag:], b[: n - lag]
    elif lag < 0:
        a, b = a[: n + lag], b[-lag:]
    if len(a) < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def price_phase_lag(a: np.ndarray, b: np.ndarray, *, max_lag: int = 12) -> float:
    """
    Signed hour-offset of b relative to a, from the cross-correlation function.

    phase_lag = Σ lag·r(lag) / Σ|r(lag)|.
    Positive ⇒ b's price curve is delayed vs a; negative ⇒ b leads a.
    """
    corrs = cross_correlation_lag(a, b, max_lag=max_lag)
    lags = np.array(sorted(corrs.keys()), dtype=float)
    vals = np.array([corrs[int(k)] for k in lags], dtype=float)
    denom = float(np.sum(np.abs(vals)))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(lags, vals) / denom)


def optimal_anti_corr_lag(a: np.ndarray, b: np.ndarray, *, max_lag: int = 12) -> tuple[int, float]:
    """
    Signed lag minimizing Pearson r (most complementary alignment).

    Tie-break: prefer larger |lag| so ±30° offsets are not collapsed to lag 0.
    """
    corrs = cross_correlation_lag(a, b, max_lag=max_lag)
    min_r = min(corrs.values())
    candidates = [lag for lag, r in corrs.items() if r <= min_r + 1e-9]
    best_lag = max(candidates, key=lambda lag: abs(lag))
    return int(best_lag), float(corrs[best_lag])


def directed_pair_metrics(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_lag: int = 12,
) -> DirectedPairMetrics:
    corr0 = pearson_corr(a, b, lag=0)
    phase = price_phase_lag(a, b, max_lag=max_lag)
    anti_lag, corr_anti = optimal_anti_corr_lag(a, b, max_lag=max_lag)
    return DirectedPairMetrics(
        corr_lag0=float(corr0),
        phase_lag=float(phase),
        anti_corr_lag=int(anti_lag),
        corr_at_anti_lag=float(corr_anti),
    )


def _metric_matrix(
    states: list[StatePriceSeries],
    *,
    max_lag: int,
    attr: str,
) -> pd.DataFrame:
    labels = [s.state for s in states]
    n = len(labels)
    mat = np.zeros((n, n), dtype=float)
    np.fill_diagonal(mat, 0.0 if attr != "corr_lag0" else 1.0)
    for i, j in combinations(range(n), 2):
        m = directed_pair_metrics(states[i].prices, states[j].prices, max_lag=max_lag)
        val = float(getattr(m, attr))
        mat[i, j] = val
        if attr in ("corr_lag0", "corr_at_anti_lag"):
            mat[j, i] = val
        else:
            # Directed: reverse direction flips sign for signed phase metrics
            m_rev = directed_pair_metrics(states[j].prices, states[i].prices, max_lag=max_lag)
            mat[j, i] = float(getattr(m_rev, attr))
    if attr == "corr_lag0":
        np.fill_diagonal(mat, 1.0)
    return pd.DataFrame(mat, index=labels, columns=labels)


def pairwise_correlation_matrix(states: list[StatePriceSeries], *, max_lag: int = 12) -> pd.DataFrame:
    return _metric_matrix(states, max_lag=max_lag, attr="corr_lag0")


def pairwise_phase_lag_matrix(states: list[StatePriceSeries], *, max_lag: int = 12) -> pd.DataFrame:
    return _metric_matrix(states, max_lag=max_lag, attr="phase_lag")


def pairwise_anti_corr_lag_matrix(states: list[StatePriceSeries], *, max_lag: int = 12) -> pd.DataFrame:
    return _metric_matrix(states, max_lag=max_lag, attr="anti_corr_lag")


def pairwise_corr_at_anti_lag_matrix(states: list[StatePriceSeries], *, max_lag: int = 12) -> pd.DataFrame:
    return _metric_matrix(states, max_lag=max_lag, attr="corr_at_anti_lag")


def compute_all_directed_matrices(
    states: list[StatePriceSeries],
    *,
    max_lag: int = 12,
) -> dict[str, pd.DataFrame]:
    return {
        "corr_lag0": pairwise_correlation_matrix(states, max_lag=max_lag),
        "phase_lag": pairwise_phase_lag_matrix(states, max_lag=max_lag),
        "anti_corr_lag": pairwise_anti_corr_lag_matrix(states, max_lag=max_lag),
        "corr_at_anti_lag": pairwise_corr_at_anti_lag_matrix(states, max_lag=max_lag),
    }


def pairwise_directed_long(
    states: list[StatePriceSeries],
    *,
    max_lag: int = 12,
) -> pd.DataFrame:
    rows = []
    for i, j in combinations(range(len(states)), 2):
        a, b = states[i], states[j]
        m_ab = directed_pair_metrics(a.prices, b.prices, max_lag=max_lag)
        m_ba = directed_pair_metrics(b.prices, a.prices, max_lag=max_lag)
        rows.append(
            {
                "state_a": a.state,
                "state_b": b.state,
                **{f"a_to_b_{k}": v for k, v in m_ab.as_dict().items()},
                **{f"b_to_a_{k}": v for k, v in m_ba.as_dict().items()},
                "lon_a": a.lon,
                "lat_a": a.lat,
                "lon_b": b.lon,
                "lat_b": b.lat,
                "iso_a": a.iso,
                "iso_b": b.iso,
                "hub_a": a.hub,
                "hub_b": b.hub,
                "same_hub": a.iso == b.iso and a.hub == b.hub,
            }
        )
    return pd.DataFrame(rows)


def reference_field(matrix: pd.DataFrame, ref_state: str) -> dict[str, float]:
    if ref_state not in matrix.index:
        raise KeyError(ref_state)
    return {st: float(matrix.loc[ref_state, st]) for st in matrix.index}


def mean_abs_lmp_spread_matrix(states: list[StatePriceSeries]) -> pd.DataFrame:
    """Mean absolute interstate LMP difference ($/MWh): row = reference, col = partner."""
    codes = [s.state for s in states]
    n = len(codes)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            n_h = min(len(states[i].prices), len(states[j].prices))
            mat[i, j] = float(np.mean(np.abs(states[j].prices[:n_h] - states[i].prices[:n_h])))
    return pd.DataFrame(mat, index=codes, columns=codes)


def reference_correlation_field(states, corr, ref_state: str) -> dict[str, float]:
    return reference_field(corr, ref_state)


def mean_pairwise_correlation(corr: pd.DataFrame) -> pd.Series:
    n = len(corr)
    if n <= 1:
        return pd.Series(dtype=float)
    totals = corr.sum(axis=1) - np.diag(corr)
    return totals / (n - 1)


def mean_phase_lag_vs_others(phase_mat: pd.DataFrame) -> pd.Series:
    """For each reference state, mean signed phase lag to all other states."""
    n = len(phase_mat)
    if n <= 1:
        return pd.Series(dtype=float)
    totals = phase_mat.sum(axis=1)
    return totals / (n - 1)


# Back-compat aliases
mean_directed_lag_centroid = mean_phase_lag_vs_others
pairwise_lag_centroid_matrix = pairwise_phase_lag_matrix
correlation_lag_centroid = price_phase_lag


def metric_description(metric: MetricName) -> str:
    desc = {
        "corr_lag0": "Pearson r at lag 0 (symmetric; legacy)",
        "phase_lag": "+ = partner price delayed vs reference (hours)",
        "anti_corr_lag": "Signed lag (h) at minimum r (tie-break |lag|)",
        "corr_at_anti_lag": "Pearson r at anti-correlation lag (symmetric magnitude)",
    }
    return desc[metric]
