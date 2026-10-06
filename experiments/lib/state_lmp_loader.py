"""Load per-state LMP price arrays from lmp_cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.lib.download_lmp import download_prices
from experiments.lib.lmp_io import season_hours
from experiments.lib.state_lmp_registry import (
    DEFAULT_CACHE,
    _state_cache_path,
    hub_cached,
    load_registry,
)

STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AL": (-86.9, 32.8), "AK": (-152.0, 64.0), "AZ": (-111.7, 34.3), "AR": (-92.4, 34.8),
    "CA": (-119.4, 37.2), "CO": (-105.5, 39.0), "CT": (-72.7, 41.6), "DE": (-75.5, 39.0),
    "DC": (-77.0, 38.9), "FL": (-81.5, 27.8), "GA": (-83.4, 32.2), "HI": (-157.8, 21.3),
    "ID": (-114.5, 44.4), "IL": (-89.4, 40.0), "IN": (-86.3, 39.8), "IA": (-93.5, 42.0),
    "KS": (-98.5, 38.5), "KY": (-85.3, 37.5), "LA": (-91.9, 30.9), "ME": (-69.0, 45.4),
    "MD": (-76.8, 39.0), "MA": (-71.5, 42.3), "MI": (-84.5, 44.3), "MN": (-94.2, 46.3),
    "MS": (-89.7, 32.7), "MO": (-92.5, 38.4), "MT": (-110.0, 47.0), "NE": (-99.8, 41.5),
    "NV": (-116.4, 39.3), "NH": (-71.6, 43.5), "NJ": (-74.5, 40.1), "NM": (-106.0, 34.5),
    "NY": (-75.5, 43.0), "NC": (-79.0, 35.5), "ND": (-100.5, 47.5), "OH": (-82.8, 40.3),
    "OK": (-97.5, 35.5), "OR": (-120.5, 44.0), "PA": (-77.8, 40.9), "RI": (-71.5, 41.7),
    "SC": (-80.9, 33.9), "SD": (-100.2, 44.4), "TN": (-86.3, 35.8), "TX": (-99.9, 31.4),
    "UT": (-111.5, 39.3), "VT": (-72.7, 44.0), "VA": (-78.7, 37.5), "WA": (-120.5, 47.4),
    "WV": (-80.6, 38.6), "WI": (-89.6, 44.6), "WY": (-107.5, 43.0),
}


def _load_prices_for_state(
    state: str,
    info: dict,
    season: str,
    *,
    cache_dir: Path,
) -> list[float]:
    iso, hub = str(info["iso"]), str(info["hub"])
    hours = season_hours(season)
    sc_path = _state_cache_path(cache_dir, state, season)
    if sc_path.exists():
        payload = json.loads(sc_path.read_text(encoding="utf-8"))
        cached_hub = str(payload.get("hub", payload.get("fetch_hub", "")))
        if cached_hub.upper() == hub.upper():
            prices = [float(x) for x in payload.get("prices", [])]
            if len(prices) >= hours:
                return prices[:hours]

    if hub_cached(cache_dir, iso, hub, season):
        from experiments.lib.download_lmp import _cache_path, _load_cache

        cp = _cache_path(cache_dir, iso, hub, season)
        cached = _load_cache(cp, hours=hours)
        if cached:
            return cached

    prices, _ = download_prices(
        iso,
        hub,
        season,
        cache_dir=cache_dir,
        cache_key=f"state_{state}",
    )
    return prices


def load_state_prices(
    state: str,
    season: str,
    *,
    cache_dir: Path | None = None,
) -> np.ndarray:
    reg = load_registry()
    info = reg["states"][state]
    cache_dir = cache_dir or DEFAULT_CACHE
    return np.asarray(_load_prices_for_state(state, info, season, cache_dir=cache_dir), dtype=float)
