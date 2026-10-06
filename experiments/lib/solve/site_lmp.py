from __future__ import annotations

from experiments.lib.lmp_io import load_hourly_prices
from experiments.lib.texas_config import get_site_lmp_csv


def load_site_lmp(
    system_data: dict,
    site_lmp_zones: dict[int, str],
    hours: int,
    season: str,
) -> dict:
    if season not in ("winter", "summer"):
        raise ValueError(f"season must be winter or summer, got {season!r}")

    out: dict[int, list[float]] = {}
    for site in system_data["dc"]["sites"]:
        prepared = get_site_lmp_csv(site, season)
        if prepared is None or not prepared.exists():
            raise FileNotFoundError(
                f"Missing prepared LMP for site {site} season {season}: {prepared}. "
                "Run experiments/E01/scripts/prepare_data.py first."
            )
        out[site] = load_hourly_prices(prepared, hours_expected=hours)
    system_data["dc"]["day_ahead_prices"] = out
    return out
