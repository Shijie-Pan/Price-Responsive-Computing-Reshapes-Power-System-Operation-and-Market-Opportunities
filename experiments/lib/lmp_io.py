from __future__ import annotations

import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

STANDARD_COLUMNS = ("interval_start_utc", "interval_end_utc", "location", "spp")
HOURS_DEFAULT = 744

SEASONS: dict[str, dict] = {
    "winter": {
        "year": 2026,
        "month": 1,
        "start_utc": "2026-01-01 00:00:00+00:00",
        "label": "2026-01",
        "hours": 744,
    },
    "summer": {
        "year": 2026,
        "month": 7,
        "start_utc": "2026-07-01 00:00:00+00:00",
        "label": "2026-07",
        "hours": 744,
    },
    # Full-year window for Fig1 robustness / ExtData (keeps summer/winter month caches intact).
    "annual": {
        "start_date": date(2025, 8, 1),
        "end_date": date(2026, 7, 31),
        "start_utc": "2025-08-01 00:00:00+00:00",
        "label": "2025-08_2026-07",
        "hours": 8760,  # 365 days
    },
}


def season_hours(season: str) -> int:
    spec = SEASONS[season]
    if "hours" in spec:
        return int(spec["hours"])
    start, end = season_dates(season)
    return ((end - start).days + 1) * 24


def season_dates(season: str) -> tuple[date, date]:
    spec = SEASONS[season]
    if "start_date" in spec and "end_date" in spec:
        return spec["start_date"], spec["end_date"]
    y, m = int(spec["year"]), int(spec["month"])
    last = calendar.monthrange(y, m)[1]
    return date(y, m, 1), date(y, m, last)


def iter_month_spans(start: date, end: date) -> list[tuple[date, date]]:
    """Inclusive calendar-month chunks covering [start, end]."""
    spans: list[tuple[date, date]] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        last = calendar.monthrange(cursor.year, cursor.month)[1]
        month_end = date(cursor.year, cursor.month, last)
        span_start = max(cursor, start)
        span_end = min(month_end, end)
        spans.append((span_start, span_end))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return spans


def pad_hourly_prices(prices: list[float], hours: int = HOURS_DEFAULT) -> list[float]:
    if len(prices) >= hours:
        return prices[:hours]
    pad = prices[-1] if prices else 0.0
    return prices + [pad] * (hours - len(prices))


def _normalize_ercot_zone(zone: str) -> str:
    if zone == "LZ_WEST":
        return "LZ_SOUTH"
    return zone


def load_hourly_prices(csv_path: Path, hours_expected: int) -> list[float]:
    df = pd.read_csv(csv_path)
    price_col = "lmp" if "lmp" in df.columns else "spp"
    time_col = "interval_start_utc" if "interval_start_utc" in df.columns else "interval_start"
    df = df[[time_col, price_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df["hour"] = df[time_col].dt.floor("h")
    hourly = df.groupby("hour")[price_col].mean().reset_index().sort_values("hour")
    prices = hourly[price_col].astype(float).tolist()
    if len(prices) >= hours_expected:
        return prices[:hours_expected]
    pad = prices[-1] if prices else 0.0
    return prices + [pad] * (hours_expected - len(prices))


def write_hourly_lmp_csv(
    prices: list[float],
    out_path: Path,
    *,
    location: str,
    start: str = "2025-01-01 00:00:00+00:00",
    price_column: str = "spp",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    rows = []
    for i, price in enumerate(prices):
        t0 = start_ts + pd.Timedelta(hours=i)
        t1 = t0 + pd.Timedelta(hours=1)
        rows.append(
            {
                "interval_start_utc": t0.isoformat(),
                "interval_end_utc": t1.isoformat(),
                "location": location,
                price_column: float(price),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
