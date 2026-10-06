"""Load merged E01 power CSVs for E02."""

from __future__ import annotations

import csv
from pathlib import Path


def load_merged_power_csv(path: Path) -> dict[int, dict[int, float]]:
    """Return {hour: {site_id: MW}} from merged 7-site CSV."""
    by_hour: dict[int, dict[int, float]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        site_cols = [
            (int(k.replace("Site", "").replace("_Power_MW", "")), k)
            for k in (reader.fieldnames or [])
            if k.startswith("Site") and k.endswith("_Power_MW")
        ]
        for row in reader:
            t = int(row["Time_Period"])
            by_hour[t] = {sid: float(row[col]) for sid, col in site_cols}
    return by_hour
