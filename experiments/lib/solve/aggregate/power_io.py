"""Load/save 7-site hour×site power profiles (MW)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

Profile = dict[int, dict[int, float]]  # {site_id: {hour: MW}}


def merged_csv_to_profile(path: Path, sites: list[int], hours: int) -> Profile:
    out: Profile = {s: {} for s in sites}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["Time_Period"])
            if t >= hours:
                continue
            for s in sites:
                col = f"Site{s}_Power_MW"
                if col in row:
                    out[s][t] = float(row[col])
    return out


def profile_to_merged_csv(profile: Profile, path: Path, sites: list[int], hours: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["Time_Period"] + [f"Site{s}_Power_MW" for s in sites] + ["Total_Power_MW"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in range(hours):
            row = [profile[s].get(t, 0.0) for s in sites]
            w.writerow([t, *row, sum(row)])


def latest_merged_csv(merged_dir: Path, scenario: str, season: str) -> Path:
    matches = sorted(
        merged_dir.glob(f"power_merged_{scenario}_{season}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No merged CSV for {scenario}/{season} in {merged_dir}")
    return matches[0]


def build_fixed_dc_profile(reference: Profile, sites: list[int], hours: int) -> Profile:
    out: Profile = {s: {} for s in sites}
    for s in sites:
        mean_mw = sum(reference[s].get(t, 0.0) for t in range(hours)) / max(hours, 1)
        for t in range(hours):
            out[s][t] = mean_mw
    return out


def dump_profile_json(profile: Profile, path: Path, sites: list[int], hours: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(s): [profile[s].get(t, 0.0) for t in range(hours)] for s in sites}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
