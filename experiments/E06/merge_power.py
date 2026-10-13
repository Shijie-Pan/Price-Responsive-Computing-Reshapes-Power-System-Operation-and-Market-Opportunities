#!/usr/bin/env python3
"""Merge nationwide E01 per-operator power CSVs for USA LOPF."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from experiments.lib.paths import package_root
ROOT = package_root()
sys.path.insert(0, str(ROOT))

from experiments.lib.solve.usa_lopf import PRIORITY_SCENARIOS  # noqa: E402

OPS_CFG = ROOT / "experiments/E06/config/nationwide_operators.json"
E01_ROOT = ROOT / "experiments/E06/results/scheduling"
DEFAULT_OUT = ROOT / "experiments/E06/data/merged_power"
OPERATOR_IDS = ["coreweave", "oracle", "google", "meta", "microsoft", "amazon"]
SEASONS = ("winter", "summer")


def _latest_power_csv(run_dir: Path, scenario: str, season: str) -> Path | None:
    matches = sorted(
        run_dir.glob(f"power_consumption_{scenario}_{season}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _run_stamp(path: Path, season: str) -> str | None:
    m = re.search(
        rf"power_consumption_\w+_{season}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})\.csv",
        path.name,
    )
    return m.group(1) if m else None


def _load_power_csv(path: Path) -> tuple[list[int], dict[int, dict[int, float]]]:
    hours: list[int] = []
    by_hour: dict[int, dict[int, float]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        site_cols = [
            k
            for k in reader.fieldnames or []
            if k.startswith("Site") and k.endswith("_Power_MW")
        ]
        site_ids = [int(c.replace("Site", "").replace("_Power_MW", "")) for c in site_cols]
        for row in reader:
            t = int(row["Time_Period"])
            hours.append(t)
            by_hour[t] = {sid: float(row[col]) for sid, col in zip(site_ids, site_cols)}
    return site_ids, by_hour


def merge_season_scenario(season: str, scenario: str, out_dir: Path, e01_root: Path) -> dict:
    ops = json.loads(OPS_CFG.read_text(encoding="utf-8"))
    site_by_op = {
        op_id: [int(s) for s in (op.get("site_ids") or [])]
        for op_id, op in ops["operators"].items()
    }
    all_sites = sorted({sid for sids in site_by_op.values() for sid in sids})
    merged: dict[int, dict[int, float]] = {}
    sources: dict[str, str] = {}

    for op in OPERATOR_IDS:
        run_dir = e01_root / op / season
        csv_path = _latest_power_csv(run_dir, scenario, season)
        if csv_path is None:
            raise FileNotFoundError(f"Missing E01 power CSV: {op}/{season}/{scenario}")
        site_ids, by_hour = _load_power_csv(csv_path)
        expected = set(site_by_op[op])
        if set(site_ids) != expected:
            raise ValueError(
                f"{op}/{season}/{scenario}: CSV sites {sorted(site_ids)} != expected {sorted(expected)}"
            )
        sources[op] = str(csv_path.relative_to(ROOT)).replace("\\", "/")
        for t, sites in by_hour.items():
            bucket = merged.setdefault(t, {})
            for sid, mw in sites.items():
                if sid in bucket:
                    raise ValueError(f"Duplicate site {sid} at t={t}")
                bucket[sid] = mw

    hours = sorted(merged)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _run_stamp(ROOT / sources[OPERATOR_IDS[0]], season) or "merged"
    out_path = out_dir / f"power_merged_{scenario}_{season}_{stamp}.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["Time_Period"] + [f"Site{sid}_Power_MW" for sid in all_sites] + [
            "Total_Power_MW"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in hours:
            row: dict = {"Time_Period": t}
            total = 0.0
            for sid in all_sites:
                mw = float(merged[t].get(sid, 0.0))
                row[f"Site{sid}_Power_MW"] = mw
                total += mw
            row["Total_Power_MW"] = total
            writer.writerow(row)

    return {
        "season": season,
        "scenario": scenario,
        "output_csv": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        "site_ids": all_sites,
        "n_hours": len(hours),
        "sources_by_operator": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", choices=[*SEASONS, "both"], default="both")
    parser.add_argument(
        "--scenario",
        default="priority",
        help="One scenario name, 'priority' (FCFS+Full flex), or 'all'",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--e01-root", type=Path, default=E01_ROOT)
    args = parser.parse_args()

    if args.scenario == "priority":
        scenarios = list(PRIORITY_SCENARIOS)
    elif args.scenario == "all":
        scenarios = list(PRIORITY_SCENARIOS)  # extend later if needed
    else:
        scenarios = [args.scenario]

    seasons = list(SEASONS) if args.season == "both" else [args.season]
    entries = []
    for season in seasons:
        for scenario in scenarios:
            info = merge_season_scenario(season, scenario, args.out_dir, args.e01_root)
            entries.append(info)
            print(
                f"[merge] {season}/{scenario}: hours={info['n_hours']} -> {info['output_csv']}",
                flush=True,
            )

    manifest = {
        "merged_power_dir": str(args.out_dir.relative_to(ROOT)).replace("\\", "/"),
        "entries": entries,
    }
    man_path = args.out_dir / "merge_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[merge] wrote {man_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
