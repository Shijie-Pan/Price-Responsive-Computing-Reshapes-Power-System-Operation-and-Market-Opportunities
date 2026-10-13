#!/usr/bin/env python3
"""Merge per-operator E01 power CSVs into unified site profiles for E02 PyPSA."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from experiments.lib.paths import package_root
ROOT = package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.texas_config import (  # noqa: E402
    OPERATOR_IDS,
    SEASONS,
    TEXAS_ROOT,
    load_texas7_config,
)
from experiments.lib.scenarios import E01_SCENARIOS  # noqa: E402

E01_ROOT = TEXAS_ROOT / "results"
DEFAULT_OUT = ROOT / "experiments" / "E08" / "merged_power"


def _site_ids_by_operator(cfg: dict) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for op_id, op in cfg["operators"].items():
        out[op_id] = [int(s) for s in (op.get("site_ids") or op.get("site_ids") or [])]
    return out


def _latest_power_csv(run_dir: Path, scenario: str, season: str) -> Path | None:
    exact = run_dir / f"power_consumption_{scenario}_{season}.csv"
    if exact.is_file():
        return exact
    matches = sorted(
        run_dir.glob(f"power_consumption_{scenario}_{season}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _run_stamp(path: Path, season: str) -> str | None:
    m = re.search(rf"power_consumption_\w+_{season}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})\.csv", path.name)
    return m.group(1) if m else None


def _load_power_csv(path: Path) -> tuple[list[int], dict[int, dict[int, float]]]:
    hours: list[int] = []
    by_hour: dict[int, dict[int, float]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        site_cols = [k for k in reader.fieldnames or [] if k.startswith("Site") and k.endswith("_Power_MW")]
        site_ids = [int(c.replace("Site", "").replace("_Power_MW", "")) for c in site_cols]
        for row in reader:
            t = int(row["Time_Period"])
            hours.append(t)
            by_hour[t] = {}
            for sid, col in zip(site_ids, site_cols):
                by_hour[t][sid] = float(row[col])
    return site_ids, by_hour


def merge_season_scenario(
    cfg: dict,
    season: str,
    scenario: str,
    out_dir: Path,
    e01_root: Path | None = None,
) -> dict:
    e01_root = Path(e01_root) if e01_root is not None else E01_ROOT
    site_by_op = _site_ids_by_operator(cfg)
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
                f"{op}/{season}/{scenario}: CSV sites {site_ids} != expected {sorted(expected)}"
            )
        sources[op] = str(csv_path.relative_to(ROOT)).replace("\\", "/")
        for t, sites in by_hour.items():
            merged.setdefault(t, {})
            for sid, mw in sites.items():
                if sid in merged[t]:
                    raise ValueError(f"Duplicate site {sid} at t={t} when merging operators")
                merged[t][sid] = mw

    hours = sorted(merged)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _run_stamp(Path(sources[OPERATOR_IDS[0]]), season)
    name = (
        f"power_merged_{scenario}_{season}_{stamp}.csv"
        if stamp
        else f"power_merged_{scenario}_{season}.csv"
    )
    out_path = out_dir / name

    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["Time_Period"] + [f"Site{sid}_Power_MW" for sid in all_sites] + ["Total_Power_MW"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in hours:
            row = {"Time_Period": t}
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
    parser.add_argument("--scenario", default=None, help="Single scenario (default: all 9)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--e01-root", type=Path, default=None, help="Override E01 results root")
    args = parser.parse_args()

    cfg = load_texas7_config()
    e01_root = args.e01_root or E01_ROOT
    print(f"[merge] study={TEXAS_ROOT.name}  e01={e01_root}  out={args.out_dir}")
    seasons = list(SEASONS) if args.season == "both" else [args.season]
    scenarios = [args.scenario] if args.scenario else list(E01_SCENARIOS)

    entries: list[dict] = []
    for season in seasons:
        for scenario in scenarios:
            if scenario not in E01_SCENARIOS:
                raise ValueError(f"Unknown scenario {scenario!r}")
            info = merge_season_scenario(
                cfg, season, scenario, args.out_dir, e01_root=e01_root
            )
            entries.append(info)
            print(f"[merge] {season}/{scenario} -> {info['output_csv']} ({info['n_hours']} h)")

    manifest = {
        "description": "Merged E01 power profiles (same Time_Period, ignore trace offsets)",
        "n_files": len(entries),
        "files": entries,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
