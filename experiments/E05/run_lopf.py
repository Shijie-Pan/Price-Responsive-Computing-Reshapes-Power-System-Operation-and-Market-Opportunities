#!/usr/bin/env python3
"""Run USA 134c LOPF: No-DC / FCFS / Full flexibility (paper experiment C)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.lib.paths import package_root
ROOT = package_root()
sys.path.insert(0, str(ROOT))

from experiments.lib.solve.usa_lopf import (  # noqa: E402
    PRIORITY_SCENARIOS,
    SCENARIO_LABELS,
    USA_NETWORK,
    solve_scenario,
    write_lmp_artifacts,
)

E02 = ROOT / "experiments/E05"
CONFIG = E02 / "config"
MERGED = E02 / "data/merged_power"
RESULTS = E02 / "results"


def _latest_merged(scenario: str, season: str) -> Path:
    exact = MERGED / f"power_merged_{scenario}_{season}.csv"
    matches = [exact] if exact.is_file() else sorted(
        MERGED.glob(f"power_merged_{scenario}_{season}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No merged power for {scenario}/{season} under {MERGED}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Nationwide E02 USA LOPF")
    parser.add_argument("--season", choices=["winter", "summer", "both"], default="both")
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--eval-length", type=int, default=24)
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument(
        "--scenarios",
        default="priority",
        help="Comma list, or 'priority' (= no_dc + baseline + full flex)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tag", default="priority_24h")
    args = parser.parse_args()

    bus_map = json.loads((CONFIG / "dc_bus_map.json").read_text(encoding="utf-8"))
    if args.scenarios == "priority":
        scenarios = ["no_dc", *PRIORITY_SCENARIOS]
    else:
        scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    seasons = ["winter", "summer"] if args.season == "both" else [args.season]
    out_root = RESULTS / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "network": str(USA_NETWORK.relative_to(ROOT)).replace("\\", "/"),
        "eval_start": args.eval_start,
        "eval_length": args.eval_length,
        "seasons": {},
    }

    for season in seasons:
        season_dir = out_root / season
        season_dir.mkdir(parents=True, exist_ok=True)
        portfolios: dict[str, dict] = {}
        for scenario in scenarios:
            sc_dir = season_dir / scenario
            metrics_path = sc_dir / "metrics.json"
            if args.resume and metrics_path.exists():
                print(f"[skip] {season}/{scenario} (resume)", flush=True)
                portfolios[scenario] = json.loads(metrics_path.read_text(encoding="utf-8"))
                continue

            power_csv = None if scenario == "no_dc" else _latest_merged(scenario, season)
            print(
                f"[LOPF] {season}/{scenario} ({SCENARIO_LABELS.get(scenario, scenario)}) "
                f"len={args.eval_length} power={power_csv}",
                flush=True,
            )
            metrics, n = solve_scenario(
                season=season,
                scenario=scenario,
                bus_map=bus_map,
                power_csv=power_csv,
                eval_start=args.eval_start,
                eval_length=args.eval_length,
                solver=args.solver,
            )
            paths = write_lmp_artifacts(sc_dir, n, bus_map, metrics=metrics)
            metrics["artifacts"] = paths
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            portfolios[scenario] = metrics
            print(
                f"  ok={metrics.get('lopf_ok')} runtime={metrics.get('runtime_seconds'):.1f}s "
                f"obj={metrics.get('objective_usd')} "
                f"campus_mae={metrics.get('campus_mean_abs_delta_lmp')} "
                f"pair_mae={metrics.get('mean_abs_pairwise_lmp'):.3f} "
                f"cong_mean={metrics.get('mean_n_congested'):.2f}",
                flush=True,
            )

        # before/after deltas vs no_dc
        ref = portfolios.get("no_dc", {})
        compare = {}
        for sc, m in portfolios.items():
            compare[sc] = {
                "label": SCENARIO_LABELS.get(sc, sc),
                "campus_mean_abs_delta_lmp": m.get("campus_mean_abs_delta_lmp"),
                "mean_abs_pairwise_lmp": m.get("mean_abs_pairwise_lmp"),
                "mean_hourly_lmp_std": m.get("mean_hourly_lmp_std"),
                "mean_n_congested": m.get("mean_n_congested"),
                "peak_loading_pct": m.get("peak_loading_pct"),
                "total_dc_mwh": m.get("total_dc_mwh"),
                "objective_usd": m.get("objective_usd"),
                "delta_campus_mae_vs_no_dc": (
                    None
                    if not ref
                    else round(
                        float(m.get("campus_mean_abs_delta_lmp") or 0)
                        - float(ref.get("campus_mean_abs_delta_lmp") or 0),
                        4,
                    )
                ),
                "delta_pairwise_mae_vs_no_dc": (
                    None
                    if not ref
                    else round(
                        float(m.get("mean_abs_pairwise_lmp") or 0)
                        - float(ref.get("mean_abs_pairwise_lmp") or 0),
                        4,
                    )
                ),
            }
        cmp_path = season_dir / "before_after_summary.json"
        cmp_path.write_text(json.dumps(compare, indent=2) + "\n", encoding="utf-8")
        summary["seasons"][season] = compare
        print(f"[summary] {season} -> {cmp_path.relative_to(ROOT)}", flush=True)

    sum_path = out_root / "run_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {sum_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
