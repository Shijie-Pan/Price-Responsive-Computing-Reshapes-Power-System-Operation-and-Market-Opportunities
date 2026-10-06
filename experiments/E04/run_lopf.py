#!/usr/bin/env python3
"""Run Texas E02ex PyPSA flexibility LOPF on dense s1250 (B2 accounting)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.lib.paths import package_root
ROOT = package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.solve.flex_lopf import run_flexibility_comparison  # noqa: E402
from experiments.lib.texas_config import REPO_ROOT, study_e02_root  # noqa: E402
from experiments.lib.scenarios import (  # noqa: E402
    E01_SCENARIOS,
    E02EX_PRIORITY_SCENARIOS,
)

E02_ROOT = study_e02_root()
CONFIG_DIR = E02_ROOT / "config"
RESULTS_DIR = E02_ROOT / "results" / "flexibility"


def main() -> None:
    parser = argparse.ArgumentParser(description="Texas E02ex s1250 PyPSA flexibility LOPF")
    parser.add_argument("--season", choices=["winter", "summer", "both"], default="winter")
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--eval-length", type=int, default=168)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Fast check: no_dc reference + initial_assignment only",
    )
    mode.add_argument(
        "--priority",
        action="store_true",
        help="No-DC + FCFS (baseline) + Full flex only; write into results/flexibility/full",
    )
    mode.add_argument(
        "--fill-remaining",
        action="store_true",
        help="Solve remaining E01 scenarios; skip those with hourly pickles",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenarios that already have hourly_*.pkl",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override the output folder. The default priority folder is the deposited result.",
    )
    parser.add_argument("--skip-voltage-pf", action="store_true")
    parser.add_argument("--voltage-all-hours", action="store_true")
    args = parser.parse_args()

    pypsa_cfg = json.loads((CONFIG_DIR / "pypsa_texas7.json").read_text(encoding="utf-8"))
    merged_dir = REPO_ROOT / pypsa_cfg["merged_power_dir"]
    network_path = REPO_ROOT / pypsa_cfg["network_path"]

    for req in ("dc_bus_map.json", "dc_load_accounting.json"):
        if not (CONFIG_DIR / req).exists():
            raise SystemExit(
                f"Missing {CONFIG_DIR / req}. Run prepare_pypsa_bus_map.py in pypsa-usa env first."
            )
    if not merged_dir.exists():
        raise SystemExit(f"Missing merged power dir (reused from E02): {merged_dir}")
    if not network_path.exists():
        raise SystemExit(f"Missing s1250 network: {network_path}")

    resume = bool(args.resume or args.fill_remaining)
    if args.smoke:
        scenarios = ["initial_assignment"]
        tag = "smoke"
    elif args.priority:
        scenarios = list(E02EX_PRIORITY_SCENARIOS)
        tag = "full"
    else:
        scenarios = list(E01_SCENARIOS)
        tag = "full"
    seasons = ["winter", "summer"] if args.season == "both" else [args.season]

    for season in seasons:
        print(
            f"[E02ex] {E02_ROOT.parent.name}/{season} s1250 LOPF "
            f"mode={'priority' if args.priority else 'fill-remaining' if args.fill_remaining else tag} "
            f"scenarios={scenarios} resume={resume} merged={merged_dir}",
            flush=True,
        )
        run_flexibility_comparison(
            season=season,
            merged_dir=merged_dir,
            config_dir=CONFIG_DIR,
            results_dir=args.results_dir or (RESULTS_DIR / tag),
            solver=args.solver,
            eval_start=args.eval_start,
            eval_length=args.eval_length,
            scenarios=scenarios,
            network_path=network_path,
            run_voltage_pf=not args.skip_voltage_pf,
            voltage_eval_all_hours=args.voltage_all_hours,
            resume=resume,
        )


if __name__ == "__main__":
    main()
