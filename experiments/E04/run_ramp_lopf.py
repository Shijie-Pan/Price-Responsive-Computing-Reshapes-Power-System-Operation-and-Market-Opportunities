#!/usr/bin/env python3
"""Optional B1 ramp LOPF using the same flex_lopf entry as Fig. 4.

A0 / A1 / no_dc loadings are never re-solved here: ``--seed-only`` (and every
full run) copies the deposited Fig. 4 hourly frames into
``results/flexibility/ramp`` under those aliases.

B1 power CSVs are staged from the package-local
``results/ramp_limit/b1_merged_power`` tables.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _package_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "experiments" / "lib").is_dir() and (
            candidate / "experiments" / "E04"
        ).is_dir():
            return candidate
    raise RuntimeError("Cannot find the deposit root")


ROOT = _package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.solve.flex_lopf import run_flexibility_comparison  # noqa: E402
from experiments.lib.map_hour import load_hourly_frames, save_hourly_scenario  # noqa: E402

CONFIG_DIR = ROOT / "experiments/E04/config"
RESULTS_DIR = ROOT / "experiments/E04/results/flexibility/ramp"
FIG4_DIR = ROOT / "experiments/E04/results/flexibility/full"
MERGED_DIR = ROOT / "experiments/E04/results/ramp_limit/merged_power"
B1_MERGED = ROOT / "experiments/E04/results/ramp_limit/b1_merged_power"

B1_CASES = ("B1_ramp50", "B1_ramp25", "B1_ramp10")
FIG4_ALIAS = {
    "A0": "baseline",
    "A1": "reallocation_slack_termination",
    "no_dc": "no_dc",
}


def _stage_merged_power() -> None:
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    fig4_merged = ROOT / "experiments/E08/merged_power"
    mapping = {"A0": "baseline", "A1": "reallocation_slack_termination"}
    for season in ("summer", "winter"):
        for alias, src_name in mapping.items():
            src = fig4_merged / f"power_merged_{src_name}_{season}.csv"
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, MERGED_DIR / f"power_merged_{alias}_{season}_fig4.csv")
        for case in B1_CASES:
            src = B1_MERGED / f"power_merged_{case}_{season}.csv"
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, MERGED_DIR / f"power_merged_{case}_{season}_fig4pipe.csv")


def _seed_fig4_aliases() -> None:
    """Copy Fig. 4 hourly loadings to A0 / A1 / no_dc aliases."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for season in ("summer", "winter"):
        frames, meta = load_hourly_frames(FIG4_DIR, season)
        for alias, key in FIG4_ALIAS.items():
            if key not in frames:
                raise FileNotFoundError(f"Fig. 4 missing {key} ({season})")
            save_hourly_scenario(
                frames[key], results_dir=RESULTS_DIR, season=season, scenario=alias
            )
        out_meta = RESULTS_DIR / f"hourly_{season}" / "meta.json"
        if meta and not out_meta.is_file():
            out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", choices=["winter", "summer", "both"], default="both")
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Stage power CSVs and copy Fig. 4 A0/A1/no_dc loadings only.",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()

    pypsa_cfg = json.loads((CONFIG_DIR / "pypsa_texas7.json").read_text(encoding="utf-8"))
    network_path = ROOT / pypsa_cfg["network_path"]
    _stage_merged_power()
    _seed_fig4_aliases()
    if args.seed_only:
        print(f"staged merged power -> {MERGED_DIR.relative_to(ROOT)}")
        print(f"seeded Fig. 4 aliases -> {RESULTS_DIR.relative_to(ROOT)}")
        return

    seasons = ["winter", "summer"] if args.season == "both" else [args.season]
    for season in seasons:
        print(f"[ramp-lopf] {season} B1 via Fig. 4 flex_lopf", flush=True)
        run_flexibility_comparison(
            season=season,
            merged_dir=MERGED_DIR,
            config_dir=CONFIG_DIR,
            results_dir=RESULTS_DIR,
            solver=args.solver,
            eval_start=0,
            eval_length=168,
            scenarios=list(B1_CASES),
            network_path=network_path,
            run_voltage_pf=False,
            voltage_eval_all_hours=False,
            resume=bool(args.resume),
        )


if __name__ == "__main__":
    main()
