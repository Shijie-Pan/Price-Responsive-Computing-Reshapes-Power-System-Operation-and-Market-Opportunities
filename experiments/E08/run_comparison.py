#!/usr/bin/env python3
"""Texas E03 — job-level vs aggregate scheduling comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.lib.paths import package_root
ROOT = package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.solve.aggregate.runner import (  # noqa: E402
    load_config,
    run_season_comparison,
)
from experiments.lib.texas_config import study_e03_root  # noqa: E402

E03_ROOT = study_e03_root()


def main() -> None:
    parser = argparse.ArgumentParser(description="E03 private-side model comparison (Texas 7-site)")
    parser.add_argument("--season", choices=["winter", "summer", "both"], default="both")
    parser.add_argument("--config", type=Path, default=E03_ROOT / "config" / "comparison.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seasons = ["winter", "summer"] if args.season == "both" else [args.season]
    combined = {}
    for season in seasons:
        print(f"\n=== E03 scheduling · {season} ===")
        combined[season] = run_season_comparison(season, cfg=cfg)

    summary = E03_ROOT / "results" / "comparison_summary.json"
    summary.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\n[done] {summary}")


if __name__ == "__main__":
    main()
