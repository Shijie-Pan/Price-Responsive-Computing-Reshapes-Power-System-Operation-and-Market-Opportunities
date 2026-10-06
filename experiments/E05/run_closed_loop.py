#!/usr/bin/env python3
"""USA endogenous closed loop: LOPF λ → E01 Full flex → merge → LOPF.

Round budget (this machine, historical E01 wall times):
  - USA LOPF 24h/168h: seconds–tens of seconds
  - Nationwide E01 Full-flex-only × 1 operator: often 0.5–2+ h
  - Full 6-operator season ≈ several hours → ~1 closed-loop round / night

Usage examples:
  # Export λ from existing Full-flex 168h summer LOPF (iter 00)
  py -3 .../run_closed_loop.py export-lmp --tag priority_168h --season summer --iter 0

  # Run E01 Full flex for all ops under that λ
  py -3 .../run_closed_loop.py e01 --season summer --iter 1

  # Merge + LOPF
  python .../run_closed_loop.py lopf --season summer --iter 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from experiments.lib.paths import package_root
ROOT = package_root()
sys.path.insert(0, str(ROOT))

NATIONWIDE = ROOT / "experiments/E05"
E02 = NATIONWIDE
LOOP = E02 / "results" / "closed_loop"
FULL_FLEX = "reallocation_slack_termination"
OPERATORS = ["coreweave", "oracle", "google", "meta", "microsoft", "amazon"]


def _iter_dir(iter_k: int) -> Path:
    return LOOP / f"iter_{iter_k:02d}"


def cmd_export_lmp(args: argparse.Namespace) -> None:
    os.environ["TEXAS_STUDY_ROOT"] = str(NATIONWIDE)
    os.environ["TEXAS_OPERATORS_JSON"] = str(NATIONWIDE / "config/nationwide_operators.json")

    import pandas as pd

    from experiments.lib.lmp_io import HOURS_DEFAULT, SEASONS, load_hourly_prices, write_hourly_lmp_csv
    from experiments.lib.texas_config import get_site_lmp_csv

    if args.from_iter is not None:
        src = _iter_dir(args.from_iter) / "lopf" / args.season / "site_lmp_hourly.csv"
    else:
        src = E02 / "results" / args.tag / args.season / FULL_FLEX / "site_lmp_hourly.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}; run open-loop LOPF or prior closed-loop lopf first")
    df = pd.read_csv(src)
    site_cols = [c for c in df.columns if c.startswith("site_")]
    out = _iter_dir(args.iter) / "lmp"
    out.mkdir(parents=True, exist_ok=True)
    bus_map = json.loads((E02 / "config/dc_bus_map.json").read_text(encoding="utf-8"))
    start = SEASONS[args.season]["start_utc"]
    n_eval = len(df)
    stats = []
    for col in site_cols:
        sid = int(col.replace("site_", ""))
        lam = [float(v) for v in df[col].tolist()]
        hist_path = get_site_lmp_csv(sid, args.season, allow_override=False)
        if hist_path is None or not hist_path.is_file():
            raise FileNotFoundError(hist_path)
        hist = load_hourly_prices(hist_path, HOURS_DEFAULT)
        prices: list[float] = []
        last = float(hist[0]) if hist else 0.0
        for i in range(HOURS_DEFAULT):
            if i < n_eval:
                raw = lam[i] if i < len(lam) else last
                if raw != raw:
                    raw = hist[i] if i < len(hist) else last
                else:
                    raw = float(raw)
                    last = float(raw)
                prices.append(float(raw))
            elif i < n_eval + 24:
                prices.append(last)
            else:
                prices.append(float(hist[i] if i < len(hist) else last))
        write_hourly_lmp_csv(
            prices,
            out / f"site_{sid}" / f"{args.season}.csv",
            location=f"pypsa:{bus_map.get(str(sid), '?')}",
            start=start,
            price_column="lmp",
        )
        stats.append({"site_id": sid, "lambda_mean": sum(lam) / max(len(lam), 1)})
    meta = {
        "source_site_lmp": str(src.relative_to(ROOT)).replace("\\", "/"),
        "season": args.season,
        "n_eval": n_eval,
        "sites": stats,
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[export] wrote {out} ({len(site_cols)} sites)")


def cmd_e01(args: argparse.Namespace) -> None:
    os.environ["TEXAS_STUDY_ROOT"] = str(NATIONWIDE)
    os.environ["TEXAS_OPERATORS_JSON"] = str(NATIONWIDE / "config/nationwide_operators.json")
    lmp_dir = _iter_dir(args.iter - 1) / "lmp"
    if not lmp_dir.exists():
        raise SystemExit(f"Missing LMP dir {lmp_dir}; run export-lmp for iter {args.iter - 1}")
    os.environ["TEXAS_LMP_OVERRIDE_DIR"] = str(lmp_dir.resolve())

    from experiments.lib.solve.runner import run_operator_study

    out_root = _iter_dir(args.iter) / "e01"
    ops = [args.operator] if args.operator else list(OPERATORS)
    for op in ops:
        print(f"[e01] iter={args.iter} op={op} season={args.season} lmp={lmp_dir}", flush=True)
        run_operator_study(
            op,
            args.season,
            scenario_names=(FULL_FLEX,),
            results_root=out_root / op,
            make_plots=False,
        )
    print(f"[e01] done -> {out_root}")


def cmd_lopf(args: argparse.Namespace) -> None:
    from experiments.E05.merge_power import (
        merge_season_scenario,
    )
    from experiments.lib.solve.usa_lopf import (
        solve_scenario,
        write_lmp_artifacts,
    )

    iter_dir = _iter_dir(args.iter)
    e01_root = iter_dir / "e01"
    merged_dir = iter_dir / "merged_power"
    info = merge_season_scenario(args.season, FULL_FLEX, merged_dir, e01_root)
    power_csv = ROOT / info["output_csv"]
    bus_map = json.loads((E02 / "config/dc_bus_map.json").read_text(encoding="utf-8"))
    print(f"[lopf] {power_csv}", flush=True)
    metrics, n = solve_scenario(
        season=args.season,
        scenario=FULL_FLEX,
        bus_map=bus_map,
        power_csv=power_csv,
        eval_start=0,
        eval_length=args.eval_length,
        solver="gurobi",
    )
    out = iter_dir / "lopf" / args.season
    paths = write_lmp_artifacts(out, n, bus_map, metrics=metrics)
    metrics["artifacts"] = paths
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    # also refresh λ for next round
    site_csv = out / "site_lmp_hourly.csv"
    # stash under this iter as the λ source for iter+1 export convenience
    print(f"[lopf] ok runtime={metrics.get('runtime_seconds'):.1f}s campus_mae={metrics.get('campus_mean_abs_delta_lmp')}")
    print(f"[lopf] site_lmp={site_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("export-lmp")
    p0.add_argument("--tag", default="priority_168h")
    p0.add_argument("--season", choices=["winter", "summer"], default="summer")
    p0.add_argument("--iter", type=int, default=0, help="Output iter folder for LMP CSVs")
    p0.add_argument(
        "--from-iter",
        type=int,
        default=None,
        help="If set, read site_lmp from closed_loop/iter_XX/lopf instead of open-loop tag",
    )
    p0.set_defaults(func=cmd_export_lmp)

    p1 = sub.add_parser("e01")
    p1.add_argument("--season", choices=["winter", "summer"], default="summer")
    p1.add_argument("--iter", type=int, default=1)
    p1.add_argument("--operator", default=None)
    p1.set_defaults(func=cmd_e01)

    p2 = sub.add_parser("lopf")
    p2.add_argument("--season", choices=["winter", "summer"], default="summer")
    p2.add_argument("--iter", type=int, default=1)
    p2.add_argument("--eval-length", type=int, default=168)
    p2.set_defaults(func=cmd_lopf)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
