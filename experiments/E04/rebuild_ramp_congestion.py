#!/usr/bin/env python3
"""Rebuild ramp-limit congestion tables on the Fig. 4 LOPF footing.

A0 / A1 / no_dc use deposited Fig. 4 hourly loadings. B1_ramp* use local ramp
pickles when present, otherwise the package-local
``results/ramp_limit/b1_loadings`` tables. Binding threshold is 99.9%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


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

from experiments.lib.map_hour import load_hourly_frames  # noqa: E402
from experiments.lib.pf_compare import BINDING_PCT  # noqa: E402

FIG4_DIR = ROOT / "experiments/E04/results/flexibility/full"
RAMP_LOPF_DIR = ROOT / "experiments/E04/results/flexibility/ramp"
B1_LOADING_DIR = ROOT / "experiments/E04/results/ramp_limit/b1_loadings"
OUT_CSV = ROOT / "experiments/E04/results/ramp_limit/congestion_vs_reference.csv"

CASES = ["no_dc", "A0", "A1", "B1_ramp50", "B1_ramp25", "B1_ramp10"]
CASE_TO_FIG4 = {
    "no_dc": "no_dc",
    "A0": "baseline",
    "A1": "reallocation_slack_termination",
}


def _binding_frame(df: pd.DataFrame, *, threshold: float = BINDING_PCT) -> pd.DataFrame:
    return df.astype(float) >= (threshold - 1e-9)


def _load_b1_loading(case: str, season: str) -> pd.DataFrame | None:
    folder = B1_LOADING_DIR / case / season
    parquet = folder / "line_loading_percent.parquet"
    csv = folder / "line_loading_percent.csv"
    if parquet.is_file():
        return pd.read_parquet(parquet).astype(float)
    if csv.is_file():
        return pd.read_csv(csv, index_col=0).astype(float)
    return None


def _load_case_loading(case: str, season: str) -> pd.DataFrame:
    if case in CASE_TO_FIG4:
        frames, _meta = load_hourly_frames(FIG4_DIR, season)
        key = CASE_TO_FIG4[case]
        if key not in frames:
            raise FileNotFoundError(f"Fig. 4 missing {key} for {season} under {FIG4_DIR}")
        return frames[key].astype(float)

    ramp_frames, _ = load_hourly_frames(RAMP_LOPF_DIR, season)
    if case in ramp_frames:
        return ramp_frames[case].astype(float)

    b1 = _load_b1_loading(case, season)
    if b1 is not None:
        return b1
    raise FileNotFoundError(
        f"No loading for {case}/{season} in {RAMP_LOPF_DIR} or {B1_LOADING_DIR}"
    )


def rebuild(*, threshold: float = BINDING_PCT) -> Path:
    rows: list[dict] = []
    sources: list[dict] = []
    for season in ("summer", "winter"):
        ref = _binding_frame(_load_case_loading("A0", season), threshold=threshold)
        nodc = _binding_frame(_load_case_loading("no_dc", season), threshold=threshold)
        for case in CASES:
            cur_load = _load_case_loading(case, season)
            cur = _binding_frame(cur_load, threshold=threshold)
            aligned = cur.reindex(index=ref.index, columns=ref.columns).fillna(False)
            base = ref.reindex(index=aligned.index, columns=aligned.columns).fillna(False)
            added = int((aligned & ~base).to_numpy().sum())
            relieved = int((~aligned & base).to_numpy().sum())
            other = nodc.reindex(index=aligned.index, columns=aligned.columns).fillna(False)
            add_n = int((aligned & ~other).to_numpy().sum())
            rel_n = int((~aligned & other).to_numpy().sum())
            rows.append(
                {
                    "case": case,
                    "season": season,
                    "congested_line_hours": int(aligned.to_numpy().sum()),
                    "added_vs_A0": added,
                    "relieved_vs_A0": relieved,
                    "net_vs_A0": added - relieved,
                    "added_vs_no_dc": add_n,
                    "relieved_vs_no_dc": rel_n,
                    "net_vs_no_dc": add_n - rel_n,
                    "binding_pct": threshold,
                }
            )
            src = "fig4_hourly"
            if case not in CASE_TO_FIG4:
                ramp_frames, _ = load_hourly_frames(RAMP_LOPF_DIR, season)
                if case in ramp_frames:
                    src = "e04_ramp_lopf"
                else:
                    src = "b1_loadings"
            sources.append({"case": case, "season": season, "loading_source": src})
            print(
                f"{case:12} {season:6} added={added} relieved={relieved} "
                f"net={added - relieved} total={int(aligned.to_numpy().sum())} src={src}",
                flush=True,
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    src_path = OUT_CSV.with_name("congestion_vs_reference_sources.csv")
    pd.DataFrame(sources).to_csv(src_path, index=False)
    print(OUT_CSV.relative_to(ROOT).as_posix())
    return OUT_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=BINDING_PCT)
    args = parser.parse_args()
    rebuild(threshold=float(args.threshold))


if __name__ == "__main__":
    main()
