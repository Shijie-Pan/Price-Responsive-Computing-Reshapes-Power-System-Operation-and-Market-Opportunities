"""Split the Alibaba spot-GPU trace into the operator job windows.

Download ``job_info_df.csv`` from the Alibaba cluster trace and place it at
``experiments/data/sources/job_info_df.csv``. This script keeps the A10 rows
and writes one CSV per trace window. It does not upload or redistribute the
trace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def _package_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "experiments" / "data" / "networks").is_dir() and (
            candidate / "experiments" / "lib"
        ).is_dir():
            return candidate
    raise RuntimeError("Cannot find the deposit root")


ROOT = _package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.job_pool import split_jobs_by_trace_windows  # noqa: E402
from experiments.lib.texas_config import load_texas7_config  # noqa: E402

SOURCE = ROOT / "experiments" / "data" / "sources" / "job_info_df.csv"
A10 = ROOT / "experiments" / "data" / "sources" / "job_info_df_A10.csv"
OUT = ROOT / "experiments" / "data" / "jobs"


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(
            "Place the Alibaba job_info_df.csv at experiments/data/sources/job_info_df.csv. "
            "See experiments/E03/README.md."
        )
    frame = pd.read_csv(SOURCE)
    if "gpu_model" not in frame.columns:
        raise SystemExit(f"{_relative(SOURCE)} has no gpu_model column")
    a10 = frame[frame["gpu_model"].astype(str) == "A10"].copy()
    if a10.empty:
        raise SystemExit("No A10 rows in job_info_df.csv")
    A10.parent.mkdir(parents=True, exist_ok=True)
    a10.to_csv(A10, index=False)

    manifest = split_jobs_by_trace_windows(A10, OUT, include_carryover=True)
    manifest["source_csv"] = _relative(A10)
    for info in manifest["trace_windows"].values():
        info["output_csv"] = _relative(Path(info["output_csv"]))
    manifest_path = OUT / "jobs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"A10 rows: {len(a10)}")
    print(f"windows: {_relative(OUT / 'by_trace')}")
    for operator_id, spec in load_texas7_config()["operators"].items():
        segment = spec["trace_segment"]
        print(
            f"  {operator_id}: experiments/data/jobs/by_trace/{segment}/job_info.csv"
        )


if __name__ == "__main__":
    main()
