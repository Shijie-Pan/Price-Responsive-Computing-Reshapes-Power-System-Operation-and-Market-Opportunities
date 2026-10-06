# workload/block_jobs_loader.py

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class BlockJob:
    """Container for a block job (HP or spot)."""
    job_id: int
    submit_hour_abs: int   # absolute hour index on the full trace
    start_hour_rel: int    # relative hour index in the optimization window
    end_hour_rel: int      # end (exclusive) in the optimization window
    cpu_total: float       # total CPU (per worker * number of workers)
    gpu_total: float       # total GPU (per worker * number of workers)
    job_type: str          # e.g. "HP" / "Spot"


def load_block_jobs(
    job_csv_path: str,
    start_hour: int = 1,
    horizon_hours: int = 25,
    job_type_filter: str | list[str] | None = ["HP", "Spot"],
    min_submit_time_sec: float = 0.0,
    max_submit_time_sec: float | None = None,
    spot_extra_hours: int = 24,
    hp_to_spot_submit_time_sec: float | None = None,
    job_id_mod: int | None = None,
    original_window_end: int | None = None,
) -> Tuple[Dict[int, BlockJob], Dict[Tuple[int, int], int]]:
    """
    Load block jobs (HP or spot) from an HPC trace and map them into a given
    optimization window [start_hour, start_hour + horizon_hours).

    Jobs submitted at or before `min_submit_time_sec` are dropped entirely.
    Job type filter can keep HP only, spot only, or both.

    Parameters
    ----------
    job_csv_path : str
        Path to the raw job_info_df.csv file.
    start_hour : int
        Absolute starting hour index of the optimization window.
    horizon_hours : int
        Length of the optimization window in hours.
    job_type_filter : str | list[str] | None
        Job type(s) to keep (e.g., "HP", ["HP","spot"]). If None, keep all.
    min_submit_time_sec : float
        Drop all jobs with submit_time <= min_submit_time_sec.
        For example, 0.0 removes all jobs that start exactly at time 0s.
    max_submit_time_sec : float | None
        If set, drop jobs with submit_time >= max_submit_time_sec (operator trace upper bound).

    Returns
    -------
    jobs : dict[int, BlockJob]
        Metadata for each job indexed by job_id.
    job_active : dict[(job_id, t), int]
        Activity indicator in the relative time index t ∈ {0, ..., H-1}.
        Only entries equal to 1 are stored explicitly.
    """
    # ------------------------------------------------------------------
    # 1. Read raw job table
    # ------------------------------------------------------------------
    df = pd.read_csv(job_csv_path)

    # Filter by job type:
    # - str: exact match
    # - list[str]: keep if in list (e.g., ["HP", "spot"])
    # - None: keep all
    if job_type_filter is not None and "job_type" in df.columns:
        if isinstance(job_type_filter, (list, tuple, set)):
            df = df[df["job_type"].isin(job_type_filter)].copy()
        else:
            df = df[df["job_type"] == job_type_filter].copy()

    # Optional: downsample by mod (allowed: 1, 2, 5, 10, 20, 100)
    if "job_name" not in df.columns:
        raise ValueError("Column 'job_name' not found in job CSV.")
    df["job_id_int"] = df["job_name"].astype(int)
    if job_id_mod is not None:
        allowed_mods = {1, 2, 4, 5, 10, 20, 100}
        if job_id_mod not in allowed_mods:
            raise ValueError(f"job_id_mod must be one of {sorted(allowed_mods)}, got {job_id_mod}")
        if job_id_mod > 1:
            df = df[df["job_id_int"] % int(job_id_mod) == 0].copy()

    # Drop jobs that already exist at or before min_submit_time_sec
    if "submit_time" not in df.columns:
        raise ValueError("Column 'submit_time' not found in job CSV.")
    df = df[df["submit_time"] > 0].copy()
    df = df[df["submit_time"] > min_submit_time_sec].copy()
    if max_submit_time_sec is not None:
        df = df[df["submit_time"] < max_submit_time_sec].copy()

    # Optional: convert early HP jobs to Spot for experimentation (after submit_time filter)
    hp_to_spot_count = 0
    if hp_to_spot_submit_time_sec is not None:
        mask_hp_early = (df.get("job_type") == "HP") & (df["submit_time"] <= hp_to_spot_submit_time_sec)
        hp_to_spot_count = int(mask_hp_early.sum())
        df.loc[mask_hp_early, "job_type"] = "Spot"

    # ------------------------------------------------------------------
    # 2. Map from seconds to absolute hour indices
    # ------------------------------------------------------------------
    df["submit_hour_abs"] = (df["submit_time"] // 3600).astype(int)
    df["duration_hours"] = np.ceil(df["duration"] / 3600.0).astype(int)
    df["end_hour_abs"] = df["submit_hour_abs"] + df["duration_hours"]

    # ------------------------------------------------------------------
    # 3. Restrict to the chosen optimization window [H_start, H_end)
    # ------------------------------------------------------------------
    H_start = int(start_hour)
    # All jobs (both HP and Spot) should be filtered by original_window_end only
    # The extension window (SPOT_EXTRA_HOURS) is only for allowing existing Spot jobs
    # to complete their execution, not for including new jobs submitted during extension
    if original_window_end is not None:
        H_end = H_start + int(original_window_end)
    else:
        H_end = H_start + int(horizon_hours)
    
    mask = (df["submit_hour_abs"] < H_end) & (df["end_hour_abs"] > H_start)
    df = df[mask].copy()

    # Optional quick stats (after all filters applied)
    print(f"[loader] rows after filters: {len(df)}")
    print(df["gpu_model"].value_counts(dropna=False))
    jt_counts = df["job_type"].value_counts(dropna=False)
    print(jt_counts)
    if hp_to_spot_count:
        print(f"[loader] HP converted to Spot: {hp_to_spot_count}")
    # Duration buckets (seconds)
    n = len(df)
    if n > 0:
        buckets = {
            "<1min": (df["duration"] < 60),
            "<10min": (df["duration"] >= 60) & (df["duration"] < 600),
            "<1hour": (df["duration"] >= 600) & (df["duration"] < 3600),
            "<1day": (df["duration"] >= 3600) & (df["duration"] < 86400),
            ">=1day": (df["duration"] >= 86400),
        }
        print("[loader] duration stats (seconds) after filters:")
        for k, m in buckets.items():
            cnt = int(m.sum())
            pct = 100.0 * cnt / n
            print(f"  {k}: {cnt:8d} ({pct:6.2f}%)")

    # Relative hour indices inside the window [0, horizon_hours)
    df["start_hour_rel"] = (df["submit_hour_abs"] - H_start).clip(lower=0)
    # Original window end: all jobs must finish by original_window_end (if specified)
    # Otherwise, use horizon_hours as before
    if original_window_end is not None:
        df["end_hour_rel"] = (df["end_hour_abs"] - H_start).clip(upper=original_window_end)
    else:
        df["end_hour_rel"] = (df["end_hour_abs"] - H_start).clip(upper=horizon_hours)

    # ------------------------------------------------------------------
    # 4. Build job objects and activity map
    # ------------------------------------------------------------------
    jobs: Dict[int, BlockJob] = {}
    job_active: Dict[Tuple[int, int], int] = {}

    for _, row in df.iterrows():
        j = int(row["job_id_int"])

        cpu_total = float(row["cpu_request"] * row["worker_num"])
        gpu_total = float(row["gpu_request"] * row["worker_num"])
        start_rel = int(row["start_hour_rel"])
        end_rel = int(row["end_hour_rel"])

        jobs[j] = BlockJob(
            job_id=j,
            submit_hour_abs=int(row["submit_hour_abs"]),
            start_hour_rel=start_rel,
            end_hour_rel=end_rel,
            cpu_total=cpu_total,
            gpu_total=gpu_total,
            job_type=str(row["job_type"]),
        )

        # Active window:
        # - HP: exactly [start_rel, end_rel)
        # - Spot: extend availability by spot_extra_hours after original_window_end to allow pause/resume,
        #   but runtime requirement remains the original duration.
        #   Extension window size = spot_extra_hours (fixed value)
        job_type_lower = str(row["job_type"]).lower()
        if job_type_lower == "spot":
            # Extension window size = spot_extra_hours (fixed value, not based on original length)
            extension_hours = spot_extra_hours
            # Available window: [start_rel, end_rel + extension_hours)
            end_active = min(horizon_hours, end_rel + extension_hours)
        else:
            end_active = end_rel

        for h in range(start_rel, end_active):
            job_active[(j, h)] = 1

    return jobs, job_active


# Backward compatibility alias
load_hp_block_jobs = load_block_jobs
