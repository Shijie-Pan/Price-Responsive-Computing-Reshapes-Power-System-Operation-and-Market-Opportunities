from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SEC_PER_DAY = 86400

# E01 evaluation window length (hours = days * 24).
TRACE_JOB_WINDOW_DAYS = 7

# Skip the first N days inside each 14-day segment (steady-state eval vs cold-start ramp).
TRACE_JOB_WINDOW_OFFSET_DAYS = 0

# Canonical trace segments (Alibaba trace spans ~184 days).
TRACE_WINDOWS: dict[str, tuple[int, int]] = {
    "days_0_14": (0, 14),
    "days_14_28": (14, 28),
    "days_28_42": (28, 42),
    "days_42_56": (42, 56),
    "days_56_70": (56, 70),
    "days_70_84": (70, 84),
    # Texas7 steady-state 30-day segments (day 30–150)
    "days_30_60": (30, 60),
    "days_60_90": (60, 90),
    "days_90_120": (90, 120),
    "days_120_150": (120, 150),
    "days_122_152": (122, 152),
    # Texas6 MW-matched segments (2026-08 sweep)
    "days_0_30": (0, 30),
    "days_19_49": (19, 49),
    "days_75_105": (75, 105),
    "days_121_151": (121, 151),
    "days_155_185": (155, 185),
}

# One operator -> one non-overlapping segment (no reuse across operator names).
OPERATOR_TRACE: dict[str, str] = {
    "google": "days_0_14",
    "meta": "days_14_28",
    "coreweave": "days_28_42",
    "oracle": "days_42_56",
    "microsoft": "days_56_70",
    "amazon": "days_70_84",
}

TEXAS_OPERATOR_TRACE = {
    "microsoft": "days_0_30",
    "meta": "days_30_60",
    "oracle": "days_75_105",
    "coreweave": "days_121_151",
    "google": "days_155_185",
}
NATIONWIDE_OPERATOR_TRACE = {
    k: OPERATOR_TRACE[k] for k in ("coreweave", "oracle", "google", "meta", "microsoft", "amazon")
}


def split_jobs_by_trace_windows(
    source_csv: Path,
    output_dir: Path,
    *,
    min_submit_floor: float = -1.0,
    include_carryover: bool = True,
) -> dict:
    """Split Alibaba trace into canonical day-window CSVs under output_dir/by_trace/.

    Each segment CSV contains:
    - jobs submitted inside [day_start, day_end), and
    - optionally jobs submitted before day_start that are still running at segment open
      (steady-state backlog at the segment boundary).
    """
    df = pd.read_csv(source_csv)
    if "submit_time" not in df.columns:
        raise ValueError(f"Missing submit_time in {source_csv}")

    # submit_time==0 means missing/invalid submission timestamp in cluster trace
    n_invalid = int((df["submit_time"] <= 0).sum())
    df = df[df["submit_time"] > 0].copy()

    trace_dir = output_dir / "by_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "source_csv": str(source_csv),
        "dropped_invalid_submit_time_leq_0": n_invalid,
        "include_carryover": include_carryover,
        "trace_windows": {},
    }

    for key, (day_start, day_end) in TRACE_WINDOWS.items():
        min_sec = max(min_submit_floor, day_start * SEC_PER_DAY - 1)
        max_sec = day_end * SEC_PER_DAY
        in_segment = (df["submit_time"] > min_sec) & (df["submit_time"] < max_sec)
        if include_carryover:
            segment_open = day_start * SEC_PER_DAY
            still_running = df["submit_time"] + df["duration"] > segment_open
            carryover = (df["submit_time"] <= segment_open) & still_running
            sub = df[in_segment | carryover].copy()
            n_carry = int(carryover.sum())
        else:
            sub = df[in_segment].copy()
            n_carry = 0
        out_path = trace_dir / key / "job_info.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)

        hp = int((sub["job_type"] == "HP").sum()) if "job_type" in sub.columns else 0
        spot = int((sub["job_type"] == "Spot").sum()) if "job_type" in sub.columns else 0
        manifest["trace_windows"][key] = {
            "output_csv": str(out_path),
            "day_start": day_start,
            "day_end": day_end,
            "start_hour": day_start * 24,
            "job_count": int(len(sub)),
            "carryover_job_count": n_carry,
            "hp_count": hp,
            "spot_count": spot,
        }

    manifest_path = output_dir / "jobs_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def operator_job_csv(output_dir: Path, operator_id: str, mapping: dict[str, str]) -> str:
    key = mapping[operator_id]
    rel = output_dir / "by_trace" / key / "job_info.csv"
    return str(rel).replace("\\", "/")


def attach_operator_job_paths(cfg_operators: dict, output_dir: Path, mapping: dict[str, str]) -> None:
    for op_id in cfg_operators:
        key = mapping[op_id]
        cfg_operators[op_id]["trace_segment"] = key
        cfg_operators[op_id]["job_csv"] = operator_job_csv(output_dir, op_id, mapping).replace(
            str(output_dir.parent.parent) + "/", ""
        )
        # relative from repo root
        cfg_operators[op_id]["job_csv"] = str(
            Path("experiments/data/jobs/by_trace") / key / "job_info.csv"
        ).replace("\\", "/")
        cfg_operators[op_id]["jobs_manifest"] = "experiments/data/jobs/jobs_manifest.json"


def trace_window_from_operator_spec(
    op: dict,
    *,
    default_job_window_days: int = TRACE_JOB_WINDOW_DAYS,
    default_job_window_offset_days: int = TRACE_JOB_WINDOW_OFFSET_DAYS,
) -> dict:
    """Map operator trace config to E01 evaluation + job-load window.

    Evaluation uses ``[segment_start + offset, segment_start + offset + window)``.
    Job loading excludes jobs submitted before ``trace_min_submit_day`` (via
    ``get_trace_min_submit_time_sec``) and includes submissions through eval week.
    """
    segment_day_start = int(op["trace_day_start"])
    segment_day_end = int(op["trace_day_end"])
    job_window_days = int(op.get("trace_job_window_days", default_job_window_days))
    offset_days = int(op.get("trace_job_window_offset_days", default_job_window_offset_days))
    eval_day_start = segment_day_start + offset_days
    eval_day_end = eval_day_start + job_window_days
    if eval_day_end > segment_day_end:
        raise ValueError(
            f"trace eval window [{eval_day_start}, {eval_day_end}) exceeds operator segment "
            f"[{segment_day_start}, {segment_day_end})"
        )
    return {
        "segment_day_start": segment_day_start,
        "segment_day_end": segment_day_end,
        "trace_job_window_offset_days": offset_days,
        "day_start": eval_day_start,
        "day_end": eval_day_end,
        "job_window_days": job_window_days,
        "start_hour": eval_day_start * 24,
        # Segment CSV includes carryover; allow submissions through the eval week.
        "min_submit_time_sec": -1.0,
        "max_submit_time_sec": eval_day_end * SEC_PER_DAY,
    }
