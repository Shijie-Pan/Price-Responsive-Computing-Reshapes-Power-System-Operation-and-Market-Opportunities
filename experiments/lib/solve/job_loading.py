"""Texas eval-window job loading (submit filters + carryover long-job cap)."""

from __future__ import annotations

from experiments.lib.texas_config import (
    get_operator_job_csv,
    get_trace_carryover_cap_policy,
    get_trace_carryover_long_min_hours,
    get_trace_max_eval_carryover_long_jobs,
    get_trace_min_submit_time_sec,
    get_trace_window,
)
from experiments.lib.solve.block_jobs_loader import BlockJob, load_block_jobs

CARRY_CAP_POLICIES = ("lightest", "prefer_ends_in_eval", "ends_in_eval_only")


def cap_eval_carryover_long_jobs(
    jobs: dict[int, BlockJob],
    job_active: dict[tuple[int, int], int],
    *,
    eval_start_hour_abs: int,
    horizon_hours: int,
    max_jobs: int | None,
    min_active_hours: int = 24,
    cap_policy: str = "lightest",
) -> tuple[dict[int, BlockJob], dict[tuple[int, int], int], int]:
    """Keep at most ``max_jobs`` pre-eval long jobs still running at eval week open."""
    if not max_jobs or max_jobs <= 0:
        return jobs, job_active, 0
    if cap_policy not in CARRY_CAP_POLICIES:
        raise ValueError(f"Unknown carryover cap policy {cap_policy!r}; expected {CARRY_CAP_POLICIES}")

    candidates: list[tuple[int, float, bool, int]] = []
    for jid, job in jobs.items():
        if job.submit_hour_abs >= eval_start_hour_abs:
            continue
        if job.start_hour_rel != 0:
            continue
        active_h = min(job.end_hour_rel, horizon_hours) - job.start_hour_rel
        if active_h < min_active_hours:
            continue
        ends_in_eval = job.end_hour_rel < horizon_hours
        candidates.append((jid, job.cpu_total * active_h, ends_in_eval, job.end_hour_rel))

    if len(candidates) <= max_jobs:
        return jobs, job_active, 0

    if cap_policy == "ends_in_eval_only":
        ending = [c for c in candidates if c[2]]
        ending.sort(key=lambda c: (c[3], c[1], c[0]))
        keep_set = {c[0] for c in ending[:max_jobs]}
    elif cap_policy == "prefer_ends_in_eval":
        ending = [c for c in candidates if c[2]]
        surviving = [c for c in candidates if not c[2]]
        ending.sort(key=lambda c: (c[3], c[1], c[0]))
        surviving.sort(key=lambda c: (c[1], c[0]))
        keep_ids: list[int] = []
        for jid, _, _, _ in ending:
            if len(keep_ids) >= max_jobs:
                break
            keep_ids.append(jid)
        for jid, _, _, _ in surviving:
            if len(keep_ids) >= max_jobs:
                break
            keep_ids.append(jid)
        keep_set = set(keep_ids)
    else:
        candidates.sort(key=lambda c: (c[1], c[0]))
        keep_set = {c[0] for c in candidates[:max_jobs]}

    drop = {c[0] for c in candidates if c[0] not in keep_set}
    jobs_out = {k: v for k, v in jobs.items() if k not in drop}
    job_active_out = {(j, h): v for (j, h), v in job_active.items() if j not in drop}
    return jobs_out, job_active_out, len(drop)


def load_eval_block_jobs(
    operator_id: str,
    *,
    eval_day_start: int | None = None,
    horizon_hours: int,
    spot_extra_hours: int = 0,
    job_id_mod: int | None = None,
    job_type_filter: str | list[str] | None = None,
    original_window_end: int | None = None,
    hp_to_spot_submit_time_sec: float | None = None,
    quiet: bool = False,
) -> tuple[dict[int, BlockJob], dict[tuple[int, int], int], dict]:
    """Load eval-window jobs with Texas trace filters and carryover long-job cap."""
    trace = get_trace_window(operator_id)
    day_start = int(eval_day_start if eval_day_start is not None else trace["day_start"])
    start_hour = day_start * 24
    job_types = job_type_filter if job_type_filter is not None else ["HP", "Spot"]
    win_end = original_window_end if original_window_end is not None else horizon_hours

    jobs, job_active = load_block_jobs(
        job_csv_path=str(get_operator_job_csv(operator_id)),
        start_hour=start_hour,
        horizon_hours=horizon_hours,
        job_type_filter=job_types,
        min_submit_time_sec=get_trace_min_submit_time_sec(),
        max_submit_time_sec=(day_start + 7) * 86400,
        spot_extra_hours=spot_extra_hours,
        hp_to_spot_submit_time_sec=hp_to_spot_submit_time_sec,
        job_id_mod=job_id_mod,
        original_window_end=win_end,
    )

    max_carry = get_trace_max_eval_carryover_long_jobs(operator_id)
    min_h = get_trace_carryover_long_min_hours()
    cap_policy = get_trace_carryover_cap_policy(operator_id)
    jobs, job_active, dropped = cap_eval_carryover_long_jobs(
        jobs,
        job_active,
        eval_start_hour_abs=start_hour,
        horizon_hours=win_end,
        max_jobs=max_carry,
        min_active_hours=min_h,
        cap_policy=cap_policy,
    )
    if dropped and not quiet:
        print(
            f"[loader] eval carryover long-job cap: kept {max_carry}, "
            f"dropped {dropped} (min_active_h={min_h}, policy={cap_policy})"
        )
    elif max_carry is None and not quiet:
        pre_long = sum(
            1
            for j in jobs.values()
            if j.submit_hour_abs < start_hour
            and j.start_hour_rel == 0
            and min(j.end_hour_rel, win_end) - j.start_hour_rel >= min_h
        )
        print(
            f"[loader] eval carryover long-job cap: off, kept all {pre_long} "
            f"(min_active_h={min_h}, policy={cap_policy})"
        )

    trace_out = dict(trace)
    trace_out["day_start"] = day_start
    trace_out["day_end"] = day_start + 7
    trace_out["start_hour"] = start_hour
    trace_out["trace_job_window_offset_days"] = day_start - int(trace["segment_day_start"])
    return jobs, job_active, trace_out
