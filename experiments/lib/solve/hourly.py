"""Hourly active-job / CPU / power series from block-job traces (no MILP)."""

from __future__ import annotations

import numpy as np

KAPPA_HIGH = 1.0


def hourly_series(
    jobs: dict,
    job_active: dict,
    hours: int,
    l_bar: float,
    p_idle_tot: float,
    p_busy_span: float,
    *,
    workload_scale: float = 1.0,
    eval_freeze: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build hourly trace series: active count, CPU load, power, util."""
    active = np.zeros(hours, dtype=int)
    cpu_load = np.zeros(hours, dtype=float)
    power = np.zeros(hours, dtype=float)
    util = np.zeros(hours, dtype=float)
    by_hour: dict[int, list[int]] = {t: [] for t in range(hours)}
    for (j, t), flag in job_active.items():
        if flag and 0 <= t < hours:
            by_hour[t].append(j)
    freeze_h0, freeze_h1, submit_cutoff = eval_freeze or (-1, -1, -1)
    for t in range(hours):
        js = by_hour[t]
        if eval_freeze is not None and freeze_h0 <= t < freeze_h1:
            js = [j for j in js if jobs[j].submit_hour_abs < submit_cutoff]
        active[t] = len(js)
        load = sum(float(jobs[j].cpu_total) for j in js) * workload_scale
        cpu_load[t] = load
        util[t] = min(1.0, load / l_bar) if l_bar > 0 else 0.0
        power[t] = p_idle_tot + p_busy_span * util[t]
    return active, cpu_load, power, util
