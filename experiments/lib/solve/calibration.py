"""N_cpu calibration and MILP-consistent util/power series (E01 + trace_validation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.lib.texas_config import (
    get_trace_watts_per_cpu_target,
    load_texas7_config,
    target_util,
)
from experiments.lib.solve.system_data import build_operator_system_data
from experiments.lib.solve.hourly import KAPPA_HIGH, hourly_series

HOURS_EVAL = 168


@dataclass(frozen=True)
class TraceCalibration:
    n_cpu_by_site: dict[int, float]
    job_scale: float
    wscale: float
    vcpu_to_fleet_scale: float
    site_peak_targets: dict[int, float]
    util_mean: float
    util_min: float
    util_max: float
    util_swing: float


def power_totals(operator_id: str) -> tuple[float, float, float]:
    sd = build_operator_system_data(operator_id, "summer", days=14, spot_extra_hours=0)
    sites = sd["dc"]["sites"]
    p_idle = sum(float(sd["dc"]["PUE"][s]) * float(sd["dc"]["Pfixed_MW"][s]) for s in sites)
    span = sum(
        float(sd["dc"]["PUE"][s])
        * (float(sd["dc"]["capacity_MW"][s]) - float(sd["dc"]["Pfixed_MW"][s]))
        for s in sites
    )
    l_bar = sum(KAPPA_HIGH * float(sd["dc"]["N_cpu"][s]) for s in sites)
    return p_idle, span, l_bar


def nominal_n_cpu_by_site(operator_id: str, job_sample_mod: int | None = None) -> dict[int, float]:
    """Nominal fleet ``N_cpu`` by site (MW-scaled nameplate)."""
    sd = build_operator_system_data(operator_id, "summer", days=7, spot_extra_hours=0)
    return {int(s): float(v) for s, v in sd["dc"]["N_cpu"].items()}


def effective_l_bar(l_bar: float, job_sample_mod: int | None = None) -> float:
    """Legacy helper; sample compensation is applied to job load, not ``l_bar``."""
    return l_bar


def apply_trace_sample_compensation(jobs: dict, job_sample_mod: int | None) -> float:
    """Deprecated: sampling debiasing is folded into ``vcpu_to_fleet_scale``."""
    mod = float(job_sample_mod) if job_sample_mod is not None and job_sample_mod > 1 else 1.0
    return mod


def _mean_trace_load(
    jobs: dict,
    job_active: dict,
    hours: int,
    calibrate_slice: slice,
) -> float:
    _, cpu, _, _ = hourly_series(jobs, job_active, hours, 1.0, 0.0, 1.0)
    return float(cpu[calibrate_slice].mean())


def vcpu_to_fleet_scale(
    operator_id: str,
    jobs: dict,
    job_active: dict,
    hours: int,
    *,
    job_sample_mod: int | None = None,
    calibrate_slice: slice | None = None,
) -> float:
    """Optional diagnostic scale if ``trace_watts_per_cpu_target`` is set; default path returns 1."""
    target_w = get_trace_watts_per_cpu_target()
    if target_w is None:
        return 1.0

    mu = util_mean_cap(operator_id)
    _, p_span, _ = power_totals(operator_id)
    sl = calibration_slice(hours, calibrate_slice=calibrate_slice)
    load_mean = _mean_trace_load(jobs, job_active, hours, sl)
    if load_mean < 1e-9:
        return 1.0

    l_bar_target = p_span * mu * 1e6 / float(target_w)
    return float(mu * l_bar_target / load_mean)


def util_mean_cap(operator_id: str | None = None) -> float:
    cfg = load_texas7_config()
    cap = float(cfg.get("trace_util_mean_hi", 0.70))
    if operator_id:
        op = cfg.get("operators", {}).get(operator_id, {})
        if "trace_util_mean_target" in op:
            return min(cap, float(op["trace_util_mean_target"]))
    return cap


def util_mean_band() -> tuple[float, float]:
    cfg = load_texas7_config()
    return float(cfg.get("trace_util_mean_lo", 0.35)), float(cfg.get("trace_util_mean_hi", 0.70))


def util_mean_target() -> float:
    return float(load_texas7_config().get("trace_util_mean_target", 0.50))


def util_peak_target() -> float:
    return float(load_texas7_config().get("trace_util_peak_target", 0.90))


def util_peak_target_for(operator_id: str) -> float:
    cfg = load_texas7_config()
    cap = float(cfg.get("trace_util_peak_cap", 1.0))
    peak = float(cfg.get("trace_util_peak_target", 0.98))
    op = cfg.get("operators", {}).get(operator_id, {})
    if "trace_util_peak_target" in op:
        peak = float(op["trace_util_peak_target"])
    return min(cap, peak)


def util_peak_cap() -> float:
    return float(load_texas7_config().get("trace_util_peak_cap", 1.0))


def util_floor_target() -> float:
    return float(load_texas7_config().get("trace_util_floor_target", 0.30))


def site_util_spread() -> float:
    return float(load_texas7_config().get("trace_site_util_spread", 0.0))


def calibrate_trim_hours() -> int:
    return int(load_texas7_config().get("trace_calibrate_trim_hours", 24))


def eval_week_slice(seg_start_day: int, eval_day_start: int) -> slice:
    h0 = (eval_day_start - seg_start_day) * 24
    return slice(h0, h0 + HOURS_EVAL)


def calibration_slice(hours: int, *, calibrate_slice: slice | None = None) -> slice:
    if calibrate_slice is not None:
        return calibrate_slice
    trim = calibrate_trim_hours()
    if trim > 0 and hours > 2 * trim:
        return slice(trim, hours - trim)
    return slice(0, hours)


def apply_job_workload_scale(jobs: dict, scale: float) -> None:
    if abs(scale - 1.0) < 1e-12:
        return
    for job in jobs.values():
        job.cpu_total = float(job.cpu_total) * scale


def _site_peak_util_targets(
    site_ids: list[int],
    *,
    peak_center: float,
    spread: float,
    cap: float,
) -> dict[int, float]:
    ordered = sorted(site_ids)
    if len(ordered) <= 1 or spread <= 1e-9:
        peak = min(peak_center, cap)
        return {s: peak for s in ordered}
    half = spread / 2.0
    lo = max(0.05, peak_center - half)
    hi = min(cap, peak_center + half)
    if hi <= lo + 1e-9:
        peak = min(peak_center, cap)
        return {s: peak for s in ordered}
    return {s: lo + (hi - lo) * i / (len(ordered) - 1) for i, s in enumerate(ordered)}


def _round_n_cpu_site(value: float, *, round_to: int = 100) -> float:
    if round_to <= 0:
        return value
    return float(max(round_to, int(round(value / round_to) * round_to)))


def _series_from_load(
    jobs: dict,
    job_active: dict,
    hours: int,
    l_bar: float,
    p_idle: float,
    p_span: float,
    *,
    workload_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active, cpu, _, _ = hourly_series(
        jobs, job_active, hours, l_bar, p_idle, p_span, workload_scale=workload_scale,
    )
    util = np.divide(cpu, l_bar, out=np.zeros_like(cpu), where=l_bar > 0)
    util = np.clip(util, 0.0, 1.0)
    power = p_idle + p_span * util
    return active, cpu, util, power


def _eval_load_stats(
    jobs: dict,
    job_active: dict,
    hours: int,
    l_bar_nom: float,
    p_idle: float,
    p_span: float,
    calibrate_slice: slice,
    *,
    workload_scale: float = 1.0,
) -> tuple[float, float, float]:
    _, cpu, _, _ = hourly_series(
        jobs, job_active, hours, l_bar_nom, p_idle, p_span, workload_scale=workload_scale,
    )
    load = cpu[calibrate_slice]
    return float(load.min()), float(load.max()), float(load.mean())


def _achieved_util(
    jobs: dict,
    job_active: dict,
    hours: int,
    l_bar: float,
    p_idle: float,
    p_span: float,
    job_scale: float,
) -> tuple[float, float, float]:
    _, _, util, _ = _series_from_load(
        jobs, job_active, hours, l_bar, p_idle, p_span, workload_scale=job_scale,
    )
    return float(util.min()), float(util.max()), float(util.mean())


def calibrate_trace(
    n_cpu_by_site: dict[int, float],
    jobs: dict,
    job_active: dict,
    hours: int,
    operator_id: str,
    *,
    calibrate_slice: slice | None = None,
    round_to: int = 100,
) -> TraceCalibration:
    """Size ``N_cpu`` only (``job_scale=1``); jobs keep raw ``cpu_total``."""
    p_idle, p_span, _ = power_totals(operator_id)
    l_bar_nom = float(sum(n_cpu_by_site.values()))
    mean_lo, mean_hi = util_mean_band()
    mean_cap = util_mean_cap(operator_id)
    peak_tgt = util_peak_target_for(operator_id)
    cap = util_peak_cap()
    spread = site_util_spread()
    sl = calibration_slice(hours, calibrate_slice=calibrate_slice)
    job_scale = 1.0

    load_min, load_max, load_mean = _eval_load_stats(
        jobs, job_active, hours, l_bar_nom, p_idle, p_span, sl, workload_scale=job_scale,
    )

    if load_mean < 1e-9 or l_bar_nom <= 1e-9 or load_max <= 1e-9:
        l_bar_agg = l_bar_nom
    else:
        l_bar_agg = load_max / peak_tgt
        mean_util = load_mean / l_bar_agg
        if mean_util > mean_cap:
            l_bar_agg = load_mean / mean_cap
        elif mean_util > mean_hi:
            l_bar_agg = load_mean / mean_hi
        elif mean_util < mean_lo:
            l_bar_agg = load_mean / mean_lo
        if load_max / l_bar_agg > cap:
            l_bar_agg = load_max / cap
        if load_max / l_bar_agg > peak_tgt:
            l_bar_agg = load_max / peak_tgt

    shares = {s: float(v) / l_bar_nom for s, v in n_cpu_by_site.items()}
    achieved_peak = min(load_max / l_bar_agg, cap) if l_bar_agg > 0 else peak_tgt
    peak_center = min(achieved_peak, peak_tgt, cap)
    peaks = _site_peak_util_targets(
        list(n_cpu_by_site.keys()), peak_center=peak_center, spread=spread, cap=cap,
    )

    weights = {s: shares[s] / max(peaks[s], 1e-9) for s in n_cpu_by_site}
    w_sum = sum(weights.values())
    n_cpu_cal = {
        s: l_bar_agg * weights[s] / w_sum if w_sum > 0 else l_bar_nom * shares[s]
        for s in n_cpu_by_site
    }
    n_cpu_cal = {s: _round_n_cpu_site(float(v), round_to=round_to) for s, v in n_cpu_cal.items()}
    l_bar_eff = float(sum(n_cpu_cal.values()))
    wscale = l_bar_nom / l_bar_eff if l_bar_eff > 1e-9 else 1.0

    mean_limit = mean_cap
    peak_limit = min(cap, peak_tgt)
    for _ in range(4):
        u_min, u_max, u_mean = _achieved_util(
            jobs, job_active, hours, l_bar_eff, p_idle, p_span, job_scale,
        )
        if u_mean <= mean_limit + 1e-6 and u_max <= peak_limit + 1e-6:
            break
        bump = 1.0
        if u_mean > mean_limit + 1e-6:
            bump = max(bump, u_mean / mean_limit)
        if u_max > peak_limit + 1e-6:
            bump = max(bump, u_max / peak_limit)
        n_cpu_cal = {
            s: _round_n_cpu_site(float(v) * bump, round_to=round_to) for s, v in n_cpu_cal.items()
        }
        l_bar_eff = float(sum(n_cpu_cal.values()))
        wscale = l_bar_nom / l_bar_eff if l_bar_eff > 1e-9 else 1.0

    u_min, u_max, u_mean = _achieved_util(
        jobs, job_active, hours, l_bar_eff, p_idle, p_span, job_scale,
    )
    return TraceCalibration(
        n_cpu_by_site=n_cpu_cal,
        job_scale=float(job_scale),
        wscale=float(wscale),
        vcpu_to_fleet_scale=1.0,
        site_peak_targets=peaks,
        util_mean=u_mean,
        util_min=u_min,
        util_max=u_max,
        util_swing=u_max - u_min,
    )


def implied_watts_per_cpu(operator_id: str, l_bar_eff: float, util_mean: float) -> float:
    """Diagnostic: dynamic facility W per calibrated ``N_cpu`` unit at ``util_mean``."""
    if l_bar_eff <= 1e-9:
        return 0.0
    _, p_span, _ = power_totals(operator_id)
    return float(p_span * util_mean * 1e6 / l_bar_eff)


def calibrate_trace_for_operator(
    operator_id: str,
    jobs: dict,
    job_active: dict,
    hours: int,
    *,
    job_sample_mod: int | None = None,
    n_cpu_by_site: dict[int, float] | None = None,
    calibrate_slice: slice | None = None,
    round_to: int = 100,
) -> TraceCalibration:
    """Calibrate ``N_cpu`` from **raw trace vCPU** load (no ``job.cpu_total`` scaling)."""
    del job_sample_mod  # sampling affects |jobs| only; MILP uses raw per-job vCPU
    n_cpu_nom = n_cpu_by_site or nominal_n_cpu_by_site(operator_id)
    cal = calibrate_trace(
        n_cpu_nom,
        jobs,
        job_active,
        hours,
        operator_id,
        calibrate_slice=calibrate_slice,
        round_to=round_to,
    )
    return TraceCalibration(
        n_cpu_by_site=cal.n_cpu_by_site,
        job_scale=cal.job_scale,
        wscale=cal.wscale,
        vcpu_to_fleet_scale=1.0,
        site_peak_targets=cal.site_peak_targets,
        util_mean=cal.util_mean,
        util_min=cal.util_min,
        util_max=cal.util_max,
        util_swing=cal.util_swing,
    )


def diagnostic_series(
    jobs: dict,
    job_active: dict,
    hours: int,
    operator_id: str,
    job_sample_mod: int | None,
    *,
    n_cpu_by_site: dict[int, float] | None = None,
    calibrate_slice: slice | None = None,
):
    """Return (active, cpu, power, util, l_bar_eff, wscale, p_idle, p_span, job_scale)."""
    p_idle, p_span, _ = power_totals(operator_id)
    cal = calibrate_trace_for_operator(
        operator_id,
        jobs,
        job_active,
        hours,
        job_sample_mod=job_sample_mod,
        n_cpu_by_site=n_cpu_by_site,
        calibrate_slice=calibrate_slice,
    )
    l_bar_eff = float(sum(cal.n_cpu_by_site.values()))
    active, cpu, util, power = _series_from_load(
        jobs, job_active, hours, l_bar_eff, p_idle, p_span, workload_scale=cal.job_scale,
    )
    return active, cpu, power, util, l_bar_eff, cal.wscale, p_idle, p_span, cal.job_scale
