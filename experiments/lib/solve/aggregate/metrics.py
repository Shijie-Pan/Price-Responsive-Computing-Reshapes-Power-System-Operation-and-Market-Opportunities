"""Metrics comparing aggregate vs job-level power profiles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from experiments.lib.solve.aggregate.power_io import Profile


def energy_cost_mwh(profile: Profile, prices: dict[int, list[float]], sites: list[int], hours: int) -> float:
    return sum(profile[s].get(t, 0.0) * prices[s][t] for s in sites for t in range(hours))


def peak_mw(profile: Profile, sites: list[int], hours: int, *, total: bool = False) -> float:
    if total:
        return max(sum(profile[s].get(t, 0.0) for s in sites) for t in range(hours))
    return max(profile[s].get(t, 0.0) for s in sites for t in range(hours))


def max_ramp(profile: Profile, sites: list[int], hours: int) -> float:
    mx = 0.0
    for s in sites:
        for t in range(1, hours):
            mx = max(mx, abs(profile[s].get(t, 0.0) - profile[s].get(t - 1, 0.0)))
    return mx


def total_energy_mwh(profile: Profile, sites: list[int], hours: int) -> float:
    return sum(profile[s].get(t, 0.0) for s in sites for t in range(hours))


def site_energy_mwh(profile: Profile, sites: list[int], hours: int) -> dict[int, float]:
    return {s: sum(profile[s].get(t, 0.0) for t in range(hours)) for s in sites}


def sync_coefficient(profile: Profile, sites: list[int], hours: int) -> float:
    if len(sites) < 2:
        return 1.0
    mats = [np.array([profile[s].get(t, 0.0) for t in range(hours)]) for s in sites]
    corrs = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            if np.std(mats[i]) < 1e-9 or np.std(mats[j]) < 1e-9:
                corrs.append(1.0)
            else:
                corrs.append(float(np.corrcoef(mats[i], mats[j])[0, 1]))
    return float(np.mean(corrs))


def profile_rmse_vs_ref(candidate: Profile, reference: Profile, sites: list[int], hours: int) -> float:
    errs = [(candidate[s].get(t, 0.0) - reference[s].get(t, 0.0)) ** 2 for s in sites for t in range(hours)]
    return math.sqrt(sum(errs) / max(len(errs), 1))


def peak_hour_spread(profile: Profile, sites: list[int], hours: int) -> float:
    peak_hours = [float(np.argmax([profile[s].get(t, 0.0) for t in range(hours)])) for s in sites]
    return float(np.std(peak_hours))


def relative_error(candidate: float, reference: float) -> float | None:
    if abs(reference) < 1e-9:
        return None
    return (candidate - reference) / reference


def summarize_profile(
    name: str,
    profile: Profile,
    *,
    sites: list[int],
    hours: int,
    prices: dict[int, list[float]],
    reference: Profile | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cost = energy_cost_mwh(profile, prices, sites, hours)
    out: dict[str, Any] = {
        "model": name,
        "energy_cost_usd": round(cost, 2),
        "total_energy_mwh": round(total_energy_mwh(profile, sites, hours), 2),
        "site_energy_mwh": {str(s): round(v, 2) for s, v in site_energy_mwh(profile, sites, hours).items()},
        "peak_total_mw": round(peak_mw(profile, sites, hours, total=True), 4),
        "peak_site_mw": round(peak_mw(profile, sites, hours, total=False), 4),
        "max_ramp_mw": round(max_ramp(profile, sites, hours), 4),
        "sync_coefficient": round(sync_coefficient(profile, sites, hours), 4),
        "peak_hour_spread_h": round(peak_hour_spread(profile, sites, hours), 2),
    }
    if reference is not None:
        ref_cost = energy_cost_mwh(reference, prices, sites, hours)
        out["rmse_vs_full_job_mw"] = round(profile_rmse_vs_ref(profile, reference, sites, hours), 4)
        out["energy_cost_error_pct"] = round(100 * (cost - ref_cost) / ref_cost, 3) if ref_cost else None
        ref_peak = peak_mw(reference, sites, hours, total=True)
        ref_ramp = max_ramp(reference, sites, hours)
        out["peak_total_error_pct"] = (
            round(100 * relative_error(out["peak_total_mw"], ref_peak), 3) if ref_peak else None
        )
        out["max_ramp_error_pct"] = (
            round(100 * relative_error(out["max_ramp_mw"], ref_ramp), 3) if ref_ramp else None
        )
    if extra:
        out.update(extra)
    return out
