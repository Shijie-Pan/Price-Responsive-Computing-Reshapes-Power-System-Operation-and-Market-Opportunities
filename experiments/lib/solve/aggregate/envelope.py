"""Per-site envelope bounds for aggregate LPs."""

from __future__ import annotations

from experiments.lib.solve.aggregate.power_io import Profile


def build_envelope_from_reference(
    ref: Profile,
    sites: list[int],
    hours: int,
    *,
    p_min_floor: float,
    p_max_cap: float,
    ramp_scale: float = 1.25,
    ramp_cap: float = 400.0,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    p_min, p_max, ramp, energy = {}, {}, {}, {}
    for s in sites:
        series = [ref[s].get(t, p_min_floor) for t in range(hours)]
        p_min[s] = max(p_min_floor, min(series))
        p_max[s] = min(p_max_cap, max(series))
        if p_max[s] < p_min[s]:
            p_max[s] = p_max_cap
        diffs = [abs(series[t] - series[t - 1]) for t in range(1, hours)]
        ramp[s] = min(ramp_cap, max(diffs) * ramp_scale if diffs else ramp_cap)
        energy[s] = sum(series)
    return p_min, p_max, ramp, energy


def operator_energy_pools(
    ref: Profile,
    operator_sites: dict[str, list[int]],
    hours: int,
) -> dict[str, float]:
    return {
        op: sum(ref[s].get(t, 0.0) for s in sites for t in range(hours))
        for op, sites in operator_sites.items()
    }
