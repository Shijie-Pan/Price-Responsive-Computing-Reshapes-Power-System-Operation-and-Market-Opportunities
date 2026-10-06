"""B2 load accounting: subtract installed proportional to PyPSA base load, add full E01 MW."""

from __future__ import annotations

from typing import Mapping


def embedded_installed_mw(
    base_mw: float,
    *,
    installed_mw: float,
    base_peak_mw: float,
) -> float:
    """P_embedded(t) = installed × P_base(t) / P_base_peak (capped at base_mw)."""
    if base_peak_mw <= 0 or installed_mw <= 0:
        return 0.0
    embedded = installed_mw * (base_mw / base_peak_mw)
    return min(embedded, base_mw)


def net_bus_load_mw(
    base_mw: float,
    e01_mw: float,
    *,
    installed_mw: float,
    base_peak_mw: float,
) -> float:
    """P_bus(t) = P_base(t) − P_embedded(t) + P_e01(t)."""
    embedded = embedded_installed_mw(
        base_mw, installed_mw=installed_mw, base_peak_mw=base_peak_mw
    )
    return max(0.0, base_mw - embedded) + e01_mw


def base_peak_by_bus(
    base_profile: Mapping[int, float],
) -> float:
    if not base_profile:
        return 0.0
    return max(float(v) for v in base_profile.values())
