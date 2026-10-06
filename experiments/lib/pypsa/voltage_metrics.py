"""Post-LOPF AC power flow voltage violation metrics (PyPSA pf)."""

from __future__ import annotations

from typing import Any

DEFAULT_VMIN_PU = 0.94
DEFAULT_VMAX_PU = 1.06


def _violation_stats(
    v_mag: dict[str, float],
    *,
    vmin: float,
    vmax: float,
) -> dict[str, Any]:
    violation_count = 0
    max_deviation = 0.0
    total_deviation = 0.0
    violating_buses: list[str] = []

    for bus, v in v_mag.items():
        dev = 0.0
        if v < vmin:
            dev = vmin - float(v)
        elif v > vmax:
            dev = float(v) - vmax
        if dev > 0:
            violation_count += 1
            total_deviation += dev
            max_deviation = max(max_deviation, dev)
            violating_buses.append(bus)

    n_buses = len(v_mag)
    return {
        "voltage_violation_count": int(violation_count),
        "max_voltage_deviation_pu": round(max_deviation, 6),
        "mean_voltage_deviation_pu": round(total_deviation / violation_count, 6) if violation_count else 0.0,
        "n_buses_checked": int(n_buses),
        "voltage_limits_pu": [vmin, vmax],
        "violating_bus_ids": violating_buses[:50],
    }


def voltage_metrics_at_snapshot(
    n,
    snapshot_idx: int,
    *,
    vmin: float = DEFAULT_VMIN_PU,
    vmax: float = DEFAULT_VMAX_PU,
) -> dict[str, Any]:
    """Run AC-PF at one snapshot and count |V| violations."""
    if not len(n.snapshots):
        return _empty_metrics(vmin=vmin, vmax=vmax, note="no snapshots")

    idx = int(snapshot_idx)
    snap = n.snapshots[idx]
    try:
        # Slice the index so MultiIndex snapshot names are preserved (list([snap]) drops names).
        snaps = n.snapshots[idx : idx + 1]
        try:
            n.pf(snapshots=snaps, use_seed=True)
        except TypeError:
            n.pf(snapshots=snaps)
    except Exception as exc:
        return _empty_metrics(
            vmin=vmin,
            vmax=vmax,
            pf_converged=False,
            note=f"pf failed at snapshot {idx}: {exc}",
            snapshot_idx=idx,
            snapshot=str(snap),
        )

    if "v_mag_pu" not in n.buses_t:
        return _empty_metrics(
            vmin=vmin,
            vmax=vmax,
            pf_converged=False,
            note="v_mag_pu missing after pf",
            snapshot_idx=idx,
            snapshot=str(snap),
        )

    v_series = n.buses_t.v_mag_pu.loc[snap]
    v_mag = {str(bus): float(v_series[bus]) for bus in v_series.index}
    out = _violation_stats(v_mag, vmin=vmin, vmax=vmax)
    out.update(
        {
            "pf_converged": True,
            "pf_backend_voltage": "pypsa_ac_pf",
            "snapshot_idx": idx,
            "snapshot": str(snap),
        }
    )
    return out


def voltage_metrics_over_window(
    n,
    *,
    snapshot_indices: list[int] | None = None,
    vmin: float = DEFAULT_VMIN_PU,
    vmax: float = DEFAULT_VMAX_PU,
) -> dict[str, Any]:
    """Run AC-PF over snapshots and aggregate violation counts."""
    if snapshot_indices is None:
        snapshot_indices = list(range(len(n.snapshots)))

    if not snapshot_indices:
        return _empty_metrics(vmin=vmin, vmax=vmax, note="empty snapshot window")

    per_hour: list[dict[str, Any]] = []
    total_violations = 0
    max_dev = 0.0
    converged_hours = 0
    failed_hours = 0

    for idx in snapshot_indices:
        m = voltage_metrics_at_snapshot(n, idx, vmin=vmin, vmax=vmax)
        per_hour.append(m)
        if m.get("pf_converged"):
            converged_hours += 1
            total_violations += int(m.get("voltage_violation_count") or 0)
            max_dev = max(max_dev, float(m.get("max_voltage_deviation_pu") or 0.0))
        else:
            failed_hours += 1

    crit = max(
        (m for m in per_hour if m.get("pf_converged")),
        key=lambda m: int(m.get("voltage_violation_count") or 0),
        default=None,
    )

    out: dict[str, Any] = {
        "voltage_violation_count_total": int(total_violations),
        "max_voltage_deviation_pu_window": round(max_dev, 6),
        "pf_converged_hours": converged_hours,
        "pf_failed_hours": failed_hours,
        "n_snapshots_checked": len(snapshot_indices),
        "voltage_limits_pu": [vmin, vmax],
        "pf_backend_voltage": "pypsa_ac_pf",
    }

    if crit is not None:
        out["voltage_violation_count"] = int(crit.get("voltage_violation_count") or 0)
        out["max_voltage_deviation_pu"] = float(crit.get("max_voltage_deviation_pu") or 0.0)
        out["critical_voltage_snapshot_idx"] = crit.get("snapshot_idx")
        out["critical_voltage_snapshot"] = crit.get("snapshot")
        out["n_buses_checked"] = crit.get("n_buses_checked")
        out["violating_bus_ids"] = crit.get("violating_bus_ids", [])
        out["pf_converged"] = True
    else:
        out.update(_empty_metrics(vmin=vmin, vmax=vmax, note="no converged pf in window"))
        out.pop("note", None)

    return out


def attach_voltage_metrics(
    metrics: dict[str, Any],
    n,
    *,
    map_snapshot_idx: int | None = None,
    eval_all_hours: bool = False,
    vmin: float = DEFAULT_VMIN_PU,
    vmax: float = DEFAULT_VMAX_PU,
) -> dict[str, Any]:
    """Merge voltage metrics into an existing LOPF metrics dict."""
    crit_idx = map_snapshot_idx
    if crit_idx is None:
        crit_idx = metrics.get("map_snapshot_idx") or metrics.get("critical_snapshot_idx") or 0

    crit = voltage_metrics_at_snapshot(n, int(crit_idx), vmin=vmin, vmax=vmax)
    metrics["max_voltage_deviation"] = crit.get("max_voltage_deviation_pu")
    metrics.update({k: v for k, v in crit.items() if k not in metrics})

    if eval_all_hours and len(n.snapshots):
        window = voltage_metrics_over_window(n, vmin=vmin, vmax=vmax)
        metrics["voltage_violation_count_total"] = window.get("voltage_violation_count_total")
        metrics["max_voltage_deviation_pu_window"] = window.get("max_voltage_deviation_pu_window")
        metrics["pf_converged_hours"] = window.get("pf_converged_hours")
        metrics["pf_failed_hours"] = window.get("pf_failed_hours")

    return metrics


def _empty_metrics(
    *,
    vmin: float,
    vmax: float,
    pf_converged: bool = False,
    note: str = "",
    snapshot_idx: int | None = None,
    snapshot: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "voltage_violation_count": 0,
        "max_voltage_deviation_pu": 0.0,
        "mean_voltage_deviation_pu": 0.0,
        "n_buses_checked": 0,
        "voltage_limits_pu": [vmin, vmax],
        "violating_bus_ids": [],
        "pf_converged": pf_converged,
        "pf_backend_voltage": "pypsa_ac_pf",
        "notes": note or None,
    }
    if snapshot_idx is not None:
        out["snapshot_idx"] = snapshot_idx
    if snapshot is not None:
        out["snapshot"] = snapshot
    return out
