"""USA LOPF helpers: seasonal window, DC inject, duals, congestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.lib.paths import package_root
ROOT = package_root()
# Prefer the deposited network under experiments/data (public-repo path).
USA_NETWORK_DEPOSIT = ROOT / "experiments/data/networks/usa/elec_s500_c134c.nc"
USA_NETWORK = USA_NETWORK_DEPOSIT

PRIORITY_SCENARIOS = ("baseline", "reallocation_slack_termination")
SCENARIO_LABELS = {
    "no_dc": "No-DC",
    "baseline": "First come first serve",
    "reallocation_slack_termination": "Full flexibility",
}


def open_usa_network_window(
    *,
    season: str = "winter",
    eval_start: int = 0,
    eval_length: int = 24,
    network_path: Path | None = None,
):
    import pypsa

    from experiments.lib.network_season import (
        prepare_seasonal_network,
    )

    path = Path(network_path) if network_path else USA_NETWORK
    n = pypsa.Network(str(path))
    return prepare_seasonal_network(
        n,
        season=season,
        network_path=path,
        eval_start=eval_start,
        eval_length=eval_length,
    )


def inject_dc_profile(
    n,
    power_full: dict[int, dict[int, float]],
    bus_map: dict[str, str],
    *,
    dc_offset: int = 0,
    eval_length: int | None = None,
) -> float:
    """Add Epoch DC loads on top of ReEDS native demand (no B2 subtract)."""
    n_hours = len(n.snapshots) if eval_length is None else min(eval_length, len(n.snapshots))
    total_mwh = 0.0
    for local_h in range(n_hours):
        snap = n.snapshots[local_h]
        sites = power_full.get(dc_offset + local_h, {})
        for site_str, pypsa_bus in bus_map.items():
            load_name = f"dc_load_{site_str}"
            if load_name not in n.loads.index:
                n.add("Load", load_name, bus=pypsa_bus)
            mw = float(sites.get(int(site_str), 0.0))
            n.loads_t.p_set.loc[snap, load_name] = mw
            total_mwh += mw
    return total_mwh


def extract_bus_lmp_frame(n) -> pd.DataFrame:
    mp = n.buses_t.marginal_price
    if mp is None or mp.empty:
        raise RuntimeError("buses_t.marginal_price empty after LOPF")
    return mp.astype(float).copy()


def extract_site_lmps(n, bus_map: dict[str, str]) -> dict[int, list[float]]:
    mp = extract_bus_lmp_frame(n)
    out: dict[int, list[float]] = {}
    cols = {str(c): c for c in mp.columns}
    for sid_str, bus in bus_map.items():
        key = cols.get(str(bus), bus if bus in mp.columns else None)
        if key is None:
            raise KeyError(f"Bus {bus!r} missing from marginal_price")
        out[int(sid_str)] = [float(v) for v in mp[key].to_numpy().ravel()]
    return out


def congestion_summary(n) -> dict[str, float]:
    from experiments.lib.pypsa.geo_network_plot import hourly_loading_frame

    df = hourly_loading_frame(n)
    if df.empty:
        return {
            "mean_n_congested": 0.0,
            "max_n_congested": 0.0,
            "n_hours_any_congested": 0.0,
            "n_hours": float(len(n.snapshots)),
            "peak_loading_pct": 0.0,
            "mean_loading_pct": 0.0,
        }
    n_cong = (df >= 100.0 - 1e-9).sum(axis=1)
    return {
        "mean_n_congested": float(n_cong.mean()),
        "max_n_congested": float(n_cong.max()),
        "n_hours_any_congested": float((n_cong > 0).sum()),
        "n_hours": float(len(n_cong)),
        "peak_loading_pct": float(df.max().max()),
        "mean_loading_pct": float(df.mean().mean()),
    }


def lmp_spatial_metrics(mp: pd.DataFrame) -> dict[str, float]:
    """Summarize cross-bus LMP structure over the solved window."""
    arr = mp.to_numpy(dtype=float)
    # per-hour pairwise mean absolute difference (sampled upper triangle)
    mae_h = []
    for h in range(arr.shape[0]):
        v = arr[h]
        # mean |vi-vj| over i<j
        # use (n * mean(|x - mean|) * 2 * (n/(n-1))) is Gini-related; simpler:
        diffs = np.abs(v[:, None] - v[None, :])
        mae_h.append(float(diffs[np.triu_indices(len(v), k=1)].mean()))
    std_h = np.nanstd(arr, axis=1)
    return {
        "mean_abs_pairwise_lmp": float(np.mean(mae_h)),
        "mean_hourly_lmp_std": float(np.mean(std_h)),
        "max_hourly_lmp_std": float(np.max(std_h)),
        "global_lmp_mean": float(np.nanmean(arr)),
        "global_lmp_std": float(np.nanstd(arr)),
        "global_lmp_min": float(np.nanmin(arr)),
        "global_lmp_max": float(np.nanmax(arr)),
        "frac_clipped_ge_100": float(np.mean(np.abs(arr - 100.0) < 0.05)),
    }


def campus_mean_abs_delta(site_lmps: dict[int, list[float]]) -> float:
    ids = sorted(site_lmps)
    if len(ids) < 2:
        return 0.0
    mat = np.asarray([site_lmps[i] for i in ids], dtype=float)
    # sites x hours → mean over hours of mean pairwise |Δ|
    mae = []
    for h in range(mat.shape[1]):
        v = mat[:, h]
        diffs = np.abs(v[:, None] - v[None, :])
        mae.append(float(diffs[np.triu_indices(len(v), k=1)].mean()))
    return float(np.mean(mae))


def solve_scenario(
    *,
    season: str,
    scenario: str,
    bus_map: dict[str, str],
    power_csv: Path | None,
    eval_start: int = 0,
    eval_length: int = 24,
    dc_offset: int = 0,
    solver: str = "gurobi",
    network_path: Path | None = None,
) -> tuple[dict[str, Any], object]:
    from experiments.lib.pypsa.lopf_prep import run_lopf
    from experiments.lib.solve.flex_power_io import load_merged_power_csv

    n = open_usa_network_window(
        season=season,
        eval_start=eval_start,
        eval_length=eval_length,
        network_path=network_path,
    )
    total_dc = 0.0
    if scenario != "no_dc":
        if power_csv is None or not Path(power_csv).is_file():
            raise FileNotFoundError(f"Need power CSV for scenario {scenario}")
        power = load_merged_power_csv(Path(power_csv))
        total_dc = inject_dc_profile(
            n, power, bus_map, dc_offset=dc_offset, eval_length=eval_length
        )

    metrics = run_lopf(n, solver_name=solver)
    mp = extract_bus_lmp_frame(n)
    site_lmps = extract_site_lmps(n, bus_map)
    cong = congestion_summary(n)
    spatial = lmp_spatial_metrics(mp)
    metrics.update(cong)
    metrics.update(spatial)
    metrics["scenario"] = scenario
    metrics["scenario_label"] = SCENARIO_LABELS.get(scenario, scenario)
    metrics["season"] = season
    metrics["eval_start"] = int(eval_start)
    metrics["eval_length"] = int(len(n.snapshots))
    metrics["total_dc_mwh"] = round(float(total_dc), 3)
    metrics["campus_mean_abs_delta_lmp"] = round(campus_mean_abs_delta(site_lmps), 4)
    metrics["lopf_ok"] = metrics.get("solver_status") in (None, "ok", "optimal") or metrics.get(
        "objective_usd"
    ) is not None
    metrics["power_csv"] = (
        str(Path(power_csv).resolve().relative_to(ROOT)).replace("\\", "/")
        if power_csv
        else None
    )
    return metrics, n


def write_lmp_artifacts(
    out_dir: Path,
    n,
    bus_map: dict[str, str],
    *,
    metrics: dict[str, Any],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mp = extract_bus_lmp_frame(n)
    site_lmps = extract_site_lmps(n, bus_map)

    bus_csv = out_dir / "bus_lmp_hourly.csv"
    mp.to_csv(bus_csv, float_format="%.6f")

    site_csv = out_dir / "site_lmp_hourly.csv"
    site_df = pd.DataFrame({f"site_{sid}": vals for sid, vals in sorted(site_lmps.items())})
    site_df.index.name = "hour"
    site_df.to_csv(site_csv, float_format="%.6f")

    meta_path = out_dir / "metrics.json"
    slim = {k: v for k, v in metrics.items() if k != "edges"}
    meta_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")

    # bus coordinates for mapping
    buses = n.buses.copy()
    geo = pd.DataFrame(
        {
            "bus": buses.index.astype(str),
            "lon": buses["x"].astype(float).to_numpy(),
            "lat": buses["y"].astype(float).to_numpy(),
            "mean_lmp": [float(mp[c].mean()) if c in mp.columns else float("nan") for c in buses.index],
            "std_lmp": [float(mp[c].std()) if c in mp.columns else float("nan") for c in buses.index],
        }
    )
    geo_path = out_dir / "bus_lmp_summary.csv"
    geo.to_csv(geo_path, index=False, float_format="%.6f")

    return {
        "bus_lmp_hourly": str(bus_csv.relative_to(ROOT)).replace("\\", "/"),
        "site_lmp_hourly": str(site_csv.relative_to(ROOT)).replace("\\", "/"),
        "metrics": str(meta_path.relative_to(ROOT)).replace("\\", "/"),
        "bus_lmp_summary": str(geo_path.relative_to(ROOT)).replace("\\", "/"),
    }
