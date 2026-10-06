"""Core E03 comparison runner (scheduling layer)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from experiments.lib.solve.aggregate.scheduler import AggregateSpec, solve_aggregate
from experiments.lib.solve.aggregate.envelope import build_envelope_from_reference, operator_energy_pools
from experiments.lib.solve.aggregate.models import (
    FCFS_SCENARIO,
    FULL_JOB_SCENARIO,
    MODEL_NOTES,
    SCHEDULING_MODELS,
)
from experiments.lib.solve.aggregate.metrics import summarize_profile
from experiments.lib.solve.aggregate.power_io import (
    build_fixed_dc_profile,
    latest_merged_csv,
    merged_csv_to_profile,
    profile_to_merged_csv,
)
from experiments.lib.texas_config import OPERATOR_IDS, load_texas7_config, study_e03_root
from experiments.lib.solve.site_lmp import load_site_lmp

from experiments.lib.paths import package_root
ROOT = package_root()
E03_ROOT = study_e03_root()
DEFAULT_CFG = E03_ROOT / "config" / "comparison.yaml"


def load_config(path: Path | None = None) -> dict:
    p = path or DEFAULT_CFG
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _operator_site_map(cfg_texas7: dict) -> dict[str, list[int]]:
    ops = cfg_texas7.get("operators", {})
    return {
        op: [int(s) for s in (spec.get("site_ids") or spec.get("site_ids") or [])]
        for op, spec in ops.items()
    }


def _load_prices(season: str, sites: list[int], hours: int) -> dict[int, list[float]]:
    from experiments.lib.solve.grid_template import data as grid_template

    sd = {"dc": {"sites": sites, "day_ahead_prices": {}}}
    lmp_zones = {s: "ERCOT" for s in sites}
    load_site_lmp(sd, lmp_zones, hours=hours, season=season)
    return sd["dc"]["day_ahead_prices"]


def _power_csv_name(model: str, season: str) -> str:
    key = model.lower()
    if model == "FULL_JOB":
        return f"power_full_job_{season}.csv"
    return f"power_{key}_{season}.csv"


def run_season_comparison(season: str, *, cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    hours = int(cfg["hours"])
    sites = [int(s) for s in cfg["site_ids"]]
    merged_dir = ROOT / cfg["merged_power_dir"]
    results_dir = E03_ROOT / "results" / season
    results_dir.mkdir(parents=True, exist_ok=True)

    texas7 = load_texas7_config()
    op_sites = _operator_site_map(texas7)
    prices = _load_prices(season, sites, hours)

    fj_path = latest_merged_csv(merged_dir, cfg["full_job_scenario"], season)
    full_profile = merged_csv_to_profile(fj_path, sites, hours)

    p_min, p_max, _ramp, energy = build_envelope_from_reference(
        full_profile,
        sites,
        hours,
        p_min_floor=float(cfg["p_min_floor_mw"]),
        p_max_cap=float(cfg["p_max_cap_mw"]),
    )
    op_energy = operator_energy_pools(full_profile, op_sites, hours)

    models: dict[str, dict] = {}
    profiles: dict[str, dict] = {"FULL_JOB": full_profile}

    # --- Aggregate LPs ---
    agg_specs = [
        ("AGG_CONT", "none", None),
        ("AGG_ENERGY", "site", energy),
        ("AGG_POOL", "operator", op_energy),
    ]
    for name, mode, targets in agg_specs:
        spec = AggregateSpec(
            sites=sites,
            hours=hours,
            prices=prices,
            p_min=p_min,
            p_max=p_max,
            energy_mode=mode,  # type: ignore[arg-type]
            energy_target_mwh=targets if mode == "site" else None,
            operator_sites=op_sites if mode == "operator" else None,
            operator_energy_mwh=targets if mode == "operator" else None,
        )
        t0 = time.perf_counter()
        prof, meta = solve_aggregate(spec, solver=cfg.get("solver", "gurobi"))
        meta["runtime_seconds"] = round(time.perf_counter() - t0, 3)
        profiles[name] = prof
        out_csv = results_dir / _power_csv_name(name, season)
        profile_to_merged_csv(prof, out_csv, sites, hours)
        models[name] = summarize_profile(
            name, prof, sites=sites, hours=hours, prices=prices, reference=full_profile, extra=meta,
        )
        print(f"  [{season}] {name}: cost={models[name]['energy_cost_usd']:.0f} USD, "
              f"RMSE={models[name].get('rmse_vs_full_job_mw')} MW ({meta['runtime_seconds']}s)")

    # --- FULL_JOB (reference) ---
    fj_out = results_dir / _power_csv_name("FULL_JOB", season)
    profile_to_merged_csv(full_profile, fj_out, sites, hours)
    models["FULL_JOB"] = summarize_profile(
        "FULL_JOB",
        full_profile,
        sites=sites,
        hours=hours,
        prices=prices,
        reference=full_profile,
        extra={"source_csv": str(fj_path.relative_to(ROOT)).replace("\\", "/")},
    )

    # --- FCFS ---
    fcfs_path = latest_merged_csv(merged_dir, cfg.get("fcfs_scenario", FCFS_SCENARIO), season)
    fcfs_profile = merged_csv_to_profile(fcfs_path, sites, hours)
    profiles["FCFS"] = fcfs_profile
    profile_to_merged_csv(fcfs_profile, results_dir / _power_csv_name("FCFS", season), sites, hours)
    models["FCFS"] = summarize_profile(
        "FCFS", fcfs_profile, sites=sites, hours=hours, prices=prices, reference=full_profile,
        extra={"source_csv": str(fcfs_path.relative_to(ROOT)).replace("\\", "/")},
    )

    # --- FIXED_DC ---
    fixed = build_fixed_dc_profile(full_profile, sites, hours)
    profiles["FIXED_DC"] = fixed
    profile_to_merged_csv(fixed, results_dir / _power_csv_name("FIXED_DC", season), sites, hours)
    models["FIXED_DC"] = summarize_profile(
        "FIXED_DC", fixed, sites=sites, hours=hours, prices=prices, reference=full_profile,
    )

    envelope = {
        "p_min_mw": {str(s): round(p_min[s], 2) for s in sites},
        "p_max_mw": {str(s): round(p_max[s], 2) for s in sites},
        "hourly_ramp_limit": False,
        "energy_target_mwh": {str(s): round(energy[s], 2) for s in sites},
        "operator_energy_mwh": {op: round(v, 2) for op, v in op_energy.items()},
    }

    result: dict[str, Any] = {
        "case": "texas7",
        "season": season,
        "hours": hours,
        "sites": sites,
        "operators": op_sites,
        "full_job_scenario": cfg["full_job_scenario"],
        "envelope": envelope,
        "models": models,
        "model_definitions": {m: MODEL_NOTES[m] for m in SCHEDULING_MODELS},
        "necessity_summary": _necessity_summary(models),
    }

    out_json = results_dir / f"comparison_{season}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  -> {out_json}")
    return result


def _necessity_summary(models: dict[str, dict]) -> dict:
    fj = models.get("FULL_JOB", {})
    cont = models.get("AGG_CONT", {})
    eng = models.get("AGG_ENERGY", {})
    pool = models.get("AGG_POOL", {})
    return {
        "full_job_energy_cost_usd": fj.get("energy_cost_usd"),
        "agg_cont": {
            "cost_error_pct": cont.get("energy_cost_error_pct"),
            "peak_error_pct": cont.get("peak_total_error_pct"),
            "rmse_mw": cont.get("rmse_vs_full_job_mw"),
        },
        "agg_energy": {
            "cost_error_pct": eng.get("energy_cost_error_pct"),
            "peak_error_pct": eng.get("peak_total_error_pct"),
            "rmse_mw": eng.get("rmse_vs_full_job_mw"),
        },
        "agg_pool": {
            "cost_error_pct": pool.get("energy_cost_error_pct"),
            "peak_error_pct": pool.get("peak_total_error_pct"),
            "rmse_mw": pool.get("rmse_vs_full_job_mw"),
        },
        "interpretation": (
            "Aggregate models optimize continuous site power under LMP, assuming idle–busy "
            "load can shift smoothly in time and across sites. Even with matched energy "
            "(AGG_ENERGY per site, AGG_POOL per operator), RMSE / peak / ramp vs FULL_JOB "
            "show that job windows, discrete CPU allocation, and task-site affinity make "
            "the implementable flexibility set much smaller than aggregate models suggest."
        ),
    }
