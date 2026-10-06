"""Texas E02 PyPSA LOPF with B2 load accounting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.lib.solve.load_accounting import embedded_installed_mw
from experiments.lib.solve.flex_power_io import load_merged_power_csv
from experiments.lib.texas_config import study_e02_root
from experiments.lib.scenarios import E01_SCENARIOS, E02_REF_SCENARIO

from experiments.lib.paths import package_root
ROOT = package_root()
E02_ROOT = study_e02_root()


def open_network_window(
    net_path: Path,
    *,
    eval_start: int,
    eval_length: int,
    season: str = "winter",
):
    import pypsa

    from experiments.lib.network_season import (
        prepare_seasonal_network,
    )

    n = pypsa.Network(str(net_path))
    return prepare_seasonal_network(
        n,
        season=season,
        network_path=Path(net_path),
        eval_start=eval_start,
        eval_length=eval_length,
    )


def _native_bus_load_mw(n, snap, bus: str) -> float:
    total = 0.0
    for load_name in n.loads.index:
        if str(n.loads.at[load_name, "bus"]) != bus:
            continue
        if str(load_name).startswith("dc_load_"):
            continue
        total += float(n.loads_t.p_set.loc[snap, load_name])
    return total


def _subtract_from_native_loads(n, snap, bus: str, amount: float) -> None:
    if amount <= 0:
        return
    names = [
        ln
        for ln in n.loads.index
        if str(n.loads.at[ln, "bus"]) == bus and not str(ln).startswith("dc_load_")
    ]
    if not names:
        return
    total = sum(float(n.loads_t.p_set.loc[snap, ln]) for ln in names)
    if total <= 0:
        return
    for ln in names:
        share = float(n.loads_t.p_set.loc[snap, ln]) / total
        cur = float(n.loads_t.p_set.loc[snap, ln])
        n.loads_t.p_set.loc[snap, ln] = max(0.0, cur - amount * share)


def inject_b2_dc_profile(
    n,
    power_full: dict[int, dict[int, float]],
    bus_map: dict[str, str],
    accounting: dict[str, Any],
    *,
    dc_offset: int,
    eval_length: int,
) -> None:
    """Apply B2: reduce native load by embedded installed, add full E01 MW."""
    site_cfg = accounting["sites"]
    buses = sorted(set(bus_map.values()))

    base_peak: dict[str, float] = {bus: 0.0 for bus in buses}
    base_profile: dict[tuple[str, int], float] = {}
    n_hours = min(eval_length, len(n.snapshots))
    for local_h in range(n_hours):
        snap = n.snapshots[local_h]
        for bus in buses:
            mw = _native_bus_load_mw(n, snap, bus)
            base_profile[(bus, local_h)] = mw
            base_peak[bus] = max(base_peak[bus], mw)

    for local_h in range(n_hours):
        snap = n.snapshots[local_h]
        dc_h = dc_offset + local_h
        sites_at_t = power_full.get(dc_h, {})
        for site_str, pypsa_bus in bus_map.items():
            cfg = site_cfg[site_str]
            installed = float(cfg["installed_mw"])
            base_mw = base_profile[(pypsa_bus, local_h)]
            embedded = embedded_installed_mw(
                base_mw,
                installed_mw=installed,
                base_peak_mw=base_peak[pypsa_bus],
            )
            _subtract_from_native_loads(n, snap, pypsa_bus, embedded)

            e01_mw = float(sites_at_t.get(int(site_str), 0.0))
            load_name = f"dc_load_{site_str}"
            if load_name not in n.loads.index:
                n.add("Load", load_name, bus=pypsa_bus)
            n.loads_t.p_set.loc[snap, load_name] = e01_mw


def metrics_from_network(
    n, *, map_snapshot_idx: int | None = None, season: str | None = None
) -> dict:
    from experiments.lib.pypsa.geo_network_plot import (
        edge_critical_hour_loading,
        edge_loading_at_snapshot,
    )
    from experiments.lib.map_hour import snapshot_label

    crit_idx, peak_edges = edge_critical_hour_loading(n)
    map_idx = crit_idx if map_snapshot_idx is None else int(map_snapshot_idx)
    edges = edge_loading_at_snapshot(n, map_idx)
    map_loadings = edges["loading_pct"] if len(edges) else None
    n_overloaded = int((map_loadings >= 100.0 - 1e-9).sum()) if map_loadings is not None else 0
    n_near = int((map_loadings >= 90.0).sum()) if map_loadings is not None else 0
    congested = []
    if map_loadings is not None and n_overloaded:
        hit = edges.loc[map_loadings >= 100.0 - 1e-9].sort_values("loading_pct", ascending=False)
        congested = [
            {
                "line": r.get("line"),
                "loading_pct": round(float(r["loading_pct"]), 4),
                "capacity_mw": round(float(r.get("capacity_mw") or 0.0), 3),
            }
            for r in hit.head(25).to_dict(orient="records")
        ]
    week_peak = float(peak_edges["loading_pct"].max()) if len(peak_edges) else 0.0
    map_peak = float(map_loadings.max()) if map_loadings is not None and len(map_loadings) else 0.0
    return {
        "week_critical_snapshot_idx": int(crit_idx),
        "week_peak_loading_pct": round(week_peak, 4),
        "critical_snapshot_idx": int(map_idx),
        "map_snapshot_idx": int(map_idx),
        "peak_loading_pct": round(map_peak, 4),
        "mean_loading_pct": round(float(edges["loading_pct"].mean()) if len(edges) else 0.0, 4),
        "n_overloaded": n_overloaded,
        "n_lines_at_or_above_90pct": n_near,
        "congested_lines": congested,
        "edges": edges.to_dict(orient="records"),
        "critical_snapshot": str(n.snapshots[map_idx]) if len(n.snapshots) else None,
        "map_snapshot": str(n.snapshots[map_idx]) if len(n.snapshots) else None,
        "map_snapshot_label": snapshot_label(n, map_idx, season),
        "week_critical_snapshot": str(n.snapshots[crit_idx]) if len(n.snapshots) else None,
    }


def _attach_voltage_metrics(
    metrics: dict,
    n,
    *,
    map_snapshot_idx: int | None,
    run_voltage_pf: bool,
    voltage_eval_all_hours: bool,
) -> None:
    if not run_voltage_pf:
        metrics["pf_backend_voltage"] = None
        return
    from experiments.lib.pypsa.voltage_metrics import attach_voltage_metrics

    attach_voltage_metrics(
        metrics,
        n,
        map_snapshot_idx=map_snapshot_idx,
        eval_all_hours=voltage_eval_all_hours,
    )


def solve_no_dc(
    *,
    network_path: Path,
    solver: str,
    eval_start: int,
    eval_length: int,
    season: str = "winter",
    run_voltage_pf: bool = True,
    voltage_eval_all_hours: bool = False,
) -> tuple[dict, object]:
    """Baseline grid LOPF with no DC load injection (B2 not applied)."""
    from experiments.lib.pypsa.lopf_prep import run_lopf

    n = open_network_window(
        network_path, eval_start=eval_start, eval_length=eval_length, season=season
    )
    metrics = run_lopf(n, solver_name=solver)
    geo = metrics_from_network(n, season=season)
    metrics.update(geo)
    _attach_voltage_metrics(
        metrics,
        n,
        map_snapshot_idx=geo.get("map_snapshot_idx"),
        run_voltage_pf=run_voltage_pf,
        voltage_eval_all_hours=voltage_eval_all_hours,
    )
    metrics["scenario"] = E02_REF_SCENARIO
    metrics["power_csv"] = None
    metrics["lopf_ok"] = metrics.get("solver_status") in (None, "ok", "optimal") or metrics.get(
        "objective_usd"
    ) is not None
    return metrics, n


def solve_scenario(
    *,
    network_path: Path,
    power_csv: Path,
    bus_map: dict[str, str],
    accounting: dict[str, Any],
    solver: str,
    eval_start: int,
    eval_length: int,
    dc_offset: int = 0,
    scenario: str,
    season: str = "winter",
    run_voltage_pf: bool = True,
    voltage_eval_all_hours: bool = False,
) -> tuple[dict, object]:
    from experiments.lib.pypsa.lopf_prep import run_lopf

    power_full = load_merged_power_csv(power_csv)
    n = open_network_window(
        network_path, eval_start=eval_start, eval_length=eval_length, season=season
    )
    inject_b2_dc_profile(
        n,
        power_full,
        bus_map,
        accounting,
        dc_offset=dc_offset,
        eval_length=eval_length,
    )
    metrics = run_lopf(n, solver_name=solver)
    geo = metrics_from_network(n, season=season)
    metrics.update(geo)
    _attach_voltage_metrics(
        metrics,
        n,
        map_snapshot_idx=geo.get("map_snapshot_idx"),
        run_voltage_pf=run_voltage_pf,
        voltage_eval_all_hours=voltage_eval_all_hours,
    )
    metrics["scenario"] = scenario
    metrics["power_csv"] = str(power_csv.relative_to(ROOT)).replace("\\", "/")
    metrics["lopf_ok"] = metrics.get("solver_status") in (None, "ok", "optimal") or metrics.get(
        "objective_usd"
    ) is not None
    return metrics, n


def _delta_voltage_vs_ref(portfolios: dict[str, dict], ref_key: str) -> None:
    ref = portfolios.get(ref_key, {})
    ref_v = int(ref.get("voltage_violation_count") or 0)
    ref_dev = float(ref.get("max_voltage_deviation_pu") or 0.0)
    ref_total = int(ref.get("voltage_violation_count_total") or ref_v)
    for sc, m in portfolios.items():
        if sc == ref_key:
            m["delta_voltage_violation_count_vs_no_dc"] = 0
            m["delta_max_voltage_deviation_pu_vs_no_dc"] = 0.0
            m["delta_voltage_violation_count_total_vs_no_dc"] = 0
            continue
        v = int(m.get("voltage_violation_count") or 0)
        dev = float(m.get("max_voltage_deviation_pu") or 0.0)
        total = int(m.get("voltage_violation_count_total") or v)
        m["delta_voltage_violation_count_vs_no_dc"] = v - ref_v
        m["delta_max_voltage_deviation_pu_vs_no_dc"] = round(dev - ref_dev, 6)
        m["delta_voltage_violation_count_total_vs_no_dc"] = total - ref_total


def run_flexibility_comparison(
    *,
    season: str,
    merged_dir: Path,
    config_dir: Path,
    results_dir: Path,
    solver: str = "gurobi",
    eval_start: int = 0,
    eval_length: int = 168,
    scenarios: list[str] | None = None,
    network_path: Path | None = None,
    run_voltage_pf: bool = True,
    voltage_eval_all_hours: bool = False,
    resume: bool = False,
) -> dict:
    pypsa_cfg = json.loads((config_dir / "pypsa_texas7.json").read_text(encoding="utf-8"))
    if network_path is None:
        network_path = ROOT / pypsa_cfg["network_path"]
    bus_map = json.loads((config_dir / "dc_bus_map.json").read_text(encoding="utf-8"))
    accounting = json.loads((config_dir / "dc_load_accounting.json").read_text(encoding="utf-8"))

    scenario_list = list(scenarios or E01_SCENARIOS)
    wanted = {E02_REF_SCENARIO, *scenario_list}
    portfolios: dict[str, dict] = {}
    networks: dict[str, object] = {}

    from experiments.lib.pypsa.geo_network_plot import hourly_loading_frame
    from experiments.lib.map_hour import (
        hourly_dir,
        load_hourly_frames,
        portfolios_from_hourly,
        save_hourly_frames,
        save_hourly_scenario,
        select_hour_max_change_vs_ref,
        snapshot_labels,
        strip_voltage_fields,
    )

    existing_frames, existing_meta = load_hourly_frames(results_dir, season)
    prev_json = results_dir / f"flexibility_comparison_{season}_full.json"
    prev_portfolios: dict[str, dict] = {}
    if prev_json.exists():
        prev_portfolios = json.loads(prev_json.read_text(encoding="utf-8")).get("portfolios") or {}

    def _has_hourly(sc: str) -> bool:
        return (hourly_dir(results_dir, season) / f"{sc}.pkl").exists()

    def _checkpoint(sc: str, n) -> None:
        save_hourly_scenario(
            hourly_loading_frame(n), results_dir=results_dir, season=season, scenario=sc
        )

    to_solve = [E02_REF_SCENARIO, *[s for s in scenario_list if s != E02_REF_SCENARIO]]
    skip = [s for s in to_solve if resume and _has_hourly(s)]
    print(
        f"[E02] {season} planned={to_solve} resume={resume} skip={skip}",
        flush=True,
    )

    if resume and _has_hourly(E02_REF_SCENARIO):
        print(f"[E02] {season}/{E02_REF_SCENARIO} resume (hourly pickle)", flush=True)
        portfolios[E02_REF_SCENARIO] = dict(prev_portfolios.get(E02_REF_SCENARIO) or {})
    else:
        print(f"[E02] {season}/{E02_REF_SCENARIO} LOPF (no DC)...", flush=True)
        ref_metrics, ref_net = solve_no_dc(
            network_path=network_path,
            solver=solver,
            eval_start=eval_start,
            eval_length=eval_length,
            season=season,
            run_voltage_pf=False,
            voltage_eval_all_hours=False,
        )
        portfolios[E02_REF_SCENARIO] = ref_metrics
        networks[E02_REF_SCENARIO] = ref_net
        _checkpoint(E02_REF_SCENARIO, ref_net)
        print(
            f"  -> ok={ref_metrics.get('lopf_ok')} peak={ref_metrics.get('peak_loading_pct')}% "
            f"obj={ref_metrics.get('objective_usd')}",
            flush=True,
        )

    for sc in scenario_list:
        if sc == E02_REF_SCENARIO:
            continue
        if resume and _has_hourly(sc):
            print(f"[E02] {season}/{sc} resume (hourly pickle)", flush=True)
            portfolios[sc] = dict(prev_portfolios.get(sc) or {})
            continue
        matches = sorted(
            merged_dir.glob(f"power_merged_{sc}_{season}_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            raise FileNotFoundError(f"No merged CSV for {sc}/{season} in {merged_dir}")
        csv_path = matches[0]
        print(f"[E02] {season}/{sc} LOPF...", flush=True)
        m, n = solve_scenario(
            network_path=network_path,
            power_csv=csv_path,
            bus_map=bus_map,
            accounting=accounting,
            solver=solver,
            eval_start=eval_start,
            eval_length=eval_length,
            scenario=sc,
            season=season,
            run_voltage_pf=False,
            voltage_eval_all_hours=False,
        )
        portfolios[sc] = m
        networks[sc] = n
        _checkpoint(sc, n)
        print(
            f"  -> ok={m.get('lopf_ok')} peak={m.get('peak_loading_pct')}% "
            f"obj={m.get('objective_usd')}",
            flush=True,
        )

    frames = {k: v for k, v in existing_frames.items() if k in wanted}
    frames.update({sc: hourly_loading_frame(n) for sc, n in networks.items()})
    if not frames:
        raise RuntimeError(f"No hourly frames for {season}: nothing solved and no pickles on disk")
    map_idx, map_score, hour_scores = select_hour_max_change_vs_ref(frames)
    if networks:
        labels = snapshot_labels(next(iter(networks.values())), season)
    else:
        labels = list(existing_meta.get("snapshots") or [])
    if not labels:
        static_for_labels = open_network_window(
            network_path, eval_start=eval_start, eval_length=eval_length, season=season
        )
        labels = snapshot_labels(static_for_labels, season)
    map_label = labels[map_idx] if labels else ""
    print(
        f"[E02] {season} map hour idx={map_idx} {map_label} "
        f"d_branches vs no-DC={map_score} (max over {len(hour_scores)} h)",
        flush=True,
    )
    save_hourly_frames(
        frames,
        results_dir=results_dir,
        season=season,
        labels=labels,
        map_idx=map_idx,
        map_label=map_label,
        selection="max_delta_branches_vs_no_dc",
        score=map_score,
        scores=hour_scores,
    )

    static_n = None
    for sc, df in frames.items():
        if sc in networks:
            geo = metrics_from_network(
                networks[sc], map_snapshot_idx=map_idx, season=season
            )
            portfolios.setdefault(sc, {}).update(geo)
            if run_voltage_pf:
                strip_voltage_fields(portfolios[sc])
                _attach_voltage_metrics(
                    portfolios[sc],
                    networks[sc],
                    map_snapshot_idx=map_idx,
                    run_voltage_pf=True,
                    voltage_eval_all_hours=voltage_eval_all_hours,
                )
            continue
        if static_n is None:
            static_n = open_network_window(
                network_path,
                eval_start=eval_start,
                eval_length=eval_length,
                season=season,
            )
        rebuilt = portfolios_from_hourly(
            {sc: df}, hour=map_idx, label=map_label, static_n=static_n
        )[sc]
        keep = portfolios.get(sc) or {}
        keep.update(rebuilt)
        portfolios[sc] = keep

    ref = portfolios.get(E02_REF_SCENARIO, {})
    for sc, m in portfolios.items():
        if sc == E02_REF_SCENARIO:
            m["delta_peak_loading_pct_vs_no_dc"] = 0.0
            continue
        peak = float(m.get("peak_loading_pct") or 0.0)
        ref_peak = float(ref.get("peak_loading_pct") or 0.0)
        m["delta_peak_loading_pct_vs_no_dc"] = round(peak - ref_peak, 4)
    if run_voltage_pf:
        _delta_voltage_vs_ref(portfolios, E02_REF_SCENARIO)

    out = {
        "case": "texas7",
        "season": season,
        "network": str(network_path),
        "load_accounting": accounting.get("mode"),
        "reference_scenario": E02_REF_SCENARIO,
        "window": {
            "eval_start": eval_start,
            "eval_length": eval_length,
            "map_snapshot_idx": map_idx,
            "map_snapshot_label": map_label,
            "map_snapshot_selection": "max_delta_branches_vs_no_dc",
            "map_impact_score": map_score,
        },
        "portfolios": portfolios,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"flexibility_comparison_{season}_full.json"
    # strip edges from JSON main file to keep size manageable; save per-scenario edge files
    slim = json.loads(json.dumps(out))
    for sc in slim["portfolios"]:
        slim["portfolios"][sc].pop("edges", None)
    out_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")

    full_path = results_dir / f"flexibility_comparison_{season}_full_with_edges.json"
    full_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    from experiments.lib.pf_maps import write_grid_health_summary

    write_grid_health_summary(slim["portfolios"], results_dir / f"grid_health_{season}.csv")
    print(f"[done] {out_path}")
    return out
