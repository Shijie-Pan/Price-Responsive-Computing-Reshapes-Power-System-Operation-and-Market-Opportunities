"""Actual vs future E02 comparison figures (written under texas_UNUSED_FUTURE/E02)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from experiments.lib.pypsa.geo_network_plot import (
    DELTA_DEC,
    DELTA_INC,
    apply_geo_style,
    draw_geo_panel_abs_loading,
    draw_geo_panel_delta_loading,
)
from experiments.lib.figures.nc_style import (
    DISPLAY_SHORT,
    PORTFOLIO_COLORS,
    apply_figure_style,
    apply_nature_comm_style,
    panel_label,
    save_figure,
)
from experiments.lib.texas_config import REPO_ROOT
from experiments.lib.scenarios import (
    E01_SCENARIOS,
    E02_REF_SCENARIO,
    FCFS_SCENARIO,
    FULL_FLEX_SCENARIO,
)
from experiments.lib.map_hour import (
    load_hourly_frames,
    portfolios_from_hourly,
    select_hour_max_study_delta,
)
from experiments.lib.pf_maps import (
    MAP_DC_COORDS,
    MAP_DC_SIZE_CTX,
    SCALE,
    _health_subtitle,
    _operator_legend_handles,
    _platecarree_aspect,
    annotation_style_for,
    backfill_congestion_metrics,
    captioned_geo_figure,
    load_e02_geo_context,
    place_delta_legend,
    plot_scale_for,
    set_panel_caption,
    write_panel_caption,
)

ACTUAL_E02 = REPO_ROOT / "experiments" / "E04"
FUTURE_E02 = ACTUAL_E02

STUDY_ACTUAL = "#3C5488"
STUDY_FUTURE = "#E64B35"

SCENARIO_KEYS: tuple[str, ...] = (E02_REF_SCENARIO, *E01_SCENARIOS)
MAP_KEYS: tuple[str, ...] = (E02_REF_SCENARIO, "baseline", FULL_FLEX_SCENARIO)
DELTA_KEYS: tuple[str, ...] = ("baseline", "reallocation", FULL_FLEX_SCENARIO)


def _label(key: str) -> str:
    if key == E02_REF_SCENARIO:
        return "No DC"
    return DISPLAY_SHORT.get(key, key)


def _json_path(e02_root: Path, season: str, results_tag: str) -> Path:
    folder = e02_root / "results" / "flexibility" / results_tag
    with_edges = folder / f"flexibility_comparison_{season}_full_with_edges.json"
    if with_edges.exists():
        return with_edges
    return folder / f"flexibility_comparison_{season}_full.json"


def _load_portfolios(e02_root: Path, season: str, results_tag: str) -> dict:
    path = _json_path(e02_root, season, results_tag)
    if not path.exists():
        raise FileNotFoundError(f"Missing E02 results JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    portfolios = data["portfolios"]
    backfill_congestion_metrics(portfolios)
    return portfolios


def _hour_tag(label: str | None, season: str | None = None) -> str:
    from experiments.lib.network_season import shift_snapshot_label_to_season

    text = shift_snapshot_label_to_season(label, season)
    return f" · {text}" if text else ""


def _aligned_studies(
    season: str, results_tag: str, threshold: float
) -> tuple[dict, dict, str | None]:
    """Same-hour actual/future portfolios. Prefer hourly scan; fall back to stored JSON edges."""
    actual_dir = ACTUAL_E02 / "results" / "flexibility" / results_tag
    future_dir = FUTURE_E02 / "results" / "flexibility" / results_tag
    frames_a, meta_a = load_hourly_frames(actual_dir, season)
    frames_f, meta_f = load_hourly_frames(future_dir, season)
    if frames_a and frames_f:
        hour, score, scores = select_hour_max_study_delta(
            frames_a, frames_f, keys=list(SCENARIO_KEYS), threshold=threshold
        )
        labels = list(meta_a.get("snapshots") or meta_f.get("snapshots") or [])
        label = labels[hour] if hour < len(labels) else f"h{hour}"
        n, *_ = load_e02_geo_context(FUTURE_E02)
        actual = portfolios_from_hourly(frames_a, hour=hour, label=label, static_n=n)
        future = portfolios_from_hourly(frames_f, hour=hour, label=label, static_n=n)
        scan_dir = future_dir / f"hourly_{season}"
        scan_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "hour": list(range(len(scores))),
                "snapshot": labels[: len(scores)],
                "n_delta_branches_future_vs_actual": scores,
            }
        ).to_csv(scan_dir / "compare_hour_scan.csv", index=False)
        meta = {
            "season": season,
            "map_snapshot_idx": hour,
            "map_snapshot_label": label,
            "map_snapshot_selection": "max_delta_branches_future_vs_actual",
            "map_impact_score": int(score),
            "delta_threshold_pp": threshold,
        }
        (scan_dir / "compare_hour_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(
            f"[compare] {season} shared hour idx={hour} {label} "
            f"Δbranches Future−Actual={score}",
            flush=True,
        )
        return actual, future, label
    return (
        _load_portfolios(ACTUAL_E02, season, results_tag),
        _load_portfolios(FUTURE_E02, season, results_tag),
        None,
    )


def _metric(portfolios: dict, key: str, field: str, default: float = 0.0) -> float:
    port = portfolios.get(key) or {}
    val = port.get(field)
    if val is None:
        return default
    return float(val)


def write_compare_csv(
    seasons: list[str],
    results_tag: str,
    out_csv: Path,
) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "season",
        "scenario",
        "map_snapshot_label",
        "actual_peak_loading_pct",
        "future_peak_loading_pct",
        "delta_peak_pp",
        "actual_n_overloaded",
        "future_n_overloaded",
        "actual_n_lines_ge_90pct",
        "future_n_lines_ge_90pct",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for season in seasons:
            actual, future, hour_label = _aligned_studies(season, results_tag, 5.0)
            for key in SCENARIO_KEYS:
                if key not in actual and key not in future:
                    continue
                a_peak = _metric(actual, key, "peak_loading_pct")
                f_peak = _metric(future, key, "peak_loading_pct")
                w.writerow(
                    {
                        "season": season,
                        "scenario": key,
                        "map_snapshot_label": hour_label or "",
                        "actual_peak_loading_pct": round(a_peak, 4),
                        "future_peak_loading_pct": round(f_peak, 4),
                        "delta_peak_pp": round(f_peak - a_peak, 4),
                        "actual_n_overloaded": int(_metric(actual, key, "n_overloaded")),
                        "future_n_overloaded": int(_metric(future, key, "n_overloaded")),
                        "actual_n_lines_ge_90pct": int(
                            _metric(actual, key, "n_lines_at_or_above_90pct")
                        ),
                        "future_n_lines_ge_90pct": int(
                            _metric(future, key, "n_lines_at_or_above_90pct")
                        ),
                    }
                )
    return out_csv


def plot_compare_metrics(seasons: list[str], results_tag: str, out_path: Path) -> Path:
    apply_figure_style()
    nrows = len(seasons)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(7.4, 2.55 * nrows + 0.55),
        facecolor="white",
        squeeze=False,
    )
    letters = "abcdefgh"

    for row, season in enumerate(seasons):
        actual, future, hour_label = _aligned_studies(season, results_tag, 5.0)
        keys = [k for k in SCENARIO_KEYS if k in actual or k in future]
        x = np.arange(len(keys))
        width = 0.36
        labels = [_label(k) for k in keys]
        a_peak = [_metric(actual, k, "peak_loading_pct") for k in keys]
        f_peak = [_metric(future, k, "peak_loading_pct") for k in keys]
        a_ov = [_metric(actual, k, "n_overloaded") for k in keys]
        f_ov = [_metric(future, k, "n_overloaded") for k in keys]

        ax_p = axes[row, 0]
        ax_c = axes[row, 1]
        ax_p.bar(x - width / 2, a_peak, width, color=STUDY_ACTUAL, label="Actual")
        ax_p.bar(x + width / 2, f_peak, width, color=STUDY_FUTURE, label="Future")
        ax_c.bar(x - width / 2, a_ov, width, color=STUDY_ACTUAL, label="Actual")
        ax_c.bar(x + width / 2, f_ov, width, color=STUDY_FUTURE, label="Future")

        cap = season.capitalize()
        ax_p.set_ylabel("Peak line loading [%]")
        ax_c.set_ylabel("Congested lines")
        ax_p.set_title(f"{cap} · peak loading{_hour_tag(hour_label, season)}", fontsize=9)
        ax_c.set_title(f"{cap} · lines at 100%{_hour_tag(hour_label, season)}", fontsize=9)
        for ax in (ax_p, ax_c):
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=40, ha="right")
            ax.set_xlim(-0.6, len(keys) - 0.4)
        ax_p.set_ylim(0, max(max(a_peak + f_peak, default=0.0) * 1.12, 60.0))
        ax_c.set_ylim(0, max(max(a_ov + f_ov, default=0.0) + 1.2, 2.0))
        ax_c.yaxis.set_major_locator(MaxNLocator(integer=True))
        panel_label(ax_p, letters[2 * row], x=-0.10, y=1.12)
        panel_label(ax_c, letters[2 * row + 1], x=-0.10, y=1.12)

    handles = [
        Line2D([0], [0], color=STUDY_ACTUAL, lw=6, label="Actual (public campuses)"),
        Line2D([0], [0], color=STUDY_FUTURE, lw=6, label="Future (actual + fictional)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("E02 grid health · Actual vs Future", fontsize=10, y=1.08)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return save_figure(fig, out_path)


def plot_compare_abs_maps(
    *,
    season: str,
    results_tag: str,
    out_path: Path,
) -> Path:
    import cartopy.crs as ccrs

    apply_nature_comm_style()
    apply_geo_style()
    actual, future, hour_label = _aligned_studies(season, results_tag, 5.0)
    n_a, dc_map_a, meta_a, refs_a, colors_a = load_e02_geo_context(ACTUAL_E02)
    n_f, dc_map_f, meta_f, refs_f, colors_f = load_e02_geo_context(FUTURE_E02)
    scale = plot_scale_for(FUTURE_E02)
    _, _, abs_thr, abs_label_fs = annotation_style_for(scale)
    keys = [k for k in MAP_KEYS if k in actual or k in future]
    aspect = _platecarree_aspect(scale)
    pw = 3.15
    nrows, ncols = len(keys), 2
    fig = plt.figure(
        figsize=(pw * ncols + 1.35, (pw / aspect) * nrows + 0.70),
        facecolor="white",
    )
    lc_ref = None
    letter = 0
    studies = (
        ("Actual", actual, n_a, dc_map_a, meta_a, refs_a, colors_a, len(refs_a)),
        ("Future", future, n_f, dc_map_f, meta_f, refs_f, colors_f, len(refs_f)),
    )

    for row, key in enumerate(keys):
        for col, (study, ports, n, dc_map, meta, refs, colors, n_sites) in enumerate(studies):
            ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1, projection=ccrs.PlateCarree())
            edges = ports.get(key, {}).get("edges") or []
            if not edges:
                ax.set_visible(False)
                letter += 1
                continue
            lc = draw_geo_panel_abs_loading(
                ax,
                edges,
                n,
                scale=scale,
                dc_map=dc_map,
                dc_meta=meta,
                dc_refs=refs,
                loading_threshold=abs_thr,
                label_fontsize=abs_label_fs,
                dc_size_ctx=MAP_DC_SIZE_CTX,
                dc_coords=MAP_DC_COORDS,
                site_colors=colors,
            )
            if lc is not None:
                lc_ref = lc
            ax.set_title("")
            ax.set_anchor("N")
            set_panel_caption(
                ax,
                chr(ord("a") + letter),
                f"{study} · {_label(key)}  ·  {_health_subtitle(ports.get(key, {}))}",
            )
            letter += 1
            if row == 0:
                ax.text(
                    0.5,
                    1.04,
                    f"{study} ({n_sites} sites)",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    clip_on=False,
                )

    fig.subplots_adjust(left=0.03, right=0.89, top=0.91, bottom=0.070, wspace=0.05, hspace=0.14)
    if lc_ref is not None:
        cax = fig.add_axes([0.915, 0.12, 0.012, 0.70])
        cbar = fig.colorbar(lc_ref, cax=cax)
        cbar.set_label(f"Line loading (≥{abs_thr:.0f}% highlighted) [%]", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    fig.legend(
        handles=_operator_legend_handles(),
        loc="upper center",
        ncol=6,
        fontsize=7,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.15,
        borderaxespad=0.0,
        bbox_to_anchor=(0.46, 0.058),
    )
    fig.suptitle(
        f"Absolute line loading · Actual vs Future · Texas · {season.capitalize()}"
        f"{_hour_tag(hour_label, season)}",
        fontsize=10,
        y=0.995,
    )
    return save_figure(fig, out_path)


def plot_compare_delta_maps(
    *,
    season: str,
    results_tag: str,
    out_path: Path,
    threshold: float = 5.0,
    label_threshold: float | None = None,
) -> Path:
    apply_nature_comm_style()
    apply_geo_style()
    actual, future, hour_label = _aligned_studies(season, results_tag, 5.0)
    n, dc_map, meta, refs, colors = load_e02_geo_context(FUTURE_E02)
    scale = plot_scale_for(FUTURE_E02)
    color_pp, label_pp, _, _ = annotation_style_for(scale)
    if threshold != 5.0:
        color_pp = threshold
    if label_threshold is not None:
        label_pp = label_threshold
    keys = [k for k in DELTA_KEYS if k in actual and k in future]
    ncols = min(3, max(1, len(keys)))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, panels, leg_ax = captioned_geo_figure(nrows=nrows, ncols=ncols, scale=scale, pw=3.45)

    for idx, key in enumerate(keys):
        ax, cap = panels[idx]
        fut_edges = future.get(key, {}).get("edges") or []
        act_edges = actual.get(key, {}).get("edges") or []
        if not fut_edges or not act_edges:
            ax.set_visible(False)
            continue
        n_inc, n_dec = draw_geo_panel_delta_loading(
            ax,
            fut_edges,
            act_edges,
            n,
            scale=scale,
            dc_map=dc_map,
            dc_meta=meta,
            dc_refs=refs,
            delta_threshold=color_pp,
            label_threshold=label_pp,
            dc_size_ctx=MAP_DC_SIZE_CTX,
            dc_coords=MAP_DC_COORDS,
            site_colors=colors,
        )
        extra = f"+{n_inc}/−{n_dec} br"
        ax.set_anchor("N")
        write_panel_caption(
            cap,
            chr(ord("a") + idx),
            f"{_label(key)} · Future − Actual  ·  {extra}",
        )

    place_delta_legend(fig, threshold=color_pp, ax=leg_ax)
    fig.suptitle(
        f"Δline loading · Future − Actual · Texas · {season.capitalize()}"
        f"{_hour_tag(hour_label, season)}",
        fontsize=10,
        y=0.99,
    )
    return save_figure(fig, out_path)


def plot_compare_delta_vs_nodc(
    *,
    season: str,
    results_tag: str,
    out_path: Path,
    threshold: float = 5.0,
    label_threshold: float | None = None,
) -> Path:
    """2×2: FCFS / Full flex vs No-DC, Future on top, Actual on bottom."""
    apply_nature_comm_style()
    apply_geo_style()
    actual, future, hour_label = _aligned_studies(season, results_tag, 5.0)
    n_a, dc_map_a, meta_a, refs_a, colors_a = load_e02_geo_context(ACTUAL_E02)
    n_f, dc_map_f, meta_f, refs_f, colors_f = load_e02_geo_context(FUTURE_E02)
    scale = plot_scale_for(FUTURE_E02)
    color_pp, label_pp, _, _ = annotation_style_for(scale)
    if threshold != 5.0:
        color_pp = threshold
    if label_threshold is not None:
        label_pp = label_threshold

    keys = [k for k in (FCFS_SCENARIO, FULL_FLEX_SCENARIO) if k in actual or k in future]
    rows = (
        ("Future", future, n_f, dc_map_f, meta_f, refs_f, colors_f),
        ("Actual", actual, n_a, dc_map_a, meta_a, refs_a, colors_a),
    )
    fig, panels, leg_ax = captioned_geo_figure(nrows=2, ncols=max(1, len(keys)), scale=scale, pw=3.45)
    letter = 0
    for row, (study, ports, n, dc_map, meta, refs, colors) in enumerate(rows):
        ref_edges = ports.get(E02_REF_SCENARIO, {}).get("edges") or []
        for col, key in enumerate(keys):
            ax, cap = panels[row * len(keys) + col]
            edges = ports.get(key, {}).get("edges") or []
            if not edges or not ref_edges:
                ax.set_visible(False)
                letter += 1
                continue
            n_inc, n_dec = draw_geo_panel_delta_loading(
                ax,
                edges,
                ref_edges,
                n,
                scale=scale,
                dc_map=dc_map,
                dc_meta=meta,
                dc_refs=refs,
                delta_threshold=color_pp,
                label_threshold=label_pp,
                dc_size_ctx=MAP_DC_SIZE_CTX,
                dc_coords=MAP_DC_COORDS,
                site_colors=colors,
            )
            extra = f"+{n_inc}/−{n_dec} br"
            ax.set_anchor("N")
            write_panel_caption(
                cap,
                chr(ord("a") + letter),
                f"{_label(key)} − No DC · {study}  ·  {extra}",
            )
            letter += 1

    place_delta_legend(fig, threshold=color_pp, ax=leg_ax)
    fig.suptitle(
        f"Δline loading vs No DC · Texas · {season.capitalize()}"
        f"{_hour_tag(hour_label, season)}",
        fontsize=10,
        y=0.99,
    )
    return save_figure(fig, out_path)


def _hourly_dirs_ready(e02_root: Path, results_tag: str, season: str) -> bool:
    d = e02_root / "results" / "flexibility" / results_tag
    return (d / f"hourly_{season}" / "meta.json").exists() or (
        d / f"flexibility_comparison_{season}_full_with_edges.json"
    ).exists()


def _select_hour_summer_vs_winter(
    results_tag: str, threshold: float
) -> tuple[int, int, str]:
    """Shared eval-week hour maximizing |Summer−Winter| branches on FCFS + Full flex."""
    keys = [FCFS_SCENARIO, FULL_FLEX_SCENARIO]
    combined: list[int] | None = None
    labels: list[str] = []
    for e02 in (FUTURE_E02, ACTUAL_E02):
        folder = e02 / "results" / "flexibility" / results_tag
        frames_w, meta_w = load_hourly_frames(folder, "winter")
        frames_s, meta_s = load_hourly_frames(folder, "summer")
        if not frames_w or not frames_s:
            continue
        _, _, scores = select_hour_max_study_delta(
            frames_w, frames_s, keys=keys, threshold=threshold
        )
        if combined is None:
            combined = list(scores)
            labels = list(meta_w.get("snapshots") or meta_s.get("snapshots") or [])
        else:
            n = min(len(combined), len(scores))
            combined = [combined[i] + scores[i] for i in range(n)]
    if not combined:
        return 0, 0, ""
    hour = int(max(range(len(combined)), key=lambda i: combined[i]))
    label = labels[hour] if hour < len(labels) else f"h{hour}"
    return hour, combined[hour], label


def plot_compare_delta_summer_vs_winter_study(
    e02_root: Path,
    *,
    results_tag: str,
    out_path: Path,
    study_label: str,
    threshold: float = 5.0,
    label_threshold: float | None = None,
) -> Path | None:
    """1×2 FCFS / Full flex: Summer − Winter for one study (Actual or Future)."""
    from experiments.lib.network_season import shift_snapshot_label_to_season

    folder = e02_root / "results" / "flexibility" / results_tag
    frames_w, meta_w = load_hourly_frames(folder, "winter")
    frames_s, meta_s = load_hourly_frames(folder, "summer")
    keys = [FCFS_SCENARIO, FULL_FLEX_SCENARIO]
    if not frames_w or not frames_s or any(k not in frames_w or k not in frames_s for k in keys):
        return None

    hour, score, _scores = select_hour_max_study_delta(
        frames_w, frames_s, keys=keys, threshold=5.0
    )
    labels = list(meta_w.get("snapshots") or meta_s.get("snapshots") or [])
    winter_lab = labels[hour] if hour < len(labels) else f"h{hour}"
    summer_lab = shift_snapshot_label_to_season(winter_lab, "summer")
    hour_label = (
        f"{winter_lab} / {summer_lab}" if winter_lab != summer_lab else winter_lab
    )

    n, dc_map, meta, refs, colors = load_e02_geo_context(e02_root)
    scale = plot_scale_for(e02_root)
    color_pp, label_pp, _, _ = annotation_style_for(scale)
    if threshold != 5.0:
        color_pp = threshold
    if label_threshold is not None:
        label_pp = label_threshold

    winter = portfolios_from_hourly(frames_w, hour=hour, label=hour_label, static_n=n)
    summer = portfolios_from_hourly(frames_s, hour=hour, label=hour_label, static_n=n)

    print(
        f"[compare] {study_label} Summer−Winter hour idx={hour} {hour_label} "
        f"Δbranches={score}",
        flush=True,
    )
    scan_dir = folder / "hourly_summer"
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / "compare_summer_vs_winter_meta.json").write_text(
        json.dumps(
            {
                "study": study_label,
                "map_snapshot_idx": hour,
                "map_snapshot_label": hour_label,
                "map_snapshot_selection": "max_delta_branches_summer_vs_winter",
                "map_impact_score": int(score),
                "delta_threshold_pp": 5.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    apply_nature_comm_style()
    apply_geo_style()
    fig, panels, leg_ax = captioned_geo_figure(nrows=1, ncols=2, scale=scale, pw=3.45)
    for col, key in enumerate(keys):
        ax, cap = panels[col]
        s_edges = summer.get(key, {}).get("edges") or []
        w_edges = winter.get(key, {}).get("edges") or []
        if not s_edges or not w_edges:
            ax.set_visible(False)
            continue
        n_inc, n_dec = draw_geo_panel_delta_loading(
            ax,
            s_edges,
            w_edges,
            n,
            scale=scale,
            dc_map=dc_map,
            dc_meta=meta,
            dc_refs=refs,
            delta_threshold=color_pp,
            label_threshold=label_pp,
            dc_size_ctx=MAP_DC_SIZE_CTX,
            dc_coords=MAP_DC_COORDS,
            site_colors=colors,
        )
        extra = f"+{n_inc}/−{n_dec} br"
        ax.set_anchor("N")
        write_panel_caption(
            cap,
            chr(ord("a") + col),
            f"{_label(key)} · {study_label}  ·  {extra}",
        )

    place_delta_legend(fig, threshold=color_pp, ax=leg_ax)
    fig.suptitle(
        f"Δline loading · Summer − Winter · {study_label} · Texas{_hour_tag(hour_label)}",
        fontsize=10,
        y=0.99,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, out_path)


def plot_compare_delta_summer_vs_winter(
    *,
    results_tag: str,
    out_path: Path,
    threshold: float = 5.0,
    label_threshold: float | None = None,
) -> Path:
    """2×2: FCFS / Full flex, Future on top, Actual on bottom; Summer − Winter."""
    apply_nature_comm_style()
    apply_geo_style()
    hour, score, hour_label = _select_hour_summer_vs_winter(results_tag, 5.0)
    n_a, dc_map_a, meta_a, refs_a, colors_a = load_e02_geo_context(ACTUAL_E02)
    n_f, dc_map_f, meta_f, refs_f, colors_f = load_e02_geo_context(FUTURE_E02)
    scale = plot_scale_for(FUTURE_E02)
    color_pp, label_pp, _, _ = annotation_style_for(scale)
    if threshold != 5.0:
        color_pp = threshold
    if label_threshold is not None:
        label_pp = label_threshold

    keys = [FCFS_SCENARIO, FULL_FLEX_SCENARIO]
    rows = []
    for study, e02, n, dc_map, meta, refs, colors in (
        ("Future", FUTURE_E02, n_f, dc_map_f, meta_f, refs_f, colors_f),
        ("Actual", ACTUAL_E02, n_a, dc_map_a, meta_a, refs_a, colors_a),
    ):
        folder = e02 / "results" / "flexibility" / results_tag
        frames_w, _ = load_hourly_frames(folder, "winter")
        frames_s, _ = load_hourly_frames(folder, "summer")
        winter = portfolios_from_hourly(frames_w, hour=hour, label=hour_label, static_n=n)
        summer = portfolios_from_hourly(frames_s, hour=hour, label=hour_label, static_n=n)
        rows.append((study, summer, winter, n, dc_map, meta, refs, colors))

    print(
        f"[compare] Summer−Winter shared hour idx={hour} {hour_label} "
        f"Δbranches={score}",
        flush=True,
    )
    scan_dir = FUTURE_E02 / "results" / "flexibility" / results_tag / "hourly_summer"
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / "compare_summer_vs_winter_meta.json").write_text(
        json.dumps(
            {
                "map_snapshot_idx": hour,
                "map_snapshot_label": hour_label,
                "map_snapshot_selection": "max_delta_branches_summer_vs_winter",
                "map_impact_score": int(score),
                "delta_threshold_pp": 5.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, panels, leg_ax = captioned_geo_figure(nrows=2, ncols=max(1, len(keys)), scale=scale, pw=3.45)
    letter = 0
    for row, (study, summer, winter, n, dc_map, meta, refs, colors) in enumerate(rows):
        for col, key in enumerate(keys):
            ax, cap = panels[row * len(keys) + col]
            s_edges = summer.get(key, {}).get("edges") or []
            w_edges = winter.get(key, {}).get("edges") or []
            if not s_edges or not w_edges:
                ax.set_visible(False)
                letter += 1
                continue
            n_inc, n_dec = draw_geo_panel_delta_loading(
                ax,
                s_edges,
                w_edges,
                n,
                scale=scale,
                dc_map=dc_map,
                dc_meta=meta,
                dc_refs=refs,
                delta_threshold=color_pp,
                label_threshold=label_pp,
                dc_size_ctx=MAP_DC_SIZE_CTX,
                dc_coords=MAP_DC_COORDS,
                site_colors=colors,
            )
            extra = f"+{n_inc}/−{n_dec} br"
            ax.set_anchor("N")
            write_panel_caption(
                cap,
                chr(ord("a") + letter),
                f"{_label(key)} · {study}  ·  {extra}",
            )
            letter += 1

    place_delta_legend(fig, threshold=color_pp, ax=leg_ax)
    fig.suptitle(
        f"Δline loading · Summer − Winter · Texas{_hour_tag(hour_label)}",
        fontsize=10,
        y=0.99,
    )
    return save_figure(fig, out_path)


CONGESTION_PLOT_KEYS: tuple[str, ...] = (
    E02_REF_SCENARIO,
    FCFS_SCENARIO,
    FULL_FLEX_SCENARIO,
)
CONGESTION_LINE_COLORS: dict[str, str] = {
    E02_REF_SCENARIO: "#7E6148",
    FCFS_SCENARIO: PORTFOLIO_COLORS["baseline"],
    FULL_FLEX_SCENARIO: PORTFOLIO_COLORS["reallocation_slack_termination"],
}


def _congestion_count(df: pd.DataFrame, *, threshold: float = 100.0) -> np.ndarray:
    return (df.to_numpy(dtype=float) >= threshold - 1e-9).sum(axis=1).astype(int)


def write_hourly_congestion_csv(results_tag: str, out_csv: Path) -> Path:
    keys = list(CONGESTION_PLOT_KEYS)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["study", "season", "scenario", "hour", "snapshot", "n_congested"],
        )
        w.writeheader()
        for study, e02 in (("actual", ACTUAL_E02), ("future", FUTURE_E02)):
            for season in ("winter", "summer"):
                folder = e02 / "results" / "flexibility" / results_tag
                frames, meta = load_hourly_frames(folder, season)
                if not frames:
                    continue
                labels = list(meta.get("snapshots") or [])
                for key in keys:
                    if key not in frames:
                        continue
                    counts = _congestion_count(frames[key])
                    for hour, n_cong in enumerate(counts):
                        w.writerow(
                            {
                                "study": study,
                                "season": season,
                                "scenario": key,
                                "hour": hour,
                                "snapshot": labels[hour] if hour < len(labels) else "",
                                "n_congested": int(n_cong),
                            }
                        )
    return out_csv


def plot_hourly_congestion_counts(
    results_tag: str,
    out_path: Path,
    *,
    e02_root: Path,
    study_label: str,
) -> Path:
    """One study: Winter | Summer congested-line count vs hour."""
    apply_figure_style()
    keys = list(CONGESTION_PLOT_KEYS)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.45),
        facecolor="white",
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    y_max = 1.0
    n_hours = 168
    folder_base = e02_root / "results" / "flexibility" / results_tag
    for col, season in enumerate(("winter", "summer")):
        ax = axes[0, col]
        frames, _meta = load_hourly_frames(folder_base, season)
        for key in keys:
            if key not in frames:
                continue
            y = _congestion_count(frames[key])
            x = np.arange(len(y))
            n_hours = max(n_hours, len(y))
            ax.plot(x, y, color=CONGESTION_LINE_COLORS[key], lw=1.2, zorder=3)
            if len(y):
                y_max = max(y_max, float(np.max(y)))
        for d in range(1, 7):
            ax.axvline(d * 24, color="#E6E6E6", lw=0.5, zorder=0)
        ax.set_xlim(0, max(n_hours - 1, 1))
        ax.set_title(season.capitalize(), fontsize=9)
        ax.grid(True, axis="y", alpha=0.25, lw=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        panel_label(ax, "ab"[col], x=-0.08, y=1.08)
        ax.set_xlabel("Hour in eval week")
        ax.set_xticks(range(0, 169, 24))
        if col == 0:
            ax.set_ylabel("Congested lines")

    for ax in axes.ravel():
        ax.set_ylim(0, y_max * 1.08)

    handles = [
        Line2D([0], [0], color=CONGESTION_LINE_COLORS[k], lw=1.6, label=_label(k))
        for k in keys
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.15,
    )
    fig.suptitle(
        f"Hourly congested lines · {study_label} · Texas · s1250",
        fontsize=10,
        y=1.10,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.18, wspace=0.12)
    return save_figure(fig, out_path)


def plot_actual_vs_future(
    *,
    seasons: list[str],
    results_tag: str = "full",
    threshold: float = 5.0,
    fig_dir: Path | None = None,
    actual_e02: Path | None = None,
    future_e02: Path | None = None,
) -> list[Path]:
    global ACTUAL_E02, FUTURE_E02
    if actual_e02 is not None:
        ACTUAL_E02 = Path(actual_e02)
    if future_e02 is not None:
        FUTURE_E02 = Path(future_e02)
    fig_dir = fig_dir or (FUTURE_E02 / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        FUTURE_E02
        / "results"
        / "flexibility"
        / results_tag
        / "compare_actual_vs_future.csv"
    )
    written = [write_compare_csv(seasons, results_tag, csv_path)]
    written.append(
        plot_compare_metrics(
            seasons,
            results_tag,
            fig_dir / "fig_compare_actual_vs_future_metrics.png",
        )
    )
    for season in seasons:
        written.append(
            plot_compare_abs_maps(
                season=season,
                results_tag=results_tag,
                out_path=fig_dir / f"fig_compare_actual_vs_future_abs_{season}.png",
            )
        )
        written.append(
            plot_compare_delta_maps(
                season=season,
                results_tag=results_tag,
                out_path=fig_dir / f"fig_compare_actual_vs_future_delta_{season}.png",
                threshold=threshold,
            )
        )
        written.append(
            plot_compare_delta_vs_nodc(
                season=season,
                results_tag=results_tag,
                out_path=fig_dir / f"fig_compare_vs_nodc_delta_{season}.png",
                threshold=threshold,
            )
        )
    if (
        _hourly_dirs_ready(ACTUAL_E02, results_tag, "winter")
        and _hourly_dirs_ready(ACTUAL_E02, results_tag, "summer")
        and _hourly_dirs_ready(FUTURE_E02, results_tag, "winter")
        and _hourly_dirs_ready(FUTURE_E02, results_tag, "summer")
    ):
        written.append(
            plot_compare_delta_summer_vs_winter(
                results_tag=results_tag,
                out_path=fig_dir / "fig_compare_summer_vs_winter_delta.png",
                threshold=threshold,
            )
        )
        written.append(
            write_hourly_congestion_csv(
                results_tag,
                FUTURE_E02
                / "results"
                / "flexibility"
                / results_tag
                / "hourly_congestion_counts.csv",
            )
        )
        written.append(
            plot_hourly_congestion_counts(
                results_tag,
                fig_dir / "fig_compare_hourly_congestion_future.png",
                e02_root=FUTURE_E02,
                study_label="Future",
            )
        )
        actual_fig = ACTUAL_E02 / "figures" / "fig_hourly_congestion.png"
        written.append(
            plot_hourly_congestion_counts(
                results_tag,
                fig_dir / "fig_compare_hourly_congestion_actual.png",
                e02_root=ACTUAL_E02,
                study_label="Actual",
            )
        )
        if actual_fig.resolve() != (fig_dir / "fig_compare_hourly_congestion_actual.png").resolve():
            written.append(
                plot_hourly_congestion_counts(
                    results_tag,
                    actual_fig,
                    e02_root=ACTUAL_E02,
                    study_label="Actual",
                )
            )
    return written
