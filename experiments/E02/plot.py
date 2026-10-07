from __future__ import annotations
"""Fig. 2 and Supplementary Figs. 9 and 10."""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib import cm as mpl_cm

from experiments.lib.paths import package_root

ROOT = package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.figure_style import *  # noqa: F403
from experiments.lib.figure_common import *  # noqa: F403

OUT = ROOT / "experiments/E02/figures"

def build_fig1() -> Path:
    return _build_spread_state_board("summer", OUT / "Fig2_interstate_LMP_phase_lag.png")


def _states_from_deposited_metrics(df: pd.DataFrame):
    """State locations for contouring when the raw price cache is not present.

    The plotted D and G values still come from the deposited metrics table.
    """
    from experiments.lib.state_lmp_loader import STATE_CENTROIDS

    class _Point:
        def __init__(self, state: str, lon: float, lat: float) -> None:
            self.state = state
            self.lon = lon
            self.lat = lat

    points = []
    for state in sorted(set(df["state"].astype(str))):
        lon, lat = STATE_CENTROIDS.get(state, (-98.0, 39.0))
        points.append(_Point(state, float(lon), float(lat)))
    return points


def _build_spread_state_board(season: str, out: Path) -> Path:
    """Fig1: D and G of state hubs vs California, Texas, New York (3 columns)."""
    from experiments.lib.us_map import draw_conus_borders, draw_state_borders

    fs_lab, fs_tick, fs_tag, fs_clabel = MAP_FS_LAB, MAP_FS_TICK, MAP_FS_TAG, MAP_FS_CLABEL
    apply_figure_style()
    plt.rcParams.update(
        {"font.size": fs_lab, "axes.labelsize": fs_lab, "xtick.labelsize": fs_tick, "ytick.labelsize": fs_tick}
    )
    geo = load_geojson()
    if geo is None:
        raise RuntimeError("US geojson unavailable for Fig1 land mask")

    spread_path = E00_RESULTS / "price_spread_vs_refs" / f"state_spread_metrics_{season}.csv"
    if not spread_path.exists():
        raise FileNotFoundError(
            f"Missing {spread_path}."
        )
    df = pd.read_csv(spread_path)
    states = _states_from_deposited_metrics(df)
    conus = _conus_states(states)
    refs = ("CA", "TX", "NY")

    d_cmap = _direction_cmap()
    d_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    d_levels = np.linspace(-1.0, 1.0, 21)

    m_vals: list[float] = []
    fields_d: dict[str, dict[str, float]] = {}
    fields_m: dict[str, dict[str, float]] = {}
    for ref in refs:
        sub = df[df["reference_state"] == ref]
        fields_d[ref] = {
            str(r.state): float(r.directional_bias_D)
            for r in sub.itertuples()
            if str(r.state) not in CONUS_EXCLUDE
        }
        # Gross magnitude G = A+ + A- (not mean |Δp|)
        fields_m[ref] = {
            str(r.state): float(r.A_plus_plus_A_minus)
            for r in sub.itertuples()
            if str(r.state) not in CONUS_EXCLUDE
        }
        m_vals.extend(v for st, v in fields_m[ref].items() if st != ref)
    vmax_m = _vmax_gross_magnitude(m_vals, floor=5_000.0, cap=80_000.0, step=5_000.0)
    m_norm = Normalize(vmin=0.0, vmax=vmax_m)
    m_cmap = _magnitude_cmap()
    m_levels = np.linspace(0.0, vmax_m, MAE_N_LEVELS)

    dx = float(CONUS_XLIM[1] - CONUS_XLIM[0])
    dy = float(CONUS_YLIM[1] - CONUS_YLIM[0])
    fig_w = float(figsize_double_col(1.0)[0])
    panel_w = fig_w * (MAP_RIGHT - MAP_LEFT) / 3.0
    panel_h = panel_w * (dy / dx)
    fig, outer = _contour_board_figure(panel_h, fig_w, bottom=MAP_FIG1_BOTTOM, fig_extra=MAP_FIG_EXTRA)

    row_titles = (
        "a  Directional price bias to reference states",
        "b  Gross magnitude relative to reference states",
    )
    left_axes: list = []
    map_rows: list[list] = []
    title_artists: list = []

    for block, fields, is_dir in ((0, fields_d, True), (1, fields_m, False)):
        sub, title_artist = _add_row_title_strip(
            fig, outer[block], row_titles[block], fontsize=MAP_FS_ROW_TITLE, ncols=3
        )
        title_artists.append(title_artist)
        row_axes: list = []
        for idx, ref in enumerate(refs):
            ax = fig.add_subplot(sub[1, idx])
            row_axes.append(ax)
            if idx == 0:
                left_axes.append(ax)
            draw_base_map(ax, geo, {}, alpha=0.95, default_color="#ececec")
            field = fields[ref]
            grid_x, grid_y, z_show, _ = _interpolate_field(conus, field, grid_n=300, geo=geo)
            z_m = np.ma.masked_invalid(z_show)
            draw_conus_borders(ax, geo, zorder=3)
            draw_state_borders(
                ax,
                geo,
                ref,
                edgecolor=FIG1_REF_BORDER_COLOR,
                linewidth=FIG1_REF_BORDER_LW,
                zorder=4,
            )
            if is_dir:
                ax.contourf(
                    grid_x, grid_y, z_m,
                    levels=d_levels, cmap=d_cmap, norm=d_norm,
                    alpha=FILL_ALPHA, extend="neither", zorder=2,
                )
                cs = ax.contour(
                    grid_x, grid_y, z_m,
                    levels=np.linspace(-1, 1, 9), colors=PHASE_CONTOUR_COLOR,
                    linewidths=PHASE_CONTOUR_LW, alpha=0.9, zorder=5,
                )
                ax.clabel(cs, inline=True, fontsize=fs_clabel, fmt="%.1f")
            else:
                ax.contourf(
                    grid_x, grid_y, z_m,
                    levels=m_levels, cmap=m_cmap, norm=m_norm,
                    alpha=FILL_ALPHA, extend="max", zorder=2,
                )
                cs = ax.contour(
                    grid_x, grid_y, z_m,
                    levels=m_levels[::2], colors=MAE_CONTOUR_COLOR,
                    linewidths=MAE_CONTOUR_LW, alpha=0.85, zorder=5,
                )
                ax.clabel(cs, inline=True, fontsize=fs_clabel, fmt=_fmt_gross_tick)
            ax.scatter(
                [s.lon for s in conus],
                [s.lat for s in conus],
                s=FIG1_STATE_DOT_S,
                c="0.2",
                zorder=6,
                alpha=0.85,
                linewidths=0,
            )
            ax.set_xlim(CONUS_XLIM)
            ax.set_ylim(CONUS_YLIM)
            ax.set_aspect("auto")
            ax.tick_params(labelsize=fs_tick, length=1.0, pad=0.4)
            if idx == 0:
                ax.set_ylabel("Latitude", fontsize=fs_lab)
            else:
                ax.set_yticklabels([])
            if block == 1:
                ax.set_xlabel("Longitude", fontsize=fs_lab)
            else:
                ax.set_xticklabels([])
            ax.text(
                0.01,
                0.02,
                state_display_name(ref),
                transform=ax.transAxes,
                fontsize=fs_tag,
                fontweight="bold",
                va="bottom",
                ha="left",
                color="0.1",
                zorder=10,
            )
        map_rows.append(row_axes)

    _center_row_titles(fig, map_rows, title_artists, fontsize=MAP_FS_ROW_TITLE, ha="left")

    fig.canvas.draw()
    right_x = max(ax.get_position().x1 for row in map_rows for ax in row)
    cbar_x = right_x + MAP_CBAR_PAD
    p0, p1 = left_axes[0].get_position(), left_axes[1].get_position()
    cax1 = fig.add_axes([cbar_x, p0.y0, MAP_CBAR_W, p0.height])
    grad = np.linspace(1.0, -1.0, 256).reshape(256, 1)
    cax1.imshow(grad, aspect="auto", cmap=d_cmap, norm=d_norm, extent=[0, 1, -1, 1], origin="upper")
    cax1.set_xlim(0, 1)
    cax1.set_ylim(-1, 1)
    cax1.set_xticks([])
    cax1.yaxis.set_label_position("right")
    cax1.yaxis.tick_right()
    cax1.set_yticks([-1, 0, 1])
    cax1.set_yticklabels(["-1", "0", "+1"])
    cax1.tick_params(labelsize=fs_tick, length=1.0, pad=1.0)
    cax1.set_ylabel("Directional bias", fontsize=fs_lab, labelpad=2)
    for spine in cax1.spines.values():
        spine.set_linewidth(0.4)

    cax2 = fig.add_axes([cbar_x, p1.y0, MAP_CBAR_W, p1.height])
    grad2 = np.linspace(vmax_m, 0.0, 256).reshape(256, 1)
    cax2.imshow(grad2, aspect="auto", cmap=m_cmap, norm=m_norm, extent=[0, 1, 0.0, vmax_m], origin="upper")
    cax2.set_xlim(0, 1)
    cax2.set_ylim(0.0, vmax_m)
    cax2.set_xticks([])
    cax2.yaxis.set_label_position("right")
    cax2.yaxis.tick_right()
    cax2.set_yticks(np.linspace(0.0, vmax_m, 5))
    cax2.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gross_tick))
    cax2.tick_params(labelsize=fs_tick, length=1.0, pad=1.0)
    cax2.set_ylabel("Gross magnitude ($/MW)", fontsize=fs_lab, labelpad=2)
    for spine in cax2.spines.values():
        spine.set_linewidth(0.4)

    return save_figure(fig, out, pad_inches=0.12)


def _build_phase_lag_board(season: str, out: Path) -> Path:
    """ExtData only: legacy peak-hour offset / mean |ΔLMP| vs CA, TX, NY."""
    fs_lab, fs_tick, fs_tag, fs_clabel = MAP_FS_LAB, MAP_FS_TICK, MAP_FS_TAG, MAP_FS_CLABEL
    apply_figure_style()
    plt.rcParams.update(
        {"font.size": fs_lab, "axes.labelsize": fs_lab, "xtick.labelsize": fs_tick, "ytick.labelsize": fs_tick}
    )
    geo = load_geojson()
    if geo is None:
        raise RuntimeError("US geojson unavailable for ExtData phase-lag board")
    states = _load_all_state_prices(season)
    phase = _load_matrix(E00_RESULTS, season)
    mae_path = E00_RESULTS / f"mae_lmp_matrix_{season}.csv"
    if mae_path.exists():
        mae = pd.read_csv(mae_path, index_col=0)
    else:
        mae = mean_abs_lmp_spread_matrix(states)
        mae.round(4).to_csv(mae_path)

    conus = _conus_states(states)
    refs = ("CA", "TX", "NY")
    vshow = PHASE_VSHOW
    phase_norm = TwoSlopeNorm(vmin=-vshow, vcenter=0.0, vmax=vshow)
    phase_levels = np.arange(-vshow, vshow + 0.1, PHASE_LEVEL_STEP)

    mae_vals = []
    for ref in refs:
        field = reference_field(mae, ref)
        mae_vals.extend(float(field[st]) for st in field if st not in CONUS_EXCLUDE and st != ref)
    vmax_mae = float(np.ceil(np.percentile(mae_vals, 95) / 5.0) * 5.0) if mae_vals else 80.0
    vmax_mae = max(40.0, min(vmax_mae, 120.0))
    mae_norm = Normalize(vmin=0.0, vmax=vmax_mae)
    mae_cmap = _magnitude_cmap()
    phase_cmap = CMAP.copy()
    phase_cmap.set_bad("#f7f7f7")
    mae_levels = np.linspace(0.0, vmax_mae, MAE_N_LEVELS)

    dx = float(CONUS_XLIM[1] - CONUS_XLIM[0])
    dy = float(CONUS_YLIM[1] - CONUS_YLIM[0])
    fig_w = float(figsize_double_col(1.0)[0])
    panel_w = fig_w * 0.78 / 3.0
    panel_h = panel_w * (dy / dx)
    fig, outer = _contour_board_figure(panel_h, fig_w)

    row_titles = ("a  Peak-hour offset (ExtData)", "b  Mean |ΔLMP| (ExtData)")
    left_axes: list = []
    map_rows: list[list] = []
    title_artists: list = []
    for block, matrix, is_phase in ((0, phase, True), (1, mae, False)):
        sub, title_artist = _add_row_title_strip(
            fig, outer[block], row_titles[block], fontsize=MAP_FS_ROW_TITLE, ncols=3
        )
        title_artists.append(title_artist)
        row_axes: list = []
        for idx, ref in enumerate(refs):
            ax = fig.add_subplot(sub[1, idx])
            row_axes.append(ax)
            if idx == 0:
                left_axes.append(ax)
            draw_base_map(ax, geo, {}, alpha=0.95, default_color="#ececec")
            field = reference_field(matrix, ref)
            field_conus = {st: field[st] for st in field if st not in CONUS_EXCLUDE}
            grid_x, grid_y, z_show, _ = _interpolate_field(
                conus, field_conus, grid_n=300, geo=geo
            )
            rlon, rlat = STATE_CENTROIDS[ref]
            z_m = np.ma.masked_invalid(z_show)
            draw_conus_borders(ax, geo, zorder=3)
            if is_phase:
                ax.contourf(
                    grid_x, grid_y, z_m,
                    levels=phase_levels, cmap=phase_cmap, norm=phase_norm,
                    alpha=FILL_ALPHA, extend="neither", zorder=2,
                )
                cs = ax.contour(
                    grid_x, grid_y, z_m,
                    levels=phase_levels, colors=PHASE_CONTOUR_COLOR,
                    linewidths=PHASE_CONTOUR_LW, alpha=0.9, zorder=5,
                )
                ax.clabel(cs, inline=True, fontsize=fs_clabel, fmt="%.0f")
            else:
                ax.contourf(
                    grid_x, grid_y, z_m,
                    levels=mae_levels, cmap=mae_cmap, norm=mae_norm,
                    alpha=FILL_ALPHA, extend="max", zorder=2,
                )
                cs = ax.contour(
                    grid_x, grid_y, z_m,
                    levels=mae_levels[::2], colors=MAE_CONTOUR_COLOR,
                    linewidths=MAE_CONTOUR_LW, alpha=0.85, zorder=5,
                )
                ax.clabel(cs, inline=True, fontsize=fs_clabel, fmt="%.0f")
            ax.scatter(
                [s.lon for s in conus], [s.lat for s in conus],
                s=FIG1_STATE_DOT_S, c="0.2", zorder=5, alpha=0.85, linewidths=0,
            )
            _draw_reference_marker(ax, rlon, rlat)
            ax.set_xlim(CONUS_XLIM)
            ax.set_ylim(CONUS_YLIM)
            ax.set_aspect("auto")
            ax.tick_params(labelsize=fs_tick, length=1.0, pad=0.4)
            if idx == 0:
                ax.set_ylabel("Latitude", fontsize=fs_lab)
            else:
                ax.set_yticklabels([])
            if block == 1:
                ax.set_xlabel("Longitude", fontsize=fs_lab)
            else:
                ax.set_xticklabels([])
            ax.text(
                0.01, 0.02, state_display_name(ref),
                transform=ax.transAxes, fontsize=fs_tag, fontweight="bold",
                va="bottom", ha="left", color="0.1", zorder=10,
            )
        map_rows.append(row_axes)

    _center_row_titles(fig, map_rows, title_artists, fontsize=MAP_FS_ROW_TITLE)
    fig.canvas.draw()
    right_x = max(ax.get_position().x1 for row in map_rows for ax in row)
    cbar_x = right_x + MAP_CBAR_PAD
    p0, p1 = left_axes[0].get_position(), left_axes[1].get_position()
    cax1 = fig.add_axes([cbar_x, p0.y0, MAP_CBAR_W, p0.height])
    grad = np.linspace(vshow, -vshow, 256).reshape(256, 1)
    cax1.imshow(grad, aspect="auto", cmap=phase_cmap, norm=phase_norm, extent=[0, 1, -vshow, vshow], origin="upper")
    cax1.set_xlim(0, 1)
    cax1.set_ylim(-vshow, vshow)
    cax1.set_xticks([])
    cax1.yaxis.set_label_position("right")
    cax1.yaxis.tick_right()
    cax1.set_yticks([-8, -4, 0, 4, 8])
    cax1.tick_params(labelsize=fs_tick, length=1.0, pad=1.0)
    cax1.set_ylabel("Phase lag (h)", fontsize=fs_lab, labelpad=1)
    for spine in cax1.spines.values():
        spine.set_linewidth(0.4)
    cax2 = fig.add_axes([cbar_x, p1.y0, MAP_CBAR_W, p1.height])
    grad2 = np.linspace(vmax_mae, 0.0, 256).reshape(256, 1)
    cax2.imshow(grad2, aspect="auto", cmap=mae_cmap, norm=mae_norm, extent=[0, 1, 0.0, vmax_mae], origin="upper")
    cax2.set_xlim(0, 1)
    cax2.set_ylim(0.0, vmax_mae)
    cax2.set_xticks([])
    cax2.yaxis.set_label_position("right")
    cax2.yaxis.tick_right()
    cax2.set_yticks(np.linspace(0.0, vmax_mae, 5))
    cax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(round(v))}"))
    cax2.tick_params(labelsize=fs_tick, length=1.0, pad=1.0)
    cax2.set_ylabel("Mean |ΔLMP| (USD/MWh)", fontsize=fs_lab, labelpad=1)
    for spine in cax2.spines.values():
        spine.set_linewidth(0.4)
    return save_figure(fig, out)


def build_extdata_directional_motivation() -> Path:
    """ExtData schematic: why directional bias D is needed (not only |Δ|)."""
    from matplotlib.patches import Patch

    from experiments.E02.model import compute_spread_metrics
    from experiments.lib.state_lmp_loader import (
        load_state_prices,
    )

    apply_figure_style()
    season = "summer"
    ref = np.asarray(load_state_prices("TX", season), dtype=float)
    cases = [
        ("a", "IN", r"$D\approx 0$"),
        ("b", "MD", r"$D\to +1$"),
        ("c", "AL", r"$D\to -1$"),
    ]

    best_start = 24 * 10
    best_score = -1.0
    for s0 in range(0, max(1, len(ref) - 48), 12):
        win = slice(s0, s0 + 48)
        p = np.asarray(load_state_prices("IN", season), dtype=float)[win]
        r = ref[win]
        crosses = int(np.sum((p[1:] - r[1:]) * (p[:-1] - r[:-1]) < 0))
        m = compute_spread_metrics(p, r)
        score = crosses - 20.0 * abs(m.directional_bias_D)
        if score > best_score:
            best_score = score
            best_start = s0
    show = slice(best_start, best_start + 48)
    ts = np.arange(48, dtype=float)

    series = []
    for letter, state, d_label in cases:
        partner = np.asarray(load_state_prices(state, season), dtype=float)
        m_full = compute_spread_metrics(partner, ref)
        series.append(
            {
                "letter": letter,
                "state": state,
                "d_label": d_label,
                "partner": partner,
                "diff": partner - ref,
                "m_full": m_full,
            }
        )

    color_ref = "#3C5488"
    color_p = "#E64B35"
    color_pos = "#00A087"
    color_neg = "#E64B35"
    color_diff = "#3C5488"

    fig = plt.figure(figsize=figsize_double_col(3.2), facecolor="white")
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.2, 1.0],
        hspace=0.38,
        wspace=0.22,
        left=0.09,
        right=0.985,
        top=0.90,
        bottom=0.14,
    )

    for j, s in enumerate(series):
        ax = fig.add_subplot(gs[0, j])
        ax_d = fig.add_subplot(gs[1, j])
        mf = s["m_full"]

        ax.plot(ts, ref[show], color=color_ref, lw=1.0)
        ax.plot(ts, s["partner"][show], color=color_p, lw=1.0)
        ax.set_xlim(0, 47)
        y0 = min(float(np.nanmin(ref[show])), float(np.nanmin(s["partner"][show]))) - 2
        y1 = max(float(np.nanmax(ref[show])), float(np.nanmax(s["partner"][show]))) + 2
        y1 = min(y1, float(np.nanpercentile(s["partner"][show], 98)) * 1.15 + 5)
        ax.set_ylim(y0, y1)
        ax.set_xticks([0, 24, 47])
        if j == 0:
            ax.set_ylabel("LMP ($/MWh)", fontsize=FS)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=FS_TICK_COMPOSED, length=2.5, width=0.5)
        ax.set_xticklabels([])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        _panel_tag(
            ax,
            s["letter"],
            f"{s['state']} vs TX  {s['d_label']}",
            x=0.0,
            y=1.02,
            fontsize=FS_TITLE,
        )

        d = s["diff"][show]
        y_d = max(float(np.nanmax(np.abs(d))), 1.0) * 1.08
        ax_d.axhline(0.0, color="0.55", lw=0.5, zorder=1)
        ax_d.fill_between(
            ts, 0.0, d, where=(d >= 0), interpolate=True,
            facecolor=color_pos, alpha=0.30, linewidth=0, zorder=2,
        )
        ax_d.fill_between(
            ts, 0.0, d, where=(d < 0), interpolate=True,
            facecolor=color_neg, alpha=0.30, linewidth=0, zorder=2,
        )
        ax_d.plot(ts, d, color=color_diff, lw=0.9, zorder=3)
        ax_d.set_xlim(0, 47)
        ax_d.set_ylim(-y_d, y_d)
        ax_d.set_xticks([0, 24, 47])
        ax_d.set_xlabel("Hour", fontsize=FS)
        if j == 0:
            # Avoid broken mathtext: use unicode delta, not nested $...$ with \mathrm.
            ax_d.set_ylabel("Δp vs TX ($/MWh)", fontsize=FS)
        else:
            ax_d.tick_params(labelleft=False)
        ax_d.tick_params(labelsize=FS_TICK_COMPOSED, length=2.5, width=0.5)
        for spine in ("top", "right"):
            ax_d.spines[spine].set_visible(False)

    handles = [
        Line2D([0], [0], color=color_ref, lw=1.2, label="Texas"),
        Line2D([0], [0], color=color_p, lw=1.2, label="Partner"),
        Patch(facecolor=color_pos, alpha=0.35, edgecolor="none", label=r"$A_+$"),
        Patch(facecolor=color_neg, alpha=0.35, edgecolor="none", label=r"$A_-$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=FS_LEGEND,
        frameon=False,
        bbox_to_anchor=(0.54, 0.995),
        columnspacing=1.2,
        handletextpad=0.35,
        handlelength=1.4,
    )

    return save_figure(fig, OUT / "ExtData_Fig2_directional_motivation.png")


_SITE_SHORT = {
    701: "Helios",
    702: "Denton",
    711: "Abilene",
    712: "Shackelford",
    713: "Milam",
    731: "DFW",
    733: "Goodnight",
    721: "Temple",
    722: "Fort Worth",
    741: "San Antonio",
    751: "Pecos",
}




def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(build_fig1())
    print(_build_spread_state_board("winter", OUT / "ExtData_Fig2_price_direction_magnitude_winter.png"))
    print(build_extdata_directional_motivation())


if __name__ == "__main__":
    main()
