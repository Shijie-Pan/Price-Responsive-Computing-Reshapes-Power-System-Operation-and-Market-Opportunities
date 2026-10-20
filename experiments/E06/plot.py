from __future__ import annotations
"""Fig. 6 and Supplementary Fig. 9 (winter open-loop maps)."""

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

OUT = ROOT / "experiments/E06/figures"

def build_fig5() -> Path:
    """USA endogenous LOPF dual maps.

    No-DC and price-independent are the initial prescribed injections.
    Price-responsive is the summer dispatch after six updates.
    """
    spread = (
        USA_E02_RESULTS
        / "closed_loop"
        / "iter_06"
        / "price_spread_vs_texas"
        / "bus_spread_metrics_summer.csv"
    )
    return _build_usa_endogenous_contour_board(
        "summer",
        OUT / "Fig6_endogenous_lmp_callback.png",
        tag="priority_168h",
        spread_csv=spread,
    )


def _build_usa_endogenous_contour_board(
    season: str,
    out: Path,
    *,
    tag: str = "priority_168h",
    spread_csv: Path | None = None,
) -> Path:
    from experiments.lib.us_map import draw_conus_borders, draw_state_borders

    fs_lab, fs_tick, fs_tag, fs_clabel = MAP_FS_LAB, MAP_FS_TICK, MAP_FS_TAG, MAP_FS_CLABEL
    apply_figure_style()
    plt.rcParams.update(
        {"font.size": fs_lab, "axes.labelsize": fs_lab, "xtick.labelsize": fs_tick, "ytick.labelsize": fs_tick}
    )
    geo = load_geojson()
    if geo is None:
        raise RuntimeError("US geojson unavailable for Fig5 land mask")

    if spread_csv is None:
        spread_csv = USA_E02_RESULTS / tag / "price_spread_vs_texas" / f"bus_spread_metrics_{season}.csv"
    if not spread_csv.exists():
        raise FileNotFoundError(
            f"Missing {spread_csv}. For Fig4 run compute_spread_vs_texas.py --fig4-converged; "
            f"for the open-loop table run it with --season {season}."
        )
    all_df = pd.read_csv(spread_csv)

    scenarios = [
        ("no_dc", NODC),
        ("baseline", FCFS),
        ("reallocation_slack_termination", FULL),
    ]
    fields: dict[str, pd.DataFrame] = {
        sc: all_df[all_df["scenario"] == sc].copy() for sc, _ in scenarios
    }
    dc_sites = _load_epoch_dc_sites()
    paper_op_colors = {**OPERATOR_COLORS, "meta": FIG5_META_COLOR}
    dc_colors = [paper_op_colors.get(str(op).lower(), "#E64B35") for op in dc_sites["operator"]]

    d_cmap = _direction_cmap()
    d_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    d_levels = np.linspace(-1.0, 1.0, 21)

    m_vals = np.concatenate([fields[sc]["A_plus_plus_A_minus"].to_numpy(float) for sc, _ in scenarios])
    vmax_m = _vmax_gross_magnitude(m_vals, floor=500.0, cap=8_000.0, step=500.0)
    m_norm = Normalize(vmin=0.0, vmax=vmax_m)
    m_cmap = _magnitude_cmap()
    m_levels = np.linspace(0.0, vmax_m, MAE_N_LEVELS)

    dx = float(CONUS_XLIM[1] - CONUS_XLIM[0])
    dy = float(CONUS_YLIM[1] - CONUS_YLIM[0])
    fig_w = float(figsize_double_col(1.0)[0])
    panel_w = fig_w * (MAP_RIGHT - MAP_LEFT) / 3.0
    panel_h = panel_w * (dy / dx)
    fig, outer = _contour_board_figure(panel_h, fig_w, bottom=MAP_FIG5_BOTTOM, fig_extra=MAP_FIG5_EXTRA)
    row_titles = (
        "a  Directional price bias relative to Texas",
        "b  Gross magnitude from Texas",
    )
    short_labels = {
        "no_dc": NODC,
        "baseline": FCFS,
        "reallocation_slack_termination": FULL,
    }
    left_axes: list = []
    map_rows: list[list] = []
    title_artists: list = []

    for block, key, is_dir in (
        (0, "directional_bias_D", True),
        (1, "A_plus_plus_A_minus", False),
    ):
        sub, title_artist = _add_row_title_strip(
            fig, outer[block], row_titles[block], fontsize=MAP_FS_ROW_TITLE, ncols=3
        )
        title_artists.append(title_artist)
        row_axes: list = []
        for idx, (sc, label) in enumerate(scenarios):
            ax = fig.add_subplot(sub[1, idx])
            row_axes.append(ax)
            if idx == 0:
                left_axes.append(ax)
            draw_base_map(ax, geo, {}, alpha=0.95, default_color="#ececec")
            df = fields[sc]
            grid_x, grid_y, z = _interpolate_bus_field(
                df["longitude"].to_numpy(float),
                df["latitude"].to_numpy(float),
                df[key].to_numpy(float),
                geo,
                grid_n=300,
            )
            z_m = np.ma.masked_invalid(z)
            draw_conus_borders(ax, geo, zorder=3)
            draw_state_borders(
                ax,
                geo,
                "TX",
                edgecolor=FIG5_REF_BORDER_COLOR,
                linewidth=FIG5_REF_BORDER_LW,
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
            if sc != "no_dc":
                ax.scatter(
                    dc_sites["lon"], dc_sites["lat"],
                    s=FIG5_STAR_S, marker="*", c=dc_colors,
                    edgecolors=FIG5_STAR_EDGE, linewidths=FIG5_STAR_EDGE_LW, zorder=8,
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
            shown = short_labels.get(sc, label)
            if shown == NODC:
                shown = "Without\ndata centers"
            elif shown in {FCFS, FULL}:
                head, tail = shown.split("-", 1)
                shown = f"{head}-\n{tail}"
            ax.text(
                0.01, 0.03, shown,
                transform=ax.transAxes, fontsize=fs_tag, fontweight="bold",
                va="bottom", ha="left", color="0.1", zorder=10,
                linespacing=0.95,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15, "alpha": 0.92},
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

    x0 = float(map_rows[1][0].get_position().x0)
    x1 = float(map_rows[1][-1].get_position().x1)
    # Legend sits below the Longitude axis labels (not between maps and xlabel).
    fig.canvas.draw()
    y_axes = float(map_rows[1][0].get_position().y0)
    leg_h = MAP_LEGEND_H
    leg_y = min(MAP_LEGEND_Y0, max(0.008, y_axes - 0.10))
    leg_ax = fig.add_axes([x0, leg_y, x1 - x0, leg_h])
    leg_ax.set_axis_off()
    handles = [
        Line2D(
            [0], [0], marker="*", color="w",
            markerfacecolor=paper_op_colors[op],
            markeredgecolor=FIG5_STAR_EDGE,
            markeredgewidth=FIG5_LEGEND_EDGE_LW,
            markersize=FIG5_LEGEND_MARKERSIZE,
            linestyle="None",
            label=OPERATOR_LABELS[op],
        )
        for op in OPERATOR_ORDER
    ]
    leg_ax.legend(
        handles=handles,
        loc="center",
        ncol=len(OPERATOR_ORDER),
        fontsize=MAP_FS_LAB,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    return save_figure(fig, out, pad_inches=0.08)





def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(build_fig5())
    print(_build_usa_endogenous_contour_board(
        "winter", OUT / "ExtData_Fig6_usa_endogenous_maps_winter.png", tag="priority_168h"
    ))


if __name__ == "__main__":
    main()
