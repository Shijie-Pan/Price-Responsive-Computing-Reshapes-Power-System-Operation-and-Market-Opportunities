from __future__ import annotations
"""Fig. 4 transmission loading."""

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

OUT = ROOT / "experiments/E04/figures"

def build_fig3() -> Path:
    """Summer then winter: No-DC / FCFS / Full flex maps (a–f) + congested-line series (g–h).

    Map hour = argmax congested lines under Full flex (same hour for all three columns).
    """
    import cartopy.crs as ccrs
    from experiments.lib.pypsa.geo_network_plot import panel_aspect
    from experiments.lib.map_hour import portfolios_from_hourly
    from experiments.lib.pf_maps import (
        annotation_style_for,
        draw_geo_panel_abs_loading,
        load_e02_geo_context,
        place_abs_legend,
        plot_scale_for,
    )
    from experiments.lib.pf_compare import (
        CONGESTION_LINE_COLORS,
        CONGESTION_PLOT_KEYS,
        _congestion_count,
        load_hourly_frames,
    )

    apply_figure_style()
    plt.rcParams.update(
        {
            "font.size": FS,
            "axes.labelsize": FS,
            "xtick.labelsize": FS_TICK_COMPOSED,
            "ytick.labelsize": FS_TICK_COMPOSED,
            "legend.fontsize": FS_LEGEND,
        }
    )

    from experiments.lib.pypsa.geo_network_plot import edge_coord_key

    e02 = E02_ROOT
    scale = plot_scale_for(e02)
    _, _, abs_thr, abs_label_fs = annotation_style_for(scale)
    n, dc_map, meta, dc_refs, site_colors = load_e02_geo_context(e02)
    paper_op_colors = {**OPERATOR_COLORS, "meta": FIG5_META_COLOR}
    meta_blue = OPERATOR_COLORS.get("meta", "#0866ff")
    for sid, color in list(site_colors.items()):
        if color == meta_blue:
            site_colors[sid] = "#111111"

    scenario_cols = (
        ("no_dc", NODC),
        ("baseline", FCFS),
        ("reallocation_slack_termination", FULL),
    )
    scenario_keys = [k for k, _ in scenario_cols]
    season_rows = (("summer", "Summer"), ("winter", "Winter"))
    panel_letters = (("a", "b", "c"), ("d", "e", "f"))

    def _common_congestion_keys(port_row: dict, *, threshold: float, eps: float = 0.05) -> set:
        """Edges with the same loading (≥threshold) across No-DC / FCFS / Full flex.

        Matching colored lines are drawn at half width so relocated congestion stands out.
        """
        load_maps: list[dict[tuple, float]] = []
        for key in scenario_keys:
            edges = port_row.get(key, {}).get("edges") or []
            load_maps.append(
                {
                    edge_coord_key(e): float(e.get("loading_pct") or 0.0)
                    for e in edges
                }
            )
        if not load_maps:
            return set()
        keys = set().union(*(m.keys() for m in load_maps))
        common = set()
        for ek in keys:
            vals = [m.get(ek, 0.0) for m in load_maps]
            # Only de-emphasize if every config highlights the line and loadings match.
            if all(v >= threshold for v in vals) and (max(vals) - min(vals)) <= eps:
                common.add(ek)
        return common

    folder = e02 / "results/flexibility/full"
    ports: dict[str, dict] = {}
    map_hours: dict[str, int] = {}
    common_keys: dict[str, set] = {}
    for season in ("summer", "winter"):
        frames, hmeta = load_hourly_frames(folder, season)
        if not frames or "reallocation_slack_termination" not in frames:
            raise FileNotFoundError(f"Missing hourly frames for Fig3 maps: {folder}/hourly_{season}")
        counts = _congestion_count(frames["reallocation_slack_termination"])
        hour = int(np.argmax(counts))
        labels = list(hmeta.get("snapshots") or [])
        label = labels[hour] if hour < len(labels) else f"hour {hour}"
        ports[season] = portfolios_from_hourly(frames, hour=hour, label=label, static_n=n)
        map_hours[season] = hour
        common_keys[season] = _common_congestion_keys(ports[season], threshold=abs_thr)

    # Manual layout in inches so cartopy aspect cannot letterbox into white gaps.
    fig_w = float(figsize_double_col(1.0)[0])
    left_in = fig_w * FIG3_LEFT
    right_in = fig_w * (1.0 - FIG3_RIGHT)
    usable_w = fig_w - left_in - right_in
    # Three map panels + two gutters (wspace as fraction of panel width).
    panel_w = usable_w / (3.0 + 2.0 * FIG3_MAP_WSPACE)
    gutter = FIG3_MAP_WSPACE * panel_w
    panel_h = panel_w / max(panel_aspect(scale), 1e-6)
    line_h = panel_h * (FIG3_MAP_LINE_HRATIO[2] / FIG3_MAP_LINE_HRATIO[0])
    top_pad, bot_pad = 0.28, 0.14
    gap_map_rows = 0.40  # room for a two-line season / strategy title between map rows
    op_leg_h = 0.18
    gap_maps_to_op = 0.04 * 1.30  # +30% before operator legend
    gap_op_to_lines = 0.22  # clear band for g/h titles above the line panels
    bot_leg_h = 0.34  # legend sits below Hour labels
    fig_h = (
        top_pad
        + panel_h
        + gap_map_rows
        + panel_h
        + gap_maps_to_op
        + op_leg_h
        + gap_op_to_lines
        + line_h
        + bot_leg_h
        + bot_pad
    )
    if FIG3_HEIGHT_IN is not None:
        fig_h = float(FIG3_HEIGHT_IN)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    map_axes: list = []
    lc_ref = None
    panel_fs = FIG3_PANEL_FS

    def _add_map(
        x_in: float,
        y_in: float,
        edges,
        letter: str,
        season_title: str,
        mode_label: str,
        shared_keys,
        *,
        show_sites: bool,
    ):
        nonlocal lc_ref
        ax = fig.add_axes(
            [x_in / fig_w, y_in / fig_h, panel_w / fig_w, panel_h / fig_h],
            projection=ccrs.PlateCarree(),
        )
        map_axes.append(ax)
        lc = draw_geo_panel_abs_loading(
            ax,
            edges,
            n,
            scale=scale,
            dc_map=dc_map if show_sites else {},
            dc_meta=meta,
            dc_refs=dc_refs,
            loading_threshold=abs_thr,
            label_fontsize=abs_label_fs,
            site_colors=site_colors,
            common_edge_keys=shared_keys,
            common_lw_scale=0.5,
        )
        if lc is not None:
            lc_ref = lc
        ax.text(
            0.0,
            1.02,
            f"{letter}  {season_title}\n{mode_label}",
            transform=ax.transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            va="bottom",
            ha="left",
            linespacing=1.05,
            clip_on=False,
            zorder=30,
        )
        return ax

    y_summer = fig_h - top_pad - panel_h
    y_winter = y_summer - gap_map_rows - panel_h
    for row_i, ((season, season_title), letters, y_in) in enumerate(
        zip(season_rows, panel_letters, (y_summer, y_winter))
    ):
        for col_i, ((key, mode_label), letter) in enumerate(zip(scenario_cols, letters)):
            x_in = left_in + col_i * (panel_w + gutter)
            edges = pd.DataFrame(ports[season].get(key, {}).get("edges") or [])
            _add_map(
                x_in,
                y_in,
                edges,
                letter,
                season_title,
                mode_label,
                common_keys[season],
                show_sites=(key != "no_dc"),
            )

    apply_figure_style()

    y_op = y_winter - gap_maps_to_op - op_leg_h
    y_lines = y_op - gap_op_to_lines - line_h
    line_gutter = FIG3_LINE_WSPACE * (usable_w / (2.0 + FIG3_LINE_WSPACE))
    line_w = (usable_w - line_gutter) / 2.0
    y_max = 1.0
    line_axes = []
    for col, season in enumerate(("summer", "winter")):
        x_in = left_in + col * (line_w + line_gutter)
        ax = fig.add_axes([x_in / fig_w, y_lines / fig_h, line_w / fig_w, line_h / fig_h])
        line_axes.append(ax)
        frames, _ = load_hourly_frames(folder, season)
        for key in CONGESTION_PLOT_KEYS:
            if key not in frames:
                continue
            y = _congestion_count(frames[key])
            color = CONGESTION_LINE_COLORS[key]
            label = {"no_dc": NODC, "baseline": FCFS, "reallocation_slack_termination": FULL}.get(key, key)
            ax.plot(np.arange(len(y)), y, color=color, lw=1.0, label=label)
            if len(y):
                y_max = max(y_max, float(np.max(y)))
        ax.axvline(map_hours[season], color="#888888", lw=0.7, ls="--", zorder=1)
        for d in range(1, 7):
            ax.axvline(d * 24, color="#E6E6E6", lw=0.4)
        ax.set_xlim(0, 167)
        ax.set_xlabel("Hour", fontsize=FS, labelpad=1)
        if col == 0:
            ax.set_ylabel("Congested lines", fontsize=FS)
        ax.tick_params(labelsize=FS_TICK_COMPOSED)
    for ax in line_axes:
        ax.set_ylim(0, y_max * 1.20)

    left = min(ax.get_position().x0 for ax in map_axes[3:])
    right = max(ax.get_position().x1 for ax in map_axes[3:])
    op_ax = fig.add_axes([left, y_op / fig_h, right - left, op_leg_h / fig_h])
    place_abs_legend(fig, ax=op_ax, fontsize=FIG3_OP_LEGEND_FS, operator_colors=paper_op_colors)

    # g/h titles in figure coords clearly above the panels.
    fig.canvas.draw()
    for col, season in enumerate(("summer", "winter")):
        pos = line_axes[col].get_position()
        fig.text(
            pos.x0,
            pos.y1 + 0.016,
            f"{'gh'[col]}  {season.capitalize()}",
            fontsize=panel_fs,
            fontweight="bold",
            va="bottom",
            ha="left",
            color="0.1",
        )

    y_lines_bot = line_axes[0].get_position().y0
    x0_lines = min(ax.get_position().x0 for ax in line_axes)
    x1_lines = max(ax.get_position().x1 for ax in line_axes)

    if lc_ref is not None:
        top_pos = map_axes[2].get_position()
        bot_pos = map_axes[5].get_position()
        cbar_w = 0.012
        cax = fig.add_axes([top_pos.x1 + 0.010, bot_pos.y0, cbar_w, top_pos.y1 - bot_pos.y0])
        cbar = fig.colorbar(lc_ref, cax=cax, orientation="vertical")
        cbar.set_label("Line loading [%]", fontsize=FIG3_CBAR_LABEL_FS, labelpad=2)
        cbar.ax.tick_params(labelsize=FS_TICK_COMPOSED, pad=1)
        cbar.outline.set_linewidth(0.4)

    # Legend below the Hour axis labels (use tight bbox so it clears xlabels).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    y_below_hour = min(
        float(fig.transFigure.inverted().transform((0, ax.get_tightbbox(renderer).y0))[1])
        for ax in line_axes
    )
    bot_h = 0.028
    bot_y = max(0.006, y_below_hour - bot_h - 0.012)
    bot = fig.add_axes([x0_lines, bot_y, x1_lines - x0_lines, bot_h])
    bot.set_axis_off()
    bot.set_xlim(0, 1)
    bot.set_ylim(0, 1)
    bot_entries = [
        (COLOR_NODC, NODC),
        (COLOR_FCFS, FCFS),
        (COLOR_FULL, FULL),
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_w_px = max(bot.get_window_extent(renderer=renderer).width, 1.0)
    dpi = float(fig.dpi)
    fs = FIG3_BOT_LEGEND_FS
    line_pt = 14.0
    pad_px = 6.0
    gap_px = 28.0

    def _tw(s: str) -> float:
        t = bot.text(0.0, 2.0, s, fontsize=fs, alpha=0.0)
        w = t.get_window_extent(renderer=renderer).width
        t.remove()
        return float(w)

    widths = [(line_pt / 72.0) * dpi + pad_px + _tw(lab) for _, lab in bot_entries]
    x_px = 0.5 * ax_w_px - 0.5 * (sum(widths) + gap_px * (len(bot_entries) - 1))
    for (color, lab), w_px in zip(bot_entries, widths):
        x = x_px / ax_w_px
        line_frac = (line_pt / 72.0) * dpi / ax_w_px
        bot.plot(
            [x, x + line_frac],
            [0.55, 0.55],
            color=color,
            lw=1.8,
            solid_capstyle="butt",
            transform=bot.transAxes,
            clip_on=False,
        )
        bot.text(
            x + line_frac + pad_px / ax_w_px,
            0.50,
            lab,
            fontsize=fs,
            va="center",
            ha="left",
            color="#222222",
            transform=bot.transAxes,
            clip_on=False,
        )
        x_px += w_px + gap_px

    out = OUT / "Fig3_congestion_relocation.png"
    return save_figure(fig, out, pad_inches=FIG3_SAVE_PAD)




def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(build_fig3())


if __name__ == "__main__":
    main()
