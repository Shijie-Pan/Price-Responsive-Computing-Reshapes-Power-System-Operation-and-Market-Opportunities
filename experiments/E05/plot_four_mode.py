"""Redraw Fig. 5 from deposited four-mode line loadings (summer)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.paths import package_root  # noqa: E402

ROOT = package_root()

from experiments.lib.figures.nc_style import (  # noqa: E402
    OPERATOR_COLORS,
    apply_nature_comm_style,
    save_figure,
)
from experiments.lib.pf_maps import (  # noqa: E402
    annotation_style_for,
    load_e02_geo_context,
    place_abs_legend,
)
from experiments.lib.pypsa.geo_network_plot import (  # noqa: E402
    draw_geo_panel_abs_loading,
    panel_aspect,
)

E05 = ROOT / "experiments" / "E05"
RES = E05 / "results" / "four_mode"
OUT = E05 / "figures"
BINDING = 99.9
WEEK_HOURS = 168

MODES = (
    {"key": "fcfs", "label": "Price-independent", "color": "#4C4C4C"},
    {"key": "temporal_only", "label": "Temporal only", "color": "#4E79A7"},
    {"key": "spatial_only", "label": "Spatial only", "color": "#F28E2B"},
    {"key": "spatiotemporal", "label": "Spatio-temporal", "color": "#E15759"},
)


def _loading(mode: str) -> pd.DataFrame:
    folder = RES / mode / "summer"
    pq = folder / "line_loading_percent.parquet"
    csv = folder / "line_loading_percent.csv"
    if pq.is_file():
        return pd.read_parquet(pq).astype(float)
    if csv.is_file():
        return pd.read_csv(csv, index_col=0).astype(float)
    raise FileNotFoundError(folder)


def _line_static() -> pd.DataFrame:
    for mode in MODES:
        folder = RES / mode["key"] / "summer"
        for name in ("line_static.parquet", "line_static.csv"):
            p = folder / name
            if p.is_file():
                return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0)
    raise FileNotFoundError("line_static")


def _counts(loading: pd.DataFrame) -> np.ndarray:
    return (loading.to_numpy(float) >= BINDING).sum(axis=1).astype(float)


def _edges(loading: pd.DataFrame, hour: int, static: pd.DataFrame, buses: pd.DataFrame) -> list[dict]:
    row = loading.iloc[int(hour)]
    out = []
    for line, pct in row.items():
        if line not in static.index:
            continue
        b0 = str(static.at[line, "bus0"])
        b1 = str(static.at[line, "bus1"])
        if b0 not in buses.index or b1 not in buses.index:
            continue
        out.append(
            {
                "line": line,
                "bus0": b0,
                "bus1": b1,
                "lon0": float(buses.at[b0, "x"]),
                "lat0": float(buses.at[b0, "y"]),
                "lon1": float(buses.at[b1, "x"]),
                "lat1": float(buses.at[b1, "y"]),
                "loading_pct": float(pct),
                "capacity_mw": float(static.at[line, "s_nom"]),
            }
        )
    return out


def build() -> Path:
    apply_nature_comm_style()
    frames = {m["key"]: _loading(m["key"]) for m in MODES}
    n_hours = min(len(df) for df in frames.values())
    frames = {k: df.iloc[:n_hours].copy() for k, df in frames.items()}
    counts = {k: _counts(df) for k, df in frames.items()}
    hour = int(np.argmax(counts["spatiotemporal"]))
    static = _line_static()
    scale = "texas"
    _, _, abs_thr, abs_label_fs = annotation_style_for(scale)
    n, dc_map, meta, dc_refs, site_colors = load_e02_geo_context(E05)
    meta_blue = OPERATOR_COLORS.get("meta", "#0866ff")
    for sid, color in list(site_colors.items()):
        if color == meta_blue:
            site_colors[sid] = "#111111"
    paper_op_colors = {**OPERATOR_COLORS, "meta": "#111111"}
    edge_lists = {k: _edges(frames[k], hour, static, n.buses) for k in frames}

    import cartopy.crs as ccrs

    # Leave room on the right for the shared loading colorbar + tick labels.
    fig_w = 7.55
    aspect = panel_aspect(scale)
    panel_w = 1.48
    panel_h = panel_w / max(aspect, 1e-6)
    line_h = 1.35
    title_e_h = 0.20
    legend_h = 0.20
    op_leg_h = 0.22
    top_pad = 0.40
    bot_pad = 0.32
    gap_maps_to_op = 0.08
    gap_op_to_title = 0.08
    gap_title_legend = 0.02
    gap_legend_line = 0.04
    panel_fs = 7.0
    fig_h = (
        top_pad
        + panel_h
        + gap_maps_to_op
        + op_leg_h
        + gap_op_to_title
        + title_e_h
        + gap_title_legend
        + legend_h
        + gap_legend_line
        + line_h
        + bot_pad
    )
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    x0 = 0.28
    gap = 0.12
    y_map = fig_h - top_pad - panel_h
    map_axes = []
    lc_ref = None
    for i, mode in enumerate(MODES):
        ax = fig.add_axes(
            [(x0 + i * (panel_w + gap)) / fig_w, y_map / fig_h, panel_w / fig_w, panel_h / fig_h],
            projection=ccrs.PlateCarree(),
        )
        map_axes.append(ax)
        lc = draw_geo_panel_abs_loading(
            ax,
            edge_lists[mode["key"]],
            n,
            scale=scale,
            dc_map=dc_map,
            dc_meta=meta,
            dc_refs=dc_refs,
            loading_threshold=abs_thr,
            label_fontsize=abs_label_fs,
            site_colors=site_colors,
        )
        if lc is not None:
            lc_ref = lc
        n_cong = int(counts[mode["key"]][hour])
        ax.text(
            0.0,
            1.02,
            f"{'abcd'[i]}  {mode['label']}\n{n_cong} lines",
            transform=ax.transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            va="bottom",
            ha="left",
            linespacing=1.05,
            clip_on=False,
            zorder=30,
        )

    if lc_ref is not None:
        right = max(ax.get_position().x1 for ax in map_axes)
        top = max(ax.get_position().y1 for ax in map_axes)
        bot = min(ax.get_position().y0 for ax in map_axes)
        cbar_w = 0.012
        cbar_gap = 0.010
        cax_left = min(right + cbar_gap, 0.955 - cbar_w)
        cax = fig.add_axes([cax_left, bot, cbar_w, top - bot])
        cbar = fig.colorbar(lc_ref, cax=cax, orientation="vertical")
        cbar.set_label("Line loading [%]", fontsize=6.5, labelpad=1)
        cbar.ax.tick_params(labelsize=5.5, pad=0.5, length=2, width=0.4)
        cbar.outline.set_linewidth(0.4)

    y_op = y_map - gap_maps_to_op - op_leg_h
    left = min(ax.get_position().x0 for ax in map_axes)
    right = max(ax.get_position().x1 for ax in map_axes)
    op_ax = fig.add_axes([left, y_op / fig_h, right - left, op_leg_h / fig_h])
    place_abs_legend(fig, ax=op_ax, fontsize=6.5, operator_colors=paper_op_colors)

    y_title = y_op - gap_op_to_title - title_e_h
    ext_note = (
        f"; solid band after {WEEK_HOURS} = extension"
        if n_hours > WEEK_HOURS
        else ""
    )
    fig.text(
        0.10,
        (y_title + 0.5 * title_e_h) / fig_h,
        f"e  Congested-line count (summer, {n_hours} h{ext_note})",
        fontsize=panel_fs,
        fontweight="bold",
        va="center",
        ha="left",
    )
    y_leg = y_title - gap_title_legend - legend_h
    ax_leg = fig.add_axes([0.10, y_leg / fig_h, 0.84, legend_h / fig_h])
    ax_leg.axis("off")

    ax_e = fig.add_axes([0.10, bot_pad / fig_h, 0.84, line_h / fig_h])
    t = np.arange(n_hours)
    handles = []
    for mode in MODES:
        (ln,) = ax_e.plot(
            t,
            counts[mode["key"]],
            color=mode["color"],
            lw=1.0,
            label=mode["label"],
        )
        handles.append(ln)
    ax_e.set_xlim(0, n_hours - 1)
    ax_e.axvline(hour, color="#888888", lw=0.7, ls="--", zorder=2)
    if n_hours > WEEK_HOURS:
        ax_e.axvspan(WEEK_HOURS - 0.5, n_hours - 0.5, color="#BDBDBD", alpha=0.25, zorder=0)
    ax_e.set_xlabel("Hour", fontsize=8)
    ax_e.set_ylabel("Congested lines", fontsize=8)
    ax_e.tick_params(labelsize=7)
    ax_leg.legend(
        handles=handles,
        labels=[m["label"] for m in MODES],
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=7,
        columnspacing=1.0,
        handlelength=1.6,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    meta_out = {
        "figure": "Fig5",
        "season": "summer",
        "map_hour": hour,
        "binding_threshold_pct": BINDING,
        "eval_hours": n_hours,
    }
    (OUT / "summary_fig5_summer.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    return save_figure(fig, OUT / "Fig5_four_mode_dc_summer.png")


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
