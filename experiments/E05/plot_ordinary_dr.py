"""Redraw Supplementary Fig. 5 (ordinary-load demand response) from deposited loadings."""

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
from experiments.lib.pf_compare import _congestion_count, load_hourly_frames  # noqa: E402
from experiments.lib.pf_maps import (  # noqa: E402
    annotation_style_for,
    load_e02_geo_context,
    place_abs_legend,
)
from experiments.lib.pypsa.geo_network_plot import (  # noqa: E402
    draw_geo_panel_abs_loading,
    edge_coord_key,
    panel_aspect,
)

E05 = ROOT / "experiments" / "E05"
E04_FLEX = ROOT / "experiments" / "E04" / "results" / "flexibility" / "full"
RES = E05 / "results" / "ordinary_load_dr"
OUT = E05 / "figures"
BINDING_PLOT = 99.9
BINDING_STRICT = 100.0 - 1e-9

CASES = (
    ("no_dr", "Without data centers", "#8491B4", "baseline_no_dr"),
    ("ordinary_dr", "Ordinary-load demand response", "#2ca02c", "ordinary_load_dr"),
    ("dc_pi_plus_dr", "Price-independent\n+ demand response", "#3C5488", "dc_pi_plus_dr"),
    ("dc_pr_plus_dr", "Price-responsive\n+ demand response", "#E64B35", "dc_pr_plus_dr"),
)
LEGEND_LABELS = {
    "no_dr": "Without data centers",
    "ordinary_dr": "Ordinary-load demand response",
    "dc_pi_plus_dr": "Price-independent + demand response",
    "dc_pr_plus_dr": "Price-responsive + demand response",
}
SHOW_DC_SITES = {"dc_pi_plus_dr", "dc_pr_plus_dr"}


def _read_loading(folder: Path) -> pd.DataFrame:
    pq = folder / "line_loading_percent.parquet"
    csv = folder / "line_loading_percent.csv"
    if pq.is_file():
        return pd.read_parquet(pq).astype(float)
    if csv.is_file():
        return pd.read_csv(csv, index_col=0).astype(float)
    raise FileNotFoundError(folder)


def _loading(key: str, folder_name: str) -> pd.DataFrame:
    if key == "no_dr":
        pkl = E04_FLEX / "hourly_summer" / "no_dc.pkl"
        if pkl.is_file():
            return pd.read_pickle(pkl).astype(float)
    return _read_loading(RES / folder_name / "summer")


def _fig4_summer_hour() -> int:
    frames, _ = load_hourly_frames(E04_FLEX, "summer")
    counts = _congestion_count(frames["reallocation_slack_termination"])
    return int(np.argmax(counts))


def _line_static() -> pd.DataFrame:
    for folder_name in ("ordinary_load_dr", "baseline_no_dr", "dc_pi_plus_dr"):
        folder = RES / folder_name / "summer"
        for name in ("line_static.parquet", "line_static.csv"):
            p = folder / name
            if p.is_file():
                return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0)
    raise FileNotFoundError("line_static")


def _hourly_counts(loading: pd.DataFrame, thr: float) -> np.ndarray:
    return (loading.to_numpy(float) >= thr).sum(axis=1).astype(float)


def _edges_at_hour(
    loading: pd.DataFrame,
    hour: int,
    line_static: pd.DataFrame,
    buses: pd.DataFrame,
) -> list[dict]:
    row = loading.iloc[int(hour)]
    edges = []
    for line, pct in row.items():
        if line not in line_static.index:
            continue
        b0 = str(line_static.at[line, "bus0"])
        b1 = str(line_static.at[line, "bus1"])
        if b0 not in buses.index or b1 not in buses.index:
            continue
        edges.append(
            {
                "line": line,
                "bus0": b0,
                "bus1": b1,
                "lon0": float(buses.at[b0, "x"]),
                "lat0": float(buses.at[b0, "y"]),
                "lon1": float(buses.at[b1, "x"]),
                "lat1": float(buses.at[b1, "y"]),
                "loading_pct": float(pct),
                "capacity_mw": float(line_static.at[line, "s_nom"]),
            }
        )
    return edges


def _common_keys(edge_lists: list[list[dict]], threshold: float, eps: float = 0.05) -> set:
    maps = [
        {edge_coord_key(e): float(e.get("loading_pct") or 0.0) for e in edges}
        for edges in edge_lists
    ]
    keys = set().union(*(m.keys() for m in maps))
    common = set()
    for ek in keys:
        vals = [m.get(ek, 0.0) for m in maps]
        if all(v >= threshold for v in vals) and (max(vals) - min(vals)) <= eps:
            common.add(ek)
    return common


def build(season: str = "summer") -> Path:
    apply_nature_comm_style()
    frames = {k: _loading(k, folder) for k, _, _, folder in CASES}
    common_cols = sorted(set.intersection(*(set(df.columns) for df in frames.values())))
    frames = {k: df.reindex(columns=common_cols).fillna(0.0) for k, df in frames.items()}
    hour = _fig4_summer_hour()
    counts = {k: _hourly_counts(df, BINDING_PLOT) for k, df in frames.items()}
    static = _line_static()
    scale = "texas"
    _, _, abs_thr, abs_label_fs = annotation_style_for(scale)
    n, dc_map, meta, dc_refs, site_colors = load_e02_geo_context(E05)
    meta_blue = OPERATOR_COLORS.get("meta", "#0866ff")
    for sid, color in list(site_colors.items()):
        if color == meta_blue:
            site_colors[sid] = "#111111"
    paper_op_colors = {**OPERATOR_COLORS, "meta": "#111111"}
    edge_lists = {
        k: _edges_at_hour(frames[k], hour, static, n.buses) for k, _, _, _ in CASES
    }
    shared = _common_keys(list(edge_lists.values()), threshold=abs_thr)

    import cartopy.crs as ccrs

    # Leave room on the right for the shared loading colorbar + tick labels.
    fig_w = 7.55
    aspect = panel_aspect(scale)
    panel_w = 1.48
    panel_h = panel_w / max(aspect, 1e-6)
    gap = 0.12
    line_h = 1.35
    title_e_h = 0.20
    legend_h = 0.20
    op_leg_h = 0.22
    gap_maps_to_op = 0.08
    gap_op_to_title = 0.08
    gap_title_legend = 0.02
    gap_legend_line = 0.04
    top_pad = 0.52
    bot_pad = 0.32
    panel_fs = 6.5
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
    y_map = fig_h - top_pad - panel_h

    map_axes = []
    lc_ref = None
    for i, (key, label, _color, _folder) in enumerate(CASES):
        ax = fig.add_axes(
            [(x0 + i * (panel_w + gap)) / fig_w, y_map / fig_h, panel_w / fig_w, panel_h / fig_h],
            projection=ccrs.PlateCarree(),
        )
        map_axes.append(ax)
        show_sites = key in SHOW_DC_SITES
        lc = draw_geo_panel_abs_loading(
            ax,
            edge_lists[key],
            n,
            scale=scale,
            dc_map=dc_map if show_sites else {},
            dc_meta=meta if show_sites else {},
            dc_refs=dc_refs if show_sites else {},
            loading_threshold=abs_thr,
            label_fontsize=abs_label_fs,
            site_colors=site_colors if show_sites else {},
            common_edge_keys=shared,
            common_lw_scale=0.5,
        )
        if lc is not None:
            lc_ref = lc
        n_cong = int(counts[key][hour])
        ax.text(
            0.0,
            1.02,
            f"{'abcd'[i]}  {label}\n{n_cong} lines",
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
    fig.text(
        0.10,
        (y_title + 0.5 * title_e_h) / fig_h,
        f"e  Congested-line count ({season})",
        fontsize=panel_fs,
        fontweight="bold",
        va="center",
        ha="left",
    )

    y_leg = y_title - gap_title_legend - legend_h
    ax_leg = fig.add_axes([0.10, y_leg / fig_h, 0.84, legend_h / fig_h])
    ax_leg.axis("off")

    ax_e = fig.add_axes([0.10, bot_pad / fig_h, 0.84, line_h / fig_h])
    t = np.arange(len(counts["no_dr"]))
    handles = []
    for key, label, color, _folder in CASES:
        (ln,) = ax_e.plot(
            t, counts[key], color=color, lw=1.0, label=LEGEND_LABELS.get(key, label)
        )
        handles.append(ln)
    ax_e.set_xlim(0, len(t) - 1)
    ax_e.axvline(hour, color="#888888", lw=0.7, ls="--", zorder=2)
    ax_e.set_xlabel("Hour", fontsize=8)
    ax_e.set_ylabel("Congested lines", fontsize=8)
    ax_e.tick_params(labelsize=7)
    ax_leg.legend(
        handles=handles,
        labels=[LEGEND_LABELS[c[0]] for c in CASES],
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        columnspacing=1.0,
        handlelength=1.6,
    )

    summary = {
        "figure": "ExtData_Fig5",
        "season": season,
        "map_hour": hour,
        "binding_threshold_pct_for_plot_series": BINDING_PLOT,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"summary_extdata_fig5_{season}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return save_figure(fig, OUT / "ExtData_Fig5_ordinary_load_dr_summer.png")


def main() -> None:
    print(build("summer"))


if __name__ == "__main__":
    main()
