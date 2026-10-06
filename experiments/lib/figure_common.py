from __future__ import annotations
"""Shared helpers for the manuscript figures."""

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

from experiments.lib.figure_style import (  # noqa: E402
    COLOR_FCFS,
    COLOR_FULL,
    COLOR_NODC,
    EXT_LOOP_BOTTOM,
    EXT_LOOP_LEFT,
    EXT_LOOP_MARKER,
    EXT_LOOP_RIGHT,
    EXT_LOOP_TOP,
    EXT_LOOP_WSPACE,
    EXT_LOOP_YLABEL_PAD,
    FCFS,
    NODC,
    FIG1_STATE_DOT_S,
    FIG1_REF_BORDER_COLOR,
    FIG1_REF_BORDER_LW,
    FIG1_REF_SQUARE_S,
    FIG2_BOTTOM,
    FIG2_DOT_S,
    FIG2_HSPACE,
    FIG2_LEFT,
    FIG2_LEGEND_MS,
    FIG2_RIGHT,
    FIG2_TOP,
    FIG2_WSPACE,
    FIG3_BOTTOM,
    FIG3_BOT_LEGEND_FS,
    FIG3_CBAR_LABEL_FS,
    FIG3_HEIGHT_IN,
    FIG3_LEFT,
    FIG3_MAP_HSPACE,
    FIG3_MAP_LINE_HRATIO,
    FIG3_MAP_WSPACE,
    FIG3_LINE_WSPACE,
    FIG3_OP_LEGEND_FS,
    FIG3_OUTER_HSPACE,
    FIG3_PANEL_FS,
    FIG3_RIGHT,
    FIG3_SAVE_PAD,
    FIG3_TOP,
    FIG4_BOTTOM,
    FIG4_HSPACE,
    FIG4_LEFT,
    FIG4_PCT_FS,
    FIG4_RIGHT,
    FIG4_TOP,
    FIG5_LEGEND_EDGE_LW,
    FIG5_LEGEND_FONTSCALE,
    FIG5_LEGEND_MARKERSIZE,
    FIG5_META_COLOR,
    FIG5_REF_BORDER_COLOR,
    FIG5_REF_BORDER_LW,
    FIG5_STAR_EDGE,
    FIG5_STAR_EDGE_LW,
    FIG5_STAR_S,
    FS,
    FS_LEGEND,
    FS_TICK_COMPOSED,
    FS_TITLE,
    FULL,
    MAE_CONTOUR_COLOR,
    MAE_CONTOUR_LW,
    MAE_N_LEVELS,
    MAP_CBAR_W,
    MAP_CBAR_X,
    MAP_CBAR_PAD,
    MAP_FIG1_BOTTOM,
    MAP_FIG5_BOTTOM,
    MAP_FIG5_EXTRA,
    MAP_FIG_EXTRA,
    MAP_FS_CLABEL,
    MAP_FS_LAB,
    MAP_FS_ROW_TITLE,
    MAP_FS_TAG,
    MAP_FS_TICK,
    MAP_HSPACE,
    MAP_LEFT,
    MAP_LEGEND_H,
    MAP_LEGEND_Y0,
    MAP_RIGHT,
    MAP_TITLE_HEIGHT,
    MAP_TITLE_HSPACE,
    MAP_TOP,
    MAP_WSPACE,
    PHASE_CONTOUR_COLOR,
    PHASE_CONTOUR_LW,
    PHASE_LEVEL_STEP,
    PHASE_VSHOW,
)
from experiments.lib.figures.nc_style import (  # noqa: E402
    FIG_DPI,
    OPERATOR_COLORS,
    OPERATOR_LABELS,
    OPERATOR_ORDER,
    apply_figure_style,
    figsize_double_col,
    save_figure,
)
from experiments.lib.state_correlation import (  # noqa: E402
    _load_all_state_prices,
    mean_abs_lmp_spread_matrix,
    price_phase_lag,
    reference_field,
)
from experiments.lib.state_lmp_loader import (  # noqa: E402
    STATE_CENTROIDS,
)
from experiments.lib.us_map import (  # noqa: E402
    CONUS_XLIM,
    CONUS_YLIM,
    draw_base_map,
    draw_conus_borders,
    draw_state_borders,
    load_geojson,
    state_display_name,
)
from experiments.lib.plot_correlation_contours import (  # noqa: E402
    CONUS_EXCLUDE,
    CMAP,
    FILL_ALPHA,
    _conus_states,
    _interpolate_field,
    _load_matrix,
)
from experiments.lib.lmp_io import load_hourly_prices  # noqa: E402
from experiments.lib.texas_config import get_operator_sites, get_site_lmp_csv, load_texas7_config  # noqa: E402

_PFIXED_RATIO = 0.2
E00_RESULTS = ROOT / "experiments/E02/results"
E01_MASTER = ROOT / "experiments/E03/results/e01_master_summary.csv"
E02_ROOT = ROOT / "experiments/E04"
E03_SUMMER = ROOT / "experiments/E07/results/summer/comparison_summer.json"
E03_WINTER = ROOT / "experiments/E07/results/winter/comparison_winter.json"
E03_RES = ROOT / "experiments/E07/results"
USA_E02_RESULTS = ROOT / "experiments/E05/results"
USA_E02_CONFIG = ROOT / "experiments/E05/config"

def _panel_tag(
    ax,
    letter: str,
    title: str = "",
    *,
    x: float = 0.0,
    y: float = 1.02,
    fontsize: float | None = None,
    fontweight: str = "bold",
) -> None:
    text = f"{letter}  {title}".rstrip() if title else letter
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=FS_TITLE if fontsize is None else fontsize,
        fontweight=fontweight,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _draw_texas_ref_marker(ax, lon: float, lat: float, *, annotate: bool = False) -> None:
    """Black bullseye (⊙) for the shared Texas reference."""
    ax.scatter(
        [lon],
        [lat],
        s=FIG1_REF_SQUARE_S * 2.4,
        marker="o",
        facecolors="none",
        edgecolors="0.05",
        linewidths=1.35,
        zorder=9,
    )
    ax.scatter(
        [lon],
        [lat],
        s=FIG1_REF_SQUARE_S * 0.55,
        marker="o",
        c="0.05",
        linewidths=0,
        zorder=10,
    )
    if annotate:
        ax.annotate(
            "Texas reference bus",
            xy=(lon, lat),
            xytext=(lon - 14.0, lat - 7.5),
            fontsize=MAP_FS_TAG,
            color="0.1",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="0.25", lw=0.7),
            zorder=12,
            ha="right",
            va="top",
        )


def _vmax_gross_magnitude(
    vals: list[float] | np.ndarray,
    *,
    floor: float,
    cap: float,
    step: float,
) -> float:
    """Nice upper bound for Gross magnitude colour scales."""
    arr = np.asarray(list(vals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(floor)
    p95 = float(np.percentile(arr, 95))
    vmax = float(np.ceil(p95 / step) * step)
    return float(max(floor, min(vmax, cap)))


def _fmt_gross_tick(v: float, _pos=None) -> str:
    """Compact colour-bar ticks for Gross magnitude (A₊+A₋).

    Values that are not a whole number of thousands keep one decimal so
    that 1.5k and 2k are not both drawn as 2k.
    """
    av = abs(float(v))
    if av >= 1000.0:
        x = float(v) / 1000.0
        if abs(x - round(x)) < 0.05:
            return f"{x:.0f}k"
        return f"{x:.1f}k"
    return f"{int(round(v))}"


def _draw_reference_marker(ax, lon: float, lat: float) -> None:
    """Backward-compatible alias used by ExtData phase-lag boards."""
    ax.scatter(
        [lon],
        [lat],
        s=FIG1_REF_SQUARE_S,
        marker="s",
        c="0.1",
        linewidths=0,
        zorder=8,
    )


def _contour_board_figure(
    panel_h: float,
    fig_w: float,
    *,
    bottom: float = 0.08,
    fig_extra: float = MAP_FIG_EXTRA,
):
    """Two map rows with dedicated title strips above each (titles never overlap the map)."""
    fig_h = 2.0 * panel_h + fig_extra
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        hspace=MAP_HSPACE,
        left=MAP_LEFT,
        right=MAP_RIGHT,
        top=MAP_TOP,
        bottom=bottom,
    )
    return fig, outer


def _add_row_title_strip(fig, outer_slot, title: str, *, fontsize: float, ncols: int = 3):
    """Title row above map panels; returns (map_subgridspec, title_text_artist)."""
    sub = outer_slot.subgridspec(
        2,
        ncols,
        height_ratios=[MAP_TITLE_HEIGHT, 1.0],
        hspace=MAP_TITLE_HSPACE,
        wspace=MAP_WSPACE,
    )
    # Span the full row width so the provisional title is already centered.
    tax = fig.add_subplot(sub[0, :])
    tax.set_axis_off()
    title_artist = tax.text(
        0.5,
        0.5,
        title,
        transform=tax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="center",
        ha="center",
        color="0.1",
        clip_on=False,
    )
    return sub, title_artist


def _center_row_titles(
    fig, map_rows, title_artists, *, fontsize: float, ha: str = "center"
) -> None:
    """Place a/b titles above each map row (outside the axes)."""
    fig.canvas.draw()
    for tax_text, row_axes in zip(title_artists, map_rows):
        p0 = row_axes[0].get_position()
        p2 = row_axes[-1].get_position()
        x = p0.x0 if ha == "left" else 0.5 * (p0.x0 + p2.x1)
        y_mid = float(fig.transFigure.inverted().transform(tax_text.get_window_extent().p0)[1])
        y_top = float(fig.transFigure.inverted().transform(tax_text.get_window_extent().p1)[1])
        y_title = 0.5 * (y_mid + y_top)
        tax_text.set_visible(False)
        fig.text(
            x,
            y_title,
            tax_text.get_text(),
            ha=ha,
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="0.1",
        )


def _direction_cmap():
    """Blue = Texas higher (D=-1), white = balanced, red = other higher (D=+1)."""
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f7f7f7")
    return cmap


def _magnitude_cmap():
    cmap = mpl_cm.YlOrRd.copy()
    cmap.set_bad("#f7f7f7")
    return cmap


def _corr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 12) -> tuple[int, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
        return 0, 0.0
    best_lag, best_r = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = a[-lag:], b[: len(b) + lag]
        elif lag > 0:
            x, y = a[: len(a) - lag], b[lag:]
        else:
            x, y = a, b
        if len(x) < 3:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(r) and r > best_r:
            best_r, best_lag = r, lag
    return best_lag, best_r


def _load_usa_bus_lmp(scenario: str, season: str, tag: str = "priority_168h") -> tuple[pd.DataFrame, pd.DataFrame]:
    d = USA_E02_RESULTS / tag / season / scenario
    raw = pd.read_csv(d / "bus_lmp_hourly.csv")
    # Bus ids are p1, p100, ... — not "period" / "timestep".
    bus_cols = [c for c in raw.columns if str(c).startswith("p") and str(c)[1:].isdigit()]
    mp = raw[bus_cols].apply(pd.to_numeric, errors="coerce")
    geo = pd.read_csv(d / "bus_lmp_summary.csv")
    geo["bus"] = geo["bus"].astype(str)
    return mp, geo


def _load_epoch_dc_sites() -> pd.DataFrame:
    meta = json.loads((USA_E02_CONFIG / "dc_bus_map_meta.json").read_text(encoding="utf-8"))
    rows = []
    for _sid, s in meta["sites"].items():
        rows.append(
            {
                "site_id": int(s["site_id"]),
                "operator": str(s["operator"]),
                "lon": float(s["reference_lon"]),
                "lat": float(s["reference_lat"]),
                "pypsa_bus": str(s["pypsa_bus"]),
            }
        )
    return pd.DataFrame(rows)


def _resolve_texas_ref_bus(mp: pd.DataFrame, bus_geo: pd.DataFrame) -> str:
    """ERCOT bus with the most mapped Texas DC sites (Fig1 Texas reference)."""
    sites = _load_epoch_dc_sites()
    meta = json.loads((USA_E02_CONFIG / "dc_bus_map_meta.json").read_text(encoding="utf-8"))
    tx_buses: list[str] = []
    for s in meta["sites"].values():
        if s.get("reeds_ba") == "ERCO" or s.get("reeds_state") == "TX":
            tx_buses.append(str(s["pypsa_bus"]))
    if not tx_buses:
        return str(bus_geo.sort_values("lon").iloc[0]["bus"])
    ref = pd.Series(tx_buses).value_counts().index[0]
    cols = {str(c): c for c in mp.columns}
    return cols.get(str(ref), str(ref))


def _bus_fields_vs_ref(
    mp: pd.DataFrame,
    geo: pd.DataFrame,
    *,
    ref_bus: str | None = None,
) -> tuple[pd.DataFrame, str]:
    if ref_bus is None:
        ref_bus = _resolve_texas_ref_bus(mp, geo)
    if ref_bus not in mp.columns:
        cols = {str(c): c for c in mp.columns}
        ref_bus = cols[ref_bus]
    ref = mp[ref_bus].to_numpy(dtype=float)
    geo_ix = geo.set_index("bus")
    rows = []
    for bus in mp.columns:
        series = mp[bus].to_numpy(dtype=float)
        lag, _ = _corr_lag(ref, series)
        mae = float(np.mean(np.abs(series - ref)))
        g = geo_ix.loc[str(bus)]
        rows.append(
            {
                "bus": str(bus),
                "lon": float(g["lon"]),
                "lat": float(g["lat"]),
                "phase_lag_h": int(lag),
                "mean_abs_delta_vs_ref": mae,
            }
        )
    return pd.DataFrame(rows), str(ref_bus)


def _interpolate_bus_field(
    lons: np.ndarray,
    lats: np.ndarray,
    vals: np.ndarray,
    geo: dict,
    *,
    grid_n: int = 300,
):
    from scipy.interpolate import RBFInterpolator
    from experiments.lib.us_map import mask_field_to_conus

    gx = np.linspace(CONUS_XLIM[0], CONUS_XLIM[1], grid_n)
    gy = np.linspace(CONUS_YLIM[0], CONUS_YLIM[1], grid_n)
    grid_x, grid_y = np.meshgrid(gx, gy)
    rbf = RBFInterpolator(
        np.column_stack([lons, lats]),
        vals,
        kernel="linear",
        smoothing=1.0,
    )
    z = rbf(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    return grid_x, grid_y, mask_field_to_conus(geo, grid_x, grid_y, z)



__all__ = [n for n in globals() if not n.startswith("__")]
