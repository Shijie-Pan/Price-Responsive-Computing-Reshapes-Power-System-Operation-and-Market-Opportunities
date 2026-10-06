#!/usr/bin/env python3
"""E00: reference-state phase-lag contour maps on US geography."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

from experiments.lib.paths import package_root

ROOT = package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.state_correlation import (  # noqa: E402
    _load_all_state_prices,
    compute_all_directed_matrices,
    reference_field,
)
from experiments.lib.state_lmp_loader import STATE_CENTROIDS  # noqa: E402
from experiments.lib.us_map import (  # noqa: E402
    CONUS_XLIM,
    CONUS_YLIM,
    draw_base_map,
    load_geojson,
    state_display_name,
)

DEFAULT_RESULTS = ROOT / "experiments" / "E02" / "results"
DEFAULT_FIGURES = ROOT / "experiments" / "E02" / "figures"
DEFAULT_REFS = ("MO", "TX", "IL", "CA", "VA", "NV")
CONUS_EXCLUDE = frozenset({"AK", "HI"})
PHASE_LAG_LEVELS = np.array([-12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12], dtype=float)
FILL_ALPHA = 0.52
CMAP = plt.colormaps["RdBu_r"].copy()
CMAP.set_bad("#f0f0f0")


def _conus_states(states) -> list:
    return [s for s in states if s.state not in CONUS_EXCLUDE]


def _interpolate_field(states, values: dict[str, float], *, grid_n: int = 220, geo: dict | None = None):
    """Return (grid_x, grid_y, z, z) with values on every CONUS land cell.

    Uses RBF over state centroids so the field extends to the national border
    without convex-hull cutoffs (no nearest-Voronoi coastal artifacts).
    If geo is given, ocean cells are set to NaN here; otherwise the caller masks.
    """
    from scipy.interpolate import RBFInterpolator

    lons = np.array([s.lon for s in states], dtype=float)
    lats = np.array([s.lat for s in states], dtype=float)
    vals = np.array([values[s.state] for s in states], dtype=float)
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
    if geo is not None:
        from experiments.lib.us_map import mask_field_to_conus

        z = mask_field_to_conus(geo, grid_x, grid_y, z)
    return grid_x, grid_y, z, z


def _plot_panel(
    ax,
    geo,
    states,
    values,
    *,
    ref_state: str,
    norm: TwoSlopeNorm,
    show_ylabel: bool = False,
    show_xlabel: bool = False,
):
    draw_base_map(ax, geo, {}, alpha=0.95, default_color="#ececec")
    grid_x, grid_y, z, _ = _interpolate_field(states, values, geo=geo)
    cf = ax.contourf(
        grid_x, grid_y, np.ma.masked_invalid(z),
        levels=PHASE_LAG_LEVELS,
        cmap=CMAP,
        norm=norm,
        alpha=FILL_ALPHA,
        extend="both",
    )
    cs = ax.contour(
        grid_x, grid_y, np.ma.masked_invalid(z),
        levels=PHASE_LAG_LEVELS,
        colors="0.15",
        linewidths=0.45,
        alpha=0.9,
    )
    ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.0f")
    ax.scatter([s.lon for s in states], [s.lat for s in states], s=16, c="0.1", zorder=5, alpha=0.75)
    rlon, rlat = STATE_CENTROIDS[ref_state]
    ax.scatter([rlon], [rlat], s=110, c="gold", edgecolors="0.15", linewidths=0.9, zorder=6, marker="*")
    ax.set_title(state_display_name(ref_state), fontsize=10, fontweight="bold", pad=2)
    if show_ylabel:
        ax.set_ylabel("Latitude", fontsize=8)
    else:
        ax.set_yticklabels([])
    if show_xlabel:
        ax.set_xlabel("Longitude", fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=7)
    return cf


def plot_reference_panel(
    *,
    season: str,
    states,
    matrix: pd.DataFrame,
    geo: dict,
    out_dir: Path,
    ref_states: tuple[str, ...],
) -> Path:
    from experiments.lib.figures.nc_style import (  # local import keeps plot deps light
        apply_figure_style,
        figsize_double_col,
        save_figure,
    )

    apply_figure_style()
    conus = _conus_states(states)
    ncols = 3
    nrows = 2
    norm = TwoSlopeNorm(
        vmin=float(PHASE_LAG_LEVELS.min()), vcenter=0.0, vmax=float(PHASE_LAG_LEVELS.max()),
    )

    fig = plt.figure(figsize=figsize_double_col(5.0), facecolor="white")
    gs = fig.add_gridspec(
        nrows, ncols,
        hspace=-0.18,
        wspace=0.08,
        left=0.06,
        right=0.86,
        top=0.90,
        bottom=0.08,
        height_ratios=[1, 1],
    )
    cf = None
    for idx, ref in enumerate(ref_states):
        if idx >= nrows * ncols:
            break
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        if ref not in matrix.index:
            ax.set_visible(False)
            continue
        field = reference_field(matrix, ref)
        field_conus = {st: field[st] for st in field if st not in CONUS_EXCLUDE}
        cf = _plot_panel(
            ax, geo, conus, field_conus,
            ref_state=ref,
            norm=norm,
            show_ylabel=(col == 0),
            show_xlabel=(row == nrows - 1),
        )

    cax = fig.add_axes([0.88, 0.10, 0.025, 0.78])
    fig.colorbar(cf, cax=cax, label="Phase lag (h)")
    fig.suptitle(f"State LMP phase lag — {season}", fontsize=10, y=0.97)
    refs_tag = "-".join(ref_states[:6]).lower()
    out = out_dir / f"fig_phase_lag_refs_{refs_tag}_{season}.png"
    return save_figure(fig, out)

def _load_matrix(results_dir: Path, season: str) -> pd.DataFrame:
    for name in (f"phase_lag_matrix_{season}.csv", f"lag_centroid_matrix_{season}.csv"):
        path = results_dir / name
        if path.exists():
            return pd.read_csv(path, index_col=0)
    raise FileNotFoundError(results_dir / f"phase_lag_matrix_{season}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="E01 phase-lag contour maps")
    parser.add_argument("--season", choices=["winter", "summer", "combined", "annual", "all"], default="all")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--ref-states", default=",".join(DEFAULT_REFS))
    args = parser.parse_args()

    seasons = ["winter", "summer", "combined", "annual"] if args.season == "all" else [args.season]
    ref_states = tuple(s.strip().upper() for s in args.ref_states.split(",") if s.strip())
    geo = load_geojson()
    if not geo:
        raise SystemExit("US geojson unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        states = _load_all_state_prices(season)
        try:
            matrix = _load_matrix(args.results_dir, season)
        except FileNotFoundError:
            matrix = compute_all_directed_matrices(states)["phase_lag"]
        plot_reference_panel(
            season=season, states=states, matrix=matrix, geo=geo,
            out_dir=args.out_dir, ref_states=ref_states,
        )
        print(f"[E00 plot] -> {args.out_dir}")

if __name__ == "__main__":
    main()
