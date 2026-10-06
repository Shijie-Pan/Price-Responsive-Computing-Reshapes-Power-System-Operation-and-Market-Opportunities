"""Geographic PyPSA network maps — unified loading-colormap style (geo_network_baseline spec)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from experiments.lib.paths import package_root

ROOT = package_root()
# Deposited geospatial only (optional; map replots degrade gracefully if absent).
_WF_CANDIDATES = (
    ROOT / "experiments" / "data" / "geospatial",
    ROOT / "data" / "geospatial",
)
WF = next((p for p in _WF_CANDIDATES if p.is_dir()), _WF_CANDIDATES[0])

# Basemap / element palette (matches geo_network_baseline.png)
LAND = "#FFFFFF"
OCEAN = "#D6E4EE"
STATE_EDGE = "#4A5568"
COAST = "#4A5568"
BUS_NEUTRAL = "#5C6B7A"
BUS_LOAD = "#3C5488"
DC_FILL = "#E64B35"
DC_EDGE = "#2B0700"

LOADING_CMAP = LinearSegmentedColormap.from_list(
    "tx_loading",
    ["#4DBBD5", "#91D1C2", "#F39B7F", "#E64B35"],
    N=256,
)
ABS_HI_CMAP = LinearSegmentedColormap.from_list(
    "abs_hi",
    ["#FFE066", "#FDAE61", "#F46D43", "#D73027", "#A50026"],
    N=256,
)
BG_LINE = "#C5C5C5"
DELTA_INC = "#D62728"
DELTA_DEC = "#2CA02C"
LOADING_VMAX = 100.0
ABS_LOADING_MIN = 20.0
DELTA_LOADING_MIN = 5.0

FIG_DPI = 600

# Texas / texas_c: readable at 600 dpi; USA: heavier for national readability
LINE_LW = {"texas_c": 0.55, "texas": 0.55, "usa": 0.50}
LOAD_BUS_SIZE = {"texas_c": (1.2, 4.5), "texas": (1.2, 4.5), "usa": (1.2, 4.5)}
DC_STAR_SIZE = {
    "texas_c": {"hero": 110, "single": 85, "panel": 55, "small": 38},
    "texas": {"hero": 110, "single": 85, "panel": 55, "small": 38},
    "usa": {"hero": 120, "single": 95, "panel": 62, "small": 42},
}

Scale = Literal["texas_c", "texas", "usa"]
PlotStyle = Literal["texas", "usa"]
SizeCtx = Literal["hero", "single", "panel", "small"]

EXTENTS = {
    "texas_c": (-106.55, -93.55, 25.82, 36.58),
    "texas": (-106.55, -93.55, 25.82, 36.58),
    "usa": (-124.8, -66.5, 24.2, 49.8),
}

# Kim et al. NC palette (optional gray style)
BG = "#FFFFFF"
LINE_GRAY = "#B8B8B8"
BUS_BLUE = "#4A86C5"
BUS_BLUE_EDGE = "#2E5F8A"
DC_AMBER = "#E89B0C"
DC_AMBER_EDGE = "#B36B00"
HIGHLIGHT_THRESHOLD_PCT = 80.0
HIGH_LOAD_CMAP = LinearSegmentedColormap.from_list(
    "high_load", ["#FDB462", "#FB8072", "#E64B35", "#B2182B"], N=256
)


def apply_geo_style() -> None:
    """Publication style for geographic loading maps (matches NC Times New Roman)."""
    from experiments.lib.figures.nc_style import apply_figure_style

    apply_figure_style()


apply_nc_style = apply_geo_style  # backward compat


def load_dc_meta(scale: Scale) -> tuple[dict[str, str], dict, dict]:
    from experiments.lib.pypsa.scale_registry import OUT_DIRS

    base = OUT_DIRS[scale]
    bus_map = json.loads((base / "config" / "dc_bus_map.json").read_text(encoding="utf-8"))
    meta_path = base / "config" / "dc_bus_map_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    refs = {
        str(sid): {
            "lat": info.get("reference_lat"),
            "lon": info.get("reference_lon"),
            "name": info.get("name", sid),
        }
        for sid, info in (meta.get("sites") or {}).items()
    }
    return bus_map, meta, refs


def solve_network(n, *, solver: str = "gurobi") -> None:
    from experiments.lib.pypsa.lopf_prep import prepare_for_lopf

    prepare_for_lopf(n)
    n.optimize(solver_name=solver)


def _bus_coords(n) -> pd.DataFrame:
    b = n.buses.copy()
    b["lon"] = b["x"].astype(float)
    b["lat"] = b["y"].astype(float)
    return b


def _peak_bus_load(n) -> pd.Series:
    if not len(n.loads):
        return pd.Series(dtype=float)
    p = n.loads_t.p_set.abs()
    peak = p.max(axis=0)
    return peak.groupby(n.loads["bus"]).sum()


def _edge_df_and_cap(n):
    if len(n.lines):
        return n.lines, "s_nom", "lines_t"
    if len(n.links):
        return n.links, "p_nom", "links_t"
    return None, None, None


def _line_loading_at_snapshot(n, snap_idx: int) -> pd.Series:
    edge_df, cap_col, ts_attr = _edge_df_and_cap(n)
    if edge_df is None:
        return pd.Series(dtype=float)
    cap = edge_df[cap_col].replace(0, np.nan)
    ts = getattr(n, ts_attr)
    if not len(ts) or "p0" not in ts:
        return pd.Series(0.0, index=edge_df.index)
    flow = ts.p0.iloc[snap_idx].abs() / cap * 100.0
    return flow.fillna(0.0)


def hourly_loading_frame(n) -> pd.DataFrame:
    """Line loading (%) for every snapshot in the current window (rows=hours, cols=lines)."""
    edge_df, cap_col, ts_attr = _edge_df_and_cap(n)
    if edge_df is None:
        return pd.DataFrame()
    cap = edge_df[cap_col].replace(0, np.nan)
    ts = getattr(n, ts_attr)
    if not len(ts) or "p0" not in ts:
        return pd.DataFrame(0.0, index=range(len(n.snapshots)), columns=edge_df.index)
    flow = ts.p0.abs().div(cap, axis=1) * 100.0
    out = flow.fillna(0.0)
    out.index = range(len(out))
    return out


def edges_from_loading_series(n, loading: pd.Series) -> pd.DataFrame:
    """Build map edge records from a static network plus one hour of loading (%)."""
    rows: list[dict] = []
    buses = _bus_coords(n)
    edge_df, cap_col, _ = _edge_df_and_cap(n)
    if edge_df is None:
        return pd.DataFrame(rows)
    aligned = loading.reindex(edge_df.index)
    if aligned.isna().all():
        aligned = loading.reindex(edge_df.index.map(str))
    aligned = aligned.fillna(0.0)
    for name, row in edge_df.iterrows():
        b0, b1 = str(row.bus0), str(row.bus1)
        if b0 not in buses.index or b1 not in buses.index:
            continue
        if name in aligned.index:
            loading_pct = float(aligned.loc[name])
        elif str(name) in aligned.index:
            loading_pct = float(aligned.loc[str(name)])
        else:
            loading_pct = 0.0
        rows.append(
            {
                "line": str(name),
                "lon0": buses.loc[b0, "lon"],
                "lat0": buses.loc[b0, "lat"],
                "lon1": buses.loc[b1, "lon"],
                "lat1": buses.loc[b1, "lat"],
                "loading_pct": loading_pct,
                "capacity_mw": float(row[cap_col]),
            }
        )
    return pd.DataFrame(rows)


def critical_snapshot_index(n) -> int:
    """Snapshot index where the network-wide maximum line loading occurs."""
    if not len(n.snapshots):
        return 0
    best_idx, best_peak = 0, -1.0
    for i in range(len(n.snapshots)):
        loading = _line_loading_at_snapshot(n, i)
        peak = float(loading.max()) if len(loading) else 0.0
        if peak > best_peak:
            best_peak = peak
            best_idx = i
    return best_idx


def edge_loading_at_snapshot(n, snap_idx: int) -> pd.DataFrame:
    rows: list[dict] = []
    buses = _bus_coords(n)
    edge_df, cap_col, _ = _edge_df_and_cap(n)
    if edge_df is None:
        return pd.DataFrame(rows)
    flow = _line_loading_at_snapshot(n, snap_idx)
    for name, row in edge_df.iterrows():
        b0, b1 = str(row.bus0), str(row.bus1)
        if b0 not in buses.index or b1 not in buses.index:
            continue
        rows.append(
            {
                "line": str(name),
                "lon0": buses.loc[b0, "lon"],
                "lat0": buses.loc[b0, "lat"],
                "lon1": buses.loc[b1, "lon"],
                "lat1": buses.loc[b1, "lat"],
                "loading_pct": float(flow.get(name, 0.0)),
                "capacity_mw": float(row[cap_col]),
            }
        )
    return pd.DataFrame(rows)


def edge_critical_hour_loading(n) -> tuple[int, pd.DataFrame]:
    """Line loadings at the single most congested hour in the snapshot window."""
    idx = critical_snapshot_index(n)
    return idx, edge_loading_at_snapshot(n, idx)


def edge_peak_loading(n, *, per_line_peak: bool = False) -> pd.DataFrame:
    """Default: all branches at the network-critical hour (not 24h mean, not per-line peak mix)."""
    if not per_line_peak:
        _, edges = edge_critical_hour_loading(n)
        return edges

    rows: list[dict] = []
    buses = _bus_coords(n)
    edge_df, cap_col, ts_attr = _edge_df_and_cap(n)
    if edge_df is None:
        return pd.DataFrame(rows)
    cap = edge_df[cap_col].replace(0, np.nan)
    ts = getattr(n, ts_attr)
    flow = (ts.p0.abs().max(axis=0) / cap * 100.0).fillna(0.0) if len(ts) and "p0" in ts else pd.Series(0.0, index=edge_df.index)
    for name, row in edge_df.iterrows():
        b0, b1 = str(row.bus0), str(row.bus1)
        if b0 not in buses.index or b1 not in buses.index:
            continue
        rows.append(
            {
                "lon0": buses.loc[b0, "lon"],
                "lat0": buses.loc[b0, "lat"],
                "lon1": buses.loc[b1, "lon"],
                "lat1": buses.loc[b1, "lat"],
                "loading_pct": float(flow.get(name, 0.0)),
                "capacity_mw": float(row[cap_col]),
            }
        )
    return pd.DataFrame(rows)


def _geodata_paths(scale: Scale) -> tuple[Path, Path | None]:
    # Geospatial bundles live under interconnect folders (texas / usa), not texas_c.
    geo_key = "texas" if scale in ("texas", "texas_c") else scale
    geo = WF / geo_key / "Geospatial"
    states = geo / "state_boundaries.geojson"
    land = geo / ("country_shapes.geojson" if geo_key == "usa" else "onshore_shapes.geojson")
    return states, land if land.is_file() else None


def panel_aspect(scale: Scale) -> float:
    lon0, lon1, lat0, lat1 = EXTENTS[scale]
    mid_lat = 0.5 * (lat0 + lat1)
    return (lon1 - lon0) / max(lat1 - lat0, 1e-6) * np.cos(np.radians(mid_lat))


def _edges_to_arrays(edges) -> tuple[list, np.ndarray]:
    if isinstance(edges, pd.DataFrame):
        if edges.empty:
            return [], np.array([], dtype=float)
        segments = [[(r.lon0, r.lat0), (r.lon1, r.lat1)] for r in edges.itertuples(index=False)]
        loadings = edges["loading_pct"].to_numpy(float)
        return segments, loadings
    if not edges:
        return [], np.array([], dtype=float)
    segments = [[(e["lon0"], e["lat0"]), (e["lon1"], e["lat1"])] for e in edges]
    loadings = np.array([e["loading_pct"] for e in edges], dtype=float)
    return segments, loadings


def draw_basemap_geo(
    ax,
    scale: Scale,
    *,
    left: bool = False,
    bottom: bool = False,
    right: bool = False,
    top: bool = False,
) -> None:
    """Ocean + land + state outlines + lat/lon grid."""
    import cartopy.crs as ccrs
    import geopandas as gpd

    ax.set_extent(EXTENTS[scale], crs=ccrs.PlateCarree())
    ax.set_facecolor(OCEAN)

    states_path, land_path = _geodata_paths(scale)
    if scale in ("texas", "texas_c") and states_path.is_file():
        # Political Texas, not ERCOT-onshore. El Paso / east-TX sites sit outside
        # the interconnect land polygon (west bound ~-105°) but inside the state.
        states = gpd.read_file(states_path)
        tx = states
        if "name" in states.columns:
            filt = states[states["name"].str.contains("Texas", case=False, na=False)]
            if len(filt):
                tx = filt
        tx.plot(
            ax=ax, facecolor=LAND, edgecolor=STATE_EDGE, linewidth=0.9,
            transform=ccrs.PlateCarree(), zorder=1,
        )
        tx.plot(
            ax=ax, facecolor="none", edgecolor=STATE_EDGE, linewidth=0.9,
            transform=ccrs.PlateCarree(), zorder=2,
        )
    else:
        if land_path:
            gpd.read_file(land_path).plot(
                ax=ax, facecolor=LAND, edgecolor=COAST, linewidth=0.35,
                transform=ccrs.PlateCarree(), zorder=1,
            )
        if states_path.is_file():
            states = gpd.read_file(states_path)
            if scale in ("texas", "texas_c") and "name" in states.columns:
                tx = states[states["name"].str.contains("Texas", case=False, na=False)]
                if len(tx):
                    states = tx
            states.plot(
                ax=ax, facecolor="none", edgecolor=STATE_EDGE, linewidth=0.5,
                transform=ccrs.PlateCarree(), zorder=2,
            )

    add_geo_gridlabels(ax, scale, left=left, bottom=bottom, right=right, top=top)


_draw_basemap_nc = draw_basemap_geo  # backward compat


def add_geo_gridlabels(
    ax,
    scale: Scale,
    *,
    left: bool = False,
    bottom: bool = False,
    right: bool = False,
    top: bool = False,
) -> None:
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    any_labels = left or bottom or right or top
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=any_labels,
        linewidth=0.3,
        color="#BBBBBB",
        alpha=0.75,
        linestyle="--",
    )
    gl.top_labels = top
    gl.right_labels = right
    gl.bottom_labels = bottom
    gl.left_labels = left
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 8, "color": "#444444"}
    gl.ylabel_style = {"size": 8, "color": "#444444"}


def network_peak_loading_timeseries(n) -> pd.Series:
    """Max line loading (%) at each snapshot in the current window."""
    if not len(n.snapshots):
        return pd.Series(dtype=float)
    peaks = []
    for i in range(len(n.snapshots)):
        loading = _line_loading_at_snapshot(n, i)
        peaks.append(float(loading.max()) if len(loading) else 0.0)
    return pd.Series(peaks)


def select_impact_map_snapshot_index(reference_n, portfolio_networks: list) -> tuple[int, float]:
    """
    Pick the hour where DC flexibility impact vs the reference (FCFS) is largest.

    score(t) = max(0, max_pf_peak(t) - fcfs_peak(t)) + portfolio_spread(t)
    """
    ref = network_peak_loading_timeseries(reference_n)
    pf_ts = [network_peak_loading_timeseries(n) for n in portfolio_networks]
    if not len(ref):
        return 0, 0.0
    best_idx, best_score = 0, -1.0
    for i in range(len(ref)):
        fcfs = float(ref.iloc[i])
        pf_peaks = [float(ts.iloc[i]) for ts in pf_ts] if pf_ts else []
        max_pf = max(pf_peaks) if pf_peaks else fcfs
        min_pf = min(pf_peaks) if pf_peaks else fcfs
        uplift = max(max_pf - fcfs, 0.0)
        spread = max_pf - min_pf if len(pf_peaks) > 1 else 0.0
        score = uplift + spread
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


def delta_annotation_counts(
    fcfs_edges,
    portfolio_edges,
    *,
    threshold: float = DELTA_LOADING_MIN,
) -> tuple[int, int]:
    """Count |Δ|>=threshold branches vs FCFS at one hour (inc, dec)."""
    ref = {
        edge_coord_key(e): float(e["loading_pct"] if isinstance(e, dict) else e.loading_pct)
        for e in _as_edge_list(fcfs_edges)
    }
    n_inc = n_dec = 0
    for e in _as_edge_list(portfolio_edges):
        loading = float(e["loading_pct"] if isinstance(e, dict) else e.loading_pct)
        delta = loading - ref.get(edge_coord_key(e), 0.0)
        if abs(delta) >= threshold:
            if delta > 0:
                n_inc += 1
            else:
                n_dec += 1
    return n_inc, n_dec


def annotation_score_at_snapshot(
    fcfs_n,
    portfolio_networks: list,
    snap_idx: int,
    *,
    threshold: float = DELTA_LOADING_MIN,
) -> tuple[int, int, int]:
    """Total Δ labels (inc+dec) summed over non-FCFS portfolios at one hour."""
    fcfs_edges = edge_loading_at_snapshot(fcfs_n, snap_idx)
    total_inc = total_dec = 0
    for n in portfolio_networks:
        edges = edge_loading_at_snapshot(n, snap_idx)
        n_inc, n_dec = delta_annotation_counts(fcfs_edges, edges, threshold=threshold)
        total_inc += n_inc
        total_dec += n_dec
    return total_inc + total_dec, total_inc, total_dec


def select_max_annotation_map_snapshot_index(
    fcfs_n,
    portfolio_networks: list,
    *,
    threshold: float = DELTA_LOADING_MIN,
) -> tuple[int, int, int, int]:
    """Pick hour maximizing total Δ labels vs FCFS across all portfolios."""
    if not len(fcfs_n.snapshots):
        return 0, 0, 0, 0
    best_idx, best_total, best_inc, best_dec = 0, -1, 0, 0
    for i in range(len(fcfs_n.snapshots)):
        total, n_inc, n_dec = annotation_score_at_snapshot(
            fcfs_n, portfolio_networks, i, threshold=threshold,
        )
        if total > best_total:
            best_idx, best_total, best_inc, best_dec = i, total, n_inc, n_dec
    return best_idx, best_total, best_inc, best_dec


def edge_coord_key(edge) -> tuple[float, float, float, float]:
    if isinstance(edge, dict):
        vals = (edge["lon0"], edge["lat0"], edge["lon1"], edge["lat1"])
    else:
        vals = (edge.lon0, edge.lat0, edge.lon1, edge.lat1)
    return tuple(round(float(v), 4) for v in vals)


def _emphasis_delta_threshold(edges, reference_edges, *, floor: float = 1.0) -> float:
    ref = {edge_coord_key(e): float(e["loading_pct"] if isinstance(e, dict) else e.loading_pct) for e in reference_edges}
    deltas = []
    for e in edges:
        key = edge_coord_key(e)
        loading = float(e["loading_pct"] if isinstance(e, dict) else e.loading_pct)
        deltas.append(abs(loading - ref.get(key, 0.0)))
    if not deltas:
        return floor
    return max(floor, float(np.percentile(deltas, 70)))


def draw_loading_edges(
    ax,
    edges,
    *,
    scale: Scale,
    reference_edges=None,
    emphasize_changes: bool = False,
    delta_threshold: float | None = None,
) -> LineCollection | None:
    """Draw line loadings; optionally de-emphasize unchanged branches."""
    import cartopy.crs as ccrs

    if isinstance(edges, pd.DataFrame):
        edge_list = edges.to_dict(orient="records")
    else:
        edge_list = list(edges)

    if not edge_list:
        return None

    if not emphasize_changes:
        segments, loadings = _edges_to_arrays(edge_list)
        lc = LineCollection(
            segments,
            cmap=LOADING_CMAP,
            norm=Normalize(0, LOADING_VMAX),
            linewidths=LINE_LW[scale],
            alpha=0.92,
            capstyle="round",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        lc.set_array(loadings)
        ax.add_collection(lc)
        return lc

    ref = None
    if reference_edges is not None:
        ref = {edge_coord_key(e): float(e["loading_pct"] if isinstance(e, dict) else e.loading_pct) for e in reference_edges}
    thr = delta_threshold
    if thr is None and ref is not None:
        thr = _emphasis_delta_threshold(edge_list, reference_edges)

    bg_segments: list = []
    bg_loadings: list[float] = []
    hi_segments: list = []
    hi_loadings: list[float] = []
    base_lw = LINE_LW[scale]
    for e in edge_list:
        seg = [(e["lon0"], e["lat0"]), (e["lon1"], e["lat1"])]
        loading = float(e["loading_pct"])
        ref_loading = ref.get(edge_coord_key(e), 0.0) if ref is not None else 0.0
        delta = abs(loading - ref_loading)
        highlight = (ref is not None and delta >= thr) or (ref is None and loading >= 40.0)
        if highlight:
            hi_segments.append(seg)
            hi_loadings.append(loading)
        else:
            bg_segments.append(seg)
            bg_loadings.append(loading)

    lc_ref = None
    if bg_segments:
        lc_bg = LineCollection(
            bg_segments,
            cmap=LOADING_CMAP,
            norm=Normalize(0, LOADING_VMAX),
            linewidths=base_lw * 0.55,
            alpha=0.72,
            capstyle="round",
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        lc_bg.set_array(np.array(bg_loadings, dtype=float))
        ax.add_collection(lc_bg)
        lc_ref = lc_bg

    if hi_segments:
        lc = LineCollection(
            hi_segments,
            cmap=LOADING_CMAP,
            norm=Normalize(0, LOADING_VMAX),
            linewidths=base_lw * 1.45,
            alpha=0.98,
            capstyle="round",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
        lc.set_array(np.array(hi_loadings, dtype=float))
        ax.add_collection(lc)
        lc_ref = lc

    return lc_ref


def _as_edge_list(edges) -> list[dict]:
    if isinstance(edges, pd.DataFrame):
        return edges.to_dict(orient="records")
    return list(edges)


def _abs_loading_callout_xy(seg, *, scale: Scale) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (line midpoint, offset label position) for abs-loading callouts."""
    mx = 0.5 * (seg[0][0] + seg[1][0])
    my = 0.5 * (seg[0][1] + seg[1][1])
    seg_dx = seg[1][0] - seg[0][0]
    seg_dy = seg[1][1] - seg[0][1]
    px, py = -seg_dy, seg_dx
    norm = float(np.hypot(px, py)) or 1.0
    px /= norm
    py /= norm
    # Prefer labels toward open map margin (SW for Texas panels).
    if scale in ("texas", "texas_c") and (px > 0 or py > 0):
        px, py = -px, -py
    elif scale == "usa" and py < 0:
        px, py = -px, -py
    offset = 0.55 if scale in ("texas", "texas_c") else 0.85
    return (mx, my), (mx + offset * px, my + offset * py)


def _annotate_abs_loading_callouts(
    ax,
    hi_segments: list,
    hi_loadings: list[float],
    *,
    scale: Scale,
    label_fontsize: float = 5.0,
) -> None:
    """Leader-line callouts for highlighted absolute line loadings."""
    import cartopy.crs as ccrs

    transform = ccrs.PlateCarree()
    for seg, loading in zip(hi_segments, hi_loadings):
        anchor, text_xy = _abs_loading_callout_xy(seg, scale=scale)
        ax.annotate(
            f"{loading:.0f}%",
            xy=anchor,
            xytext=text_xy,
            fontsize=label_fontsize,
            fontweight="bold",
            color="#A50026",
            ha="center",
            va="center",
            transform=transform,
            zorder=8,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="#DDDDDD",
                alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#888888",
                lw=0.7,
                shrinkA=2,
                shrinkB=2,
            ),
        )


def draw_abs_loading_threshold(
    ax,
    edges,
    *,
    scale: Scale,
    threshold: float = ABS_LOADING_MIN,
    label_fontsize: float | None = 5.0,
    common_edge_keys: set[tuple[float, float, float, float]] | None = None,
    common_lw_scale: float = 0.5,
) -> LineCollection | None:
    """Absolute line loading at map hour: grey background, >=threshold in yellow-red.

    If ``common_edge_keys`` is set, highlighted edges in that set are drawn at
    ``common_lw_scale`` × the normal highlight width (Fig3: de-emphasize lines
    whose congestion status matches across No-DC / FCFS / Full flex).
    """
    import cartopy.crs as ccrs

    edge_list = _as_edge_list(edges)
    if not edge_list:
        return None

    bg_segments: list = []
    hi_segments: list = []
    hi_loadings: list[float] = []
    hi_linewidths: list[float] = []
    base_lw = LINE_LW[scale]
    hi_lw = base_lw * 1.55
    shared = common_edge_keys or set()
    for e in edge_list:
        seg = [(e["lon0"], e["lat0"]), (e["lon1"], e["lat1"])]
        loading = float(e["loading_pct"])
        if loading >= threshold:
            hi_segments.append(seg)
            hi_loadings.append(loading)
            lw = hi_lw * common_lw_scale if edge_coord_key(e) in shared else hi_lw
            hi_linewidths.append(lw)
        else:
            bg_segments.append(seg)

    if bg_segments:
        ax.add_collection(
            LineCollection(
                bg_segments,
                colors=BG_LINE,
                linewidths=base_lw * 0.70,
                alpha=0.55,
                capstyle="round",
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
        )

    if not hi_segments:
        return None

    lc = LineCollection(
        hi_segments,
        cmap=ABS_HI_CMAP,
        norm=Normalize(threshold, LOADING_VMAX),
        linewidths=hi_linewidths,
        alpha=0.98,
        capstyle="round",
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    lc.set_array(np.array(hi_loadings, dtype=float))
    ax.add_collection(lc)
    if label_fontsize is not None:
        _annotate_abs_loading_callouts(
            ax,
            hi_segments,
            hi_loadings,
            scale=scale,
            label_fontsize=label_fontsize,
        )
    return lc


def draw_delta_loading_threshold(
    ax,
    edges,
    reference_edges,
    *,
    scale: Scale,
    threshold: float = DELTA_LOADING_MIN,
    label_threshold: float | None = None,
    label_fontsize: float = 5.0,
) -> tuple[int, int]:
    """Δloading vs reference: color |Δ|>=threshold; text only if |Δ|>=label_threshold."""
    import cartopy.crs as ccrs

    edge_list = _as_edge_list(edges)
    ref = {
        edge_coord_key(e): float(e["loading_pct"] if isinstance(e, dict) else e.loading_pct)
        for e in reference_edges
    }
    if not edge_list:
        return 0, 0

    bg_segments: list = []
    inc_segments: list = []
    dec_segments: list = []
    base_lw = LINE_LW[scale]
    n_inc = n_dec = 0
    label_thr = threshold if label_threshold is None else float(label_threshold)

    for e in edge_list:
        seg = [(e["lon0"], e["lat0"]), (e["lon1"], e["lat1"])]
        loading = float(e["loading_pct"])
        delta = loading - ref.get(edge_coord_key(e), 0.0)
        if abs(delta) >= threshold:
            if delta > 0:
                inc_segments.append(seg)
                n_inc += 1
            else:
                dec_segments.append(seg)
                n_dec += 1
            if abs(delta) >= label_thr:
                mx = 0.5 * (seg[0][0] + seg[1][0])
                my = 0.5 * (seg[0][1] + seg[1][1])
                sign = "+" if delta > 0 else ""
                color = DELTA_INC if delta > 0 else DELTA_DEC
                ax.text(
                    mx, my, f"{sign}{delta:.0f}%",
                    fontsize=label_fontsize,
                    fontweight="bold",
                    color=color,
                    ha="center",
                    va="center",
                    transform=ccrs.PlateCarree(),
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#DDDDDD", alpha=0.92),
                )
        else:
            bg_segments.append(seg)

    if bg_segments:
        ax.add_collection(
            LineCollection(
                bg_segments,
                colors=BG_LINE,
                linewidths=base_lw * 0.70,
                alpha=0.55,
                capstyle="round",
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
        )
    if inc_segments:
        ax.add_collection(
            LineCollection(
                inc_segments,
                colors=DELTA_INC,
                linewidths=base_lw * 1.55,
                alpha=0.95,
                capstyle="round",
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
        )
    if dec_segments:
        ax.add_collection(
            LineCollection(
                dec_segments,
                colors=DELTA_DEC,
                linewidths=base_lw * 1.55,
                alpha=0.95,
                capstyle="round",
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
        )
    return n_inc, n_dec


def draw_geo_panel_abs_loading(
    ax,
    edges,
    n,
    *,
    scale: Scale,
    dc_map: dict[str, str],
    dc_meta: dict | None = None,
    dc_refs: dict | None = None,
    left: bool = False,
    bottom: bool = False,
    right: bool = False,
    top: bool = False,
    dc_size_ctx: SizeCtx = "panel",
    loading_threshold: float = ABS_LOADING_MIN,
    label_fontsize: float | None = 5.0,
    dc_coords: Literal["bus", "reference"] = "bus",
    site_colors: dict[str, str] | None = None,
    common_edge_keys: set[tuple[float, float, float, float]] | None = None,
    common_lw_scale: float = 0.5,
) -> LineCollection | None:
    draw_basemap_geo(ax, scale, left=left, bottom=bottom, right=right, top=top)
    lc = draw_abs_loading_threshold(
        ax,
        edges,
        scale=scale,
        threshold=loading_threshold,
        label_fontsize=label_fontsize,
        common_edge_keys=common_edge_keys,
        common_lw_scale=common_lw_scale,
    )
    draw_load_buses(ax, n, dc_buses=set(dc_map.values()), scale=scale)
    draw_dc_sites(
        ax, dc_map, scale=scale, n=n, dc_meta=dc_meta, dc_refs=dc_refs,
        size_ctx=dc_size_ctx, show_labels=False, coords=dc_coords,
        site_colors=site_colors,
    )
    return lc


def draw_geo_panel_delta_loading(
    ax,
    edges,
    reference_edges,
    n,
    *,
    scale: Scale,
    dc_map: dict[str, str],
    dc_meta: dict | None = None,
    dc_refs: dict | None = None,
    left: bool = False,
    bottom: bool = False,
    right: bool = False,
    top: bool = False,
    dc_size_ctx: SizeCtx = "panel",
    delta_threshold: float = DELTA_LOADING_MIN,
    label_threshold: float | None = None,
    label_fontsize: float = 5.0,
    dc_coords: Literal["bus", "reference"] = "bus",
    site_colors: dict[str, str] | None = None,
) -> tuple[int, int]:
    draw_basemap_geo(ax, scale, left=left, bottom=bottom, right=right, top=top)
    counts = draw_delta_loading_threshold(
        ax, edges, reference_edges, scale=scale,
        threshold=delta_threshold, label_threshold=label_threshold,
        label_fontsize=label_fontsize,
    )
    draw_load_buses(ax, n, dc_buses=set(dc_map.values()), scale=scale)
    draw_dc_sites(
        ax, dc_map, scale=scale, n=n, dc_meta=dc_meta, dc_refs=dc_refs,
        size_ctx=dc_size_ctx, show_labels=False, coords=dc_coords,
        site_colors=site_colors,
    )
    return counts


def draw_load_buses(
    ax,
    n,
    *,
    dc_buses: set[str] | None = None,
    scale: Scale,
) -> None:
    """Small load-bus markers (transmission-connected only); DC buses omitted here."""
    import cartopy.crs as ccrs

    dc_buses = dc_buses or set()
    buses = _bus_coords(n)
    peak_load = _peak_bus_load(n)
    pmax = float(peak_load.max()) if len(peak_load) else 1.0
    q75 = float(peak_load.quantile(0.75)) if len(peak_load) else 0.0
    smin, smax = LOAD_BUS_SIZE[scale]

    if len(n.lines):
        tx_buses = set(n.lines["bus0"]) | set(n.lines["bus1"])
        sub = buses.loc[buses.index.isin(tx_buses)]
    elif len(n.links):
        tx_buses = set(n.links["bus0"]) | set(n.links["bus1"])
        sub = buses.loc[buses.index.isin(tx_buses)]
    else:
        sub = buses

    regular = sub[~sub.index.isin(dc_buses)]
    if not len(regular):
        return

    sizes, colors = [], []
    for b in regular.index:
        mw = float(peak_load.get(b, 0.0))
        rel = np.sqrt(mw / max(pmax, 1e-9)) if mw > 0 else 0.0
        sizes.append(smin + (smax - smin) * rel)
        colors.append(BUS_LOAD if mw > q75 else BUS_NEUTRAL)

    ax.scatter(
        regular["lon"], regular["lat"],
        s=sizes, c=colors, alpha=0.82,
        edgecolors="none",
        transform=ccrs.PlateCarree(), zorder=4,
    )


def draw_dc_sites(
    ax,
    dc_map: dict[str, str],
    *,
    scale: Scale,
    n=None,
    dc_meta: dict | None = None,
    dc_refs: dict | None = None,
    marker_size: float | None = None,
    size_ctx: SizeCtx = "single",
    show_labels: bool = True,
    coords: Literal["bus", "reference"] = "bus",
    site_colors: dict[str, str] | None = None,
) -> None:
    """Stars at DC sites (injection bus or facility reference coordinates)."""
    import cartopy.crs as ccrs

    from experiments.lib.pypsa.scale_registry import OUT_DIRS

    meta_path = OUT_DIRS[scale] / "config" / "dc_bus_map_meta.json"
    meta = dc_meta or (json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {})
    sites = meta.get("sites", {})
    refs = dc_refs or {}

    lons, lats, labels, colors = [], [], [], []
    for site_id, _bus in dc_map.items():
        sm = sites.get(str(site_id), {})
        ref = refs.get(str(site_id), refs.get(site_id, {}))
        lon, lat = None, None
        if coords == "reference":
            lat = sm.get("reference_lat") or ref.get("lat")
            lon = sm.get("reference_lon") or ref.get("lon")
        if lon is None or lat is None:
            lon = sm.get("bus_lon")
            lat = sm.get("bus_lat")
        if lon is None or lat is None:
            if n is not None and _bus in _bus_coords(n).index:
                lon = float(_bus_coords(n).loc[_bus, "lon"])
                lat = float(_bus_coords(n).loc[_bus, "lat"])
            else:
                continue
        lons.append(float(lon))
        lats.append(float(lat))
        city = ref.get("reference_city", sm.get("reference_city", f"Site {site_id}"))
        labels.append(f"{city}\n({site_id})")
        colors.append((site_colors or {}).get(str(site_id), DC_FILL))

    if not lons:
        return

    s = marker_size if marker_size is not None else DC_STAR_SIZE[scale][size_ctx]
    ax.scatter(
        lons, lats, s=s, marker="*", c=colors, edgecolors=DC_EDGE, linewidths=0.5,
        transform=ccrs.PlateCarree(), zorder=6,
    )

    if show_labels and size_ctx in ("hero", "single"):
        for lon, lat, label in zip(lons, lats, labels):
            txt = ax.text(
                lon, lat, label, fontsize=7.5, fontweight="bold", color=DC_EDGE,
                ha="left", va="bottom", transform=ccrs.PlateCarree(), zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.88),
            )
            txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def geo_map_legend(ax, lc: LineCollection | None, *, loc: str = "lower left") -> None:
    handles = [
        Line2D([0], [0], color=LOADING_CMAP(0.25), lw=1.8, label="Transmission element"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BUS_LOAD, markersize=4.5, label="Load bus"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=DC_FILL, markeredgecolor=DC_EDGE,
               markersize=11, label="DC injection site"),
    ]
    ax.legend(
        handles=handles, loc=loc, frameon=True, framealpha=0.92,
        edgecolor="#DDDDDD", fontsize=8, handletextpad=0.45, borderpad=0.35,
    )


def draw_geo_panel(
    ax,
    edges,
    n,
    *,
    scale: Scale,
    dc_map: dict[str, str],
    dc_meta: dict | None = None,
    dc_refs: dict | None = None,
    left: bool = False,
    bottom: bool = False,
    right: bool = False,
    top: bool = False,
    dc_size_ctx: SizeCtx = "panel",
    show_dc_labels: bool = False,
    show_legend: bool = False,
    reference_edges=None,
    emphasize_changes: bool = False,
    delta_threshold: float | None = None,
) -> LineCollection | None:
    draw_basemap_geo(ax, scale, left=left, bottom=bottom, right=right, top=top)
    lc = draw_loading_edges(
        ax, edges, scale=scale, reference_edges=reference_edges,
        emphasize_changes=emphasize_changes, delta_threshold=delta_threshold,
    )
    draw_load_buses(ax, n, dc_buses=set(dc_map.values()), scale=scale)
    draw_dc_sites(
        ax, dc_map, scale=scale, n=n, dc_meta=dc_meta, dc_refs=dc_refs,
        size_ctx=dc_size_ctx, show_labels=show_dc_labels,
    )
    if show_legend and lc is not None:
        geo_map_legend(ax, lc)
    return lc


def plot_geo_map(
    edges,
    n,
    *,
    scale: Scale,
    title: str,
    subtitle: str,
    dc_bus_map: dict[str, str],
    dc_meta: dict,
    dc_refs: dict,
    out_path: Path | str,
    figsize: tuple[float, float] | None = None,
) -> Path:
    import cartopy.crs as ccrs

    apply_geo_style()
    if figsize is None:
        figsize = (9.2, 8.6) if scale in ("texas", "texas_c") else (11.0, 6.8)
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lc = draw_geo_panel(
        ax, edges, n, scale=scale, dc_map=dc_bus_map, dc_meta=dc_meta, dc_refs=dc_refs,
        left=True, bottom=True, show_dc_labels=True, show_legend=True, dc_size_ctx="single",
    )
    if lc is not None:
        cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02, shrink=0.78)
        cbar.set_label("Peak transmission loading (%)", fontsize=9)
        cbar.set_ticks([0, 25, 50, 75, 100])
        cbar.ax.tick_params(labelsize=8)

    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold", pad=10)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_geo_comparison(
    panels: list[tuple],
    *,
    scale: Scale,
    suptitle: str,
    out_path: Path | str,
) -> Path:
    """panels: [(edges, n, subtitle, dc_map), ...]"""
    import cartopy.crs as ccrs

    apply_geo_style()
    ncols = len(panels)
    pw = 4.8 if scale in ("texas", "texas_c") else 4.2
    fig = plt.figure(figsize=(pw * ncols + 0.6, 6.4 if scale in ("texas", "texas_c") else 5.6), facecolor="white")

    lc_ref = None
    dc_meta = panels[0][4] if len(panels[0]) > 4 else {}
    dc_refs = panels[0][5] if len(panels[0]) > 5 else {}

    for i, item in enumerate(panels):
        edges, n, subtitle, dc_map = item[:4]
        ax = fig.add_subplot(1, ncols, i + 1, projection=ccrs.PlateCarree())
        lc = draw_geo_panel(
            ax, edges, n, scale=scale, dc_map=dc_map, dc_meta=dc_meta, dc_refs=dc_refs,
            left=(i == 0), bottom=True, dc_size_ctx="panel", show_dc_labels=False,
            show_legend=(i == 0),
        )
        if lc is not None:
            lc_ref = lc
        ax.set_title(subtitle, fontsize=10, fontweight="bold", pad=6)
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    if lc_ref is not None:
        cbar = fig.colorbar(lc_ref, ax=fig.axes, fraction=0.025, pad=0.02, shrink=0.82)
        cbar.set_label("Peak transmission loading (%)", fontsize=9)
        cbar.set_ticks([0, 25, 50, 75, 100])
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def network_stats(n, edges) -> str:
    seg, _ = _edges_to_arrays(edges)
    peak = float(np.nanmax(_edges_to_arrays(edges)[1])) if seg else 0.0
    return f"Peak loading: {peak:.1f}%  ·  {len(n.buses)} buses  ·  {len(seg)} branches"


def load_static_network(scale: Scale):
    import pypsa
    from experiments.lib.pypsa.scale_registry import NETWORKS

    return pypsa.Network(str(NETWORKS[scale]))


# ── Optional NC gray style (legacy) ───────────────────────────────────────

def _draw_transmission_nc(ax, edges: pd.DataFrame, *, scale: Scale) -> None:
    import cartopy.crs as ccrs

    if isinstance(edges, list):
        edges = pd.DataFrame(edges)
    if edges.empty:
        return
    segments = [[(r.lon0, r.lat0), (r.lon1, r.lat1)] for r in edges.itertuples(index=False)]
    lw = LINE_LW[scale]
    lc = LineCollection(
        segments, colors=LINE_GRAY, linewidths=lw, alpha=0.85,
        capstyle="round", transform=ccrs.PlateCarree(), zorder=2,
    )
    ax.add_collection(lc)


def plot_nc_single(
    n, *, scale: Scale, panel_title: str, dc_bus_map: dict[str, str],
    out_path: Path | str, panel_label: str | None = None,
) -> Path:
    edges = edge_peak_loading(n)
    subtitle = network_stats(n, edges)
    dc_map, dc_meta, dc_refs = load_dc_meta(scale)
    return plot_geo_map(
        edges, n, scale=scale, title=panel_title, subtitle=subtitle,
        dc_bus_map=dc_bus_map or dc_map, dc_meta=dc_meta, dc_refs=dc_refs, out_path=out_path,
    )


def plot_nc_comparison(
    panels: list[tuple],
    *,
    scale: Scale,
    out_path: Path | str,
    panel_labels: list[str] | None = None,
) -> Path:
    dc_map, dc_meta, dc_refs = load_dc_meta(scale)
    from experiments.lib.pypsa.scale_registry import SIZE_TAG

    size_tag = SIZE_TAG.get(scale, scale)
    geo_panels = []
    for i, (n, subtitle, dc_m) in enumerate(panels):
        edges = edge_peak_loading(n)
        geo_panels.append((edges, n, subtitle, dc_m or dc_map, dc_meta, dc_refs))
    return plot_geo_comparison(
        geo_panels, scale=scale,
        suptitle=f"PyPSA-USA {size_tag}-bus network comparison",
        out_path=out_path,
    )


def plot_loading_single(
    n, *, scale: Scale, title: str, dc_bus_map: dict[str, str],
    dc_meta: dict, dc_refs: dict, out_path: Path | str,
) -> Path:
    edges = edge_peak_loading(n)
    return plot_geo_map(
        edges, n, scale=scale, title=title, subtitle=network_stats(n, edges),
        dc_bus_map=dc_bus_map, dc_meta=dc_meta, dc_refs=dc_refs, out_path=out_path,
    )
