"""Shared PyPSA geographic line-loading map helpers (E02/E03)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from experiments.lib.pypsa.geo_network_plot import (
    ABS_LOADING_MIN,
    DELTA_DEC,
    DELTA_INC,
    EXTENTS,
    apply_geo_style,
    draw_geo_panel_abs_loading,
    draw_geo_panel_delta_loading,
    load_static_network,
    panel_aspect,
)
from experiments.lib.figures.nc_style import (
    OPERATOR_COLORS,
    OPERATOR_LABELS,
    OPERATOR_ORDER,
    apply_nature_comm_style,
    save_figure,
)

SCALE = "texas_c"
MAP_DC_SIZE_CTX = "small"
MAP_DC_COORDS = "reference"

# Dense s1250 maps: color |Δ|≥50 pp, text |Δ|≥80 pp; abs highlight ≥80%, no callouts.
S1250_DELTA_COLOR_PP = 50.0
S1250_DELTA_LABEL_PP = 80.0
S1250_ABS_HIGHLIGHT_PCT = 80.0


def annotation_style_for(scale: str) -> tuple[float, float, float, float | None]:
    """Return (delta_color_pp, delta_label_pp, abs_highlight_pct, abs_label_fontsize)."""
    if scale == "texas":
        return S1250_DELTA_COLOR_PP, S1250_DELTA_LABEL_PP, S1250_ABS_HIGHLIGHT_PCT, None
    return 5.0, 5.0, ABS_LOADING_MIN, 5.0


def plot_scale_for(e02_root: Path) -> str:
    cfg_path = e02_root / "config" / "pypsa_texas7.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return str(cfg.get("plot_scale") or SCALE)
    return SCALE


def load_e02_geo_context(e02_root: Path) -> tuple[object, dict, dict, dict, dict[str, str]]:
    meta_path = e02_root / "config" / "dc_bus_map_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"sites": {}}
    dc_map = json.loads((e02_root / "config" / "dc_bus_map.json").read_text(encoding="utf-8"))
    accounting_path = e02_root / "config" / "dc_load_accounting.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8")) if accounting_path.exists() else {}
    dc_refs = {
        sid: {
            "lat": info["reference_lat"],
            "lon": info["reference_lon"],
            "name": info.get("name", sid),
        }
        for sid, info in meta.get("sites", {}).items()
    }
    site_colors: dict[str, str] = {}
    for sid, info in (accounting.get("sites") or {}).items():
        op = str(info.get("operator") or "").lower()
        site_colors[str(sid)] = OPERATOR_COLORS.get(op, "#E64B35")
    n = load_static_network(plot_scale_for(e02_root))
    return n, dc_map, meta, dc_refs, site_colors


def backfill_congestion_metrics(portfolios: dict) -> None:
    """Count lines at thermal limit from stored edges (critical-hour snapshot)."""
    for port in portfolios.values():
        edges = port.get("edges") or []
        if not edges:
            continue
        loadings = [float(e.get("loading_pct") or 0.0) for e in edges]
        port["n_overloaded"] = int(sum(v >= 99.9 - 1e-9 for v in loadings))
        port["n_lines_at_or_above_90pct"] = int(sum(v >= 90.0 for v in loadings))
        if port.get("peak_loading_pct") is None and loadings:
            port["peak_loading_pct"] = round(max(loadings), 4)


def write_grid_health_summary(portfolios: dict, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "peak_loading_pct",
        "n_overloaded",
        "n_lines_at_or_above_90pct",
        "voltage_violation_count",
        "max_voltage_deviation_pu",
        "pf_converged",
        "voltage_notes",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sc, port in portfolios.items():
            w.writerow(
                {
                    "scenario": sc,
                    "peak_loading_pct": port.get("peak_loading_pct"),
                    "n_overloaded": port.get("n_overloaded"),
                    "n_lines_at_or_above_90pct": port.get("n_lines_at_or_above_90pct"),
                    "voltage_violation_count": port.get("voltage_violation_count"),
                    "max_voltage_deviation_pu": port.get("max_voltage_deviation_pu"),
                    "pf_converged": port.get("pf_converged"),
                    "voltage_notes": port.get("notes") or "",
                }
            )
    return out_csv


def _health_subtitle(port: dict, extra: str | None = None) -> str:
    peak = float(port.get("peak_loading_pct") or 0.0)
    n_ov = port.get("n_overloaded")
    n_v = port.get("voltage_violation_count")
    pf_ok = port.get("pf_converged")
    bits = [f"peak {peak:.1f}%"]
    if n_ov is not None:
        bits.append(f"{int(n_ov)} cong.")
    if pf_ok is False:
        bits.append("V n/a")
    elif n_v is not None:
        bits.append(f"{int(n_v)} V viol.")
    if extra:
        bits.insert(0, extra)
    return " · ".join(bits)


def _operator_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=OPERATOR_COLORS[op],
            markeredgecolor="#2B0700",
            markersize=9,
            linestyle="None",
            label=OPERATOR_LABELS[op],
        )
        for op in OPERATOR_ORDER
    ]


def write_panel_caption(cap_ax, letter: str, text: str, *, fontsize: float = 7.5) -> None:
    cap_ax.set_axis_off()
    cap_ax.text(
        0.5,
        0.50,
        f"({letter}) {text}",
        transform=cap_ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def _platecarree_aspect(scale: str) -> float:
    """Width/height of a PlateCarree axes whose extent is in degrees (no cosφ)."""
    lon0, lon1, lat0, lat1 = EXTENTS[scale]
    return (lon1 - lon0) / max(lat1 - lat0, 1e-6)


def captioned_geo_figure(
    *,
    nrows: int,
    ncols: int,
    scale: str,
    pw: float = 3.4,
) -> tuple:
    """Map slots match PlateCarree aspect so cartopy does not letterbox the rows."""
    import cartopy.crs as ccrs

    aspect = _platecarree_aspect(scale)
    ph = pw / aspect
    cap_h = 0.14
    title_h = 0.26
    leg_h = 0.14
    pair_gap = 0.0
    fig_w = ncols * pw + 0.18
    pair_h = ph + cap_h
    fig_h = title_h + nrows * pair_h + nrows * pair_gap + leg_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    n_outer = nrows + 1
    mean_h = (nrows * pair_h + leg_h) / n_outer
    outer = fig.add_gridspec(
        n_outer,
        1,
        height_ratios=[pair_h] * nrows + [leg_h],
        left=0.03,
        right=0.97,
        top=1.0 - title_h / fig_h,
        bottom=0.01,
        hspace=pair_gap / mean_h,
    )
    panels: list[tuple] = []
    for row in range(nrows):
        inner = outer[row].subgridspec(
            2,
            ncols,
            height_ratios=[ph, cap_h],
            hspace=0.0,
            wspace=0.05,
        )
        for col in range(ncols):
            ax = fig.add_subplot(inner[0, col], projection=ccrs.PlateCarree())
            cap = fig.add_subplot(inner[1, col])
            cap.set_axis_off()
            panels.append((ax, cap))
    leg_ax = fig.add_subplot(outer[nrows])
    leg_ax.set_axis_off()
    return fig, panels, leg_ax


def set_panel_caption(ax, letter: str, text: str, *, fontsize: float = 7.5) -> None:
    """Place '(a) subtitle' centered just below the map panel."""
    ax.set_title("")
    ax.text(
        0.5,
        -0.025,
        f"({letter}) {text}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        clip_on=False,
        linespacing=1.15,
    )


def place_delta_legend(
    fig,
    *,
    threshold: float,
    ax=None,
    fontsize: float = 6.5,
    operator_colors: dict[str, str] | None = None,
) -> None:
    """One-row legend: star/line and label share the same vertical center."""
    if ax is None:
        ax = fig.add_axes([0.03, 0.01, 0.94, 0.055])
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    colors = operator_colors or OPERATOR_COLORS
    entries: list[tuple[str, str, str]] = [
        ("star", colors.get(op, OPERATOR_COLORS[op]), OPERATOR_LABELS[op]) for op in OPERATOR_ORDER
    ]
    entries += [
        ("line", DELTA_INC, f"Increase ≥{threshold:.0f} pp"),
        ("line", DELTA_DEC, f"Decrease ≥{threshold:.0f} pp"),
    ]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_w_px = max(ax.get_window_extent(renderer=renderer).width, 1.0)
    dpi = float(fig.dpi)
    fs = float(fontsize)
    # Center marks with text in a short legend strip (no empty band above icons).
    y_text = 0.50
    y_mark = 0.58
    pad_px = 5.5
    gap_px = 13.0
    group_gap_px = 0.0
    star_pt = max(8.0, fs + 1.5)
    line_pt = max(12.0, fs + 3.0)

    def _text_w(s: str) -> float:
        t = ax.text(0.0, 2.0, s, fontsize=fs, alpha=0.0)
        w = t.get_window_extent(renderer=renderer).width
        t.remove()
        return float(w)

    widths_px: list[float] = []
    for kind, _color, lab in entries:
        icon_px = (star_pt / 72.0) * dpi if kind == "star" else (line_pt / 72.0) * dpi
        widths_px.append(icon_px + pad_px + _text_w(lab))

    total_px = sum(widths_px) + gap_px * (len(entries) - 1) + group_gap_px
    x_px = 0.5 * ax_w_px - 0.5 * total_px

    for i, (kind, color, lab) in enumerate(entries):
        x_px += group_gap_px if i == len(OPERATOR_ORDER) else 0.0
        x = x_px / ax_w_px
        if kind == "star":
            star_frac = (star_pt / 72.0) * dpi / ax_w_px
            ax.plot(
                x + 0.5 * star_frac,
                y_mark,
                marker="*",
                markersize=star_pt,
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor="#2B0700",
                markeredgewidth=0.25,
                transform=ax.transAxes,
                clip_on=False,
                zorder=3,
            )
            ax.text(
                x + star_frac + pad_px / ax_w_px,
                y_text,
                lab,
                fontsize=fs,
                color="#222222",
                ha="left",
                va="center",
                transform=ax.transAxes,
                clip_on=False,
            )
        else:
            line_frac = (line_pt / 72.0) * dpi / ax_w_px
            ax.plot(
                [x, x + line_frac],
                [y_mark, y_mark],
                color=color,
                lw=1.8,
                solid_capstyle="butt",
                transform=ax.transAxes,
                clip_on=False,
                zorder=3,
            )
            ax.text(
                x + line_frac + pad_px / ax_w_px,
                y_text,
                lab,
                fontsize=fs,
                color="#222222",
                ha="left",
                va="center",
                transform=ax.transAxes,
                clip_on=False,
            )
        x_px += widths_px[i] + gap_px


def place_abs_legend(
    fig,
    *,
    ax=None,
    fontsize: float = 6.5,
    operator_colors: dict[str, str] | None = None,
) -> None:
    """Operator stars only (absolute-map colorbar is placed separately)."""
    if ax is None:
        ax = fig.add_axes([0.03, 0.01, 0.94, 0.055])
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    colors = operator_colors or OPERATOR_COLORS
    entries = [(colors.get(op, OPERATOR_COLORS[op]), OPERATOR_LABELS[op]) for op in OPERATOR_ORDER]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_w_px = max(ax.get_window_extent(renderer=renderer).width, 1.0)
    dpi = float(fig.dpi)
    fs = float(fontsize)
    y_text = 0.50
    y_mark = 0.58
    pad_px = 5.5
    gap_px = 13.0
    star_pt = max(8.0, fs + 1.5)

    def _text_w(s: str) -> float:
        t = ax.text(0.0, 2.0, s, fontsize=fs, alpha=0.0)
        w = t.get_window_extent(renderer=renderer).width
        t.remove()
        return float(w)

    widths_px = [(star_pt / 72.0) * dpi + pad_px + _text_w(lab) for _, lab in entries]
    total_px = sum(widths_px) + gap_px * (len(entries) - 1)
    x_px = 0.5 * ax_w_px - 0.5 * total_px

    for (color, lab), w_px in zip(entries, widths_px):
        x = x_px / ax_w_px
        star_frac = (star_pt / 72.0) * dpi / ax_w_px
        ax.plot(
            x + 0.5 * star_frac,
            y_mark,
            marker="*",
            markersize=star_pt,
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="#2B0700",
            markeredgewidth=0.25,
            transform=ax.transAxes,
            clip_on=False,
            zorder=3,
        )
        ax.text(
            x + star_frac + pad_px / ax_w_px,
            y_text,
            lab,
            fontsize=fs,
            color="#222222",
            ha="left",
            va="center",
            transform=ax.transAxes,
            clip_on=False,
        )
        x_px += w_px + gap_px


def plot_pf_delta_grid(
    *,
    portfolios: dict,
    panel_keys: list[str],
    ref_key: str,
    ref_edges: list,
    title: str,
    out_path: Path,
    label_fn: Callable[[str], str],
    threshold: float = 5.0,
    label_threshold: float | None = None,
    e02_root: Path,
) -> Path:
    import cartopy.crs as ccrs

    apply_nature_comm_style()
    apply_geo_style()
    backfill_congestion_metrics(portfolios)
    n, dc_map, meta, dc_refs, site_colors = load_e02_geo_context(e02_root)
    scale = plot_scale_for(e02_root)

    ncols = min(5, max(1, len(panel_keys)))
    nrows = (len(panel_keys) + ncols - 1) // ncols
    aspect = panel_aspect(scale)
    pw = 3.2
    fig = plt.figure(figsize=(pw * ncols + 0.6, (pw / aspect) * nrows + 0.75), facecolor="white")

    for idx, key in enumerate(panel_keys):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection=ccrs.PlateCarree())
        edges = pd.DataFrame(portfolios.get(key, {}).get("edges") or [])
        if edges.empty:
            ax.set_visible(False)
            continue
        n_inc, n_dec = draw_geo_panel_delta_loading(
            ax,
            edges,
            ref_edges,
            n,
            scale=scale,
            dc_map=dc_map,
            dc_meta=meta,
            dc_refs=dc_refs,
            delta_threshold=threshold,
            label_threshold=label_threshold,
            dc_size_ctx=MAP_DC_SIZE_CTX,
            dc_coords=MAP_DC_COORDS,
            site_colors=site_colors,
        )
        extra = f"+{n_inc}/−{n_dec} br"
        set_panel_caption(
            ax,
            chr(ord("a") + idx),
            f"{label_fn(key)}  ·  {_health_subtitle(portfolios.get(key, {}), extra)}",
        )

    place_delta_legend(fig, threshold=threshold)
    fig.suptitle(title, fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.08, wspace=0.06, hspace=0.16)
    return save_figure(fig, out_path)


def plot_pf_abs_grid(
    *,
    portfolios: dict,
    panel_keys: list[str],
    title: str,
    out_path: Path,
    label_fn: Callable[[str], str],
    loading_threshold: float = ABS_LOADING_MIN,
    label_fontsize: float | None = 5.0,
    e02_root: Path,
) -> Path:
    import cartopy.crs as ccrs

    apply_nature_comm_style()
    apply_geo_style()
    backfill_congestion_metrics(portfolios)
    n, dc_map, meta, dc_refs, site_colors = load_e02_geo_context(e02_root)
    scale = plot_scale_for(e02_root)

    ncols = min(5, max(1, len(panel_keys)))
    nrows = (len(panel_keys) + ncols - 1) // ncols
    aspect = panel_aspect(scale)
    pw = 3.2
    fig = plt.figure(figsize=(pw * ncols + 1.35, (pw / aspect) * nrows + 0.70), facecolor="white")
    lc_ref = None

    for idx, key in enumerate(panel_keys):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection=ccrs.PlateCarree())
        edges = portfolios.get(key, {}).get("edges") or []
        if not edges:
            ax.set_visible(False)
            continue
        lc = draw_geo_panel_abs_loading(
            ax,
            edges,
            n,
            scale=scale,
            dc_map=dc_map,
            dc_meta=meta,
            dc_refs=dc_refs,
            loading_threshold=loading_threshold,
            label_fontsize=label_fontsize,
            dc_size_ctx=MAP_DC_SIZE_CTX,
            dc_coords=MAP_DC_COORDS,
            site_colors=site_colors,
        )
        if lc is not None:
            lc_ref = lc
        set_panel_caption(
            ax,
            chr(ord("a") + idx),
            f"{label_fn(key)}  ·  {_health_subtitle(portfolios.get(key, {}))}",
        )

    fig.subplots_adjust(left=0.03, right=0.90, top=0.93, bottom=0.070, wspace=0.06, hspace=0.14)
    if lc_ref is not None:
        cax = fig.add_axes([0.915, 0.12, 0.012, 0.70])
        cbar = fig.colorbar(lc_ref, cax=cax)
        cbar.set_label(f"Line loading (≥{loading_threshold:.0f}% highlighted) [%]", fontsize=7)
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
    fig.suptitle(title, fontsize=10, y=0.98)
    return save_figure(fig, out_path)
