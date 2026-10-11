"""Plot the Texas scheduling sites used in the manuscript map."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator

import sys

def _package_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "experiments" / "data" / "networks").is_dir() and (candidate / "experiments" / "lib").is_dir():
            return candidate
    raise RuntimeError("Cannot find the deposit root")


ROOT = _package_root()
sys.path.insert(0, str(ROOT))
TEXAS_DIR = ROOT / "experiments" / "E03"

# Text anchors in data coordinates. The north-central sites sit too close
# for offset-point labels, so each name is pulled into open map space.
# (lon, lat, ha, va)
LABEL_AT = {
    733: (-100.42, 35.72, "right", "center"),  # Goodnight, inside the panhandle
    701: (-101.25, 33.55, "right", "center"),  # Helios, west and inside Texas
    712: (-100.70, 33.15, "right", "center"),  # Shackelford, west
    711: (-100.70, 31.72, "right", "center"),  # Abilene, below Shackelford
    722: (-98.35, 33.62, "right", "center"),  # Fort Worth, northwest of the cluster
    702: (-96.45, 33.48, "left", "center"),   # Denton, east and south of the Red River
    731: (-95.55, 32.18, "left", "center"),   # DFW, southeast
    721: (-98.75, 31.02, "right", "center"),  # Temple, west
    713: (-95.45, 30.28, "left", "center"),   # Milam, southeast
    751: (-101.35, 30.22, "left", "center"),  # Pecos, southeast of the largest marker
    741: (-98.68, 28.22, "center", "top"),    # San Antonio, south
}
_MARKER_AREA_PER_SQRT_MW = 12.0

_PREFIXES = (
    "Google ",
    "Meta ",
    "Microsoft ",
    "OpenAI Stargate ",
    "OpenAI ",
    "CoreWeave ",
    "Amazon ",
)


def _short_name(site: dict) -> str:
    name = str(site["name"])
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace(" TX", "")


def _marker_area(mw: float) -> float:
    return _MARKER_AREA_PER_SQRT_MW * np.sqrt(float(mw))


def _load_sites() -> list[dict]:
    path = TEXAS_DIR / "config" / "texas7_operators.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return [
        s
        for s in cfg["sites"]
        if s.get("kind") != "fictional" and "✳" not in str(s.get("name", ""))
    ]


def _draw_states(ax) -> None:
    from experiments.lib.us_map import NAME_TO_ABBR, _rings, load_geojson

    geo = load_geojson()
    if geo is None:
        raise RuntimeError("US state boundaries are not available")
    neighbors = {"NM", "OK", "AR", "LA", "CO", "KS"}
    neighbor_patches: list[Polygon] = []
    texas_patches: list[Polygon] = []
    for feat in geo.get("features", []):
        abbr = NAME_TO_ABBR.get(feat.get("properties", {}).get("name", ""))
        rings = _rings(feat.get("geometry", {}))
        if abbr == "TX":
            texas_patches.extend(Polygon(ring, closed=True) for ring in rings)
        elif abbr in neighbors:
            neighbor_patches.extend(Polygon(ring, closed=True) for ring in rings)
    if neighbor_patches:
        ax.add_collection(
            PatchCollection(
                neighbor_patches,
                facecolor="#f7f7f7",
                edgecolor="#efefef",
                linewidth=0.25,
                zorder=1,
            )
        )
    ax.add_collection(
        PatchCollection(
            texas_patches,
            facecolor="#f8f7f4",
            edgecolor="#5a5a5a",
            linewidth=0.45,
            zorder=2,
        )
    )


def plot_roster(out: Path) -> Path:
    from experiments.lib.figures.nc_style import (
        OPERATOR_COLORS,
        OPERATOR_LABELS,
        OPERATOR_ORDER,
        apply_figure_style,
        figsize_double_col,
        save_figure,
    )

    apply_figure_style()
    colors = {**OPERATOR_COLORS, "meta": "#111111"}
    sites = sorted(_load_sites(), key=lambda s: float(s["projected_total_mw"]), reverse=True)
    fig, ax = plt.subplots(figsize=figsize_double_col(6.85), facecolor="white")
    ax.set_facecolor("white")
    _draw_states(ax)

    for site in sites:
        sid = int(site["site_id"])
        lat = float(site["reference_lat"])
        lon = float(site["reference_lon"])
        color = colors[site["operator"]]
        tx, ty, ha, va = LABEL_AT[sid]
        mw = int(round(float(site["projected_total_mw"])))
        area = _marker_area(mw)
        ax.scatter(
            [lon],
            [lat],
            s=area,
            c=color,
            edgecolors="white",
            linewidths=0.45,
            zorder=5,
        )
        ax.annotate(
            f"{_short_name(site)}  {mw:,} MW",
            xy=(lon, lat),
            xytext=(tx, ty),
            textcoords="data",
            ha=ha,
            va=va,
            fontsize=6,
            color="#333333",
            arrowprops={
                "arrowstyle": "-",
                "color": "#b5b5b5",
                "lw": 0.35,
                "shrinkA": float(np.sqrt(area) / 2.0 + 0.6),
                "shrinkB": 2.5,
            },
            zorder=6,
            clip_on=False,
            annotation_clip=False,
        )

    lat0 = np.deg2rad(31.5)
    ax.set_xlim(-106.65, -93.45)
    ax.set_ylim(25.55, 36.65)
    ax.set_aspect(1.0 / np.cos(lat0), adjustable="box")
    ax.set_xlabel("Longitude", color="#555555")
    ax.set_ylabel("Latitude", color="#555555")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{abs(v):.0f}\u00b0W"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}\u00b0N"))
    ax.tick_params(labelsize=6, length=2.2, width=0.35, color="0.65", labelcolor="0.45")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.35)
        spine.set_color("0.75")
    ax.grid(False)

    operator_legend = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colors[op],
                markeredgecolor="0.35",
                markeredgewidth=0.3,
                markersize=4.6,
                label=OPERATOR_LABELS[op],
            )
            for op in OPERATOR_ORDER
        ],
        title="Operator",
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=False,
        fontsize=6,
        title_fontsize=7,
        ncol=2,
        borderaxespad=0.15,
        handletextpad=0.3,
        columnspacing=0.8,
        labelspacing=0.25,
    )
    plt.setp(operator_legend.get_title(), fontweight="regular")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.055)
    out.parent.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, out)


def main() -> None:
    out = ROOT / "experiments" / "E03" / "figures" / "ExtData_Fig3_texas_campus_map.png"
    print(plot_roster(out))


if __name__ == "__main__":
    main()
