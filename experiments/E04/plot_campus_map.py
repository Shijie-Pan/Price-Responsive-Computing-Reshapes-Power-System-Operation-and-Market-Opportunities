"""Plot the Texas scheduling sites used in the manuscript map."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

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
    733: (-100.45, 35.72, "left", "center"),   # Goodnight, east inside the panhandle
    701: (-102.55, 33.85, "right", "center"),    # Helios, west of its marker
    712: (-101.15, 32.95, "right", "center"),   # Shackelford, due west
    711: (-101.15, 31.85, "right", "center"),   # Abilene, southwest; stays under Shackelford
    722: (-98.05, 34.35, "center", "bottom"),   # Fort Worth, north of its marker
    702: (-96.05, 33.72, "left", "center"),     # Denton, east
    731: (-95.70, 32.15, "left", "center"),     # DFW, southeast
    721: (-98.95, 31.05, "right", "center"),   # Temple, west
    713: (-95.45, 30.35, "left", "center"),     # Milam, southeast
    751: (-101.20, 30.15, "left", "center"),   # Pecos, southeast
    741: (-98.68, 28.42, "center", "top"),      # San Antonio, south
}

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
                facecolor="#f3f4f6",
                edgecolor="0.72",
                linewidth=0.35,
                zorder=1,
            )
        )
    ax.add_collection(
        PatchCollection(
            texas_patches,
            facecolor="#fbfaf6",
            edgecolor="0.25",
            linewidth=0.85,
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
    sites = _load_sites()
    fig, ax = plt.subplots(figsize=figsize_double_col(6.15), facecolor="white")
    ax.set_facecolor("white")
    _draw_states(ax)

    for site in sites:
        sid = int(site["site_id"])
        lat = float(site["reference_lat"])
        lon = float(site["reference_lon"])
        color = colors[site["operator"]]
        tx, ty, ha, va = LABEL_AT[sid]
        mw = int(round(float(site["projected_total_mw"])))
        ax.scatter(
            [lon],
            [lat],
            s=28,
            c=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )
        ax.annotate(
            f"{_short_name(site)} {mw}",
            xy=(lon, lat),
            xytext=(tx, ty),
            textcoords="data",
            ha=ha,
            va=va,
            fontsize=7,
            color="#1a1a1a",
            arrowprops={
                "arrowstyle": "-",
                "color": "0.45",
                "lw": 0.45,
                "shrinkA": 1,
                "shrinkB": 4,
            },
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.94,
            },
            zorder=6,
            clip_on=False,
        )

    ax.set_xlim(-106.9, -93.3)
    ax.set_ylim(25.7, 36.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Site capacity (MW)", fontsize=8, pad=4)
    ax.tick_params(labelsize=6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("0.35")
    ax.grid(True, color="0.88", linewidth=0.35, zorder=0)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colors[op],
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=5.5,
                label=OPERATOR_LABELS[op],
            )
            for op in OPERATOR_ORDER
        ],
        loc="lower left",
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        framealpha=1,
        fontsize=7,
        ncol=2,
        borderpad=0.45,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.07)
    out.parent.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, out)


def main() -> None:
    out = ROOT / "experiments" / "E04" / "figures" / "ExtData_Fig4_texas_campus_map.png"
    plot_roster(out)


if __name__ == "__main__":
    main()
