"""Shared US state base map for nationwide E00 geographic figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from experiments.lib.paths import package_root

DEFAULT_GEO_CACHE = package_root() / "experiments" / "E02" / "data" / "geo" / "us-states.json"
GEO_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

ABBR_TO_NAME = {v: k for k, v in NAME_TO_ABBR.items()}


def state_display_name(abbr: str) -> str:
    return ABBR_TO_NAME.get(abbr.upper(), abbr)

CONUS_XLIM = (-125, -66)
CONUS_YLIM = (24, 50)


def load_geojson(cache: Path | None = None) -> dict | None:
    cache = cache or DEFAULT_GEO_CACHE
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        import urllib.request

        with urllib.request.urlopen(GEO_URL, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception as exc:
        print(f"[warn] geojson: {exc}")
        return None


def _rings(geom: dict) -> list[list[tuple[float, float]]]:
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    rings: list[list[tuple[float, float]]] = []
    if gtype == "Polygon" and coords:
        rings.append([(x, y) for x, y in coords[0]])
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                rings.append([(x, y) for x, y in poly[0]])
    return rings


def draw_base_map(
    ax: plt.Axes,
    geo: dict,
    state_colors: dict[str, str] | None = None,
    *,
    default_color: str = "#e8e8e8",
    alpha: float = 0.85,
    xlim: tuple[float, float] = CONUS_XLIM,
    ylim: tuple[float, float] = CONUS_YLIM,
) -> None:
    state_colors = state_colors or {}
    patches, colors = [], []
    for feat in geo.get("features", []):
        abbr = NAME_TO_ABBR.get(feat.get("properties", {}).get("name", ""))
        if not abbr:
            continue
        col = state_colors.get(abbr, default_color)
        for ring in _rings(feat.get("geometry", {})):
            patches.append(Polygon(ring, closed=True))
            colors.append(col)
    pc = PatchCollection(patches, facecolor=colors, edgecolor="0.35", linewidth=0.25, alpha=alpha)
    ax.add_collection(pc)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")


def conus_land_mask(geo: dict, grid_x, grid_y):
    """Boolean mask True on CONUS land (excludes AK/HI), matching US state borders."""
    import numpy as np
    from matplotlib.path import Path

    paths: list = []
    for feat in geo.get("features", []):
        abbr = NAME_TO_ABBR.get(feat.get("properties", {}).get("name", ""))
        if not abbr or abbr in {"AK", "HI"}:
            continue
        for ring in _rings(feat.get("geometry", {})):
            if len(ring) >= 3:
                paths.append(Path(ring))
    pts = np.column_stack([np.asarray(grid_x).ravel(), np.asarray(grid_y).ravel()])
    inside = np.zeros(pts.shape[0], dtype=bool)
    for path in paths:
        inside |= path.contains_points(pts)
    return inside.reshape(np.asarray(grid_x).shape)


def mask_field_to_conus(geo: dict, grid_x, grid_y, grid_z, *, dilate: int = 1):
    """Set ocean / non-CONUS cells to NaN so contour fill follows national borders."""
    import numpy as np
    from scipy.ndimage import binary_dilation

    out = np.asarray(grid_z, dtype=float).copy()
    land = conus_land_mask(geo, grid_x, grid_y)
    if dilate > 0:
        land = binary_dilation(land, iterations=int(dilate))
    out[~land] = np.nan
    return out


def draw_conus_borders(
    ax: plt.Axes,
    geo: dict,
    *,
    edgecolor: str = "0.35",
    linewidth: float = 0.30,
    zorder: float = 3,
) -> None:
    """Redraw CONUS state borders (keep zorder below isolines)."""
    patches = []
    for feat in geo.get("features", []):
        abbr = NAME_TO_ABBR.get(feat.get("properties", {}).get("name", ""))
        if not abbr or abbr in {"AK", "HI"}:
            continue
        for ring in _rings(feat.get("geometry", {})):
            patches.append(Polygon(ring, closed=True))
    pc = PatchCollection(
        patches,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_collection(pc)


def draw_state_borders(
    ax: plt.Axes,
    geo: dict,
    states: str | list[str] | tuple[str, ...],
    *,
    edgecolor: str = "0.05",
    linewidth: float = 0.60,
    zorder: float = 4,
) -> None:
    """Redraw one or more state borders (thicker highlight for reference states)."""
    want = {str(s).upper() for s in ([states] if isinstance(states, str) else states)}
    patches = []
    for feat in geo.get("features", []):
        abbr = NAME_TO_ABBR.get(feat.get("properties", {}).get("name", ""))
        if abbr not in want:
            continue
        for ring in _rings(feat.get("geometry", {})):
            patches.append(Polygon(ring, closed=True))
    if not patches:
        return
    pc = PatchCollection(
        patches,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_collection(pc)


def style_conus_map(ax: plt.Axes, *, title: str | None = None) -> None:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
