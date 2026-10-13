"""Central registry for PyPSA case-study scales (texas_c / texas / usa).

Paths are relative to the deposit root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from experiments.lib.paths import package_root

ROOT = package_root()
NET = ROOT / "experiments" / "data" / "networks"

ScaleId = Literal["texas_c", "texas", "usa"]
PlotScale = Literal["texas_c", "texas", "usa"]

# Default case studies: clustered Texas + USA zonal (not dense s1250).
DEFAULT_SCALES: list[ScaleId] = ["texas_c", "usa"]

# Deposit ships unclustered ERCOT s1250 and USA 134-zone; texas_c aliases s1250.
_TEXAS_NC = NET / "texas" / "elec_s1250.nc"
_USA_NC = NET / "usa" / "elec_s500_c134c.nc"

NETWORKS: dict[str, Path] = {
    "texas_c": _TEXAS_NC,
    "texas": _TEXAS_NC,
    "usa": _USA_NC,
}

OUT_DIRS: dict[str, Path] = {
    "texas_c": ROOT / "experiments" / "E04",
    "texas": ROOT / "experiments" / "E04",
    "usa": ROOT / "experiments" / "E06",
}

# dc_geo_sites.json key for DC coordinates
GEO_SITES_KEY: dict[str, str] = {
    "texas_c": "texas",
    "texas": "texas",
    "usa": "usa_nationwide",
}

# dc_capacity_scale.json key (texas_c shares Epoch MW with texas)
CAPACITY_KEY: dict[str, str] = {
    "texas_c": "texas",
    "texas": "texas",
    "usa": "usa",
}

# Geo-map styling (texas_c reuses texas cartography)
PLOT_STYLE: dict[str, PlotScale] = {
    "texas_c": "texas",
    "texas": "texas",
    "usa": "usa",
}

MIN_BRANCHES: dict[str, int] = {
    "texas_c": 100,
    "texas": 1500,
    "usa": 400,
}

SIZE_TAG: dict[str, str] = {
    "texas_c": "c1250",
    "texas": "s1250",
    "usa": "134",
}

NETWORK_TITLE: dict[str, str] = {
    "texas_c": "ERCOT Transmission Network (PyPSA-USA s1250)",
    "texas": "ERCOT Transmission Network (PyPSA-USA s1250, 1890 lines)",
    "usa": "US Interconnection Network (PyPSA-USA 134-zone)",
}

ALL_SCALES = ("texas_c", "texas", "usa")


def resolve_scales(arg: str) -> list[str]:
    if arg == "both":
        return list(DEFAULT_SCALES)
    return [arg]


def bus_map_path(scale: str) -> Path:
    return OUT_DIRS[scale] / "config" / "dc_bus_map.json"


def capacity_scale(scale: str) -> str:
    return CAPACITY_KEY.get(scale, scale)


def plot_style(scale: str) -> PlotScale:
    return PLOT_STYLE.get(scale, scale)  # type: ignore[return-value]
