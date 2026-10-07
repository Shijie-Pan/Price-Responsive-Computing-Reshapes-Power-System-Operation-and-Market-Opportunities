#!/usr/bin/env python3
"""USA load phase-propagation maps — mixed open demand sources.

Sources (UTC hour-of-day mean over a full calendar month):
  • PUDL FERC-714 estimated hourly state demand (48 CONUS states)
  • PUDL EIA-930 hourly BA demand (~50+ BAs)

Display metric (per node): log10 of the fraction by which this hour's
load sits below that node's own peak load,
  f(h) = (P_peak − P(h)) / P_peak
  colour is log10(f), floored at 0.1% and capped at 10%.
  Darker means closer to the local peak load.

Three panels: East / Central / West peak. Uniform Fig1-sized markers.

Outputs:
  ExtData_Fig_usa_load_propagation_{summer,winter}.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

def _package_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "experiments" / "data" / "networks").is_dir() and (candidate / "experiments" / "lib").is_dir():
            return candidate
    raise RuntimeError("Cannot find the deposit root")


ROOT = _package_root()
sys.path.insert(0, str(ROOT))

from experiments.lib.figures.nc_style import (  # noqa: E402
    apply_figure_style,
    figsize_double_col,
    save_figure,
)
from experiments.lib.us_map import (  # noqa: E402
    CONUS_XLIM,
    CONUS_YLIM,
    NAME_TO_ABBR,
    draw_base_map,
    draw_conus_borders,
    load_geojson,
)
from experiments.lib.figure_common import _interpolate_bus_field  # noqa: E402
from experiments.lib.figure_style import (  # noqa: E402
    FIG1_STATE_DOT_S,
    FILL_ALPHA,
    MAP_FS_LAB,
    MAP_FS_ROW_TITLE,
    MAP_FS_TAG,
    MAP_FS_TICK,
    MAP_WSPACE,
)

MERGED = ROOT / "experiments/E07/results"
NATIONWIDE_CFG = ROOT / "experiments/E03/config/texas7_operators.json"
DATA = ROOT / "experiments/data/eia930"
STATE_DEMAND = DATA / "out_ferc714__hourly_estimated_state_demand.parquet"
BA_OPS = DATA / "core_eia930__hourly_operations.parquet"
BA_CODES = DATA / "core_eia__codes_balancing_authorities.parquet"
PLANTS = DATA / "out_eia__yearly_plants.parquet"
OUT_RESULTS = ROOT / "experiments/E07/results"
OUT_FIG1 = ROOT / "experiments/E01/figures"
OUT_FIG7 = ROOT / "experiments/E07/figures"

DEMAND_YEAR = 2025
# Inclusive UTC month windows (hour-of-day mean over the full month).
MONTH_BOUNDS = {
    "summer": (f"{DEMAND_YEAR}-07-01 00:00:00", f"{DEMAND_YEAR}-08-01 00:00:00"),
    "winter": (f"{DEMAND_YEAR}-01-01 00:00:00", f"{DEMAND_YEAR}-02-01 00:00:00"),
}

FIPS_TO_STATE = {
    "01": "AL",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

MAP_HOURS = {
    # ~3–4 h East→West evening lag (medians ≈ summer 22/23/01, winter eve 00/01/03).
    "summer": (21, 23, 1),
    "winter": (23, 1, 3),
}
MAP_PEAK_LABEL = {
    "summer": {21: "East peak", 23: "Central peak", 1: "West peak"},
    "winter": {23: "East peak", 1: "Central peak", 3: "West peak"},
}
TEXAS_UTC_OFFSET = {"summer": -5, "winter": -6}
# Peak hour = argmax over evening window only (avoids winter morning↔evening mix).
EVENING_HOURS = tuple(list(range(18, 24)) + list(range(0, 6)))
PEAK_SIGMA_H = 1.5
MANUAL_BA_XY: dict[str, tuple[float, float]] = {
    "BHBA": (44.08, -103.23),
    "SWPW": (42.0, -108.0),
}


def _require_data() -> None:
    needed = [STATE_DEMAND, BA_OPS, PLANTS, BA_CODES]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing:\n  " + "\n  ".join(str(p) for p in missing))


def _rings(geom: dict) -> list[list[tuple[float, float]]]:
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    rings: list[list[tuple[float, float]]] = []
    if gtype == "Polygon" and coords:
        rings.append([(float(x), float(y)) for x, y in coords[0]])
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                rings.append([(float(x), float(y)) for x, y in poly[0]])
    return rings


def build_state_centroids(geo: dict) -> pd.DataFrame:
    rows = []
    for feat in geo.get("features", []):
        abbr = NAME_TO_ABBR.get(feat.get("properties", {}).get("name", ""))
        if abbr not in FIPS_TO_STATE.values():
            continue
        xs, ys = [], []
        for ring in _rings(feat.get("geometry", {})):
            xs.extend(x for x, _ in ring)
            ys.extend(y for _, y in ring)
        if xs:
            rows.append({"node": f"ST:{abbr}", "lon": float(np.mean(xs)), "lat": float(np.mean(ys)), "kind": "state"})
    return pd.DataFrame(rows).set_index("node")


def build_ba_centroids() -> pd.DataFrame:
    plants = pd.read_parquet(PLANTS)
    plants = plants.dropna(subset=["latitude", "longitude", "balancing_authority_code_eia"])
    plants = plants[
        plants["latitude"].between(24.0, 50.0) & plants["longitude"].between(-125.0, -66.0)
    ]
    plants["year"] = pd.to_datetime(plants["report_date"]).dt.year
    p = plants[plants["year"] == plants["year"].max()]
    cent = p.groupby("balancing_authority_code_eia").agg(
        lat=("latitude", "mean"), lon=("longitude", "mean")
    )
    for ba, (lat, lon) in MANUAL_BA_XY.items():
        if ba not in cent.index:
            cent.loc[ba] = {"lat": lat, "lon": lon}
    codes = pd.read_parquet(BA_CODES).set_index("code")
    gen_only = codes["is_generation_only"].fillna(False)
    keep = [b for b in cent.index if not bool(gen_only.get(b, False))]
    cent = cent.loc[keep]
    out = cent.rename_axis("ba").reset_index()
    out["node"] = "BA:" + out["ba"].astype(str)
    out["kind"] = "ba"
    return out.set_index("node")[["lon", "lat", "kind"]]


def _month_bounds(season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_s, end_s = MONTH_BOUNDS[season]
    return pd.Timestamp(start_s, tz="UTC"), pd.Timestamp(end_s, tz="UTC")


def load_month_hod_mean_state(season: str) -> pd.DataFrame:
    """24 × state mean demand by UTC hour-of-day over the season month."""
    start, end = _month_bounds(season)
    df = pd.read_parquet(STATE_DEMAND)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["state_id_fips"] = df["state_id_fips"].astype(str).str.zfill(2)
    win = df[(df["datetime_utc"] >= start) & (df["datetime_utc"] < end)].copy()
    win["state"] = win["state_id_fips"].map(FIPS_TO_STATE)
    win = win[win["state"].notna()]
    col = (
        "scaled_demand_mwh"
        if win["scaled_demand_mwh"].notna().mean() > 0.9
        else "demand_mwh"
    )
    pivot = win.pivot_table(
        index="datetime_utc", columns="state", values=col, aggfunc="mean"
    ).sort_index()
    pivot.columns = [f"ST:{c}" for c in pivot.columns]
    hod = pivot.copy()
    hod.index = pivot.index.hour
    return hod.groupby(level=0).mean().reindex(range(24)).interpolate(axis=0).ffill().bfill()


def load_month_hod_mean_ba(season: str, ba_nodes: pd.DataFrame) -> pd.DataFrame:
    start, end = _month_bounds(season)
    cols = [
        "balancing_authority_code_eia",
        "datetime_utc",
        "demand_adjusted_mwh",
        "demand_imputed_eia_mwh",
        "demand_reported_mwh",
    ]
    ops = pd.read_parquet(BA_OPS, columns=cols)
    ops["datetime_utc"] = pd.to_datetime(ops["datetime_utc"], utc=True)
    win = ops[(ops["datetime_utc"] >= start) & (ops["datetime_utc"] < end)].copy()
    for c in ("demand_adjusted_mwh", "demand_imputed_eia_mwh", "demand_reported_mwh"):
        if win[c].notna().any():
            win["demand"] = win[c].astype(float)
            break
    else:
        raise RuntimeError("No BA demand column")
    bas = [n.split(":", 1)[1] for n in ba_nodes.index]
    win = win[win["balancing_authority_code_eia"].isin(bas)]
    pivot = win.pivot_table(
        index="datetime_utc",
        columns="balancing_authority_code_eia",
        values="demand",
        aggfunc="mean",
    ).sort_index()
    ok = (pivot.notna().mean(axis=0) >= 0.5) & (pivot.mean(axis=0) > 1.0)
    pivot = pivot.loc[:, ok]
    pivot.columns = [f"BA:{c}" for c in pivot.columns]
    hod = pivot.copy()
    hod.index = pivot.index.hour
    return hod.groupby(level=0).mean().reindex(range(24)).interpolate(axis=0).ffill().bfill()


def _latest_merged(scenario: str, season: str) -> Path:
    matches = sorted(MERGED.glob(f"power_merged_{scenario}_{season}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No merged power for {scenario}/{season} in {MERGED}")
    return matches[-1]


def _site_coords() -> pd.DataFrame:
    cfg = json.loads(NATIONWIDE_CFG.read_text(encoding="utf-8"))
    rows = [
        {
            "site_id": int(s["site_id"]),
            "lat": float(s["reference_lat"]),
            "lon": float(s["reference_lon"]),
        }
        for s in cfg["sites"]
    ]
    return pd.DataFrame(rows)


def _dc_hour_of_day(power_csv: Path) -> pd.DataFrame:
    """UTC hour-of-day mean site power over the 168 h scheduling window."""
    df = pd.read_csv(power_csv)
    df = df.iloc[:168].copy()
    cols = [c for c in df.columns if c.startswith("Site") and c.endswith("_Power_MW")]
    out = df[cols].astype(float)
    out.index = df["Time_Period"].astype(int).to_numpy() % 24
    return out.groupby(level=0).mean().reindex(range(24)).fillna(0.0)


def _add_dc_to_nearest_nodes(
    avg: pd.DataFrame,
    lon: pd.Series,
    lat: pd.Series,
    power_csv: Path,
) -> tuple[pd.DataFrame, dict]:
    """Add each site's power once, onto the nearest plotted state or BA node."""
    sites = _site_coords()
    hod = _dc_hour_of_day(power_csv)
    node_lon = lon.to_numpy(float)
    node_lat = lat.to_numpy(float)
    nodes = list(lon.index)
    extra = pd.DataFrame(0.0, index=avg.index, columns=avg.columns)
    assigned: dict[str, list[int]] = {}
    for row in sites.itertuples(index=False):
        col = f"Site{int(row.site_id)}_Power_MW"
        if col not in hod.columns:
            continue
        dlon = (node_lon - float(row.lon)) * np.cos(np.deg2rad(float(row.lat)))
        dlat = node_lat - float(row.lat)
        j = int(np.argmin(dlon**2 + dlat**2))
        node = nodes[j]
        extra[node] = extra[node] + hod[col].to_numpy(float)
        assigned.setdefault(node, []).append(int(row.site_id))
    added = avg + extra
    info = {
        "power_csv": str(power_csv.relative_to(ROOT)).replace("\\", "/"),
        "n_sites": int(sum(len(v) for v in assigned.values())),
        "n_nodes_with_dc": int(len(assigned)),
        "mean_added_mw": float(extra.to_numpy().mean()),
    }
    return added, info


def _phase_vs_local_peak(avg: pd.DataFrame, sigma_h: float = PEAK_SIGMA_H) -> pd.DataFrame:
    """Gaussian proximity to each node's evening peak hour (1 at peak)."""
    eve = avg.reindex(list(EVENING_HOURS))
    peak_h = eve.idxmax(axis=0).astype(float)
    hours = np.asarray(avg.index, dtype=float)[:, None]
    peaks = peak_h.to_numpy(dtype=float)[None, :]
    delta = np.abs((hours - peaks + 12.0) % 24.0 - 12.0)
    z = np.exp(-0.5 * (delta / sigma_h) ** 2)
    return pd.DataFrame(z, index=avg.index, columns=avg.columns)


def alignment_check(season: str) -> dict:
    """CONUS state-sum vs EIA-930 CONUS for the season month."""
    start, end = _month_bounds(season)
    st = pd.read_parquet(STATE_DEMAND)
    st["datetime_utc"] = pd.to_datetime(st["datetime_utc"], utc=True)
    st["state_id_fips"] = st["state_id_fips"].astype(str).str.zfill(2)
    win = st[(st["datetime_utc"] >= start) & (st["datetime_utc"] < end)]
    win = win[~win["state_id_fips"].isin(["02", "15"])]
    st_sum = win.groupby("datetime_utc")["scaled_demand_mwh"].sum()
    agg = pd.read_parquet(DATA / "out_eia930__hourly_aggregated_demand.parquet")
    agg["datetime_utc"] = pd.to_datetime(agg["datetime_utc"], utc=True)
    conus = agg.loc[
        (agg["aggregation_level"] == "conus")
        & (agg["datetime_utc"] >= start)
        & (agg["datetime_utc"] < end)
    ].set_index("datetime_utc")["demand_imputed_pudl_mwh"]
    both = pd.DataFrame({"state_sum": st_sum, "eia930_conus": conus}).dropna()
    return {
        "n_hours": int(len(both)),
        "corr": float(both["state_sum"].corr(both["eia930_conus"])),
        "mean_ratio_state_over_eia930": float((both["state_sum"] / both["eia930_conus"]).mean()),
    }


def build_propagation_board(season: str, case: str = "background") -> Path:
    apply_figure_style()
    plt.rcParams.update(
        {
            "font.size": MAP_FS_LAB,
            "axes.labelsize": MAP_FS_LAB,
            "xtick.labelsize": MAP_FS_TICK,
            "ytick.labelsize": MAP_FS_TICK,
        }
    )
    geo = load_geojson()
    if geo is None:
        raise RuntimeError("US geojson unavailable")

    _require_data()
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)

    st_xy = build_state_centroids(geo)
    ba_xy = build_ba_centroids()
    st_avg = load_month_hod_mean_state(season)
    ba_avg = load_month_hod_mean_ba(season, ba_xy)
    # Keep BA nodes that have both demand and coordinates
    ba_avg = ba_avg[[c for c in ba_avg.columns if c in ba_xy.index]]
    avg = pd.concat([st_avg, ba_avg], axis=1)
    xy = pd.concat([st_xy, ba_xy.loc[ba_avg.columns]])
    cols = [c for c in avg.columns if c in xy.index]
    avg = avg[cols]
    lon = xy.loc[cols, "lon"].astype(float)
    lat = xy.loc[cols, "lat"].astype(float)

    case_key = {"background": None, "with_dc": "baseline", "full_flex": "reallocation_slack_termination"}
    if case not in case_key:
        raise ValueError(case)
    dc_info = None
    if case_key[case] is not None:
        power_csv = _latest_merged(case_key[case], season)
        avg, dc_info = _add_dc_to_nearest_nodes(avg, lon, lat, power_csv)
        print(
            f"[dc] {case}: sites={dc_info['n_sites']} on {dc_info['n_nodes_with_dc']} nodes, "
            f"file={power_csv.name}",
            flush=True,
        )

    suffix = "" if case == "background" else f"_{case}"
    avg.to_csv(
        OUT_RESULTS / f"hybrid_demand_hod_mean_{season}_{DEMAND_YEAR}{suffix}.csv",
        float_format="%.4f",
    )
    xy.loc[cols].to_csv(OUT_RESULTS / f"hybrid_nodes_{season}.csv")

    hours = MAP_HOURS[season]
    z = _log_fraction_below_peak(avg)
    vmin = float(np.log10(PEAK_GAP_FLOOR))
    vmax = float(np.log10(PEAK_GAP_CAP))
    align = alignment_check(season)

    n_state = sum(c.startswith("ST:") for c in cols)
    n_ba = sum(c.startswith("BA:") for c in cols)
    start, end = _month_bounds(season)
    meta = {
        "sources": [
            "PUDL out_ferc714__hourly_estimated_state_demand",
            "PUDL core_eia930__hourly_operations",
        ],
        "year": DEMAND_YEAR,
        "month_start_utc": str(start),
        "month_end_utc_exclusive": str(end),
        "aggregation": "UTC hour-of-day mean over full calendar month",
        "n_nodes": int(len(cols)),
        "n_states": n_state,
        "n_bas": n_ba,
        "alignment_conus": align,
        "map_hours_utc": list(hours),
        "panel_letters": ["a", "b", "c"],
        "peak_roles": [MAP_PEAK_LABEL[season][h] for h in hours],
        "case": case,
        "dc_overlay": dc_info,
        "display": (
            "log10((P_peak-P)/P_peak); peak is the max hour-of-day mean; "
            f"floor {PEAK_GAP_FLOOR:g}, cap {PEAK_GAP_CAP:g}"
        ),
        "vmin": vmin,
        "vmax": vmax,
    }
    (OUT_RESULTS / f"meta_{season}{suffix}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[align] {season}: corr={align['corr']:.3f}, "
        f"ratio={align['mean_ratio_state_over_eia930']:.3f}, "
        f"nodes={n_state} states + {n_ba} BAs",
        flush=True,
    )

    # Reversed so a smaller gap (closer to the peak load) is darker.
    cmap = plt.get_cmap("YlOrRd_r").copy()
    cmap.set_bad("#f0f0f0")
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 17)

    fig_w = float(figsize_double_col(1.0)[0])
    dx = float(CONUS_XLIM[1] - CONUS_XLIM[0])
    dy = float(CONUS_YLIM[1] - CONUS_YLIM[0])
    n_maps = len(hours)
    gutter_frac = 0.04
    left_m, right_m, bot_m, top_m = 0.06, 0.90, 0.16, 0.88
    panel_w_in = fig_w * (right_m - left_m) / (n_maps + (n_maps - 1) * gutter_frac)
    map_h_in = panel_w_in * (dy / dx)
    fig_h = map_h_in / (top_m - bot_m)
    fig, axes = plt.subplots(
        1,
        n_maps,
        figsize=(fig_w, fig_h),
        facecolor="white",
        gridspec_kw={"wspace": gutter_frac},
    )
    fig.subplots_adjust(left=left_m, right=right_m, bottom=bot_m, top=top_m)
    if n_maps == 1:
        axes = [axes]

    panel_letters = ("a", "b", "c")
    for i, (ax, hour) in enumerate(zip(axes, hours)):
        draw_base_map(ax, geo, {}, alpha=0.95, default_color="#ececec")
        vals = z.loc[hour]
        grid_x, grid_y, field = _interpolate_bus_field(
            lon.to_numpy(float),
            lat.to_numpy(float),
            vals.reindex(lon.index).to_numpy(float),
            geo,
            grid_n=280,
        )
        z_m = np.ma.masked_invalid(field)
        draw_conus_borders(ax, geo, zorder=3)
        ax.contourf(
            grid_x,
            grid_y,
            z_m,
            levels=levels,
            cmap=cmap,
            norm=norm,
            alpha=FILL_ALPHA,
            extend="both",
            zorder=2,
        )
        ax.scatter(
            lon,
            lat,
            s=FIG1_STATE_DOT_S,
            c="0.2",
            alpha=0.85,
            linewidths=0,
            zorder=6,
        )
        ax.set_xlim(CONUS_XLIM)
        ax.set_ylim(CONUS_YLIM)
        ax.set_aspect("auto")
        ax.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=0.4)
        if i == 0:
            ax.set_ylabel("Latitude", fontsize=MAP_FS_LAB)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        ax.set_xlabel("Longitude" if i == 1 else "", fontsize=MAP_FS_LAB, labelpad=2)
        ax.set_title(
            f"{panel_letters[i]}  {MAP_PEAK_LABEL[season][hour]}",
            fontsize=MAP_FS_TAG,
            fontweight="bold",
            pad=4,
            loc="left",
        )
        ax.text(
            0.03,
            0.04,
            f"UTC {hour:02d}",
            transform=ax.transAxes,
            fontsize=MAP_FS_TAG,
            fontweight="bold",
            va="bottom",
            ha="left",
            color="0.1",
            zorder=10,
        )

    cax = fig.add_axes([right_m + 0.012, bot_m, 0.014, top_m - bot_m])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Fraction below peak load", fontsize=MAP_FS_LAB, labelpad=3)
    cb.set_ticks([-3.0, -2.0, -1.0])
    cb.set_ticklabels(["0.1%", "1%", "10%"])
    cb.ax.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=1.0)

    if season == "summer" and case == "background":
        out = OUT_FIG1 / "Fig1_summer_peak_load.png"
    else:
        out = OUT_FIG1 / f"ExtData_Fig_usa_load_propagation_{season}{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    path = save_figure(fig, out, pad_inches=0.08, bbox_inches="tight")
    print(f"[ok] {season} {case}: hours={hours}, metric=log10 fraction below own peak", flush=True)
    return path


# Log color scale for (peak − load) / peak. Exact peaks are pinned to the floor.
PEAK_GAP_FLOOR = 1e-3
PEAK_GAP_CAP = 0.2
# The with-DC and full-flex series already use the 2030 Epoch campus projection.
# Scale only that increment by the 2050/2030 ratio in EIA AEO2026 Table 5
# (April 2026, Counterfactual Baseline): commercial electricity for
# Data Center Servers, 1.520461 / 0.607868 quadrillion Btu.
# Written on Fig. 7.
AEO_DC_QUAD_2030 = 0.6078678369522095
AEO_DC_QUAD_2050 = 1.520461201667786
FORECAST_YEAR = 2050
DC_LOAD_SCALE = AEO_DC_QUAD_2050 / AEO_DC_QUAD_2030


def _log_fraction_below_peak(avg: pd.DataFrame) -> pd.DataFrame:
    """log10 of how far this hour's load sits below that node's own peak load."""
    peak = avg.max(axis=0).replace(0.0, np.nan)
    frac = ((peak - avg) / peak).clip(lower=PEAK_GAP_FLOOR, upper=PEAK_GAP_CAP)
    return np.log10(frac)


def build_summary_board(season: str = "summer", only: str | None = None) -> Path:
    """Rows show how far this hour's load sits below that node's own peak.

    ``only='full_flex'`` keeps that row and omits its side label.
    """
    apply_figure_style()
    plt.rcParams.update(
        {
            "font.size": MAP_FS_LAB,
            "axes.labelsize": MAP_FS_LAB,
            "xtick.labelsize": MAP_FS_TICK,
            "ytick.labelsize": MAP_FS_TICK,
        }
    )
    geo = load_geojson()
    if geo is None:
        raise RuntimeError("US geojson unavailable")
    xy = pd.read_csv(OUT_RESULTS / f"hybrid_nodes_{season}.csv", index_col=0)
    cases = (
        ("background", f"hybrid_demand_hod_mean_{season}_{DEMAND_YEAR}.csv", "No DC"),
        ("with_dc", f"hybrid_demand_hod_mean_{season}_{DEMAND_YEAR}_with_dc.csv", "With DC"),
        ("full_flex", f"hybrid_demand_hod_mean_{season}_{DEMAND_YEAR}_full_flex.csv", "Price-responsive"),
    )
    frames = []
    for _key, name, label in cases:
        path = OUT_RESULTS / name
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append((label, pd.read_csv(path, index_col=0)))
    bg = frames[0][1]
    frames = [
        (label, avg if i == 0 else bg + DC_LOAD_SCALE * (avg - bg))
        for i, (label, avg) in enumerate(frames)
    ]
    row_years = None
    letters = ("a", "b", "c")
    if only == "full_flex" and season == "summer":
        frames = [frames[0], frames[2]]
        row_years = (DEMAND_YEAR, FORECAST_YEAR)
        letters = ("a", "b", "c", "d", "e", "f")
    elif only == "full_flex":
        frames = [frames[2]]

    hours = MAP_HOURS[season]
    lon = xy["lon"].astype(float)
    lat = xy["lat"].astype(float)
    log_min = float(np.log10(PEAK_GAP_FLOOR))
    log_max = float(np.log10(PEAK_GAP_CAP))
    # Reversed so a smaller gap (closer to the peak load) is darker.
    cmap = plt.get_cmap("YlOrRd_r").copy()
    cmap.set_bad("#f0f0f0")
    norm = Normalize(vmin=log_min, vmax=log_max)
    levels = np.linspace(log_min, log_max, 17)

    fig_w = float(figsize_double_col(1.0)[0])
    dx = float(CONUS_XLIM[1] - CONUS_XLIM[0])
    dy = float(CONUS_YLIM[1] - CONUS_YLIM[0])
    n_col = len(hours)
    n_row = len(frames)
    gutter_w, gutter_h = 0.035, 0.48 if row_years is not None else 0.22
    left_m = 0.07 if only == "full_flex" else 0.09
    left_m, right_m, bot_m, top_m = left_m, 0.90, 0.10 if n_row == 1 else 0.07, 0.90 if n_row == 1 else 0.94
    panel_w_in = fig_w * (right_m - left_m) / (n_col + (n_col - 1) * gutter_w)
    map_h_in = panel_w_in * (dy / dx)
    fig_h = map_h_in * (n_row + (n_row - 1) * gutter_h) / (top_m - bot_m)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(fig_w, fig_h),
        facecolor="white",
        gridspec_kw={"wspace": gutter_w, "hspace": gutter_h},
    )
    fig.subplots_adjust(left=left_m, right=right_m, bottom=bot_m, top=top_m)
    if n_row == 1:
        axes = np.atleast_2d(axes)

    for r, (label, avg) in enumerate(frames):
        score = _log_fraction_below_peak(avg)
        for c, hour in enumerate(hours):
            ax = axes[r, c]
            draw_base_map(ax, geo, {}, alpha=0.95, default_color="#ececec")
            vals = score.loc[hour].reindex(lon.index)
            grid_x, grid_y, field = _interpolate_bus_field(
                lon.to_numpy(float),
                lat.to_numpy(float),
                vals.to_numpy(float),
                geo,
                grid_n=220,
            )
            draw_conus_borders(ax, geo, zorder=3)
            ax.contourf(
                grid_x,
                grid_y,
                np.ma.masked_invalid(field),
                levels=levels,
                cmap=cmap,
                norm=norm,
                alpha=FILL_ALPHA,
                extend="both",
                zorder=2,
            )
            ax.scatter(
                lon, lat, s=FIG1_STATE_DOT_S, c="0.2", alpha=0.85, linewidths=0, zorder=6,
            )
            ax.set_xlim(CONUS_XLIM)
            ax.set_ylim(CONUS_YLIM)
            ax.set_aspect("auto")
            ax.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=0.3)
            if c == 0:
                ax.set_ylabel("Latitude", fontsize=MAP_FS_LAB)
                if only is None:
                    ax.text(
                        -0.28,
                        0.5,
                        label,
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=MAP_FS_TAG,
                        fontweight="bold",
                        clip_on=False,
                    )
            else:
                ax.set_yticklabels([])
            if row_years is not None and r < n_row - 1:
                ax.set_xticklabels([])
            if r == n_row - 1 and c == 1:
                ax.set_xlabel("Longitude", fontsize=MAP_FS_LAB, labelpad=1)
            else:
                ax.set_xlabel("")
            if row_years is not None or r == 0:
                peak = MAP_PEAK_LABEL[season][hour]
                letter = letters[r * n_col + c]
                year_tag = f" ({row_years[r]})" if row_years is not None else ""
                ax.set_title(
                    f"{letter}  {peak}{year_tag}",
                    fontsize=MAP_FS_TAG,
                    fontweight="bold",
                    pad=3,
                    loc="left",
                )
            ax.text(
                0.03,
                0.05,
                f"UTC {hour:02d}",
                transform=ax.transAxes,
                fontsize=MAP_FS_TICK,
                fontweight="bold",
                va="bottom",
                ha="left",
                color="0.1",
                zorder=10,
            )

    cax = fig.add_axes([right_m + 0.015, bot_m, 0.014, top_m - bot_m])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Fraction below peak load", fontsize=MAP_FS_LAB, labelpad=3)
    cb.set_ticks([-3.0, -2.0, -1.0])
    cb.set_ticklabels(["0.1%", "1%", "10%"])
    cb.ax.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=1.0)

    if only == "full_flex" and season == "summer":
        out = OUT_FIG7 / "Fig7_load_propagation_full_flex.png"
    else:
        suffix = "_full_flex" if only == "full_flex" else "_summary"
        out = OUT_FIG7 / f"ExtData_Fig_usa_load_propagation_{season}{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    path = save_figure(fig, out, pad_inches=0.08, bbox_inches="tight")
    print(
        f"[ok] summary {season}: log10 fraction below own peak; DC increment x{DC_LOAD_SCALE:g}",
        flush=True,
    )
    return path


def build_contrast_board(season: str = "summer") -> Path:
    """Same hours, but demand change in MW so FCFS and full flex separate."""
    apply_figure_style()
    plt.rcParams.update(
        {
            "font.size": MAP_FS_LAB,
            "axes.labelsize": MAP_FS_LAB,
            "xtick.labelsize": MAP_FS_TICK,
            "ytick.labelsize": MAP_FS_TICK,
        }
    )
    geo = load_geojson()
    if geo is None:
        raise RuntimeError("US geojson unavailable")
    xy = pd.read_csv(OUT_RESULTS / f"hybrid_nodes_{season}.csv", index_col=0)
    bg = pd.read_csv(OUT_RESULTS / f"hybrid_demand_hod_mean_{season}_{DEMAND_YEAR}.csv", index_col=0)
    dc = pd.read_csv(OUT_RESULTS / f"hybrid_demand_hod_mean_{season}_{DEMAND_YEAR}_with_dc.csv", index_col=0)
    ff = pd.read_csv(OUT_RESULTS / f"hybrid_demand_hod_mean_{season}_{DEMAND_YEAR}_full_flex.csv", index_col=0)
    rows = (
        ("With DC\n− No DC", dc - bg),
        ("Price-responsive\n− No DC", ff - bg),
        ("Price-responsive\n− With DC", ff - dc),
    )
    hours = MAP_HOURS[season]
    lon = xy["lon"].astype(float)
    lat = xy["lat"].astype(float)
    vmax = 4000.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f0f0f0")
    norm = Normalize(vmin=-vmax, vmax=vmax)
    levels = np.linspace(-vmax, vmax, 21)

    fig_w = float(figsize_double_col(1.0)[0])
    dx = float(CONUS_XLIM[1] - CONUS_XLIM[0])
    dy = float(CONUS_YLIM[1] - CONUS_YLIM[0])
    n_col, n_row = len(hours), len(rows)
    gutter_w, gutter_h = 0.035, 0.06
    left_m, right_m, bot_m, top_m = 0.11, 0.90, 0.06, 0.94
    panel_w_in = fig_w * (right_m - left_m) / (n_col + (n_col - 1) * gutter_w)
    map_h_in = panel_w_in * (dy / dx)
    fig_h = map_h_in * (n_row + (n_row - 1) * gutter_h) / (top_m - bot_m)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(fig_w, fig_h),
        facecolor="white",
        gridspec_kw={"wspace": gutter_w, "hspace": gutter_h},
    )
    fig.subplots_adjust(left=left_m, right=right_m, bottom=bot_m, top=top_m)
    letters = ("a", "b", "c")
    for r, (label, delta) in enumerate(rows):
        for c, hour in enumerate(hours):
            ax = axes[r, c]
            draw_base_map(ax, geo, {}, alpha=0.95, default_color="#ececec")
            vals = delta.loc[hour].reindex(lon.index).astype(float)
            grid_x, grid_y, field = _interpolate_bus_field(
                lon.to_numpy(float),
                lat.to_numpy(float),
                vals.to_numpy(float),
                geo,
                grid_n=220,
            )
            draw_conus_borders(ax, geo, zorder=3)
            ax.contourf(
                grid_x,
                grid_y,
                np.ma.masked_invalid(field),
                levels=levels,
                cmap=cmap,
                norm=norm,
                alpha=FILL_ALPHA,
                extend="both",
                zorder=2,
            )
            ax.scatter(lon, lat, s=FIG1_STATE_DOT_S, c="0.25", alpha=0.7, linewidths=0, zorder=6)
            ax.set_xlim(CONUS_XLIM)
            ax.set_ylim(CONUS_YLIM)
            ax.set_aspect("auto")
            ax.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=0.3)
            if c == 0:
                ax.set_ylabel("Latitude", fontsize=MAP_FS_LAB)
                ax.text(
                    -0.42,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=MAP_FS_TAG,
                    fontweight="bold",
                    linespacing=1.05,
                    clip_on=False,
                )
            else:
                ax.set_yticklabels([])
            if r == n_row - 1 and c == 1:
                ax.set_xlabel("Longitude", fontsize=MAP_FS_LAB, labelpad=1)
            else:
                ax.set_xlabel("")
            if r == 0:
                ax.set_title(
                    f"{letters[c]}  {MAP_PEAK_LABEL[season][hour]}",
                    fontsize=MAP_FS_TAG,
                    fontweight="bold",
                    pad=3,
                    loc="left",
                )
            ax.text(
                0.03,
                0.05,
                f"UTC {hour:02d}",
                transform=ax.transAxes,
                fontsize=MAP_FS_TICK,
                fontweight="bold",
                va="bottom",
                ha="left",
                color="0.1",
                zorder=10,
            )

    cax = fig.add_axes([right_m + 0.015, bot_m, 0.014, top_m - bot_m])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Demand change (MW)", fontsize=MAP_FS_LAB, labelpad=2)
    cb.set_ticks([-4000, -2000, 0, 2000, 4000])
    cb.ax.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=1.0)

    out = OUT_FIG1 / f"ExtData_Fig_usa_load_propagation_{season}_contrast.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    path = save_figure(fig, out, pad_inches=0.08, bbox_inches="tight")
    print(f"[ok] contrast {season}", flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="both", choices=["summer", "winter", "both"])
    ap.add_argument(
        "--case",
        default="background",
        choices=["background", "with_dc", "full_flex", "dc_pair", "summary", "full_flex_only", "contrast"],
    )
    args = ap.parse_args()
    seasons = ["summer", "winter"] if args.season == "both" else [args.season]
    if args.case == "summary":
        for season in seasons:
            print("[ok]", build_summary_board(season))
        return
    if args.case == "full_flex_only":
        for season in seasons:
            print("[ok]", build_summary_board(season, only="full_flex"))
        return
    if args.case == "contrast":
        for season in seasons:
            print("[ok]", build_contrast_board(season))
        return
    cases = ["with_dc", "full_flex"] if args.case == "dc_pair" else [args.case]
    for season in seasons:
        for case in cases:
            p = build_propagation_board(season, case)
            print("[ok]", p)


if __name__ == "__main__":
    main()
