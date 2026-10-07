#!/usr/bin/env python3
"""Fig. 6: summer temporal price range and price-taking battery value.

Baseline prices are the open-loop price-independent (FCFS) nodal duals.
Comparison prices are the last closed-loop iteration of price-responsive
scheduling. Both are existing 168 h solves; nothing is re-dispatched.

Each bus hosts one 1 MW / 4 MWh storage, round-trip efficiency 90% split
symmetrically, SOC starts and ends at 50%, and charge and discharge cannot
occur in the same hour. The battery takes prices as given.

Panel c prices the same battery on observed day-ahead LMPs. The duals enter
only as the fractional reduction in intraday price range, which scales the
price-responsive revenue. Installation cost is the 2030 mid-case overnight
cost of a 4-hour system (Cole and Karmakar, NREL/TP-6A40-85332, 326 $/kWh
in 2022$).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

def _package_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "experiments" / "data" / "networks").is_dir() and (candidate / "experiments" / "lib").is_dir():
            return candidate
    raise RuntimeError("Cannot find the deposit root")


ROOT = _package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.figures.nc_style import (  # noqa: E402
    apply_figure_style,
    figsize_double_col,
    save_figure,
)
from experiments.lib.us_map import (  # noqa: E402
    CONUS_XLIM,
    CONUS_YLIM,
    draw_base_map,
    draw_conus_borders,
    load_geojson,
)
from experiments.lib.figure_common import _interpolate_bus_field  # noqa: E402
from experiments.lib.figure_style import (  # noqa: E402
    FIG1_STATE_DOT_S,
    FILL_ALPHA,
    MAP_FS_LAB,
    MAP_FS_TAG,
    MAP_FS_TICK,
)

USA = ROOT / "experiments/E05"
PI_CSV = USA / "results/priority_168h/summer/baseline/bus_lmp_hourly.csv"
PR_CSV = USA / "results/closed_loop/iter_06/lopf/summer/bus_lmp_hourly.csv"
GEO_CSV = USA / "results/priority_168h/summer/baseline/bus_lmp_summary.csv"
DC_META = USA / "config/dc_bus_map_meta.json"
OUT = ROOT / "experiments/E06/figures/Fig6_battery_arbitrage.png"
TABLE = ROOT / "experiments/E06/results/summer_battery_vs_price_independent.csv"

POWER_MW = 1.0
ENERGY_MWH = 4.0
SOC0 = 0.5 * ENERGY_MWH
RTE = 0.9
ETA = RTE ** 0.5  # symmetric charge and discharge efficiencies
HOURS = 168
HOURS_PER_DAY = 24
WEEKS_PER_YEAR = 52
# 2030 mid projection for a complete 4-hour system, 2022$.
CAPEX_USD_PER_KWH = 326.0
CAPEX_USD_PER_MW = CAPEX_USD_PER_KWH * 1000.0 * ENERGY_MWH / POWER_MW
LIFE_YEARS = 15


def _bus_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = [c for c in raw.columns if str(c).startswith("p") and str(c)[1:].isdigit()]
    frame = raw[cols].apply(pd.to_numeric, errors="coerce")
    if len(frame) != HOURS:
        raise RuntimeError(f"{path.name} has {len(frame)} rows, expected {HOURS}")
    return frame


def _observed_prices(region: str) -> np.ndarray:
    """Day-ahead LMPs in $/MWh. Texas has January and July 2026 only."""
    cache = ROOT / "experiments/data/lmp_cache"

    def read(path: Path) -> np.ndarray:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(payload["prices"], float)

    if region == "Texas":
        return np.concatenate(
            [
                read(cache / "ERCOT/LZ_NORTH/winter_2026-01.json"),
                read(cache / "ERCOT/LZ_NORTH/summer_2026-07.json"),
            ]
        )
    if region == "Michigan":
        return read(cache / "MISO/MICHIGAN.HUB/annual_2025-08_2026-07.json")
    if region == "California":
        return read(cache / "CAISO/SP15/annual_2025-08_2026-07.json")
    raise KeyError(region)


def _annual_revenue(prices: np.ndarray) -> float:
    """Mean weekly arbitrage extended to 52 weeks, in $/MW/year."""
    n = len(prices) // HOURS * HOURS
    weeks = [_battery_value(prices[i : i + HOURS]) for i in range(0, n, HOURS)]
    return float(np.mean(weeks)) * WEEKS_PER_YEAR


def _intraday_range(prices: np.ndarray) -> float:
    days = prices.reshape(-1, HOURS_PER_DAY)
    return float(np.mean(days.max(axis=1) - days.min(axis=1)))


def _battery_value(prices: np.ndarray) -> float:
    """Optimal sale revenue minus charging cost over the week, in $."""
    T = len(prices)
    m = gp.Model("bus_battery")
    m.Params.OutputFlag = 0
    c = m.addVars(T, lb=0.0, ub=POWER_MW, name="c")
    d = m.addVars(T, lb=0.0, ub=POWER_MW, name="d")
    u = m.addVars(T, vtype=gp.GRB.BINARY, name="u")
    e = m.addVars(T + 1, lb=0.0, ub=ENERGY_MWH, name="e")
    m.addConstr(e[0] == SOC0)
    m.addConstr(e[T] == SOC0)
    for t in range(T):
        m.addConstr(c[t] <= POWER_MW * u[t])
        m.addConstr(d[t] <= POWER_MW * (1 - u[t]))
        m.addConstr(e[t + 1] == e[t] + ETA * c[t] - d[t] / ETA)
    m.setObjective(gp.quicksum(float(prices[t]) * (d[t] - c[t]) for t in range(T)), gp.GRB.MAXIMIZE)
    m.optimize()
    if m.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"battery status {m.Status}")
    return float(m.ObjVal)


def build_table() -> pd.DataFrame:
    pi = _bus_frame(PI_CSV)
    pr = _bus_frame(PR_CSV)
    buses = [c for c in pi.columns if c in pr.columns]
    geo = pd.read_csv(GEO_CSV).set_index("bus")
    rows = []
    for i, bus in enumerate(buses):
        pi_p = pi[bus].to_numpy(float)
        pr_p = pr[bus].to_numpy(float)
        if not (np.isfinite(pi_p).all() and np.isfinite(pr_p).all()):
            continue
        pi_range = _intraday_range(pi_p)
        pr_range = _intraday_range(pr_p)
        pi_value = _battery_value(pi_p)
        pr_value = _battery_value(pr_p)
        rows.append(
            {
                "bus": bus,
                "lon": float(geo.loc[bus, "lon"]),
                "lat": float(geo.loc[bus, "lat"]),
                "pi_intraday_range": pi_range,
                "pr_intraday_range": pr_range,
                "delta_intraday_range": pr_range - pi_range,
                "pi_arbitrage_usd_per_mw_week": pi_value / POWER_MW,
                "pr_arbitrage_usd_per_mw_week": pr_value / POWER_MW,
                "delta_arbitrage_usd_per_mw_week": (pr_value - pi_value) / POWER_MW,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"[battery] {i + 1}/{len(buses)}", flush=True)
    out = pd.DataFrame(rows)
    meta = json.loads(DC_META.read_text(encoding="utf-8"))
    dc_buses = {str(spec["pypsa_bus"]) for spec in meta["sites"].values()}
    out["data_centre_bus"] = out["bus"].isin(dc_buses)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE, index=False, float_format="%.6f")
    print(f"[battery] wrote {TABLE}", flush=True)
    return out


def _symmetric_limit(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    limit = float(np.max(np.abs(finite)))
    if limit <= 0:
        return 1.0
    return limit


def build_figure(table: pd.DataFrame) -> Path:
    apply_figure_style()
    plt.rcParams.update(
        {
            "font.size": MAP_FS_LAB,
            "axes.labelsize": MAP_FS_LAB,
            "axes.titlesize": MAP_FS_TAG,
            "xtick.labelsize": MAP_FS_TICK,
            "ytick.labelsize": MAP_FS_TICK,
            "legend.fontsize": MAP_FS_LAB,
        }
    )
    geo = load_geojson()
    if geo is None:
        raise RuntimeError("US geojson unavailable")

    fig_w = float(figsize_double_col(1.0)[0])
    left, right, bottom, top = 0.048, 0.992, 0.145, 0.90
    fig_h = 1.80 * 0.85 + 0.02
    # Same degree box as the other CONUS boards, then 10% narrower so b and c can widen.
    dx = float(CONUS_XLIM[1] - CONUS_XLIM[0])
    dy = float(CONUS_YLIM[1] - CONUS_YLIM[0])
    map_h_in = fig_h * (top - bottom)
    cb_fraction, cb_pad = 0.046, 0.03
    map_w_in = map_h_in * dx / dy
    map_cell_in = map_w_in / (1.0 - cb_fraction - cb_pad)
    freed = 0.10 * map_cell_in
    map_cell_in -= freed
    gap_ab, gap_bc = 0.40, 0.24
    price_w_in = 1.52 + 0.5 * freed
    x_in = left * fig_w
    c_w = right * fig_w - (x_in + map_cell_in + gap_ab + price_w_in + gap_bc)
    c_cut = 0.10 * c_w
    gap_ab += 0.5 * c_cut
    gap_bc += 0.5 * c_cut
    c_w -= c_cut
    span = top - bottom
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def _axes_at(x0: float, width: float):
        return fig.add_axes([x0 / fig_w, bottom, width / fig_w, span])

    ax_b = _axes_at(x_in, map_cell_in)
    x_in += map_cell_in + gap_ab
    ax_a = _axes_at(x_in, price_w_in)
    x_in += price_w_in + gap_bc
    ax_c = _axes_at(x_in, c_w)

    pi = _bus_frame(PI_CSV)
    pr = _bus_frame(PR_CSV)
    # p62 sits in the deepest blue of the arbitrage-change field.
    # Michigan is a modest northern decrease; California is unchanged.
    traces = (
        ("p62", "Texas", "#E64B35"),
        ("p74", "Michigan", "#3C5488"),
        ("p11", "California", "#009E73"),
    )
    hours = np.arange(HOURS)
    for bus, name, color in traces:
        ax_a.plot(hours, pr[bus].to_numpy(float), color=color, lw=1.05, ls="-", zorder=3)
        ax_a.plot(hours, pi[bus].to_numpy(float), color=color, lw=0.9, ls="--", zorder=2)
    ax_a.set_xlim(0, HOURS - 1)
    ax_a.set_ylim(-2, 46)
    ax_a.set_xlabel("Hour", fontsize=MAP_FS_LAB, labelpad=1)
    ax_a.set_ylabel("Price ($/MWh)", fontsize=MAP_FS_LAB, labelpad=0)
    ax_a.set_title("b  Nodal price", loc="left", fontsize=MAP_FS_TAG, fontweight="bold", pad=3)
    ax_a.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=0.4)
    ax_a.tick_params(axis="y", pad=0.15)
    ax_a.xaxis.set_major_locator(MaxNLocator(5))
    ax_a.yaxis.set_major_locator(MaxNLocator(5))
    handles = [
        Line2D([0], [0], color=color, lw=1.2, label=name) for _bus, name, color in traces
    ] + [
        Line2D([0], [0], color="0.2", lw=0.9, ls="--", label="Price-independent"),
        Line2D([0], [0], color="0.2", lw=1.05, ls="-", label="Price-responsive"),
    ]
    ax_a.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=MAP_FS_LAB,
        borderaxespad=0.15,
        handlelength=1.15,
        columnspacing=0.45,
        handletextpad=0.35,
        labelspacing=0.15,
    )

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f7f7f7")
    lon = table["lon"].to_numpy(float)
    lat = table["lat"].to_numpy(float)
    vals = table["delta_arbitrage_usd_per_mw_week"].to_numpy(float)
    limit = _symmetric_limit(vals)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    draw_base_map(ax_b, geo, {}, alpha=0.95, default_color="#ececec")
    grid_x, grid_y, field = _interpolate_bus_field(lon, lat, vals, geo, grid_n=220)
    draw_conus_borders(ax_b, geo, zorder=3)
    ax_b.contourf(
        grid_x,
        grid_y,
        np.ma.masked_invalid(field),
        levels=np.linspace(-limit, limit, 21),
        cmap=cmap,
        norm=norm,
        alpha=FILL_ALPHA,
        extend="both",
        zorder=2,
    )
    for bus, _name, color in traces:
        row = table.loc[table["bus"] == bus].iloc[0]
        ax_b.scatter(
            [row["lon"]],
            [row["lat"]],
            s=18,
            c=color,
            marker="o",
            linewidths=0.4,
            edgecolors="0.1",
            zorder=7,
        )
    ax_b.set_xlim(*CONUS_XLIM)
    ax_b.set_ylim(*CONUS_YLIM)
    ax_b.set_aspect("auto")
    ax_b.xaxis.set_major_locator(MaxNLocator(3))
    ax_b.yaxis.set_major_locator(MaxNLocator(4))
    ax_b.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=0.4)
    ax_b.tick_params(axis="y", pad=0.15)
    ax_b.set_ylabel("Latitude", fontsize=MAP_FS_LAB, labelpad=0)
    ax_b.set_xlabel("Longitude", fontsize=MAP_FS_LAB, labelpad=1)
    ax_b.set_title("a  Temporal arbitrage value change", loc="left", fontsize=MAP_FS_TAG, fontweight="bold", pad=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_b, fraction=0.046, pad=0.03)
    cb.set_label("Value change ($/MW/week)", fontsize=MAP_FS_LAB, labelpad=2)
    cb.ax.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=1.0)

    # Real day-ahead revenue, scaled for the responsive case by the dual's
    # intraday-range ratio. The dual is not added onto the observed prices.
    annual = {}
    for bus, name, _color in traces:
        row = table.loc[table["bus"] == bus].iloc[0]
        keep = float(row["pr_intraday_range"]) / float(row["pi_intraday_range"])
        pi_annual = _annual_revenue(_observed_prices(name))
        pr_annual = pi_annual * keep
        annual[bus] = (pi_annual, pr_annual)
        print(
            f"[payback] {name} range kept {keep * 100:.1f}%  "
            f"{CAPEX_USD_PER_MW / pi_annual:.1f} -> {CAPEX_USD_PER_MW / pr_annual:.1f} yr",
            flush=True,
        )
    slowest = max(CAPEX_USD_PER_MW / min(pair) for pair in annual.values())
    xmax = slowest * 1.08
    years = np.linspace(0.0, xmax, 400)
    capex_m = CAPEX_USD_PER_MW / 1e6
    for bus, _name, color in traces:
        pi_annual, pr_annual = annual[bus]
        ax_c.plot(years, pr_annual * years / 1e6, color=color, lw=1.05, ls="-", zorder=3)
        ax_c.plot(years, pi_annual * years / 1e6, color=color, lw=0.9, ls="--", zorder=2)
        pi_year = CAPEX_USD_PER_MW / pi_annual
        for revenue, which in ((pi_annual, "pi"), (pr_annual, "pr")):
            year = CAPEX_USD_PER_MW / revenue
            if which == "pr" and abs(year - pi_year) < 1.5:
                continue
            # Texas and Michigan cross within one year, so their labels sit on opposite sides.
            ax_c.plot([year], [capex_m], marker="o", ms=3.2, color=color, ls="none", zorder=4)
            if bus == "p62" and which == "pi":
                text_x, text_y, ha, va = year + 1.6, 0.72, "center", "center"
            elif bus == "p62" and which == "pr":
                text_x, text_y, ha, va = year + 0.6, capex_m - 0.16, "left", "top"
            elif bus == "p74":
                text_x, text_y, ha, va = year, capex_m + 0.08, "right", "bottom"
            else:
                text_x, text_y, ha, va = year, capex_m + 0.08, "right", "bottom"
            ax_c.text(
                text_x,
                text_y,
                f"{year:.0f} yr",
                color=color,
                fontsize=MAP_FS_TICK,
                ha=ha,
                va=va,
                zorder=5,
            )
    ax_c.axhline(capex_m, color="0.15", lw=0.8, zorder=1)
    ax_c.axvline(LIFE_YEARS, color="0.45", lw=0.6, ls=":", zorder=1)
    fastest = max(pair[0] for pair in annual.values())
    ymax = fastest * xmax / 1e6 * 1.06
    ax_c.legend(
        handles=[
            Line2D([0], [0], color="0.15", lw=0.8, label="Installation cost"),
            Line2D([0], [0], color="0.45", lw=0.6, ls=":", label="15-year life"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=MAP_FS_LAB,
        borderaxespad=0.15,
        handlelength=1.4,
        labelspacing=0.15,
    )
    ax_c.set_xlim(0, xmax)
    ax_c.set_ylim(0, ymax)
    ax_c.set_xlabel("Year", fontsize=MAP_FS_LAB, labelpad=1)
    ax_c.set_ylabel("Cumulative revenue (M$/MW)", fontsize=MAP_FS_LAB, labelpad=0)
    ax_c.set_title("c  Cumulative revenue", loc="left", fontsize=MAP_FS_TAG, fontweight="bold", pad=3)
    ax_c.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=0.4)
    ax_c.tick_params(axis="y", pad=0.15)
    ax_c.xaxis.set_major_locator(MaxNLocator(4))
    ax_c.yaxis.set_major_locator(MaxNLocator(4))

    path = save_figure(fig, OUT, pad_inches=0.02)
    print(f"[ok] {path}", flush=True)
    return path


def main() -> None:
    table = build_table() if not TABLE.is_file() else pd.read_csv(TABLE)
    if "data_centre_bus" not in table.columns:
        table = build_table()
    print(
        "median delta range",
        float(table["delta_intraday_range"].median()),
        "median delta value",
        float(table["delta_arbitrage_usd_per_mw_week"].median()),
        flush=True,
    )
    build_figure(table)


if __name__ == "__main__":
    main()
