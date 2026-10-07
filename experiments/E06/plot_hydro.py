#!/usr/bin/env python3
"""Pumped-storage arbitrage on the same summer prices as the battery figure.

Baseline prices are the open-loop price-independent nodal duals. Comparison
prices are the closed-loop prices after six scheduling updates. The plant
takes those prices as given and is not added to network demand.

The unit is 1 MW with 10 MWh of storage, the 10-hour duration tabulated in
the 2025 Electricity ATB. Round-trip efficiency is 80%, the 2025 ATB central
value. Charge and discharge efficiencies are equal. Storage starts and ends
at half of capacity, and pumping and generation do not overlap.

Capital cost in the payback panel is the median of the 15 equal-capacity
closed-loop classes for 10-hour storage: 3,706 $/kW in 2021$, ATB class 8
(Electricity ATB 2025). Panel c uses observed day-ahead prices, scaled for
the price-responsive case by the dual intraday-range ratio. It does not
compare the plant with the battery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def _package_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "experiments" / "data" / "networks").is_dir() and (
            candidate / "experiments" / "lib"
        ).is_dir():
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
OUT = ROOT / "experiments/E06/figures/ExtData_Fig6_hydro_arbitrage.png"
TABLE = ROOT / "experiments/E06/results/summer_hydro_vs_price_independent.csv"
PAYBACK_JSON = ROOT / "experiments/E06/results/summer_hydro_observed_payback.json"
HOURS = 168
HOURS_PER_DAY = 24
WEEKS_PER_YEAR = 52

POWER_MW = 1.0
DURATION_H = 10.0
ENERGY_MWH = POWER_MW * DURATION_H
SOC0 = 0.5 * ENERGY_MWH
RTE = 0.80
ETA = RTE ** 0.5
# 2025 Electricity ATB, closed-loop 10-hour class 8 (median of 15 equal-capacity classes), 2021$.
CAPEX_USD_PER_KW = 3706.0
CAPEX_USD_PER_MW = CAPEX_USD_PER_KW * 1000.0

TRACES = (
    ("p62", "Texas", "#E64B35"),
    ("p74", "Michigan", "#3C5488"),
    ("p11", "California", "#009E73"),
)


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


def _intraday_range(prices: np.ndarray) -> float:
    days = prices.reshape(-1, HOURS_PER_DAY)
    return float(np.mean(days.max(axis=1) - days.min(axis=1)))


def _symmetric_limit(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    limit = float(np.max(np.abs(finite)))
    if limit <= 0:
        return 1.0
    return limit


def _hydro_schedule(prices: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Optimal weekly value ($), net power (MW), and stored energy (MWh)."""
    import gurobipy as gp

    horizon = len(prices)
    model = gp.Model("bus_pumped_storage")
    model.Params.OutputFlag = 0
    pump = model.addVars(horizon, lb=0.0, ub=POWER_MW, name="pump")
    generate = model.addVars(horizon, lb=0.0, ub=POWER_MW, name="gen")
    pumping = model.addVars(horizon, vtype=gp.GRB.BINARY, name="pumping")
    stored = model.addVars(horizon + 1, lb=0.0, ub=ENERGY_MWH, name="stored")
    model.addConstr(stored[0] == SOC0)
    model.addConstr(stored[horizon] == SOC0)
    for t in range(horizon):
        model.addConstr(pump[t] <= POWER_MW * pumping[t])
        model.addConstr(generate[t] <= POWER_MW * (1 - pumping[t]))
        model.addConstr(stored[t + 1] == stored[t] + ETA * pump[t] - generate[t] / ETA)
    model.setObjective(
        gp.quicksum(float(prices[t]) * (generate[t] - pump[t]) for t in range(horizon)),
        gp.GRB.MAXIMIZE,
    )
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"pumped storage status {model.Status}")
    net = np.array([generate[t].X - pump[t].X for t in range(horizon)], float)
    stored_mwh = np.array([stored[t].X for t in range(horizon)], float)
    return float(model.ObjVal), net, stored_mwh


def build_table() -> pd.DataFrame:
    pi = _bus_frame(PI_CSV)
    pr = _bus_frame(PR_CSV)
    buses = [c for c in pi.columns if c in pr.columns]
    geo = pd.read_csv(GEO_CSV).set_index("bus")
    rows = []
    schedules: dict[tuple[str, str], np.ndarray] = {}
    for i, bus in enumerate(buses):
        pi_p = pi[bus].to_numpy(float)
        pr_p = pr[bus].to_numpy(float)
        if not (np.isfinite(pi_p).all() and np.isfinite(pr_p).all()):
            continue
        pi_value, _pi_net, pi_stored = _hydro_schedule(pi_p)
        pr_value, _pr_net, pr_stored = _hydro_schedule(pr_p)
        if bus in {name for name, _label, _color in TRACES}:
            schedules[(bus, "pi")] = pi_stored
            schedules[(bus, "pr")] = pr_stored
        rows.append(
            {
                "bus": bus,
                "lon": float(geo.loc[bus, "lon"]),
                "lat": float(geo.loc[bus, "lat"]),
                "pi_arbitrage_usd_per_mw_week": pi_value / POWER_MW,
                "pr_arbitrage_usd_per_mw_week": pr_value / POWER_MW,
                "delta_arbitrage_usd_per_mw_week": (pr_value - pi_value) / POWER_MW,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"[hydro] {i + 1}/{len(buses)}", flush=True)
    out = pd.DataFrame(rows)
    meta = json.loads(DC_META.read_text(encoding="utf-8"))
    dc_buses = {str(spec["pypsa_bus"]) for spec in meta["sites"].values()}
    out["data_centre_bus"] = out["bus"].isin(dc_buses)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE, index=False, float_format="%.6f")
    np.savez(
        TABLE.with_suffix(".npz"),
        **{f"{bus}_{which}": series for (bus, which), series in schedules.items()},
    )
    print(f"[hydro] wrote {TABLE}", flush=True)
    return out


def _load_schedules() -> dict[tuple[str, str], np.ndarray]:
    path = TABLE.with_suffix(".npz")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(path)
    out = {}
    for bus, _label, _color in TRACES:
        out[(bus, "pi")] = payload[f"{bus}_pi"]
        out[(bus, "pr")] = payload[f"{bus}_pr"]
    return out


def _annual_hydro_revenue(prices: np.ndarray) -> float:
    """Mean weekly arbitrage extended to 52 weeks, in $/MW/year."""
    n = len(prices) // HOURS * HOURS
    weeks = [_hydro_schedule(prices[i : i + HOURS])[0] for i in range(0, n, HOURS)]
    return float(np.mean(weeks)) * WEEKS_PER_YEAR


def _payback() -> dict[str, tuple[float, float, float]]:
    """Observed-price revenue. The dual enters only as the intraday-range ratio."""
    if PAYBACK_JSON.is_file():
        saved = json.loads(PAYBACK_JSON.read_text(encoding="utf-8"))
        return {bus: (row["pi_annual"], row["pr_annual"], row["keep"]) for bus, row in saved.items()}
    pi = _bus_frame(PI_CSV)
    pr = _bus_frame(PR_CSV)
    out = {}
    saved = {}
    for bus, name, _color in TRACES:
        keep = _intraday_range(pr[bus].to_numpy(float)) / _intraday_range(pi[bus].to_numpy(float))
        pi_annual = _annual_hydro_revenue(_observed_prices(name))
        pr_annual = pi_annual * keep
        out[bus] = (pi_annual, pr_annual, keep)
        saved[bus] = {"name": name, "pi_annual": pi_annual, "pr_annual": pr_annual, "keep": keep}
        print(
            f"[hydro payback] {name} range kept {keep * 100:.1f}%  "
            f"{CAPEX_USD_PER_MW / pi_annual:.1f} -> {CAPEX_USD_PER_MW / pr_annual:.1f} yr",
            flush=True,
        )
    PAYBACK_JSON.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return out


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
    schedules = _load_schedules()
    payback = _payback()

    fig_w = float(figsize_double_col(1.0)[0])
    left, right, bottom, top = 0.048, 0.992, 0.145, 0.90
    fig_h = 1.80 * 0.85 + 0.02
    dx = float(CONUS_XLIM[1] - CONUS_XLIM[0])
    dy = float(CONUS_YLIM[1] - CONUS_YLIM[0])
    map_h_in = fig_h * (top - bottom)
    cb_fraction, cb_pad = 0.046, 0.03
    map_w_in = map_h_in * dx / dy
    map_cell_in = map_w_in / (1.0 - cb_fraction - cb_pad)
    freed = 0.10 * map_cell_in
    map_cell_in -= freed
    gap_ab, gap_bc = 0.40, 0.24
    mid_w_in = 1.52 + 0.5 * freed
    x_in = left * fig_w
    c_w = right * fig_w - (x_in + map_cell_in + gap_ab + mid_w_in + gap_bc)
    c_cut = 0.10 * c_w
    gap_ab += 0.5 * c_cut
    gap_bc += 0.5 * c_cut
    c_w -= c_cut
    span = top - bottom
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def _axes_at(x0: float, width: float):
        return fig.add_axes([x0 / fig_w, bottom, width / fig_w, span])

    ax_a = _axes_at(x_in, map_cell_in)
    x_in += map_cell_in + gap_ab
    ax_b = _axes_at(x_in, mid_w_in)
    x_in += mid_w_in + gap_bc
    ax_c = _axes_at(x_in, c_w)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f7f7f7")
    vals = table["delta_arbitrage_usd_per_mw_week"].to_numpy(float)
    limit = _symmetric_limit(vals)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    draw_base_map(ax_a, geo, {}, alpha=0.95, default_color="#ececec")
    grid_x, grid_y, field = _interpolate_bus_field(
        table["lon"].to_numpy(float),
        table["lat"].to_numpy(float),
        vals,
        geo,
        grid_n=220,
    )
    draw_conus_borders(ax_a, geo, zorder=3)
    ax_a.contourf(
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
    for bus, _name, color in TRACES:
        row = table.loc[table["bus"] == bus].iloc[0]
        ax_a.scatter(
            [row["lon"]],
            [row["lat"]],
            s=18,
            c=color,
            marker="o",
            linewidths=0.4,
            edgecolors="0.1",
            zorder=7,
        )
    ax_a.set_xlim(*CONUS_XLIM)
    ax_a.set_ylim(*CONUS_YLIM)
    ax_a.set_aspect("auto")
    ax_a.xaxis.set_major_locator(MaxNLocator(3))
    ax_a.yaxis.set_major_locator(MaxNLocator(4))
    ax_a.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=0.4)
    ax_a.tick_params(axis="y", pad=0.15)
    ax_a.set_ylabel("Latitude", fontsize=MAP_FS_LAB, labelpad=0)
    ax_a.set_xlabel("Longitude", fontsize=MAP_FS_LAB, labelpad=1)
    ax_a.set_title("a  Pumped-storage value change", loc="left", fontsize=MAP_FS_TAG, fontweight="bold", pad=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_a, fraction=0.046, pad=0.03)
    cb.set_label("Value change ($/MW/week)", fontsize=MAP_FS_LAB, labelpad=2)
    cb.ax.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=1.0)

    hours = np.arange(HOURS)
    for bus, name, color in TRACES:
        ax_b.plot(hours, schedules[(bus, "pr")], color=color, lw=1.05, ls="-", zorder=3)
        ax_b.plot(hours, schedules[(bus, "pi")], color=color, lw=0.9, ls="--", zorder=2)
    ax_b.set_xlim(0, HOURS - 1)
    ax_b.set_ylim(-0.8, 16.6)
    ax_b.set_xlabel("Hour", fontsize=MAP_FS_LAB, labelpad=1)
    ax_b.set_ylabel("Stored energy (MWh)", fontsize=MAP_FS_LAB, labelpad=0)
    ax_b.set_title("b  Reservoir energy", loc="left", fontsize=MAP_FS_TAG, fontweight="bold", pad=3)
    ax_b.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=0.4)
    ax_b.tick_params(axis="y", pad=0.15)
    ax_b.xaxis.set_major_locator(MaxNLocator(5))
    ax_b.yaxis.set_major_locator(MaxNLocator(5))
    ax_b.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1.2, label=name) for _bus, name, color in TRACES
        ]
        + [
            Line2D([0], [0], color="0.2", lw=0.9, ls="--", label="Price-independent"),
            Line2D([0], [0], color="0.2", lw=1.05, ls="-", label="Price-responsive"),
        ],
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

    capex_m = CAPEX_USD_PER_MW / 1e6
    slowest = max(CAPEX_USD_PER_MW / min(pair[:2]) for pair in payback.values())
    xmax = slowest * 1.08
    years = np.linspace(0.0, xmax, 400)
    fastest = max(pair[0] for pair in payback.values())
    for bus, _name, color in TRACES:
        pi_annual, pr_annual, _keep = payback[bus]
        ax_c.plot(years, pr_annual * years / 1e6, color=color, lw=1.05, ls="-", zorder=3)
        ax_c.plot(years, pi_annual * years / 1e6, color=color, lw=0.9, ls="--", zorder=2)
        pi_year = CAPEX_USD_PER_MW / pi_annual
        for revenue, which in ((pi_annual, "pi"), (pr_annual, "pr")):
            year = CAPEX_USD_PER_MW / revenue
            if which == "pr" and abs(year - pi_year) < 1.5:
                continue
            ax_c.plot([year], [capex_m], marker="o", ms=3.2, color=color, ls="none", zorder=4)
            # A few points off that crossing. 31 sits left of its dot, 34 sits right of its dot.
            if bus == "p62" and which == "pi":
                dx, dy, ha, va = 0, 6, "center", "bottom"
            elif bus == "p62" and which == "pr":
                dx, dy, ha, va = 2, -7, "left", "top"
            elif bus == "p74" and which == "pi":
                dx, dy, ha, va = -1, -7, "right", "top"
            elif bus == "p74" and which == "pr":
                dx, dy, ha, va = 0, 7, "center", "bottom"
            else:
                dx, dy, ha, va = 0, 6, "center", "bottom"
            ax_c.annotate(
                f"{year:.0f} yr",
                xy=(year, capex_m),
                xytext=(dx, dy),
                textcoords="offset points",
                color=color,
                fontsize=MAP_FS_TICK,
                ha=ha,
                va=va,
                zorder=5,
            )
    ax_c.axhline(capex_m, color="0.15", lw=0.8, zorder=1)
    ax_c.legend(
        handles=[
            Line2D([0], [0], color="0.15", lw=0.8, label="Installation cost"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=MAP_FS_LAB,
        borderaxespad=0.15,
        handlelength=1.4,
        labelspacing=0.15,
    )
    ax_c.set_xlim(0, xmax)
    ax_c.set_ylim(0, fastest * xmax / 1e6 * 1.06)
    ax_c.set_xlabel("Year", fontsize=MAP_FS_LAB, labelpad=1)
    ax_c.set_ylabel("Cumulative revenue (M$/MW)", fontsize=MAP_FS_LAB, labelpad=0)
    ax_c.set_title("c  Cumulative revenue", loc="left", fontsize=MAP_FS_TAG, fontweight="bold", pad=3)
    ax_c.tick_params(labelsize=MAP_FS_TICK, length=1.0, pad=0.4)
    ax_c.tick_params(axis="y", pad=0.15)
    ax_c.xaxis.set_major_locator(MaxNLocator(4))
    ax_c.yaxis.set_major_locator(MaxNLocator(4))
    print("[hydro] median delta", float(table["delta_arbitrage_usd_per_mw_week"].median()), flush=True)

    path = save_figure(fig, OUT, pad_inches=0.02)
    print(f"[ok] {path}", flush=True)
    return path


def _trace_schedules() -> None:
    """Store reservoir energy at the three mapped buses. The value table is unchanged."""
    pi = _bus_frame(PI_CSV)
    pr = _bus_frame(PR_CSV)
    schedules = {}
    for bus, _label, _color in TRACES:
        _value, _net, stored = _hydro_schedule(pi[bus].to_numpy(float))
        schedules[(bus, "pi")] = stored
        _value, _net, stored = _hydro_schedule(pr[bus].to_numpy(float))
        schedules[(bus, "pr")] = stored
    np.savez(
        TABLE.with_suffix(".npz"),
        **{f"{bus}_{which}": series for (bus, which), series in schedules.items()},
    )


def main() -> None:
    if TABLE.is_file():
        table = pd.read_csv(TABLE)
        if not TABLE.with_suffix(".npz").is_file():
            _trace_schedules()
    else:
        table = build_table()
    build_figure(table)


if __name__ == "__main__":
    main()
