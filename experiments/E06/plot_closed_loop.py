from __future__ import annotations
"""Supplementary Fig. 8: closed-loop iteration residuals."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

from experiments.lib.figure_common import _panel_tag  # noqa: E402
from experiments.lib.figure_style import COLOR_FCFS, FS, FS_TITLE  # noqa: E402
from experiments.lib.figures.nc_style import apply_figure_style, figsize_double_col, save_figure  # noqa: E402

OUT = ROOT / "experiments/E06/figures/ExtData_Fig6_usa_endogenous_closed_loop.png"
SUMMARY = ROOT / "experiments/E06/results/closed_loop/iteration_summary_summer.csv"


def build(season: str = "summer") -> Path:
    path = ROOT / "experiments/E06/results/closed_loop" / f"iteration_summary_{season}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df_all = pd.read_csv(path)
    apply_figure_style()
    plt.rcParams.update(
        {
            "font.size": FS_TITLE,
            "axes.labelsize": FS,
            "axes.titlesize": FS_TITLE,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": FS,
        }
    )
    fig = plt.figure(figsize=figsize_double_col(2.55), facecolor="white")
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.0, 1.05, 1.05], wspace=0.46, left=0.08, right=0.98, top=0.90, bottom=0.16
    )
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    energy_all = df_all["total_dc_mwh"].to_numpy(dtype=float) / 1e6
    site_all = df_all["campus_mean_abs_delta_lmp"].to_numpy(dtype=float)
    cong_all = df_all["mean_n_congested"].to_numpy(dtype=float)
    df = df_all.loc[df_all["iter"] >= 1].reset_index(drop=True)
    iters = df["iter"].to_numpy()
    x_pad = 0.35

    def _centered(full: np.ndarray, shown: np.ndarray, margin: float) -> tuple[float, float]:
        lo, hi = float(np.min(full)), float(np.max(full))
        pad = margin * max(hi - lo, 1e-9)
        span = (hi + pad) - (lo - pad)
        mid = 0.5 * (float(np.min(shown)) + float(np.max(shown)))
        return mid - 0.5 * span, mid + 0.5 * span

    ax_a.plot(iters, df["total_dc_mwh"] / 1e6, color=COLOR_FCFS, lw=1.2, marker="o", markersize=3.5)
    ax_a.set_xlabel("Iteration")
    ax_a.set_ylabel("Total data center energy (TWh/week)", labelpad=4)
    ax_a.set_ylim(*_centered(energy_all, df["total_dc_mwh"].to_numpy(dtype=float) / 1e6, 0.05))
    ax_a.set_xlim(float(iters.min()) - x_pad, float(iters.max()) + x_pad)
    ax_a.set_xticks(iters)
    ax_a.tick_params(axis="y", pad=2)
    ax_a.yaxis.grid(True, color="#D0D0D0", lw=0.45, zorder=0)
    ax_a.set_axisbelow(True)
    _panel_tag(ax_a, "a", "Total energy")

    ax_b.plot(iters, df["campus_mean_abs_delta_lmp"], color=COLOR_FCFS, lw=1.2, marker="o", markersize=3.5)
    ax_b.set_xlabel("Iteration")
    ax_b.set_ylabel(r"Average $|\Delta\mathrm{LMP}|$", labelpad=4)
    ax_b.tick_params(axis="y", pad=2)
    ax_b.set_ylim(*_centered(site_all, df["campus_mean_abs_delta_lmp"].to_numpy(dtype=float), 0.08))
    ax_b.set_xlim(float(iters.min()) - x_pad, float(iters.max()) + x_pad)
    ax_b.set_xticks(iters)
    ax_b.yaxis.grid(True, color="#D0D0D0", lw=0.45, zorder=0)
    ax_b.set_axisbelow(True)
    _panel_tag(ax_b, "b", "Data center sites")

    ax_c.plot(iters, df["mean_n_congested"], color=COLOR_FCFS, lw=1.2, marker="o", markersize=3.5)
    ax_c.set_xlabel("Iteration")
    ax_c.set_ylabel("Average congested lines", labelpad=4)
    ax_c.set_ylim(*_centered(cong_all, df["mean_n_congested"].to_numpy(dtype=float), 0.05))
    ax_c.set_xlim(float(iters.min()) - x_pad, float(iters.max()) + x_pad)
    ax_c.set_xticks(iters)
    ax_c.tick_params(axis="y", pad=2)
    ax_c.yaxis.grid(True, color="#D0D0D0", lw=0.45, zorder=0)
    ax_c.set_axisbelow(True)
    _panel_tag(ax_c, "c", "Congestion")

    out = OUT if season == "summer" else OUT.with_name(f"ExtData_Fig_usa_endogenous_closed_loop_{season}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, out, pad_inches=0.08)


def main() -> None:
    print(build("summer"))


if __name__ == "__main__":
    main()
