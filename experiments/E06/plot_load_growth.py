from __future__ import annotations
"""Supplementary Fig. 10: load-growth battery arbitrage."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

from experiments.lib.figure_style import FS, FS_TITLE, FS_TICK_COMPOSED  # noqa: E402
from experiments.lib.figures.nc_style import apply_figure_style, figsize_double_col, save_figure  # noqa: E402

OUT = ROOT / "experiments/E06/figures/ExtData_Fig6_load_growth_battery_arbitrage.png"
SUMMARY = ROOT / "experiments/E06/results/load_growth/summary_fixed_sample.csv"

SCENARIOS = ("ordinary_low_price_siting", "dc_fixed_profile_scaling")
LABELS = {
    "ordinary_low_price_siting": "Ordinary load: low-price siting",
    "dc_fixed_profile_scaling": "Data centers: fixed-profile scaling",
}
COLORS = {
    "ordinary_low_price_siting": "#4C72B0",
    "dc_fixed_profile_scaling": "#E64B35",
}


def build() -> Path:
    if not SUMMARY.is_file():
        raise FileNotFoundError(SUMMARY)
    summary = pd.read_csv(SUMMARY)
    apply_figure_style()
    plt.rcParams.update(
        {
            "font.size": FS_TITLE,
            "axes.labelsize": FS_TITLE,
            "xtick.labelsize": FS_TICK_COMPOSED,
            "ytick.labelsize": FS_TICK_COMPOSED,
            "legend.fontsize": FS,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize_double_col(3.2), sharex=True)
    titles = (
        "a  Mean battery-arbitrage change vs zero-growth baseline",
        "b  Median battery-arbitrage change vs zero-growth baseline",
    )
    ycols = ("mean_change_vs_stage0_pct", "median_change_vs_stage0_pct")
    ylabels = ("Mean arbitrage change (%)", "Median arbitrage change (%)")
    for ax, title, ycol, ylab in zip(axes, titles, ycols, ylabels):
        for scenario in SCENARIOS:
            g = summary[summary["scenario"] == scenario].sort_values("stage")
            ax.plot(
                g["cumulative_added_twh_per_week"],
                g[ycol],
                marker="o",
                ms=3.5,
                lw=1.2,
                color=COLORS[scenario],
                label=LABELS[scenario],
            )
        ax.axhline(0.0, color="0.5", lw=0.7, zorder=0)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=FS_TITLE, pad=3)
        ax.set_xlabel("Added electricity consumption over 168 h (TWh)", fontsize=FS)
        ax.set_ylabel(ylab, fontsize=FS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    y_leg = axes[0].get_position().y1 + 0.028
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, y_leg),
        columnspacing=1.2,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, OUT)


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
