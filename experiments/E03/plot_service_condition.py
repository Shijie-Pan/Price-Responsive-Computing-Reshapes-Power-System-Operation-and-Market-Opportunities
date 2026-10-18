from __future__ import annotations
"""Supplementary Fig. 3: service-condition electricity cost."""

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

from experiments.lib.figure_style import FS, FS_TITLE, FS_TICK_COMPOSED  # noqa: E402
from experiments.lib.figures.nc_style import apply_figure_style, figsize_double_col, save_figure  # noqa: E402

OUT = ROOT / "experiments/E03/figures/ExtData_Fig3_service_condition_cost.png"
TABLE = ROOT / "experiments/E03/results/service_condition/operator_metrics.csv"
FLEET = ROOT / "experiments/E03/results/service_condition/fleet_metrics.csv"


def build() -> Path:
    table = pd.read_csv(TABLE)
    fleet = pd.read_csv(FLEET)
    need = {"A0", "A1", "A2"}
    if not need.issubset(set(fleet["case"])) or (fleet[fleet["case"].isin(need)]["n_operators"] < 6).any():
        raise RuntimeError("service-condition table incomplete for A0/A1/A2")

    apply_figure_style()
    plt.rcParams.update(
        {
            "font.size": FS_TITLE,
            "axes.labelsize": FS_TITLE,
            "xtick.labelsize": FS_TICK_COMPOSED,
            "ytick.labelsize": FS_TICK_COMPOSED,
        }
    )
    op_order = list("ABCDEF")
    case_order = ("A0", "A1", "A2")
    colors = {"A0": "#4C72B0", "A1": "#E64B35", "A2": "#009E73"}
    labels = {"A0": "Price-independent", "A1": "95% + 24 h", "A2": "100%, no extension"}
    fig, axes = plt.subplots(2, 2, figsize=figsize_double_col(5.4), sharex="col")
    for col, season in enumerate(("summer", "winter")):
        sub = table[(table["season"] == season) & (table["case"].isin(case_order))]
        x = np.arange(len(op_order))
        width = 0.24
        for ax, metric, ylab in (
            (axes[0, col], "electricity_expenditure_usd", "Electricity expenditure ($)"),
            (axes[1, col], "usd_per_delivered_cpu_hour", "Expenditure per delivered CPU-hour ($)"),
        ):
            for i, case in enumerate(case_order):
                vals = []
                for op in op_order:
                    hit = sub[(sub["operator"] == op) & (sub["case"] == case)]
                    vals.append(float(hit[metric].iloc[0]) if len(hit) else np.nan)
                ax.bar(
                    x + (i - 1) * width,
                    vals,
                    width=width,
                    color=colors[case],
                    label=labels[case] if col == 0 and ax is axes[0, 0] else None,
                )
            if col == 0:
                ax.set_ylabel(ylab, fontsize=FS)
        axes[1, col].set_xticks(x, [f"Operator {op}" for op in op_order], rotation=30, ha="right")
    panel = {
        (0, 0): "a  Summer",
        (0, 1): "b  Winter",
        (1, 0): "c  Summer",
        (1, 1): "d  Winter",
    }
    for key, title in panel.items():
        axes[key].set_title(title, loc="left", fontweight="bold", fontsize=FS_TITLE)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    y_leg = axes[0, 0].get_position().y1 + 0.035
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, y_leg),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, OUT)


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
