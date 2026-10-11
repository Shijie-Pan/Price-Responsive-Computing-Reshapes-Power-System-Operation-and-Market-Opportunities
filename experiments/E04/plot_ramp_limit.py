from __future__ import annotations
"""Supplementary Fig. 4: ramp-limit sensitivity (95% + 24 h)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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

OUT = ROOT / "experiments/E04/figures/ExtData_Fig4_ramp_limit_sensitivity.png"
FLEET = ROOT / "experiments/E04/results/ramp_limit/fleet_metrics.csv"
CONG = ROOT / "experiments/E04/results/ramp_limit/congestion_vs_reference.csv"

CASES = ["A1", "B1_ramp50", "B1_ramp25", "B1_ramp10"]
LABELS = {
    "A1": "No ramp\nlimit",
    "B1_ramp50": "50%\nramp limit",
    "B1_ramp25": "25%\nramp limit",
    "B1_ramp10": "10%\nramp limit",
}
SUMMER = "#E64B35"
WINTER = "#4C72B0"


def build() -> Path:
    if not FLEET.is_file() or not CONG.is_file():
        raise FileNotFoundError(f"missing deposited ramp-limit tables under {FLEET.parent}")
    fleet = pd.read_csv(FLEET)
    cong = pd.read_csv(CONG)
    rows = []
    for season in ("summer", "winter"):
        for case in CASES:
            f = fleet[(fleet["case"] == case) & (fleet["season"] == season)]
            c = cong[(cong["case"] == case) & (cong["season"] == season)]
            if f.empty or c.empty or int(f["n_operators"].iloc[0]) < 6:
                raise RuntimeError(f"incomplete {case} {season}")
            unit_chg = float(f["unit_cost_change_vs_a0_pct"].iloc[0])
            e0 = float(f["energy_mwh_0_167"].iloc[0])
            e24 = float(f["energy_mwh_168_191"].iloc[0])
            rows.append(
                {
                    "case": case,
                    "season": season,
                    "unit_cost_reduction_vs_price_independent_pct": -unit_chg,
                    "energy_mwh_0_191": e0 + e24,
                    "net_congested_line_hours_vs_price_independent": float(c["net_vs_A0"].iloc[0]),
                    "added_congested_line_hours_vs_price_independent": float(c["added_vs_A0"].iloc[0]),
                    "relieved_congested_line_hours_vs_price_independent": float(c["relieved_vs_A0"].iloc[0]),
                }
            )
    table = pd.DataFrame(rows)

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
    fig, axes = plt.subplots(2, 2, figsize=figsize_double_col(5.2), sharex=False)
    x = np.arange(len(CASES))
    tick_labels = [LABELS[c] for c in CASES]

    for season, color in (("summer", SUMMER), ("winter", WINTER)):
        sub = table[table["season"] == season].set_index("case").loc[CASES]
        axes[0, 0].plot(x, sub["unit_cost_reduction_vs_price_independent_pct"], marker="o", color=color)
        axes[0, 1].plot(x, sub["energy_mwh_0_191"] / 1e3, marker="o", color=color)
        axes[1, 0].plot(x, sub["net_congested_line_hours_vs_price_independent"], marker="o", color=color)

    width = 0.18
    offsets = {
        ("summer", "added"): -1.5 * width,
        ("summer", "relieved"): -0.5 * width,
        ("winter", "added"): 0.5 * width,
        ("winter", "relieved"): 1.5 * width,
    }
    for season, color in (("summer", SUMMER), ("winter", WINTER)):
        sub = table[table["season"] == season].set_index("case").loc[CASES]
        axes[1, 1].bar(
            x + offsets[(season, "added")],
            sub["added_congested_line_hours_vs_price_independent"],
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )
        axes[1, 1].bar(
            x + offsets[(season, "relieved")],
            -sub["relieved_congested_line_hours_vs_price_independent"],
            width=width,
            color=color,
            alpha=0.45,
            edgecolor="black",
            linewidth=0.3,
            hatch="///",
        )

    panel_titles = {
        (0, 0): "a  Unit electricity-cost reduction vs price-independent scheduling",
        (0, 1): "b  Total energy, hours 0–167 + 24 h extension",
        (1, 0): "c  Net congested line-hours vs price-independent scheduling",
        (1, 1): "d  Added / relieved congested line-hours vs price-independent",
    }
    ylabels = {
        (0, 0): "Unit cost reduction (%)",
        (0, 1): "Total energy (GWh)",
        (1, 0): "Net congested line-hours",
        (1, 1): "Congested line-hours",
    }
    for key, title in panel_titles.items():
        ax = axes[key]
        ax.set_title(title, loc="left", fontsize=FS_TITLE, fontweight="bold", pad=3)
        ax.set_ylabel(ylabels[key], fontsize=FS)
        ax.set_xticks(x, tick_labels)
        ax.tick_params(axis="x", labelbottom=True)

    axes[1, 0].axhline(0, color="0.5", lw=0.6)
    axes[1, 1].axhline(0, color="0.5", lw=0.6)
    handles = [
        Line2D([0], [0], color=SUMMER, marker="o", lw=1.2, label="Summer"),
        Line2D([0], [0], color=WINTER, marker="o", lw=1.2, label="Winter"),
        Patch(facecolor=SUMMER, edgecolor="black", linewidth=0.3, label="Summer added congestion"),
        Patch(
            facecolor=SUMMER,
            alpha=0.45,
            hatch="///",
            edgecolor="black",
            linewidth=0.3,
            label="Summer relieved congestion",
        ),
        Patch(facecolor=WINTER, edgecolor="black", linewidth=0.3, label="Winter added congestion"),
        Patch(
            facecolor=WINTER,
            alpha=0.45,
            hatch="///",
            edgecolor="black",
            linewidth=0.3,
            label="Winter relieved congestion",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        columnspacing=1.2,
        handlelength=1.4,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.88)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, OUT)


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
