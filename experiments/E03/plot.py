from __future__ import annotations
"""Fig. 3 and Supplementary Fig. 12."""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib import cm as mpl_cm

from experiments.lib.paths import package_root

ROOT = package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.figure_style import *  # noqa: F403
from experiments.lib.figure_common import *  # noqa: F403

OUT = ROOT / "experiments/E03/figures"

def _load_e01_master() -> list[dict]:
    with E01_MASTER.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_submission_manifest() -> dict:
    path = ROOT / "experiments" / "E03" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/build_submission_manifest.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _pinned_power(op: str, season: str, scenario: str) -> Path:
    """Resolve power CSV from submission manifest (never auto-latest by mtime)."""
    man = _load_submission_manifest()
    if not man.get("policy", {}).get("forbid_auto_latest", False):
        raise RuntimeError("submission manifest must set policy.forbid_auto_latest=true")
    key = f"{op}/{season}/{scenario}"
    entry = man.get("power_trajectories", {}).get(key)
    if not entry:
        raise KeyError(f"manifest missing power_trajectories[{key!r}]")
    rel = str(entry["path"]).replace("\\", "/")
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _latest_power(op: str, season: str, scenario: str) -> Path:
    """Deprecated alias kept for call sites; delegates to manifest pins."""
    return _pinned_power(op, season, scenario)


def _fleet_power(season: str, scenario: str, hours: int = 168) -> np.ndarray:
    total = np.zeros(hours)
    for op in OPERATOR_ORDER:
        sites = [int(s["site_id"]) for s in get_operator_sites(op)]
        rows = list(csv.DictReader(_latest_power(op, season, scenario).open(encoding="utf-8")))
        for t in range(min(hours, len(rows))):
            row = rows[t]
            if "Total_Power_MW" in row:
                total[t] += float(row["Total_Power_MW"])
            else:
                total[t] += sum(float(row[f"Site{sid}_Power_MW"]) for sid in sites)
    return total


def build_fig2() -> Path:
    apply_figure_style()
    plt.rcParams.update(
        {
            "font.size": FS,
            "axes.labelsize": FS,
            "axes.titlesize": FS_TITLE,
            "xtick.labelsize": FS_TICK_COMPOSED,
            "ytick.labelsize": FS_TICK_COMPOSED,
            "legend.fontsize": FS_LEGEND,
        }
    )
    rows = _load_e01_master()
    # Shorter board; flatter c/d; right pad so b/d are not flush-cut.
    fig = plt.figure(figsize=figsize_double_col(4.15), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.95],
        hspace=FIG2_HSPACE,
        wspace=FIG2_WSPACE,
        left=FIG2_LEFT,
        right=FIG2_RIGHT,
        top=FIG2_TOP,
        bottom=FIG2_BOTTOM,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_c)
    top_axes = [ax_a, ax_b]
    bot_axes = [ax_c, ax_d]

    xmin = 0.0
    for col, season in enumerate(("summer", "winter")):
        ax = top_axes[col]
        ops = list(OPERATOR_ORDER)
        y = np.arange(len(ops))
        left, right = [], []
        for op in ops:
            by = {r["scenario"]: r for r in rows if r["operator"] == op and r["season"] == season}
            left.append(0.0)
            right.append(
                float(by.get("reallocation_slack_termination", {}).get("delta_energy_cost_pct_vs_baseline", 0) or 0)
            )
        for i in range(len(ops)):
            ax.plot([left[i], right[i]], [y[i], y[i]], color="#8491B4", lw=1.0, zorder=1)
            ax.scatter(left[i], y[i], color=COLOR_FCFS, s=FIG2_DOT_S, zorder=2)
            ax.scatter(right[i], y[i], color=COLOR_FULL, s=FIG2_DOT_S, zorder=2)
        ax.axvline(0.0, color="#CCCCCC", lw=0.5)
        ax.set_yticks(y)
        if col == 0:
            ax.set_yticklabels([OPERATOR_LABELS[o] for o in ops], fontsize=FS)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xlabel(r"$\Delta$Energy cost (%)", fontsize=FS)
        ax.tick_params(axis="both", labelsize=FS_TICK_COMPOSED)
        ax.xaxis.grid(True, color="#D0D0D0", lw=0.45, zorder=0)
        ax.set_axisbelow(True)
        xmin = min(xmin, min(right))
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_FCFS, markersize=FIG2_LEGEND_MS, label=FCFS),
                Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_FULL, markersize=FIG2_LEGEND_MS, label=FULL),
            ],
            loc="upper left",
            frameon=False,
            fontsize=FS_LEGEND,
            handletextpad=0.25,
            borderaxespad=0.15,
            labelspacing=0.2,
        )
        _panel_tag(ax, "ab"[col], season.capitalize())
    for ax in top_axes:
        ax.set_xlim(xmin - 2.5, 1.5)

    for col, season in enumerate(("summer", "winter")):
        ax = bot_axes[col]
        fcfs = _fleet_power(season, "baseline")
        full = _fleet_power(season, "reallocation_slack_termination")
        t = np.arange(len(fcfs))
        ax.fill_between(t, fcfs, full, where=full >= fcfs, color=COLOR_FULL, alpha=0.15, interpolate=True)
        ax.fill_between(t, fcfs, full, where=full < fcfs, color=COLOR_FCFS, alpha=0.12, interpolate=True)
        ax.plot(t, fcfs, color=COLOR_FCFS, lw=0.9)
        ax.plot(t, full, color=COLOR_FULL, lw=0.9)
        ax.set_xlim(0, len(fcfs) - 1)
        ax.set_xlabel("Hour", fontsize=FS)
        if col == 0:
            # Pull the label toward panel c; default pad leaves a wide gap.
            ax.set_ylabel("Combined power (MW)", fontsize=FS, labelpad=-14)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.yaxis.grid(True, color="#D0D0D0", lw=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=FS_TICK_COMPOSED)
        _panel_tag(ax, "cd"[col], season.capitalize())

    handles = [
        Line2D([0], [0], color=COLOR_FCFS, lw=1.2, label=FCFS),
        Line2D([0], [0], color=COLOR_FULL, lw=1.2, label=FULL),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=FS_LEGEND,
        frameon=False,
        bbox_to_anchor=(0.54, 0.01),
        columnspacing=1.4,
    )
    fig.align_ylabels([ax_a, ax_c])
    out = OUT / "Fig2_job_level_flexibility.png"
    return save_figure(fig, out)


def _site_power_gw(op: str, season: str, scenario: str, sid: int, hours: int = 168) -> np.ndarray:
    rows = list(csv.DictReader(_pinned_power(op, season, scenario).open(encoding="utf-8")))
    return np.array([float(r[f"Site{sid}_Power_MW"]) for r in rows[:hours]], dtype=float) / 1e3


_SITE_COLORS = ("#111111", "#0072B2", "#D55E00")
_SITE_PREFIXES = ("CoreWeave ", "OpenAI Stargate ", "Google ", "Meta ", "Microsoft ", "Amazon ")


def _site_label(spec: dict) -> str:
    name = str(spec.get("name") or spec["site_id"])
    for prefix in _SITE_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace(" TX", "")


def build_extdata_site_trajectories() -> Path:
    """One panel per operator. Sites of that operator share the axes.

    Solid lines are Full flex and dashed lines are FCFS. Colour distinguishes
    sites. FCFS does not depend on the season. The largest one-hour step is
    at Shackelford.
    """
    apply_figure_style()
    plt.rcParams.update(
        {
            "font.size": FS,
            "axes.labelsize": FS,
            "xtick.labelsize": FS_TICK_COMPOSED,
            "ytick.labelsize": FS_TICK_COMPOSED,
            "legend.fontsize": FS_LEGEND,
        }
    )
    operators: list[tuple[str, list[tuple[int, str]]]] = []
    for op in OPERATOR_ORDER:
        sites = [
            (int(spec["site_id"]), _site_label(spec))
            for spec in get_operator_sites(op)
        ]
        operators.append((op, sites))

    n = len(operators)
    fig_w = float(figsize_double_col(1.0)[0])
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(fig_w, 6.35),
        sharex=True,
        sharey="row",
        facecolor="white",
        gridspec_kw={"hspace": 0.72, "wspace": 0.10},
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.05, top=0.88)

    hours = 168
    t = np.arange(hours)
    for r, (op, sites) in enumerate(operators):
        peak = 0.0
        for c, season in enumerate(("summer", "winter")):
            ax = axes[r, c]
            handles = []
            for i, (sid, label) in enumerate(sites):
                color = _SITE_COLORS[i]
                fcfs = _site_power_gw(op, "summer", "baseline", sid, hours)
                full = _site_power_gw(op, season, "reallocation_slack_termination", sid, hours)
                ax.plot(t, fcfs, color=color, lw=0.65, ls="--", zorder=2)
                ax.plot(t, full, color=color, lw=0.9, ls="-", zorder=3)
                handles.append(Line2D([0], [0], color=color, lw=1.1, label=label))
                peak = max(peak, float(fcfs.max()), float(full.max()))
            ax.set_xlim(0, hours - 1)
            ax.tick_params(axis="both", labelsize=FS_TICK_COMPOSED, length=2.0, pad=0.6)
            if c == 1:
                ax.tick_params(labelleft=False)
            if r == n - 1:
                ax.set_xlabel("Hour", fontsize=FS)
                ax.set_xticks([0, 84, 168])
            else:
                ax.tick_params(labelbottom=False)
            leg = ax.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                ncol=len(sites),
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=1.0,
                fontsize=6,
                title=OPERATOR_LABELS[op] if c == 0 else None,
                title_fontsize=FS,
                handlelength=1.15,
                handletextpad=0.3,
                columnspacing=0.7,
                borderaxespad=0.0,
                labelspacing=0.15,
            )
            leg.set_zorder(6)
            if c == 0:
                leg.get_title().set_fontweight("bold")
                leg._legend_box.align = "left"
        axes[r, 0].set_ylim(0.0, peak * 1.12 if peak > 0 else 1.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_left = []
    for ax in axes[:, 0]:
        for lab in ax.get_yticklabels():
            if not lab.get_text():
                continue
            bb = lab.get_window_extent(renderer=renderer)
            tick_left.append(fig.transFigure.inverted().transform((bb.x0, bb.y0))[0])
    y_top = axes[0, 0].get_position().y1
    y_bot = axes[-1, 0].get_position().y0
    x_power = (min(tick_left) if tick_left else 0.04) - 0.008
    fig.text(
        x_power,
        0.5 * (y_top + y_bot),
        "Power (GW)",
        rotation=90,
        va="center",
        ha="right",
        fontsize=FS,
    )
    fig.text(0.31, 0.955, "a  Summer", fontsize=FS_TITLE, fontweight="bold", ha="center", va="bottom")
    fig.text(0.77, 0.955, "b  Winter", fontsize=FS_TITLE, fontweight="bold", ha="center", va="bottom")
    fig.legend(
        handles=[
            Line2D([0], [0], color="0.2", lw=1.2, ls="-", label=FULL),
            Line2D([0], [0], color="0.2", lw=1.0, ls="--", label=FCFS),
        ],
        loc="upper center",
        ncol=2,
        fontsize=FS_LEGEND,
        frameon=False,
        bbox_to_anchor=(0.50, 0.998),
        columnspacing=1.4,
        handletextpad=0.35,
    )

    return save_figure(fig, OUT / "ExtData_Fig_texas_site_trajectories.png")



def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(build_fig2())
    print(build_extdata_site_trajectories())


if __name__ == "__main__":
    main()
