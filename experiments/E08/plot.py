from __future__ import annotations
"""Fig. 7 aggregate and job-level schedules."""

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
from matplotlib import patheffects as mpe


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

from experiments.lib.figure_style import *  # noqa: F403
from experiments.lib.figure_common import *  # noqa: F403

OUT = ROOT / "experiments/E08/figures"

def _idle_facility_mw() -> dict[int, float]:
    """Facility idle power, PUE times the configured IT idle, as in the MILP."""
    idle: dict[int, float] = {}
    for site in load_texas7_config()["sites"]:
        idle[int(site["site_id"])] = (
            float(site["pue"]) * _PFIXED_RATIO * float(site["projected_total_mw"])
        )
    return idle


def _fig4_expenditure_musd(season: str, payload: dict) -> dict[str, float]:
    """Operator electricity expenditure for Fig. 4, in $M, on the first 168 h.

    Every schedule is billed as price times facility power for t < 168.
    The job-level source trajectory has 192 h; later hours are omitted.
    Aggregate schedules are already 168 h and are unchanged.
    """
    original_hours = 168
    sites = [int(s) for s in payload["sites"]]
    idle = _idle_facility_mw()
    try:
        prices = {
            s: load_hourly_prices(get_site_lmp_csv(s, season), original_hours + 24)
            for s in sites
        }
    except FileNotFoundError:
        # Deposited schedules stay at or above facility idle, so the idle split
        # below equals price times scheduled power. That sum is already stored.
        return {
            model: float(payload["models"][model]["energy_cost_usd"]) / 1e6
            for model in ("FULL_JOB", "AGG_ENERGY", "AGG_POOL", "AGG_CONT")
        }
    stems = {
        "FULL_JOB": "power_full_job",
        "AGG_ENERGY": "power_agg_energy",
        "AGG_POOL": "power_agg_pool",
        "AGG_CONT": "power_agg_cont",
    }
    out: dict[str, float] = {}
    for model, stem in stems.items():
        if model == "FULL_JOB":
            path = ROOT / str(payload["models"]["FULL_JOB"]["source_csv"])
        else:
            path = E03_RES / season / f"{stem}_{season}.csv"
        frame = pd.read_csv(path)
        total = 0.0
        for s in sites:
            series = frame[f"Site{s}_Power_MW"].to_numpy(dtype=float)
            lam = prices[s]
            idle_s = idle[s]
            for t, power in enumerate(series):
                if t >= original_hours:
                    break
                total += float(lam[t]) * max(0.0, float(power) - idle_s)
                total += float(lam[t]) * idle_s
        out[model] = total / 1e6
    return out


def build_fig4(*, include_agg_cont: bool = True) -> Path:
    """Aggregate-model error.

    a  Summer–winter means. Expenditure is price times facility power
       on the original 168 h for every model. The job-level trajectory
       is longer; hours at or after 168 are omitted.
       Peak is the combined fleet peak. Ramp and mean absolute error are
       reported for Operator A (Pecos) and for the combined fleet. MAE is
       MAE = (1/T) sum_t |P^AGG_t − P^job_t| on the corresponding curve.
    b  Combined fleet power, summer.
    c  Summer power at Operator A (Pecos), the site where AGG Site departs
       most from the job-level trace.
    """
    apply_figure_style()
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_TICK_COMPOSED, "ytick.labelsize": FS_TICK_COMPOSED, "legend.fontsize": FS_LEGEND})
    summer = json.loads(E03_SUMMER.read_text(encoding="utf-8"))
    winter = json.loads(E03_WINTER.read_text(encoding="utf-8"))

    models_all = ["FULL_JOB", "AGG_ENERGY", "AGG_POOL", "AGG_CONT"]
    models = models_all if include_agg_cont else ["FULL_JOB", "AGG_ENERGY", "AGG_POOL"]
    labels = {
        "FULL_JOB": "Job-level",
        "AGG_ENERGY": "AGG Site",
        "AGG_POOL": "AGG Operator",
        "AGG_CONT": "AGG Continuous",
    }
    colors = {
        "FULL_JOB": "#111111",
        "AGG_ENERGY": "#D62728",
        "AGG_POOL": "#1F77B4",
        "AGG_CONT": "#FF7F0E",
    }
    # Operator A / Pecos: largest summer |AGG Site − job-level| among sites.
    focus_site = 751
    focus_site_label = "Operator A (Pecos)"

    def avg(key: str, metric: str) -> float:
        return 0.5 * (
            float(summer["models"][key].get(metric) or 0)
            + float(winter["models"][key].get(metric) or 0)
        )

    def _power(season: str, model: str, column: str, hours: int = 168) -> np.ndarray:
        stem = {
            "FULL_JOB": "power_full_job",
            "AGG_ENERGY": "power_agg_energy",
            "AGG_POOL": "power_agg_pool",
            "AGG_CONT": "power_agg_cont",
        }[model]
        df = pd.read_csv(E03_RES / season / f"{stem}_{season}.csv")
        return df[column].to_numpy(dtype=float)[:hours]

    hours = 168
    combined = {
        season: {m: _power(season, m, "Total_Power_MW", hours) for m in models_all}
        for season in ("summer", "winter")
    }
    site_traces = {
        season: {m: _power(season, m, f"Site{focus_site}_Power_MW", hours) for m in models_all}
        for season in ("summer", "winter")
    }

    def _ramp_mae_gw(traces: dict[str, dict[str, np.ndarray]], model: str) -> tuple[float, float]:
        ramps, maes = [], []
        for season in ("summer", "winter"):
            y = traces[season][model]
            y_job = traces[season]["FULL_JOB"]
            ramps.append(float(np.max(np.abs(np.diff(y)))))
            maes.append(float(np.mean(np.abs(y - y_job))))
        return float(np.mean(ramps)) / 1e3, float(np.mean(maes)) / 1e3

    # Common 168 h bill: price times facility power. Job-level hours at
    # or after 168 are omitted. Aggregate schedules are already 168 h.
    expenditure_m = {
        season: _fig4_expenditure_musd(season, payload)
        for season, payload in (("summer", summer), ("winter", winter))
    }
    # Three-line ticks (metric / scope / unit) widen each category column
    # so horizontal bar annotations have room.
    xlabels = [
        "Operator electricity\nexpenditure\n($M)",
        "Peak\n(combined)\nGW",
        f"Ramp\n({focus_site_label})\nGW/h",
        "Ramp\n(combined)\nGW/h",
        f"MAE\n({focus_site_label})\nGW",
        "MAE\n(combined)\nGW",
    ]
    vals_by_model = {}
    for m in models_all:
        site_ramp, site_mae = _ramp_mae_gw(site_traces, m)
        comb_ramp, comb_mae = _ramp_mae_gw(combined, m)
        vals_by_model[m] = [
            0.5 * (expenditure_m["summer"][m] + expenditure_m["winter"][m]),
            avg(m, "peak_total_mw") / 1e3,
            site_ramp,
            comb_ramp,
            site_mae,
            comb_mae,
        ]
    job_vals = vals_by_model["FULL_JOB"]

    fig = plt.figure(figsize=figsize_double_col(4.70), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.32, 1.0],
        hspace=0.30,
        wspace=0.28,
        left=FIG4_LEFT,
        right=FIG4_RIGHT,
        top=0.915,
        bottom=0.055,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    x = np.arange(len(xlabels))
    n_models = len(models)
    w = 0.15 if n_models == 4 else 0.22
    offset_center = (n_models - 1) / 2.0
    agg_models = [m for m in models if m != "FULL_JOB"]
    label_fs = max(FS - 1, 6)
    # Horizontal labels, one per AGG model (never merged). Sit just above
    # each bar's own top (e.g. orange AGG Cont stays on its short bar).
    label_y0_pts = 3.0
    label_dy_pts = 11.0
    label_stroke = [mpe.withStroke(linewidth=2.5, foreground="white")]
    bar_artists: dict[str, list] = {}
    for i, m in enumerate(models):
        vals = vals_by_model[m]
        bars = ax_a.bar(
            x + (i - offset_center) * w,
            vals,
            width=w,
            color=colors[m],
            edgecolor="0.15",
            lw=0.35,
            label=labels[m],
            zorder=2,
        )
        bar_artists[m] = list(bars)

    def _ann_text(m: str, j: int) -> str | None:
        val = vals_by_model[m][j]
        if abs(job_vals[j]) < 1e-9:
            return None if val < 1e-9 else f"{val:.2f}"
        pct = 100.0 * (val - job_vals[j]) / job_vals[j]
        return f"{pct:+.0f}%"

    for j in range(len(xlabels)):
        # Place near-equal bars with a small vertical stagger so identical
        # Site/Operator numbers remain both visible and on-bar.
        heights = [float(vals_by_model[m][j]) for m in agg_models]
        for k, m in enumerate(agg_models):
            text = _ann_text(m, j)
            if text is None:
                continue
            bar = bar_artists[m][j]
            xc = bar.get_x() + bar.get_width() / 2
            h = float(vals_by_model[m][j])
            # Extra lift only when a neighbor bar of similar height would
            # put two labels on top of each other.
            bump = 0
            for k2, h2 in enumerate(heights):
                if k2 == k:
                    continue
                if abs(h - h2) < 0.35 and k2 < k:
                    bump += 1
            ax_a.annotate(
                text,
                xy=(xc, h),
                xytext=(0.0, label_y0_pts + bump * label_dy_pts),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=0,
                fontsize=label_fs,
                color=colors[m],
                path_effects=label_stroke,
                zorder=6,
                annotation_clip=False,
            )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabels, fontsize=max(FS - 1, 6), linespacing=0.98)
    ax_a.set_ylabel("Absolute value", fontsize=FS, labelpad=2)
    ax_a.tick_params(axis="y", labelsize=6, pad=1.5)
    ax_a.tick_params(axis="x", pad=1.5)
    ax_a.set_xlim(-0.55, len(xlabels) - 0.45)
    _panel_tag(ax_a, "a", "Summer–winter mean")
    y_max_bars = max(max(vals_by_model[m]) for m in models)
    ax_a.set_ylim(0, y_max_bars * 1.18)

    t = np.arange(hours)
    style = {
        "FULL_JOB": dict(lw=1.6, ls="-"),
        "AGG_ENERGY": dict(lw=1.25, ls="-"),
        "AGG_POOL": dict(lw=1.25, ls="--"),
        "AGG_CONT": dict(lw=1.35, ls="-"),
    }
    # Draw AGG Operator above AGG Site so the dashed blue remains visible
    # when the two site trajectories coincide (as at Operator A / Pecos).
    zord = {"FULL_JOB": 4, "AGG_POOL": 3.5, "AGG_ENERGY": 3, "AGG_CONT": 1}

    def _draw_profile(ax, traces: dict[str, np.ndarray], ylabel: str) -> None:
        for m in models:
            ax.plot(
                t,
                traces[m],
                color=colors[m],
                zorder=zord[m],
                solid_capstyle="round",
                **style[m],
            )
        ax.set_xlim(0, hours - 1)
        ax.set_xlabel("Hour", fontsize=FS, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=FS, labelpad=2)
        ax.tick_params(labelsize=6, pad=1.5)

    _draw_profile(ax_b, combined["summer"], "Combined power (MW)")
    _panel_tag(ax_b, "b", "Combined · summer")
    _draw_profile(ax_c, site_traces["summer"], "Site power (MW)")
    _panel_tag(ax_c, "c", f"Site-level · {focus_site_label} · summer")

    handles = [Line2D([0], [0], color=colors[m], lw=2.0, ls=style[m]["ls"], label=labels[m]) for m in models]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=n_models,
        fontsize=FS_LEGEND,
        frameon=False,
        bbox_to_anchor=(0.55, 0.995),
        columnspacing=1.1,
        handletextpad=0.35,
    )

    return save_figure(fig, OUT / "Fig8_aggregate_model_error.png", pad_inches=0.02)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(build_fig4(include_agg_cont=True))


if __name__ == "__main__":
    main()
