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

from experiments.lib.paths import package_root

ROOT = package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.figure_style import *  # noqa: F403
from experiments.lib.figure_common import *  # noqa: F403

OUT = ROOT / "experiments/E07/figures"

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


def build_fig4() -> Path:
    """Aggregate-model error.

    a  Summer–winter means. Expenditure is price times facility power
       on the original 168 h for every model. The job-level trajectory
       is longer; hours at or after 168 are omitted.
       Peak is the combined fleet peak. Ramp is the largest hour-to-hour
       step at Amazon Pecos. Range is that site's interquartile range:
       max−min is the same for every model because each one touches the
       same power limits.
    b  Combined fleet power, summer.
    c  Summer power at Amazon Pecos, the site where AGG Site departs
       most from the job-level trace.
    """
    apply_figure_style()
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_TICK_COMPOSED, "ytick.labelsize": FS_TICK_COMPOSED, "legend.fontsize": FS_LEGEND})
    summer = json.loads(E03_SUMMER.read_text(encoding="utf-8"))
    winter = json.loads(E03_WINTER.read_text(encoding="utf-8"))

    models = ["FULL_JOB", "AGG_ENERGY", "AGG_POOL", "AGG_CONT"]
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
    # Amazon Pecos: largest summer |AGG Site − job-level| among sites.
    # Panel a ramp is this site's max hour-to-hour step. Panel a IQR is
    # the same site: max−min is identical across models.
    focus_site = 751
    focus_site_label = "Amazon(Pecos)"

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
        season: {m: _power(season, m, "Total_Power_MW", hours) for m in models}
        for season in ("summer", "winter")
    }
    site_traces = {
        season: {m: _power(season, m, f"Site{focus_site}_Power_MW", hours) for m in models}
        for season in ("summer", "winter")
    }

    def _site_ramp_iqr_gw(model: str) -> tuple[float, float]:
        ramps, iqrs = [], []
        for season in ("summer", "winter"):
            y = site_traces[season][model]
            ramps.append(float(np.max(np.abs(np.diff(y)))))
            q75, q25 = np.percentile(y, [75, 25])
            iqrs.append(float(q75 - q25))
        return float(np.mean(ramps)) / 1e3, float(np.mean(iqrs)) / 1e3

    # Common 168 h bill: price times facility power. Job-level hours at
    # or after 168 are omitted. Aggregate schedules are already 168 h.
    expenditure_m = {
        season: _fig4_expenditure_musd(season, payload)
        for season, payload in (("summer", summer), ("winter", winter))
    }
    xlabels = [
        "Operator electricity\nexpenditure ($M)",
        "Peak\n(combined, GW)",
        f"Ramp\n({focus_site_label}, GW/h)",
        f"Interquartile range\n({focus_site_label}, GW)",
    ]
    vals_by_model = {}
    for m in models:
        ramp_gw, iqr_gw = _site_ramp_iqr_gw(m)
        vals_by_model[m] = [
            0.5 * (expenditure_m["summer"][m] + expenditure_m["winter"][m]),
            avg(m, "peak_total_mw") / 1e3,
            ramp_gw,
            iqr_gw,
        ]
    job_vals = vals_by_model["FULL_JOB"]

    fig = plt.figure(figsize=figsize_double_col(4.55), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.12, 0.92],
        hspace=0.48,
        wspace=0.28,
        left=FIG4_LEFT,
        right=FIG4_RIGHT,
        top=0.90,
        bottom=FIG4_BOTTOM,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    x = np.arange(len(xlabels))
    w = 0.18
    labeled_pct: dict[int, float] = {}
    for i, m in enumerate(models):
        vals = vals_by_model[m]
        bars = ax_a.bar(
            x + (i - 1.5) * w,
            vals,
            width=w,
            color=colors[m],
            edgecolor="0.15",
            lw=0.35,
            label=labels[m],
            zorder=2,
        )
        if m == "FULL_JOB":
            continue
        for j, bar in enumerate(bars):
            if abs(job_vals[j]) < 1e-9:
                continue
            pct = 100.0 * (vals[j] - job_vals[j]) / job_vals[j]
            # Wide IQR labels collide; AGG Site and AGG Operator match here.
            if j == len(xlabels) - 1 and j in labeled_pct and abs(labeled_pct[j] - pct) < 0.5:
                continue
            labeled_pct[j] = pct
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.35,
                f"{pct:+.0f}%",
                ha="center",
                va="bottom",
                fontsize=FIG4_PCT_FS,
                color=colors[m],
                clip_on=False,
            )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabels, fontsize=FS)
    ax_a.set_ylabel("Absolute value", fontsize=FS)
    ax_a.tick_params(labelsize=6)
    _panel_tag(ax_a, "a", "Summer–winter mean")
    ax_a.set_ylim(0, ax_a.get_ylim()[1] * 1.18)

    t = np.arange(hours)
    style = {
        "FULL_JOB": dict(lw=1.6, ls="-"),
        "AGG_ENERGY": dict(lw=1.25, ls="-"),
        "AGG_POOL": dict(lw=1.25, ls="--"),
        "AGG_CONT": dict(lw=1.35, ls="-"),
    }
    zord = {"FULL_JOB": 4, "AGG_ENERGY": 3, "AGG_POOL": 2, "AGG_CONT": 1}

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
        ax.set_xlabel("Hour", fontsize=FS)
        ax.set_ylabel(ylabel, fontsize=FS)
        ax.tick_params(labelsize=6)

    _draw_profile(ax_b, combined["summer"], "Combined power (MW)")
    _panel_tag(ax_b, "b", "Combined · summer")
    _draw_profile(ax_c, site_traces["summer"], "Site power (MW)")
    _panel_tag(ax_c, "c", f"Site-level · {focus_site_label} · summer")

    handles = [Line2D([0], [0], color=colors[m], lw=2.0, ls=style[m]["ls"], label=labels[m]) for m in models]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=FS_LEGEND,
        frameon=False,
        bbox_to_anchor=(0.55, 0.995),
        columnspacing=1.1,
        handletextpad=0.35,
    )

    out = OUT / "Fig6_aggregate_model_error.png"
    return save_figure(fig, out)




def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(build_fig4())


if __name__ == "__main__":
    main()
