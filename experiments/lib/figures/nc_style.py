"""Nature Communications figure style (Times New Roman, PNG only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

NC_SINGLE_COL_MM = 89
NC_DOUBLE_COL_MM = 183
FIG_DPI = 300
FIG_FORMAT = "png"

NPG_PALETTE = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]

PORTFOLIO_COLORS = {
    "baseline": NPG_PALETTE[3],
    "initial_assignment": NPG_PALETTE[2],
    "termination": NPG_PALETTE[7],
    "slack": NPG_PALETTE[4],
    "reallocation": NPG_PALETTE[0],
    "reallocation_termination": NPG_PALETTE[7],
    "reallocation_slack": NPG_PALETTE[6],
    "slack_termination": NPG_PALETTE[4],
    "reallocation_slack_termination": NPG_PALETTE[0],
}

DISPLAY_SHORT = {
    "baseline": "FCFS",
    "initial_assignment": "Asgn.",
    "termination": "Term.",
    "slack": "Slack",
    "reallocation": "Ralc.",
    "reallocation_termination": "Ralc.+Term.",
    "reallocation_slack": "Ralc.+Slack",
    "slack_termination": "Slack+Term.",
    "reallocation_slack_termination": "Full flex",
}

SCENARIO_ORDER = [
    "baseline",
    "initial_assignment",
    "termination",
    "slack",
    "reallocation",
    "reallocation_termination",
    "reallocation_slack",
    "slack_termination",
    "reallocation_slack_termination",
]

# Manuscript-facing labels only. Internal ids / folders stay amazon…oracle.
OPERATOR_ORDER = ["amazon", "coreweave", "google", "meta", "microsoft", "oracle"]
OPERATOR_LABELS = {
    "amazon": "Operator A",
    "coreweave": "Operator B",
    "google": "Operator C",
    "meta": "Operator D",
    "microsoft": "Operator E",
    "oracle": "Operator F",
}
OPERATOR_COLORS = {
    "amazon": "#ff9900",
    "coreweave": "#7b2cbf",
    "google": "#1a73e8",
    "meta": "#0866ff",
    "microsoft": "#107c10",
    "oracle": "#c74634",
}
# Source-registry vendors (not used on figure legends). Internal id `oracle`
# is the Stargate / Oracle-linked group; manuscript label is Operator F.
OPERATOR_VENDOR = {
    "amazon": "Amazon",
    "coreweave": "CoreWeave",
    "google": "Google",
    "meta": "Meta",
    "microsoft": "Microsoft",
    "oracle": "Oracle",
}
SEASON_ORDER = ["summer", "winter"]


def mm_to_inches(mm: float) -> float:
    return mm / 25.4


def figsize_double_col(height_in: float = 4.2) -> tuple[float, float]:
    return (mm_to_inches(NC_DOUBLE_COL_MM), height_in)


def apply_figure_style() -> None:
    plt.rcParams.update(
        {
            # Force TNR so maps cannot fall back to Arial via apply_geo_style.
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": FIG_DPI,
            "savefig.format": FIG_FORMAT,
            "text.usetex": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.grid": False,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# Backward-compatible alias (former experiments.travis.lib.figure_utils).
apply_nature_comm_style = apply_figure_style


def panel_label(ax, label: str, *, x: float = -0.12, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(
    fig,
    path: Path | str,
    *,
    also_pdf: bool = False,
    also_svg: bool = False,
    pad_inches: float = 0.02,
    bbox_inches: str | None = None,
) -> Path:
    """Save PNG at fixed figure size (default).

    Do **not** use ``bbox_inches='tight'`` for paper boards: tight cropping makes
    fonts appear different sizes across figures when included at the same width.
    """
    out = Path(path)
    if out.suffix.lower() != f".{FIG_FORMAT}":
        out = out.with_suffix(f".{FIG_FORMAT}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        format=FIG_FORMAT,
        dpi=FIG_DPI,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        facecolor="white",
    )
    if also_pdf:
        fig.savefig(
            out.with_suffix(".pdf"),
            format="pdf",
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            facecolor="white",
        )
    if also_svg:
        fig.savefig(
            out.with_suffix(".svg"),
            format="svg",
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            facecolor="white",
        )
    plt.close(fig)
    return out
