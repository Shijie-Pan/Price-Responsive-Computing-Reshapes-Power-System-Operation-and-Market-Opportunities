"""
Bar charts from experiment CSVs (IEEE-style figures for flexibility studies).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_categorical_bar_figure(
    file_path: str,
    file_type: str = "csv",
    x_column: str | None = None,
    y_column: str | None = None,
    order: list | None = None,
    x_label: str = "",
    y_label: str = "",
    rotate_xticks: float = 0,
    show_values: bool = False,
    save_pdf: bool = False,
    output_file: str = "out.png",
    color_map: dict[str, str] | None = None,
    default_color: str = "#3C5488",
    bar_edgecolor: str = "black",
    bar_linewidth: float = 0.5,
) -> None:
    if file_type != "csv":
        raise ValueError(f"Only csv supported, got {file_type}")
    df = pd.read_csv(file_path)
    if x_column not in df.columns or y_column not in df.columns:
        raise KeyError(f"CSV missing columns: {x_column}, {y_column}")

    plot_df = df[[x_column, y_column]].dropna(subset=[y_column])
    if order is not None:
        plot_df[x_column] = pd.Categorical(plot_df[x_column], categories=order, ordered=True)
        plot_df = plot_df.sort_values(x_column)

    fig, ax = plt.subplots(figsize=(12, 6))
    xlabs = plot_df[x_column].astype(str).tolist()
    ys = plot_df[y_column].astype(float).tolist()
    colors = [color_map.get(x, default_color) if color_map else default_color for x in xlabs]
    bars = ax.bar(
        range(len(xlabs)),
        ys,
        color=colors,
        edgecolor=bar_edgecolor,
        linewidth=bar_linewidth,
        width=0.72,
    )
    ax.set_xticks(range(len(xlabs)))
    ax.set_xticklabels(xlabs, rotation=rotate_xticks, ha="right" if rotate_xticks else "center")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.3)
    if show_values:
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = out.suffix.lstrip(".").lower() or "png"
    if fmt == "jpg":
        fmt = "jpeg"
    fig.savefig(out, format=fmt, dpi=300, bbox_inches="tight")
    plt.close(fig)
