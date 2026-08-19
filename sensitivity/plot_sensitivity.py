#!/usr/bin/env python3
"""
Simple sensitivity plotting for RankIC across hyperparameters.
Data is defined directly in the code for easy modification.
"""

import os
import re
from typing import Any, Dict, List

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")


# Experiment data — edit values here directly.
SENSITIVITY_DATA = {
    "codebook_size": {
        "title": "Codebook Size",
        "x_label": "Codebook Size (K)",
        "y_label": "RankIC",
        "series": [
            {"label": "CSI300", "x": [128, 256, 512], "y": [0.0533, 0.0573, 0.0607], "marker": "o"},
            {"label": "S&P500", "x": [128, 256, 512], "y": [0.0096, 0.0035, 0.0078], "marker": "s"},
        ]
    },
    "experts": {
        "title": "Number of Experts", 
        "x_label": "Number of Experts",
        "y_label": "RankIC",
        "series": [
            {"label": "CSI300", "x": [2, 4, 8], "y": [0.0598, 0.0607, 0.0581], "marker": "^"},
            {"label": "S&P500", "x": [2, 4, 8], "y": [0.0053, 0.0048, 0.0096], "marker": "v"},
        ]
    },
    "transformer_hidden": {
        "title": "Transformer Hidden",
        "x_label": "Transformer Hidden (d_model)", 
        "y_label": "RankIC",
        "series": [
            {"label": "CSI300", "x": [128, 256, 512, 768], "y": [0.040, 0.045, 0.046, 0.044], "marker": "o"},
            {"label": "S&P500", "x": [128, 256, 512, 768], "y": [0.039, 0.043, 0.045, 0.043], "marker": "s"},
        ]
    },
    "transformer_heads": {
        "title": "Transformer Heads",
        "x_label": "Transformer Heads",
        "y_label": "RankIC", 
        "series": [
            {"label": "CSI300", "x": [4, 8, 16], "y": [0.043, 0.046, 0.045], "marker": "o"},
            {"label": "S&P500", "x": [4, 8, 16], "y": [0.041, 0.044, 0.043], "marker": "s"},
        ]
    }
}


def _sanitize_filename(title: str) -> str:
    """Normalize a title into a safe filename."""
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "plot"


def _annotate_points(ax: plt.Axes, x_values: List[float], y_values: List[float], color: str = None, fontsize: int = 8) -> None:
    """Annotate each point with its value rounded to 4 decimals."""
    if not x_values or not y_values:
        return
    y_min = min(y_values)
    y_max = max(y_values)
    y_range = y_max - y_min
    offset = (y_range if y_range > 0 else 1e-6) * 0.02
    for xi, yi in zip(x_values, y_values):
        ax.text(xi, yi + offset, f"{yi:.4f}", ha="center", va="bottom", fontsize=fontsize, color=color)


def plot_sensitivity(data: Dict[str, Any], save_dir: str = "sensitivity/plots", show: bool = False, dpi: int = 180):
    """Render a sensitivity-analysis line plot for the given data block."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for series in data["series"]:
        x = series["x"]
        y = series["y"] 
        label = series.get("label")
        marker = series.get("marker", "o")
        linestyle = series.get("linestyle", "-")
        color = series.get("color")
        yerr = series.get("yerr")
        
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, label=label, marker=marker, 
                       linestyle=linestyle, color=color, capsize=3)
        else:
            ax.plot(x, y, label=label, marker=marker,
                   linestyle=linestyle, color=color)

        _annotate_points(ax, x, y, color=color)

    ax.set_xlabel(data["x_label"])
    ax.set_ylabel(data["y_label"])
    ax.set_title(data["title"])

    # Explicit x-ticks from the first series.
    if data["series"]:
        x_values = data["series"][0]["x"]
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values])

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = _sanitize_filename(data["title"]) + ".png"
        out_path = os.path.join(save_dir, fname)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


def main():
    """Generate every sensitivity-analysis plot."""
    print("Generating sensitivity-analysis plots...")

    for experiment_name, data in SENSITIVITY_DATA.items():
        print(f"  plotting: {data['title']}")
        plot_sensitivity(data, save_dir="sensitivity/plots", show=False, dpi=180)

    print("  plotting: Codebook Size & Experts subplot")
    plot_codebook_experts_subplot(save_dir="sensitivity/plots", show=False, dpi=180)

    print("All plots saved under sensitivity/plots/.")


def plot_specific(experiment_name: str, show: bool = True):
    """Render only one experiment's plot."""
    if experiment_name not in SENSITIVITY_DATA:
        print(f"Available experiments: {list(SENSITIVITY_DATA.keys())}")
        return

    data = SENSITIVITY_DATA[experiment_name]
    plot_sensitivity(data, save_dir="sensitivity/plots", show=show, dpi=180)


def plot_codebook_experts_subplot(save_dir: str = "sensitivity/plots", show: bool = False, dpi: int = 180):
    """Render Codebook Size and Number of Experts side by side as a 1x2 subplot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    codebook_data = SENSITIVITY_DATA["codebook_size"]
    for series in codebook_data["series"]:
        x = series["x"]
        y = series["y"]
        label = series.get("label")
        marker = series.get("marker", "o")
        linestyle = series.get("linestyle", "-")
        color = series.get("color")
        yerr = series.get("yerr")
        
        if yerr is not None:
            ax1.errorbar(x, y, yerr=yerr, label=label, marker=marker, 
                        linestyle=linestyle, color=color, capsize=3)
        else:
            ax1.plot(x, y, label=label, marker=marker,
                    linestyle=linestyle, color=color)

        _annotate_points(ax1, x, y, color=color)
    
    ax1.set_xlabel(codebook_data["x_label"])
    ax1.set_ylabel(codebook_data["y_label"])
    # ax1.set_title(codebook_data["title"])
    ax1.set_xticks([128, 256, 512])
    ax1.set_xticklabels(['128', '256', '512'])
    ax1.grid(True, linestyle=":", alpha=0.6)

    experts_data = SENSITIVITY_DATA["experts"]
    for series in experts_data["series"]:
        x = series["x"]
        y = series["y"]
        label = series.get("label")
        marker = series.get("marker", "o")
        linestyle = series.get("linestyle", "-")
        color = series.get("color")
        yerr = series.get("yerr")
        
        if yerr is not None:
            ax2.errorbar(x, y, yerr=yerr, label=label, marker=marker, 
                        linestyle=linestyle, color=color, capsize=3)
        else:
            ax2.plot(x, y, label=label, marker=marker,
                    linestyle=linestyle, color=color)

        _annotate_points(ax2, x, y, color=color)
    
    ax2.set_xlabel(experts_data["x_label"])
    ax2.set_ylabel(experts_data["y_label"])
    # ax2.set_title(experts_data["title"])
    ax2.set_xticks([2, 4, 8])
    ax2.set_xticklabels(['2', '4', '8'])
    ax2.grid(True, linestyle=":", alpha=0.6)

    # One shared legend outside the figure on the right.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(0.92, 0.5),
        frameon=False,
        fontsize=8,
        markerscale=0.9,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.3,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, right=0.92)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = "codebook_experts_subplot.png"
        out_path = os.path.join(save_dir, fname)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved subplot: {out_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


if __name__ == "__main__":
    main()
