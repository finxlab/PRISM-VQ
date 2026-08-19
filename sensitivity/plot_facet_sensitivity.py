#!/usr/bin/env python3
"""
Faceted Sensitivity Analysis for TKDE style publication
Main panel: Codebook×MoE 2D heatmap with d as facets (d=32, d=64)
Bottom panel: Interaction map ΔZ = Z(d=64) - Z(d=32)

This layout clearly shows:
1. How performance varies across Codebook×MoE for each d
2. Where d makes the biggest difference (interaction effect)
"""

import argparse
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_RES_DIR = str(Path(__file__).resolve().parent.parent / "res")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import matplotlib
from matplotlib.patches import Rectangle

# IEEE style configuration
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 13
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Clean white background for IEEE style
plt.style.use('default')
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['grid.alpha'] = 0.3
matplotlib.rcParams['grid.linestyle'] = ':'


def parse_folder_name(folder_name: str) -> Dict:
    """Parse folder name to extract hyperparameters."""
    params = {}

    vqk_match = re.search(r'VQK(\d+)', folder_name)
    if vqk_match:
        params['VQK'] = int(vqk_match.group(1))

    if 'sp500' in folder_name:
        params['market'] = 'sp500'
    elif 'csi300' in folder_name:
        params['market'] = 'csi300'

    mo_match = re.search(r'_mo(\d+)', folder_name)
    if mo_match:
        params['mo'] = int(mo_match.group(1))

    dm_match = re.search(r'_dm(\d+)', folder_name)
    if dm_match:
        params['dm'] = int(dm_match.group(1))

    nh_match = re.search(r'_nh(\d+)', folder_name)
    if nh_match:
        params['nh'] = int(nh_match.group(1))

    return params


def read_rank_ic(metric_file: str) -> float:
    """Read RankIC from metric CSV file."""
    try:
        with open(metric_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('') == 'RankIC':
                    return float(row['values'])
    except Exception as e:
        print(f"Error reading {metric_file}: {e}")
    return None


def collect_sensitivity_data(res_dir: str, market: str) -> List[Dict]:
    """Collect sensitivity data from experiment results."""
    VALID_VQK = [128, 256, 512]
    VALID_DM = [32, 64]
    VALID_MO = [2, 4, 8]

    if market == 'csi300':
        search_dir = os.path.join(res_dir, 'csi300')
        required_nh = 2
    elif market == 'sp500':
        search_dir = os.path.join(res_dir, 'sp500')
        required_nh = 4
    else:
        return []

    if not os.path.exists(search_dir):
        print(f"Directory not found: {search_dir}")
        return []

    experiments = {}

    for dir_name in os.listdir(search_dir):
        dir_path = os.path.join(search_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        if market not in dir_name:
            continue

        params = parse_folder_name(dir_name)
        if 'VQK' not in params or 'mo' not in params or 'dm' not in params or 'nh' not in params:
            continue

        if params['VQK'] not in VALID_VQK:
            continue
        if params['dm'] not in VALID_DM:
            continue
        if params['mo'] not in VALID_MO:
            continue
        if params['nh'] != required_nh:
            continue

        rank_ics = []
        for i in range(5):
            metric_file = os.path.join(dir_path, f'{i}_metric.csv')
            if os.path.exists(metric_file):
                rank_ic = read_rank_ic(metric_file)
                if rank_ic is not None:
                    rank_ics.append(rank_ic)

        if rank_ics:
            key = (params['VQK'], params['mo'], params['dm'])
            if key not in experiments:
                experiments[key] = {
                    'VQK': params['VQK'],
                    'mo': params['mo'],
                    'dm': params['dm'],
                    'RankIC': np.mean(rank_ics),
                    'RankIC_std': np.std(rank_ics),
                    'folder': dir_name
                }

    data = list(experiments.values())
    return data


def plot_facet_sensitivity(
    data: List[Dict],
    market: str,
    save_dir: str = "sensitivity/plots",
):
    """
    Create faceted sensitivity plot:
    - Top row: Two panels showing Codebook×MoE for d=32 and d=64
    - Bottom row: One panel showing interaction effect ΔZ = Z(d=64) - Z(d=32)
    """
    if not data:
        print(f"No data available for {market}")
        return

    # Create figure with 2 rows: top has 2 panels, bottom has 1 panel
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.05],
                          hspace=0.35, wspace=0.25, left=0.08, right=0.90, top=0.94, bottom=0.08)

    # Top row: d=32 and d=64 panels
    ax_d32 = fig.add_subplot(gs[0, 0])
    ax_d64 = fig.add_subplot(gs[0, 1])

    # Bottom row: interaction map (spans both columns)
    ax_delta = fig.add_subplot(gs[1, :2])

    # Colorbar axes
    cbar_ax_top = fig.add_subplot(gs[0, 2])
    cbar_ax_bottom = fig.add_subplot(gs[1, 2])

    # Filter data by d
    data_d32 = [d for d in data if d['dm'] == 32]
    data_d64 = [d for d in data if d['dm'] == 64]

    # Get global min/max for consistent color scale
    all_ric = np.array([d['RankIC'] for d in data])
    vmin, vmax = all_ric.min(), all_ric.max()

    # Plot d=32 panel
    if data_d32:
        plot_heatmap_panel(ax_d32, data_d32, vmin, vmax,
                          title='(a) $d = 32$',
                          xlabel='Codebook Size', ylabel='MoE Experts')

    # Plot d=64 panel
    scatter_d64 = None
    if data_d64:
        scatter_d64 = plot_heatmap_panel(ax_d64, data_d64, vmin, vmax,
                                        title='(b) $d = 64$',
                                        xlabel='Codebook Size', ylabel='MoE Experts')

    # Add colorbar for top panels
    if scatter_d64:
        cbar_top = fig.colorbar(scatter_d64, cax=cbar_ax_top)
        cbar_top.set_label('RankIC', fontsize=11, weight='bold')
        cbar_top.ax.tick_params(labelsize=9)

    # Plot interaction map
    if data_d32 and data_d64:
        scatter_delta = plot_interaction_map(ax_delta, data_d32, data_d64,
                                            title='(c) Interaction Effect: $\Delta$RankIC = RankIC($d$=64) − RankIC($d$=32)',
                                            xlabel='Codebook Size', ylabel='MoE Experts')

        # Add colorbar for interaction map
        if scatter_delta:
            cbar_bottom = fig.colorbar(scatter_delta, cax=cbar_ax_bottom)
            cbar_bottom.set_label('$\Delta$RankIC', fontsize=11, weight='bold')
            cbar_bottom.ax.tick_params(labelsize=9)

    # Add overall title
    market_name = 'CSI300' if market == 'csi300' else 'S&P500'
    fig.suptitle(f'{market_name}: Hyperparameter Sensitivity Analysis',
                 fontsize=14, fontweight='bold', y=0.97)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'facet_sensitivity_{market}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved faceted sensitivity plot: {out_path}")
    plt.close(fig)


def plot_heatmap_panel(ax, data, vmin, vmax, title, xlabel, ylabel):
    """Plot a single heatmap panel for Codebook×MoE at fixed d."""
    # Extract data
    x_values = np.array([d['VQK'] for d in data])
    y_values = np.array([d['mo'] for d in data])
    z_values = np.array([d['RankIC'] for d in data])

    # Get unique values
    x_unique = np.array(sorted(set(x_values)))
    y_unique = np.array(sorted(set(y_values)))

    # Create smooth interpolation
    if len(x_unique) >= 2 and len(y_unique) >= 2:
        xi = np.linspace(x_values.min(), x_values.max(), 150)
        yi = np.linspace(y_values.min(), y_values.max(), 150)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        zi = griddata((x_values, y_values), z_values, (xi_grid, yi_grid), method='cubic')

        # Filled contour
        contourf = ax.contourf(xi_grid, yi_grid, zi, levels=20, cmap='Blues',
                               vmin=vmin, vmax=vmax, alpha=0.9)

        # Contour lines
        contour = ax.contour(xi_grid, yi_grid, zi, levels=8, colors='darkblue',
                            linewidths=0.8, alpha=0.5, linestyles='solid')

    # Plot data points
    scatter = ax.scatter(x_values, y_values, c=z_values, cmap='Blues',
                        s=350, edgecolors='black', linewidths=2.0,
                        vmin=vmin, vmax=vmax, zorder=10, alpha=1.0)

    # Find best and worst
    z_max_idx = np.argmax(z_values)
    z_min_idx = np.argmin(z_values)

    # Highlight best with thick green border
    ax.scatter(x_values[z_max_idx], y_values[z_max_idx],
              s=450, facecolors='none', edgecolors='darkgreen',
              linewidths=4.0, zorder=11, marker='o')

    # Highlight worst with thick red border
    ax.scatter(x_values[z_min_idx], y_values[z_min_idx],
              s=450, facecolors='none', edgecolors='darkred',
              linewidths=4.0, zorder=11, marker='o')

    # Annotate values
    for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
        if i == z_max_idx:
            color = 'darkgreen'
            fontsize = 11
            fontweight = 'bold'
        elif i == z_min_idx:
            color = 'darkred'
            fontsize = 11
            fontweight = 'bold'
        else:
            color = 'black'
            fontsize = 9
            fontweight = 'normal'

        ax.annotate(f'{z:.4f}',
                   xy=(x, y), xytext=(0, -30),
                   textcoords='offset points',
                   ha='center', va='top',
                   fontsize=fontsize, fontweight=fontweight, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', alpha=0.95, linewidth=0.8))

    # Labels
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, pad=10, fontweight='bold')

    # Ticks
    ax.set_xticks(x_unique)
    ax.set_yticks(y_unique)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0, length=5)

    # Grid
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5, color='gray', zorder=0)

    # Limits with padding
    x_range = x_unique.max() - x_unique.min()
    y_range = y_unique.max() - y_unique.min()
    ax.set_xlim(x_unique.min() - x_range * 0.15, x_unique.max() + x_range * 0.15)
    ax.set_ylim(y_unique.min() - y_range * 0.20, y_unique.max() + y_range * 0.20)

    return scatter


def plot_interaction_map(ax, data_d32, data_d64, title, xlabel, ylabel):
    """
    Plot interaction effect: ΔZ = Z(d=64) - Z(d=32)
    Shows where increasing d from 32 to 64 has the biggest impact.
    """
    # Build dictionaries for quick lookup
    lookup_d32 = {(d['VQK'], d['mo']): d['RankIC'] for d in data_d32}
    lookup_d64 = {(d['VQK'], d['mo']): d['RankIC'] for d in data_d64}

    # Find common (VQK, mo) pairs
    common_keys = set(lookup_d32.keys()) & set(lookup_d64.keys())

    if not common_keys:
        print("No common data points for interaction map")
        return None

    # Calculate differences
    x_values = []
    y_values = []
    delta_values = []

    for vqk, mo in common_keys:
        x_values.append(vqk)
        y_values.append(mo)
        delta = lookup_d64[(vqk, mo)] - lookup_d32[(vqk, mo)]
        delta_values.append(delta)

    x_values = np.array(x_values)
    y_values = np.array(y_values)
    delta_values = np.array(delta_values)

    # Get unique values
    x_unique = np.array(sorted(set(x_values)))
    y_unique = np.array(sorted(set(y_values)))

    # Symmetric color scale around zero
    delta_max = max(abs(delta_values.min()), abs(delta_values.max()))
    vmin_delta = -delta_max
    vmax_delta = delta_max

    # Create smooth interpolation
    if len(x_unique) >= 2 and len(y_unique) >= 2:
        xi = np.linspace(x_values.min(), x_values.max(), 150)
        yi = np.linspace(y_values.min(), y_values.max(), 150)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        zi = griddata((x_values, y_values), delta_values, (xi_grid, yi_grid), method='cubic')

        # Use diverging colormap: blue (negative) to white (zero) to red (positive)
        # But we want Blues for consistency, so use Blues but centered
        # Actually, for interaction, a diverging colormap makes more sense
        # Let's use a custom Blues-based diverging: light blue (negative) to dark blue (positive)
        contourf = ax.contourf(xi_grid, yi_grid, zi, levels=20, cmap='RdBu_r',
                               vmin=vmin_delta, vmax=vmax_delta, alpha=0.9)

        # Contour lines
        contour = ax.contour(xi_grid, yi_grid, zi, levels=8, colors='black',
                            linewidths=0.8, alpha=0.5, linestyles='solid')

        # Add zero contour line (important reference)
        ax.contour(xi_grid, yi_grid, zi, levels=[0], colors='black',
                  linewidths=2.0, alpha=0.8, linestyles='dashed')

    # Plot data points
    scatter = ax.scatter(x_values, y_values, c=delta_values, cmap='RdBu_r',
                        s=350, edgecolors='black', linewidths=2.0,
                        vmin=vmin_delta, vmax=vmax_delta, zorder=10, alpha=1.0)

    # Find max positive and max negative
    max_pos_idx = np.argmax(delta_values)
    max_neg_idx = np.argmin(delta_values)

    # Highlight biggest positive effect (green = d=64 helps most)
    ax.scatter(x_values[max_pos_idx], y_values[max_pos_idx],
              s=450, facecolors='none', edgecolors='darkgreen',
              linewidths=4.0, zorder=11, marker='s')

    # Highlight biggest negative effect (red = d=64 hurts or d=32 is better)
    if delta_values[max_neg_idx] < 0:
        ax.scatter(x_values[max_neg_idx], y_values[max_neg_idx],
                  s=450, facecolors='none', edgecolors='darkred',
                  linewidths=4.0, zorder=11, marker='s')

    # Annotate values
    for i, (x, y, delta) in enumerate(zip(x_values, y_values, delta_values)):
        if i == max_pos_idx:
            color = 'darkgreen'
            fontsize = 11
            fontweight = 'bold'
        elif i == max_neg_idx and delta < 0:
            color = 'darkred'
            fontsize = 11
            fontweight = 'bold'
        else:
            color = 'black'
            fontsize = 9
            fontweight = 'normal'

        # Use + for positive, - already included
        sign = '+' if delta >= 0 else ''
        ax.annotate(f'{sign}{delta:.4f}',
                   xy=(x, y), xytext=(0, -30),
                   textcoords='offset points',
                   ha='center', va='top',
                   fontsize=fontsize, fontweight=fontweight, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', alpha=0.95, linewidth=0.8))

    # Labels
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, pad=10, fontweight='bold')

    # Ticks
    ax.set_xticks(x_unique)
    ax.set_yticks(y_unique)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0, length=5)

    # Grid
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5, color='gray', zorder=0)

    # Limits with padding
    x_range = x_unique.max() - x_unique.min()
    y_range = y_unique.max() - y_unique.min()
    ax.set_xlim(x_unique.min() - x_range * 0.15, x_unique.max() + x_range * 0.15)
    ax.set_ylim(y_unique.min() - y_range * 0.20, y_unique.max() + y_range * 0.20)

    return scatter


def main():
    """Generate faceted sensitivity analysis plots for both markets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res-dir", default=DEFAULT_RES_DIR,
                        help=f"Path to the results directory (default: {DEFAULT_RES_DIR})")
    args = parser.parse_args()
    res_dir = args.res_dir

    for market in ['csi300', 'sp500']:
        print(f"\n{'='*60}")
        print(f"Processing {market.upper()} market...")
        print(f"{'='*60}")

        # Collect data
        data = collect_sensitivity_data(res_dir, market)
        print(f"Found {len(data)} experiments for {market}")

        if data:
            # Print summary
            print("\nData summary:")
            for d in sorted(data, key=lambda x: (x['VQK'], x['dm'], x['mo'])):
                print(f"  VQK={d['VQK']:3d}, dm={d['dm']:2d}, mo={d['mo']:1d} -> "
                      f"RankIC={d['RankIC']:.4f} ± {d['RankIC_std']:.4f}")

            # Generate plots
            print(f"\nGenerating faceted sensitivity plots for {market}...")
            plot_facet_sensitivity(data, market)
        else:
            print(f"No data found for {market}")

    print(f"\n{'='*60}")
    print("All plots saved to sensitivity/plots/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
