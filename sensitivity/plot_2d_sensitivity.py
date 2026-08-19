#!/usr/bin/env python3
"""
2D Sensitivity Analysis for VQK, d_model, and MoE experts
Generates 2D contour/heatmap plots showing how RankIC varies with these hyperparameters.
IEEE TKDE style optimized - cleaner and more readable than 3D plots.
"""

import argparse
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

DEFAULT_RES_DIR = str(Path(__file__).resolve().parent.parent / "res")
from scipy.interpolate import griddata
import matplotlib

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
    """
    Parse folder name to extract hyperparameters.
    Example: VQK512_sp500_mo8_k4_mh64_md0.1_dm64_nh4_l1_d0.1_au0.001_...
    """
    params = {}

    # Extract VQK (codebook size)
    vqk_match = re.search(r'VQK(\d+)', folder_name)
    if vqk_match:
        params['VQK'] = int(vqk_match.group(1))

    # Extract market
    if 'sp500' in folder_name:
        params['market'] = 'sp500'
    elif 'csi300' in folder_name:
        params['market'] = 'csi300'

    # Extract mo (number of experts)
    mo_match = re.search(r'_mo(\d+)', folder_name)
    if mo_match:
        params['mo'] = int(mo_match.group(1))

    # Extract dm (d_model)
    dm_match = re.search(r'_dm(\d+)', folder_name)
    if dm_match:
        params['dm'] = int(dm_match.group(1))

    # Extract nh (number of heads)
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
    """
    Collect sensitivity data from experiment results.
    Returns list of dicts with keys: VQK, mo, dm, RankIC
    """
    # Define the hyperparameters we want to analyze
    VALID_VQK = [128, 256, 512]
    VALID_DM = [32, 64]
    VALID_MO = [2, 4, 8]

    # Define the specific directory and nh requirement
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

    # Collect exactly one experiment per (VQK, mo, dm) combination
    experiments = {}

    # Only look at direct subdirectories (not recursive)
    for dir_name in os.listdir(search_dir):
        dir_path = os.path.join(search_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        if market not in dir_name:
            continue

        # Parse parameters from folder name
        params = parse_folder_name(dir_name)
        if 'VQK' not in params or 'mo' not in params or 'dm' not in params or 'nh' not in params:
            continue

        # Filter: only include desired hyperparameter values
        if params['VQK'] not in VALID_VQK:
            continue
        if params['dm'] not in VALID_DM:
            continue
        if params['mo'] not in VALID_MO:
            continue
        if params['nh'] != required_nh:
            continue

        # Read RankIC from all cross-validation folds
        rank_ics = []

        for i in range(5):  # 5 seeds cross-validation
            metric_file = os.path.join(dir_path, f'{i}_metric.csv')
            if os.path.exists(metric_file):
                rank_ic = read_rank_ic(metric_file)
                if rank_ic is not None:
                    rank_ics.append(rank_ic)

        # Store this experiment (only first one per combination)
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

    # Convert to list
    data = list(experiments.values())
    return data


def plot_2d_contour(
    data: List[Dict],
    market: str,
    save_dir: str = "sensitivity/plots",
    include_suptitle: bool = False,
):
    """
    Create 2D contour plots showing sensitivity analysis.
    Much cleaner and more readable than 3D plots.
    """
    if not data:
        print(f"No data available for {market}")
        return

    # IEEE 2-column width optimization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    BEST_DM = 64
    BEST_VQK = 512
    BEST_MO = 2 if market == 'csi300' else 8

    all_ric = np.array([d['RankIC'] for d in data])
    vmin, vmax = all_ric.min(), all_ric.max()

    # Plot 1: Codebook vs MoE (fixed dm=64)
    ax1 = axes[0]
    filtered_data1 = [d for d in data if d['dm'] == BEST_DM]
    if filtered_data1:
        plot_2d_subplot(
            ax1, filtered_data1,
            x_key='VQK', y_key='mo',
            xlabel='Codebook Size', ylabel='MoE Experts',
            title='(a) Codebook vs MoE', vmin=vmin, vmax=vmax
        )

    # Plot 2: Codebook vs d (fixed mo)
    ax2 = axes[1]
    filtered_data2 = [d for d in data if d['mo'] == BEST_MO]
    if filtered_data2:
        plot_2d_subplot(
            ax2, filtered_data2,
            x_key='VQK', y_key='dm',
            xlabel='Codebook Size', ylabel='$d$',
            title='(b) Codebook vs $d$', vmin=vmin, vmax=vmax
        )

    # Plot 3: d vs MoE (fixed VQK=512)
    ax3 = axes[2]
    filtered_data3 = [d for d in data if d['VQK'] == BEST_VQK]
    if filtered_data3:
        contour = plot_2d_subplot(
            ax3, filtered_data3,
            x_key='dm', y_key='mo',
            xlabel='$d$', ylabel='MoE Experts',
            title='(c) $d$ vs MoE', vmin=vmin, vmax=vmax
        )

    # Add a single colorbar on the right
    if filtered_data1:
        # Create colorbar axis
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('RankIC', fontsize=12, weight='bold')
        cbar.ax.tick_params(labelsize=10)

    if include_suptitle:
        market_name = 'CSI300' if market == 'csi300' else 'S&P500'
        fig.suptitle(f'{market_name}: Hyperparameter Sensitivity Analysis',
                     fontsize=14, y=0.98, fontweight='bold')
        plt.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.12, wspace=0.30)
    else:
        plt.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.12, wspace=0.30)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'2d_sensitivity_{market}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved 2D contour plot: {out_path}")
    plt.close(fig)


def plot_2d_subplot(ax, data, x_key, y_key, xlabel, ylabel, title, vmin=None, vmax=None):
    """
    Plot 2D contour/heatmap with scatter points and value annotations.
    """
    # Extract data
    x_values = np.array([d[x_key] for d in data])
    y_values = np.array([d[y_key] for d in data])
    z_values = np.array([d['RankIC'] for d in data])

    # Get unique values for grid
    x_unique = np.array(sorted(set(x_values)))
    y_unique = np.array(sorted(set(y_values)))

    # Create meshgrid for contour plot
    if len(x_unique) >= 2 and len(y_unique) >= 2:
        # Create finer grid for smooth contours
        xi = np.linspace(x_values.min(), x_values.max(), 100)
        yi = np.linspace(y_values.min(), y_values.max(), 100)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate
        zi = griddata((x_values, y_values), z_values, (xi_grid, yi_grid), method='cubic')

        # Plot filled contour
        contourf = ax.contourf(xi_grid, yi_grid, zi, levels=15, cmap='Blues',
                               vmin=vmin, vmax=vmax, alpha=0.8)

        # Plot contour lines
        contour = ax.contour(xi_grid, yi_grid, zi, levels=10, colors='black',
                            linewidths=0.5, alpha=0.4, linestyles='solid')

        # Add contour labels
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.4f')

    # Plot actual data points as scatter
    scatter = ax.scatter(x_values, y_values, c=z_values, cmap='Blues',
                        s=300, edgecolors='black', linewidths=2.0,
                        vmin=vmin, vmax=vmax, zorder=10, alpha=1.0)

    # Find best and worst values
    z_max_idx = np.argmax(z_values)
    z_min_idx = np.argmin(z_values)

    # Annotate each point with its value
    for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
        if i == z_max_idx:
            # Best value
            color = 'darkgreen'
            fontsize = 11
            fontweight = 'bold'
        elif i == z_min_idx:
            # Worst value
            color = 'darkred'
            fontsize = 11
            fontweight = 'bold'
        else:
            color = 'black'
            fontsize = 10
            fontweight = 'bold'

        ax.annotate(f'{z:.4f}',
                   xy=(x, y), xytext=(0, -25),
                   textcoords='offset points',
                   ha='center', va='top',
                   fontsize=fontsize, fontweight=fontweight, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', alpha=0.9, linewidth=1.0))

    # Set labels
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    # Add title at bottom
    ax.text(0.5, -0.18, title, transform=ax.transAxes,
            fontsize=13, fontweight='bold', ha='center', va='top')

    # Set ticks to actual values
    ax.set_xticks(x_unique)
    ax.set_yticks(y_unique)

    # Grid
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5, color='gray', zorder=0)

    # Set axis limits with some padding
    x_range = x_unique.max() - x_unique.min()
    y_range = y_unique.max() - y_unique.min()
    ax.set_xlim(x_unique.min() - x_range * 0.1, x_unique.max() + x_range * 0.1)
    ax.set_ylim(y_unique.min() - y_range * 0.15, y_unique.max() + y_range * 0.15)

    # Enhanced tick formatting
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.0, length=5)

    return scatter


def main():
    """Generate 2D sensitivity analysis plots for both markets."""
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
            print(f"\nGenerating 2D contour plots for {market}...")
            plot_2d_contour(data, market)
        else:
            print(f"No data found for {market}")

    print(f"\n{'='*60}")
    print("All plots saved to sensitivity/plots/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
