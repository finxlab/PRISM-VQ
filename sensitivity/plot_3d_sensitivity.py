#!/usr/bin/env python3
"""
3D Sensitivity Analysis for VQK, d_model, and MoE experts
Generates 3D surface plots showing how RankIC varies with these hyperparameters.
IEEE TKDE style optimized.
"""

import argparse
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import matplotlib
from matplotlib.patches import FancyBboxPatch

DEFAULT_RES_DIR = str(Path(__file__).resolve().parent.parent / "res")

# IEEE style configuration
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 12
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Clean white background for IEEE style
plt.style.use('default')
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = 0.8
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
    Only includes VQK=[128,256,512], dm=64 (best baseline), mo=[2,4,8]
    Only looks in specific directories: res/csi300/ (nh2) and res/sp500/ (nh4)
    Selects exactly one experiment per (VQK, mo, dm) combination.
    """
    # Define the hyperparameters we want to analyze
    VALID_VQK = [128, 256, 512]
    VALID_DM = [32, 64]  # Both dm=32 and dm=64 for comparison
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


def plot_3d_surface(
    data: List[Dict],
    market: str,
    save_dir: str = "sensitivity/plots",
    include_suptitle: bool = False,
):
    if not data:
        print(f"No data available for {market}")
        return

    # IEEE 2-column width: ~3.5 inches per column, full width ~7.16 inches
    fig = plt.figure(figsize=(18, 5.5))

    BEST_DM = 64
    BEST_VQK = 512
    BEST_MO = 2 if market == 'csi300' else 8

    all_ric = np.array([d['RankIC'] for d in data])
    vmin, vmax = all_ric.min(), all_ric.max()

    ax1 = fig.add_subplot(131, projection='3d')
    filtered_data1 = [d for d in data if d['dm'] == BEST_DM]
    scatter1 = None
    if filtered_data1:
        vqk_vals = np.array([d['VQK'] for d in filtered_data1])
        mo_vals = np.array([d['mo'] for d in filtered_data1])
        ric_vals = np.array([d['RankIC'] for d in filtered_data1])
        scatter1 = plot_3d_subplot(
            ax1, vqk_vals, mo_vals, ric_vals,
            'Codebook Size', 'MoE Experts', 'RankIC',
            '(a) Codebook vs MoE', vmin, vmax, add_colorbar=False
        )
        ax1.text2D(0.5, -0.10, '(a) Codebook vs MoE', transform=ax1.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top')

    ax2 = fig.add_subplot(132, projection='3d')
    filtered_data2 = [d for d in data if d['mo'] == BEST_MO]
    if filtered_data2:
        vqk_vals = np.array([d['VQK'] for d in filtered_data2])
        dm_vals = np.array([d['dm'] for d in filtered_data2])
        ric_vals = np.array([d['RankIC'] for d in filtered_data2])
        scatter2 = plot_3d_subplot(
            ax2, vqk_vals, dm_vals, ric_vals,
            'Codebook Size', '$d_t$', 'RankIC',
            '(b) Codebook vs $d_t$', vmin, vmax, add_colorbar=False
        )
        ax2.text2D(0.5, -0.10, '(b) Codebook vs $d_t$', transform=ax2.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top')

    ax3 = fig.add_subplot(133, projection='3d')
    filtered_data3 = [d for d in data if d['VQK'] == BEST_VQK]
    if filtered_data3:
        dm_vals = np.array([d['dm'] for d in filtered_data3])
        mo_vals = np.array([d['mo'] for d in filtered_data3])
        ric_vals = np.array([d['RankIC'] for d in filtered_data3])
        scatter3 = plot_3d_subplot(
            ax3, dm_vals, mo_vals, ric_vals,
            '$d_t$', 'MoE Experts', 'RankIC',
            '(c) $d_t$ vs MoE', vmin, vmax, add_colorbar=False
        )
        ax3.text2D(0.5, -0.10, '(c) $d_t$ vs MoE', transform=ax3.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top')

    # Enhanced colorbar with better visibility
    if filtered_data1:
        cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.65])
        cbar = fig.colorbar(scatter1, cax=cbar_ax)
        cbar.set_label('RankIC', fontsize=11, weight='bold')
        cbar.ax.tick_params(labelsize=9)
        cbar.outline.set_linewidth(0.8)

    if include_suptitle:
        market_name = 'CSI300' if market == 'csi300' else 'S&P500'
        fig.suptitle(f'{market_name}: Hyperparameter Sensitivity Analysis',
                     fontsize=13, y=0.98, fontweight='bold')
        plt.subplots_adjust(left=0.04, right=0.91, top=0.92, bottom=0.10, wspace=0.25)
    else:
        plt.subplots_adjust(left=0.04, right=0.91, top=0.96, bottom=0.10, wspace=0.25)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'3d_sensitivity_{market}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved 3D plot: {out_path}")
    plt.close(fig)


def plot_3d_subplot(ax, x_values, y_values, z_values, xlabel, ylabel, zlabel, title,
                    vmin=None, vmax=None, add_colorbar=True):
    """Plot 3D scatter with surface interpolation and enhanced value visibility."""

    # Use a professional colormap suitable for papers
    cmap = 'Blues'  # Professional single-color gradient

    # Create scatter plot with larger markers and better contrast
    scatter = ax.scatter(x_values, y_values, z_values, c=z_values,
                        cmap=cmap, s=250, alpha=0.95, edgecolors='navy',
                        linewidths=1.5, vmin=vmin, vmax=vmax, depthshade=True)

    # Find best and worst values for highlighting
    z_max_idx = np.argmax(z_values)
    z_min_idx = np.argmin(z_values)

    # Calculate smart text offset based on data range to avoid overlap
    z_range = z_values.max() - z_values.min()
    z_offset = z_range * 0.08  # 8% of range for vertical offset

    # Group points by y-value to handle overlaps
    y_unique = sorted(set(y_values))
    y_spacing = {}
    for y_val in y_unique:
        mask = y_values == y_val
        indices = np.where(mask)[0]
        y_spacing[y_val] = indices

    # Add value labels with smart positioning to avoid overlaps
    for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
        # Determine text properties based on value ranking
        if i == z_max_idx:
            # Highlight best value
            fontsize = 11
            fontweight = 'bold'
            color = 'darkgreen'
        elif i == z_min_idx:
            # Highlight worst value
            fontsize = 11
            fontweight = 'bold'
            color = 'darkred'
        else:
            # Regular values
            fontsize = 10
            fontweight = 'bold'
            color = 'black'

        # Smart positioning: offset based on x position within same y-group
        indices_in_group = y_spacing[y]
        position_in_group = np.where(indices_in_group == i)[0][0]
        num_in_group = len(indices_in_group)

        # Alternate left/right for same y-value to reduce overlap
        if num_in_group > 1:
            if position_in_group == 0:
                ha_align = 'right'
                x_offset = -0.02 * (x_values.max() - x_values.min())
            elif position_in_group == num_in_group - 1:
                ha_align = 'left'
                x_offset = 0.02 * (x_values.max() - x_values.min())
            else:
                ha_align = 'center'
                x_offset = 0
        else:
            ha_align = 'center'
            x_offset = 0

        # Position text with offset
        ax.text(x + x_offset, y, z + z_offset, f'{z:.4f}',
                fontsize=fontsize, fontweight=fontweight, color=color,
                ha=ha_align, va='bottom',
                zorder=1000)  # Ensure text is on top

    # Try to create surface interpolation with contour lines if we have enough points
    if len(x_values) >= 4:
        try:
            # Create grid for interpolation
            xi = np.linspace(x_values.min(), x_values.max(), 30)
            yi = np.linspace(y_values.min(), y_values.max(), 30)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate
            zi = griddata((x_values, y_values), z_values, (xi, yi), method='cubic')

            # Plot surface with transparency
            surf = ax.plot_surface(xi, yi, zi, cmap=cmap, alpha=0.25,
                                  linewidth=0, antialiased=True, vmin=vmin, vmax=vmax)

            # Add contour lines on the surface for better readability
            # Project contours at different z-levels
            contour_levels = np.linspace(np.nanmin(zi), np.nanmax(zi), 8)

            # Contour lines on the surface
            ax.contour(xi, yi, zi, levels=contour_levels, colors='darkblue',
                      linewidths=0.8, alpha=0.6, linestyles='solid')

            # Optional: Add contour projection on the bottom plane for reference
            offset_z = z_values.min() - (z_values.max() - z_values.min()) * 0.05
            ax.contour(xi, yi, zi, levels=contour_levels, colors='gray',
                      linewidths=0.5, alpha=0.3, linestyles='dashed',
                      offset=offset_z, zdir='z')

        except Exception as e:
            pass  # Silently skip if interpolation fails

    # Labels with better formatting
    ax.set_xlabel(xlabel, fontsize=11, labelpad=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, labelpad=10, fontweight='bold')
    ax.set_zlabel(zlabel, fontsize=11, labelpad=10, fontweight='bold')

    # Set ticks to actual values only
    x_unique = sorted(set(x_values))
    y_unique = sorted(set(y_values))
    ax.set_xticks(x_unique)
    ax.set_yticks(y_unique)

    # Enhanced tick formatting
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0, length=5)
    ax.tick_params(axis='z', which='major', labelsize=9, pad=5)

    # Adjust viewing angle for better visibility of labels
    ax.view_init(elev=22, azim=50)

    # Enhanced grid
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.6, color='gray')

    # Set pane colors to white for cleaner look
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.9)
    ax.yaxis.pane.set_alpha(0.9)
    ax.zaxis.pane.set_alpha(0.9)

    return scatter



def _grid_from_points(x_vals, y_vals, z_vals, x_ticks, y_ticks):
    """
    Build a (len(y_ticks), len(x_ticks)) grid with NaN for missing entries.
    y is rows, x is columns.
    """
    grid = np.full((len(y_ticks), len(x_ticks)), np.nan, dtype=float)
    x_to_idx = {x: j for j, x in enumerate(x_ticks)}
    y_to_idx = {y: i for i, y in enumerate(y_ticks)}
    for x, y, z in zip(x_vals, y_vals, z_vals):
        if x in x_to_idx and y in y_to_idx:
            grid[y_to_idx[y], x_to_idx[x]] = z
    return grid

def _annotate_heatmap(ax, grid, vmin, vmax):
    """Add cell annotations with appropriate text color."""
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                val = grid[i, j]
                norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                text_color = 'white' if norm_val > 0.5 else 'black'
                ax.text(j, i, f"{val:.4f}", ha='center', va='center',
                        fontsize=9, fontweight='bold', color=text_color)


def _setup_heatmap_ax(ax, grid, x_ticks, y_ticks, vmin, vmax, xlabel, ylabel):
    """Setup a single heatmap panel."""
    im = ax.imshow(grid, origin='lower', aspect='equal', interpolation='nearest',
                   vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks)
    ax.set_yticks(range(len(y_ticks)))
    ax.set_yticklabels(y_ticks)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(x_ticks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_ticks), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

    _annotate_heatmap(ax, grid, vmin, vmax)
    return im


def plot_2d_grid_heatmaps(
    data,
    market: str,
    save_dir: str = "sensitivity/plots",
    fname_prefix: str = "2d_grid_sensitivity",
):
    """
    Clean heatmap for IJCAI-style publication.
    3 panels: (a) Codebook vs MoE, (b) Codebook vs d, (c) d vs MoE
    """
    if not data:
        print(f"No data available for {market}")
        return

    BEST_DM = 64
    BEST_VQK = 512
    BEST_MO = 2 if market == 'csi300' else 8

    all_ric = np.array([d['RankIC'] for d in data], dtype=float)
    vmin, vmax = float(np.nanmin(all_ric)), float(np.nanmax(all_ric))

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    plt.subplots_adjust(wspace=0.35, left=0.06, right=0.88, top=0.88)

    # (a) Codebook vs MoE at dm=BEST_DM
    fd = [d for d in data if d['dm'] == BEST_DM]
    x_ticks = sorted({d['VQK'] for d in fd})
    y_ticks = sorted({d['mo'] for d in fd})
    grid = _grid_from_points(
        np.array([d['VQK'] for d in fd]),
        np.array([d['mo'] for d in fd]),
        np.array([d['RankIC'] for d in fd]),
        x_ticks, y_ticks
    )
    _setup_heatmap_ax(axes[0], grid, x_ticks, y_ticks, vmin, vmax,
                      "Codebook Size", "MoE Experts")

    # (b) Codebook vs d at mo=BEST_MO
    fd = [d for d in data if d['mo'] == BEST_MO]
    x_ticks = sorted({d['VQK'] for d in fd})
    y_ticks = sorted({d['dm'] for d in fd})
    grid = _grid_from_points(
        np.array([d['VQK'] for d in fd]),
        np.array([d['dm'] for d in fd]),
        np.array([d['RankIC'] for d in fd]),
        x_ticks, y_ticks
    )
    _setup_heatmap_ax(axes[1], grid, x_ticks, y_ticks, vmin, vmax,
                      "Codebook Size", r"$d_t$")

    # (c) d vs MoE at VQK=BEST_VQK
    fd = [d for d in data if d['VQK'] == BEST_VQK]
    x_ticks = sorted({d['dm'] for d in fd})
    y_ticks = sorted({d['mo'] for d in fd})
    grid = _grid_from_points(
        np.array([d['dm'] for d in fd]),
        np.array([d['mo'] for d in fd]),
        np.array([d['RankIC'] for d in fd]),
        x_ticks, y_ticks
    )
    im = _setup_heatmap_ax(axes[2], grid, x_ticks, y_ticks, vmin, vmax,
                           r"$d_t$", "MoE Experts")

    # Titles at fixed y position
    title_y = 0.95
    fig.text(0.18, title_y, "(a) Codebook vs MoE", ha='center', fontsize=11)
    fig.text(0.48, title_y, r"(b) Codebook vs $d_t$", ha='center', fontsize=11)
    fig.text(0.78, title_y, r"(c) $d_t$ vs MoE", ha='center', fontsize=11)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("RankIC")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{fname_prefix}_{market}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved 2D grid heatmap: {out_path}")

def main():
    """Generate 3D sensitivity analysis plots for both markets."""
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
            print(f"\nGenerating 3D plots for {market}...")
            # plot_3d_surface(data, market)
            plot_2d_grid_heatmaps(data, market)
        else:
            print(f"No data found for {market}")

    print(f"\n{'='*60}")
    print("All plots saved to sensitivity/plots/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
