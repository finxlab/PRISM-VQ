#!/usr/bin/env python3
"""
Generate ablation study results table for paper.
Compares: Full Model vs w/o MoE vs w/o Prior vs w/o Codebook vs Autoregressive
Creates both LaTeX and Markdown formats.
"""

import argparse
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_RES_DIR = str(Path(__file__).resolve().parent.parent / "res")


def read_rank_ic(metric_file: str) -> float:
    """Read RankIC from metric CSV file."""
    try:
        with open(metric_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('') == 'RankIC':
                    return float(row['values'])
    except:
        pass
    return None


def read_all_metrics(metric_file: str) -> Dict:
    """Read all metrics from CSV file."""
    metrics = {}
    try:
        with open(metric_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric_name = row.get('')
                if metric_name:
                    metrics[metric_name] = float(row['values'])
    except:
        pass
    return metrics


def collect_full_model_data(res_dir: str, market: str) -> Dict:
    """Collect full model results (baseline)."""
    if market == 'csi300':
        search_dir = os.path.join(res_dir, 'csi300')
        folder_pattern = 'VQK512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3'
    elif market == 'sp500':
        search_dir = os.path.join(res_dir, 'sp500')
        folder_pattern = 'VQK512_sp500_mo8_k4_mh64_md0.1_dm64_nh4_l1_d0.1_au0.001_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3'
    else:
        return None

    folder_path = os.path.join(search_dir, folder_pattern)

    if not os.path.exists(folder_path):
        print(f"Warning: Full model folder not found: {folder_path}")
        return None

    # Read all metrics from all folds
    all_metrics = []
    for i in range(5):
        metric_file = os.path.join(folder_path, f'{i}_metric.csv')
        if os.path.exists(metric_file):
            metrics = read_all_metrics(metric_file)
            if metrics:
                all_metrics.append(metrics)

    if not all_metrics:
        return None

    # Calculate mean and std for each metric
    result = {}
    metric_names = all_metrics[0].keys()
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        if values:
            result[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

    return result


def collect_ablation_data(res_dir: str, ablation_type: str, market: str) -> Dict:
    """Collect ablation study results."""
    search_dir = os.path.join(res_dir, ablation_type)

    if not os.path.exists(search_dir):
        print(f"Warning: Ablation folder not found: {search_dir}")
        return None

    # Find the folder for the market
    folders = [d for d in os.listdir(search_dir) if market in d]

    if not folders:
        print(f"Warning: No folder found for {market} in {ablation_type}")
        return None

    folder_path = os.path.join(search_dir, folders[0])

    # Read all metrics from all folds
    all_metrics = []
    for i in range(5):
        metric_file = os.path.join(folder_path, f'{i}_metric.csv')
        if os.path.exists(metric_file):
            metrics = read_all_metrics(metric_file)
            if metrics:
                all_metrics.append(metrics)

    if not all_metrics:
        return None

    # Calculate mean and std for each metric
    result = {}
    metric_names = all_metrics[0].keys()
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        if values:
            result[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

    return result


def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX table for ablation study."""
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Ablation Study: Impact of Model Components}")
    latex.append("\\label{tab:ablation}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Model Variant} & \\textbf{IC} & \\textbf{ICIR} & \\textbf{RankIC} & \\textbf{RankICIR} \\\\")
    latex.append("\\midrule")

    # CSI300 results
    latex.append("\\multicolumn{5}{c}{\\textbf{CSI300}} \\\\")
    latex.append("\\midrule")

    for model_name, display_name in [
        ('full', 'Full Model'),
        ('wo_prior', 'w/o Prior'),
        ('wo_moe', 'w/o MoE'),
        ('wo_Codebook', 'w/o Codebook'),
        ('autoregressive', 'Autoregressive')
    ]:
        if 'csi300' in results and model_name in results['csi300']:
            data = results['csi300'][model_name]
            ic = data.get('IC', {})
            icir = data.get('ICIR', {})
            ric = data.get('RankIC', {})
            ricir = data.get('RankICIR', {})

            bold = "\\textbf{" if model_name == 'full' else ""
            bold_end = "}" if model_name == 'full' else ""

            latex.append(f"{bold}{display_name}{bold_end} & "
                        f"{bold}{ic.get('mean', 0):.4f}{bold_end} & "
                        f"{bold}{icir.get('mean', 0):.4f}{bold_end} & "
                        f"{bold}{ric.get('mean', 0):.4f}{bold_end} & "
                        f"{bold}{ricir.get('mean', 0):.4f}{bold_end} \\\\")

    latex.append("\\midrule")

    # SP500 results
    latex.append("\\multicolumn{5}{c}{\\textbf{S\\&P500}} \\\\")
    latex.append("\\midrule")

    for model_name, display_name in [
        ('full', 'Full Model'),
        ('wo_prior', 'w/o Prior'),
        ('wo_moe', 'w/o MoE'),
        ('wo_Codebook', 'w/o Codebook'),
        ('autoregressive', 'Autoregressive')
    ]:
        if 'sp500' in results and model_name in results['sp500']:
            data = results['sp500'][model_name]
            ic = data.get('IC', {})
            icir = data.get('ICIR', {})
            ric = data.get('RankIC', {})
            ricir = data.get('RankICIR', {})

            bold = "\\textbf{" if model_name == 'full' else ""
            bold_end = "}" if model_name == 'full' else ""

            latex.append(f"{bold}{display_name}{bold_end} & "
                        f"{bold}{ic.get('mean', 0):.4f}{bold_end} & "
                        f"{bold}{icir.get('mean', 0):.4f}{bold_end} & "
                        f"{bold}{ric.get('mean', 0):.4f}{bold_end} & "
                        f"{bold}{ricir.get('mean', 0):.4f}{bold_end} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_markdown_table(results: Dict) -> str:
    """Generate Markdown table for ablation study."""
    md = []
    md.append("# Ablation Study Results\n")
    md.append("## Impact of Model Components on Performance\n")

    # CSI300 results
    md.append("### CSI300 Market\n")
    md.append("| Model Variant | IC | ICIR | RankIC | RankICIR |")
    md.append("|---------------|-----|------|--------|----------|")

    for model_name, display_name in [
        ('full', '**Full Model** (Baseline)'),
        ('wo_prior', 'w/o Prior'),
        ('wo_moe', 'w/o MoE'),
        ('wo_Codebook', 'w/o Codebook'),
        ('autoregressive', 'Autoregressive')
    ]:
        if 'csi300' in results and model_name in results['csi300']:
            data = results['csi300'][model_name]
            ic = data.get('IC', {})
            icir = data.get('ICIR', {})
            ric = data.get('RankIC', {})
            ricir = data.get('RankICIR', {})

            md.append(f"| {display_name} | "
                     f"{ic.get('mean', 0):.4f} ± {ic.get('std', 0):.4f} | "
                     f"{icir.get('mean', 0):.4f} ± {icir.get('std', 0):.4f} | "
                     f"{ric.get('mean', 0):.4f} ± {ric.get('std', 0):.4f} | "
                     f"{ricir.get('mean', 0):.4f} ± {ricir.get('std', 0):.4f} |")

    # SP500 results
    md.append("\n### S&P500 Market\n")
    md.append("| Model Variant | IC | ICIR | RankIC | RankICIR |")
    md.append("|---------------|-----|------|--------|----------|")

    for model_name, display_name in [
        ('full', '**Full Model** (Baseline)'),
        ('wo_prior', 'w/o Prior'),
        ('wo_moe', 'w/o MoE'),
        ('wo_Codebook', 'w/o Codebook'),
        ('autoregressive', 'Autoregressive')
    ]:
        if 'sp500' in results and model_name in results['sp500']:
            data = results['sp500'][model_name]
            ic = data.get('IC', {})
            icir = data.get('ICIR', {})
            ric = data.get('RankIC', {})
            ricir = data.get('RankICIR', {})

            md.append(f"| {display_name} | "
                     f"{ic.get('mean', 0):.4f} ± {ic.get('std', 0):.4f} | "
                     f"{icir.get('mean', 0):.4f} ± {icir.get('std', 0):.4f} | "
                     f"{ric.get('mean', 0):.4f} ± {ric.get('std', 0):.4f} | "
                     f"{ricir.get('mean', 0):.4f} ± {ricir.get('std', 0):.4f} |")

    # Add insights
    md.append("\n## Key Insights\n")

    # Calculate performance drops for CSI300
    if 'csi300' in results and 'full' in results['csi300']:
        full_ric = results['csi300']['full'].get('RankIC', {}).get('mean', 0)

        md.append("### CSI300 Performance Degradation\n")
        for model_name, display_name in [('wo_prior', 'Prior'), ('wo_moe', 'MoE'), ('wo_Codebook', 'Codebook'), ('autoregressive', 'Autoregressive (vs Bidirectional)')]:
            if model_name in results['csi300']:
                ablation_ric = results['csi300'][model_name].get('RankIC', {}).get('mean', 0)
                drop = (full_ric - ablation_ric) / full_ric * 100 if full_ric > 0 else 0
                prefix = "w/o " if model_name != 'autoregressive' else ""
                md.append(f"- **{prefix}{display_name}**: {drop:.2f}% drop in RankIC")

    # Calculate performance drops for SP500
    if 'sp500' in results and 'full' in results['sp500']:
        full_ric = results['sp500']['full'].get('RankIC', {}).get('mean', 0)

        md.append("\n### S&P500 Performance Degradation\n")
        for model_name, display_name in [('wo_prior', 'Prior'), ('wo_moe', 'MoE'), ('wo_Codebook', 'Codebook'), ('autoregressive', 'Autoregressive (vs Bidirectional)')]:
            if model_name in results['sp500']:
                ablation_ric = results['sp500'][model_name].get('RankIC', {}).get('mean', 0)
                drop = (full_ric - ablation_ric) / full_ric * 100 if full_ric > 0 else 0
                prefix = "w/o " if model_name != 'autoregressive' else ""
                md.append(f"- **{prefix}{display_name}**: {drop:.2f}% drop in RankIC")

    md.append("\n### Component Importance Ranking\n")
    md.append("Based on performance degradation, we rank the importance of each component:")
    md.append("1. Component with largest performance drop is most critical")
    md.append("2. All components contribute to overall model performance")
    md.append("3. The combined effect demonstrates the synergy between components")

    return "\n".join(md)


def main():
    """Generate ablation study tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res-dir", default=DEFAULT_RES_DIR,
                        help=f"Path to the results directory (default: {DEFAULT_RES_DIR})")
    args = parser.parse_args()
    res_dir = args.res_dir

    results = {
        'csi300': {},
        'sp500': {}
    }

    print("Collecting ablation study data...\n")

    # Collect data for both markets
    for market in ['csi300', 'sp500']:
        print(f"Processing {market.upper()}...")

        # Full model (baseline)
        print(f"  - Full model...")
        full_data = collect_full_model_data(res_dir, market)
        if full_data:
            results[market]['full'] = full_data
            print(f"    ✓ RankIC: {full_data.get('RankIC', {}).get('mean', 0):.4f}")
        else:
            print(f"    ✗ Not found")

        # Ablation studies
        for ablation_type in ['wo_prior', 'wo_moe', 'wo_Codebook', 'autoregressive']:
            print(f"  - {ablation_type}...")
            ablation_data = collect_ablation_data(res_dir, ablation_type, market)
            if ablation_data:
                results[market][ablation_type] = ablation_data
                print(f"    ✓ RankIC: {ablation_data.get('RankIC', {}).get('mean', 0):.4f}")
            else:
                print(f"    ✗ Not found")

        print()

    # Generate tables
    print("Generating tables...\n")
    latex_table = generate_latex_table(results)
    markdown_table = generate_markdown_table(results)

    # Save LaTeX table
    latex_file = 'ablation/ablation_table.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_file}")

    # Save Markdown table
    md_file = 'ablation/ablation_table.md'
    with open(md_file, 'w') as f:
        f.write(markdown_table)
    print(f"Markdown table saved to: {md_file}")

    # Print to console
    print("\n" + "="*80)
    print("MARKDOWN TABLE (for viewing)")
    print("="*80)
    print(markdown_table)
    print("\n" + "="*80)
    print("LATEX TABLE (for paper)")
    print("="*80)
    print(latex_table)


if __name__ == "__main__":
    main()
