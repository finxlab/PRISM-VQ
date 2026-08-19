#!/usr/bin/env python3
"""
Generate sensitivity analysis results table for paper.
Creates both LaTeX and Markdown formats.
"""

import argparse
import os
import re
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

DEFAULT_RES_DIR = str(Path(__file__).resolve().parent.parent / "res")


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
    except:
        pass
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
                }

    return list(experiments.values())


def generate_latex_table(csi300_data: List[Dict], sp500_data: List[Dict]) -> str:
    """Generate LaTeX table for paper."""

    # Sort data
    csi300_data = sorted(csi300_data, key=lambda x: (x['dm'], x['VQK'], x['mo']))
    sp500_data = sorted(sp500_data, key=lambda x: (x['dm'], x['VQK'], x['mo']))

    # Create lookup dictionaries
    csi300_dict = {(d['VQK'], d['mo'], d['dm']): d for d in csi300_data}
    sp500_dict = {(d['VQK'], d['mo'], d['dm']): d for d in sp500_data}

    # Find best results
    best_csi300 = max(csi300_data, key=lambda x: x['RankIC'])
    best_sp500 = max(sp500_data, key=lambda x: x['RankIC'])

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Hyperparameter Sensitivity Analysis: RankIC Performance}")
    latex.append("\\label{tab:sensitivity}")
    latex.append("\\begin{tabular}{ccc|cc|cc}")
    latex.append("\\toprule")
    latex.append("\\multicolumn{3}{c|}{\\textbf{Hyperparameters}} & \\multicolumn{2}{c|}{\\textbf{CSI300}} & \\multicolumn{2}{c}{\\textbf{S\\&P500}} \\\\")
    latex.append("\\textbf{Codebook} & \\textbf{$d_{model}$} & \\textbf{MoE} & \\textbf{RankIC} & \\textbf{Std} & \\textbf{RankIC} & \\textbf{Std} \\\\")
    latex.append("\\midrule")

    # Group by dm
    for dm in [32, 64]:
        latex.append(f"\\multicolumn{{7}}{{c}}{{\\textit{{$d_{{model}}$ = {dm}}}}} \\\\")
        for vqk in [128, 256, 512]:
            for mo in [2, 4, 8]:
                key = (vqk, mo, dm)

                csi_data = csi300_dict.get(key)
                sp_data = sp500_dict.get(key)

                if csi_data and sp_data:
                    # Check if this is the best result
                    csi_best = (csi_data['VQK'] == best_csi300['VQK'] and
                               csi_data['mo'] == best_csi300['mo'] and
                               csi_data['dm'] == best_csi300['dm'])
                    sp_best = (sp_data['VQK'] == best_sp500['VQK'] and
                              sp_data['mo'] == best_sp500['mo'] and
                              sp_data['dm'] == best_sp500['dm'])

                    csi_val = f"\\textbf{{{csi_data['RankIC']:.4f}}}" if csi_best else f"{csi_data['RankIC']:.4f}"
                    sp_val = f"\\textbf{{{sp_data['RankIC']:.4f}}}" if sp_best else f"{sp_data['RankIC']:.4f}"

                    latex.append(f"{vqk} & {dm} & {mo} & {csi_val} & {csi_data['RankIC_std']:.4f} & {sp_val} & {sp_data['RankIC_std']:.4f} \\\\")

        if dm == 32:
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_markdown_table(csi300_data: List[Dict], sp500_data: List[Dict]) -> str:
    """Generate Markdown table for viewing."""

    # Sort data
    csi300_data = sorted(csi300_data, key=lambda x: (x['dm'], x['VQK'], x['mo']))
    sp500_data = sorted(sp500_data, key=lambda x: (x['dm'], x['VQK'], x['mo']))

    # Create lookup dictionaries
    csi300_dict = {(d['VQK'], d['mo'], d['dm']): d for d in csi300_data}
    sp500_dict = {(d['VQK'], d['mo'], d['dm']): d for d in sp500_data}

    # Find best results
    best_csi300 = max(csi300_data, key=lambda x: x['RankIC'])
    best_sp500 = max(sp500_data, key=lambda x: x['RankIC'])

    md = []
    md.append("# Hyperparameter Sensitivity Analysis Results\n")
    md.append("## RankIC Performance Across Different Configurations\n")
    md.append("| Codebook | d_model | MoE | CSI300 RankIC | CSI300 Std | S&P500 RankIC | S&P500 Std |")
    md.append("|----------|---------|-----|---------------|------------|---------------|------------|")

    # Group by dm
    for dm in [32, 64]:
        md.append(f"| **d_model = {dm}** | | | | | | |")
        for vqk in [128, 256, 512]:
            for mo in [2, 4, 8]:
                key = (vqk, mo, dm)

                csi_data = csi300_dict.get(key)
                sp_data = sp500_dict.get(key)

                if csi_data and sp_data:
                    # Check if this is the best result
                    csi_best = (csi_data['VQK'] == best_csi300['VQK'] and
                               csi_data['mo'] == best_csi300['mo'] and
                               csi_data['dm'] == best_csi300['dm'])
                    sp_best = (sp_data['VQK'] == best_sp500['VQK'] and
                              sp_data['mo'] == best_sp500['mo'] and
                              sp_data['dm'] == best_sp500['dm'])

                    csi_val = f"**{csi_data['RankIC']:.4f}**" if csi_best else f"{csi_data['RankIC']:.4f}"
                    sp_val = f"**{sp_data['RankIC']:.4f}**" if sp_best else f"{sp_data['RankIC']:.4f}"

                    md.append(f"| {vqk} | {dm} | {mo} | {csi_val} | {csi_data['RankIC_std']:.4f} | {sp_val} | {sp_data['RankIC_std']:.4f} |")

    md.append("\n## Key Insights\n")
    md.append(f"### CSI300 Best Configuration")
    md.append(f"- **Codebook Size**: {best_csi300['VQK']}")
    md.append(f"- **d_model**: {best_csi300['dm']}")
    md.append(f"- **MoE Experts**: {best_csi300['mo']}")
    md.append(f"- **RankIC**: {best_csi300['RankIC']:.4f} ± {best_csi300['RankIC_std']:.4f}\n")

    md.append(f"### S&P500 Best Configuration")
    md.append(f"- **Codebook Size**: {best_sp500['VQK']}")
    md.append(f"- **d_model**: {best_sp500['dm']}")
    md.append(f"- **MoE Experts**: {best_sp500['mo']}")
    md.append(f"- **RankIC**: {best_sp500['RankIC']:.4f} ± {best_sp500['RankIC_std']:.4f}\n")

    md.append("### Observations")
    md.append(f"1. **Codebook Size Impact**: Both markets achieve best performance with codebook size = 512")
    md.append(f"2. **d_model Impact**: Both markets prefer d_model = 64 over 32")
    md.append(f"3. **MoE Experts**: CSI300 benefits from fewer experts (mo=2), while S&P500 benefits from more experts (mo=8)")
    md.append(f"4. **Market Differences**: CSI300 shows significantly higher RankIC ({best_csi300['RankIC']:.4f}) compared to S&P500 ({best_sp500['RankIC']:.4f})")

    return "\n".join(md)


def main():
    """Generate sensitivity analysis tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res-dir", default=DEFAULT_RES_DIR,
                        help=f"Path to the results directory (default: {DEFAULT_RES_DIR})")
    args = parser.parse_args()
    res_dir = args.res_dir

    print("Collecting data...")
    csi300_data = collect_sensitivity_data(res_dir, 'csi300')
    sp500_data = collect_sensitivity_data(res_dir, 'sp500')

    print(f"Found {len(csi300_data)} CSI300 experiments")
    print(f"Found {len(sp500_data)} SP500 experiments\n")

    # Generate tables
    latex_table = generate_latex_table(csi300_data, sp500_data)
    markdown_table = generate_markdown_table(csi300_data, sp500_data)

    # Save LaTeX table
    latex_file = 'sensitivity/sensitivity_table.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_file}")

    # Save Markdown table
    md_file = 'sensitivity/sensitivity_table.md'
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
