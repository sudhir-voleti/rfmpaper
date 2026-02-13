#!/usr/bin/env python3
"""
smc_ablation_table.py
=====================
Generate ablation table from saved SMC results.
Reads .pkl files and displays log-evidence + power parameters in order of complexity.

Usage:
    python exec_gitcode.py \
        https://github.com/sudhir-voleti/rfmpaper/blob/main/smc_ablation_table.py \
        /path/to/results_dir \
        --dataset uci

Output:
    - Console table (log-ev, p values, complexity order)
    - CSV file for LaTeX import
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import re


def extract_from_pkl(pkl_path: Path) -> Dict:
    """Extract log-evidence and power parameters from pickle."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    idata = data['idata']
    res = data['res']
    
    # Basic info
    result = {
        'file': pkl_path.name,
        'dataset': res.get('dataset', 'unknown'),
        'K': res.get('K', 1),
        'N': res.get('N', 0),
        'D': res.get('draws', 0),
        'log_ev': res.get('log_evidence', np.nan),
        'state_specific_p': res.get('state_specific_p', False),
        'p_fixed': res.get('p_fixed', None),
        'use_gam': res.get('use_gam', True),
        'time_min': res.get('time_min', 0),
    }
    
    # Power parameters
    if 'p' in idata.posterior:
        p_vals = idata.posterior['p'].values
        # Mean across chains and draws, per state
        p_mean = p_vals.mean(axis=(0, 1))  # (K,) or scalar
        
        if result['K'] == 1:
            result['p_mean'] = float(p_mean)
            result['p_range'] = f"{p_mean:.3f}"
        else:
            result['p_mean'] = p_mean.mean()  # average across states
            result['p_range'] = f"{p_mean.min():.3f}-{p_mean.max():.3f}"
    else:
        result['p_mean'] = res.get('p_fixed', 1.5)
        result['p_range'] = f"{result['p_mean']:.3f}"
    
    # Model type label
    if result['K'] == 1:
        result['model_type'] = 'Static'
    else:
        result['model_type'] = f'HMM K={result["K"]}'
    
    # Complexity score for sorting
    result['complexity'] = result['K'] * 10 + (1 if result['state_specific_p'] else 0)
    
    return result


def parse_filename_info(filename: str) -> Dict:
    """Extract model info from filename for categorization."""
    info = {
        'is_statep': 'statep' in filename,
        'is_p_fixed': 'p1.5' in filename or 'p_' in filename,
        'K': 1
    }
    
    # Extract K
    k_match = re.search(r'K(\d+)', filename)
    if k_match:
        info['K'] = int(k_match.group(1))
    
    return info


def generate_ablation_table(results_dir: str, dataset: str = None) -> pd.DataFrame:
    """Generate complete ablation table from all pickles in directory."""
    
    results_path = Path(results_dir)
    
    # Find all .pkl files
    pkls = list(results_path.glob("smc_*.pkl"))
    
    if not pkls:
        raise FileNotFoundError(f"No .pkl files found in {results_dir}")
    
    print(f"Found {len(pkls)} pickle files")
    
    # Extract from all
    rows = []
    for pkl in sorted(pkls):
        try:
            info = extract_from_pkl(pkl)
            
            # Filter by dataset if specified
            if dataset and info['dataset'] != dataset:
                continue
            
            rows.append(info)
            print(f"  ✓ {pkl.name}: log-ev={info['log_ev']:.2f}, K={info['K']}, "
                  f"state_p={info['state_specific_p']}")
        except Exception as e:
            print(f"  ✗ {pkl.name}: {e}")
    
    if not rows:
        raise ValueError(f"No valid results found for dataset={dataset}")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by complexity (K, then state-specific p)
    df = df.sort_values(['dataset', 'complexity', 'log_ev'], 
                       ascending=[True, True, False])
    
    return df


def format_ablation_display(df: pd.DataFrame) -> str:
    """Format nice console display."""
    
    lines = []
    lines.append("=" * 100)
    lines.append("ABLATION TABLE: Model Comparison by Complexity")
    lines.append("=" * 100)
    
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        
        lines.append(f"\n{dataset.upper()}")
        lines.append("-" * 100)
        lines.append(f"{'Model':<20} {'K':>3} {'Type':<12} {'Log-Ev':>12} "
                    f"{'p (power)':<12} {'N':>5} {'D':>5} {'Time(min)':>10}")
        lines.append("-" * 100)
        
        for _, row in df_ds.iterrows():
            model_desc = row['model_type']
            if row['K'] > 1:
                if row['state_specific_p']:
                    model_desc += " (state-p)"
                else:
                    model_desc += f" (p={row['p_fixed']})"
            
            lines.append(f"{model_desc:<20} {row['K']:>3} "
                        f"{'GAM' if row['use_gam'] else 'GLM':<12} "
                        f"{row['log_ev']:>12.2f} "
                        f"{row['p_range']:<12} "
                        f"{row['N']:>5} {row['D']:>5} {row['time_min']:>10.1f}")
        
        # Add deltas
        if len(df_ds) > 1:
            base_logev = df_ds.iloc[0]['log_ev']
            lines.append("")
            lines.append("Improvement vs simplest model:")
            for i, row in df_ds.iloc[1:].iterrows():
                delta = row['log_ev'] - base_logev
                lines.append(f"  {row['model_type']}: +{delta:.2f} log-ev units")
    
    lines.append("=" * 100)
    
    return "\n".join(lines)


def export_for_latex(df: pd.DataFrame, out_path: Path):
    """Export clean CSV for LaTeX table import."""
    
    # Select and rename columns
    export = df[['dataset', 'model_type', 'K', 'state_specific_p', 
                'log_ev', 'p_range', 'N', 'D']].copy()
    
    export.columns = ['Dataset', 'Model', 'States', 'State_Specific_p', 
                     'Log_Evidence', 'p_Range', 'N', 'Draws']
    
    # Add delta column per dataset
    export['Delta_vs_K1'] = np.nan
    for dataset in export['Dataset'].unique():
        mask = export['Dataset'] == dataset
        base = export.loc[mask, 'Log_Evidence'].iloc[0]
        export.loc[mask, 'Delta_vs_K1'] = export.loc[mask, 'Log_Evidence'] - base
    
    export.to_csv(out_path, index=False, float_format='%.2f')
    print(f"\nExported LaTeX-ready CSV to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate ablation table from SMC results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('results_dir', type=str,
                       help='Directory containing .pkl result files')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['uci', 'cdnow'],
                       help='Filter to single dataset')
    parser.add_argument('--out_dir', type=str, default=None,
                       help='Output directory for CSV')
    
    args = parser.parse_args()
    
    # Generate table
    df = generate_ablation_table(args.results_dir, args.dataset)
    
    # Display
    print(format_ablation_display(df))
    
    # Export
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    export_for_latex(df, out_dir / 'ablation_table_formatted.csv')
    
    # Also save full data
    df.to_pickle(out_dir / 'ablation_data.pkl')
    print(f"Saved full data to: {out_dir / 'ablation_data.pkl'}")


if __name__ == "__main__":
    main()
