#!/usr/bin/env python3
"""
smc_unified_postproc_full.py
============================
Comprehensive post-processing for HMM-Tweedie-SMC full runs.
Extracts all quantities needed for Marketing Science Sections 4-6.

Usage:
    python exec_gitcode.py \
        https://github.com/sudhir-voleti/rfmpaper/blob/main/smc_unified_postproc_full.py \
        /path/to/results_dir \
        --dataset uci \
        --output_format all

Output:
    - CSV tables for LaTeX
    - Pickled extraction objects
    - Summary reports
"""

import argparse
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

STATE_LABELS = {0: 'Cold', 1: 'Warm', 2: 'Hot', 3: 'Whale', 4: 'VIP'}

OUTPUT_FORMATS = ['csv', 'latex', 'json', 'pkl', 'all']


# =============================================================================
# CORE EXTRACTION FUNCTIONS
# =============================================================================

def load_results(pkl_path: str) -> Tuple[object, Dict]:
    """Load idata and metadata from SMC result pickle."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['idata'], data['res']


def summarize_posterior(arr: np.ndarray, ci: float = 0.94) -> Dict:
    """Compute posterior summaries: mean, sd, median, credible interval."""
    flat = arr.flatten()
    alpha = (1 - ci) / 2
    return {
        'mean': float(np.mean(flat)),
        'sd': float(np.std(flat)),
        'median': float(np.median(flat)),
        'lo': float(np.quantile(flat, alpha)),
        'hi': float(np.quantile(flat, 1 - alpha)),
        'n_eff': len(flat)
    }


def extract_state_emissions(idata: object, K: int) -> pd.DataFrame:
    """
    Extract state-specific emission parameters for Table 6.
    
    Returns: beta0, phi, p, psi, spend with credible intervals
    """
    rows = []
    
    for k in range(K):
        row = {'state': k, 'label': STATE_LABELS.get(k, f'S{k}')}
        
        # beta0 (baseline log-spend)
        beta0 = idata.posterior['beta0'].values[:, :, k]
        s = summarize_posterior(beta0)
        row.update({f'beta0_{k}': s[k] for k in ['mean', 'sd', 'lo', 'hi']})
        
        # spend = exp(beta0)
        spend_samples = np.exp(beta0)
        s_spend = summarize_posterior(spend_samples)
        row.update({
            'spend_mean': s_spend['mean'],
            'spend_lo': s_spend['lo'],
            'spend_hi': s_spend['hi']
        })
        
        # phi (dispersion)
        phi = idata.posterior['phi'].values[:, :, k]
        s = summarize_posterior(phi)
        row.update({f'phi_{k}': s[k] for k in ['mean', 'sd', 'lo', 'hi']})
        
        # p (power parameter)
        if 'p' in idata.posterior:
            p = idata.posterior['p'].values[:, :, k]
            s = summarize_posterior(p)
            row.update({f'p_{k}': s[k] for k in ['mean', 'sd', 'lo', 'hi']})
            p_mean = s['mean']
        else:
            p_mean = 1.5
            row['p_mean'] = p_mean
        
        # psi (structural zero probability) - ZIG formula
        exponent = 2.0 - p_mean
        psi = np.exp(-(s_spend['mean'] ** exponent) / (s['mean'] * exponent))
        row['psi_mean'] = float(psi)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_transitions(idata: object, K: int) -> Dict:
    """
    Extract transition dynamics for Table 5.
    
    Returns: Gamma matrix, persistence, dwell times, resurrection probabilities
    """
    Gamma = idata.posterior['Gamma'].values  # (chains, draws, K, K)
    Gamma_mean = Gamma.mean(axis=(0, 1))
    Gamma_std = Gamma.std(axis=(0, 1))
    Gamma_lo = np.percentile(Gamma, 2.5, axis=(0, 1))
    Gamma_hi = np.percentile(Gamma, 97.5, axis=(0, 1))
    
    # Persistence (diagonal elements)
    gamma_diag = np.einsum('...kk->...k', Gamma)
    
    persistence = {}
    dwell_times = {}
    
    for k in range(K):
        s = summarize_posterior(gamma_diag[:, :, k])
        persistence[k] = s
        dwell_times[k] = 1 / (1 - s['mean']) if s['mean'] < 0.999 else 999.0
    
    # Resurrection and abandonment probabilities
    resurrection = {}
    if K >= 3:
        resurrection['cold_to_hot'] = float(Gamma[:, :, 0, 2].mean())
        resurrection['hot_to_cold'] = float(Gamma[:, :, 2, 0].mean())
    
    return {
        'Gamma_mean': Gamma_mean,
        'Gamma_std': Gamma_std,
        'Gamma_lo': Gamma_lo,
        'Gamma_hi': Gamma_hi,
        'persistence': persistence,
        'dwell_times': dwell_times,
        'resurrection': resurrection
    }


def extract_clv_metrics(idata: object, K: int) -> pd.DataFrame:
    """
    Extract CLV proxy and churn risk by state.
    """
    rows = []
    
    for k in range(K):
        row = {'state': k, 'label': STATE_LABELS.get(k, f'S{k}')}
        
        # CLV proxy
        if 'clv_proxy' in idata.posterior:
            clv = idata.posterior['clv_proxy'].values[:, :, k]
            s = summarize_posterior(clv)
            row.update({
                'clv_mean': s['mean'],
                'clv_sd': s['sd'],
                'clv_lo': s['lo'],
                'clv_hi': s['hi']
            })
        
        # Churn risk
        if 'churn_risk' in idata.posterior:
            churn = idata.posterior['churn_risk'].values[:, :, k]
            s = summarize_posterior(churn)
            row.update({
                'churn_mean': s['mean'],
                'churn_sd': s['sd'],
                'churn_lo': s['lo'],
                'churn_hi': s['hi']
            })
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def compute_stationary_distribution(Gamma_mean: np.ndarray) -> np.ndarray:
    """Compute stationary distribution from transition matrix."""
    K = Gamma_mean.shape[0]
    eigvals, eigvecs = np.linalg.eig(Gamma_mean.T)
    stationary = eigvecs[:, np.isclose(eigvals, 1)].real
    stationary = stationary / stationary.sum()
    return stationary.flatten()


def format_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Format DataFrame as LaTeX table."""
    latex = df.to_latex(index=False, float_format='%.3f')
    return f"\\begin{{table}}[ht]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{latex}\n\\end{{table}}"


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_dataset(results_dir: str, dataset: str, 
                   output_format: str = 'all', out_dir: Optional[str] = None) -> Dict:
    """
    Process single dataset and generate all outputs.
    """
    results_path = Path(results_dir)
    
    # Find .pkl file
    patterns = [
        f"smc_{dataset}_K*_GAM_*_N*_D*.pkl",
        f"smc_K*_GAM_*_N*_D*_C*.pkl"
    ]
    
    pkls = []
    for pattern in patterns:
        pkls.extend(list(results_path.glob(pattern)))
    
    if not pkls:
        raise FileNotFoundError(f"No .pkl files found in {results_dir} for {dataset}")
    
    pkl_path = max(pkls, key=lambda p: p.stat().st_mtime)
    print(f"Processing: {pkl_path.name}")
    
    # Load
    idata, metadata = load_results(str(pkl_path))
    K = idata.posterior['Gamma'].shape[2]
    
    print(f"  Dataset: {metadata.get('dataset', dataset)}")
    print(f"  N: {metadata.get('N', 'unknown')}, K: {K}")
    print(f"  Log-evidence: {metadata.get('log_evidence', 'unknown'):.2f}")
    
    # Extract
    print("  Extracting emissions...")
    emissions = extract_state_emissions(idata, K)
    
    print("  Extracting transitions...")
    transitions = extract_transitions(idata, K)
    
    print("  Extracting CLV metrics...")
    clv = extract_clv_metrics(idata, K)
    
    # Stationary distribution
    pi_inf = compute_stationary_distribution(transitions['Gamma_mean'])
    
    # Compile results
    results = {
        'metadata': metadata,
        'dataset': dataset,
        'K': K,
        'emissions': emissions,
        'transitions': transitions,
        'clv': clv,
        'stationary': pi_inf
    }
    
    # Output directory
    out_path = Path(out_dir) if out_dir else results_path
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save outputs
    base_name = f"{dataset}_K{K}_full"
    
    if output_format in ['csv', 'all']:
        emissions.to_csv(out_path / f"{base_name}_emissions.csv", index=False)
        clv.to_csv(out_path / f"{base_name}_clv.csv", index=False)
        pd.DataFrame(transitions['Gamma_mean']).to_csv(
            out_path / f"{base_name}_Gamma.csv"
        )
        print(f"  Saved CSV files to {out_path}")
    
    if output_format in ['pkl', 'all']:
        with open(out_path / f"{base_name}_extracted.pkl", 'wb') as f:
            pickle.dump(results, f)
        print(f"  Saved pickle: {base_name}_extracted.pkl")
    
    if output_format in ['json', 'all']:
        # Convert numpy arrays to lists for JSON
        json_results = {
            'metadata': metadata,
            'emissions': emissions.to_dict(),
            'clv': clv.to_dict(),
            'stationary': pi_inf.tolist()
        }
        with open(out_path / f"{base_name}_summary.json", 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    
    if output_format in ['latex', 'all']:
        # Generate LaTeX tables
        latex_emissions = format_latex_table(
            emissions[['state', 'label', 'spend_mean', 'p_mean', 'phi_mean']],
            f"State Emission Parameters ({dataset.upper()})",
            f"tab:{dataset}_emissions"
        )
        with open(out_path / f"{base_name}_emissions.tex", 'w') as f:
            f.write(latex_emissions)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataset.upper()}")
    print(f"{'='*60}")
    print(f"\nState Emissions:")
    print(emissions[['state', 'label', 'spend_mean', 'p_mean']].to_string())
    
    print(f"\nTransition Persistence:")
    for k in range(K):
        pers = transitions['persistence'][k]
        dwell = transitions['dwell_times'][k]
        print(f"  {STATE_LABELS.get(k, f'S{k}')}: "
              f"γ={pers['mean']:.3f}, Dwell={dwell:.1f}w")
    
    if 'resurrection' in transitions and transitions['resurrection']:
        print(f"\nResurrection (Cold→Hot): "
              f"{transitions['resurrection'].get('cold_to_hot', 0):.3f}")
    
    print(f"\nStationary Distribution: {pi_inf.round(3)}")
    print(f"{'='*60}\n")
    
    return results


def compare_datasets(uci_results: Dict, cdnow_results: Dict, 
                    out_dir: Path) -> pd.DataFrame:
    """Generate comparison table for paper."""
    
    comparison_data = []
    
    # Basic info
    comparison_data.extend([
        ['Log-Evidence', 
         f"{uci_results['metadata'].get('log_evidence', 0):.2f}",
         f"{cdnow_results['metadata'].get('log_evidence', 0):.2f}"],
        ['Sample Size (N)',
         str(uci_results['metadata'].get('N', 'N/A')),
         str(cdnow_results['metadata'].get('N', 'N/A'))],
        ['States (K)',
         str(uci_results['K']),
         str(cdnow_results['K'])]
    ])
    
    # State comparisons
    for k in range(min(uci_results['K'], cdnow_results['K'])):
        uci_row = uci_results['emissions'].iloc[k]
        cdnow_row = cdnow_results['emissions'].iloc[k]
        
        comparison_data.extend([
            [f'State {k} Spend ($)',
             f"{uci_row['spend_mean']:.2f}",
             f"{cdnow_row['spend_mean']:.2f}"],
            [f'State {k} Power (p)',
             f"{uci_row['p_mean']:.3f}",
             f"{cdnow_row['p_mean']:.3f}"],
            [f'State {k} Persistence',
             f"{uci_results['transitions']['persistence'][k]['mean']:.3f}",
             f"{cdnow_results['transitions']['persistence'][k]['mean']:.3f}"]
        ])
    
    df = pd.DataFrame(comparison_data, columns=['Metric', 'UCI', 'CDNOW'])
    
    # Save
    df.to_csv(out_dir / 'comparison_uci_vs_cdnow.csv', index=False)
    print(f"\nComparison saved to: {out_dir / 'comparison_uci_vs_cdnow.csv'}")
    print(df.to_string())
    
    return df


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Full post-processing for HMM-Tweedie-SMC (Sections 4-6)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('results_dir', type=str,
                       help='Directory containing .pkl result files')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['uci', 'cdnow', 'both'],
                       help='Which dataset to process')
    parser.add_argument('--output_format', type=str, default='all',
                       choices=OUTPUT_FORMATS,
                       help='Output format(s)')
    parser.add_argument('--out_dir', type=str, default=None,
                       help='Output directory (default: same as results_dir)')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir)
    
    # Process single or both datasets
    if args.dataset == 'both':
        print("Processing UCI...")
        uci = process_dataset(args.results_dir, 'uci', args.output_format, str(out_dir))
        
        print("\nProcessing CDNOW...")
        cdnow = process_dataset(args.results_dir, 'cdnow', args.output_format, str(out_dir))
        
        print("\nGenerating comparison...")
        compare_datasets(uci, cdnow, out_dir)
        
    else:
        process_dataset(args.results_dir, args.dataset, args.output_format, str(out_dir))
    
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
