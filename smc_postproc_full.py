#!/usr/bin/env python3
"""
smc_postproc_full.py
====================
Comprehensive post-processing for HMM-Tweedie-SMC full runs (N=1000).
Extracts all quantities needed for Marketing Science manuscript Sections 4-6.

Usage:
    # Via exec_gitcode.py (GitHub execution)
    python exec_gitcode.py \
        https://github.com/sudhir-voleti/rfmpaper/blob/main/smc_postproc_full.py \
        /path/to/results_dir \
        --dataset uci \
        --step full_extraction
    
    # Local execution
    python smc_postproc_full.py \
        /path/to/results_dir \
        --dataset uci \
        --step full_extraction

Steps:
    - full_extraction: All tables and diagnostics (default)
    - transitions: Gamma matrix, persistence, dwell times
    - emissions: State parameters (beta0, phi, p, psi, spend)
    - clv: CLV proxy and churn risk by state
    - viterbi: Decode state sequences (if available)
    - comparison: Side-by-side UCI vs CDNOW
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

STATE_LABELS = {0: 'Cold', 1: 'Warm', 2: 'Hot', 3: 'Whale', 4: 'VIP'}

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_results(pkl_path: str) -> Tuple[object, Dict]:
    """Load idata and metadata from pickle."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['idata'], data['res']


def summarize_posterior(arr: np.ndarray, ci: float = 0.94) -> Dict:
    """Compute mean, sd, and credible interval."""
    flat = arr.flatten()
    alpha = (1 - ci) / 2
    return {
        'mean': float(np.mean(flat)),
        'sd': float(np.std(flat)),
        'lo': float(np.quantile(flat, alpha)),
        'hi': float(np.quantile(flat, 1 - alpha)),
        'median': float(np.median(flat))
    }


def compute_psi_zig(mu: float, phi: float, p: float) -> float:
    """Compute structural zero probability from ZIG formula."""
    exponent = 2.0 - p
    return float(np.exp(-(mu ** exponent) / (phi * exponent)))


def extract_state_emissions(idata: object, K: int) -> pd.DataFrame:
    """
    Extract state-specific emission parameters.
    Returns DataFrame with beta0, phi, p, psi, spend for each state.
    """
    rows = []
    
    for k in range(K):
        row = {
            'state': k,
            'label': STATE_LABELS.get(k, f'S{k}')
        }
        
        # beta0 (baseline log-spend)
        beta0 = idata.posterior['beta0'].values[:, :, k]
        s = summarize_posterior(beta0)
        row.update({
            'beta0_mean': s['mean'],
            'beta0_sd': s['sd'],
            'beta0_lo': s['lo'],
            'beta0_hi': s['hi']
        })
        
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
        row.update({
            'phi_mean': s['mean'],
            'phi_sd': s['sd'],
            'phi_lo': s['lo'],
            'phi_hi': s['hi']
        })
        
        # p (power parameter) - state-specific
        if 'p' in idata.posterior:
            p = idata.posterior['p'].values[:, :, k]
            s = summarize_posterior(p)
            row.update({
                'p_mean': s['mean'],
                'p_sd': s['sd'],
                'p_lo': s['lo'],
                'p_hi': s['hi']
            })
            p_mean = s['mean']
        else:
            p_mean = 1.5  # default
            row['p_mean'] = p_mean
        
        # psi (structural zero prob) - computed from ZIG
        psi = compute_psi_zig(s_spend['mean'], s['mean'], p_mean)
        row['psi_mean'] = psi
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_transitions(idata: object, K: int) -> Dict:
    """
    Extract transition matrix dynamics.
    Returns Gamma mean, persistence, dwell times, resurrection probs.
    """
    Gamma = idata.posterior['Gamma'].values  # (chains, draws, K, K)
    Gamma_mean = Gamma.mean(axis=(0, 1))
    Gamma_std = Gamma.std(axis=(0, 1))
    
    # Persistence (diagonal)
    gamma_diag = np.einsum('...kk->...k', Gamma)
    
    persistence = {}
    dwell_times = {}
    
    for k in range(K):
        s = summarize_posterior(gamma_diag[:, :, k])
        persistence[k] = s
        dwell_times[k] = 1 / (1 - s['mean']) if s['mean'] < 0.999 else 999
    
    # Resurrection and abandonment
    results = {
        'Gamma_mean': Gamma_mean,
        'Gamma_std': Gamma_std,
        'persistence': persistence,
        'dwell_times': dwell_times
    }
    
    if K >= 3:
        # Cold -> Hot (resurrection)
        cold_to_hot = Gamma[:, :, 0, 2].mean() if K > 2 else 0
        # Hot -> Cold (abandonment)
        hot_to_cold = Gamma[:, :, 2, 0].mean() if K > 2 else 0
        
        results['resurrection'] = float(cold_to_hot)
        results['abandonment'] = float(hot_to_cold)
    
    return results


def extract_clv_quantities(idata: object, K: int) -> pd.DataFrame:
    """Extract CLV proxy and churn risk by state."""
    rows = []
    
    has_clv = 'clv_proxy' in idata.posterior
    has_churn = 'churn_risk' in idata.posterior
    
    for k in range(K):
        row = {'state': k, 'label': STATE_LABELS.get(k, f'S{k}')}
        
        if has_clv:
            clv = idata.posterior['clv_proxy'].values[:, :, k]
            s = summarize_posterior(clv)
            row.update({
                'clv_mean': s['mean'],
                'clv_sd': s['sd'],
                'clv_lo': s['lo'],
                'clv_hi': s['hi']
            })
        
        if has_churn:
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
    # Solve pi = pi * Gamma
    eigvals, eigvecs = np.linalg.eig(Gamma_mean.T)
    stationary = eigvecs[:, np.isclose(eigvals, 1)].real
    stationary = stationary / stationary.sum()
    return stationary.flatten()


def format_transition_table(transitions: Dict, K: int) -> pd.DataFrame:
    """Format transition matrix for LaTeX table."""
    Gamma_mean = transitions['Gamma_mean']
    
    df = pd.DataFrame(
        Gamma_mean,
        index=[f'From {STATE_LABELS.get(i, f"S{i}")}' for i in range(K)],
        columns=[f'To {STATE_LABELS.get(j, f"S{j}")}' for j in range(K)]
    )
    
    return df.round(3)


def generate_summary_report(dataset: str, emissions: pd.DataFrame, 
                           transitions: Dict, clv: pd.DataFrame,
                           metadata: Dict) -> str:
    """Generate text summary for manuscript."""
    
    K = len(emissions)
    
    report = f"""
{'='*70}
SUMMARY REPORT: {dataset.upper()} K={K} N={metadata.get('N', 'unknown')}
{'='*70}

STATE EMISSIONS:
{emissions[['state', 'label', 'spend_mean', 'p_mean', 'psi_mean']].to_string(index=False)}

TRANSITION DYNAMICS:
"""
    
    for k in range(K):
        pers = transitions['persistence'][k]
        dwell = transitions['dwell_times'][k]
        report += f"  {STATE_LABELS.get(k, f'S{k}')}: "
        report += f"γ={pers['mean']:.3f} [{pers['lo']:.3f}, {pers['hi']:.3f}], "
        report += f"Dwell={dwell:.1f}w\n"
    
    if 'resurrection' in transitions:
        report += f"\nResurrection (Cold→Hot): {transitions['resurrection']:.3f}\n"
        report += f"Abandonment (Hot→Cold): {transitions['abandonment']:.3f}\n"
    
    if not clv.empty and 'clv_mean' in clv.columns:
        report += f"\nCLV BY STATE:\n"
        for _, row in clv.iterrows():
            report += f"  {row['label']}: ${row['clv_mean']:.2f} "
            report += f"[${row['clv_lo']:.2f}, ${row['clv_hi']:.2f}]\n"
    
    report += f"\n{'='*70}\n"
    
    return report


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_single_dataset(results_dir: str, dataset: str, 
                          step: str = 'full_extraction') -> Dict:
    """
    Process single dataset results.
    
    Args:
        results_dir: Path to directory containing .pkl files
        dataset: 'uci' or 'cdnow'
        step: Which extraction step to run
    """
    results_path = Path(results_dir)
    
    # Find the correct .pkl file
    pattern = f"smc_{dataset}_K*_GAM_*_N*_D*_C*.pkl"
    pkls = list(results_path.glob(pattern))
    
    if not pkls:
        # Try without dataset prefix (older naming)
        pattern = f"smc_K*_GAM_*_N*_D*_C*.pkl"
        pkls = list(results_path.glob(pattern))
    
    if not pkls:
        raise FileNotFoundError(f"No .pkl files found in {results_dir} matching {pattern}")
    
    # Use most recent if multiple
    pkl_path = max(pkls, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {pkl_path.name}")
    
    # Load data
    idata, metadata = load_results(str(pkl_path))
    K = idata.posterior['Gamma'].shape[2]
    
    print(f"Dataset: {metadata.get('dataset', dataset)}")
    print(f"N: {metadata.get('N', 'unknown')}, K: {K}")
    print(f"Log-evidence: {metadata.get('log_evidence', 'unknown')}")
    
    output = {
        'metadata': metadata,
        'dataset': dataset,
        'K': K
    }
    
    # Run requested step
    if step in ['full_extraction', 'emissions']:
        print("\nExtracting state emissions...")
        emissions = extract_state_emissions(idata, K)
        output['emissions'] = emissions
        print(emissions[['state', 'label', 'spend_mean', 'p_mean']].to_string())
    
    if step in ['full_extraction', 'transitions']:
        print("\nExtracting transition dynamics...")
        transitions = extract_transitions(idata, K)
        output['transitions'] = transitions
        
        # Stationary distribution
        pi_inf = compute_stationary_distribution(transitions['Gamma_mean'])
        output['stationary'] = pi_inf
        print(f"Stationary distribution: {pi_inf.round(3)}")
        
        # Transition table
        gamma_df = format_transition_table(transitions, K)
        print(f"\nTransition matrix:\n{gamma_df}")
    
    if step in ['full_extraction', 'clv']:
        print("\nExtracting CLV quantities...")
        clv = extract_clv_quantities(idata, K)
        output['clv'] = clv
        if not clv.empty:
            print(clv[['state', 'label', 'clv_mean', 'churn_mean']].to_string())
    
    # Generate report
    if step == 'full_extraction':
        report = generate_summary_report(
            dataset, 
            output.get('emissions', pd.DataFrame()),
            output.get('transitions', {}),
            output.get('clv', pd.DataFrame()),
            metadata
        )
        output['report'] = report
        print(report)
    
    # Save extraction
    output_path = pkl_path.with_suffix('_full_extract.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved extraction to: {output_path}")
    
    return output


def compare_datasets(uci_results: Dict, cdnow_results: Dict) -> pd.DataFrame:
    """Generate side-by-side comparison table."""
    
    comparison = pd.DataFrame({
        'Metric': [],
        'UCI': [],
        'CDNOW': []
    })
    
    # Log-evidence
    comparison = pd.concat([comparison, pd.DataFrame({
        'Metric': ['Log-Evidence', 'N', 'K', 'Zeros'],
        'UCI': [uci_results['metadata'].get('log_evidence', 'N/A'),
                uci_results['metadata'].get('N', 'N/A'),
                uci_results['K'],
                f"{uci_results['metadata'].get('zeros', 0):.1%}"],
        'CDNOW': [cdnow_results['metadata'].get('log_evidence', 'N/A'),
                  cdnow_results['metadata'].get('N', 'N/A'),
                  cdnow_results['K'],
                  f"{cdnow_results['metadata'].get('zeros', 0):.1%}"]
    })], ignore_index=True)
    
    # State comparisons
    if 'emissions' in uci_results and 'emissions' in cdnow_results:
        for k in range(min(uci_results['K'], cdnow_results['K'])):
            uci_spend = uci_results['emissions'].loc[k, 'spend_mean']
            cdnow_spend = cdnow_results['emissions'].loc[k, 'spend_mean']
            uci_p = uci_results['emissions'].loc[k, 'p_mean']
            cdnow_p = cdnow_results['emissions'].loc[k, 'p_mean']
            
            comparison = pd.concat([comparison, pd.DataFrame({
                'Metric': [f'State {k} Spend', f'State {k} p'],
                'UCI': [f"${uci_spend:.2f}", f"{uci_p:.3f}"],
                'CDNOW': [f"${cdnow_spend:.2f}", f"{cdnow_p:.3f}"]
            })], ignore_index=True)
    
    # Persistence
    if 'transitions' in uci_results and 'transitions' in cdnow_results:
        for k in range(min(uci_results['K'], cdnow_results['K'])):
            uci_pers = uci_results['transitions']['persistence'][k]['mean']
            cdnow_pers = cdnow_results['transitions']['persistence'][k]['mean']
            
            comparison = pd.concat([comparison, pd.DataFrame({
                'Metric': [f'State {k} Persistence'],
                'UCI': [f"{uci_pers:.3f}"],
                'CDNOW': [f"{cdnow_pers:.3f}"]
            })], ignore_index=True)
    
    return comparison


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Full post-processing for HMM-Tweedie-SMC results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('results_dir', type=str,
                       help='Directory containing .pkl result files')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['uci', 'cdnow', 'both'],
                       help='Which dataset to process')
    parser.add_argument('--step', type=str, default='full_extraction',
                       choices=['full_extraction', 'emissions', 'transitions', 
                               'clv', 'viterbi', 'comparison'],
                       help='Which extraction step to run')
    parser.add_argument('--out_dir', type=str, default=None,
                       help='Output directory (default: same as results_dir)')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process single dataset
    if args.dataset in ['uci', 'cdnow']:
        results = process_single_dataset(
            args.results_dir, 
            args.dataset, 
            args.step
        )
        
        # Save CSV tables for easy import
        if 'emissions' in results:
            results['emissions'].to_csv(
                out_dir / f'{args.dataset}_emissions.csv', index=False
            )
        if 'transitions' in results:
            pd.DataFrame(results['transitions']['Gamma_mean']).to_csv(
                out_dir / f'{args.dataset}_Gamma.csv'
            )
    
    # Comparison mode
    elif args.dataset == 'both':
        print("Processing UCI...")
        uci_results = process_single_dataset(args.results_dir, 'uci', args.step)
        
        print("\nProcessing CDNOW...")
        cdnow_results = process_single_dataset(args.results_dir, 'cdnow', args.step)
        
        print("\nGenerating comparison...")
        comparison = compare_datasets(uci_results, cdnow_results)
        print(comparison.to_string())
        
        comparison.to_csv(out_dir / 'comparison_uci_cdnow.csv', index=False)


if __name__ == "__main__":
    main()
