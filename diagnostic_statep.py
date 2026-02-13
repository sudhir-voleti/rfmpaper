#!/usr/bin/env python3
"""
diagnostic_statep.py
====================
Compare fixed-p vs state-p on predictive metrics, not just log-ev.
"""

import pickle
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path


def load_results(results_dir, dataset, K, p_type):
    """Load pickle for specific model configuration."""
    pattern = f"smc_{dataset}_K{K}_GAM_{p_type}_N500_D*.pkl"
    files = list(Path(results_dir).glob(pattern))
    if not files:
        return None, None
    # Prefer D=1000 if available
    files = sorted(files, key=lambda x: int(x.stem.split('D')[1].split('_')[0]), reverse=True)
    with open(files[0], 'rb') as f:
        data = pickle.load(f)
    return data['idata'], data['res']


def check_p_separation(idata, K):
    """Check if state-p values are actually separated or collapsed."""
    if 'p' not in idata.posterior:
        return None
    
    p_post = idata.posterior['p'].values  # shape: (chains, draws, K) or (chains, draws)
    if p_post.ndim == 2:
        # K=1 case
        return {'means': [p_post.mean()], 'stds': [p_post.std()], 'separation': None}
    
    # K>1: check if states have distinct p values
    p_means = p_post.mean(axis=(0,1))  # mean per state
    p_stds = p_post.std(axis=(0,1))
    
    # Check overlap: are posterior distributions separated?
    # Compute pairwise separation in standard deviations
    separations = []
    for i in range(K):
        for j in range(i+1, K):
            sep = abs(p_means[i] - p_means[j]) / np.sqrt(p_stds[i]**2 + p_stds[j]**2)
            separations.append((i, j, sep))
    
    return {
        'means': p_means.tolist(),
        'stds': p_stds.tolist(),
        'separations': separations,
        'range': float(p_means.max() - p_means.min())
    }


def posterior_predictive_stats(idata, y_obs=None):
    """Extract key PPC statistics."""
    stats = {}
    
    # Try to get posterior predictive
    if 'posterior_predictive' in idata and 'y' in idata.posterior_predictive:
        y_pp = idata.posterior_predictive['y'].values
    else:
        # Need to compute from model — skip for now
        y_pp = None
    
    if y_pp is not None and y_obs is not None:
        # Calibration: proportion of zeros
        stats['zero_rate_obs'] = (y_obs == 0).mean()
        stats['zero_rate_pred'] = (y_pp == 0).mean(axis=(0,1)).mean()
        stats['zero_rate_bias'] = stats['zero_rate_pred'] - stats['zero_rate_obs']
        
        # Mean calibration
        stats['mean_obs'] = y_obs.mean()
        stats['mean_pred'] = y_pp.mean(axis=(0,1)).mean()
        
        # Variance calibration
        stats['var_obs'] = y_obs.var()
        stats['var_pred'] = y_pp.var(axis=(0,1)).mean()
    
    # From observed log-likelihood (in-sample)
    if 'log_likelihood' in idata.observed_data:
        loglik = idata.observed_data['log_likelihood'].values
        stats['mean_loglik'] = loglik.mean()
        stats['total_loglik'] = loglik.sum()
    
    return stats


def compare_models(results_dir, dataset):
    """Compare fixed-p vs state-p for K=3."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: {dataset.upper()}")
    print(f"{'='*70}")
    
    # Load models
    idata_fixed, res_fixed = load_results(results_dir, dataset, 3, 'p1.5')
    idata_state, res_state = load_results(results_dir, dataset, 3, 'statep')
    
    if idata_fixed is None or idata_state is None:
        print("Missing models!")
        return
    
    # 1. Log-evidence comparison
    print(f"\n--- Log-Evidence ---")
    print(f"Fixed p=1.5:  {res_fixed.get('log_evidence', np.nan):.2f}")
    print(f"State-p:      {res_state.get('log_evidence', np.nan):.2f}")
    diff = res_state.get('log_evidence', np.nan) - res_fixed.get('log_evidence', np.nan)
    print(f"Difference:   {diff:.2f} (negative = fixed wins)")
    
    # 2. State-p separation check
    print(f"\n--- State-p Identification ---")
    p_info = check_p_separation(idata_state, 3)
    if p_info:
        print(f"State p means: {['%.3f' % p for p in p_info['means']]}")
        print(f"State p stds:  {['%.3f' % s for s in p_info['stds']]}")
        print(f"Range: {p_info['range']:.3f}")
        print(f"\nPairwise separations (in SD units):")
        for i, j, sep in p_info['separations']:
            status = "✓ separated" if sep > 2 else "✗ overlapping"
            print(f"  State {i} vs {j}: {sep:.2f} SD {status}")
    
    # 3. In-sample likelihood (alternative to log-ev)
    print(f"\n--- In-Sample Fit (Point Estimate) ---")
    stats_fixed = posterior_predictive_stats(idata_fixed)
    stats_state = posterior_predictive_stats(idata_state)
    
    print(f"Fixed p:  total loglik = {stats_fixed.get('total_loglik', np.nan):.2f}")
    print(f"State-p:  total loglik = {stats_state.get('total_loglik', np.nan):.2f}")
    
    # 4. Check if state-p is just fitting noise
    print(f"\n--- Effective Parameters (approximate) ---")
    # Rough proxy: variance of posterior means
    if 'phi' in idata_fixed.posterior and 'phi' in idata_state.posterior:
        phi_fixed = idata_fixed.posterior['phi'].values.mean(axis=(0,1))
        phi_state = idata_state.posterior['phi'].values.mean(axis=(0,1))
        print(f"Fixed p phi:  {phi_fixed}")
        print(f"State-p phi:  {phi_state}")
    
    # 5. State occupancy (for HMM)
    if 'Gamma' in idata_state.posterior:
        gamma_post = idata_state.posterior['Gamma'].values.mean(axis=(0,1))
        print(f"\n--- Transition Dynamics ---")
        print(f"Diagonal (persistence): {np.diag(gamma_post).round(3)}")
        print(f"State usage (stationary): {get_stationary(gamma_post).round(3)}")


def get_stationary(gamma):
    """Compute stationary distribution of transition matrix."""
    eigvals, eigvecs = np.linalg.eig(gamma.T)
    stationary = eigvecs[:, np.argmax(np.isclose(eigvals, 1))]
    return stationary / stationary.sum()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_statep.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    for dataset in ['cdnow', 'uci']:
        try:
            compare_models(results_dir, dataset)
        except Exception as e:
            print(f"Error on {dataset}: {e}")
            import traceback
            traceback.print_exc()
