#!/usr/bin/env python3
"""
extract_logev_comparison.py
===========================
Extract log-evidence with multiple aggregation methods for comparison.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy.special import logsumexp


def extract_chain_values(idata):
    """Extract final log-evidence values from each chain."""
    chain_vals = []
    
    try:
        lm = idata.sample_stats.log_marginal_likelihood.values
        
        if isinstance(lm, np.ndarray) and lm.dtype == object:
            n_chains = lm.shape[1] if lm.ndim > 1 else 1
            
            for c in range(n_chains):
                if lm.ndim > 1:
                    chain_list = lm[-1, c] if lm.shape[0] > 0 else lm[0, c]
                else:
                    chain_list = lm[c] if lm.ndim == 1 else lm[0]
                
                if isinstance(chain_list, list):
                    valid = [float(x) for x in chain_list 
                            if isinstance(x, (int, float, np.floating)) and np.isfinite(x)]
                    if valid:
                        chain_vals.append(valid[-1])
                elif isinstance(chain_list, (int, float, np.floating)) and np.isfinite(chain_list):
                    chain_vals.append(float(chain_list))
        else:
            flat = np.array(lm).flatten()
            valid = flat[np.isfinite(flat)]
            chain_vals = valid.tolist() if len(valid) > 0 else []
            
    except Exception as e:
        print(f"  Warning: chain extraction failed: {e}")
    
    return chain_vals


def compute_log_ev_metrics(chain_vals):
    """
    Compute log-evidence using multiple methods.
    
    Returns dict with:
    - simple_mean: arithmetic mean of all chains
    - trimmed_mean_25: mean with worst 25% dropped
    - median: 50th percentile
    - max: best chain (most optimistic)
    - min: worst chain (most pessimistic)
    - n_chains: number of chains
    - chain_range: max - min (convergence diagnostic)
    """
    if not chain_vals or len(chain_vals) == 0:
        return {
            'simple_mean': np.nan,
            'trimmed_mean_25': np.nan,
            'median': np.nan,
            'max': np.nan,
            'min': np.nan,
            'n_chains': 0,
            'chain_range': np.nan
        }
    
    arr = np.array(chain_vals)
    n = len(arr)
    
    # Simple mean
    simple_mean = float(np.mean(arr))
    
    # Trimmed mean (drop lowest 25%)
    if n >= 4:
        sorted_arr = np.sort(arr)
        trim_n = max(1, n // 4)  # Drop worst 25%
        trimmed_mean_25 = float(np.mean(sorted_arr[trim_n:]))
    else:
        trimmed_mean_25 = simple_mean  # Not enough chains to trim
    
    # Median
    median = float(np.median(arr))
    
    # Max and min
    max_val = float(np.max(arr))
    min_val = float(np.min(arr))
    
    # Range for convergence check
    chain_range = max_val - min_val
    
    return {
        'simple_mean': simple_mean,
        'trimmed_mean_25': trimmed_mean_25,
        'median': median,
        'max': max_val,
        'min': min_val,
        'n_chains': n,
        'chain_range': chain_range
    }


def extract_model_metrics(pkl_path):
    """Extract full metrics including all log-ev methods."""
    pkl_path = Path(pkl_path)
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return None
    
    if not isinstance(data, dict):
        return None
    
    idata = data.get('idata')
    res = data.get('res', {})
    
    if idata is None:
        return None
    
    # Parse filename
    fname = pkl_path.name
    parts = fname.replace('.pkl', '').split('_')
    
    try:
        dataset = parts[1]
        K = int(parts[2][1:])
        model_type = parts[3]
        p_type = parts[4]
        N = int(parts[5][1:])
        D = int(parts[6][1:])
    except (IndexError, ValueError):
        return None
    
    # Determine p config
    p_fixed = None
    if p_type == 'varyingp':
        p_str = 'varying_K1'
    elif p_type == 'pNone':
        p_str = 'varying'
    elif p_type.startswith('p'):
        try:
            p_fixed = float(p_type[1:])
            p_str = f'fixed={p_fixed}'
        except:
            p_str = p_type
    else:
        p_str = p_type
    
    # Extract chain values and compute all log-ev metrics
    chain_vals = extract_chain_values(idata)
    log_ev_metrics = compute_log_ev_metrics(chain_vals)
    
    # WAIC
    waic = np.nan
    try:
        if hasattr(idata, 'posterior') and 'log_likelihood' in idata.posterior:
            loglik = idata.posterior['log_likelihood'].values
            n_chains, n_draws, n_obs = loglik.shape
            lppd = logsumexp(loglik, axis=(0,1)) - np.log(n_chains * n_draws)
            mean_loglik = logsumexp(loglik, axis=(0,1), keepdims=True) - np.log(n_chains * n_draws)
            mean_loglik = np.broadcast_to(mean_loglik, loglik.shape)
            p_waic = np.mean((loglik - mean_loglik)**2, axis=(0,1))
            waic_i = -2 * (lppd - p_waic)
            waic = -0.5 * np.sum(waic_i)
    except:
        pass
    
    # p mean
    p_mean = p_fixed if p_fixed else np.nan
    try:
        if hasattr(idata, 'posterior') and 'p' in idata.posterior:
            p_vals = idata.posterior['p'].values.flatten()
            p_mean = np.mean(p_vals)
    except:
        pass
    
    # Gamma
    gamma_diag = np.nan
    if K > 1:
        try:
            if hasattr(idata, 'posterior') and 'Gamma' in idata.posterior:
                gamma_post = idata.posterior['Gamma'].values
                gamma_mean = gamma_post.mean(axis=(0,1))
                gamma_diag = np.diag(gamma_mean).mean()
        except:
            pass
    
    # Phi sharing
    phi_shared = None
    if hasattr(idata, 'posterior') and 'phi' in idata.posterior:
        phi_shape = idata.posterior['phi'].shape
        phi_shared = len(phi_shape) == 2
    
    # Convergence flag
    flag = ""
    if log_ev_metrics['chain_range'] > 1000:
        flag = "CHAIN_DISAGREE"
    elif log_ev_metrics['chain_range'] > 100:
        flag = "MODERATE_DISAGREE"
    
    # Build result dict
    result = {
        'file': fname,
        'dataset': dataset,
        'K': K,
        'type': model_type,
        'p_type': p_str,
        'N': N,
        'D': D,
        'waic': waic,
        'p_mean': p_mean,
        'gamma_diag': gamma_diag,
        'phi_shared': phi_shared,
        'flag': flag,
    }
    
    # Add all log-ev metrics
    result.update(log_ev_metrics)
    
    return result


def scan_and_compare(dir_path, pattern="*.pkl"):
    """Scan directory and create comparison table."""
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob(pattern))
    
    print(f"Scanning {dir_path}...")
    print(f"Found {len(files)} files\n")
    
    results = []
    for f in files:
        r = extract_model_metrics(f)
        if r:
            results.append(r)
    
    df = pd.DataFrame(results)
    print(f"Successfully extracted: {len(df)} models\n")
    
    if df.empty:
        return df
    
    # Display comparison
    print("="*160)
    print("LOG-EVIDENCE COMPARISON TABLE")
    print("="*160)
    print(f"{'File':<45} {'K':>3} {'Type':<5} {'Simple':>12} {'Trim25%':>12} {'Median':>12} {'Max':>12} {'Min':>12} {'Range':>10} {'Flag':<15}")
    print("-"*160)
    
    for _, row in df.iterrows():
        print(f"{row['file']:<45} {row['K']:>3} {row['type']:<5} "
              f"{row['simple_mean']:>12.2f} {row['trimmed_mean_25']:>12.2f} "
              f"{row['median']:>12.2f} {row['max']:>12.2f} {row['min']:>12.2f} "
              f"{row['chain_range']:>10.2f} {row['flag']:<15}")
    
    print("="*160)
    
    # Summary by method
    print("\n" + "="*100)
    print("WINNER COUNT BY METHOD")
    print("="*100)
    
    # Group by dataset/K and find best for each method
    methods = ['simple_mean', 'trimmed_mean_25', 'median', 'max']
    
    for method in methods:
        print(f"\n{method.upper()}:")
        winners = {}
        
        for (ds, k), group in df.groupby(['dataset', 'K']):
            if len(group) == 0:
                continue
            best_idx = group[method].idxmax()
            best = group.loc[best_idx]
            winner = f"{best['type']}_phi{'shared' if best['phi_shared'] else 'state'}"
            
            if winner not in winners:
                winners[winner] = 0
            winners[winner] += 1
            
            print(f"  {ds}_K{k}: {winner} (log_ev={best[method]:.2f})")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract log-evidence with multiple methods')
    parser.add_argument('--dir', required=True, help='Directory with pickle files')
    parser.add_argument('--pattern', default='*.pkl', help='File pattern')
    parser.add_argument('--out', default='logev_comparison.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    df = scan_and_compare(args.dir, args.pattern)
    
    if not df.empty:
        df.to_csv(args.out, index=False)
        print(f"\nSaved to: {args.out}")
        
        # Also show summary stats
        print("\n" + "="*100)
        print("CHAIN CONVERGENCE SUMMARY")
        print("="*100)
        print(f"Total models: {len(df)}")
        print(f"Models with chain disagreement (>1000): {sum(df['flag'] == 'CHAIN_DISAGREE')}")
        print(f"Models with moderate disagreement (100-1000): {sum(df['flag'] == 'MODERATE_DISAGREE')}")
        print(f"Well-converged models (<100): {sum(df['flag'] == '')}")
        print(f"\nMean chain range: {df['chain_range'].mean():.2f}")
        print(f"Max chain range: {df['chain_range'].max():.2f}")
