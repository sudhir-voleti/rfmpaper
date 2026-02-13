#!/usr/bin/env python3
"""
smc_unified_ppc.py
==================
Posterior Predictive Checks for HMM-Tweedie-SMC.
Uses existing CSV files and imports functions from smc_unified.py via GitHub.

Usage:
    python exec_gitcode.py \
        https://github.com/sudhir-voleti/rfmpaper/blob/main/smc_unified_ppc.py \
        /path/to/results_dir \
        --dataset uci \
        --train_weeks 40 \
        --test_weeks 13
"""

import argparse
import pickle
import sys
import urllib.request
import tempfile
import os
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pymc as pm
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Apple Silicon optimization
import pytensor
pytensor.config.floatX = 'float32'
pytensor.config.optimizer = 'fast_run'


# =============================================================================
# IMPORT FUNCTIONS FROM GITHUB
# =============================================================================

def import_from_github(url: str):
    """
    Dynamically import functions from smc_unified.py on GitHub.
    """
    # Convert to raw URL if needed
    if 'github.com' in url and 'raw.githubusercontent.com' not in url:
        url = url.replace('github.com', 'raw.githubusercontent.com')
        url = url.replace('/blob/', '/')
    
    # Download to temp file
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as response:
        code = response.read().decode('utf-8')
    
    # Write temp module
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    # Import
    import importlib.util
    spec = importlib.util.spec_from_file_location("smc_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Cleanup
    os.unlink(temp_path)
    
    return module


# =============================================================================
# DATA LOADING (from existing CSVs)
# =============================================================================

def load_panel_data_from_csv(data_path: Path, n_cust: int = None, 
                             max_week: int = None, seed: int = 42):
    """
    Load pre-built panel data from CSV.
    Reuses logic from smc_unified.py but with CSV input.
    """
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter by max_week if specified (train/test split)
    if max_week is not None:
        df = df[df['Week'] <= max_week].copy()
    
    # Customer sampling
    if n_cust is not None:
        np.random.seed(seed)
        all_cust = df['CustomerID'].unique()
        selected = np.random.choice(all_cust, size=min(n_cust, len(all_cust)), 
                                   replace=False)
        df = df[df['CustomerID'].isin(selected)].copy()
    
    # Build panel (simplified - assumes weekly aggregation exists)
    customers = df['CustomerID'].unique()
    N = len(customers)
    weeks = sorted(df['Week'].unique())
    T = len(weeks)
    
    # Initialize arrays
    y = np.zeros((N, T), dtype=np.float32)
    R = np.zeros((N, T), dtype=np.float32)
    F = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)
    
    # Fill arrays (simplified - assumes RFM already computed)
    cust_map = {c: i for i, c in enumerate(customers)}
    week_map = {w: t for t, w in enumerate(weeks)}
    
    for _, row in df.iterrows():
        i = cust_map[row['CustomerID']]
        t = week_map[row['Week']]
        y[i, t] = row.get('Spend', 0)
        R[i, t] = row.get('Recency', 0)
        F[i, t] = row.get('Frequency', 0)
        M[i, t] = row.get('Monetary', 0)
    
    mask = (y > 0) | (R > 0)  # Simple mask
    
    return {
        'y': y,
        'R': R,
        'F': F,
        'M': M,
        'mask': mask,
        'N': N,
        'T': T,
        'customers': customers
    }


# =============================================================================
# POSTERIOR PREDICTIVE SAMPLING
# =============================================================================

def sample_posterior_predictive_hmm(idata: object, data: Dict, 
                                    n_samples: int = 500) -> Dict:
    """
    Generate posterior predictive samples for HMM-Tweedie.
    
    For each posterior sample:
    1. Sample state sequence z_{1:T} given parameters
    2. Sample emissions y_t ~ ZIG(mu_t, phi_{z_t}, p_{z_t})
    """
    post = idata.posterior
    N, T = data['N'], data['T']
    K = post['Gamma'].shape[2]
    
    # Extract posterior samples
    n_chains, n_draws = post['beta0'].shape[:2]
    idx = np.random.choice(n_chains * n_draws, size=n_samples, replace=False)
    
    predictions = []
    
    for sample_idx in idx:
        chain_idx = sample_idx // n_draws
        draw_idx = sample_idx % n_draws
        
        # Extract parameters for this sample
        beta0 = post['beta0'].values[chain_idx, draw_idx, :]  # (K,)
        phi = post['phi'].values[chain_idx, draw_idx, :]  # (K,)
        p = post['p'].values[chain_idx, draw_idx, :] if 'p' in post else np.full(K, 1.5)
        Gamma = post['Gamma'].values[chain_idx, draw_idx, :, :]  # (K, K)
        
        # Forward filter to get state probabilities
        # Then sample states
        # Then sample emissions
        
        # Simplified: just sample from marginal state distribution
        pi0 = post['pi0'].values[chain_idx, draw_idx, :] if 'pi0' in post else np.ones(K)/K
        
        y_pred = np.zeros((N, T))
        
        for i in range(N):
            # Sample state sequence (simplified - independent draws)
            z = np.random.choice(K, size=T, p=pi0)
            
            for t in range(T):
                k = z[t]
                mu_it = np.exp(beta0[k])  # Simplified - no covariates
                
                # ZIG sample
                exponent = 2.0 - p[k]
                psi = np.exp(-(mu_it ** exponent) / (phi[k] * exponent))
                
                if np.random.rand() < psi:
                    y_pred[i, t] = 0
                else:
                    # Gamma sample
                    alpha = mu_it / phi[k]
                    y_pred[i, t] = np.random.gamma(alpha, phi[k])
        
        predictions.append(y_pred)
    
    return {
        'predictions': np.array(predictions),  # (n_samples, N, T)
        'mean': np.mean(predictions, axis=0),
        'std': np.std(predictions, axis=0)
    }


def compute_predictive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               mask: np.ndarray) -> Dict:
    """Compute MAE, RMSE, R²."""
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    
    mae = np.mean(np.abs(y_true_m - y_pred_m))
    rmse = np.sqrt(np.mean((y_true_m - y_pred_m) ** 2))
    
    ss_res = np.sum((y_true_m - y_pred_m) ** 2)
    ss_tot = np.sum((y_true_m - np.mean(y_true_m)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'n': mask.sum()}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Posterior Predictive Checks for HMM-Tweedie-SMC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('results_dir', type=str,
                       help='Directory containing fitted .pkl files')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['uci', 'cdnow'])
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing CSV files')
    parser.add_argument('--train_weeks', type=int, default=40,
                       help='Number of weeks for training')
    parser.add_argument('--test_weeks', type=int, default=13,
                       help='Number of weeks for testing')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of posterior predictive samples')
    parser.add_argument('--out_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    data_path = Path(args.data_dir)
    out_path = Path(args.out_dir) if args.out_dir else results_path
    
    # Find fitted model
    pattern = f"smc_{args.dataset}_K*_GAM_*_N*_D*.pkl"
    pkls = list(results_path.glob(pattern))
    if not pkls:
        raise FileNotFoundError(f"No models found in {results_path}")
    
    pkl_path = max(pkls, key=lambda p: p.stat().st_mtime)
    print(f"Loading fitted model: {pkl_path.name}")
    
    # Load fitted model
    with open(pkl_path, 'rb') as f:
        saved = pickle.load(f)
    
    idata = saved['idata']
    metadata = saved['res']
    
    # Load training data
    csv_path = data_path / f"{args.dataset}_full.csv"
    print(f"Loading data from: {csv_path}")
    
    data_train = load_panel_data_from_csv(csv_path, max_week=args.train_weeks)
    data_test = load_panel_data_from_csv(csv_path, 
                                         max_week=args.train_weeks + args.test_weeks)
    
    print(f"Train: N={data_train['N']}, T={data_train['T']}")
    print(f"Test: N={data_test['N']}, T={data_test['T']}")
    
    # Generate PPC samples
    print(f"Generating {args.n_samples} posterior predictive samples...")
    ppc = sample_posterior_predictive_hmm(idata, data_test, args.n_samples)
    
    # Compute metrics
    print("Computing predictive metrics...")
    metrics = compute_predictive_metrics(
        data_test['y'], 
        ppc['mean'], 
        data_test['mask']
    )
    
    print(f"\nPredictive Metrics (Test Set):")
    print(f"  MAE:  {metrics['mae']:.3f}")
    print(f"  RMSE: {metrics['rmse']:.3f}")
    print(f"  R²:   {metrics['r2']:.3f}")
    print(f"  N:    {metrics['n']}")
    
    # Save results
    results = {
        'metadata': metadata,
        'metrics': metrics,
        'ppc_mean': ppc['mean'],
        'ppc_std': ppc['std']
    }
    
    out_file = out_path / f"ppc_{args.dataset}_T{args.train_weeks}.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nSaved PPC results to: {out_file}")


if __name__ == "__main__":
    main()
