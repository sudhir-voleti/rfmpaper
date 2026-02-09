#!/usr/bin/env python3
"""
ode_falsification_fixed.py - Extract ODE params from smc_unified.py output
"""
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_nested_idata(pkl_path):
    """Load idata from nested dict structure"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'idata' in data:
        idata = data['idata']
        res = data.get('res', {})
        print(f"Loaded: K={res.get('K', 'unknown')}, log_ev={res.get('log_evidence', 'N/A')}")
        return idata, res
    else:
        return data, {}  # Direct idata

def extract_transition_dynamics(idata):
    """Extract beta from Gamma transition matrix"""
    try:
        gamma_post = idata.posterior['Gamma']  # (chains, draws, K, K)
        gamma_mean = gamma_post.mean(dim=['chain', 'draw']).values
        
        K = gamma_mean.shape[0]
        
        # Diagonal = persistence (self-transition probability)
        persistence = np.diag(gamma_mean)
        
        # Off-diagonal = flux between states
        off_diag_sum = gamma_mean.sum(axis=1) - persistence
        
        # Beta from log-odds of persistence (higher persistence = higher momentum)
        beta = np.log(persistence / (1 - persistence + 1e-6))
        
        # Conductivity from off-diagonal rates
        conductivity = off_diag_sum / (K - 1)
        
        return pd.DataFrame({
            'state': range(K),
            'persistence': persistence,
            'beta_raw': beta,
            'conductivity': conductivity,
            'gamma_ii': persistence
        })
    except Exception as e:
        print(f"Error extracting Gamma: {e}")
        return None

def extract_recency_dissipation(idata):
    """Extract delta from w_R (Recency smooth weights)"""
    try:
        if 'w_R' not in idata.posterior:
            print("No w_R found (might be GLM, not GAM)")
            return None
            
        w_R_post = idata.posterior['w_R']
        # Shape: (chains, draws, K, n_splines) or (chains, draws, n_splines)
        
        w_R_mean = w_R_post.mean(dim=['chain', 'draw']).values
        
        # If state-specific, average absolute weight as dissipation proxy
        if w_R_mean.ndim == 2:
            # (K, n_splines)
            delta_by_state = np.mean(np.abs(w_R_mean), axis=1)
        else:
            # Shared across states
            delta_by_state = np.full(len(idata.posterior['beta0'].mean(dim=['chain', 'draw'])), 
                                    np.mean(np.abs(w_R_mean)))
        
        return pd.DataFrame({
            'state': range(len(delta_by_state)),
            'delta': delta_by_state,
            'w_R_mean': [float(np.mean(np.abs(w_R_mean[i]))) for i in range(len(delta_by_state))]
        })
    except Exception as e:
        print(f"Error extracting w_R: {e}")
        return None

def extract_power_params(idata):
    """Extract state-specific p values"""
    try:
        p_post = idata.posterior['p']
        p_mean = p_post.mean(dim=['chain', 'draw']).values
        p_std = p_post.std(dim=['chain', 'draw']).values
        
        # Check if constrained (ordered transform) or free
        # p_raw might exist if using ordered transform
        if 'p_raw' in idata.posterior:
            print("Note: Using ordered transformation for p")
        
        if p_mean.ndim == 0:
            return None, float(p_mean)
        
        df = pd.DataFrame({
            'state': range(len(p_mean)),
            'p_mean': p_mean,
            'p_std': p_std
        })
        return df, None
    except Exception as e:
        print(f"Error extracting p: {e}")
        return None, None

def compute_ode_kinetics(gamma_df, delta_df, p_df):
    """Merge and compute beta/delta ratios"""
    if gamma_df is None or delta_df is None or p_df is None:
        return None
    
    # Merge on state
    merged = gamma_df.merge(delta_df, on='state').merge(p_df, on='state')
    
    # Avoid division by zero
    merged['beta_over_delta'] = merged['beta_raw'] / (merged['delta'] + 1e-6)
    
    # Normalize for interpretability (z-score)
    merged['beta_norm'] = (merged['beta_raw'] - merged['beta_raw'].mean()) / merged['beta_raw'].std()
    merged['delta_norm'] = (merged['delta'] - merged['delta'].mean()) / merged['delta'].std()
    
    return merged

def correlation_test(df):
    """Test if p correlates with kinetic parameters"""
    from scipy.stats import pearsonr, spearmanr
    
    results = {}
    
    # Pearson correlation
    if len(df) > 2:
        r_pearson, p_pearson = pearsonr(df['beta_over_delta'], df['p_mean'])
        results['pearson_r'] = r_pearson
        results['pearson_p'] = p_pearson
        
        # Spearman (rank correlation, more robust)
        r_spear, p_spear = spearmanr(df['beta_over_delta'], df['p_mean'])
        results['spearman_r'] = r_spear
        results['spearman_p'] = p_spear
        
        # Linear fit
        slope, intercept = np.polyfit(df['beta_over_delta'], df['p_mean'], 1)
        results['slope'] = slope
        results['intercept'] = intercept
    else:
        # K=2: just report difference
        diff_p = df['p_mean'].iloc[1] - df['p_mean'].iloc[0]
        diff_beta = df['beta_over_delta'].iloc[1] - df['beta_over_delta'].iloc[0]
        results['delta_p'] = diff_p
        results['delta_beta_delta'] = diff_beta
        
    return results

def main():
    parser = argparse.ArgumentParser(description='ODE Falsification Test - Fixed')
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='unknown')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    print("="*70)
    print(f"ODE FALSIFICATION TEST: {args.dataset}")
    print("="*70)
    
    # Load
    idata, res = load_nested_idata(args.file)
    
    # Extract
    print("\n1. Transition dynamics (Gamma)...")
    gamma_df = extract_transition_dynamics(idata)
    
    print("2. Recency dissipation (w_R)...")
    delta_df = extract_recency_dissipation(idata)
    
    print("3. Power parameters (p)...")
    p_df, p_global = extract_power_params(idata)
    
    # Merge
    print("\n4. Computing ODE kinetics...")
    kinetics_df = compute_ode_kinetics(gamma_df, delta_df, p_df)
    
    if kinetics_df is not None:
        print(kinetics_df.to_string(index=False))
        
        # Test correlation
        print("\n5. Correlation test...")
        corr = correlation_test(kinetics_df)
        
        for k, v in corr.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
        
        # Interpretation
        print("\n" + "="*70)
        print("INTERPRETATION:")
        if 'pearson_r' in corr:
            if corr['pearson_r'] > 0.5 and corr['pearson_p'] < 0.1:
                print("✓ STRONG POSITIVE: High β/δ → High p (Boiling Pot confirmed)")
                print("  Hot states: persistence dominates dissipation")
            elif corr['pearson_r'] < -0.3:
                print("✗ INVERSE: Unexpected negative correlation")
            else:
                print("~ WEAK: No clear kinetic relationship")
        else:
            # K=2 case
            if kinetics_df['p_mean'].iloc[1] > kinetics_df['p_mean'].iloc[0]:
                print("✓ ORDERED: State 1 (hot) has higher p than State 0 (cold)")
            else:
                print("? UNORDERED: p values don't align with state labels")
        
        # Save
        if args.output:
            kinetics_df.to_csv(args.output, index=False)
            print(f"\n6. Saved to: {args.output}")
    else:
        print("✗ Failed to compute kinetics")
    
    print("="*70)

if __name__ == "__main__":
    main()
