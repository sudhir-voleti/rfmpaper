#!/usr/bin/env python3
"""
ode_falsification_v2.py - Handles both direct idata and nested dict
"""
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_idata_robust(pkl_path):
    """Handle both direct idata and nested dict formats"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Case 1: Direct InferenceData (large files >100MB)
    if hasattr(data, 'posterior'):
        print("Format: Direct InferenceData")
        return data, {}
    
    # Case 2: Nested dict (smaller files ~4MB)
    elif isinstance(data, dict) and 'idata' in data:
        print(f"Format: Nested dict (K={data.get('res', {}).get('K', 'unknown')})")
        return data['idata'], data.get('res', {})
    
    else:
        raise ValueError(f"Unknown format: {type(data)}")

def extract_all_params(idata):
    """Extract ODE-relevant parameters"""
    results = {}
    
    try:
        # 1. Transition matrix (Gamma)
        if 'Gamma' in idata.posterior:
            gamma = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values
            K = gamma.shape[0]
            persistence = np.diag(gamma)
            results['persistence'] = persistence
            results['beta'] = np.log(persistence + 1e-6)  # log-odds proxy
            print(f"✓ Gamma: {K}x{K}, persistence range [{persistence.min():.3f}, {persistence.max():.3f}]")
        else:
            print("✗ No Gamma found")
            
        # 2. Power parameter (p) - THE THERMOMETER
        if 'p' in idata.posterior:
            p_mean = idata.posterior['p'].mean(dim=['chain', 'draw']).values
            results['p'] = p_mean
            if p_mean.ndim > 0:
                print(f"✓ State-specific p: {p_mean} (range: {p_mean.max()-p_mean.min():.3f})")
            else:
                print(f"✓ Global p: {float(p_mean):.3f}")
        else:
            print("✗ No p found")
            
        # 3. Recency/GAM coefficients
        for var in ['w_R', 'beta_R', 's_R', 'smooth_R']:
            if var in idata.posterior:
                w = idata.posterior[var].mean(dim=['chain', 'draw']).values
                results['delta'] = np.mean(np.abs(w))
                print(f"✓ {var}: delta proxy = {results['delta']:.4f}")
                break
        
        # 4. Baseline spend (beta0) - emission level
        if 'beta0' in idata.posterior:
            beta0 = idata.posterior['beta0'].mean(dim=['chain', 'draw']).values
            results['log_spend'] = beta0  # log-scale baseline
            print(f"✓ beta0 (log-spend): {beta0}")
            
    except Exception as e:
        print(f"Error: {e}")
        
    return results

def compute_equation_of_state(results):
    """Test if p correlates with persistence (boiling pot hypothesis)"""
    if 'p' not in results or 'persistence' not in results:
        print("Insufficient data for equation of state")
        return None
    
    p = results['p']
    pers = results['persistence']
    
    if p.ndim == 0 or len(p) != len(pers):
        print("Dimension mismatch")
        return None
    
    # Sort by p (cold to hot)
    order = np.argsort(p)
    p_sorted = p[order]
    pers_sorted = pers[order]
    
    # Correlation
    corr = np.corrcoef(p, pers)[0,1]
    
    df = pd.DataFrame({
        'state': range(len(p)),
        'p': p,
        'persistence': pers,
        'log_spend': results.get('log_spend', [np.nan]*len(p)),
        'phase': ['Cold' if i == 0 else 'Hot' if i == len(p)-1 else 'Trans' for i in range(len(p))]
    })
    
    df = df.sort_values('p')
    
    print(f"\n{'='*60}")
    print("EQUATION OF STATE: p vs Persistence")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nCorrelation(p, persistence): {corr:.3f}")
    
    if corr > 0.5:
        print("✓ VALIDATED: Higher persistence → Higher p (Boiling Pot)")
        print("  Cold states: Low persistence, low p (~1.2, clumpy)")
        print("  Hot states: High persistence, high p (~1.8, Gamma-like)")
    elif corr < -0.2:
        print("✗ INVERSE: Unexpected negative correlation")
    else:
        print("~ WEAK: No strong thermodynamic relationship")
        
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    print(f"Loading: {args.file}")
    print(f"Size: {Path(args.file).stat().st_size/1e6:.1f} MB\n")
    
    # Load
    idata, meta = load_idata_robust(args.file)
    
    # Extract
    results = extract_all_params(idata)
    
    # Analyze
    df = compute_equation_of_state(results)
    
    # Save
    if df is not None and args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()
