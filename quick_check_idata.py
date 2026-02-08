#!/usr/bin/env python3
"""
quick_check_idata.py
====================
Quick diagnostic for SMC results.
Usage: python quick_check_idata.py path/to/smc_*.pkl
"""
import sys
import pickle
import numpy as np
import pandas as pd

pkl_path = sys.argv[1]

with open(pkl_path, 'rb') as f:
    bundle = pickle.load(f)

idata = bundle['idata']
res = bundle['res']

print(f"\n{'='*60}")
print(f"FILE: {pkl_path}")
print(f"{'='*60}")
print(f"K={res['K']}, N={res['N']}, log_ev={res['log_evidence']:.2f}")

# Check if state-specific p exists
if 'p' in idata.posterior:
    p_post = idata.posterior['p'].values  # (chains, draws, K)
    K = p_post.shape[2]
    
    print(f"\nSTATE-SPECIFIC p (K={K}):")
    print(f"{'State':<8} {'Mean':<8} {'Std':<8} {'2.5%':<8} {'97.5%':<8}")
    print("-" * 50)
    
    for k in range(K):
        p_k = p_post[:, :, k].flatten()
        print(f"{k:<8} {np.mean(p_k):.3f}    {np.std(p_k):.3f}    "
              f"{np.percentile(p_k, 2.5):.3f}    {np.percentile(p_k, 97.5):.3f}")
    
    # Check ordering
    ordered = np.all(p_post[:, :, :-1] < p_post[:, :, 1:])
    print(f"\nOrdering constraint (p[0]<p[1]<...): {100*np.mean(ordered):.1f}%")
    
else:
    print("\nNo state-specific p found (single p or fixed p)")

# Other key params
print(f"\nOther parameters: {list(idata.posterior.data_vars)}")
print(f"{'='*60}")
