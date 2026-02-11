#!/usr/bin/env python3
"""
smc_unified_postproc.py
=======================
Post-processing utilities for RFM-HMM-Tweedie SMC results.
Extracts CLV, Viterbi, ROI, and diagnostic tables from saved .pkl files.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_idata(pkl_path):
    """Load InferenceData from pickle."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['idata'], data['res']


def summarize(arr, ci=0.95):
    """Mean, SD, and credible interval."""
    flat = arr.flatten()
    alpha = (1 - ci) / 2
    return {
        'mean': flat.mean(),
        'sd': flat.std(),
        'lo': np.quantile(flat, alpha),
        'hi': np.quantile(flat, 1 - alpha)
    }


def compute_pi_inf(Gamma):
    """Stationary distribution from transition matrix."""
    K = Gamma.shape[-1]
    A = np.eye(K) - Gamma.T + 1.0/K
    b = np.ones(K) / K
    pi = np.linalg.solve(A, b)
    return pi / pi.sum()


def simulate_clv(Gamma, beta0, phi, psi, start_state, 
                 horizon=52, discount=0.0019, n_sims=1000):
    """
    Monte Carlo CLV simulation from given starting state.
    """
    K = Gamma.shape[0]
    clvs = []
    
    for _ in range(n_sims):
        state = start_state
        clv = 0.0
        for t in range(horizon):
            # Spend
            if np.random.rand() < psi[state]:
                spend = 0.0
            else:
                mu = np.exp(beta0[state])
                alpha = mu / phi[state]
                spend = np.random.gamma(alpha, phi[state])  # scale parameterization
            
            clv += spend / ((1 + discount) ** t)
            
            # Transition
            state = np.random.choice(K, p=Gamma[state, :])
        
        clvs.append(clv)
    
    return np.array(clvs)


def reactivation_counterfactual(Gamma, beta0, phi, psi, 
                                from_state, to_state, switch_week=4,
                                horizon=52, n_sims=2000):
    """
    ROI of forcing transition from_state -> to_state at switch_week.
    """
    baseline = []
    treatment = []
    
    for _ in range(n_sims):
        # Baseline: stay in from_state
        state = from_state
        clv_base = 0.0
        for t in range(horizon):
            if np.random.rand() < psi[state]:
                spend = 0.0
            else:
                mu = np.exp(beta0[state])
                spend = np.random.gamma(mu/phi[state], phi[state])
            clv_base += spend / (1.0019 ** t)
            state = np.random.choice(K, p=Gamma[state, :])
        baseline.append(clv_base)
        
        # Treatment: switch at switch_week
        state = from_state
        clv_treat = 0.0
        for t in range(horizon):
            if t == switch_week:
                state = to_state
            if np.random.rand() < psi[state]:
                spend = 0.0
            else:
                mu = np.exp(beta0[state])
                spend = np.random.gamma(mu/phi[state], phi[state])
            clv_treat += spend / (1.0019 ** t)
            state = np.random.choice(K, p=Gamma[state, :])
        treatment.append(clv_treat)
    
    return np.array(baseline), np.array(treatment)


# =============================================================================
# EXTRACTION PIPELINE
# =============================================================================

def extract_state_table(idata):
    """Extract state-specific parameter table with CIs."""
    K = idata.posterior['Gamma'].shape[2]
    
    # Extract parameters
    psi = idata.posterior['psi'].values
    p = idata.posterior['p'].values
    beta0 = idata.posterior['beta0'].values
    phi = idata.posterior['phi'].values
    Gamma = idata.posterior['Gamma'].values
    
    # Compute derived quantities
    gamma_diag = np.einsum('...kk->...k', Gamma)
    dwell = 1.0 / (1.0 - gamma_diag + 1e-8)
    
    # Stationary distribution per draw
    pi_inf = np.zeros(psi.shape)
    for c in range(psi.shape[0]):
        for d in range(psi.shape[1]):
            pi_inf[c, d, :] = compute_pi_inf(Gamma[c, d, :, :])
    
    # Build table
    rows = []
    for k in range(K):
        row = {'state': k}
        
        for name, arr in [('psi', psi), ('p', p), ('gamma', gamma_diag),
                          ('dwell_weeks', dwell), ('pi_inf', pi_inf),
                          ('phi', phi)]:
            s = summarize(arr[:, :, k])
            row[f'{name}_mean'] = s['mean']
            row[f'{name}_sd'] = s['sd']
            row[f'{name}_lo'] = s['lo']
            row[f'{name}_hi'] = s['hi']
        
        row['spend_mean'] = np.exp(summarize(beta0[:, :, k])['mean'])
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_clv_table(idata, n_sims=2000):
    """Extract CLV by starting state."""
    K = idata.posterior['Gamma'].shape[2]
    
    # Use posterior mean for speed (or loop over draws for full uncertainty)
    Gamma = idata.posterior['Gamma'].mean(axis=(0,1)).values
    beta0 = idata.posterior['beta0'].mean(axis=(0,1)).values
    phi = idata.posterior['phi'].mean(axis=(0,1)).values
    psi = idata.posterior['psi'].mean(axis=(0,1)).values
    
    rows = []
    for k in range(K):
        clv_sims = simulate_clv(Gamma, beta0, phi, psi, k, n_sims=n_sims)
        row = {
            'start_state': k,
            'clv_mean': clv_sims.mean(),
            'clv_sd': clv_sims.std(),
            'clv_lo': np.quantile(clv_sims, 0.025),
            'clv_hi': np.quantile(clv_sims, 0.975)
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_roi_table(idata, n_sims=2000):
    """Extract reactivation ROI counterfactuals."""
    K = idata.posterior['Gamma'].shape[2]
    
    Gamma = idata.posterior['Gamma'].mean(axis=(0,1)).values
    beta0 = idata.posterior['beta0'].mean(axis=(0,1)).values
    phi = idata.posterior['phi'].mean(axis=(0,1)).values
    psi = idata.posterior['psi'].mean(axis=(0,1)).values
    
    # Reactivate from highest state to lowest
    from_state = K - 1  # Whale
    to_state = 0        # Active
    
    baseline, treatment = reactivation_counterfactual(
        Gamma, beta0, phi, psi, from_state, to_state, n_sims=n_sims
    )
    
    lift = treatment.mean() - baseline.mean()
    pct_lift = lift / baseline.mean() * 100
    
    return pd.DataFrame([{
        'from_state': from_state,
        'to_state': to_state,
        'baseline_clv': baseline.mean(),
        'treatment_clv': treatment.mean(),
        'lift_abs': lift,
        'lift_pct': pct_lift,
        'prob_positive': (treatment > baseline).mean()
    }])


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Post-process SMC results')
    parser.add_argument('pkl_path', help='Path to .pkl file')
    parser.add_argument('--output', choices=['state', 'clv', 'roi', 'all'], 
                       default='all', help='What to extract')
    parser.add_argument('--n_sims', type=int, default=2000, help='CLV simulations')
    
    args = parser.parse_args()
    
    print(f"Loading: {args.pkl_path}")
    idata, res = load_idata(args.pkl_path)
    
    print(f"Dataset: K={res['K']}, N={res['N']}, log_ev={res['log_evidence']:.2f}")
    
    if args.output in ['state', 'all']:
        print("\n" + "="*70)
        print("STATE PARAMETERS")
        print("="*70)
        df_state = extract_state_table(idata)
        print(df_state.round(3).to_string())
        df_state.to_csv(args.pkl_path.replace('.pkl', '_state.csv'), index=False)
    
    if args.output in ['clv', 'all']:
        print("\n" + "="*70)
        print("CLV BY STARTING STATE")
        print("="*70)
        df_clv = extract_clv_table(idata, n_sims=args.n_sims)
        print(df_clv.round(2).to_string())
        df_clv.to_csv(args.pkl_path.replace('.pkl', '_clv.csv'), index=False)
    
    if args.output in ['roi', 'all']:
        print("\n" + "="*70)
        print("REACTIVATION ROI")
        print("="*70)
        df_roi = extract_roi_table(idata, n_sims=args.n_sims)
        print(df_roi.round(2).to_string())
        df_roi.to_csv(args.pkl_path.replace('.pkl', '_roi.csv'), index=False)
    
    print(f"\nDone. CSVs saved alongside {args.pkl_path}")


if __name__ == '__main__':
    main()
