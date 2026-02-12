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

def build_ablation_table(pkl_paths, output_csv=None):
    """
    Build model selection table across K for HMM-Tweedie family.
    pkl_paths: list of paths or glob pattern
    """
    if isinstance(pkl_paths, str):
        pkl_paths = list(Path('.').glob(pkl_paths))
    
    rows = []
    for pkl_path in sorted(pkl_paths):
        try:
            idata, res = load_idata(pkl_path)
            
            # Compute WAIC if available
            try:
                import arviz as az
                waic = az.waic(idata)
                waic_val = waic.waic
                waic_se = waic.waic_se
            except:
                waic_val = np.nan
                waic_se = np.nan
            
            row = {
                'file': Path(pkl_path).name,
                'dataset': res.get('dataset', 'unknown'),
                'K': res['K'],
                'model': 'GAM' if res['use_gam'] else 'GLM',
                'state_specific_p': res.get('state_specific_p', False),
                'N': res['N'],
                'draws': res['draws'],
                'log_evidence': res['log_evidence'],
                'waic': waic_val,
                'waic_se': waic_se,
                'time_min': res['time_min']
            }
            rows.append(row)
        except Exception as e:
            print(f"Skipping {pkl_path}: {e}")
    
    df = pd.DataFrame(rows)
    
    # Compute delta vs K=1 (GAM) for each dataset
    for dataset in df['dataset'].unique():
        mask = df['dataset'] == dataset
        k1_gam = df.loc[mask & (df['K'] == 1) & (df['model'] == 'GAM'), 'log_evidence']
        if len(k1_gam) > 0:
            baseline = k1_gam.values[0]
            df.loc[mask, 'delta_vs_k1gam'] = df.loc[mask, 'log_evidence'] - baseline
    
    df = df.sort_values(['dataset', 'K', 'model'])
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")
    
    return df


def compute_posterior_predictive_metrics(idata, actual_y=None):
    """
    Compute in-sample RMSE and R-squared from posterior predictive.
    Requires actual_y (N, T) to compare.
    """
    if actual_y is None:
        return {'rmse': np.nan, 'r2': np.nan, 'mae': np.nan}
    
    # Posterior predictive mean
    log_lik = idata.posterior['log_likelihood'].values  # (chains, draws, N)
    
    # Reconstruct y_pred from log_lik (approximate)
    # Better: sample from posterior predictive distribution
    # For now, return NaN - requires full posterior predictive sampling
    
    return {'rmse': np.nan, 'r2': np.nan, 'mae': np.nan}
    

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
    K = Gamma.shape[0]  # <-- ADD THIS LINE
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
    parser.add_argument('input_path', help='Path to .pkl file OR directory containing .pkl files')
    parser.add_argument('--step', choices=['state', 'clv', 'roi', 'ablation', 'all'], 
                       default='all', help='Which extraction step')
    parser.add_argument('--n_sims', type=int, default=2000, help='CLV simulations')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # ABILATION MODE: Directory containing multiple .pkl files
    if args.step == 'ablation':
        if input_path.is_dir():
            pkl_files = list(input_path.glob("*.pkl"))
            output_csv = input_path / 'ablation_table.csv'
        else:
            # Single file provided, use its parent directory
            pkl_files = list(input_path.parent.glob("*.pkl"))
            output_csv = input_path.parent / 'ablation_table.csv'
        
        print(f"Ablation mode: Found {len(pkl_files)} .pkl files")
        
        df = build_ablation_table(pkl_files, str(output_csv))
        print("\n" + "="*70)
        print("ABLATION TABLE")
        print("="*70)
        print(df.to_string())
        print(f"\nSaved: {output_csv}")
        return
    
    # SINGLE FILE MODE: Process one .pkl file
    if not input_path.is_file():
        print(f"Error: {input_path} is not a file. Use --step ablation for directories.")
        return
    
    pkl_path = str(input_path)
    print(f"Loading: {pkl_path}")
    idata, res = load_idata(pkl_path)
    
    print(f"Dataset: {res.get('dataset', 'unknown')}, K={res['K']}, N={res['N']}, log_ev={res['log_evidence']:.2f}")
    
    if args.step in ['state', 'all']:
        print("\n" + "="*70)
        print("STATE PARAMETERS")
        print("="*70)
        df_state = extract_state_table(idata)
        print(df_state.round(3).to_string())
        df_state.to_csv(pkl_path.replace('.pkl', '_state.csv'), index=False)
        print(f"Saved: {pkl_path.replace('.pkl', '_state.csv')}")
    
    if args.step in ['clv', 'all']:
        print("\n" + "="*70)
        print("CLV BY STARTING STATE")
        print("="*70)
        df_clv = extract_clv_table(idata, n_sims=args.n_sims)
        print(df_clv.round(2).to_string())
        df_clv.to_csv(pkl_path.replace('.pkl', '_clv.csv'), index=False)
        print(f"Saved: {pkl_path.replace('.pkl', '_clv.csv')}")
    
    if args.step in ['roi', 'all']:
        print("\n" + "="*70)
        print("REACTIVATION ROI")
        print("="*70)
        df_roi = extract_roi_table(idata, n_sims=args.n_sims)
        print(df_roi.round(2).to_string())
        df_roi.to_csv(pkl_path.replace('.pkl', '_roi.csv'), index=False)
        print(f"Saved: {pkl_path.replace('.pkl', '_roi.csv')}")
    
    print(f"\nDone.")


if __name__ == '__main__':
    main()
    
