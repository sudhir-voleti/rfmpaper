#!/usr/bin/env python3
"""
simulation_generator.py
=======================
Generate synthetic RFM panel data with known ground truth states.
For testing state recovery in HMM models.

Usage:
    python simulation_generator.py --n_customers 500 --n_periods 100 --output_pkl ground_truth.pkl
    
Or via exec_gitcode.py:
    exec_from_github(url, ['--n_customers', '50', '--n_periods', '20', '--output_csv', 'sim_data.csv'])
"""

import numpy as np
import pandas as pd
import pickle
import argparse
from datetime import datetime, timedelta

def rcompound_poisson_gamma(n, lambda_poisson, shape_gamma, scale_gamma, p=1.6):
    """Generate Tweedie(p) via compound Poisson-Gamma."""
    N = np.random.poisson(lambda_poisson, size=n)
    samples = np.zeros(n)
    positive_mask = N > 0
    if np.any(positive_mask):
        total_components = N[positive_mask].sum()
        gamma_draws = np.random.gamma(shape=shape_gamma, scale=scale_gamma, size=total_components)
        indices = np.concatenate([[0], np.cumsum(N[positive_mask])[:-1]])
        samples[positive_mask] = np.add.reduceat(gamma_draws, indices)
    return samples

def generate_hmm_tweedie_panel(n_customers=500, n_periods=100, p_true=1.6,
                                phi_true=2.0, mu_active=5.0, seed=42):
    """Generate panel with stochastic state transitions and Tweedie observations."""
    np.random.seed(seed)
    N, T = n_customers, n_periods

    # State transitions (logistic "vibe")
    time_norm = np.linspace(-3, 3, T)
    p_active_vibe = 1 / (1 + np.exp(-time_norm))

    states = np.zeros((N, T), dtype=int)
    states[:, 0] = 0

    for t in range(1, T):
        for i in range(N):
            if states[i, t-1] == 0:
                states[i, t] = 1 if np.random.rand() < p_active_vibe[t] else 0
            else:
                states[i, t] = 1 if np.random.rand() < 0.95 else 0

    # Tweedie observations
    lambda_poisson = (mu_active ** (2 - p_true)) / (phi_true * (2 - p_true))
    shape_gamma = (2 - p_true) / (p_true - 1)
    scale_gamma = phi_true * (p_true - 1) * (mu_active ** (p_true - 1))

    obs = np.zeros((N, T))
    for t in range(T):
        active = (states[:, t] == 1)
        if active.sum() > 0:
            obs[active, t] = rcompound_poisson_gamma(active.sum(), lambda_poisson,
                                                      shape_gamma, scale_gamma, p_true)

    # RFM metrics
    r_weeks = np.zeros((N, T))
    f_run = np.zeros((N, T))
    m_run = np.zeros((N, T))

    for i in range(N):
        last_purchase, cum_count, cum_spend = -999, 0, 0.0
        for t in range(T):
            spend = obs[i, t]
            r_weeks[i, t] = t - last_purchase if last_purchase >= 0 else 999
            if spend > 0:
                cum_count += 1
                last_purchase = t
                cum_spend += spend
            f_run[i, t] = cum_count
            m_run[i, t] = cum_spend / cum_count if cum_count > 0 else 0

    # True switch day
    true_switch_day = np.full(N, -1)
    for i in range(N):
        active_times = np.where(states[i, :] == 1)[0]
        if len(active_times) > 0:
            true_switch_day[i] = active_times[0]

    return {
        'obs_matrix': obs,
        'states_matrix': states,
        'r_matrix': r_weeks,
        'f_matrix': f_run,
        'm_matrix': m_run,
        'N': N,
        'T': T,
        'true_params': {
            'p': p_true,
            'phi': phi_true,
            'mu_active': mu_active,
            'lambda_poisson': lambda_poisson,
            'shape_gamma': shape_gamma,
            'scale_gamma': scale_gamma
        },
        'true_switch_day': true_switch_day,
        'generation_seed': seed
    }

def save_to_rfm_csv(data_dict, output_csv):
    """Convert panel data to RFM CSV format for smc_unified.py"""
    N, T = data_dict['N'], data_dict['T']
    obs = data_dict['obs_matrix']
    states = data_dict['states_matrix']
    r_weeks = data_dict['r_matrix']
    f_run = data_dict['f_matrix']
    m_run = data_dict['m_matrix']
    
    # Create synthetic dates (weekly)
    base_date = datetime(2020, 1, 1)
    dates = [base_date + timedelta(weeks=t) for t in range(T)]
    
    rows = []
    for i in range(N):
        for t in range(T):
            rows.append({
                'customer_id': f'CUST_{i:04d}',
                'WeekStart': dates[t],
                'WeeklySpend': obs[i, t],
                'R_weeks': r_weeks[i, t],
                'F_run': f_run[i, t],
                'M_run': m_run[i, t],
                'p0_cust': 0.0,
                'true_state': int(states[i, t])
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved RFM CSV: {output_csv}")
    print(f"  Shape: {df.shape}")
    print(f"  Zero rate: {(df['WeeklySpend'] == 0).mean():.1%}")
    print(f"  Mean spend (when >0): {df[df['WeeklySpend'] > 0]['WeeklySpend'].mean():.2f}")
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic RFM panel data with ground truth states'
    )
    parser.add_argument('--n_customers', type=int, default=500,
                        help='Number of customers (default: 500)')
    parser.add_argument('--n_periods', type=int, default=100,
                        help='Number of time periods (default: 100)')
    parser.add_argument('--p_true', type=float, default=1.6,
                        help='True Tweedie power parameter (default: 1.6)')
    parser.add_argument('--phi_true', type=float, default=2.0,
                        help='True Tweedie dispersion (default: 2.0)')
    parser.add_argument('--mu_active', type=float, default=5.0,
                        help='Mean spend in active state (default: 5.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output_pkl', type=str, default='ground_truth.pkl',
                        help='Output pickle file (default: ground_truth.pkl)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Optional: also save as RFM CSV for smc_unified.py')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GENERATING SYNTHETIC RFM PANEL DATA")
    print("="*60)
    print(f"Customers: {args.n_customers}")
    print(f"Periods: {args.n_periods}")
    print(f"Tweedie p: {args.p_true}, phi: {args.phi_true}")
    print(f"Seed: {args.seed}")
    print("="*60)
    
    # Generate data
    data = generate_hmm_tweedie_panel(
        n_customers=args.n_customers,
        n_periods=args.n_periods,
        p_true=args.p_true,
        phi_true=args.phi_true,
        mu_active=args.mu_active,
        seed=args.seed
    )
    
    # Save pickle
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"\nSaved ground truth to: {args.output_pkl}")
    
    # Print summary
    print(f"\nGround Truth Summary:")
    print(f"  State 0 (Dead): {(data['states_matrix'] == 0).sum()} obs ({100*(data['states_matrix'] == 0).mean():.1f}%)")
    print(f"  State 1 (Active): {(data['states_matrix'] == 1).sum()} obs ({100*(data['states_matrix'] == 1).mean():.1f}%)")
    
    n_activated = (data['true_switch_day'] >= 0).sum()
    print(f"  Customers activated: {n_activated}/{data['N']} ({100*n_activated/data['N']:.1f}%)")
    print(f"  Mean activation day: {data['true_switch_day'][data['true_switch_day'] >= 0].mean():.1f}")
    
    active_spend = data['obs_matrix'][data['states_matrix'] == 1]
    print(f"  Active state spend: mean={active_spend.mean():.2f}, std={active_spend.std():.2f}")
    print(f"  Zero rate in active state: {(active_spend == 0).mean():.1%}")
    
    # Optionally save CSV
    if args.output_csv:
        save_to_rfm_csv(data, args.output_csv)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
