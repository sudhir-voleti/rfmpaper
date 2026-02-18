#!/usr/bin/env python3
"""
simulation_generator.py
=======================
MIXTURE DGP with Recency-Dependent Churn (Gemini-enhanced)
Tests structural recovery under misspecification

Key features:
1. Recency-dependent churn (not fixed 20%)
2. High-variance Gamma(2, 4) for challenging recovery
3. Burn-in period (first 5 periods all dead)
4. Heterogeneous "vibe" activation
"""

import numpy as np
import pandas as pd
import pickle
import argparse
from datetime import datetime, timedelta

def generate_mixed_observations(n, zero_prob=0.70, gamma_shape=2.0, gamma_scale=4.0):
    """
    HIGH VARIANCE mixture: 70% zeros + 30% Gamma(2, 4)
    Mean = 8, Variance = 32 (very noisy active state)
    """
    is_zero = np.random.rand(n) < zero_prob
    result = np.zeros(n)
    n_positive = (~is_zero).sum()
    if n_positive > 0:
        result[~is_zero] = np.random.gamma(gamma_shape, gamma_scale, size=n_positive)
    return result

def generate_mixture_panel(n_customers=500, n_periods=100, 
                           zero_prob=0.70, gamma_shape=2.0, gamma_scale=4.0,
                           burn_in=5, seed=42):
    """
    GEMINI-ENHANCED mixture panel with recency-dependent churn.
    
    BAKED-IN FEATURES:
    1. BURN-IN: First 'burn_in' periods all customers are State 0
    2. Heterogeneous activation: Logistic "vibe" curve
    3. RECENCY-DEPENDENT CHURN: P(churn) = 0.20 + 0.02 * days_since_purchase
    4. High-variance emissions: Gamma(2, 4) when active
    5. Target: >75% overall zeros, challenging state recovery
    """
    np.random.seed(seed)
    N, T = n_customers, n_periods

    # BURN-IN: Everyone starts dead for first 'burn_in' periods
    # Then logistic "vibe" activation
    time_norm = np.linspace(-8, 0, T - burn_in)  # Very slow ramp
    p_active_vibe = np.concatenate([
        np.zeros(burn_in),  # Burn-in: zero activation probability
        1 / (1 + np.exp(-time_norm))  # Then slow ramp
    ])
    
    states = np.zeros((N, T), dtype=int)
    obs = np.zeros((N, T))
    
    # Track last purchase for recency-dependent churn
    last_purchase = np.full(N, -999)

    for t in range(T):
        for i in range(N):
            if t == 0:
                states[i, t] = 0
            elif states[i, t-1] == 0:
                # Dead -> Active based on vibe curve
                states[i, t] = 1 if np.random.rand() < p_active_vibe[t] else 0
                if states[i, t] == 1:
                    last_purchase[i] = t  # Just activated
            else:
                # ACTIVE: Recency-dependent churn
                # Longer since last purchase = higher churn probability
                days_since = t - last_purchase[i] if last_purchase[i] >= 0 else 0
                churn_prob = 0.20 + (0.02 * days_since)  # Increases 2% per day
                churn_prob = min(churn_prob, 0.80)  # Cap at 80%
                
                if np.random.rand() < churn_prob:
                    states[i, t] = 0  # Churn back to dead
                    # last_purchase stays as is (memory of when they were active)
                else:
                    states[i, t] = 1  # Stay active
                    # Update last_purchase if they bought something this period
                    # (handled after observation generation)

        # Generate observations for this period
        active_customers = (states[:, t] == 1)
        n_active = active_customers.sum()
        if n_active > 0:
            obs[active_customers, t] = generate_mixed_observations(
                n=n_active,
                zero_prob=zero_prob,
                gamma_shape=gamma_shape,
                gamma_scale=gamma_scale
            )
            # Update last_purchase for those who bought (positive spend)
            positive_spend = obs[:, t] > 0
            last_purchase[positive_spend] = t

    # Calculate RFM metrics
    r_weeks = np.zeros((N, T))
    f_run = np.zeros((N, T))
    m_run = np.zeros((N, T))

    for i in range(N):
        last_purchase_i, cum_count, cum_spend = -999, 0, 0.0
        for t in range(T):
            spend = obs[i, t]
            r_weeks[i, t] = t - last_purchase_i if last_purchase_i >= 0 else 999
            if spend > 0:
                cum_count += 1
                last_purchase_i = t
                cum_spend += spend
            f_run[i, t] = cum_count
            m_run[i, t] = cum_spend / cum_count if cum_count > 0 else 0

    # Ground truth
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
            'DGP_type': 'Mixture_with_Recency_Churn',
            'zero_prob': zero_prob,
            'gamma_shape': gamma_shape,
            'gamma_scale': gamma_scale,
            'burn_in': burn_in,
            'base_churn': 0.20,
            'churn_increment': 0.02
        },
        'true_switch_day': true_switch_day,
        'last_purchase': last_purchase,
        'generation_seed': seed
    }

def save_to_rfm_csv(data_dict, output_csv):
    """Convert panel to RFM CSV format"""
    N, T = data_dict['N'], data_dict['T']
    obs = data_dict['obs_matrix']
    states = data_dict['states_matrix']
    r_weeks = data_dict['r_matrix']
    f_run = data_dict['f_matrix']
    m_run = data_dict['m_matrix']
    
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
    
    # Diagnostics
    overall_zero_rate = (df['WeeklySpend'] == 0).mean()
    active_mask = df['true_state'] == 1
    active_zero_rate = (df.loc[active_mask, 'WeeklySpend'] == 0).mean() if active_mask.sum() > 0 else 0
    
    print(f"Saved RFM CSV: {output_csv}")
    print(f"  Shape: {df.shape}")
    print(f"  Overall zero rate: {overall_zero_rate:.1%}")
    print(f"  Zero rate in Active state: {active_zero_rate:.1%}")
    print(f"  Mean spend (when >0): {df[df['WeeklySpend'] > 0]['WeeklySpend'].mean():.2f}")
    print(f"  Std spend (when >0): {df[df['WeeklySpend'] > 0]['WeeklySpend'].std():.2f}")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='GEMINI-ENHANCED Mixture DGP with Recency-Dependent Churn'
    )
    parser.add_argument('--n_customers', type=int, default=500)
    parser.add_argument('--n_periods', type=int, default=100)
    parser.add_argument('--zero_prob', type=float, default=0.70)
    parser.add_argument('--gamma_shape', type=float, default=2.0)
    parser.add_argument('--gamma_scale', type=float, default=4.0,
                        help='Higher = more variance, harder recovery (default: 4.0)')
    parser.add_argument('--burn_in', type=int, default=5,
                        help='Initial periods where all are State 0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_pkl', type=str, default='ground_truth.pkl')
    parser.add_argument('--output_csv', type=str, default=None)
    
    args = parser.parse_args()
    
    print("="*70)
    print("GEMINI-ENHANCED MIXTURE DGP")
    print("="*70)
    print(f"Customers: {args.n_customers}, Periods: {args.n_periods}")
    print(f"DGP: {args.zero_prob:.0%} zeros + {100-args.zero_prob:.0%} Gamma({args.gamma_shape}, {args.gamma_scale})")
    print(f"Burn-in: {args.burn_in} periods (all State 0)")
    print(f"Churn: Recency-dependent (20% + 2% per day since purchase)")
    print("="*70)
    print("\nRECOVERY CHALLENGE:")
    print("  - HMM-Tweedie: Misspecified emission (assumes Tweedie, true is Mixture)")
    print("  - HMM-Hurdle: Closer but still misspecified (binary gate vs recency-churn)")
    print("  - Test: Can Tweedie's flexibility beat Hurdle's structural similarity?")
    print("="*70)
    
    data = generate_mixture_panel(
        n_customers=args.n_customers,
        n_periods=args.n_periods,
        zero_prob=args.zero_prob,
        gamma_shape=args.gamma_shape,
        gamma_scale=args.gamma_scale,
        burn_in=args.burn_in,
        seed=args.seed
    )
    
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"\nSaved ground truth to: {args.output_pkl}")
    
    # Summary
    print(f"\nGround Truth Summary:")
    dead_obs = (data['states_matrix'] == 0).sum()
    active_obs = (data['states_matrix'] == 1).sum()
    print(f"  State 0 (Dead): {dead_obs} obs ({100*dead_obs/(dead_obs+active_obs):.1f}%)")
    print(f"  State 1 (Active): {active_obs} obs ({100*active_obs/(dead_obs+active_obs):.1f}%)")
    
    n_activated = (data['true_switch_day'] >= 0).sum()
    print(f"  Customers activated: {n_activated}/{data['N']} ({100*n_activated/data['N']:.1f}%)")
    if n_activated > 0:
        print(f"  Mean activation day: {data['true_switch_day'][data['true_switch_day'] >= 0].mean():.1f}")
    
    active_spend = data['obs_matrix'][data['states_matrix'] == 1]
    if len(active_spend) > 0:
        print(f"  Active state - Mean: {active_spend.mean():.2f}, Std: {active_spend.std():.2f}")
        print(f"  Active state - Zero rate: {(active_spend == 0).mean():.1%}")
    
    overall_zeros = (data['obs_matrix'] == 0).mean()
    print(f"\n  *** OVERALL ZERO RATE: {overall_zeros:.1%} ***")
    print(f"  *** DGP: {data['true_params']['DGP_type']} ***")
    
    if args.output_csv:
        save_to_rfm_csv(data, args.output_csv)
    
    print("\n" + "="*70)
    print("PAPER STORY:")
    print("  If HMM-Tweedie recovers state timing better than HMM-Hurdle,")
    print("  despite Hurdle being structurally closer to true DGP,")
    print("  then Tweedie's semi-continuous kernel is the superior")
    print("  'structural lens' for latent behavioral shifts.")
    print("="*70)
    print("\nDone.")

if __name__ == "__main__":
    main()
