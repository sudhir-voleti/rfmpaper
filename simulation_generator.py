#!/usr/bin/env python3
"""
simulation_generator.py
=======================
Generate synthetic RFM panel data with known ground truth states.
Supports multiple DGPs: mixture (unimodal) and bimodal.

Usage:
    python simulation_generator.py --dgp mixture --n_customers 50 --n_periods 60
    python simulation_generator.py --dgp bimodal --n_customers 100 --n_periods 60
"""

import numpy as np
import pandas as pd
import pickle
import argparse
from datetime import datetime, timedelta


def rcompound_poisson_gamma(n, lambda_poisson, shape_gamma, scale_gamma, p=1.2):
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


def generate_mixed_observations(n, zero_prob=0.70, gamma_shape=2.0, gamma_scale=4.0):
    """
    MIXTURE DGP (unimodal): zero_prob zeros + Gamma(gamma_shape, gamma_scale)
    """
    is_zero = np.random.rand(n) < zero_prob
    result = np.zeros(n)
    n_positive = (~is_zero).sum()
    if n_positive > 0:
        result[~is_zero] = np.random.gamma(gamma_shape, gamma_scale, size=n_positive)
    return result


def generate_bimodal_observations(n, zero_prob=0.60):
    """
    BIMODAL DGP (extreme): zero_prob zeros + 50% Gamma(1,1) + 50% Gamma(10,2)
    Creates massive variance that NBD cannot capture.
    """
    is_zero = np.random.rand(n) < zero_prob
    result = np.zeros(n)
    n_positive = (~is_zero).sum()
    
    if n_positive > 0:
        is_whale = np.random.rand(n_positive) < 0.5
        result_pos = np.zeros(n_positive)
        result_pos[is_whale] = np.random.gamma(10, 2, is_whale.sum())  # Whales: mean=20
        result_pos[~is_whale] = np.random.gamma(1, 1, (~is_whale).sum())  # Low: mean=1
        result[~is_zero] = result_pos
    
    return result


def generate_panel(dgp_type='mixture', n_customers=500, n_periods=100, seed=42, 
                   burn_in=5, **obs_kwargs):
    """
    Generate panel with specified DGP type.
    
    Parameters:
    -----------
    dgp_type : str
        'mixture' or 'bimodal'
    n_customers, n_periods, seed : standard
    **kwargs : passed to observation generator (zero_prob, etc.)
    """
    np.random.seed(seed)
    N, T = n_customers, n_periods
    
    # State dynamics (same for both DGPs)
    burn_in = kwargs.get('burn_in', 5)
    time_norm = np.linspace(-6, 2, T - burn_in)
    p_active_vibe = np.concatenate([np.zeros(burn_in), 1 / (1 + np.exp(-time_norm))])
    
    states = np.zeros((N, T), dtype=int)
    obs = np.zeros((N, T))
    last_purchase = np.full(N, -999)

    for t in range(T):
        for i in range(N):
            if t == 0:
                states[i, t] = 0
            elif states[i, t-1] == 0:
                states[i, t] = 1 if np.random.rand() < p_active_vibe[t] else 0
                if states[i, t] == 1:
                    last_purchase[i] = t
            else:
                days_since = t - last_purchase[i] if last_purchase[i] >= 0 else 0
                churn_prob = 0.20 + (0.02 * days_since)
                churn_prob = min(churn_prob, 0.80)
                states[i, t] = 0 if np.random.rand() < churn_prob else 1
        
        # Generate observations based on DGP type
        active = (states[:, t] == 1)
        if active.sum() > 0:
            if dgp_type == 'bimodal':
                obs[active, t] = generate_bimodal_observations(active.sum(), **kwargs)
            else:  # mixture
                obs[active, t] = generate_mixed_observations(active.sum(), **kwargs)
            
            positive = obs[:, t] > 0
            last_purchase[positive] = t

    # RFM metrics
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

    true_switch_day = np.full(N, -1)
    for i in range(N):
        active_times = np.where(states[i, :] == 1)[0]
        if len(active_times) > 0:
            true_switch_day[i] = active_times[0]

    # DGP-specific metadata
    if dgp_type == 'bimodal':
        true_params = {
            'DGP_type': 'Bimodal_Mixture',
            'zero_prob': kwargs.get('zero_prob', 0.60),
            'low_dist': 'Gamma(1, 1)',
            'whale_dist': 'Gamma(10, 2)',
            'whale_share': 0.50
        }
    else:
        true_params = {
            'DGP_type': 'Mixture',
            'zero_prob': kwargs.get('zero_prob', 0.70),
            'gamma_shape': kwargs.get('gamma_shape', 2.0),
            'gamma_scale': kwargs.get('gamma_scale', 4.0)
        }

    return {
        'obs_matrix': obs,
        'states_matrix': states,
        'r_matrix': r_weeks,
        'f_matrix': f_run,
        'm_matrix': m_run,
        'N': N, 'T': T,
        'true_params': true_params,
        'true_switch_day': true_switch_day,
        'generation_seed': seed,
        'dgp_type': dgp_type
    }


def save_to_rfm_csv(data_dict, output_csv):
    """Convert panel to RFM CSV format."""
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
    overall_zero = (df['WeeklySpend'] == 0).mean()
    active_mask = df['true_state'] == 1
    active_df = df[active_mask]
    
    print(f"Saved: {output_csv}")
    print(f"  Shape: {df.shape}")
    print(f"  Overall zero rate: {overall_zero:.1%}")
    
    if active_mask.sum() > 0:
        pos_spends = active_df[active_df['WeeklySpend'] > 0]['WeeklySpend']
        print(f"  Active state - Zeros: {(active_df['WeeklySpend'] == 0).mean():.1%}")
        print(f"  Active state - Mean: {pos_spends.mean():.2f}, Std: {pos_spends.std():.2f}")
        
        # Bimodality check if applicable
        if pos_spends.std() > pos_spends.mean():
            low_share = (pos_spends < pos_spends.quantile(0.33)).mean()
            high_share = (pos_spends > pos_spends.quantile(0.67)).mean()
            print(f"  Low tertile (<{pos_spends.quantile(0.33):.1f}): {low_share:.1%}")
            print(f"  High tertile (>{pos_spends.quantile(0.67):.1f}): {high_share:.1%}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic RFM panel')
    parser.add_argument('--dgp', type=str, default='mixture', 
                        choices=['mixture', 'bimodal'],
                        help='DGP type: mixture (unimodal) or bimodal (extreme)')
    parser.add_argument('--n_customers', type=int, default=500)
    parser.add_argument('--n_periods', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_pkl', type=str, default=None)
    parser.add_argument('--output_csv', type=str, default=None)
    
    # DGP-specific params
    parser.add_argument('--zero_prob', type=float, default=None,
                        help='Zero probability in active state (default: 0.70 mixture, 0.60 bimodal)')
    parser.add_argument('--gamma_shape', type=float, default=2.0,
                        help='[Mixture only] Gamma shape')
    parser.add_argument('--gamma_scale', type=float, default=4.0,
                        help='[Mixture only] Gamma scale')
    
    args = parser.parse_args()
    
    # Set defaults based on DGP type
    if args.zero_prob is None:
        args.zero_prob = 0.60 if args.dgp == 'bimodal' else 0.70
    
    # Auto-generate output names if not specified
    if args.output_pkl is None:
        args.output_pkl = f'ground_truth_{args.dgp}.pkl'
    if args.output_csv is None:
        args.output_csv = f'simulation_{args.dgp}_rfm.csv'
    
    print("="*70)
    print(f"GENERATING {args.dgp.upper()} DGP")
    print("="*70)
    print(f"Customers: {args.n_customers}, Periods: {args.n_periods}")
    print(f"Zero prob: {args.zero_prob}")
    if args.dgp == 'mixture':
        print(f"Positive: Gamma({args.gamma_shape}, {args.gamma_scale})")
    else:
        print(f"Positive: 50% Gamma(1,1) + 50% Gamma(10,2) [BIMODAL]")
    print(f"Seed: {args.seed}")
    print("="*70)
    
    # Generate
    kwargs = {
        'zero_prob': args.zero_prob,
        'burn_in': 5
    }
    if args.dgp == 'mixture':
        kwargs.update({
            'gamma_shape': args.gamma_shape,
            'gamma_scale': args.gamma_scale
        })
    data = generate_panel(
        dgp_type=args.dgp,
        n_customers=args.n_customers,
        n_periods=args.n_periods,
        seed=args.seed,
        burn_in=5,  # Pass separately
        **obs_kwargs  # Only zero_prob, gamma_shape, gamma_scale
    )
    
    # Save
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"\nSaved: {args.output_pkl}")
    
    # Summary
    print(f"\nGround Truth:")
    dead_pct = 100 * (data['states_matrix'] == 0).mean()
    active_pct = 100 * (data['states_matrix'] == 1).mean()
    print(f"  Dead: {dead_pct:.1f}%, Active: {active_pct:.1f}%")
    
    n_activated = (data['true_switch_day'] >= 0).sum()
    print(f"  Activated: {n_activated}/{data['N']} ({100*n_activated/data['N']:.1f}%)")
    
    save_to_rfm_csv(data, args.output_csv)
    
    print("\n" + "="*70)
    if args.dgp == 'bimodal':
        print("BIMODAL DGP: Tests if Tweedie-HMM beats NBD on extreme data")
    else:
        print("MIXTURE DGP: Standard test with high zero-inflation")
    print("="*70)

if __name__ == "__main__":
    main()
