#!/usr/bin/env python3
"""
simulation_generator.py
=======================
Generate synthetic RFM panel with principled 3-state mixture DGP (NOT Tweedie).
Designed to test HMM-Tweedie recovery of latent regimes despite misspecification.
"""

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.special import expit
import pickle
import argparse
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional

# =============================================================================
# 1. STATIONARY MOMENT SOLVER (corrected for Dead state π₀=1.0)
# =============================================================================
def solve_stationary_moments(Gamma, target_mu, target_pi0, anchors, separation_factor=0.8):
    """
    Solve for state-specific pi0 and mu to match target aggregate moments.
    Dead state (index 0) has pi0 = 1.0 by construction.
    """
    evals, evecs = scipy.linalg.eig(Gamma.T)
    delta = evecs[:, np.isclose(evals, 1.0, atol=1e-10)].real
    delta = (delta / delta.sum()).flatten()
    
    s1_pi0_base, s1_mu_base = anchors['s1']  # Cold
    s3_pi0_base, s3_mu_base = anchors['s3']  # Hot
    
    s1_mu = target_mu + (s1_mu_base - target_mu) * separation_factor
    s3_mu = target_mu + (s3_mu_base - target_mu) * separation_factor
    s1_pi0 = target_pi0 + (s1_pi0_base - target_pi0) * separation_factor
    s3_pi0 = target_pi0 + (s3_pi0_base - target_pi0) * separation_factor
    
    # Corrected: Dead state (0) contributes delta[0] * 1.0 to total pi0
    s2_pi0 = (target_pi0 - delta[0] * 1.0 - delta[1] * s1_pi0 - delta[3] * s3_pi0) / delta[2]
    s2_pi0 = np.clip(s2_pi0, 0.05, 0.95)
    
    target_sum_mu = target_mu * (1 - target_pi0)
    term1 = delta[1] * (1 - s1_pi0) * s1_mu
    term3 = delta[3] * (1 - s3_pi0) * s3_mu
    s2_mu = (target_sum_mu - term1 - term3) / (delta[2] * (1 - s2_pi0))
    s2_mu = max(s2_mu, 2.0)
    
    return delta, [s1_pi0, s2_pi0, s3_pi0], [s1_mu, s2_mu, s3_mu]

# =============================================================================
# 2. PANEL GENERATOR
# =============================================================================
def generate_principled_panel(
    n_customers=500,
    n_periods=100,
    target_pi0=0.75,
    target_mu=45.0,
    target_cv=2.0,
    separation=0.8,
    seed=42,
    burn_in=5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate synthetic RFM panel with behavioral states.
    
    Returns:
        df: Long-format DataFrame with columns [customer_id, time_period, y, state, recency, frequency, monetary]
        meta: Dictionary with ground truth parameters
    """
    np.random.seed(seed)
    N, T = n_customers, n_periods
    K = 3  # Cold, Transitional, Hot

    # Transition matrix (4 states: Dead, Cold, Transitional, Hot)
    Gamma_base = np.array([
        [0.90, 0.07, 0.02, 0.01],  # From Dead
        [0.10, 0.80, 0.08, 0.02],  # From Cold
        [0.05, 0.10, 0.80, 0.05],  # From Transitional
        [0.02, 0.03, 0.10, 0.85]   # From Hot
    ])

    # Anchor points for extreme states (Cold and Hot)
    anchors = {'s1': (0.95, 5.0), 's3': (0.15, 150.0)}
    delta, pi_ks, mu_ks = solve_stationary_moments(Gamma_base, target_mu, target_pi0, anchors, separation)

    # State-specific CV → Gamma parameters
    cv_ks = [1.0, 1.5, target_cv * 1.5]
    gamma_shapes = [1.0 / (cv**2) for cv in cv_ks]
    gamma_scales = [mu / shape for mu, shape in zip(mu_ks, gamma_shapes)]

    # Customer-specific activation timing (logistic curve with random shift)
    time_norm = np.linspace(-8, 4, T - burn_in)
    base_vibe = 1 / (1 + np.exp(-time_norm))
    customer_shift = np.random.normal(0, 2, N)
    p_activate = np.clip(base_vibe[None, :] + customer_shift[:, None] * 0.1, 0, 1)

    # Initialize
    obs = np.zeros((N, T))
    states = np.zeros((N, T), dtype=int)
    recency = np.full((N, T), 999)

    # Burn-in: all start dead
    states[:, :burn_in] = 0
    recency[:, :burn_in] = 999

    # Simulate dynamics
    for t in range(burn_in, T):
        for i in range(N):
            curr = states[i, t-1]
            # Activation from dead
            if curr == 0 and np.random.rand() < p_activate[i, t-burn_in]:
                states[i, t] = 1
            else:
                states[i, t] = np.random.choice(4, p=Gamma_base[curr])

            # Emission (only in non-dead states)
            s = states[i, t]
            if s > 0:
                cliff_effect = expit(-(recency[i, t] - 10) * 0.6)
                prob_spend = (1 - pi_ks[s-1]) * cliff_effect
                if np.random.rand() < prob_spend:
                    obs[i, t] = np.random.gamma(gamma_shapes[s-1], gamma_scales[s-1])

            # Update recency
            if t < T-1:
                recency[i, t+1] = 0 if obs[i, t] > 0 else recency[i, t] + 1

    # Build RFM covariates
    r_weeks = recency.copy()
    f_run = np.zeros((N, T))
    m_run = np.zeros((N, T))

    for i in range(N):
        cum_count = cum_spend = 0.0
        for t in range(T):
            spend = obs[i, t]
            if spend > 0:
                cum_count += 1
                cum_spend += spend
            f_run[i, t] = cum_count
            m_run[i, t] = cum_spend / cum_count if cum_count > 0 else 0.0

    # Build long-format DataFrame
    data_list = []
    for i in range(N):
        for t in range(T):
            data_list.append({
                'customer_id': i,
                'time_period': t,
                'y': obs[i, t],
                'true_state': states[i, t],
                'recency': r_weeks[i, t],
                'frequency': f_run[i, t],
                'monetary': m_run[i, t]
            })
    
    df = pd.DataFrame(data_list)

    # Compute actual moments
    actual_pi0 = np.mean(obs == 0)
    active_obs = obs[obs > 0]
    actual_mu = np.mean(active_obs) if len(active_obs) > 0 else 0.0
    actual_cv = np.std(active_obs) / actual_mu if actual_mu > 0 else 0.0

    meta = {
        'N': N,
        'T': T,
        'seed': seed,
        'target_pi0': target_pi0,
        'target_mu': target_mu,
        'target_cv': target_cv,
        'actual_pi0': actual_pi0,
        'actual_mu': actual_mu,
        'actual_cv': actual_cv,
        'zero_probs': pi_ks,
        'mu_ks': mu_ks,
        'gamma_shapes': gamma_shapes,
        'gamma_scales': gamma_scales,
        'delta_stationary': delta,
        'Gamma_base': Gamma_base,
        'DGP_type': 'Principled_Mixture'
    }

    return df, meta

# =============================================================================
# 3. OUTPUT FUNCTIONS
# =============================================================================
def save_simulation_csv(df: pd.DataFrame, output_path: str, include_state: bool = True):
    """Save simulation to CSV format."""
    cols = ['customer_id', 'time_period', 'y']
    if include_state and 'true_state' in df.columns:
        cols.append('true_state')
    
    df[cols].to_csv(output_path, index=False)
    print(f"Saved CSV: {output_path}")
    print(f"  Rows: {len(df)}, Customers: {df['customer_id'].nunique()}, Periods: {df['time_period'].nunique()}")
    if include_state:
        print(f"  Includes: true_state column")

def save_simulation_pkl(df: pd.DataFrame, meta: Dict, output_path: str):
    """Save simulation to pickle format."""
    with open(output_path, "wb") as f:
        pickle.dump({'df': df, 'meta': meta}, f)
    print(f"Saved PKL: {output_path}")

# =============================================================================
# 4. DIAGNOSTIC VISUALIZATION
# =============================================================================
def run_diagnostics(df: pd.DataFrame, meta: Dict, output_dir: str):
    """Generate diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Recency Cliff
    plt.figure(figsize=(8, 5))
    df_active = df[df['true_state'] > 0].copy()
    df_active['is_spend'] = (df_active['y'] > 0).astype(int)
    sns.lineplot(data=df_active, x='recency', y='is_spend', err_style="bars")
    plt.title("Recency Cliff (Behavioral Non-linearity)")
    plt.ylabel("P(Spend)")
    plt.savefig(output_dir / "recency_cliff.png", dpi=150, bbox_inches='tight')
    plt.close()

    # State-Wise Intensity
    plt.figure(figsize=(8, 5))
    df_spend = df[df['y'] > 0]
    if len(df_spend) > 0:
        sns.boxplot(data=df_spend, x='true_state', y='y', palette="Set2")
        plt.yscale('log')
        plt.title("State-Wise Spending Intensity (Log Scale)")
        plt.savefig(output_dir / "state_intensity.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Transition Matrix Heatmap
    plt.figure(figsize=(6, 5))
    Gamma = meta['Gamma_base']
    sns.heatmap(Gamma, annot=True, fmt='.2f', cmap="YlGnBu",
                xticklabels=['Dead', 'Cold', 'Trans', 'Hot'],
                yticklabels=['Dead', 'Cold', 'Trans', 'Hot'])
    plt.title("Ground Truth Transition Matrix")
    plt.savefig(output_dir / "transitions.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Diagnostic plots saved to: {output_dir}")

# =============================================================================
# 5. MAIN & CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Principled synthetic RFM panel generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n_customers", type=int, default=500, help="Number of customers")
    parser.add_argument("--n_periods", type=int, default=100, help="Number of time periods")
    parser.add_argument("--target_pi0", type=float, default=0.75, help="Target zero-inflation rate")
    parser.add_argument("--target_mu", type=float, default=45.0, help="Target mean spend (active)")
    parser.add_argument("--target_cv", type=float, default=2.0, help="Target coefficient of variation")
    parser.add_argument("--separation", type=float, default=0.8, help="State separation factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="sim_principled", help="Output file prefix")
    parser.add_argument("--format", type=str, default="both", choices=["csv", "pkl", "both"], 
                       help="Output format")
    parser.add_argument("--include_state", action="store_true", default=True, 
                       help="Include true_state in CSV")
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots")
    parser.add_argument("--plot_dir", type=str, default=None, help="Plot output directory")
    
    args = parser.parse_args()

    print("="*70)
    print("PRINCIPLED SIMULATION GENERATOR")
    print("="*70)
    print(f"Parameters: N={args.n_customers}, T={args.n_periods}, seed={args.seed}")
    print(f"Targets: π₀={args.target_pi0:.2%}, μ=${args.target_mu:.1f}, CV={args.target_cv:.2f}")

    # Generate
    df, meta = generate_principled_panel(
        n_customers=args.n_customers,
        n_periods=args.n_periods,
        target_pi0=args.target_pi0,
        target_mu=args.target_mu,
        target_cv=args.target_cv,
        separation=args.separation,
        seed=args.seed
    )

    print("\n" + "-"*70)
    print("ACTUAL MOMENTS")
    print("-"*70)
    print(f"Zero rate: {meta['actual_pi0']:.2%} (target: {args.target_pi0:.2%})")
    print(f"Mean spend: ${meta['actual_mu']:.2f} (target: ${args.target_mu:.1f})")
    print(f"CV: {meta['actual_cv']:.2f} (target: {args.target_cv:.2f})")
    print(f"Stationary dist: {meta['delta_stationary'].round(3)}")

    # Save outputs
    print("\n" + "-"*70)
    print("SAVING")
    print("-"*70)
    
    if args.format in ["csv", "both"]:
        csv_path = f"{args.output}.csv"
        save_simulation_csv(df, csv_path, include_state=args.include_state)
    
    if args.format in ["pkl", "both"]:
        pkl_path = f"{args.output}.pkl"
        save_simulation_pkl(df, meta, pkl_path)

    # Plots
    if args.plot:
        print("\n" + "-"*70)
        print("GENERATING DIAGNOSTICS")
        print("-"*70)
        plot_dir = args.plot_dir or f"plots_{args.output}"
        run_diagnostics(df, meta, plot_dir)

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
