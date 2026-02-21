#!/usr/bin/env python3
"""
simulation_generator.py
=======================
Generate synthetic RFM panel with principled 3-state DGP + Dead state.
Designed to test HMM-Tweedie recovery of heterogeneous regimes.
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
from typing import Dict, Tuple, Optional, List

# =============================================================================
# 1. WORLD PRESETS (2x2 Taxonomy)
# =============================================================================
WORLD_PRESETS = {
    'poisson': {
        'target_pi0': 0.20, 
        'target_mu': 50.0, 
        'target_cv': 0.5,
        'anchors': {'s1': (0.30, 30.0), 's3': (0.05, 80.0)},
        'desc': 'Low zero, low variance — Loyalty program members'
    },
    'sporadic': {
        'target_pi0': 0.75, 
        'target_mu': 60.0, 
        'target_cv': 0.8,
        'anchors': {'s1': (0.90, 20.0), 's3': (0.20, 100.0)},
        'desc': 'High zero, low variance — Seasonal subscriptions'
    },
    'gamma': {
        'target_pi0': 0.30, 
        'target_mu': 40.0, 
        'target_cv': 3.0,
        'anchors': {'s1': (0.50, 15.0), 's3': (0.10, 120.0)},
        'desc': 'Low zero, high variance — VIP whales'
    },
    'clumpy': {
        'target_pi0': 0.75, 
        'target_mu': 45.0, 
        'target_cv': 4.0,
        'anchors': {'s1': (0.95, 5.0), 's3': (0.15, 150.0)},
        'desc': 'High zero, high variance — Episodic heavy buyers'
    }
}

# =============================================================================
# 2. STATIONARY MOMENT SOLVER (4 states: Dead + 3 Active)
# =============================================================================
def solve_stationary_moments(Gamma: np.ndarray, target_mu: float, target_pi0: float, 
                            anchors: Dict, separation: float = 0.8) -> Tuple:
    """
    Solve for state-specific pi0 and mu to match target aggregate moments.
    States: 0=Dead (pi0=1.0), 1=Cold, 2=Warm, 3=Hot
    """
    evals, evecs = scipy.linalg.eig(Gamma.T)
    delta = evecs[:, np.isclose(evals, 1.0, atol=1e-10)].real
    delta = (delta / delta.sum()).flatten()
    
    s1_pi0_base, s1_mu_base = anchors['s1']  # Cold
    s3_pi0_base, s3_mu_base = anchors['s3']  # Hot
    
    # Apply separation factor
    s1_mu = target_mu + (s1_mu_base - target_mu) * separation
    s3_mu = target_mu + (s3_mu_base - target_mu) * separation
    s1_pi0 = target_pi0 + (s1_pi0_base - target_pi0) * separation
    s3_pi0 = target_pi0 + (s3_pi0_base - target_pi0) * separation
    
    # Solve for Warm state (index 2)
    # Dead state (0) has pi0=1.0 exactly
    s2_pi0 = (target_pi0 - delta[0]*1.0 - delta[1]*s1_pi0 - delta[3]*s3_pi0) / delta[2]
    s2_pi0 = np.clip(s2_pi0, 0.05, 0.95)
    
    # Solve for Warm mu
    target_active_spend = target_mu * (1 - target_pi0)
    term1 = delta[1] * (1 - s1_pi0) * s1_mu
    term3 = delta[3] * (1 - s3_pi0) * s3_mu

    s2_mu = (target_active_spend - term1 - term3) / (delta[2] * (1 - s2_pi0))
    s2_mu = max(s2_mu, 5.0)
    
    # === ORDER CONSTRAINT: Enforce mu_Cold < mu_Warm < mu_Hot ===
    # Collect active state parameters
    active_mus = [s1_mu, s2_mu, s3_mu]
    active_pi0s = [s1_pi0, s2_pi0, s3_pi0]
    
    # Sort by mean spend (ascending: Cold < Warm < Hot)
    order = np.argsort(active_mus)
    s1_mu, s2_mu, s3_mu = [active_mus[i] for i in order]
    s1_pi0, s2_pi0, s3_pi0 = [active_pi0s[i] for i in order]
    
    # Ensure minimum separation for identifiability
    min_sep = 2.0
    if s2_mu - s1_mu < min_sep:
        s2_mu = s1_mu + min_sep
    if s3_mu - s2_mu < min_sep:
        s3_mu = s2_mu + min_sep
    
    return delta, [1.0, s1_pi0, s2_pi0, s3_pi0], [0.0, s1_mu, s2_mu, s3_mu]

# =============================================================================
# 3. PANEL GENERATOR WITH CUSTOMER HETEROGENEITY
# =============================================================================

def generate_principled_panel(
    n_customers: int = 500,
    n_periods: int = 100,
    world: str = 'clumpy',
    separation: float = 0.8,
    seed: int = 42,
    burn_in: int = 5,
    customer_heterogeneity: float = 2.0,
    target_inflation: float = 1.25,
    recency_cap: int = 15
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate synthetic RFM panel with 4-state HMM (Dead + 3 Active).
    HYBRID: Stationary init + inflated targets + original activation logic.
    """
    np.random.seed(seed)
    N, T = n_customers, n_periods
    
    if world not in WORLD_PRESETS:
        raise ValueError(f"Unknown world: {world}. Choose from {list(WORLD_PRESETS.keys())}")
    
    preset = WORLD_PRESETS[world]
    target_pi0 = preset['target_pi0']
    inflated_mu = preset['target_mu'] * target_inflation
    target_cv = preset['target_cv'] * 0.85
    anchors = preset['anchors']
    
    # Transition matrix (4 states: Dead, Cold, Warm, Hot)
    Gamma = np.array([
        [0.92, 0.06, 0.015, 0.005],  # Dead: hard to reactivate
        [0.12, 0.78, 0.08,  0.02],   # Cold: leaky
        [0.05, 0.10, 0.80,  0.05],   # Warm: balanced
        [0.02, 0.03, 0.08,  0.87]    # Hot: sticky
    ])
    
    # Solve for stationary moments
    delta, pi_ks, mu_ks = solve_stationary_moments(Gamma, inflated_mu, target_pi0, anchors, separation)
    
    # ORDER CONSTRAINT: Enforce mu_Cold < mu_Warm < mu_Hot
    active_mus = [mu_ks[1], mu_ks[2], mu_ks[3]]
    active_pi0s = [pi_ks[1], pi_ks[2], pi_ks[3]]
    order = np.argsort(active_mus)
    s1_mu, s2_mu, s3_mu = [active_mus[i] for i in order]
    s1_pi0, s2_pi0, s3_pi0 = [active_pi0s[i] for i in order]
    mu_ks = [0.0, s1_mu, s2_mu, s3_mu]
    pi_ks = [1.0, s1_pi0, s2_pi0, s3_pi0]
    
    # State-specific CVs
    cv_ks = [0.0, 0.7, target_cv, target_cv * 1.3]
    gamma_shapes = [1.0, 1.0/(cv_ks[1]**2), 1.0/(cv_ks[2]**2), 1.0/(cv_ks[3]**2)]
    gamma_scales = [0.0, mu_ks[1]/gamma_shapes[1], mu_ks[2]/gamma_shapes[2], mu_ks[3]/gamma_shapes[3]]
    
    # Customer-specific activation curves (original logic)
    time_norm = np.linspace(-6, 4, T)
    base_vibe = 1 / (1 + np.exp(-time_norm))
    customer_shifts = np.random.normal(0, customer_heterogeneity, N)
    
    # Initialize matrices
    obs = np.zeros((N, T))
    states = np.zeros((N, T), dtype=int)
    recency = np.zeros((N, T))
    
    # Stationary initialization (t=0)
    states[:, 0] = np.random.choice(4, size=N, p=delta)
    for i in range(N):
        if states[i, 0] == 0:
            recency[i, 0] = 999
        else:
            recency[i, 0] = np.random.randint(0, 10)  # Moderate initial recency
    
    # Simulation loop with burn-in handling
    for t in range(1, T):
        for i in range(N):
            prev_state = states[i, t-1]
            
            # State transition
            if prev_state == 0:
                # Activation from dead with customer-specific timing
                p_activate = base_vibe[t] + customer_shifts[i] * 0.05
                p_activate = np.clip(p_activate, 0.001, 0.5)
                if np.random.rand() < p_activate:
                    states[i, t] = 1  # Activate to Cold
                else:
                    states[i, t] = 0
            else:
                states[i, t] = np.random.choice(4, p=Gamma[prev_state])
            
            # Update recency BEFORE emission
            if states[i, t] == 0:
                recency[i, t] = 999
            elif prev_state == 0 and states[i, t] > 0:
                recency[i, t] = 5  # Fresh activation
            elif obs[i, t-1] > 0:
                recency[i, t] = 0  # Just purchased
            else:
                recency[i, t] = min(recency[i, t-1] + 1, 999)
            
            # Emission
            s = states[i, t]
            if s > 0:
                effective_recency = min(recency[i, t], recency_cap)
                cliff_effect = expit(-(effective_recency - 8) * 0.6)
                prob_spend = (1 - pi_ks[s]) * cliff_effect
                
                if np.random.rand() < prob_spend:
                    obs[i, t] = np.random.gamma(gamma_shapes[s], gamma_scales[s])
    
    # Build RFM covariates
    r_weeks = recency.copy()
    f_run = np.zeros((N, T))
    m_run = np.zeros((N, T))
    
    for i in range(N):
        cum_count = cum_spend = 0.0
        for t in range(T):
            if obs[i, t] > 0:
                cum_count += 1
                cum_spend += obs[i, t]
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
        'world': world,
        'world_desc': preset['desc'],
        'seed': seed,
        'target_pi0': target_pi0,
        'target_mu': inflated_mu / target_inflation,
        'target_cv': target_cv / 0.85,
        'actual_pi0': actual_pi0,
        'actual_mu': actual_mu,
        'actual_cv': actual_cv,
        'separation': separation,
        'customer_heterogeneity': customer_heterogeneity,
        'target_inflation': target_inflation,
        'recency_cap': recency_cap,
        'zero_probs': pi_ks,
        'mu_ks': mu_ks,
        'gamma_shapes': gamma_shapes,
        'gamma_scales': gamma_scales,
        'delta_stationary': delta,
        'Gamma': Gamma,
        'DGP_type': 'Principled_4State_HMM_Hybrid'
    }
    
    return df, meta

# =============================================================================
# 4. OUTPUT FUNCTIONS
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
# 5. DIAGNOSTIC VISUALIZATIONS
# =============================================================================
def run_diagnostics(df: pd.DataFrame, meta: Dict, output_dir: str):
    """Generate diagnostic plots including ground truth recovery."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    world = meta.get('world', 'unknown')
    
    # 1. State distribution over time
    plt.figure(figsize=(10, 5))
    state_props = df.groupby('time_period')['true_state'].value_counts(normalize=True).unstack(fill_value=0)
    state_props.plot(kind='area', stacked=True, alpha=0.7, 
                     color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    plt.title(f"State Distribution Over Time ({world})")
    plt.xlabel("Time Period")
    plt.ylabel("Proportion")
    plt.legend(['Dead', 'Cold', 'Warm', 'Hot'], loc='upper right')
    plt.savefig(output_dir / "state_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Recency Cliff
    plt.figure(figsize=(8, 5))
    df_active = df[df['true_state'] > 0].copy()
    df_active['is_spend'] = (df_active['y'] > 0).astype(int)
    recency_cliff = df_active.groupby('recency')['is_spend'].mean().reset_index()
    recency_cliff = recency_cliff[recency_cliff['recency'] <= 20]
    sns.lineplot(data=recency_cliff, x='recency', y='is_spend', marker='o')
    plt.title(f"Recency Cliff ({world})")
    plt.ylabel("P(Spend)")
    plt.xlabel("Weeks Since Last Purchase")
    plt.savefig(output_dir / "recency_cliff.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. State-wise spend distribution
    plt.figure(figsize=(10, 6))
    df_spend = df[df['y'] > 0]
    if len(df_spend) > 0:
        state_labels = {0: 'Dead', 1: 'Cold', 2: 'Warm', 3: 'Hot'}
        df_spend['state_label'] = df_spend['true_state'].map(state_labels)
        sns.boxplot(data=df_spend, x='state_label', y='y', 
                   order=['Cold', 'Warm', 'Hot'], palette="Set2")
        plt.yscale('log')
        plt.title(f"Spend Distribution by State ({world})")
        plt.ylabel("Spend ($, log scale)")
        plt.savefig(output_dir / "state_spend_dist.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Transition matrix heatmap
    plt.figure(figsize=(7, 6))
    Gamma = meta['Gamma']
    sns.heatmap(Gamma, annot=True, fmt='.3f', cmap="YlOrRd", vmin=0, vmax=1,
                xticklabels=['Dead', 'Cold', 'Warm', 'Hot'],
                yticklabels=['Dead', 'Cold', 'Warm', 'Hot'],
                cbar_kws={'label': 'Transition Probability'})
    plt.title(f"True Transition Matrix ({world})")
    plt.tight_layout()
    plt.savefig(output_dir / "true_transitions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostics saved to: {output_dir}")

# =============================================================================
# 6. MAIN & CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Principled synthetic RFM panel generator (4-world taxonomy)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--world", type=str, default='clumpy', 
                       choices=list(WORLD_PRESETS.keys()),
                       help="Simulation world (2x2 taxonomy cell)")
    parser.add_argument("--n_customers", type=int, default=300, 
                       help="Number of customers")
    parser.add_argument("--n_periods", type=int, default=100, 
                       help="Number of time periods")
    parser.add_argument("--separation", type=float, default=0.8, 
                       help="State separation factor (0-1)")
    parser.add_argument("--customer_het", type=float, default=2.0, 
                       help="Customer heterogeneity in activation")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output file prefix (default: world_name)")
    parser.add_argument("--format", type=str, default="both", 
                       choices=["csv", "pkl", "both"], 
                       help="Output format")
    parser.add_argument("--no_state", action="store_true", 
                       help="Exclude true_state from CSV")
    parser.add_argument("--plot", action="store_true", 
                       help="Generate diagnostic plots")
    parser.add_argument("--plot_dir", type=str, default=None, 
                       help="Plot output directory")
    
    args = parser.parse_args()
    
    # Set default output name
    if args.output is None:
        args.output = f"sim_{args.world}_N{args.n_customers}_T{args.n_periods}_seed{args.seed}"
    
    print("="*70)
    print("PRINCIPLED SIMULATION GENERATOR")
    print("="*70)
    print(f"World: {args.world} — {WORLD_PRESETS[args.world]['desc']}")
    print(f"Parameters: N={args.n_customers}, T={args.n_periods}, seed={args.seed}")
    
    preset = WORLD_PRESETS[args.world]
    print(f"Targets: π₀={preset['target_pi0']:.1%}, μ=${preset['target_mu']:.1f}, CV={preset['target_cv']:.1f}")
    
    # Generate
    df, meta = generate_principled_panel(
        n_customers=args.n_customers,
        n_periods=args.n_periods,
        world=args.world,
        separation=args.separation,
        seed=args.seed,
        customer_heterogeneity=args.customer_het
    )
    
    print("\n" + "-"*70)
    print("ACTUAL MOMENTS")
    print("-"*70)
    print(f"Zero rate: {meta['actual_pi0']:.2%} (target: {preset['target_pi0']:.1%})")
    print(f"Mean spend: ${meta['actual_mu']:.2f} (target: ${preset['target_mu']:.1f})")
    print(f"CV: {meta['actual_cv']:.2f} (target: {preset['target_cv']:.1f})")
    print(f"Stationary: Dead={meta['delta_stationary'][0]:.3f}, "
          f"Cold={meta['delta_stationary'][1]:.3f}, "
          f"Warm={meta['delta_stationary'][2]:.3f}, "
          f"Hot={meta['delta_stationary'][3]:.3f}")
    
    # Save
    print("\n" + "-"*70)
    print("SAVING")
    print("-"*70)
    
    if args.format in ["csv", "both"]:
        csv_path = f"{args.output}.csv"
        save_simulation_csv(df, csv_path, include_state=not args.no_state)
    
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
    print(f"Run harness with:")
    print(f"  ./launch_harness.sh ./{args.output}.csv ./runs_{args.world}/")

if __name__ == "__main__":
    main()
