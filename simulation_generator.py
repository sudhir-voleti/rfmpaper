import numpy as np
import pandas as pd
import scipy.linalg
from scipy.special import expit
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# 1. ANALYTICAL SOLVER
# =============================================================================

def solve_stationary_moments(Gamma, target_mu, target_pi0, anchors, separation_factor=0.8):
    evals, evecs = scipy.linalg.eig(Gamma.T)
    delta = evecs[:, np.isclose(evals, 1.0)].real
    delta = (delta / delta.sum()).flatten()
    
    s1_pi0_base, s1_mu_base = anchors['s1']
    s3_pi0_base, s3_mu_base = anchors['s3']
    
    s1_mu = target_mu + (s1_mu_base - target_mu) * separation_factor
    s3_mu = target_mu + (s3_mu_base - target_mu) * separation_factor
    s1_pi0 = target_pi0 + (s1_pi0_base - target_pi0) * separation_factor
    s3_pi0 = target_pi0 + (s3_pi0_base - target_pi0) * separation_factor

    s2_pi0 = (target_pi0 - delta[0] - delta[1]*s1_pi0 - delta[3]*s3_pi0) / delta[2]
    s2_pi0 = np.clip(s2_pi0, 0.05, 0.95)
    
    target_sum_mu = target_mu * (1 - target_pi0)
    term1 = delta[1] * (1 - s1_pi0) * s1_mu
    term3 = delta[3] * (1 - s3_pi0) * s3_mu
    s2_mu = (target_sum_mu - term1 - term3) / (delta[2] * (1 - s2_pi0))
    s2_mu = max(s2_mu, 2.0)
    
    return delta, [s1_pi0, s2_pi0, s3_pi0], [s1_mu, s2_mu, s3_mu]

# =============================================================================
# 2. DATA GENERATOR
# =============================================================================

def generate_principled_panel(n_customers=500, n_periods=100, target_pi0=0.75, 
                              target_mu=45.0, target_cv=2.0, separation=0.8, seed=42):
    np.random.seed(seed)
    N, T = n_customers, n_periods
    
    Gamma_base = np.array([
        [0.90, 0.07, 0.02, 0.01], 
        [0.10, 0.80, 0.08, 0.02], 
        [0.05, 0.10, 0.80, 0.05], 
        [0.02, 0.03, 0.10, 0.85]  
    ])
    
    anchors = {'s1': (0.95, 5.0), 's3': (0.15, 150.0)}
    delta, pi_ks, mu_ks = solve_stationary_moments(Gamma_base, target_mu, target_pi0, anchors, separation)
    
    cv_ks = [1.0, 1.5, target_cv * 1.5] 
    gamma_shapes = [1.0 / (cv**2) for cv in cv_ks]
    gamma_scales = [mu / shape for mu, shape in zip(mu_ks, gamma_shapes)]

    obs = np.zeros((N, T))
    states = np.zeros((N, T), dtype=int)
    recency = np.zeros((N, T))
    segments = np.random.choice(['Loyalist', 'Switcher'], size=N, p=[0.7, 0.3])
    states[:, 0] = np.random.choice(4, size=N, p=delta)

    for t in range(T):
        for i in range(N):
            s = states[i, t]
            if s > 0: 
                cliff_effect = expit(-(recency[i, t] - 10) * 0.6)
                prob_spend = (1 - pi_ks[s-1]) * cliff_effect
                if np.random.rand() < prob_spend:
                    obs[i, t] = np.random.gamma(gamma_shapes[s-1], gamma_scales[s-1])
            
            if t < T-1:
                recency[i, t+1] = 0 if obs[i, t] > 0 else recency[i, t] + 1
                G = Gamma_base.copy()
                if segments[i] == 'Switcher':
                    G = G * 0.5 + 0.5 * delta
                    G /= G.sum(axis=1, keepdims=True)
                states[i, t+1] = np.random.choice(4, p=G[states[i, t]])

    data_list = []
    for i in range(N):
        for t in range(T):
            data_list.append({
                'customer_id': i, 'segment': segments[i], 'time_period': t,
                'recency': recency[i, t], 'state': states[i, t], 'y': obs[i, t]
            })
    
    df = pd.DataFrame(data_list)
    meta = {'target_pi0': target_pi0, 'actual_pi0': np.mean(obs == 0),
            'target_mu': target_mu, 'actual_mu': np.mean(obs[obs > 0]),
            'delta': delta, 'pi_ks': pi_ks, 'mu_ks': mu_ks, 'Gamma_base': Gamma_base}
    return df, meta

# =============================================================================
# 3. DIAGNOSTIC VISUALIZATION
# =============================================================================

def run_diagnostics(df, meta, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. The Recency Cliff Plot
    plt.figure(figsize=(8, 5))
    df_active = df[df['state'] > 0].copy()
    df_active['is_spend'] = (df_active['y'] > 0).astype(int)
    sns.lineplot(data=df_active, x='recency', y='is_spend', err_style="bars")
    plt.title("The Recency Cliff (Behavioral Non-linearity)")
    plt.ylabel("P(Spend)")
    plt.savefig(output_dir / "diagnostic_recency_cliff.png")

    # 2. State-Wise Intensity Distribution
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df[df['y'] > 0], x='state', y='y', palette="Set2")
    plt.yscale('log')
    plt.title("State-Wise Spending Intensity (Log Scale)")
    plt.savefig(output_dir / "diagnostic_state_intensity.png")

    # 3. Transition Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(meta['Gamma_base'], annot=True, cmap="YlGnBu", 
                xticklabels=['Dead', 'Dormant', 'Core', 'Whale'],
                yticklabels=['Dead', 'Dormant', 'Core', 'Whale'])
    plt.title("Ground Truth Transition Matrix")
    plt.savefig(output_dir / "diagnostic_transitions.png")
    
    plt.close('all')
    print(f"Diagnostic plots saved to: {output_dir}")

# =============================================================================
# 4. MAIN & CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Principled HMM-Hurdle-Gamma Data Generator")
    parser.add_argument("--n_customers", type=int, default=500)
    parser.add_argument("--n_periods", type=int, default=100)
    parser.add_argument("--target_pi0", type=float, default=0.75, help="Agg zero-incidence")
    parser.add_argument("--target_mu", type=float, default=45.0, help="Agg mean spend")
    parser.add_argument("--target_cv", type=float, default=2.0, help="Agg volatility")
    parser.add_argument("--separation", type=float, default=0.8, help="State separation index (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="sim_data", help="Base name for output")
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots")

    args = parser.parse_args()

    df, meta = generate_principled_panel(
        n_customers=args.n_customers, n_periods=args.n_periods,
        target_pi0=args.target_pi0, target_mu=args.target_mu,
        target_cv=args.target_cv, separation=args.separation, seed=args.seed
    )

    # Save outputs
    df.to_csv(f"{args.output}.csv", index=False)
    with open(f"{args.output}.pkl", "wb") as f:
        pickle.dump({'df': df, 'meta': meta}, f)
    
    print(f"\n--- Generation Complete ---")
    print(f"Actual Zero-Rate: {meta['actual_pi0']:.2%}")
    print(f"Actual Mean Spend: {meta['actual_mu']:.2f}")

    if args.plot:
        run_diagnostics(df, meta, f"plots_{args.output}")

if __name__ == "__main__":
    main()
