import numpy as np
import pandas as pd
import scipy.linalg
from scipy.special import expit
import pickle
from pathlib import Path

def solve_stationary_moments(Gamma, target_mu, target_pi0, anchors, separation_factor=0.8):
    """
    Ensures HMM steady-state hits aggregate retail targets by solving 
    for the 'Core' state parameters.
    """
    # 1. Compute stationary distribution (delta)
    evals, evecs = scipy.linalg.eig(Gamma.T)
    delta = evecs[:, np.isclose(evals, 1.0)].real
    delta = (delta / delta.sum()).flatten()
    
    # 2. Adjust anchors based on Separation Factor
    # Pulls anchors toward the target_mu as separation decreases
    s1_pi0_base, s1_mu_base = anchors['s1']
    s3_pi0_base, s3_mu_base = anchors['s3']
    
    s1_mu = target_mu + (s1_mu_base - target_mu) * separation_factor
    s3_mu = target_mu + (s3_mu_base - target_mu) * separation_factor
    s1_pi0 = target_pi0 + (s1_pi0_base - target_pi0) * separation_factor
    s3_pi0 = target_pi0 + (s3_pi0_base - target_pi0) * separation_factor

    # 3. Solve for State 2 (Core) parameters
    # pi0_agg = delta[0]*1.0 (Dead) + sum(delta[k]*pi0_k)
    s2_pi0 = (target_pi0 - delta[0] - delta[1]*s1_pi0 - delta[3]*s3_pi0) / delta[2]
    s2_pi0 = np.clip(s2_pi0, 0.05, 0.95)
    
    # mu_agg = sum(delta_k * (1-pi0_k) * mu_k) / (1-target_pi0)
    target_sum_mu = target_mu * (1 - target_pi0)
    term1 = delta[1] * (1 - s1_pi0) * s1_mu
    term3 = delta[3] * (1 - s3_pi0) * s3_mu
    s2_mu = (target_sum_mu - term1 - term3) / (delta[2] * (1 - s2_pi0))
    s2_mu = max(s2_mu, 2.0)
    
    return delta, [s1_pi0, s2_pi0, s3_pi0], [s1_mu, s2_mu, s3_mu]

def generate_principled_panel(n_customers=500, n_periods=100, target_pi0=0.75, 
                              target_mu=45.0, target_cv=2.0, separation=0.8, seed=42):
    """
    Redesigned DGP: Non-parametric Hurdle-Gamma HMM with Recency Cliff.
    """
    np.random.seed(seed)
    N, T = n_customers, n_periods
    
    # 1. SEGMENT-BASED TRANSITION DYNAMICS
    # We create 2 segments: 'Loyalists' (sticky) and 'Switchers' (volatile)
    Gamma_base = np.array([
        [0.90, 0.07, 0.02, 0.01], # Dead
        [0.10, 0.80, 0.08, 0.02], # Dormant
        [0.05, 0.10, 0.80, 0.05], # Core
        [0.02, 0.03, 0.10, 0.85]  # Whale
    ])
    
    # 2. SOLVE MOMENTS
    anchors = {'s1': (0.95, 5.0), 's3': (0.15, 150.0)}
    delta, pi_ks, mu_ks = solve_stationary_moments(Gamma_base, target_mu, target_pi0, anchors, separation)
    
    # Emission Variability (Target CV matching)
    cv_ks = [1.0, 1.5, target_cv * 1.5] 
    gamma_shapes = [1.0 / (cv**2) for cv in cv_ks]
    gamma_scales = [mu / shape for mu, shape in zip(mu_ks, gamma_shapes)]

    # 3. INITIALIZE PANEL
    obs = np.zeros((N, T))
    states = np.zeros((N, T), dtype=int)
    recency = np.zeros((N, T))
    
    # Static Segment Assignment
    segments = np.random.choice(['Loyalist', 'Switcher'], size=N, p=[0.7, 0.3])
    
    # Starting states based on stationary distribution
    states[:, 0] = np.random.choice(4, size=N, p=delta)

    # 4. SIMULATION LOOP
    for t in range(T):
        for i in range(N):
            s = states[i, t]
            if s > 0: # If not 'Dead'
                # RECENCY CLIFF: Propensity to spend drops after 10 periods of silence
                # This is the non-linear challenge for the GAM models
                cliff_effect = expit(-(recency[i, t] - 10) * 0.6)
                prob_spend = (1 - pi_ks[s-1]) * cliff_effect
                
                if np.random.rand() < prob_spend:
                    obs[i, t] = np.random.gamma(gamma_shapes[s-1], gamma_scales[s-1])
            
            # Update Recency for t+1
            if t < T-1:
                recency[i, t+1] = 0 if obs[i, t] > 0 else recency[i, t] + 1
                
                # Transition (Segment-dependent)
                G = Gamma_base.copy()
                if segments[i] == 'Switcher':
                    # Switchers have less 'stickiness' (lower diagonal)
                    diag = np.diag(G)
                    G = G * 0.5 + 0.5 * delta # Pull toward average
                    np.fill_diagonal(G, diag * 0.7)
                    G /= G.sum(axis=1, keepdims=True)
                
                states[i, t+1] = np.random.choice(4, p=G[states[i, t]])

    # 5. PACKAGING
    data_list = []
    for i in range(N):
        for t in range(T):
            data_list.append({
                'customer_id': i,
                'segment': segments[i],
                'time_period': t,
                'recency': recency[i, t],
                'state': states[i, t],
                'y': obs[i, t]
            })
    
    df = pd.DataFrame(data_list)
    
    meta = {
        'target_pi0': target_pi0,
        'actual_pi0': np.mean(obs == 0),
        'target_mu': target_mu,
        'actual_mu': np.mean(obs[obs > 0]),
        'delta': delta,
        'pi_ks': pi_ks,
        'mu_ks': mu_ks,
        'gamma_params': list(zip(gamma_shapes, gamma_scales)),
        'Gamma_base': Gamma_base
    }
    
    return df, meta

def save_simulation(df, meta, filename="simulation_data.csv"):
    df.to_csv(filename, index=False)
    with open(filename.replace(".csv", ".pkl"), "wb") as f:
        pickle.dump({'df': df, 'meta': meta}, f)
    print(f"Saved: {filename} and associated .pkl")

if __name__ == "__main__":
    # Example: Generate 'Clumpy' World Data
    df, meta = generate_principled_panel(
        target_pi0=0.85, 
        target_mu=45.0, 
        target_cv=3.5, 
        separation=0.8
    )
    save_simulation(df, meta, "sim_clumpy_v3.csv")
