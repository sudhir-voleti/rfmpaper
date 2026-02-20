#!/usr/bin/env python3
"""
simulation_generator.py
=======================
Generate synthetic RFM panel with selectable DGP:
- mixture: Unimodal high zero-inflation
- bimodal: Two-mode (low + whales)
- 4mode: Extreme four-mode (Dead, Low, Med, High, Super)

Usage:
    python simulation_generator.py --dgp mixture --n_customers 50 --n_periods 60
    python simulation_generator.py --dgp bimodal --n_customers 100 --n_periods 60
    python simulation_generator.py --dgp 4mode --n_customers 200 --n_periods 80
"""

import numpy as np
import pandas as pd
import pickle
import csv
import argparse
from datetime import datetime, timedelta


# =============================================================================
# OBSERVATION GENERATORS
# =============================================================================

def generate_mixed_observations(n, zero_prob=0.70, gamma_shape=2.0, gamma_scale=4.0):
    """MIXTURE: zero_prob zeros + Gamma(gamma_shape, gamma_scale)"""
    is_zero = np.random.rand(n) < zero_prob
    result = np.zeros(n)
    n_positive = (~is_zero).sum()
    if n_positive > 0:
        result[~is_zero] = np.random.gamma(gamma_shape, gamma_scale, size=n_positive)
    return result


def generate_bimodal_observations(n, zero_prob=0.60):
    """BIMODAL: 60% zeros + 50% Gamma(1,1) + 50% Gamma(10,2)"""
    is_zero = np.random.rand(n) < zero_prob
    result = np.zeros(n)
    n_positive = (~is_zero).sum()
    
    if n_positive > 0:
        is_whale = np.random.rand(n_positive) < 0.5
        result_pos = np.zeros(n_positive)
        result_pos[is_whale] = np.random.gamma(10, 2, is_whale.sum())
        result_pos[~is_whale] = np.random.gamma(1, 1, (~is_whale).sum())
        result[~is_zero] = result_pos
    
    return result


def generate_4mode_observations(n, state, state_zero_probs=None):
    """4-MODE: State-dependent Gamma emissions WITH zero-inflation"""
    if state_zero_probs is None:
        # Default: high zeros for low states, lower for high states
        state_zero_probs = {0: 1.0, 1: 0.80, 2: 0.50, 3: 0.30, 4: 0.20}
    
    result = np.zeros(n)
    
    for s in [0, 1, 2, 3, 4]:
        mask = state == s
        if mask.sum() > 0:
            if s == 0:
                # Dead state: all zeros
                result[mask] = 0.0
            else:
                # Active states: zero-inflated Gamma
                n_s = mask.sum()
                is_zero = np.random.rand(n_s) < state_zero_probs[s]
                result_s = np.zeros(n_s)
                
                n_pos = (~is_zero).sum()
                if n_pos > 0:
                    if s == 1:
                        result_s[~is_zero] = np.random.gamma(1, 0.5, n_pos)
                    elif s == 2:
                        result_s[~is_zero] = np.random.gamma(5, 1, n_pos)
                    elif s == 3:
                        result_s[~is_zero] = np.random.gamma(10, 2, n_pos)
                    elif s == 4:
                        result_s[~is_zero] = np.random.gamma(15, 3, n_pos)
                
                result[mask] = result_s
    
    return result

# =============================================================================
# PANEL GENERATORS
# =============================================================================

def generate_mixture_panel(n_customers=500, n_periods=100, seed=42, **kwargs):
    """Standard mixture DGP with recency-dependent churn."""
    np.random.seed(seed)
    N, T = n_customers, n_periods
    
    zero_prob = kwargs.get('zero_prob', 0.70)
    gamma_shape = kwargs.get('gamma_shape', 2.0)
    gamma_scale = kwargs.get('gamma_scale', 4.0)
    burn_in = kwargs.get('burn_in', 5)
    
    # Activation curve
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
        
        # Generate observations
        active = (states[:, t] == 1)
        if active.sum() > 0:
            obs[active, t] = generate_mixed_observations(
                active.sum(), zero_prob, gamma_shape, gamma_scale
            )
            positive = obs[:, t] > 0
            last_purchase[positive] = t

    true_params = {
        'DGP_type': 'Mixture',
        'zero_prob': zero_prob,
        'gamma_shape': gamma_shape,
        'gamma_scale': gamma_scale
    }
    
    return _package_results(N, T, states, obs, true_params, seed)


def generate_bimodal_panel(n_customers=100, n_periods=60, seed=42, **kwargs):
    """Bimodal DGP: low spenders vs whales."""
    np.random.seed(seed)
    N, T = n_customers, n_periods
    
    zero_prob = kwargs.get('zero_prob', 0.60)
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
        
        active = (states[:, t] == 1)
        if active.sum() > 0:
            obs[active, t] = generate_bimodal_observations(active.sum(), zero_prob)
            positive = obs[:, t] > 0
            last_purchase[positive] = t

    true_params = {
        'DGP_type': 'Bimodal',
        'zero_prob': zero_prob,
        'low_dist': 'Gamma(1, 1)',
        'whale_dist': 'Gamma(10, 2)'
    }
    
    return _package_results(N, T, states, obs, true_params, seed)

## ----

def generate_3mode_panel(n_customers=300, n_periods=100, seed=42, **kwargs):
    """3-MODE DGP: Dead + Light + Regular + Heavy (merged high states)."""
    np.random.seed(seed)
    N, T = n_customers, n_periods
    
    # Start spread: 25% each state
    states = np.zeros((N, T), dtype=int)
    obs = np.zeros((N, T))
    tenure = np.zeros(N, dtype=int)
    
    initial_dist = [0.25, 0.25, 0.25, 0.25]
    states[:, 0] = np.random.choice([0, 1, 2, 3], size=N, p=initial_dist)
    
    # Gentler activation
    time_norm = np.linspace(-1, 3, T)
    p_activate = 0.15 + 0.25 / (1 + np.exp(-time_norm))
    
    for t in range(1, T):
        for i in range(N):
            current = states[i, t-1]
            
            if current == 0:
                if np.random.rand() < p_activate[t]:
                    states[i, t] = 1
                    tenure[i] = 0
                else:
                    states[i, t] = 0
                    tenure[i] += 1
            elif current == 1:
                # Upgrade to Regular
                if tenure[i] > 4 and np.random.rand() < 0.50:
                    states[i, t] = 2
                    tenure[i] = 0
                elif np.random.rand() < 0.08:
                    states[i, t] = 0
                    tenure[i] = 0
                else:
                    states[i, t] = 1
                    tenure[i] += 1
            elif current == 2:
                # Upgrade to Heavy
                if tenure[i] > 6 and np.random.rand() < 0.40:
                    states[i, t] = 3
                    tenure[i] = 0
                elif np.random.rand() < 0.06:
                    states[i, t] = 0
                    tenure[i] = 0
                else:
                    states[i, t] = 2
                    tenure[i] += 1
            elif current == 3:
                # Sticky Heavy
                if np.random.rand() < 0.04:
                    states[i, t] = 0
                    tenure[i] = 0
                else:
                    states[i, t] = 3
                    tenure[i] += 1
        
        obs[:, t] = generate_3mode_observations(N, states[:, t])
        churned = (states[:, t] == 0) & (states[:, t-1] != 0)
        tenure[churned] = 0

    true_params = {
        'DGP_type': '3Mode',
        'state_zero_probs': {0: 1.0, 1: 0.80, 2: 0.50, 3: 0.30},
        'state_gamma_params': {1: (1, 1), 2: (4, 2), 3: (10, 3)},
        'transitions': 'tenure_based_3state'
    }
    
    return _package_results(N, T, states, obs, true_params, seed)


def generate_3mode_observations(n, state, state_zero_probs=None):
    """3-MODE: State-dependent Gamma emissions WITH zero-inflation."""
    if state_zero_probs is None:
        state_zero_probs = {0: 1.0, 1: 0.80, 2: 0.50, 3: 0.30}
    
    result = np.zeros(n)
    
    for s in [0, 1, 2, 3]:
        mask = state == s
        if mask.sum() > 0:
            if s == 0:
                result[mask] = 0.0
            else:
                n_s = mask.sum()
                is_zero = np.random.rand(n_s) < state_zero_probs[s]
                result_s = np.zeros(n_s)
                
                n_pos = (~is_zero).sum()
                if n_pos > 0:
                    if s == 1:
                        result_s[~is_zero] = np.random.gamma(1, 1, n_pos)
                    elif s == 2:
                        result_s[~is_zero] = np.random.gamma(4, 2, n_pos)
                    elif s == 3:
                        result_s[~is_zero] = np.random.gamma(10, 3, n_pos)
                
                result[mask] = result_s
    
    return result

## ------

def generate_4mode_panel(n_customers=200, n_periods=80, seed=42, **kwargs):
    """Extreme 4-mode DGP with tenure-based upgrades."""
    np.random.seed(seed)
    N, T = n_customers, n_periods
    
    # Activation curve
    time_norm = np.linspace(-4, 4, T)
    p_activate = 0.05 + 0.4 / (1 + np.exp(-time_norm + 2))
    
    states = np.zeros((N, T), dtype=int)
    obs = np.zeros((N, T))
    tenure = np.zeros(N, dtype=int)
    
    for t in range(T):
        for i in range(N):
            current = states[i, t-1] if t > 0 else 0
            
            if current == 0:
                if np.random.rand() < p_activate[t]:
                    states[i, t] = 1
                    tenure[i] = 0
                else:
                    states[i, t] = 0
                    tenure[i] += 1
            elif current == 1:
                if tenure[i] > 3 and np.random.rand() < 0.3:
                    states[i, t] = 2
                    tenure[i] = 0
                elif np.random.rand() < 0.15:
                    states[i, t] = 0
                    tenure[i] = 0
                else:
                    states[i, t] = 1
                    tenure[i] += 1
            elif current == 2:
                if tenure[i] > 5 and np.random.rand() < 0.25:
                    states[i, t] = 3
                    tenure[i] = 0
                elif np.random.rand() < 0.20:
                    states[i, t] = 0
                    tenure[i] = 0
                else:
                    states[i, t] = 2
                    tenure[i] += 1
            elif current == 3:
                if tenure[i] > 7 and np.random.rand() < 0.20:
                    states[i, t] = 4
                    tenure
                else:
                    states[i, t] = 3
                    tenure[i] += 1
            elif current == 4:
                if np.random.rand() < 0.10:
                    states[i, t] = 0
                    tenure[i] = 0
                else:
                    states[i, t] = 4
                    tenure[i] += 1
        
        # Generate observations based on state
        obs[:, t] = generate_4mode_observations(N, states[:, t], 
        state_zero_probs={0: 1.0, 1: 0.85, 2: 0.60, 3: 0.40, 4: 0.25})
       
        # Reset tenure for those who churned
        churned = (states[:, t] == 0) & (states[:, t-1] != 0) if t > 0 else np.zeros(N, dtype=bool)
        tenure[churned] = 0

        true_params = {
            'DGP_type': '4Mode',
            'state_zero_probs': {0: 1.0, 1: 0.85, 2: 0.60, 3: 0.40, 4: 0.25},
            'state_gamma_params': {1: (1, 0.5), 2: (5, 1), 3: (10, 2), 4: (15, 3)},
            'transitions': 'tenure-based upgrades'
        }    
    return _package_results(N, T, states, obs, true_params, seed)

def generate_4mode_panel_v2(n_customers=200, n_periods=80, seed=42, **kwargs):
    """4-MODE DGP v2: Start spread out, stay in high states longer."""
    np.random.seed(seed)
    N, T = n_customers, n_periods
    
    # Start with spread initialization, not all in state 0
    states = np.zeros((N, T), dtype=int)
    obs = np.zeros((N, T))
    tenure = np.zeros(N, dtype=int)
    
    # Initial distribution: 20% each in states 1-4, 20% in state 0
    initial_dist = [0.2, 0.2, 0.2, 0.2, 0.2]
    states[:, 0] = np.random.choice([0, 1, 2, 3, 4], size=N, p=initial_dist)
    
    # Activation curve (less aggressive, more stable)
    time_norm = np.linspace(-2, 2, T)
    p_activate = 0.1 + 0.2 / (1 + np.exp(-time_norm))
    
    for t in range(1, T):
        for i in range(N):
            current = states[i, t-1]
            
            if current == 0:
                # Reactivation
                if np.random.rand() < p_activate[t]:
                    states[i, t] = 1
                    tenure[i] = 0
                else:
                    states[i, t] = 0
                    tenure[i] += 1
            elif current == 1:
                # Longer tenure before upgrade, less churn
                if tenure[i] > 5 and np.random.rand() < 0.4:  # Was 3, 0.3
                    states[i, t] = 2
                    tenure[i] = 0
                elif np.random.rand() < 0.08:  # Was 0.15
                    states[i, t] = 0
                    tenure[i] = 0
                else:
                    states[i, t] = 1
                    tenure[i] += 1
            elif current == 2:
                if tenure[i] > 7 and np.random.rand() < 0.35:  # Was 5, 0.25
                    states[i, t] = 3
                    tenure[i] = 0
                elif np.random.rand() < 0.10:  # Was 0.20
                    states[i, t] = 0
                    tenure[i] = 0
                else:
                    states[i, t] = 2
                    tenure[i] += 1
            elif current == 3:
                # STICKY high state - hard to leave
                if tenure[i] > 10 and np.random.rand() < 0.30:  # Was 7, 0.20
                    states[i, t] = 4
                    tenure[i] = 0
                elif np.random.rand() < 0.05:  # Was higher
                    states[i, t] = 0
                    tenure[i] = 0
                else:
                    states[i, t] = 3
                    tenure[i] += 1
            elif current == 4:
                # SUPER STICKY - whales stay whales
                if np.random.rand() < 0.03:  # Very rare churn
                    states[i, t] = 0
                    tenure[i] = 0
                else:
                    states[i, t] = 4
                    tenure[i] += 1
        
        obs[:, t] = generate_4mode_observations(N, states[:, t])
        churned = (states[:, t] == 0) & (states[:, t-1] != 0)
        tenure[churned] = 0

    true_params = {
        'DGP_type': '4Mode_v2',
        'state_zero_probs': {0: 1.0, 1: 0.85, 2: 0.60, 3: 0.40, 4: 0.25},
        'state_gamma_params': {1: (1, 0.5), 2: (5, 1), 3: (10, 2), 4: (15, 3)},
        'transitions': 'sticky_high_states'
    }
    
    return _package_results(N, T, states, obs, true_params, seed)

## ---

def generate_moment_controlled_panel(n_customers=500, n_periods=100,
                                     target_pi0=0.75, target_mu=45.0, target_cv=2.0,
                                     n_states=3, seed=42, **kwargs):
    """
    3-State Gamma Mixture HMM with exact marginal moment matching.
    
    Uses stationary distribution + linear system solver for state parameters,
    then Gamma parameters to match target CV.
    """
    import scipy.linalg
    np.random.seed(seed)
    N, T = n_customers, n_periods
    
    # --- 1. TRANSITION MATRIX (with Dead state as index 0) ---
    # Sticky diagonal, proportional off-diagonals
    Gamma = np.eye(n_states + 1) * 0.85
    Gamma[0, 0] = 0.90  # Dead state very sticky
    
    # Off-diagonal: proportional to active time
    off_diag = 0.15 / n_states
    Gamma += (np.ones((n_states+1, n_states+1)) - np.eye(n_states+1)) * off_diag
    Gamma /= Gamma.sum(axis=1, keepdims=True)
    
    # --- 2. STATIONARY DISTRIBUTION ---
    # Solve delta * Gamma = delta
    evals, evecs = scipy.linalg.eig(Gamma.T)
    delta = evecs[:, np.isclose(evals, 1.0)].real
    delta = (delta / delta.sum()).flatten()
    
    prob_dead = delta[0]
    delta_active = delta[1:] / delta[1:].sum()  # Relative weights among active
    
    # --- 3. SOLVE STATE PARAMETERS FOR TARGET MOMENTS ---
    # Anchors: State 1 (Dormant), State 3 (Whale) fixed
    # State 2 (Core) solved to hit targets
    
    anchors = {
        's1': (0.90, 2.0),    # Dormant: 90% zeros, $2 mean
        's3': (0.20, 120.0)   # Whale: 20% zeros, $120 mean
    }
    
    s1_pi0, s1_mu = anchors['s1']
    s3_pi0, s3_mu = anchors['s3']
    
    # Solve for State 2 (Core) zero-inflation
    # target_pi0 = delta[0]*1.0 + delta[1]*s1_pi0 + delta[2]*s2_pi0 + delta[3]*s3_pi0
    s2_pi0 = (target_pi0 - prob_dead - delta[1]*s1_pi0 - delta[3]*s3_pi0) / delta[2]
    s2_pi0 = np.clip(s2_pi0, 0.05, 0.95)
    
    # Solve for State 2 (Core) mean
    # target_mu = weighted average of active state means
    numerator = target_mu * (1 - target_pi0)
    term1 = delta[1] * (1 - s1_pi0) * s1_mu
    term3 = delta[3] * (1 - s3_pi0) * s3_mu
    
    s2_mu = (numerator - term1 - term3) / (delta[2] * (1 - s2_pi0))
    s2_mu = max(s2_mu, 1.0)
    
    # Collect state parameters
    pi_states = [s1_pi0, s2_pi0, s3_pi0]
    mu_states = [s1_mu, s2_mu, s3_mu]
    
    # --- 4. SOLVE GAMMA PARAMETERS FOR TARGET CV ---
    # Target variance for active observations
    target_var_active = (target_cv * target_mu) ** 2
    
    # Distribute variance across states proportional to their contribution to mean
    contributions = [delta[i+1] * (1 - pi_states[i]) * mu_states[i] 
                     for i in range(n_states)]
    total_contrib = sum(contributions)
    
    state_vars = [target_var_active * (c / total_contrib) for c in contributions]
    
    # Gamma: shape = mu^2 / var, scale = var / mu
    gamma_shapes = [mu**2 / max(var, 0.1) for mu, var in zip(mu_states, state_vars)]
    gamma_scales = [var / mu for mu, var in zip(mu_states, state_vars)]
    
    # --- 5. GENERATE PANEL ---
    states = np.zeros((N, T), dtype=int)
    obs = np.zeros((N, T))
    
    # Initialize from stationary distribution
    states[:, 0] = np.random.choice(n_states + 1, size=N, p=delta)
    
    for t in range(T):
        for i in range(N):
            s = states[i, t]
            
            if s == 0:
                # Dead state
                obs[i, t] = 0.0
            else:
                # Active state (index 0 in params = state 1)
                idx = s - 1
                if np.random.rand() < pi_states[idx]:
                    obs[i, t] = 0.0
                else:
                    obs[i, t] = np.random.gamma(gamma_shapes[idx], gamma_scales[idx])
        
        # Transition
        if t < T - 1:
            for i in range(N):
                states[i, t+1] = np.random.choice(n_states + 1, 
                                                   p=Gamma[states[i, t], :])
    
    # --- 6. VALIDATION ---
    actual_pi0 = np.mean(obs == 0)
    active_obs = obs[obs > 0]
    actual_mu = np.mean(active_obs) if len(active_obs) > 0 else 0
    actual_cv = np.std(active_obs) / actual_mu if actual_mu > 0 else 0
    
    true_params = {
        'DGP_type': 'MomentControlled',
        'target_pi0': target_pi0,
        'target_mu': target_mu,
        'target_cv': target_cv,
        'actual_pi0': actual_pi0,
        'actual_mu': actual_mu,
        'actual_cv': actual_cv,
        'delta_stationary': delta.tolist(),
        'Gamma': Gamma.tolist(),
        'pi_states': pi_states,
        'mu_states': mu_states,
        'gamma_shapes': gamma_shapes,
        'gamma_scales': gamma_scales
    }
    
    print(f"Target: π₀={target_pi0:.2%}, μ={target_mu:.1f}, CV={target_cv:.2f}")
    print(f"Actual: π₀={actual_pi0:.2%}, μ={actual_mu:.1f}, CV={actual_cv:.2f}")
    print(f"States: Dormant(π₀={pi_states[0]:.0%}, μ=${mu_states[0]:.0f}), "
          f"Core(π₀={pi_states[1]:.0%}, μ=${mu_states[1]:.0f}), "
          f"Whale(π₀={pi_states[2]:.0%}, μ=${mu_states[2]:.0f})")
    
    return _package_results(N, T, states, obs, true_params, seed)


# =============================================================================
# PACKAGING & OUTPUT
# =============================================================================

def _package_results(N, T, states, obs, true_params, seed):
    """Package simulation results into standardized format."""
    
    # Compute RFM metrics
    recency = np.zeros(N, dtype=int)
    frequency = np.zeros(N, dtype=int)
    monetary = np.zeros(N)
    
    for i in range(N):
        purchases = obs[i, :] > 0
        if purchases.any():
            last_purchase_idx = np.where(purchases)[0][-1]
            recency[i] = T - 1 - last_purchase_idx
            frequency[i] = purchases.sum()
            monetary[i] = obs[i, purchases].mean()
    
    results = {
        'N': N,
        'T': T,
        'seed': seed,
        'states': states,
        'observations': obs,
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'true_params': true_params,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def save_simulation(results, output_path=None):
    """Save simulation to pickle file."""
    
    dgp_type = results['true_params']['DGP_type'].lower()
    N = results['N']
    T = results['T']
    seed = results['seed']
    
    if output_path is None:
        output_path = f"sim_{dgp_type}_N{N}_T{T}_seed{seed}.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved: {output_path}")
    print(f"  DGP: {results['true_params']['DGP_type']}")
    print(f"  Customers: {N}, Periods: {T}")
    print(f"  Seed: {seed}")
    print(f"  Avg Frequency: {results['frequency'].mean():.2f}")
    print(f"  Avg Monetary: {results['monetary'].mean():.2f}")
    print(f"  Zero rate: {(results['observations'] == 0).mean():.2%}")
    
    return output_path


def save_simulation_csv(results, output_path=None, include_state=False):
    """Save simulation to CSV format for smc_unified_new.py compatibility.
    
    Args:
        results: Simulation results dict
        output_path: Output file path (optional)
        include_state: If True, include true_state column for validation
    """
    N, T = results['N'], results['T']
    obs = results['observations']
    states = results.get('states', None) if include_state else None

    dgp_type = results['true_params']['DGP_type'].lower()
    seed = results['seed']

    if output_path is None:
        state_tag = "_withstate" if include_state else ""
        output_path = f"sim_{dgp_type}_N{N}_T{T}_seed{seed}{state_tag}.csv"

    # Write long-format CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header with optional state column
        header = ['customer_id', 'time_period', 'y']
        if include_state:
            header.append('true_state')
        writer.writerow(header)

        for i in range(N):
            for t in range(T):
                row = [i, t, obs[i, t]]
                if include_state:
                    row.append(int(states[i, t]) if states is not None else -1)
                writer.writerow(row)

    print(f"Saved CSV: {output_path}")
    print(f" Rows: {N * T}, Customers: {N}, Periods: {T}")
    if include_state:
        print(f" Includes: true_state column")

    return output_path


def load_simulation(input_path):
    """Load simulation from pickle file."""
    with open(input_path, 'rb') as f:
        results = pickle.load(f)
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic RFM panel data')
    parser.add_argument('--n_customers', type=int, default=100,
                       help='Number of customers (default: 100)')
    parser.add_argument('--n_periods', type=int, default=60,
                       help='Number of time periods (default: 60)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (optional)')
    parser.add_argument('--zero_prob', type=float, default=None,
                       help='Zero-inflation probability (DGP-specific default if not set)')
    parser.add_argument('--gamma_shape', type=float, default=2.0,
                       help='Gamma shape parameter (mixture DGP only)')
    parser.add_argument('--gamma_scale', type=float, default=4.0,
                       help='Gamma scale parameter (mixture DGP only)')
    parser.add_argument('--format', type=str, default='pkl', choices=['pkl', 'csv', 'both'],
                   help='Output format (default: pkl)')    
    parser.add_argument('--dgp', choices=['mixture', 'bimodal', '4mode', '4mode_v2', '3mode', 'moment_controlled'])
    parser.add_argument('--target_pi0', type=float, default=0.75)
    parser.add_argument('--target_mu', type=float, default=45.0)
    parser.add_argument('--target_cv', type=float, default=2.0)
    parser.add_argument('--include_state', action='store_true',
                       help='Include true state in CSV output')

    args = parser.parse_args()
    
    # Set DGP-specific defaults
    if args.zero_prob is None:
        if args.dgp == 'mixture':
            args.zero_prob = 0.70
        elif args.dgp == 'bimodal':
            args.zero_prob = 0.60
        else:
            args.zero_prob = 0.50
    
# Generate
    kwargs = {
        'zero_prob': args.zero_prob,
        'gamma_shape': args.gamma_shape,
        'gamma_scale': args.gamma_scale
    }
    
    if args.dgp == 'mixture':
        results = generate_mixture_panel(args.n_customers, args.n_periods, args.seed, **kwargs)
    elif args.dgp == 'bimodal':
        results = generate_bimodal_panel(args.n_customers, args.n_periods, args.seed, **kwargs)
    elif args.dgp == '4mode':
        results = generate_4mode_panel(args.n_customers, args.n_periods, args.seed, **kwargs)
    elif args.dgp == '4mode_v2':
        results = generate_4mode_panel_v2(args.n_customers, args.n_periods, args.seed, **kwargs)
    elif args.dgp == '3mode':
        results = generate_3mode_panel(args.n_customers, args.n_periods, args.seed, **kwargs)
    elif args.dgp == 'moment_controlled':
        results = generate_moment_controlled_panel(
            args.n_customers, args.n_periods,
            target_pi0=args.target_pi0,
            target_mu=args.target_mu,
            target_cv=args.target_cv,
            seed=args.seed, **kwargs
        )

    # Save based on format
    if args.format in ['pkl', 'both']:
        pkl_path = args.output or f"sim_{args.dgp}_N{args.n_customers}_T{args.n_periods}_seed{args.seed}.pkl"
        save_simulation(results, pkl_path)
    

    if args.format in ['csv', 'both']:
        if args.output:
            # Use provided output path (change .pkl to .csv if needed)
            csv_path = args.output.replace('.pkl', '.csv') if args.output.endswith('.pkl') else args.output
        else:
            # Default naming
            state_tag = "_withstate" if args.include_state else ""
            csv_path = f"sim_{args.dgp}_N{args.n_customers}_T{args.n_periods}_seed{args.seed}{state_tag}.csv"
        save_simulation_csv(results, csv_path, include_state=True)


if __name__ == "__main__":
    main()
