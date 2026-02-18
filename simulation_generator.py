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
        'N': N, '
