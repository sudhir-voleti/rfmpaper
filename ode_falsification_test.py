#!/usr/bin/env python3
"""
ode_falsification_test.py
=========================
Validate "Thermodynamics of RFM" theory using HMM posterior.

Usage:
    python ode_falsification_test.py --file results/smc_uci_K2_GAM_statep_N500_D500_C4.pkl --dataset uci
"""
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import plotting libs (optional)
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except:
    HAS_PLT = False

try:
    from scipy.integrate import odeint
    from scipy.interpolate import UnivariateSpline
    HAS_SCIPY = True
except:
    HAS_SCIPY = False


def extract_gamma_diagnostics(idata):
    """Extract transition matrix and compute kinetic beta"""
    try:
        if not hasattr(idata, 'posterior') or 'Gamma' not in idata.posterior:
            return None
            
        gamma_post = idata.posterior['Gamma']  # (chains, draws, K, K)
        gamma_mean = gamma_post.mean(dim=['chain', 'draw']).values
        
        K = gamma_mean.shape[0]
        
        # Compute persistence (diagonal) as proxy for momentum
        persistence = np.diag(gamma_mean)
        
        # Compute dominant eigenvalue for system memory
        eigvals = np.linalg.eigvals(gamma_mean)
        dominant = np.max(np.real(eigvals))
        
        # beta from continuous-time approximation: beta = log(lambda_dom)/dt
        # dt = 1 week
        beta = np.log(dominant) if dominant > 0 else 0
        
        # Off-diagonal flux (conductivity)
        conductivity = (gamma_mean.sum(axis=1) - persistence) / (K-1)
        
        return {
            'Gamma': gamma_mean,
            'persistence': persistence,
            'dominant_eig': dominant,
            'beta': beta,
            'conductivity': conductivity.mean(),
            'K': K
        }
    except Exception as e:
        print(f"Error extracting Gamma: {e}")
        return None


def extract_recency_dissipation(idata):
    """Extract delta from GAM Recency smooth slope"""
    try:
        # Look for smooth terms in posterior
        if not hasattr(idata, 'posterior'):
            return None
            
        post = idata.posterior
        
        # Find recency/smooth coefficients (naming varies by model)
        recency_vars = [v for v in post.data_vars if any(x in v.lower() 
                       for x in ['recency', 's_r', 'smooth_r', 'f_r'])]
        
        if not recency_vars:
            return None
            
        # Get smooth coefficients for each state
        delta_estimates = {}
        
        for var in recency_vars:
            # Try to extract - should have shape involving states
            coeff = post[var].mean(dim=['chain', 'draw']).values
            
            # If state-specific, average absolute slope as dissipation proxy
            if coeff.ndim > 0:
                # Take mean absolute value as dissipation strength
                delta_estimates[var] = float(np.mean(np.abs(coeff)))
        
        # Overall delta (average across states)
        delta = np.mean(list(delta_estimates.values())) if delta_estimates else 0.5
        
        return {
            'delta': delta,
            'delta_by_var': delta_estimates
        }
    except Exception as e:
        print(f"Error extracting delta: {e}")
        return None


def extract_state_p_values(idata):
    """Extract state-specific p estimates"""
    try:
        if not hasattr(idata, 'posterior') or 'p' not in idata.posterior:
            return None
            
        p_post = idata.posterior['p']
        p_mean = p_post.mean(dim=['chain', 'draw']).values
        p_std = p_post.std(dim=['chain', 'draw']).values
        
        if p_mean.ndim == 0:
            return {'p_global': float(p_mean), 'p_states': None}
        
        states = {f'p_{i}': float(p_mean[i]) for i in range(len(p_mean))}
        states['p_range'] = float(p_mean.max() - p_mean.min())
        states['p_ordered'] = [float(p_mean[i]) for i in range(len(p_mean))]
        
        return {
            'p_global': None,
            'p_states': states,
            'p_std': [float(p_std[i]) for i in range(len(p_std))]
        }
    except Exception as e:
        print(f"Error extracting p: {e}")
        return None


def compute_ode_params(gamma_diag, delta):
    """
    Compute ODE parameters per state
    beta_k from diagonal persistence
    delta_k from GAM (assume constant across states or modify if state-specific GAM)
    """
    # beta_k from persistence: high persistence = high momentum
    beta_k = np.log(gamma_diag + 1e-6)  # log(persist) as proxy
    
    # Assume delta constant (or could extract state-specific if model has it)
    delta_k = np.full_like(beta_k, delta)
    
    # Ratio (thermal conductivity proxy)
    ratio = beta_k / (delta_k + 1e-6)
    
    return pd.DataFrame({
        'state': range(len(beta_k)),
        'beta': beta_k,
        'delta': delta_k,
        'beta_over_delta': ratio,
        'persistence': gamma_diag
    })


def correlation_test(ode_df, p_values):
    """Test if p_k correlates with beta/delta"""
    if p_values is None or 'p_ordered' not in p_values:
        return None
    
    p_vec = np.array(p_values['p_ordered'])
    ratio_vec = ode_df['beta_over_delta'].values
    
    if len(p_vec) != len(ratio_vec):
        return None
    
    corr = np.corrcoef(p_vec, ratio_vec)[0,1]
    
    # Linear regression slope
    slope = np.polyfit(ratio_vec, p_vec, 1)[0]
    
    return {
        'correlation': float(corr),
        'slope': float(slope),
        'p_values': p_vec.tolist(),
        'ratios': ratio_vec.tolist()
    }


def simulate_ode(beta, delta, rho0=0.5, T=53, marketing_pulse=None):
    """
    Simulate the propensity ODE:
    dρ/dt = βρ - δR(t)ρ + αM(t)
    
    For simplicity, assume R(t) = t (recency increases linearly with time since last purchase)
    and M(t) = 0 (no external marketing)
    """
    if not HAS_SCIPY:
        return None
    
    def dydt(y, t):
        recency = t  # Time since last purchase proxy
        # Simple decay model: internal momentum vs recency dissipation
        return beta * y - delta * recency * y * 0.1  # scaled recency effect
    
    t = np.linspace(0, T, 100)
    y = odeint(dydt, rho0, t)
    
    return pd.DataFrame({'time': t, 'propensity': y.flatten()})


def main():
    parser = argparse.ArgumentParser(description='ODE Falsification Test for HMM-Tweedie')
    parser.add_argument('--file', type=str, required=True, help='Path to .pkl file')
    parser.add_argument('--dataset', type=str, default='unknown', help='Dataset name (uci/cdnow)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()
    
    print("="*60)
    print(f"ODE FALSIFICATION TEST: {args.dataset}")
    print("="*60)
    
    # Load
    with open(args.file, 'rb') as f:
        idata = pickle.load(f)
    
    # Extract components
    print("\n1. Extracting kinetic parameters...")
    gamma_info = extract_gamma_diagnostics(idata)
    delta_info = extract_recency_dissipation(idata)
    p_info = extract_state_p_values(idata)
    
    if not all([gamma_info, delta_info, p_info]):
        print("✗ Failed to extract necessary components")
        return
    
    # Compute ODE params
    print(f"\n2. Computing ODE parameters for K={gamma_info['K']} states...")
    ode_df = compute_ode_params(gamma_info['persistence'], delta_info['delta'])
    
    # Add p values
    if p_info['p_states']:
        p_ordered = p_info['p_states']['p_ordered']
        ode_df['p_k'] = p_ordered[:len(ode_df)]
        
        print("\n3. State-specific estimates:")
        print(ode_df.to_string(index=False))
        
        # Correlation test
        print("\n4. Correlation test (p_k vs β/δ)...")
        corr_result = correlation_test(ode_df, p_info['p_states'])
        
        if corr_result:
            print(f"   Correlation: {corr_result['correlation']:.3f}")
            print(f"   Regression slope: {corr_result['slope']:.3f}")
            
            if corr_result['correlation'] > 0.5:
                print("   ✓ POSITIVE: Hot states (high p) have high β/δ (momentum dominates dissipation)")
            elif corr_result['correlation'] < -0.3:
                print("   ✗ NEGATIVE: Unexpected inverse relationship")
            else:
                print("   ~ WEAK: No clear kinetic relationship detected")
    
    # Simulate trajectory
    if HAS_SCIPY:
        print("\n5. Simulating ODE trajectory (representative state)...")
        mid_state = len(ode_df) // 2
        beta = ode_df.iloc[mid_state]['beta']
        delta = ode_df.iloc[mid_state]['delta']
        
        traj = simulate_ode(beta, delta, T=53)
        cliff_point = traj[traj['propensity'] < 0.5 * traj['propensity'].iloc[0]]['time'].min()
        print(f"   Recency cliff at t ≈ {cliff_point:.1f} weeks")
    
    # Save results
    if args.output:
        ode_df.to_csv(args.output, index=False)
        print(f"\n6. Saved results to: {args.output}")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print("If correlation > 0.5: Kinetic theory VALIDATED")
    print("  → p_k acts as 'equation of state' for relational reservoir")
    print("  → Hot states (p≈1.8) = high β/δ (boiling)")
    print("  → Cold states (p≈1.2) = low β/δ (frozen)")
    print("="*60)


if __name__ == "__main__":
    main()
