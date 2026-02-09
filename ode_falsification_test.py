#!/usr/bin/env python3
"""
ode_falsification_test.py - COMPLETE VALIDATION SUITE
Tests 1-4: Persistence, Equation of State, ODE Simulation, Conductivity
Usage: python ode_falsification_test.py --file results/smc_uci_K4_*.pkl --dataset uci
"""
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional imports for ODE solving
try:
    from scipy.integrate import odeint
    HAS_SCIPY = True
except:
    HAS_SCIPY = False

def load_idata(pkl_path):
    """Universal loader (nested dict or direct)"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'idata' in data:
        return data['idata'], data.get('res', {})
    return data, {}

# =============================================================================
# TEST 1: p vs Persistence (Thermometer)
# =============================================================================
def test1_persistence_correlation(idata):
    """Test 1: Correlation between p and diagonal Gamma (persistence)"""
    try:
        gamma_mean = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values
        p_mean = idata.posterior['p'].mean(dim=['chain', 'draw']).values
        
        if p_mean.ndim == 0 or len(p_mean) < 2:
            return None
            
        persistence = np.diag(gamma_mean)
        corr = np.corrcoef(p_mean, persistence)[0, 1]
        
        return {
            'test': 'p_vs_persistence',
            'correlation': float(corr),
            'p_values': p_mean.tolist(),
            'persistence': persistence.tolist(),
            'interpretation': 'Leidenfrost' if corr < -0.5 else 'Gradual' if corr > 0.5 else 'Weak'
        }
    except Exception as e:
        return {'test': 'p_vs_persistence', 'error': str(e)}

# =============================================================================
# TEST 2: Equation of State (β/δ vs p) - FULL BAYESIAN
# =============================================================================
def test2_equation_of_state(idata):
    """Test 2: Full posterior of beta/delta ratio vs p"""
    try:
        post = idata.posterior
        gamma = post['Gamma'].values
        n_chains, n_draws, K, _ = gamma.shape
        n_samp = n_chains * n_draws
        
        # Reshape
        gamma_flat = gamma.reshape(n_samp, K, K)
        gamma_diag = np.array([np.diag(g) for g in gamma_flat])
        
        # Beta (log-odds)
        beta = np.log(gamma_diag / (1 - gamma_diag + 1e-8))
        
        # Delta from w_R
        if 'w_R' in post:
            w_R = post['w_R'].values.reshape(n_samp, -1)
            delta = np.mean(np.abs(w_R), axis=1, keepdims=True)
            delta = np.tile(delta, (1, K))
        else:
            delta = np.ones((n_samp, K)) * 0.5
        
        # p
        p = post['p'].values.reshape(n_samp, K)
        
        # Compute correlation for each posterior draw
        cors = []
        for i in range(n_samp):
            if K >= 3:
                r = np.corrcoef(beta[i]/delta[i], p[i])[0,1]
                if not np.isnan(r):
                    cors.append(r)
        
        cors = np.array(cors)
        
        return {
            'test': 'equation_of_state',
            'correlation_mean': float(np.mean(cors)),
            'correlation_std': float(np.std(cors)),
            'hdi_95': (float(np.percentile(cors, 2.5)), float(np.percentile(cors, 97.5))),
            'prob_positive': float(np.mean(cors > 0)),
            'prob_strong': float(np.mean(np.abs(cors) > 0.5)),
            'n_samples': len(cors)
        }
    except Exception as e:
        return {'test': 'equation_of_state', 'error': str(e)}

# =============================================================================
# TEST 3: ODE Simulation (Recency Cliff Prediction)
# =============================================================================
def test3_ode_simulation(idata):
    """Test 3: Solve dρ/dt and check for cliff at R_crit = β/δ"""
    if not HAS_SCIPY:
        return {'test': 'ode_simulation', 'status': 'scipy_not_available'}
    
    try:
        # Extract mean parameters
        gamma_mean = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values
        beta_mean = np.log(np.diag(gamma_mean) + 1e-8)
        
        if 'w_R' in idata.posterior:
            delta_mean = float(np.abs(idata.posterior['w_R'].mean()).values.mean())
        else:
            delta_mean = 0.5
        
        # Solve ODE: dρ/dt = βρ - δRρ (R = t, time since last purchase)
        def dydt(y, t, beta, delta):
            return beta * y - delta * t * y * 0.1
        
        t = np.linspace(0, 52, 100)  # 52 weeks
        results = {}
        
        for k in range(len(beta_mean)):
            y0 = 0.5  # Initial propensity
            sol = odeint(dydt, y0, t, args=(beta_mean[k], delta_mean))
            
            # Find cliff (where propensity drops below 50% of initial)
            cliff_idx = np.where(sol < 0.25)[0]
            cliff_week = int(t[cliff_idx[0]]) if len(cliff_idx) > 0 else None
            
            results[f'state_{k}'] = {
                'beta': float(beta_mean[k]),
                'cliff_week': cliff_week,
                'final_propensity': float(sol[-1])
            }
        
        return {
            'test': 'ode_simulation',
            'delta_used': delta_mean,
            'states': results
        }
    except Exception as e:
        return {'test': 'ode_simulation', 'error': str(e)}

# =============================================================================
# TEST 4: Thermal Conductivity (κ)
# =============================================================================
def test4_conductivity(idata):
    """Test 4: Off-diagonal flux as thermal conductivity"""
    try:
        gamma_mean = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values
        K = gamma_mean.shape[0]
        
        # Conductivity = average off-diagonal rate
        off_diag_mask = ~np.eye(K, dtype=bool)
        kappa = gamma_mean[off_diag_mask].mean()
        
        # Also compute state-specific mobility
        mobility = (gamma_mean.sum(axis=1) - np.diag(gamma_mean)) / (K-1)
        
        return {
            'test': 'conductivity',
            'kappa_global': float(kappa),
            'mobility_by_state': mobility.tolist(),
            'K': K,
            'interpretation': 'High' if kappa > 0.2 else 'Low' if kappa < 0.1 else 'Moderate'
        }
    except Exception as e:
        return {'test': 'conductivity', 'error': str(e)}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='ODE Falsification - Complete Test Suite')
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='unknown')
    parser.add_argument('--output', type=str, default='results/falsification_report.csv')
    args = parser.parse_args()
    
    print("="*80)
    print(f"ODE FALSIFICATION TEST SUITE: {args.dataset}")
    print("="*80)
    
    # Load
    print(f"\nLoading: {args.file}")
    idata, meta = load_idata(args.file)
    K = meta.get('K', 'unknown')
    print(f"Model: K={K}, Dataset: {args.dataset}")
    
    # Run all tests
    results = []
    
    print("\n" + "-"*80)
    print("TEST 1: Persistence Correlation (p vs γ)")
    print("-"*80)
    r1 = test1_persistence_correlation(idata)
    results.append(r1)
    if 'correlation' in r1:
        print(f"Correlation: {r1['correlation']:.3f}")
        print(f"Pattern: {r1['interpretation']}")
    
    print("\n" + "-"*80)
    print("TEST 2: Equation of State (β/δ vs p) - Bayesian")
    print("-"*80)
    r2 = test2_equation_of_state(idata)
    results.append(r2)
    if 'correlation_mean' in r2:
        print(f"Posterior Correlation: {r2['correlation_mean']:.3f} ± {r2['correlation_std']:.3f}")
        print(f"95% HDI: [{r2['hdi_95'][0]:.3f}, {r2['hdi_95'][1]:.3f}]")
        print(f"P(corr > 0): {r2['prob_positive']:.1%}")
    
    print("\n" + "-"*80)
    print("TEST 3: ODE Simulation (Recency Cliff)")
    print("-"*80)
    r3 = test3_ode_simulation(idata)
    results.append(r3)
    if 'states' in r3:
        print(f"Delta parameter: {r3['delta_used']:.4f}")
        for state, vals in r3['states'].items():
            cliff = vals['cliff_week'] if vals['cliff_week'] else "None"
            print(f"  {state}: cliff at week {cliff}, final ρ={vals['final_propensity']:.3f}")
    
    print("\n" + "-"*80)
    print("TEST 4: Thermal Conductivity (κ)")
    print("-"*80)
    r4 = test4_conductivity(idata)
    results.append(r4)
    if 'kappa_global' in r4:
        print(f"Global κ: {r4['kappa_global']:.3f}")
        print(f"Classification: {r4['interpretation']} conductivity")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Determine regime
    if 'correlation' in r1 and 'correlation_mean' in r2:
        c1 = r1['correlation']
        c2 = r2['correlation_mean']
        
        if c1 < -0.5 and c2 < 0:
            print("REGIME: Boiling Pot with Leidenfrost Point")
            print("  - High p correlates with LOW persistence (unstable whales)")
            print("  - Equation of State: INVERSE relationship")
        elif c1 > 0.5 and c2 > 0:
            print("REGIME: Gradual Heating (Leaky Bucket)")
            print("  - High p correlates with HIGH persistence (stable loyalty)")
            print("  - Equation of State: Standard thermodynamics")
        else:
            print("REGIME: Mixed/Uncertain")
    
    if 'kappa_global' in r4:
        print(f"  - Thermal conductivity: {r4['interpretation']}")
    
    # Export detailed results
    flat_results = {
        'dataset': args.dataset,
        'file': Path(args.file).name,
        'K': K
    }
    for r in results:
        test_name = r['test']
        for k, v in r.items():
            if k != 'test':
                flat_results[f"{test_name}_{k}"] = v
    
    df = pd.DataFrame([flat_results])
    df.to_csv(args.output, index=False)
    print(f"\nDetailed report saved to: {args.output}")

if __name__ == "__main__":
    main()
