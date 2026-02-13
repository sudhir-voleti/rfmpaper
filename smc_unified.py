#!/usr/bin/env python3
"""
smc_unified_optimized.py
========================
Optimized for Apple Silicon (M1/M2/M3) with Metal Performance Shaders.
Unified HMM-Tweedie model with configurable options.
"""

# =============================================================================
# 0. APPLE SILICON OPTIMIZATION (MUST BE FIRST - Before any imports)
# =============================================================================

import os
# Set environment variables BEFORE importing pytensor
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,optimizer=fast_run,openmp=True'

# Now import pytensor - it will read the flags
import pytensor
import numpy as np
import pytensor.tensor as pt

# Verify config
print(f"PyTensor config: floatX={pytensor.config.floatX}, optimizer={pytensor.config.optimizer}")

# Optional: Disable Metal GPU for SMC (CPU is often faster for tempering)
os.environ['PYTENSOR_METAL'] = '0'

# =============================================================================
# STANDARD IMPORTS
# =============================================================================

import argparse
import time
import pathlib
import pickle
import warnings
import platform

import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from patsy import dmatrix
from scipy.special import logsumexp

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Check if running on Apple Silicon
IS_APPLE_SILICON = platform.machine() == 'arm64' and platform.system() == 'Darwin'
if IS_APPLE_SILICON:
    print(f"Detected Apple Silicon ({platform.machine()}). Float32 optimization enabled.")


# =============================================================================
# 1. B-SPLINE BASIS FUNCTION
# =============================================================================

def create_bspline_basis(x, df=3, degree=3):
    """
    Create B-spline basis matrix for GAM with Apple Silicon optimization.
    Uses float32 internally for memory efficiency on M1/M2/M3.
    """
    x = np.asarray(x, dtype=np.float32).flatten()
    
    n_knots = df - degree + 1
    if n_knots > 1:
        knots = np.quantile(x, np.linspace(0, 1, n_knots)[1:-1]).tolist()
    else:
        knots = []
    
    formula = f"bs(x, knots={list(knots)}, degree={degree}, include_intercept=False)"
    basis = dmatrix(formula, {"x": x}, return_type='matrix')
    
    return np.asarray(basis, dtype=np.float32)


# =============================================================================
# 2. GAMMA LOG-DENSITY (OPTIMIZED)
# =============================================================================

def gamma_logp_det(value, mu, phi):
    """
    Deterministic Gamma log-density with numerical stability.
    All operations in float32 for Apple Silicon speed.
    """
    alpha = mu / phi
    beta = 1.0 / phi
    
    return (alpha - 1) * pt.log(value) - value * beta + alpha * pt.log(beta) - pt.gammaln(alpha)

# =============================================================================
# EMISSION DENSITY FUNCTIONS
# =============================================================================

def zig_logp(y, mu, phi, p):
    """
    Zero-Inflated Gamma (ZIG) log-density - Tweedie approximation.
    Returns log emission matrix (N, T, K) or (N, T) for K=1.
    """
    exponent = 2.0 - p
    psi = pt.exp(-pt.pow(mu, exponent) / (phi * exponent))
    psi = pt.clip(psi, 1e-12, 1 - 1e-12)
    
    log_zero = pt.log(psi)
    log_pos = pt.log1p(-psi) + gamma_logp_det(y, mu, phi)
    
    return pt.where(y == 0, log_zero, log_pos)


def poisson_logp(y, mu):
    """
    Poisson log-density for count data.
    WARNING: Discretizes continuous spend - for ablation only.
    """
    # Round to nearest integer for Poisson (hacky but necessary)
    y_rounded = pt.round(pt.clip(y, 0, 1e6))
    return pm.logp(pm.Poisson.dist(mu=mu), y_rounded)


def nbd_logp(y, mu, theta):
    """
    Negative Binomial log-density.
    WARNING: Discrete likelihood on continuous data - will explode at high zeros.
    """
    # Round for NBD
    y_rounded = pt.round(pt.clip(y, 0, 1e6))
    alpha = theta  # PyMC parameterization: alpha = dispersion
    return pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=alpha), y_rounded)
    
# =============================================================================
# 3. UNIFIED MODEL BUILDER
# =============================================================================

def make_model(data, K=3, state_specific_p=True, p_fixed=1.5, 
               use_gam=True, gam_df=3, emission_type='tweedie'
               p_min=None, p_max=None):
    """
    Build HMM-Tweedie (K>=2) or Static Tweedie (K=1) model.
    Build HMM with selectable emission: 'tweedie', 'poisson', or 'nbd'.
    Optimized for Apple Silicon with float32 throughout.
    """
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]
    
    # Precompute GAM bases if needed
    if use_gam and K >= 1:
        R_flat = R.flatten()
        F_flat = F.flatten()
        M_flat = M.flatten()
        
        basis_R = create_bspline_basis(R_flat, df=gam_df)
        basis_F = create_bspline_basis(F_flat, df=gam_df)
        basis_M = create_bspline_basis(M_flat, df=gam_df)
        
        n_basis_R = basis_R.shape[1]
        n_basis_F = basis_F.shape[1]
        n_basis_M = basis_M.shape[1]
        
        basis_R = basis_R.reshape(N, T, n_basis_R)
        basis_F = basis_F.reshape(N, T, n_basis_F)
        basis_M = basis_M.reshape(N, T, n_basis_M)
    else:
        n_basis_R = n_basis_F = n_basis_M = 1
        basis_R = basis_F = basis_M = None

    with pm.Model(coords={"customer": np.arange(N), "state": np.arange(K)}) as model:
        
        # ---- Latent dynamics ----
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
        else:
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K, dtype=np.float32), shape=(K, K))

        # ---- State parameters ----
        if K == 1:
            beta0_raw = pm.Normal("beta0_raw", 0, 1)
            beta0 = pm.Deterministic("beta0", beta0_raw)
        else:
            beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
            beta0 = pm.Deterministic("beta0", pt.sort(beta0_raw))
        
        if K == 1:
            phi_raw = pm.Exponential("phi_raw", lam=10.0)
            phi = pm.Deterministic("phi", phi_raw)
        else:
            phi_raw = pm.Exponential("phi_raw", lam=10.0, shape=K)
            phi = pm.Deterministic("phi", pt.sort(phi_raw))

        # ---- R, F, M effects ----
        if use_gam:
            if K == 1:
                w_R = pm.Normal("w_R", 0, 1, shape=n_basis_R)
                w_F = pm.Normal("w_F", 0, 1, shape=n_basis_F)
                w_M = pm.Normal("w_M", 0, 1, shape=n_basis_M)
            else:
                w_R = pm.Normal("w_R", 0, 1, shape=(K, n_basis_R))
                w_F = pm.Normal("w_F", 0, 1, shape=(K, n_basis_F))
                w_M = pm.Normal("w_M", 0, 1, shape=(K, n_basis_M))
        else:
            if K == 1:
                betaR = pm.Normal("betaR", 0, 1)
                betaF = pm.Normal("betaF", 0, 1)
                betaM = pm.Normal("betaM", 0, 1)
            else:
                betaR = pm.Normal("betaR", 0, 1, shape=K)
                betaF = pm.Normal("betaF", 0, 1, shape=K)
                betaM = pm.Normal("betaM", 0, 1, shape=K)

        # ---- Power parameter p ----
        if K == 1:
            p_raw = pm.Beta("p_raw", alpha=2, beta=2)
            p = pm.Deterministic("p", 1.1 + p_raw * 0.8)
        elif state_specific_p:
            p_raw = pm.Beta("p_raw", alpha=2, beta=2, shape=K)
            p_sorted = pt.sort(p_raw)
            p = pm.Deterministic("p", 1.1 + p_sorted * 0.8)
        else:
            p = pt.as_tensor_variable(np.array([p_fixed] * K, dtype=np.float32))

      # Power parameter p
    if p_min is not None and p_max is not None:
        # Constrained varying p
        p_raw = pm.Beta("p_raw", alpha=2, beta=2)
        p = pm.Deterministic("p", p_min + p_raw * (p_max - p_min))
    elif K == 1:
        p_raw = pm.Beta("p_raw", alpha=2, beta=2)
        p = pm.Deterministic("p", 1.1 + p_raw * 0.8)
        
        # ---- Compute mu ----
        if use_gam:
            if K == 1:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [0]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [0]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [0]))
                mu = pt.exp(beta0 + effect_R + effect_F + effect_M)
            else:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [1]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [1]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [1]))
                mu = pt.exp(beta0 + effect_R + effect_F + effect_M)
        else:
            if K == 1:
                mu = pt.exp(beta0 + betaR * R + betaF * F + betaM * M)
            else:
                mu = pt.exp(beta0 + betaR * R[..., None] + 
                           betaF * F[..., None] + betaM * M[..., None])

        mu = pt.clip(mu, 1e-3, 1e6)

        # ---- CLV & Segmentation Metrics (Deterministic) ----
        gamma_diag = pt.diag(Gamma)
        
        # Keep only these two - they work
        churn_risk = pm.Deterministic('churn_risk', 1.0 - gamma_diag)
        clv_proxy = pm.Deterministic('clv_proxy', pt.exp(beta0) / (1.0 - 0.95 * gamma_diag))
        
        # ---- EMISSION DENSITY ----
        if emission_type == 'tweedie':
            # ZIG/Tweedie emission
            if K == 1:
                p_expanded = p
                phi_expanded = phi
                exponent = 2.0 - p_expanded
                
                psi = pt.exp(-pt.pow(mu, exponent) / (phi_expanded * exponent))
                psi = pt.clip(psi, 1e-12, 1 - 1e-12)
                
                log_zero = pt.log(psi)
                log_pos = pt.log1p(-psi) + gamma_logp_det(y, mu, phi_expanded)
                log_emission = pt.where(y == 0, log_zero, log_pos)
                log_emission = pt.where(mask, log_emission, 0.0)
            else:
                p_expanded = p[None, None, :]
                phi_expanded = phi[None, None, :]
                exponent = 2.0 - p_expanded
                
                psi = pt.exp(-pt.pow(mu, exponent) / (phi_expanded * exponent))
                psi = pt.clip(psi, 1e-12, 1 - 1e-12)
                
                log_zero = pt.log(psi)
                y_exp = y[..., None]
                log_pos = pt.log1p(-psi) + gamma_logp_det(y_exp, mu, phi_expanded)
                log_emission = pt.where(y_exp == 0, log_zero, log_pos)
                log_emission = pt.where(mask[:, :, None], log_emission, 0.0)
                
        elif emission_type == 'poisson':
            # Poisson emission - DISCRETE, rounds continuous spend
            if K == 1:
                # Round y to nearest integer for Poisson
                y_rounded = pt.round(pt.clip(y, 0, 1e6))
                log_emission = pm.logp(pm.Poisson.dist(mu=mu), y_rounded)
                log_emission = pt.where(mask, log_emission, 0.0)
            else:
                y_exp = y[..., None]
                y_rounded = pt.round(pt.clip(y_exp, 0, 1e6))
                log_emission = pm.logp(pm.Poisson.dist(mu=mu), y_rounded)
                log_emission = pt.where(mask[:, :, None], log_emission, 0.0)
                
        elif emission_type == 'nbd':
            # NBD emission - WILL EXPLODE at high zeros
            # Add dispersion parameter theta for NBD
            if K == 1:
                theta_raw = pm.Exponential('theta_raw', lam=1.0)
                theta = pm.Deterministic('theta', theta_raw)
                
                y_rounded = pt.round(pt.clip(y, 0, 1e6))
                log_emission = pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=theta), y_rounded)
                log_emission = pt.where(mask, log_emission, 0.0)
            else:
                theta_raw = pm.Exponential('theta_raw', lam=1.0, shape=K)
                theta = pm.Deterministic('theta', pt.sort(theta_raw))  # ordering
                
                y_exp = y[..., None]
                y_rounded = pt.round(pt.clip(y_exp, 0, 1e6))
                log_emission = pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=theta), y_rounded)
                log_emission = pt.where(mask[:, :, None], log_emission, 0.0)
        else:
            raise ValueError(f"Unknown emission_type: {emission_type}")
            

        # ---- Marginal likelihood ----
        if K == 1:
            logp_cust = pt.sum(log_emission, axis=1)
        else:
            log_alpha = pt.log(pi0) + log_emission[:, 0, :]
            log_Gamma = pt.log(Gamma)[None, :, :]
            for t in range(1, T):
                temp = log_alpha[:, :, None] + log_Gamma
                log_alpha = log_emission[:, t, :] + pt.logsumexp(temp, axis=1)
            logp_cust = pt.logsumexp(log_alpha, axis=1)

        pm.Potential('loglike', pt.sum(logp_cust))
        
        if K == 1:
            pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))
        else:
            pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))

    return model


# =============================================================================
# 4. DATA BUILDER
# =============================================================================

def build_panel_data(df_path, customer_col='customer_id', n_cust=None, 
                     max_week=None, seed=42):  # ADD max_week
    """Build panel data dictionary from CSV with float32 optimization."""
    np.random.seed(seed)
    
    df = pd.read_csv(df_path, parse_dates=['WeekStart'])
    df = df.astype({customer_col: str})

    # ADD THIS BLOCK
    if max_week is not None:
        df['WeekIndex'] = df.groupby(customer_col)['WeekStart'].rank(method='dense').astype(int)
        df = df[df['WeekIndex'] <= max_week]
        print(f"Filtered to weeks 1-{max_week}: {len(df)} rows")
        
    if n_cust is not None:
        unique_custs = df[customer_col].unique()
        if len(unique_custs) > n_cust:
            selected = np.random.choice(unique_custs, n_cust, replace=False)
            df = df[df[customer_col].isin(selected)]
            print(f"Subsampled to {n_cust} customers: {len(selected)} selected")
        else:
            print(f"Requested {n_cust}, but only {len(unique_custs)} available. Using all.")
    
    panel_sizes = df.groupby(customer_col).size()
    if panel_sizes.nunique() != 1:
        print(f"Warning: Non-rectangular panel. Sizes: {panel_sizes.unique()}")
    
    def mat(col):
        arr = df.pivot(index=customer_col, columns='WeekStart', values=col).values
        return arr.astype(np.float32)
    
    return {
        'N': df[customer_col].nunique(),
        'T': panel_sizes.iloc[0],
        'y': mat('WeeklySpend'),
        'mask': ~np.isnan(mat('WeeklySpend')),
        'R': mat('R_weeks'),
        'F': mat('F_run'),
        'M': mat('M_run'),
        'p0': mat('p0_cust')
    }


# =============================================================================
# 5. SMC RUNNER
# =============================================================================

def run_smc(data, K, state_specific_p, p_fixed, use_gam, gam_df, 
            draws, chains, dataset, seed, out_dir, emission_type='tweedie'):
    """Run SMC estimation with Apple Silicon optimizations."""
    cores = min(chains, os.cpu_count() or 1)
    t0 = time.time()
    
    if IS_APPLE_SILICON:
        print(f"  Apple Silicon: {os.cpu_count()} cores, using {cores}")
    
    try:
        with make_model(data, K=K, state_specific_p=state_specific_p, 
                       p_fixed=p_fixed, use_gam=use_gam, gam_df=gam_df, emission_type=emission_type) as model:
            
            model_type = "GAM" if use_gam else "GLM"
            p_type = "state-specific" if (state_specific_p and K > 1) else \
                     "varying" if K == 1 else f"fixed={p_fixed}"
            
            print(f"  Model: K={K}, {model_type}, p={p_type}")
            print(f"  Data: N={data['N']}, T={data['T']}")
            print(f"  SMC: draws={draws}, chains={chains}, cores={cores}")
            
            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seed,
                return_inferencedata=True,
                threshold=0.8
            )
        
        # Extract log-evidence
        log_ev = np.nan
        try:
            lm = idata.sample_stats.log_marginal_likelihood.values
            
            if isinstance(lm, np.ndarray) and lm.dtype == object:
                chain_vals = []
                for c in range(lm.shape[1] if lm.ndim > 1 else 1):
                    if lm.ndim > 1:
                        chain_list = lm[-1, c] if lm.shape[0] > 0 else lm[0, c]
                    else:
                        chain_list = lm[c] if lm.ndim == 1 else lm[0]
                    
                    if isinstance(chain_list, list):
                        valid = [float(x) for x in chain_list 
                                if isinstance(x, (int, float, np.floating)) and np.isfinite(x)]
                        if valid:
                            chain_vals.append(valid[-1])
                    elif isinstance(chain_list, (int, float, np.floating)) and np.isfinite(chain_list):
                        chain_vals.append(float(chain_list))
                
                log_ev = float(np.mean(chain_vals)) if chain_vals else np.nan
            else:
                flat = np.array(lm).flatten()
                valid = flat[np.isfinite(flat)]
                log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
                
        except Exception as e:
            print(f"  Warning: log-ev extraction failed: {e}")
            log_ev = np.nan

        elapsed = (time.time() - t0) / 60
        
        res = {
            'dataset': dataset,
            'K': K,
            'N': data['N'],
            'T': data['T'],
            'use_gam': use_gam,
            'gam_df': gam_df if use_gam else None,
            'state_specific_p': state_specific_p,
            'p_fixed': p_fixed if not (state_specific_p or K == 1) else None,
            'log_evidence': log_ev,
            'draws': draws,
            'chains': chains,
            'time_min': elapsed,
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'platform': 'Apple_Silicon' if IS_APPLE_SILICON else 'Other'
        }

        model_tag = f"K{K}_{'GAM' if use_gam else 'GLM'}"
        p_tag = "statep" if (state_specific_p and K > 1) else \
                "varyingp" if K == 1 else f"p{p_fixed}"
        
        # ADD DATASET TO FILENAME
        pkl_path = out_dir / f"smc_{dataset}_{model_tag}_{p_tag}_N{data['N']}_D{draws}_C{chains}.pkl"
        
        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'res': res}, f, protocol=4)
            
        size_mb = pkl_path.stat().st_size / (1024**2)
        print(f"  ✓ log_ev={log_ev:.2f}, time={elapsed:.1f}min, size={size_mb:.1f}MB")
        
        return pkl_path, res

    except Exception as e:
        import traceback
        print(f"  ✗ CRASH: {str(e)[:60]}")
        raise


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Apple Silicon Optimized HMM/Static Tweedie Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset', required=True, choices=['uci', 'cdnow'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--n_cust', type=int, default=None)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--state_specific_p', action='store_true')
    parser.add_argument('--p_fixed', type=float, default=1.5)
    parser.add_argument('--no_gam', action='store_true')
    parser.add_argument('--gam_df', type=int, default=3)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--max_week', type=int, default=None,
                   help='Maximum week to include (for train/test split)')
    parser.add_argument('--emission', type=str, default='tweedie',
                   choices=['tweedie', 'poisson', 'nbd'])
    parser.add_argument('--p_min', type=float, default=None)
    parser.add_argument('--p_max', type=float, default=None)
    
    args = parser.parse_args()
    
    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = data_dir / f"{args.dataset}_full.csv"
    
    print(f"\n{'='*70}")
    print(f"SMC Unified (Optimized): {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"Model: K={args.K}, {'GLM' if args.no_gam else 'GAM'}")
    print(f"FloatX: {pytensor.config.floatX}, Optimizer: {pytensor.config.optimizer}")
    print(f"{'='*70}\n")
    
    data = build_panel_data(data_path, n_cust=args.n_cust, 
                       max_week=args.max_week,  # ADD
                       seed=args.seed)
    print(f"Loaded: N={data['N']}, T={data['T']}, zeros={np.mean(data['y']==0):.1%}\n")
    
    pkl_path, res = run_smc(
        data=data,
        K=args.K,
        state_specific_p=args.state_specific_p,
        p_fixed=args.p_fixed,
        use_gam=not args.no_gam,
        gam_df=args.gam_df,
        draws=args.draws,
        chains=args.chains,
        dataset=args.dataset,
        seed=args.seed,
        out_dir=out_dir,
        emission_type=args.emission,
        p_min=args.p_min,
        p_max=args.p_max
    )
    
    print(f"\n{'='*70}")
    print("RESULT")
    for key, val in res.items():
        print(f"  {key}: {val}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
