#!/usr/bin/env python3
"""
smc_unified.py
==============
Unified HMM-Tweedie model with configurable options:
- K states (K=1 is static, K>=2 is HMM)
- GLM vs GAM for RFM effects
- State-specific p vs fixed p
- Configurable draws, chains, customers

Upload to: https://github.com/sudhir-voleti/rfmpaper/blob/main/smc_unified.py
"""

import argparse
import os
import time
import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from patsy import dmatrix

import pytensor
pytensor.config.floatX = 'float32'  # Add this line

warnings.filterwarnings('ignore')

# ---------- 0. Configuration ----------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- 1. B-Spline Basis Function ----------
def create_bspline_basis(x, df=3, degree=3):
    """
    Create B-spline basis matrix for GAM.
    
    Parameters:
    -----------
    x : array-like
        Input values (flattened)
    df : int
        Degrees of freedom (determines number of knots)
    degree : int
        Polynomial degree for splines
    
    Returns:
    --------
    basis : ndarray, shape (len(x), n_basis)
        B-spline basis matrix
    """
    x = np.asarray(x).flatten()
    
    # Quantile-based knots for data-adaptive placement
    n_knots = df - degree + 1
    if n_knots > 1:
        knots = np.quantile(x, np.linspace(0, 1, n_knots)[1:-1]).tolist()
    else:
        knots = []
    
    formula = f"bs(x, knots={knots}, degree={degree}, include_intercept=False)"
    basis = dmatrix(formula, {"x": x}, return_type='matrix')
    
    return np.asarray(basis)


# ---------- 2. Gamma Log-Density ----------
def gamma_logp_det(value, mu, phi):
    """Deterministic Gamma log-density for ZIG positive part."""
    alpha = mu / phi
    beta = 1.0 / phi
    return (alpha - 1) * pt.log(value) - value * beta + alpha * pt.log(beta) - pt.gammaln(alpha)


# ---------- 3. Unified Model Builder ----------
def make_model(data, K=3, state_specific_p=True, p_fixed=1.5, 
               use_gam=True, gam_df=3):
    """
    Build HMM-Tweedie (K>=2) or Static Tweedie (K=1) model.
    
    Parameters:
    -----------
    data : dict
        Contains N, T, y, mask, R, F, M
    K : int
        Number of states. K=1 is static, K>=2 is HMM.
    state_specific_p : bool
        If True and K>1, estimate p per state. If False, use p_fixed.
    p_fixed : float
        Global p value when state_specific_p=False.
    use_gam : bool
        If True, use B-spline GAM. If False, use linear GLM.
    gam_df : int
        Degrees of freedom for B-splines.
    
    Returns:
    --------
    model : pm.Model
        PyMC model object
    """
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]
    
    # Precompute GAM bases if needed
    if use_gam and K >= 1:  # Allow GAM even for K=1
        # Flatten for basis computation
        R_flat = R.flatten()
        F_flat = F.flatten()
        M_flat = M.flatten()
        
        # Create bases
        basis_R = create_bspline_basis(R_flat, df=gam_df)
        basis_F = create_bspline_basis(F_flat, df=gam_df)
        basis_M = create_bspline_basis(M_flat, df=gam_df)
        
        n_basis_R = basis_R.shape[1]
        n_basis_F = basis_F.shape[1]
        n_basis_M = basis_M.shape[1]
        
        # Reshape to (N, T, n_basis)
        basis_R = basis_R.reshape(N, T, n_basis_R)
        basis_F = basis_F.reshape(N, T, n_basis_F)
        basis_M = basis_M.reshape(N, T, n_basis_M)
    else:
        # GLM: single coefficient per state
        n_basis_R = n_basis_F = n_basis_M = 1
        basis_R = basis_F = basis_M = None

    with pm.Model() as model:
        
        # ---- Latent dynamics (K=1: static, no transitions) ----
        if K == 1:
            # Deterministic for static model
            pi0 = pt.as_tensor_variable(np.array([1.0]))
            Gamma = pt.as_tensor_variable(np.array([[1.0]]))
        else:
            # HMM: estimate initial probs and transition matrix
            pi0 = pm.Dirichlet("pi0", a=np.ones(K))
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K), shape=(K, K))

        # ---- State-specific parameters ----
        if K == 1:
            # Scalar for static model
            beta0_raw = pm.Normal("beta0_raw", 0, 1)
            beta0 = pm.Deterministic("beta0", beta0_raw)
        else:
            # Ordered for HMM (Cold < Warm < Hot)
            beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
            beta0 = pm.Deterministic("beta0", pt.sort(beta0_raw))
        
        # Dispersion phi
        if K == 1:
            phi_raw = pm.Exponential("phi_raw", lam=10.0)
            phi = pm.Deterministic("phi", phi_raw)
        else:
            phi_raw = pm.Exponential("phi_raw", lam=10.0, shape=K)
            phi = pm.Deterministic("phi", pt.sort(phi_raw))

        # ---- R, F, M effects (GAM or GLM) ----
        if use_gam:
            # GAM: weights for B-spline bases
            if K == 1:
                w_R = pm.Normal("w_R", 0, 1, shape=n_basis_R)
                w_F = pm.Normal("w_F", 0, 1, shape=n_basis_F)
                w_M = pm.Normal("w_M", 0, 1, shape=n_basis_M)
            else:
                w_R = pm.Normal("w_R", 0, 1, shape=(K, n_basis_R))
                w_F = pm.Normal("w_F", 0, 1, shape=(K, n_basis_F))
                w_M = pm.Normal("w_M", 0, 1, shape=(K, n_basis_M))
        else:
            # GLM: linear coefficients
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
            # Always estimate p for static model (comparable to HMM)
            p_raw = pm.Beta("p_raw", alpha=2, beta=2)
            p = pm.Deterministic("p", 1.1 + p_raw * 0.8)  # [1.1, 1.9]
        elif state_specific_p:
            # State-specific p for HMM
            p_raw = pm.Beta("p_raw", alpha=2, beta=2, shape=K)
            p_sorted = pt.sort(p_raw)
            p = pm.Deterministic("p", 1.1 + p_sorted * 0.8)
        else:
            # Fixed global p for HMM
            p = pt.as_tensor_variable(np.array([p_fixed] * K))

        # ---- Compute mu (state-varying mean) ----
        if use_gam:
            # GAM effects via tensor dot product
            if K == 1:
                # Static: (N, T, n_basis) dot (n_basis,) -> (N, T)
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [0]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [0]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [0]))
                mu = pt.exp(beta0 + effect_R + effect_F + effect_M)
            else:
                # HMM: (N, T, n_basis) tensordot (K, n_basis) -> (N, T, K)
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [1]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [1]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [1]))
                mu = pt.exp(beta0 + effect_R + effect_F + effect_M)
        else:
            # GLM: simple linear effects
            if K == 1:
                mu = pt.exp(beta0 + betaR * R + betaF * F + betaM * M)
            else:
                mu = pt.exp(beta0 + betaR * R[..., None] + 
                           betaF * F[..., None] + betaM * M[..., None])
        
        mu = pt.clip(mu, 1e-3, 1e6)

        # ---- ZIG emission ----
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

        # ---- Marginal likelihood ----
        if K == 1:
            # Static: sum over time for each customer
            logp_cust = pt.sum(log_emission, axis=1)
        else:
            # HMM: forward algorithm
            log_alpha = pt.log(pi0) + log_emission[:, 0, :]
            log_Gamma = pt.log(Gamma)[None, :, :]
            for t in range(1, T):
                temp = log_alpha[:, :, None] + log_Gamma
                log_alpha = log_emission[:, t, :] + pt.logsumexp(temp, axis=1)
            logp_cust = pt.logsumexp(log_alpha, axis=1)

        pm.Potential('loglike', pt.sum(logp_cust))
        
        # Store log-likelihood for diagnostics
        if K == 1:
            pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))
        else:
            pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))

    return model


# ---------- 4. Data Builder ----------
def build_panel_data(df_path, customer_col='customer_id', n_cust=None, seed=RANDOM_SEED):
    """
    Build panel data dictionary from CSV.
    
    Parameters:
    -----------
    df_path : str or Path
        Path to CSV file
    customer_col : str
        Column name for customer ID
    n_cust : int or None
        Number of customers to subsample. If None, use all.
    seed : int
        Random seed for reproducible subsampling.
    
    Returns:
    --------
    data : dict
        Dictionary with N, T, y, mask, R, F, M, p0
    """
    np.random.seed(seed)
    
    df = pd.read_csv(df_path, parse_dates=['WeekStart'])
    df = df.astype({customer_col: str})
    
    # Subsample if requested
    if n_cust is not None:
        unique_custs = df[customer_col].unique()
        if len(unique_custs) > n_cust:
            selected = np.random.choice(unique_custs, n_cust, replace=False)
            df = df[df[customer_col].isin(selected)]
            print(f"Subsampled to {n_cust} customers: {len(selected)} selected")
        else:
            print(f"Requested {n_cust}, but only {len(unique_custs)} available. Using all.")
    
    # Check panel structure
    panel_sizes = df.groupby(customer_col).size()
    if panel_sizes.nunique() != 1:
        print(f"Warning: Non-rectangular panel. Sizes: {panel_sizes.unique()}")
    
    def mat(col):
        """Create matrix from long-format data."""
        return df.pivot(index=customer_col, columns='WeekStart', values=col).values
    
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


# ---------- 5. SMC Runner ----------
def run_smc(data, K, state_specific_p, p_fixed, use_gam, gam_df, 
            draws, chains, seed, out_dir):
    """
    Run SMC estimation and save results.
    """
    cores = min(chains, os.cpu_count() or 1)
    t0 = time.time()
    
    try:
        with make_model(data, K=K, state_specific_p=state_specific_p, 
                       p_fixed=p_fixed, use_gam=use_gam, gam_df=gam_df) as model:
            
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
            
            # Handle nested structure
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
        
        # Build result dict
        res = {
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
            'timestamp': time.strftime('%Y%m%d_%H%M%S')
        }

        # Save
        model_tag = f"K{K}_{'GAM' if use_gam else 'GLM'}"
        p_tag = "statep" if (state_specific_p and K > 1) else \
                "varyingp" if K == 1 else f"p{p_fixed}"
        
        pkl_path = out_dir / f"smc_{model_tag}_{p_tag}_N{data['N']}_D{draws}_C{chains}.pkl"
        
        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'res': res}, f, protocol=4)
        
        size_mb = pkl_path.stat().st_size / (1024**2)
        print(f"  ✓ log_ev={log_ev:.2f}, time={elapsed:.1f}min, size={size_mb:.1f}MB")
        print(f"  Saved: {pkl_path.name}")
        
        return pkl_path, res

    except Exception as e:
        import traceback
        crash_path = out_dir / f"CRASH_K{K}_{int(time.time())}.pkl"
        with open(crash_path, 'wb') as f:
            pickle.dump({'error': str(e), 'trace': traceback.format_exc()}, f)
        print(f"  ✗ CRASH: {str(e)[:60]}")
        raise


# ---------- 6. Main Entry Point ----------
def main():
    parser = argparse.ArgumentParser(
        description='Unified HMM/Static Tweedie Model with GAM/GLM options',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--dataset', required=True, choices=['uci', 'cdnow'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing data files')
    parser.add_argument('--n_cust', type=int, default=None,
                       help='Number of customers to subsample (None=all)')
    
    # Model structure
    parser.add_argument('--K', type=int, default=3,
                       help='Number of states (1=static, >=2=HMM)')
    parser.add_argument('--state_specific_p', action='store_true',
                       help='Use state-specific p (only for K>=2)')
    parser.add_argument('--p_fixed', type=float, default=1.5,
                       help='Fixed p value (used if --state_specific_p not set)')
    
    # GAM/GLM
    parser.add_argument('--no_gam', action='store_true',
                       help='Use GLM instead of GAM (default: GAM)')
    parser.add_argument('--gam_df', type=int, default=3,
                       help='Degrees of freedom for B-splines')
    
    # SMC
    parser.add_argument('--draws', type=int, default=1000,
                       help='Number of SMC particles per chain')
    parser.add_argument('--chains', type=int, default=4,
                       help='Number of independent SMC chains')
    
    # Output
    parser.add_argument('--out_dir', type=str, default='./results',
                       help='Directory for output files')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = data_dir / f"{args.dataset}_full.csv"
    
    print(f"\n{'='*70}")
    print(f"SMC Unified: {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"Model: K={args.K}, {'GLM' if args.no_gam else 'GAM'}, "
          f"p={'state-specific' if args.state_specific_p else 'fixed'}")
    print(f"Data: {data_path}")
    print(f"Output: {out_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")
    
    # Load data
    data = build_panel_data(data_path, n_cust=args.n_cust, seed=args.seed)
    print(f"Loaded: N={data['N']}, T={data['T']}, "
          f"zero_incidence={np.mean(data['y']==0):.1%}\n")
    
    # Run SMC
    pkl_path, res = run_smc(
        data=data,
        K=args.K,
        state_specific_p=args.state_specific_p,
        p_fixed=args.p_fixed,
        use_gam=not args.no_gam,
        gam_df=args.gam_df,
        draws=args.draws,
        chains=args.chains,
        seed=args.seed,
        out_dir=out_dir
    )
    
    print(f"\n{'='*70}")
    print("RESULT")
    print(f"{'='*70}")
    for key, val in res.items():
        print(f"  {key}: {val}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
