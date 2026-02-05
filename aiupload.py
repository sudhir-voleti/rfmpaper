#!/usr/bin/env python3
"""
SMC State-Specific p Pilot: HMM-Tweedie with per-state power parameter.
Tests whether Cold/Warm/Hot states have different clumpiness (p) profiles.
"""
import argparse, os, time, pathlib, pickle, pandas as pd, numpy as np
import pymc as pm, pytensor.tensor as pt, arviz as az
from scipy.special import logsumexp

# ---------- 0.A  Float32 ----------
import pytensor
pytensor.config.floatX = 'float32'

# ---------- 0.B  ROOT PATHS ----------
ROOT = pathlib.Path("/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/results")
ROOT.mkdir(parents=True, exist_ok=True)
DATA_DIR = pathlib.Path("/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/data")

# ---------- 0.C  Deterministic Gamma log-density ----------
def gamma_logp_det(value, mu, phi):
    alpha = mu / phi
    beta = 1.0 / phi
    return (alpha - 1)*pt.log(value) - value*beta + alpha*pt.log(beta) - pt.gammaln(alpha)

# ---------- 1. make_model (STATE-SPECIFIC p VERSION) ----------
def make_model(data, K=2, state_specific_p=True, p_fixed=1.5):
    """
    Build HMM-Tweedie-GAM with state-specific p.
    
    state_specific_p=True: Each state k has its own p_k (ordered: Cold < Warm < Hot)
    state_specific_p=False: Global fixed p (original behavior)
    """
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]

    with pm.Model(coords={"state": np.arange(K),
                          "customer": np.arange(N),
                          "week": np.arange(T)}) as model:
        
        # ---- latent ----
        pi0 = pm.Dirichlet("pi0", a=np.ones(K))
        Gamma = pm.Dirichlet("Gamma", a=np.ones(K), shape=(K,K))

        # ---- ORDERED beta0 (Cold < Warm < Hot) ----
        beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
        beta0 = pm.Deterministic("beta0", pt.sort(beta0_raw))

        # ---- ORDERED phi (Cold < Warm < Hot) ----
        phi_raw = pm.Exponential("phi_raw", lam=10.0, shape=K)
        phi = pm.Deterministic("phi", pt.sort(phi_raw))

        # ---- coefficients ----
        betaR = pm.Normal("betaR", 0, 1, shape=K)
        betaF = pm.Normal("betaF", 0, 1, shape=K)
        betaM = pm.Normal("betaM", 0, 1, shape=K)

        # ---- STATE-SPECIFIC p (NEW) ----
        if state_specific_p:
            # Ordered p: Cold (more clumpy) < Warm < Hot (less clumpy)
            # Transform: Beta(2,2) -> [0,1] -> [1.1, 1.9]
            p_raw = pm.Beta("p_raw", alpha=2, beta=2, shape=K)
            p_sorted = pt.sort(p_raw)  # ascending: p[0] < p[1] < ...
            p = pm.Deterministic("p", 1.1 + p_sorted * 0.8)  # [1.1, 1.9]
        else:
            # Global fixed p (original behavior)
            p = pt.as_tensor_variable([p_fixed] * K)  # same p for all states

        # ---- state-varying mean μ (N,T,K) ----
        mu = pt.exp(beta0 + betaR*R[...,None] + betaF*F[...,None] + betaM*M[...,None])
        mu = pt.clip(mu, 1e-3, 1e6)

        # ---- ZIG emission with STATE-SPECIFIC p ----
        # psi[n,t,k] = f(mu[n,t,k], phi[k], p[k])
        # Shape: (N, T, K) for each parameter
        
        # Expand p to (1, 1, K) for broadcasting
        p_expanded = p[None, None, :]  # (1, 1, K)
        phi_expanded = phi[None, None, :]  # (1, 1, K)
        
        # Compute psi with state-specific p
        # psi = exp(-mu^(2-p) / (phi * (2-p)))
        exponent = 2.0 - p_expanded
        numerator = pt.pow(mu, exponent)
        denominator = phi_expanded * exponent
        psi = pt.exp(-numerator / denominator)
        psi = pt.clip(psi, 1e-12, 1 - 1e-12)

        log_zero = pt.log(psi)
        y_exp = y[...,None]
        log_pos = pt.log1p(-psi) + gamma_logp_det(y_exp, mu, phi_expanded)
        log_emission = pt.where(y_exp==0, log_zero, log_pos)
        log_emission = pt.where(mask[:,:,None], log_emission, 0.0)

        # ---- forward algorithm ----
        log_alpha = pt.log(pi0) + log_emission[:,0,:]
        log_Gamma = pt.log(Gamma)[None,:,:]
        for t in range(1, T):
            temp = log_alpha[:,:,None] + log_Gamma
            log_alpha = log_emission[:,t,:] + pt.logsumexp(temp, axis=1)

        # ---- marginal log-lik ----
        logp_cust = pt.logsumexp(log_alpha, axis=1)
        pm.Potential('loglike', logp_cust.sum())

        # point-wise log-lik
        pm.Deterministic('log_likelihood', logp_cust[:, None] * mask,
                        dims=('customer', 'week'))

    return model


# ---------- 2. Panel builder ----------
def build_panel_data(df_path, customer_col='customer_id', n_cust=None):
    """Build data dict from CSV, optionally subsetting."""
    df = pd.read_csv(df_path, parse_dates=['WeekStart'])
    df = df.astype({customer_col: str})
    
    if n_cust is not None:
        unique_custs = df[customer_col].unique()
        selected = unique_custs[:n_cust]
        df = df[df[customer_col].isin(selected)]
        print(f"Subset to {n_cust} customers: {len(selected)} found")
    
    panel_sizes = df.groupby(customer_col).size()
    if panel_sizes.nunique() != 1:
        print(f"Warning: Non-rectangular panel. Sizes: {panel_sizes.unique()}")
    
    def mat(col): 
        return df.pivot(index=customer_col, columns='WeekStart', values=col).values
    
    return dict(
        N = df[customer_col].nunique(),
        T = panel_sizes.iloc[0],
        y = mat('WeeklySpend'),
        mask = ~np.isnan(mat('WeeklySpend')),
        R = mat('R_weeks'), 
        F = mat('F_run'), 
        M = mat('M_run'),
        p0 = mat('p0_cust')
    )


# ---------- 3. Run SMC (unchanged from before) ----------
def run_smc(data, K, state_specific_p, p_fixed, draws, chains, seed, out_dir):
    cores = min(chains, os.cpu_count() or 1)
    t0 = time.time()
    
    try:
        with make_model(data, K=K, state_specific_p=state_specific_p, p_fixed=p_fixed) as model:
            print(f"  Starting SMC: K={K}, state_specific_p={state_specific_p}, "
                  f"p_fixed={p_fixed if not state_specific_p else 'N/A'}, "
                  f"N={data['N']}, draws={draws}, chains={chains}")
            
            idata = pm.sample_smc(
                draws=draws, 
                chains=chains, 
                cores=cores, 
                random_seed=seed,
                return_inferencedata=True,
                threshold=0.8
            )

        # ---- robust log-evidence extraction ----
        log_ev = np.nan
        try:
            lm = idata.sample_stats.log_marginal_likelihood.values
            
            # Handle nested list structure from SMC
            if isinstance(lm, np.ndarray) and lm.dtype == object:
                # Extract last valid value from each chain
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
                    elif isinstance(chain_list, (int, float, np.floating)):
                        if np.isfinite(chain_list):
                            chain_vals.append(float(chain_list))
                
                log_ev = float(np.mean(chain_vals)) if chain_vals else np.nan
            else:
                # Simple array
                flat = np.array(lm).flatten()
                valid = flat[np.isfinite(flat)]
                log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
                
        except Exception as e:
            print(f"  Warning: log-ev extraction failed: {e}")
            log_ev = np.nan

        res = dict(
            K=K, 
            state_specific_p=state_specific_p,
            p_fixed=p_fixed if not state_specific_p else None,
            N=data['N'],
            log_evidence=log_ev
        )

        # Save
        mode = "statep" if state_specific_p else f"p{p_fixed}"
        pkl_path = out_dir / f"smc_{mode}_K{K}_N{data['N']}_D{draws}_C{chains}.pkl"
        
        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'res': res}, f, protocol=4)

        size_mb = pkl_path.stat().st_size / (1024**2)
        print(f"  ✓ log_ev={log_ev:.2f}, time={(time.time()-t0)/60:.1f}min, "
              f"saved {size_mb:.1f}MB")
        return pkl_path

    except Exception as e:
        import traceback
        crash_path = out_dir / f"CRASH_statep_K{K}_{int(time.time())}.pkl"
        with open(crash_path, 'wb') as f:
            pickle.dump({'error': str(e), 'trace': traceback.format_exc()}, f)
        print(f"  ❌ CRASH: {str(e)[:60]}")
        raise


# ---------- 4. Comparison driver ----------
def run_comparison(dataset, K=3, draws=500, chains=4, n_cust=200, seed=42):
    """
    Compare state-specific p vs. fixed p (at various values).
    """
    data_path = DATA_DIR / f"{dataset}_full.csv"
    print(f"\nLoading data from {data_path}")
    
    data = build_panel_data(data_path, n_cust=n_cust)
    print(f"Panel: N={data['N']}, T={data['T']}, zeros={np.mean(data['y']==0):.1%}")
    
    results = []
    
    # Test 1: State-specific p
    print(f"\n{'='*60}")
    print(f">>> {dataset.upper()} | K={K} | STATE-SPECIFIC p <<<")
    print(f"{'='*60}")
    try:
        pkl_path = run_smc(data, K=K, state_specific_p=True, p_fixed=None,
                          draws=draws, chains=chains, seed=seed, out_dir=ROOT)
        with open(pkl_path, 'rb') as f:
            res = pickle.load(f)['res']
        results.append(res)
    except Exception as e:
        print(f"State-specific p failed: {e}")
        results.append(dict(K=K, state_specific_p=True, log_evidence=np.nan, error=str(e)))
    
    # Test 2-4: Fixed p at grid values
    for p_fixed in [1.35, 1.5, 1.65]:
        print(f"\n{'='*60}")
        print(f">>> {dataset.upper()} | K={K} | FIXED p={p_fixed} <<<")
        print(f"{'='*60}")
        try:
            pkl_path = run_smc(data, K=K, state_specific_p=False, p_fixed=p_fixed,
                              draws=draws, chains=chains, seed=seed, out_dir=ROOT)
            with open(pkl_path, 'rb') as f:
                res = pickle.load(f)['res']
            results.append(res)
        except Exception as e:
            print(f"Fixed p={p_fixed} failed: {e}")
            results.append(dict(K=K, state_specific_p=False, p_fixed=p_fixed, 
                              log_evidence=np.nan, error=str(e)))
    
    # Summary
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(df[['K', 'state_specific_p', 'p_fixed', 'log_evidence']].to_string())
    
    # Find best
    valid = df[df['log_evidence'].notna()]
    if len(valid) > 0:
        best = valid.loc[valid['log_evidence'].idxmax()]
        if best['state_specific_p']:
            print(f"\n>>> BEST: State-specific p, log_ev={best['log_evidence']:.2f} <<<")
        else:
            print(f"\n>>> BEST: Fixed p={best['p_fixed']}, log_ev={best['log_evidence']:.2f} <<<")
    
    # Save
    csv_path = ROOT / f"comparison_statep_{dataset}_K{K}_N{data['N']}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    return df


# ---------- 5. Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State-Specific p Pilot")
    parser.add_argument("--dataset", required=True, choices=['uci', 'cdnow'])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--n_cust", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    run_comparison(
        dataset=args.dataset,
        K=args.k,
        draws=args.draws,
        chains=args.chains,
        n_cust=args.n_cust,
        seed=args.seed
    )
