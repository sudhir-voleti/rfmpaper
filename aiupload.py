#!/usr/bin/env python3
"""
Full-panel SMC for Table-4 model selection: K=2,3,4 with *state-specific* dispersion ϕ.
Produces: log-evidence for each K.
"""
import argparse, os, time, pathlib, pickle, pandas as pd, numpy as np
import pymc as pm, pytensor.tensor as pt, arviz as az
from scipy.special import logsumexp

# ---------- 0.A  Float32
import pytensor
pytensor.config.floatX = 'float32'          # Apple-M1 friendly speed-up

# ---------- 0.B  Deterministic Gamma log-density ----------
def gamma_logp_det(value, mu, phi):
    alpha = mu / phi
    beta  = 1.0 / phi
    return (alpha - 1)*pt.log(value) - value*beta + alpha*pt.log(beta) - pt.gammaln(alpha)

# ---------- 1. make_model (WITH p_fixed) ----------
def make_model(data, K=2, p_fixed=1.5):
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]

    with pm.Model(coords={"state": np.arange(K),
                          "customer": np.arange(N),
                          "week": np.arange(T)}) as model:
        
        # ---- latent ----
        pi0   = pm.Dirichlet("pi0", a=np.ones(K))
        Gamma = pm.Dirichlet("Gamma", a=np.ones(K), shape=(K,K))

        # ---- ORDERED beta0 ----
        beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
        beta0     = pm.Deterministic("beta0", pt.sort(beta0_raw))

        # ---- ORDERED phi ----
        phi_raw = pm.Exponential("phi_raw", lam=10.0, shape=K)
        phi     = pm.Deterministic("phi", pt.sort(phi_raw))

        # ---- coefficients ----
        betaR = pm.Normal("betaR", 0, 1, shape=K)
        betaF = pm.Normal("betaF", 0, 1, shape=K)
        betaM = pm.Normal("betaM", 0, 1, shape=K)

        # ---- state-varying mean μ ----
        mu = pt.exp(beta0 + betaR*R[...,None] + betaF*F[...,None] + betaM*M[...,None])
        mu = pt.clip(mu, 1e-3, 1e6)

        # ---- ZIG emission WITH p_fixed ----
        # psi = exp(-mu^(2-p) / (phi * (2-p)))
        psi = pt.exp(-pt.pow(mu, 2 - p_fixed) / (phi * (2 - p_fixed)))
        psi = pt.clip(psi, 1e-12, 1 - 1e-12)
        
        log_zero = pt.log(psi)
        y_exp = y[...,None]
        log_pos  = pt.log1p(-psi) + gamma_logp_det(y_exp, mu, phi)
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
    

# ---------- 2. Generic panel builder ----------
def build_panel_data(df_path, customer_col='customer_id'):
    df = pd.read_csv(df_path, parse_dates=['WeekStart'])
    df = df.astype({customer_col: str})
    assert df.groupby(customer_col).size().nunique() == 1, "Panel must be rectangular"
    def mat(col): return df.pivot(index=customer_col, columns='WeekStart', values=col).values
    return dict(
        N = df[customer_col].nunique(),
        T = df.groupby(customer_col).size().iloc[0],
        y = mat('WeeklySpend'),
        mask = ~np.isnan(mat('WeeklySpend')),
        R = mat('R_weeks'), F = mat('F_run'), M = mat('M_run'),
        p0 = mat('p0_cust')
    )


# ---------- 3. Run & record (WITH p_fixed) ----------
def run_smc(data, K, p_fixed, draws, chains, seed, out_dir):
    cores = min(chains, os.cpu_count() or 1)
    t0 = time.time()
    
    try:
        with make_model(data, K, p_fixed=p_fixed) as model:
            print(f"  Starting SMC: K={K}, p={p_fixed}, N={data['N']}, draws={draws}")
            idata = pm.sample_smc(draws=draws, chains=chains, cores=cores, 
                                 random_seed=seed, return_inferencedata=True)

        # ---- robust log-evidence extraction ----
        try:
            # Handle irregular shapes from SMC
            lm = idata.sample_stats.log_marginal_likelihood
            # Flatten to list, then to float array
            lm_list = lm.values.tolist() if hasattr(lm, 'values') else lm.tolist()
            flat = []
            def flatten(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    for item in x: flatten(item)
                else:
                    try: flat.append(float(x))
                    except: pass
            flatten(lm_list)
            log_ev = float(np.mean(flat)) if flat else np.nan
        except Exception as e:
            print(f"  Warning: log-ev extraction failed: {e}")
            log_ev = np.nan

        res = dict(K=K, p=p_fixed, N=data['N'], log_evidence=log_ev)

        # Save
        pkl_path = out_dir / f"smc_p{p_fixed}_K{K}_N{data['N']}_D{draws}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'res': res}, f, protocol=4)

        print(f"  ✓ K={K}, p={p_fixed}: log-ev={log_ev:.2f}, time={(time.time()-t0)/60:.1f}min")
        return pkl_path

    except Exception as e:
        import traceback
        crash_path = out_dir / f"CRASH_p{p_fixed}_K{K}_{int(time.time())}.pkl"
        with open(crash_path, 'wb') as f:
            pickle.dump({'error': str(e), 'trace': traceback.format_exc()}, f)
        print(f"  ❌ CRASH: p={p_fixed}, K={K}: {str(e)[:60]}")
        raise
        

ROOT = pathlib.Path("/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/results")
# ---------- 5. p-grid driver (FIXED) ----------
def run_p_grid(dataset, K=2, draws=1000, chains=2, p_grid=None, n_cust=None):
    if p_grid is None:
        p_grid = [1.2, 1.35, 1.5, 1.65, 1.8]
    
    # FIX: Use proper data path
    data_path = DATA_DIR / f"{dataset}_full.csv"
    df = pd.read_csv(data_path, parse_dates=['WeekStart'])
    
    if n_cust is not None:
        cust_ids = df.customer_id.unique()[:n_cust]
        df = df[df.customer_id.isin(cust_ids)]
        print(f"Subset to {n_cust} customers: {len(cust_ids)} found")
    
    # FIX: Build data dict, not pass DataFrame
    data = build_panel_data_from_df(df)
    
    results = []
    for p in p_grid:
        print(f"\n{'='*60}")
        print(f">>> {dataset} | K={K} | p={p} | N={data['N']} <<<")
        print(f"{'='*60}")
        
        # FIX: Pass data dict and p_fixed
        pkl_path = run_smc(data, K=K, p_fixed=p, draws=draws, 
                          chains=chains, seed=42, out_dir=ROOT)
        
        # Read back and extract
        with open(pkl_path, 'rb') as f:
            bundle = pickle.load(f)
            res = bundle['res']
        
        results.append({
            "dataset": dataset, 
            "K": K, 
            "p": p, 
            "log_evidence": res['log_evidence']
        })        
    
    return pd.DataFrame(results)

# Helper (add this too)
def build_panel_data_from_df(df, customer_col='customer_id'):
    """Build data dict from already-loaded DataFrame."""
    df = df.astype({customer_col: str})
    
    # Check rectangular
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
    
# ---- single entry point -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=['uci', 'cdnow'])
    parser.add_argument("--k", nargs='+', type=int, default=[3])
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--p_grid", nargs='+', type=float, default=[1.2, 1.35, 1.5, 1.65, 1.8])
    parser.add_argument("--n_cust", type=int, help="slice to first n customers (None = full)")
    args = parser.parse_args()

    ds = args.dataset
    tbl = run_p_grid(ds, K=args.k[0], draws=args.draws, chains=args.chains,
                     p_grid=args.p_grid, n_cust=args.n_cust)
    print(f"\n{args.p_grid} grid results for {ds}")
    print(tbl)
    tbl.to_csv(f"grid_pilot_{ds}_K{args.k[0]}.csv", index=False)
