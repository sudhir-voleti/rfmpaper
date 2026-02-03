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

# ---------- 1. make_model ----------
def make_model(data, K=2):
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]

    with pm.Model(coords={"state": np.arange(K),
                      "customer": np.arange(N),
                      "week": np.arange(T)}) as model:
        #with pm.Model() as model:
        # ---- latent ----
        pi0   = pm.Dirichlet("pi0", a=np.ones(K))
        Gamma = pm.Dirichlet("Gamma", a=np.ones(K), shape=(K,K))

        # ---- ORDERED beta0 (Hot > Warm > Cold) ----
        beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
        beta0     = pm.Deterministic("beta0", pt.sort(beta0_raw))   # ascending → state 0=Cold, K-1=Hot

        # ---- ORDERED phi (Cold ≤ Warm ≤ Hot) ----
        phi_raw = pm.Exponential("phi_raw", lam=10.0, shape=K)
        phi     = pm.Deterministic("phi", pt.sort(phi_raw))

        # ---- other coefficients (unordered) ----
        betaR = pm.Normal("betaR", 0, 1, shape=K)
        betaF = pm.Normal("betaF", 0, 1, shape=K)
        betaM = pm.Normal("betaM", 0, 1, shape=K)

        # ---- state-varying mean μ  (N,T,K) ----
        mu = pt.exp(beta0 + betaR*R[...,None] + betaF*F[...,None] + betaM*M[...,None])
        mu = pt.clip(mu, 1e-3, 1e6)

        # ---- ZIG emission ----
        psi = pt.exp(-mu / phi)
        log_zero = pt.log(psi + 1e-12)
        y_exp = y[...,None]
        log_pos  = pt.log1p(-psi + 1e-12) + gamma_logp_det(y_exp, mu, phi)
        log_emission = pt.where(y_exp==0, log_zero, log_pos)
        log_emission = pt.where(mask[:,:,None], log_emission, 0.0)

        # ---- vectorised forward (no scan, no rng) ----
        log_alpha = pt.log(pi0) + log_emission[:,0,:]
        log_Gamma = pt.log(Gamma)[None,:,:]
        for t in range(1, T):
            temp = log_alpha[:,:,None] + log_Gamma
            log_alpha = log_emission[:,t,:] + pt.logsumexp(temp, axis=1)

        # ---- marginal log-lik ----
        logp_cust = pt.logsumexp(log_alpha, axis=1)          # (N,)
        pm.Potential('loglike', logp_cust.sum())

        # point-wise log-lik for LOO/WAIC  (N,T)  <-- NEW
        logp_obs = pm.Deterministic('log_likelihood',
                                logp_cust[:, None] * mask,   # broadcast to (N,T)
                                dims=('customer', 'week'))


    return model


# ---------- 1. Generic panel builder ----------
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


# ---------- 3. Run & record ----------
def run_smc(data, K, draws, chains, seed, out_dir):
    import time, traceback, pathlib, pickle, numpy as np
    from scipy.special import logsumexp
    import arviz as az

    cores = min(chains, os.cpu_count() or 1)
    t0 = time.time()
    try:
        with make_model(data, K) as model:
            idata = pm.sample_smc(draws=draws, chains=chains, cores=cores, random_seed=seed)

        
        # ---- metrics ----
        log_marg = np.concatenate([np.array(lst, dtype=float) for lst in
                                   idata.sample_stats['log_marginal_likelihood'].values.flat])
        log_ev   = float(logsumexp(log_marg[~np.isnan(log_marg)]) -
                         np.log((~np.isnan(log_marg)).sum()))
        # LOO/WAIC omitted – log-evidence is decisive
        res = dict(K=K, log_evidence=log_ev)

        pkl_path = out_dir / f"smc_full_{data['N']}cust_K{K}_D{draws}_C{chains}.pkl"
        try:
            posterior_ds = xr.Dataset({k: (['chain','draw'], v.data) for k,v in idata.posterior.items()})
            stats_ds     = xr.Dataset({k: (['chain','draw'], v.data) for k,v in idata.sample_stats.items()})
            save_idata   = az.InferenceData(posterior=posterior_ds, sample_stats=stats_ds)
        except Exception:
            save_idata = idata   # fallback raw dict

        with open(pkl_path, 'wb') as f:
            pickle.dump(dict(idata=save_idata, res=res), f)

        print(f"K={K}  log-ev={log_ev:.3f}  time={(time.time()-t0)/60:.1f}min  -> {pkl_path.name}")
        return res


    except Exception as e:
        import traceback, pathlib, time, pickle
        crash_path = out_dir / f"CRASH_K{K}_{int(time.time())}.pkl"
        with open(crash_path, 'wb') as f:
            pickle.dump({'error': str(e), 'trace': traceback.format_exc(),
                         'idata': locals().get('idata', None)}, f)
        print(f"❌ K={K} crashed – dump saved to {crash_path.name}")
        raise   # re-raise to stop script


# ---------- 5.  p-grid driver (discrete search) -----------------------------
ROOT = pathlib.Path("/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/results")
def run_p_grid(dataset, K=2, draws=1000, chains=2, p_grid=None, n_cust=None):
    if p_grid is None:
        p_grid = [1.2, 1.35, 1.5, 1.65, 1.8]
    df = pd.read_csv(f"data/{dataset}_full.csv")   # full panel
    if n_cust is not None:
        df = df[df.customer_id.isin(df.customer_id.unique()[:n_cust])]
    results = []
    for p in p_grid:
        print(f"\n>>> Pilot {dataset}  K={K}  p={p}  <<<\n")
        # modify make_model to accept p_fixed
        pkl_path = run_smc(df, K=K, draws=draws, chains=chains, seed=42, out_dir=ROOT)
        with open(pkl_path, 'rb') as f:
            bundle = pickle.load(f)
            idata  = bundle['idata']   # InferenceData
            res    = bundle['res']     # dict with N, T, log_evidence
        log_marg = np.concatenate([np.array(lst, dtype=float) for lst in
                                   idata.sample_stats['log_marginal_likelihood'].values.flat])
        log_ev = float(logsumexp(log_marg[~np.isnan(log_marg)]) - np.log((~np.isnan(log_marg)).sum()))
        results.append({"dataset": dataset, "K": K, "p": p, "log_evidence": log_ev})        
    return pd.DataFrame(results)

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
