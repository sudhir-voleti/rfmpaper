#!/usr/bin/env python3
"""
Full-panel SMC for Table-4 model selection: K=2,3,4 with *state-specific* dispersion ϕ.
Produces: log-evidence for each K.
"""
import argparse, os, time, pathlib, pickle, pandas as pd, numpy as np
import pymc as pm, pytensor.tensor as pt, arviz as az
from scipy.special import logsumexp

# ---------- 0. Deterministic Gamma log-density ----------
def gamma_logp_det(value, mu, phi):
    alpha = mu / phi
    beta  = 1.0 / phi
    return (alpha - 1)*pt.log(value) - value*beta + alpha*pt.log(beta) - pt.gammaln(alpha)

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

# ---------- 2. HMM-Tweedie-SMC model ----------
def make_model(data, K):
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]

    with pm.Model(coords={"state": np.arange(K)}) as model:
        # latent
        pi0   = pm.Dirichlet("pi0", a=np.ones(K))
        Gamma = pm.Dirichlet("Gamma", a=np.ones(K), shape=(K,K))

        # state-varying coefficients & dispersion
        beta0 = pm.Normal("beta0", 0, 1, dims="state")
        betaR = pm.Normal("betaR", 0, 1, dims="state")
        betaF = pm.Normal("betaF", 0, 1, dims="state")
        betaM = pm.Normal("betaM", 0, 1, dims="state")
        phi   = pm.Exponential("phi", lam=1.0, dims="state")   # ← free ϕ per state

        mu = pt.exp(beta0 + betaR*R[...,None] + betaF*F[...,None] + betaM*M[...,None])
        mu = pt.clip(mu, 1e-3, 1e6)

        # -- pre-compute log transition matrix --
        log_Gamma = pt.log(Gamma)          # <-- add this line

        # -- ZIG emission log-lik --
        psi = pt.exp(-mu / phi)
        log_zero = pt.log(psi + 1e-12)
        y_exp = y[...,None]
        log_pos  = pt.log1p(-psi + 1e-12) + gamma_logp_det(y_exp, mu, phi)
        log_emission = pt.where(y_exp==0, log_zero, log_pos)
        log_emission = pt.where(mask[:,:,None], log_emission, 0.0)

        # -- vectorised forward (no scan, no rng) --
        log_alpha = pt.log(pi0) + log_emission[:,0,:]
        log_Gamma = log_Gamma[None,:,:]
        for t in range(1, T):                      # Python loop – no RNG inside
            temp = log_alpha[:,:,None] + log_Gamma
            log_alpha = log_emission[:,t,:] + pt.logsumexp(temp, axis=1)

        # -- Drop LOO/WAIC --
        logp_cust = pt.logsumexp(log_alpha, axis=1)              # (N,)
        # replicate over weeks -> (N,T)  (broadcast OK)
        logp_obs = pm.Deterministic('log_likelihood', logp_cust[:,None] * mask)
        # scalarPotential still needed for SMC
        pm.Potential('loglike', logp_obs.sum())

    return model

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



# ---------- 4. CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=['uci', 'cdnow'])
    parser.add_argument("--k", nargs='+', type=int, default=[2,3,4])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    #data_path = f"data/{args.dataset}_full.csv"        # <-- your full panel file
    data_path = f"data/pilot_{args.dataset}_50.csv"        # <-- your full panel file
    out_dir   = pathlib.Path(f"results/full/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_panel_data(data_path)
    print(f"Loaded {args.dataset} full panel: {data['N']} customers, {data['T']} weeks")

    results = []
    for k in args.k:
        results.append(run_smc(data, k, args.draws, args.chains, args.seed, out_dir))

    # ---- quick Table-4 console ----
    df_tbl = pd.DataFrame(results).set_index('K')
    print("\nTable-4 summary")
    print(df_tbl[['log_evidence']].round(3))


# smc_pilot.py  (bottom)
def post_run(idata, res):
    """
    Generate Tables 4-8 from saved InferenceData + metrics dict.
    Returns dict of DataFrames ready for LaTeX.
    """
    tbl4 = pd.DataFrame({'K': res['K'], 'log_evidence': res['log_evidence']}, index=[0])
    tbl5 = az.summary(idata.posterior['Gamma'], hdi_prob=0.95)
    tbl6 = az.summary(idata, var_names=['beta0','phi'], hdi_prob=0.95)
    tbl7 = dict(log_pi0  = idata.posterior['pi0'].mean(dim=('chain','draw')).values,
                log_Gamma= idata.posterior['Gamma'].mean(dim=('chain','draw')).values)
    tbl8 = idata.posterior['betaR'].mean(dim=('chain','draw'))  # GAM slopes
    return {'tbl4': tbl4, 'tbl5': tbl5, 'tbl6': tbl6, 'tbl7': tbl7, 'tbl8': tbl8}
