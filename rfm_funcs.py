#!/usr/bin/env python
# reusable model & data builder for regime-specific SMC
import numpy as np, pandas as pd, pymc as pm, pytensor.tensor as pt, arviz as az
import patsy
#from pyhmm import GaussianHMM   # pip install pyhmm

def build_uci_data(ids, df):
    df = df[df.CustomerID.isin(ids)].copy()
    df['week_idx'] = (pd.to_datetime(df['Week']).dt.to_period('W').astype(int)) + 1
    df['cust_idx']  = pd.factorize(df.CustomerID)[0]
    df = df.sort_values(['CustomerID','week_idx']).reset_index(drop=True)
    df['seq_idx'] = np.arange(1, len(df)+1)
    start = df.groupby('CustomerID', sort=True).first()['seq_idx'].astype(int).tolist()
    len_  = df.groupby('CustomerID', sort=True).size().astype(int).tolist()
    N = df.shape[0]
    y   = df['M_roll'].values
    cust = df['cust_idx'].values.astype(int)
    # spline bases  (5 knots, drop intercept)
    P = 5
    
    unique_vals = df.M_roll.unique()
    print(f'[debug] unique_vals = {unique_vals}, size = {unique_vals.size}')
    if unique_vals.size < 5:
        knots_R = np.linspace(df.M_roll.min(), df.M_roll.max(), 5)
        print(f'[debug] knots_R (linear) = {knots_R}')
    else:
        try:
            knots_R = np.quantile(unique_vals, np.linspace(0.05, 0.95, 5))
            print(f'[debug] knots_R (quantile) = {knots_R}')
        except Exception:
            knots_R = np.linspace(df.M_roll.min(), df.M_roll.max(), 5)
    knots_R = np.clip(knots_R, df.M_roll.min(), df.M_roll.max())

    unique_freq = df.groupby('CustomerID')['Week'].count().unique()
    print(f'[debug] unique_freq = {unique_freq}, size = {unique_freq.size}')
    if unique_freq.size < 5:
        knots_F = np.linspace(unique_freq.min(), unique_freq.max(), 5)
        print(f'[debug] knots_F (linear) = {knots_F}')
    else:
        try:
            knots_F = np.fromiter((float(item) for sub in np.array(unique_freq, dtype=object) for item in sub), dtype=np.float64)
            print(f'[debug] knots_F (flatten) = {knots_F}')
        except Exception:
            knots_F = np.linspace(unique_freq.min(), unique_freq.max(), 5)
            print(f'[debug] knots_F (fallback) = {knots_F}')
    knots_F = np.clip(knots_F, unique_freq.min(), unique_freq.max())
    print(f'[debug] knots_F (clipped) = {knots_F}')

    X_R = patsy.dmatrix("bs(M_roll, knots=knots_R, degree=3, include_intercept=False)",
                        {"M_roll": df.M_roll.values}, return_type='dataframe').values
    freq_vec = df.groupby('CustomerID')['Week'].count().reindex(df.CustomerID).values
    X_F = patsy.dmatrix("bs(freq, knots=knots_F, degree=3, include_intercept=False)",
                        {"freq": freq_vec}, return_type='dataframe').values
    X_M = X_R.copy()                                    # same basis

    result = {"N": N, "S": len(start), "K": 4, "P": P,
              "y": y, "cust": cust,
              "start": np.array(start, dtype=int),
              "len": np.array(len_, dtype=int),
              "X_R": X_R, "X_F": X_F, "X_M": X_M}
    print("[DEBUG] build_uci_data returning keys:", list(result.keys()))
    print("[DEBUG] y shape:", y.shape, "start:", start[:3], "len:", len_[:3])
    return result

#    return {"N": N, "S": len(start), "K": 4, "P": P,
#            "y": y, "cust": cust,
#            "start": np.array(start, dtype=int),
#            "len": np.array(len_, dtype=int),
#            "X_R": X_R, "X_F": X_F, "X_M": X_M}

def make_model(data_uci, K):
    with pm.Model(coords={"state": np.arange(K)}) as model:
        N, y, cust = data_uci['N'], data_uci['y'], data_uci['cust']
        X_R = pm.Data("X_R", data_uci["X_R"])[:, :4]
        X_F = pm.Data("X_F", data_uci["X_F"])[:, :4]
        X_M = pm.Data("X_M", data_uci["X_M"])[:, :4]

        # ---- Real priors (must stay as-is for sampling) ----
        beta0_raw = pm.Normal("beta0_raw", mu=0, sigma=1, shape=K)
        phi_raw = pm.Exponential("phi_raw", lam=1, shape=K)
        diag_raw = pm.Beta("diag_raw", alpha=5, beta=2, shape=K)

        beta_R_col1_raw = pm.Normal("beta_R_col1_raw", 0, 0.8, shape=K)
        beta_F_col1_raw = pm.Normal("beta_F_col1_raw", 0, 0.8, shape=K)
        beta_M_col1_raw = pm.Normal("beta_M_col1_raw", 0, 0.8, shape=K)

        beta_R_rest = pm.Normal("beta_R_rest", 0, 0.5, shape=(K, 4))
        beta_F_rest = pm.Normal("beta_F_rest", 0, 0.5, shape=(K, 4))
        beta_M_rest = pm.Normal("beta_M_rest", 0, 0.5, shape=(K, 4))

        off_diag = pm.Dirichlet("off_diag", np.ones(K-1), shape=(K, K-1))

        # ---- Deterministics (for transforms and mu) ----
        beta0 = pm.Deterministic("beta0", pt.cumsum(beta0_raw))
        phi = pm.Deterministic("phi", pt.cumsum(phi_raw))
        diag_trans = pm.Deterministic("diag_trans", pt.sort(diag_raw))

        beta_R_col1 = pm.Deterministic("beta_R_col1", pt.cumsum(beta_R_col1_raw))
        beta_F_col1 = pm.Deterministic("beta_F_col1", pt.cumsum(beta_F_col1_raw))
        beta_M_col1 = pm.Deterministic("beta_M_col1", pt.cumsum(beta_M_col1_raw))

        # --- Correct: no [:, :4] — keep full (K, 5) vector ---
        # inside make_model, after building beta_R etc.
        beta_R = pm.Deterministic("beta_R", pt.concatenate([beta_R_col1[:, None], beta_R_rest], axis=1)[:, :4])
        beta_F = pm.Deterministic("beta_F", pt.concatenate([beta_F_col1[:, None], beta_F_rest], axis=1)[:, :4])
        beta_M = pm.Deterministic("beta_M", pt.concatenate([beta_M_col1[:, None], beta_M_rest], axis=1)[:, :4])

        mu = pm.Deterministic("mu", pt.exp(beta0 + pt.dot(X_R, beta_R.T) +
                                           pt.dot(X_F, beta_F.T) + pt.dot(X_M, beta_M.T)))

        trans = pm.Deterministic("trans", pt.concatenate([off_diag, diag_trans[:, None]], axis=1))

        # ---- Likelihood via Potential ----
        nz = y > 0
        phi_nz = pt.clip(phi[cust[nz]], 1e-3, 1e6)
        nz_idx = pt.arange(N)[nz]
        cust_nz = pt.clip(cust[nz], 0, K - 1) 
        mu_nz = pt.clip(mu[nz_idx, cust_nz], 1e-3, 1e6)
        beta_nz = phi_nz / mu_nz
        log_gamma = pm.logp(pm.Gamma.dist(alpha=phi_nz, beta=beta_nz), y[nz])
        log_spike = pm.logp(pm.DiracDelta.dist(0), y[~nz])
        spend = pm.Potential('spend', log_gamma.sum() + log_spike.sum())

    return model



def rebuild_deterministics(model, raw_point):
    """Rebuild deterministics for logp evaluation — correct shapes."""
    full_point = raw_point.copy()

    # Monotonic transforms
    full_point['beta0'] = np.cumsum(full_point['beta0_raw'])
    full_point['phi']   = np.cumsum(full_point['phi_raw'])
    diag = full_point['diag_raw']
    full_point['diag_trans'] = np.sort(diag) if diag.ndim > 0 else diag

    # ---------- posterior mean of TRANSFORMED params ----------
    beta_R_col1 = np.cumsum(full_point['beta_R_col1_raw'])
    beta_F_col1 = np.cumsum(full_point['beta_F_col1_raw'])
    beta_M_col1 = np.cumsum(full_point['beta_M_col1_raw'])

    beta_R_rest = np.atleast_1d(np.squeeze(full_point['beta_R_rest']))
    beta_F_rest = np.atleast_1d(np.squeeze(full_point['beta_F_rest']))
    beta_M_rest = np.atleast_1d(np.squeeze(full_point['beta_M_rest']))

    full_point['beta_R'] = np.concatenate([beta_R_col1[:, None], beta_R_rest], axis=1)[:, :4]
    full_point['beta_F'] = np.concatenate([beta_F_col1[:, None], beta_F_rest], axis=1)[:, :4]
    full_point['beta_M'] = np.concatenate([beta_M_col1[:, None], beta_M_rest], axis=1)[:, :4]

    off_diag = np.atleast_2d(full_point['off_diag'])
    diag_col = np.atleast_1d(full_point['diag_trans'])[:, None]
    full_point['trans'] = np.concatenate([off_diag, diag_col], axis=1)
    full_point['trans'] /= full_point['trans'].sum(axis=1, keepdims=True)  
  
    # mu rebuild
    X_R = model['X_R'].get_value()[:, :4]
    X_F = model['X_F'].get_value()[:, :4]
    X_M = model['X_M'].get_value()[:, :4]

    full_point['mu'] = np.exp(
        full_point['beta0'] +
        np.dot(X_R, full_point['beta_R'].T) +
        np.dot(X_F, full_point['beta_F'].T) +
        np.dot(X_M, full_point['beta_M'].T)
    )

    return full_point

import numpy as np, pandas as pd
from patsy import dmatrix

def partial_dependence_recency(posterior, data_uci, n_grid=50, n_draws=1000, model=None):
    """
    State-specific partial dependence of log-recency on expected weekly spend.
    Holds F & M at sample means.
    
    posterior : dict  – thinned posterior (already flattened)
    data_uci  : dict  – output of build_uci_data()
    n_grid    : int   – number of log-recency evaluation points
    n_draws   : int   – how many posterior draws to use (randomly sampled)
    
    Returns
    -------
    pd.DataFrame with columns
        state, log_recency, mean, cri_lower, cri_upper
    """
    K = 3
    draws_total = posterior['beta0_raw'].shape[0]
    idx = np.random.choice(draws_total, size=min(n_draws, draws_total), replace=False)
    
    # ---- 1.  grid for log-recency (observed range) ----
    R_obs = np.log(data_uci['y'] + 1)          # quick proxy; replace with real R_cl if avail
    r_seq = np.linspace(R_obs.min(), R_obs.max(), n_grid)
    
    # ---- 2.  helper matrix: hold F & M at means ----
    F_mean = data_uci['X_F'][:, 0].mean()
    M_mean = data_uci['X_M'][:, 0].mean()
    
    # ---- 3.  build design matrix for each (r, k) ----
    designs = {}
    for k in range(K):
        # dummy frame: same length as grid, recency = r_seq, F=M=mean
        dummy = pd.DataFrame({'R': r_seq, 'F': F_mean, 'M': M_mean})
        knots_r = np.linspace(dummy.R.min(), dummy.R.max(), 5)   # 5 interior knots
        formula = f"bs(R, knots={knots_r.tolist()}, degree=3, include_intercept=False) + F + M"
        designs[k] = dmatrix(formula, data=dummy, return_type='dataframe').values[:, :4]  # drop int
        
    # ---- 4.  allocate results ----
    mu_grid = np.full((n_grid, K, len(idx)), np.nan)
    
    # ---- 5.  loop over sampled draws ----
    for m, draw_idx in enumerate(idx):
        point = {k: np.atleast_1d(v[draw_idx]) for k, v in posterior.items()}
        full  = rebuild_deterministics(model, raw_point=point)  # model not needed
        
        for k in range(K):
            eta = full['beta0'][k] + designs[k] @ full['beta_R'][k]  # (n_grid,)
            mu_grid[:, k, m] = np.exp(eta)
    
    # ---- 6.  summarise across draws ----
    out = pd.DataFrame({
        'state': np.tile(['Engaged', 'Cooling', 'Churned'], n_grid),
        'log_recency': np.repeat(r_seq, K),
        'mean': mu_grid.mean(axis=2).ravel(),
        'cri_lower': np.percentile(mu_grid, 2.5, axis=2).ravel(),
        'cri_upper': np.percentile(mu_grid, 97.5, axis=2).ravel()
    })
    return out


