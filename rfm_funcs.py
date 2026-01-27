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
        full  = rebuild_deterministics(model, point)  
        
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

# ------------------------------------------------------------------
# Figure 5: ROI menu bar chart (UCI & CDNOW, +5 pp retention lift)
# ------------------------------------------------------------------
import pandas as pd, numpy as np, requests, pickle, seaborn as sns, matplotlib.pyplot as plt

def roi_menu_csv(dataset, pkl_path, csv_out, n_draws_wanted=1000, cost_ratio=0.2, lift_pp=5):
    """
    Simulate +5 pp Engaged-state retention lift → ROI posterior
    dataset : str – 'UCI' or 'CDNOW'
    """
    # 1. load posterior
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    post_full = {k: v.reshape(-1, *v.shape[2:]) for k, v in raw['posterior'].items()}
    thin = max(1, post_full['beta0_raw'].shape[0] // n_draws_wanted)
    post = {k: v[::thin] for k, v in post_full.items()}

    # 2. load data & model
    if dataset == 'UCI':
        df = pd.read_csv('/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/uci_full_panel_regime.csv')
    else:  # CDNOW
        df = pd.read_csv('/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/CDNOW/cdnow_subsample_180k.csv')
    data = build_uci_data(df.CustomerID.unique(), df)

    # 3. allocate
    K = 3
    n_draws = post['beta0_raw'].shape[0]
    clv_ctrl = np.full((K, n_draws), np.nan)
    clv_treat = np.full((K, n_draws), np.nan)

    with make_model(data, K=K) as model:
        for i in range(n_draws):
            point = {k: np.atleast_1d(v[i]) for k, v in post.items()}
            full = rebuild_deterministics(model, point)

            # baseline 52-week CLV per state
            base_p = full['trans']  # (K,K)
            base_mu = full['mu'].mean(axis=0)  # (K,)
            clv_ctrl[:, i] = base_mu / (1 - base_p.diagonal())  # perpetuity approx

            # treated: +5 pp self-transition
            treat_p = base_p.copy()
            treat_p[2, 2] += lift_pp / 100  # Engaged state only
            treat_p[2, :] /= treat_p[2, :].sum()  # re-normalise row
            clv_treat[:, i] = base_mu / (1 - treat_p.diagonal())

    # 4. ROI posterior
    roi_post = (clv_treat - clv_ctrl) / (cost_ratio * clv_ctrl)  # cost = 20 % baseline
    out = pd.DataFrame({
        'dataset': dataset,
        'state': ['Engaged', 'Cooling', 'Churned'],
        'mean_roi': roi_post.mean(axis=1),
        'cri_lower': np.percentile(roi_post, 5, axis=1),
        'cri_upper': np.percentile(roi_post, 95, axis=1)
    })
    out.to_csv(csv_out, index=False)
    print(f'[OK] {csv_out} saved')
    return out

# ---------- NEW FUNCTIONS (v1.1) ----------
def smc_forward(df, customer_col='CustomerID', K=3, draws=1000, chains=None, seed=42):
    """
    Minimal SMC forward run for lumpy-HMM.
    Returns pathlib.Path to saved pickle.
    """
    import pathlib, os, platform, time, pickle
    from scipy.special import logsumexp
    from .smc_model_grok import build_uci_data, make_model

    cores = chains if chains else min(4, os.cpu_count() or 1)
    chains = chains or min(4, os.cpu_count() or 1)
    ROOT = pathlib.Path(os.getenv('LUMPYHMM_RESULTS', './results'))
    ROOT.mkdir(exist_ok=True)
    data_uci = build_uci_data(df[customer_col].unique(), df)
    t0 = time.time()
    with make_model(data_uci, K=K) as model:
        idata = pm.sample_smc(draws=draws, chains=chains, cores=cores, random_seed=seed, progressbar=True)
        log_lik = idata.sample_stats['log_marginal_likelihood'].values  # (chains, draws)
        ll_vals = np.array([row[-1] for ch in log_lik for row in ch if len(row) and not np.isnan(row[-1])])
        log_ev = float(logsumexp(ll_vals) - np.log(len(ll_vals)))
        print(f'SMC K={K} finished in {time.time() - t0:.1f} min – log-ev = {log_ev:.5f}')
        pkl_path = ROOT / f'smc_forward_K{K}_D{draws}_C{chains}.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump({g: {k: v.values for k, v in idata[g].data_vars.items()} for g in idata.groups()}, f)
    return pkl_path


def smc_post_run(pkl_path, model=None):
    """
    Harvest log-evidence, CLV, plots from pickle.
    Returns dict with results.
    """
    import pathlib, pickle, numpy as np, time
    from scipy.special import logsumexp
    from .smc_model_grok import rebuild_deterministics, make_model

    pkl = pathlib.Path(pkl_path)
    with open(pkl, 'rb') as f:
        raw = pickle.load(f)
    post = {k: v for k, v in raw['posterior'].items()}
    n_samples = post['beta0_raw'].shape[0]

    # evidence from log-weights
    log_weights = raw['sample_stats']['log_marginal_likelihood']  # nested list
    ll_vals = np.array([row[-1] for ch in log_weights for row in ch if len(row) and not np.isnan(row[-1])])
    log_ev = float(logsumexp(ll_vals) - np.log(len(ll_vals)))

    # CLV per state (mean across draws)
    mu_mean  = post['mu'].mean(axis=2)   # (N, K)
    phi_mean = post['phi'].mean(axis=1)  # (K,)
    retain   = post['trans'].mean(axis=2).diagonal()
    disc = 0.95
    clv_state = pd.Series(index=range(3), dtype=float)
    for k in range(3):
        spend = mu_mean.mean(axis=0)[k]
        clv_state[k] = (spend * disc) / (1.0 - disc * retain[k]) if retain[k] < 1.0 else 0.0

    results = {'log_evidence': log_ev, 'clv_per_state': clv_state}
    print(f'[OK] Post-run finished – log-ev = {log_ev:.5f}')
    return results


def roi_menu_csv(dataset, pkl_path, csv_out, n_draws_wanted=1000, cost_ratio=0.2, lift_pp=5):
    """
    Simulate +5 pp Engaged-state retention lift → ROI posterior
    dataset : str – 'UCI' or 'CDNOW'
    """
    import pandas as pd, numpy as np, requests, pickle, seaborn as sns, matplotlib.pyplot as plt
    from .smc_model_grok import build_uci_data, rebuild_deterministics, make_model

    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    post_full = {k: v.reshape(-1, *v.shape[2:]) for k, v in raw['posterior'].items()}
    thin = max(1, post_full['beta0_raw'].shape[0] // n_draws_wanted)
    post = {k: v[::thin] for k, v in post_full.items()}

    if dataset == 'UCI':
        df = pd.read_csv('https://raw.githubusercontent.com/sudhir-voleti/rfmpaper/main/uci_full_panel_regime.csv')
    else:  # CDNOW
        df = pd.read_csv('https://raw.githubusercontent.com/sudhir-voleti/rfmpaper/main/cdnow_subsample_180k.csv')
    data = build_uci_data(df.CustomerID.unique(), df)

    K = 3
    n_draws = post['beta0_raw'].shape[0]
    clv_ctrl = np.full((K, n_draws), np.nan)
    clv_treat = np.full((K, n_draws), np.nan)

    with make_model(data, K=K) as model:
        for i in range(n_draws):
            point = {k: np.atleast_1d(v[i]) for k, v in post.items()}
            full = rebuild_deterministics(model, point)
            base_p = full['trans']
            base_mu = full['mu'].mean(axis=0)
            clv_ctrl[:, i] = base_mu / (1 - base_p.diagonal())
            treat_p = base_p.copy()
            treat_p[2, 2] += lift_pp / 100
            treat_p[2, :] /= treat_p[2, :].sum()
            clv_treat[:, i] = base_mu / (1 - treat_p.diagonal())

    roi_post = (clv_treat - clv_ctrl) / (cost_ratio * clv_ctrl)
    out = pd.DataFrame({
        'dataset': dataset,
        'state': ['Engaged', 'Cooling', 'Churned'],
        'mean_roi': roi_post.mean(axis=1),
        'cri_lower': np.percentile(roi_post, 5, axis=1),
        'cri_upper': np.percentile(roi_post, 95, axis=1)
    })
    out.to_csv(csv_out, index=False)
    print(f'[OK] {csv_out} saved')
    return out
# ---------- END NEW FUNCTIONS ----------
## 26 Jan 26 funcs

# rfmfuncs.py  (append below your existing helpers)

from __future__ import annotations
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az

# ----------  TABLE 5  ---------------------------------------------------------
def make_table5(
    idata: az.InferenceData,
    ds_name: str,
    out_dir: pathlib.Path | None = None,
    save_plot: bool = True,
) -> pd.DataFrame:
    """
    Posterior-mean transition matrix Γ with optional heat-map.

    Parameters
    ----------
    idata   : arviz.InferenceData  (must contain posterior['Gamma'])
    ds_name : 'uci' or 'cdnow'
    out_dir : folder for CSV/PDF; None → return DataFrame only
    save_plot : write PDF heat-map?

    Returns
    -------
    DataFrame with columns ['From', 'State0', 'State1', ..., 'StateK-1']
    """
    gamma = idata.posterior["Gamma"].mean(("chain", "draw")).values  # (K, K)
    k = gamma.shape[0]
    labs = [f"State {i}" for i in range(k)]
    df = pd.DataFrame(gamma, index=labs, columns=labs)
    df.insert(0, "From", labs)

    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"table5_{ds_name}.csv", index=False)
        if save_plot:
            plt.figure(figsize=(4, 3))
            sns.heatmap(
                df.set_index("From"),
                annot=True,
                fmt=".3f",
                cmap="Blues",
                cbar_kws={"label": "P(t+1|t)"},
            )
            plt.title(f"{ds_name.upper()} – transition matrix")
            plt.tight_layout()
            plt.savefig(out_dir / f"gamma_{ds_name}.pdf")
            plt.close()
    return df


# ----------  TABLE 6  ---------------------------------------------------------
def make_table6(
    idata: az.InferenceData,
    ds_name: str,
    out_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """
    Posterior summary (mean, sd, 95 % HDI) for state-specific β0 and φ.

    Parameters
    ----------
    idata   : arviz.InferenceData
    ds_name : 'uci' or 'cdnow'
    out_dir : folder for CSV; None → return DataFrame only

    Returns
    -------
    DataFrame with columns ['Dataset', 'parameter', 'mean', 'sd', 'hdi_2.5%', 'hdi_97.5%', ...]
    """
    summ = az.summary(idata, var_names=["beta0", "phi"], hdi_prob=0.95)
    summ.insert(0, "Dataset", ds_name.upper())

    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out_dir / f"table6_{ds_name}.csv", index=False)
    return summ

# rfmfuncs.py  (append below existing helpers)

from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gammaln

def make_table7(
    idata: az.InferenceData,
    ds_name: str,
    csv_path: str | pathlib.Path,
    out_dir: pathlib.Path | None = None,
    labels: list[str] | None = None,
    preview_rows: int = 10,
) -> pd.DataFrame:
    """
    Viterbi decoding → state-share over calendar weeks + area plot.

    Parameters
    ----------
    idata   : arviz.InferenceData (posterior group)
    ds_name : 'uci' or 'cdnow'
    csv_path: original full CSV (to rebuild R,F,M,y,mask)
    out_dir : folder for CSV/PDF; None → return DataFrame only
    labels  : state names (length must equal K); None → auto State0…K-1
    preview_rows : how many weeks to print in Jupyter

    Returns
    -------
    DataFrame (week, prop_state0, …, prop_stateK-1) with **all K states** (missing = 0 %)
    """
    # posterior-mean parameters → NumPy
    beta0 = np.asarray(idata.posterior["beta0"].mean(("chain", "draw")).values, dtype=np.float64)
    betaR = np.asarray(idata.posterior["betaR"].mean(("chain", "draw")).values, dtype=np.float64)
    betaF = np.asarray(idata.posterior["betaF"].mean(("chain", "draw")).values, dtype=np.float64)
    betaM = np.asarray(idata.posterior["betaM"].mean(("chain", "draw")).values, dtype=np.float64)
    phi   = np.asarray(idata.posterior["phi"].mean(("chain", "draw")).values, dtype=np.float64)
    K = beta0.size

    # reload covariates & spend
    df  = pd.read_csv(csv_path)
    cust = df["customer_id"].unique()[:idata.posterior.dims["customer"]]
    R   = df.pivot(index="customer_id", columns="WeekStart", values="R_weeks").loc[cust].values
    F   = df.pivot(index="customer_id", columns="WeekStart", values="F_run").loc[cust].values
    M   = df.pivot(index="customer_id", columns="WeekStart", values="M_run").loc[cust].values
    y   = df.pivot(index="customer_id", columns="WeekStart", values="WeeklySpend").loc[cust].values
    mask = ~np.isnan(y)

    # rebuild mu
    lin = beta0 + betaR * R[..., None] + betaF * F[..., None] + betaM * M[..., None]
    mu  = np.clip(np.exp(lin), 1e-3, 1e6)

    # Viterbi decode (pure NumPy)
    pi0   = np.asarray(idata.posterior["pi0"].mean(("chain", "draw")).values, dtype=np.float64)
    Gamma = np.asarray(idata.posterior["Gamma"].mean(("chain", "draw")).values, dtype=np.float64)

    def zig_logp(y_row, mu_row, phi):
        psi = np.exp(-mu_row / phi)
        log_zero = np.log(psi + 1e-12)
        alpha = mu_row / phi
        beta  = 1.0 / phi
        log_pos  = (np.log1p(-psi + 1e-12) +
                    (alpha - 1) * np.log(y_row[:, None] + 1e-12) -
                    y_row[:, None] * beta +
                    alpha * np.log(beta) -
                    gammaln(alpha))
        return np.where(y_row[:, None] == 0, log_zero, log_pos)

    def viterbi_numpy(y, mu, phi, pi0, Gamma):
        N, T, K = mu.shape
        z = np.empty((N, T), dtype=int)
        for i in range(N):
            log_ems = zig_logp(y[i], mu[i], phi)
            log_delta = np.log(pi0 + 1e-12) + log_ems[0]
            log_psi = np.zeros((T, K))
            for t in range(1, T):
                tmp = log_delta + np.log(Gamma + 1e-12)
                log_delta = tmp.max(axis=0) + log_ems[t]
                log_psi[t] = tmp.argmax(axis=0)
            z_row = np.empty(T, dtype=int)
            z_row[-1] = log_delta.argmax()
            for t in range(T - 2, -1, -1):
                z_row[t] = log_psi[t + 1, z_row[t + 1]]
            z[i] = z_row
        return z

    z_mat = viterbi_numpy(y, mu, phi, pi0, Gamma)

    # state share (pad missing states to 0 %)
    T = mu.shape[1]
    prop = pd.DataFrame({t: pd.Series(z_mat[:, t]).value_counts(normalize=True) for t in range(T)}).T.fillna(0.0)
    all_states = np.arange(K)
    prop = prop.reindex(columns=all_states, fill_value=0.0)  # ensure K columns
    if labels:
        if len(labels) != K:
            raise ValueError(f"labels length {len(labels)} != number of states {K}")
        prop.columns = labels[:K]
    else:
        prop.columns = [f"State {k}" for k in range(K)]

    # area plot
    if out_dir:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        prop.to_csv(out_dir / f"table7_{ds_name}.csv", index_label="Week")
        plt.figure(figsize=(8, 3))
        prop.plot.area(stacked=True, cmap="coolwarm")
        plt.title(f"{ds_name.upper()} – latent state share (Viterbi)")
        plt.ylabel("Proportion")
        plt.xlabel("Week")
        plt.legend(title="State", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / f"state_share_{ds_name}.pdf")
        plt.close()

    # Jupyter preview
    if preview_rows:
        from IPython.display import display
        display(prop.head(preview_rows))
    return prop
    

# rfmfuncs.py  (append below existing helpers)

from __future__ import annotations
import pathlib, pandas as pd, arviz as az

def make_table8(
    idata: az.InferenceData,
    ds_name: str,
    out_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """
    PSIS-LOO & WAIC summary (Table 8).

    Parameters
    ----------
    idata   : arviz.InferenceData (must contain log_likelihood group)
    ds_name : 'uci' or 'cdnow'
    out_dir : folder for CSV; None → return DataFrame only

    Returns
    -------
    DataFrame with columns ['Dataset', 'ELPD-LOO', 'p_loo', 'SE-LOO', 'WAIC', 'SE-WAIC']
    """
    loo  = az.loo(idata, pointwise=True)
    waic = az.waic(idata, pointwise=True)

    df = pd.DataFrame({
        "Dataset"  : [ds_name.upper()],
        "ELPD-LOO" : [loo.elpd_loo],
        "p_loo"    : [loo.p_loo],
        "SE-LOO"   : [loo.se],
        "WAIC"     : [waic.elpd_waic],   # correct attribute
        "SE-WAIC"  : [waic.se],
    })

    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        df.to_csv(out_dir / f"table8_{ds_name}.csv", index=False)
    return df

'''
## test drive and run
ROOT = pathlib.Path("/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/results/full")
PKL_uci  = ROOT / "uci" / "smc_full_4338cust_K5_D1500_C4.pkl"   # start with CDNOW-K4
PKL_cdnow  = ROOT / "cdnow" / "smc_full_2357cust_K4_D1000_C4.pkl"   # start with CDNOW-K4
OUT  = ROOT / "paper_step"
OUT.mkdir(exist_ok=True)

# %% 4.  run -------------------------------------------------------------------
idata_uci = load_idata(PKL_uci)
idata_cdnow = load_idata(PKL_cdnow)

tbl5_uci = make_table5(idata_uci, 'uci', out_dir=OUT, preview=True)
tbl5_cdnow = make_table5(idata_cdnow, 'cdnow', out_dir=OUT, preview=True)

## test-drive
tbl6_uci = make_table6(idata_uci, "uci")
tbl6_cdnow = make_table6(idata_cdnow, "cdnow")

csv_uci = "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/data/uci_full.csv"
csv_cdnow = "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/data/cdnow_full.csv"

tbl7_uci = make_table7(idata_uci, "uci", csv_uci, out_dir=OUT,
                       labels=['Cold','Cool','Warm','Luke','Hot'],
                       preview_rows=10, n_cust=None)   # ← decode ALL customers

tbl7_cdnow = make_table7(idata_cdnow, "cdnow", csv_cdnow, out_dir=OUT, labels=["Cold","Cool","Warm","Hot"], 
                         preview_rows=10, n_cust=None)

# down-sample posterior to every 5th draw (300 instead of 1500)
idata_uci1 = idata_uci.isel(draw=slice(None, None, 5))
idata_cdnow1 = idata_cdnow.isel(draw=slice(None, None, 5))

idata_uci1 = add_log_likelihood(idata_uci1, csv_path=csv_uci)
idata_cdnow1 = add_log_likelihood(idata_cdnow1, csv_path=csv_cdnow)

tbl8_uci = make_table8(idata_uci1, 'uci', out_dir=OUT, preview=True)
tbl8_cdnow = make_table8(idata_cdnow1, 'cdnow', out_dir=OUT, preview=True)
'''

