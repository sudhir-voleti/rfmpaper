import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import adjusted_rand_score
from scipy.stats import mode
import warnings


def load_ground_truth(csv_path: Path) -> Optional[Dict]:
    """
    Load ground truth states from simulation CSV.
    CSV must have columns: customer_id, time_period, y, [state]
    """
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check if state column exists
        if 'state' not in df.columns:
            # Try to infer from common names
            state_cols = [c for c in df.columns if 'state' in c.lower()]
            if state_cols:
                df = df.rename(columns={state_cols[0]: 'state'})
            else:
                print(f"Warning: No state column found in {csv_path}")
                return None
        
        # Pivot to (N, T) matrices
        N = df['customer_id'].nunique()
        T = df['time_period'].nunique()
        
        states = df.pivot(index='customer_id', columns='time_period', values='state').values
        obs = df.pivot(index='customer_id', columns='time_period', values='y').values
        
        return {
            'states': states,
            'observations': obs,
            'N': N,
            'T': T
        }
    except Exception as e:
        print(f"Error loading ground truth from {csv_path}: {e}")
        return None


## ----


def compute_viterbi_ari(idata, ground_truth_states: np.ndarray) -> float:
    """
    Computes ARI between Viterbi path and simulation ground truth.
    Uses only training period (where Viterbi exists).
    """
    if 'viterbi' not in idata.posterior:
        return np.nan

    # Viterbi shape: (chain, draw, N, T_train)
    viterbi = idata.posterior['viterbi'].values
    n_chains, n_draws, N_vit, T_vit = viterbi.shape
    
    # Slice ground truth to match training period
    if ground_truth_states.shape[1] > T_vit:
        gt_sliced = ground_truth_states[:, :T_vit]
    else:
        gt_sliced = ground_truth_states
    
    if N_vit != gt_sliced.shape[0]:
        print(f"  N mismatch: viterbi N={N_vit}, truth N={gt_sliced.shape[0]}")
        return np.nan
    
    # Reshape to (samples, N, T)
    v_paths = viterbi.reshape(n_chains * n_draws, N_vit, T_vit)

    # Take mode across samples for each (customer, time)
    v_map = mode(v_paths, axis=0, keepdims=False).mode

    # Flatten and compute ARI
    return adjusted_rand_score(gt_sliced.flatten(), v_map.flatten())


## ----



def compute_forward_prediction(idata, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bayesian forward prediction using alpha_filtered and Gamma.
    """
    if 'alpha_filtered' not in idata.posterior:
        return np.nan * np.ones(n_steps), np.nan * np.ones(n_steps)
    
    # Get shapes from actual data
    alpha_shape = idata.posterior['alpha_filtered'].shape
    # Expected: (chain, draw, N, T, K)
    
    if len(alpha_shape) != 5:
        print(f"Unexpected alpha_filtered shape: {alpha_shape}")
        return np.nan * np.ones(n_steps), np.nan * np.ones(n_steps)
    
    n_chains, n_draws, N, T, K = alpha_shape
    
    # Extract final time point: (chain, draw, N, K)
    alpha_final = idata.posterior['alpha_filtered'][:, :, :, -1, :].values
    alpha_final = alpha_final.reshape(n_chains * n_draws, N, K)
    
    # Population average: (samples, K)
    alpha_pop = alpha_final.mean(axis=1)
    
    # Transition matrix: (chain, draw, K, K) -> (samples, K, K)
    Gamma = idata.posterior['Gamma'].values.reshape(n_chains * n_draws, K, K)
    
    # State means: (chain, draw, K) or (chain, draw) for K=1
    beta0 = idata.posterior['beta0'].values
    if beta0.ndim == 2:  # K=1 case
        beta0 = beta0.reshape(n_chains * n_draws, 1)
    else:
        beta0 = beta0.reshape(n_chains * n_draws, K)
    
    mu_states = np.exp(beta0)
    
    # Forward predict
    all_preds = []
    
    for i in range(n_chains * n_draws):
        preds = []
        current_probs = alpha_pop[i]
        
        for t in range(n_steps):
            current_probs = current_probs @ Gamma[i]
            pred = np.sum(current_probs * mu_states[i])
            preds.append(pred)
        
        all_preds.append(preds)
    
    all_preds = np.array(all_preds)
    
    return all_preds.mean(axis=0), all_preds.std(axis=0)



## ----


def compute_brier_score(y_true: np.ndarray, y_prob_nonzero: float) -> float:
    """
    Brier score for zero-inflation prediction.
    """
    if len(y_true) == 0:
        return np.nan
    
    y_binary = (y_true > 0).astype(int)
    return np.mean((y_prob_nonzero - y_binary) ** 2)


def compute_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Calibration metrics.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'calibration_error': np.nan, 'max_calibration_error': np.nan}
    
    # Filter to positive observations for spend calibration
    mask_pos = y_true > 0
    if mask_pos.sum() < n_bins:
        return {'calibration_error': np.nan, 'max_calibration_error': np.nan}
    
    y_true_pos = y_true[mask_pos]
    y_pred_pos = y_pred[mask_pos]
    
    # Bin by predicted values
    try:
        bin_edges = np.percentile(y_pred_pos, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-6
    except:
        return {'calibration_error': np.nan, 'max_calibration_error': np.nan}
    
    errors = []
    for i in range(n_bins):
        mask = (y_pred_pos >= bin_edges[i]) & (y_pred_pos < bin_edges[i + 1])
        if mask.sum() > 0:
            errors.append(abs(y_pred_pos[mask].mean() - y_true_pos[mask].mean()))
    
    return {
        'calibration_error': np.mean(errors) if errors else np.nan,
        'max_calibration_error': np.max(errors) if errors else np.nan
    }

## ----

def extract_metrics_from_pkl(pkl_path: Path, 
                            ground_truth: Optional[Dict] = None,
                            metrics: List[str] = ['all']) -> Dict:
    """
    Extract comprehensive metrics from a single pkl file.
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return {'file': pkl_path.name, 'error': str(e)}
    
    res = data.get('res', {})
    idata = data.get('idata', None)
    
    # Basic metadata
    result = {
        'file': pkl_path.name,
        'K': res.get('K', np.nan),
        'model': res.get('model_type', 'unknown'),
        'use_gam': res.get('use_gam', False),
        'N': res.get('N', np.nan),
        'T': res.get('T', np.nan),
        'T_train': res.get('T', np.nan),
        'draws': res.get('draws', np.nan),
        'chains': res.get('chains', np.nan),
        'time_min': res.get('time_min', np.nan),
        'timestamp': res.get('timestamp', ''),
    }
    
    # Parse model description
    p_type = 'state-varying' if 'statep' in pkl_path.name else ('fixed-p' if 'p1.5' in pkl_path.name else 'N/A')
    glm_gam = 'GAM' if res.get('use_gam', False) else 'GLM'
    result['model_desc'] = f"K{result['K']}-{result['model']}-{glm_gam}-{p_type}"
    
    # In-sample fit
    if 'all' in metrics or 'log_ev' in metrics:
        result['log_evidence'] = res.get('log_evidence', np.nan)
    
    # ESS diagnostics
    if idata is not None and ('all' in metrics or 'ess' in metrics):
        try:
            import arviz as az
            ess = az.ess(idata)
            ess_vals = [ess[v].values.min() for v in ess.data_vars if hasattr(ess[v].values, 'min')]
            result['ess_min'] = min(ess_vals) if ess_vals else np.nan
            result['ess_median'] = np.median(ess_vals) if ess_vals else np.nan
        except:
            result['ess_min'] = np.nan
            result['ess_median'] = np.nan
    
    # OOS metrics
    y_test = res.get('y_test')
    if y_test is not None:
        result['T_test'] = res.get('T_test', y_test.shape[1])
        result['y_test_mean'] = np.mean(y_test)
        result['y_test_zeros'] = np.mean(y_test == 0)
        
        if 'all' in metrics or any(m in metrics for m in ['oos_rmse', 'oos_mae']):
            if idata is not None and 'beta0' in idata.posterior:
                # Get posterior mean of exp(beta0) for base prediction
                beta0 = idata.posterior['beta0'].values
                mu_pred = np.exp(beta0.mean())
                
                K = result['K']
                
                # For K>1 with alpha_filtered, use forward prediction
                if K > 1 and 'alpha_filtered' in idata.posterior:
                    pred_mean, _ = compute_forward_prediction(idata, result['T_test'])
                else:
                    # K=1 static: constant prediction across all time points
                    pred_mean = np.full(result['T_test'], mu_pred)
                
                # Compare to actual test data (averaged across customers per time)
                y_test_avg = y_test.mean(axis=0)
                
                result['oos_rmse'] = np.sqrt(np.mean((y_test_avg - pred_mean) ** 2))
                result['oos_mae'] = np.mean(np.abs(y_test_avg - pred_mean))
                result['oos_mape'] = np.mean(np.abs((y_test_avg - pred_mean) / (y_test_avg + 1e-6))) * 100
                
                # Calibration
                if 'all' in metrics or 'calibration' in metrics:
                    y_pred_full = np.tile(pred_mean, (y_test.shape[0], 1))
                    cal = compute_calibration(y_test, y_pred_full)
                    result.update(cal)
            else:
                result['oos_rmse'] = np.nan
                result['oos_mae'] = np.nan
                result['oos_mape'] = np.nan
        
        # Brier score for zero prediction
        if 'all' in metrics or 'brier' in metrics:
            prob_nonzero = 1 - result['y_test_zeros']
            result['brier_zero'] = compute_brier_score(y_test, prob_nonzero)
    else:
        result['T_test'] = 0
        for k in ['oos_rmse', 'oos_mae', 'oos_mape', 'brier_zero', 'calibration_error', 'max_calibration_error']:
            result[k] = np.nan
    
    # State recovery (ARI) - only for K>1
    if ground_truth is not None and result.get('K', 0) > 1:
        if 'all' in metrics or 'ari' in metrics:
            result['ari'] = compute_viterbi_ari(idata, ground_truth['states'])
    
    # Ground truth comparison (if available)
    if ground_truth is not None:
        result['gt_pi0'] = np.mean(ground_truth['observations'] == 0)
        result['gt_mean'] = np.mean(ground_truth['observations'])
    
    return result


## ---


from sklearn.metrics import adjusted_rand_score

def compute_ari(idata, true_states_csv):
    """
    Compute Adjusted Rand Index between Viterbi path and true states.
    true_states_csv: path to CSV with 'customer_id', 'time_period', 'true_state' columns
    """
    # Load true states
    true_df = pd.read_csv(true_states_csv)
    
    # Pivot to (N, T) matrix
    true_states = true_df.pivot(index='customer_id', columns='time_period', values='true_state').values
    
    # Get Viterbi path from idata
    if 'viterbi' in idata.posterior:
        viterbi = idata.posterior['viterbi'].mean(dim=['chain', 'draw']).values.astype(int)
    else:
        return np.nan
    
    # Flatten both (ignore time structure for ARI)
    true_flat = true_states.flatten()
    viterbi_flat = viterbi.flatten()
    
    # Compute ARI
    ari = adjusted_rand_score(true_flat, viterbi_flat)
    
    return ari



## ----


def process_simulation_folder(folder_path: str,
                              ground_truth_csv: Optional[str] = None,
                              metrics: List[str] = ['all'],
                              pattern: str = "smc_*.pkl",
                              y_train: Optional[np.ndarray] = None,
                              y_test: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Process all pkl files in a folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing .pkl files
    ground_truth_csv : str, optional
        Path to simulation CSV with true states for ARI computation
    metrics : List[str]
        Metrics to extract
    pattern : str
        Glob pattern for pkl files
    y_train : np.ndarray, optional
        Training data for OOS recomputation
    y_test : np.ndarray, optional
        Test data for OOS recomputation
    
    Returns:
    --------
    pd.DataFrame with all models and metrics
    """
    folder = Path(folder_path)
    pkl_files = sorted(folder.glob(pattern))
    
    if not pkl_files:
        print(f"No files found matching {pattern} in {folder_path}")
        return pd.DataFrame()
    
    # Load ground truth if provided
    ground_truth = None
    if ground_truth_csv:
        gt_path = Path(ground_truth_csv)
        if gt_path.exists():
            print(f"Loading ground truth from {gt_path}")
            ground_truth = load_ground_truth(gt_path)
        else:
            print(f"Ground truth file not found: {gt_path}")
    
    print(f"Processing {len(pkl_files)} files from {folder_path}...")
    
    results = []
    for pkl_file in pkl_files:
        print(f"  {pkl_file.name}")
        metrics_dict = extract_metrics_from_pkl(pkl_file, ground_truth, metrics)
        
        # Recompute OOS if simulation data provided
        if y_train is not None and y_test is not None:
            try:
                oos_rmse, oos_msg = recompute_oos_rmse(pkl_file, y_train, y_test)
                metrics_dict['oos_rmse_recomputed'] = oos_rmse
                metrics_dict['oos_recompute_status'] = oos_msg
                print(f"    OOS: {oos_rmse:.4f}" if not np.isnan(oos_rmse) else f"    OOS: {oos_msg}")
            except Exception as e:
                metrics_dict['oos_rmse_recomputed'] = np.nan
                metrics_dict['oos_recompute_status'] = str(e)[:50]
                print(f"    OOS ERROR: {str(e)[:50]}")
        
        results.append(metrics_dict)
    
    df = pd.DataFrame(results)
    
    # Sort by log-evidence
    if 'log_evidence' in df.columns:
        df = df.sort_values('log_evidence', ascending=False)
    
    return df


## ----


def display_results_table(df: pd.DataFrame, 
                         cols: Optional[List[str]] = None) -> None:
    """
    Pretty print results table.
    """
    if df.empty:
        print("No results to display")
        return
    
    if cols is None:
        # Default informative columns
        default_cols = ['model_desc', 'log_evidence', 'oos_rmse', 'ari', 'time_min']
        cols = [c for c in default_cols if c in df.columns]
    
    print("\n" + "="*120)
    print("SIMULATION RESULTS SUMMARY")
    print("="*120)
    
    # Format for display
    display_df = df[cols].copy()
    
    # Round floats
    for col in display_df.columns:
        if display_df[col].dtype == np.float64:
            display_df[col] = display_df[col].round(4)
    
    print(display_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "-"*120)
    if 'log_evidence' in df.columns:
        best_idx = df['log_evidence'].idxmax()
        print(f"Best model (log-ev): {df.loc[best_idx, 'model_desc']}")
        print(f"  Log-ev: {df.loc[best_idx, 'log_evidence']:.2f}")
    
    if 'oos_rmse' in df.columns and not df['oos_rmse'].isna().all():
        best_idx = df['oos_rmse'].idxmin()
        print(f"\nBest model (OOS RMSE): {df.loc[best_idx, 'model_desc']}")
        print(f"  RMSE: {df.loc[best_idx, 'oos_rmse']:.4f}")
    
    if 'ari' in df.columns and not df['ari'].isna().all():
        best_idx = df['ari'].idxmax()
        print(f"\nBest state recovery (ARI): {df.loc[best_idx, 'model_desc']}")
        print(f"  ARI: {df.loc[best_idx, 'ari']:.4f}")
    
    print("-"*120)

## ----


def load_simulation_data(csv_path, n_cust=None, train_frac=0.8, seed=42):
    """Load simulation from CSV with train/test split."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    N_full = df['customer_id'].nunique()
    T = df['time_period'].nunique()
    obs = df.pivot(index='customer_id', columns='time_period', values='y').values
    
    if n_cust is not None and n_cust < N_full:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_full, n_cust, replace=False)
        obs = obs[idx, :]
        N = n_cust
    else:
        N = N_full
    
    T_train = int(T * train_frac)
    obs_train = obs[:, :T_train]
    obs_test = obs[:, T_train:]
    
    return obs_train, obs_test, N, T_train

## ----

def compute_rfm_features_oos(y_train, y_test):
    """Compute RFM for OOS periods continuing from training."""
    N, T_test = y_test.shape
    T_train = y_train.shape[1]
    
    R = np.zeros((N, T_test), dtype=np.float32)
    F = np.zeros((N, T_test), dtype=np.float32)
    M = np.zeros((N, T_test), dtype=np.float32)
    
    for i in range(N):
        train_purchases = np.where(y_train[i, :] > 0)[0]
        if len(train_purchases) > 0:
            last_purchase = train_purchases[-1]
            cumulative_freq = len(train_purchases)
            cumulative_spend = np.sum(y_train[i, :])
        else:
            last_purchase = -1
            cumulative_freq = 0
            cumulative_spend = 0.0
        
        for t in range(T_test):
            t_absolute = T_train + t
            
            if y_test[i, t] > 0:
                last_purchase = t_absolute
                cumulative_freq += 1
                cumulative_spend += y_test[i, t]
            
            if last_purchase >= 0:
                R[i, t] = t_absolute - last_purchase
                F[i, t] = cumulative_freq
                M[i, t] = cumulative_spend / cumulative_freq if cumulative_freq > 0 else 0.0
            else:
                R[i, t] = t_absolute + 1
                F[i, t] = 0
                M[i, t] = 0.0
    
    return R, F, M

## ----

def recompute_oos_metrics(pkl_path, y_train, y_test):
    """
    Recompute OOS RMSE, MAE, and dual R² metrics from pickle.
    Handles K=1 (Static) and K>1 (Dynamic HMM) cases.
    """
    import numpy as np
    import pickle
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    res = data['res']
    idata = data['idata']
    
    K = res.get('K', 1)
    N = res.get('N')
    use_gam = res.get('use_gam', False)
    
    # GAM requires basis functions (X_spline) which aren't in the idata.
    # We rely on res['y_pred'] if it exists, otherwise skip recomputation for GAM.
    if use_gam:
        if 'y_pred' in res and res['y_pred'] is not None:
            y_pred = res['y_pred']
        else:
            return {k: np.nan for k in ['rmse', 'mae', 'r2_bayes', 'r2_ols']}, "GAM skipped (missing pred)"
    else:
        # GLM Reconstruction Logic
        try:
            beta0 = idata.posterior['beta0'].mean(dim=['chain', 'draw']).values
            betaR = idata.posterior['betaR'].mean(dim=['chain', 'draw']).values
            betaF = idata.posterior['betaF'].mean(dim=['chain', 'draw']).values
            betaM = idata.posterior['betaM'].mean(dim=['chain', 'draw']).values
            
            # Reconstruct RFM features for the test period
            # Assumes compute_rfm_features_oos is available in your namespace
            R_test, F_test, M_test = compute_rfm_features_oos(y_train, y_test)
            
            if K == 1:
                log_mu = beta0 + betaR * R_test + betaF * F_test + betaM * M_test
                y_pred = np.exp(np.clip(log_mu, -10, 15))
            else:
                if 'alpha_filtered' not in idata.posterior:
                    return {k: np.nan for k in ['rmse', 'mae', 'r2_bayes', 'r2_ols']}, "Missing alpha_filtered"
                
                # Get last state probs from training to seed OOS transitions
                alpha_final = idata.posterior['alpha_filtered'].mean(dim=['chain', 'draw']).values[:, -1, :]
                Gamma = idata.posterior['Gamma'].mean(dim=['chain', 'draw']).values
                
                # mu_states: (N, T_test, K)
                log_mu = beta0[None, None, :] + \
                         betaR[None, None, :] * R_test[:, :, None] + \
                         betaF[None, None, :] * F_test[:, :, None] + \
                         betaM[None, None, :] * M_test[:, :, None]
                mu_states = np.exp(np.clip(log_mu, -10, 15))
                
                y_pred = np.zeros((N, y_test.shape[1]))
                state_probs = alpha_final # Initial state at T_train
                
                for t in range(y_test.shape[1]):
                    # Transition to next period
                    state_probs = state_probs @ Gamma
                    # E[y] is weighted sum of state means
                    y_pred[:, t] = np.sum(state_probs * mu_states[:, t, :], axis=1)
        except Exception as e:
            return {k: np.nan for k in ['rmse', 'mae', 'r2_bayes', 'r2_ols']}, f"Recon Error: {str(e)}"

    # Final Metric Calculation
    mask = ~np.isnan(y_test) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {k: np.nan for k in ['rmse', 'mae', 'r2_bayes', 'r2_ols']}, "No overlap"
    
    y_o = y_test[mask]
    y_p = y_pred[mask]
    
    # 1. Error Metrics
    rmse = np.sqrt(np.mean((y_o - y_p)**2))
    mae = np.mean(np.abs(y_o - y_p))
    
    # 2. Bayesian R² (Variance Ratio - ignores bias, focuses on correlation/fit)
    var_pred = np.var(y_p)
    var_resid = np.var(y_o - y_p)
    r2_bayes = var_pred / (var_pred + var_resid) if (var_pred + var_resid) > 0 else 0.0
    
    # 3. OLS R² (1 - SS_res/SS_tot - penalizes bias heavily)
    ss_res = np.sum((y_o - y_p)**2)
    ss_tot = np.sum((y_o - np.mean(y_o))**2)
    r2_ols = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2_bayes': r2_bayes,
        'r2_ols': r2_ols
    }
    
    return metrics, "OK"

## ----


def compute_ground_truth_features(gt: Dict) -> Dict:
    """Compute key features from ground truth simulation."""
    obs = gt['observations']
    states = gt['states']
    
    # Basic moments
    zero_incidence = np.mean(obs == 0)
    mean_spend = np.mean(obs[obs > 0]) if np.any(obs > 0) else 0
    std_spend = np.std(obs[obs > 0]) if np.any(obs > 0) else 0
    cv_spend = std_spend / mean_spend if mean_spend > 0 else 0
    
    # State distribution
    unique_states = np.unique(states)
    n_modes = len(unique_states)
    state_shares = {f"state_{s}_share": np.mean(states == s) for s in unique_states}
    
    # Transition matrix (if T > 1)
    T = obs.shape[1]
    if T > 1:
        transitions = np.zeros((n_modes, n_modes))
        for t in range(T-1):
            for s_prev in unique_states:
                mask = states[:, t] == s_prev
                if mask.sum() > 0:
                    s_next = states[mask, t+1]
                    for s_curr in unique_states:
                        transitions[s_prev, s_curr] += np.sum(s_next == s_curr)
        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions = np.divide(transitions, row_sums, where=row_sums>0)
        diagonals = np.diag(transitions)
        mean_persistence = np.mean(diagonals)
    else:
        transitions = None
        diagonals = None
        mean_persistence = np.nan
    
    return {
        'zero_incidence': zero_incidence,
        'mean_spend': mean_spend,
        'std_spend': std_spend,
        'cv_spend': cv_spend,
        'n_modes': n_modes,
        'mean_persistence': mean_persistence,
        **state_shares
    }


def extract_model_persistence(idata, K: int) -> Dict:
    """Extract diagonal elements of transition matrix (state persistence)."""
    if K <= 1 or 'Gamma' not in idata.posterior:
        return {'persistence': np.nan, 'persistence_ci': 'N/A'}
    
    Gamma = idata.posterior['Gamma'].values  # (chains, draws, K, K)
    diagonals = np.einsum('...kk->...k', Gamma)  # Extract diagonal
    
    # Mean persistence across states
    mean_persist = np.mean(diagonals, axis=-1)  # Average across states
    persistence_mean = np.mean(mean_persist)
    persistence_lo = np.percentile(mean_persist, 2.5)
    persistence_hi = np.percentile(mean_persist, 97.5)
    
    # Per-state persistence
    per_state_mean = np.mean(diagonals, axis=(0,1))
    per_state_str = '/'.join([f"{p:.2f}" for p in per_state_mean])
    
    return {
        'persistence': persistence_mean,
        'persistence_lo': persistence_lo,
        'persistence_hi': persistence_hi,
        'persistence_ci': f"{persistence_mean:.2f} [{persistence_lo:.2f}, {persistence_hi:.2f}]",
        'per_state_persistence': per_state_str
    }

## ----


def extract_simulation_metrics(pkl_folder: str, 
                               world_name: str = "unknown",
                               ground_truth_csv: Optional[str] = None,
                               y_train: np.ndarray = None,
                               y_test: np.ndarray = None) -> pd.DataFrame:
    """
    Extract comprehensive metrics from simulation results for a single world.
    Accepts y_train and y_test to compute in-sample and OOS R² on the fly.
    """
    folder = Path(pkl_folder)
    pkl_files = sorted(folder.glob("smc_*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found in {pkl_folder}")
        return pd.DataFrame()
    
    print(f"\n{'='*70}")
    print(f"Extracting metrics for: {world_name.upper()} WORLD")
    print(f"{'='*70}")
    print(f"Found {len(pkl_files)} models")
    
    # Load ground truth if provided    
    gt = load_ground_truth(Path(ground_truth_csv)) if ground_truth_csv else None
    if gt:
        print(f"Ground truth: N={gt['N']}, T={gt['T']}")

    # Compute and print ground truth features
    gt_features = compute_ground_truth_features(gt) if gt else None
    if gt_features:
        print(f"\n{'-'*70}")
        print("GROUND TRUTH FEATURES")
        print(f"{'-'*70}")
        print(f"Zero incidence:     {gt_features['zero_incidence']:.2%}")
        print(f"Mean spend (y>0):   ${gt_features['mean_spend']:.2f}")
        print(f"CV spend:           {gt_features['cv_spend']:.2f}")
        print(f"Number of modes:    {gt_features['n_modes']}")
        print(f"Mean persistence:   {gt_features['mean_persistence']:.3f}")
        for k, v in gt_features.items():
            if 'state_' in k and 'share' in k:
                print(f"{k.replace('_', ' ').title():20s} {v:.2%}")
        print(f"{'-'*70}\n")
    
    print(f"Processing {len(pkl_files)} models...")
    
    results = []
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            res = data['res']
            idata = data['idata']
            
            # Skip exploded models
            log_ev = res.get('log_evidence', np.nan)
            if log_ev > 0 or not np.isfinite(log_ev):
                print(f"  SKIP (exploded): {pkl_path.name}")
                continue
            
            # Parse model specs from filename
            filename = pkl_path.name
            parts = filename.replace('smc_', '').replace('.pkl', '').split('_')
            
            K = int([p for p in parts if p.startswith('K')][0].replace('K', '')) if any(p.startswith('K') for p in parts) else res.get('K', 0)
            model_type = [p for p in parts if p in ['TWEEDIE', 'NBD', 'POISSON', 'HURDLE']][0] if any(p in ['TWEEDIE', 'NBD', 'POISSON', 'HURDLE'] for p in parts) else res.get('model_type', 'unknown')
            glm_gam = 'GAM' if 'GAM' in parts else 'GLM'
            p_spec = 'statep' if 'statep' in parts else ('p1.5' if 'p1.5' in parts else 'free')
            
            row = {
                'world': world_name,
                'file': filename,
                'K': K,
                'model_type': model_type,
                'glm_gam': glm_gam,
                'p_spec': p_spec,
                'model_desc': f"K{K}-{model_type}-{glm_gam}-{p_spec}",
                'N': res.get('N', np.nan),
                'draws': res.get('draws', np.nan),
                'log_evidence': log_ev,
                'time_min': res.get('time_min', np.nan),
            }
            
            # === R² & OOS METRICS RECOMPUTATION ===
            # Call the unified metric engine
            metrics, status = recompute_oos_metrics(pkl_path, y_train, y_test)
            
            row['oos_rmse'] = metrics.get('rmse', res.get('oos_rmse', np.nan))
            row['oos_mae'] = metrics.get('mae', res.get('oos_mae', np.nan))
            row['r2_oos_bayesian'] = metrics.get('r2_bayes', np.nan)
            row['r2_oos_ols'] = metrics.get('r2_ols', np.nan)
            
            # Attempt In-Sample R2 if training predictions are present in the pickle
            if 'y_pred_train' in res and y_train is not None:
                # Basic OLS R2 for training
                y_o_is = y_train.flatten()
                y_p_is = res['y_pred_train'].flatten()
                ss_res = np.sum((y_o_is - y_p_is)**2)
                ss_tot = np.sum((y_o_is - np.mean(y_o_is))**2)
                row['r2_is_ols'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                # Bayesian R2 (variance ratio)
                v_p, v_r = np.var(y_p_is), np.var(y_o_is - y_p_is)
                row['r2_is_bayesian'] = v_p / (v_p + v_r) if (v_p + v_r) > 0 else 0.0
            else:
                row['r2_is_bayesian'] = np.nan
                row['r2_is_ols'] = np.nan

            if 'y_test' in res and res['y_test'] is not None:
                row['T_test'] = res['T_test']
                row['y_test_zeros'] = np.mean(res['y_test'] == 0)
            
            # ARI (K > 1 only)
            if K > 1 and gt is not None:
                row['ari'] = compute_viterbi_ari(idata, gt['states'])
            else:
                row['ari'] = np.nan
            
            # p-parameter with CI
            if 'p' in idata.posterior:
                p_post = idata.posterior['p'].values
                if p_post.ndim == 2:  # Single p
                    p_flat = p_post.flatten()
                    row['p_mean'] = np.mean(p_flat)
                    row['p_lo'] = np.percentile(p_flat, 2.5)
                    row['p_hi'] = np.percentile(p_flat, 97.5)
                    row['p_ci'] = f"{row['p_mean']:.2f} [{row['p_lo']:.2f}, {row['p_hi']:.2f}]"
                elif p_post.ndim == 3:  # State-specific
                    n_ch, n_dr, n_states = p_post.shape
                    p_flat = p_post.reshape(n_ch * n_dr, n_states)
                    p_means = np.mean(p_flat, axis=0)
                    p_lo = np.percentile(p_flat, 2.5, axis=0)
                    p_hi = np.percentile(p_flat, 97.5, axis=0)
                    row['p_mean'] = np.mean(p_means)
                    row['p_lo'] = np.min(p_lo)
                    row['p_hi'] = np.max(p_hi)
                    if n_states == 2:
                        row['p_ci'] = f"{p_means[0]:.2f}/{p_means[1]:.2f} [{p_lo[0]:.2f},{p_hi[1]:.2f}]"
                    else:
                        row['p_ci'] = f"{np.mean(p_means):.2f} [{np.min(p_lo):.2f},{np.max(p_hi):.2f}]"
            else:
                row['p_mean'] = row['p_lo'] = row['p_hi'] = np.nan
                row['p_ci'] = "N/A"

            # Model persistence (K>1)
            if K > 1:
                persist_dict = extract_model_persistence(idata, K)
                row.update(persist_dict)
            else:
                row['persistence'] = np.nan
                row['persistence_ci'] = 'N/A'
                row['per_state_persistence'] = 'N/A'

            results.append(row)

            # Updated print string including Bayesian OOS R2
            oos_str = f"RMSE={row['oos_rmse']:6.2f}/MAE={row['oos_mae']:6.2f}" if not np.isnan(row['oos_rmse']) else "OOS=N/A"
            r2_str = f"R²oos={row.get('r2_oos_bayesian', np.nan):.3f}" if not np.isnan(row.get('r2_oos_bayesian', np.nan)) else "R²oos=N/A"
            persist_str = row.get('persistence_ci', 'N/A')
            print(f"  ✓ {filename:45s} | log_ev={log_ev:12.2f} | {r2_str} | ARI={row.get('ari', np.nan):5.3f} | {oos_str} | persist={persist_str} | p={row.get('p_ci', 'N/A')}")
            
        except Exception as e:
            print(f"  ✗ ERROR {pkl_path.name}: {str(e)[:60]}")
    
    df = pd.DataFrame(results)
    if df.empty:
        return df
    
    # Sort by log-evidence
    df = df.sort_values('log_evidence', ascending=False)
    
    # Summary
    print(f"\n{'-'*100}")
    print("SUMMARY")
    print(f"{'-'*100}")
    
    if not df['log_evidence'].isna().all():
        best = df.loc[df['log_evidence'].idxmax()]
        print(f"Best log-ev:  {best['model_desc']} ({best['log_evidence']:.2f})")
    
    if 'ari' in df.columns and not df['ari'].isna().all():
        best = df.loc[df['ari'].idxmax()]
        print(f"Best ARI:     {best['model_desc']} ({best['ari']:.4f})")
    
    if not df['oos_rmse'].isna().all():
        best = df.loc[df['oos_rmse'].idxmin()]
        print(f"Best OOS:     {best['model_desc']} (RMSE={best['oos_rmse']:.4f}, MAE={best['oos_mae']:.4f})")
    
    print(f"{'='*100}\n")
    
    return df


## ----

def extract_all_worlds(base_path: str, worlds: list, harness_dir: str = "harness_n500_d800") -> pd.DataFrame:
    """
    Extract metrics from all 4 simulation worlds and stack into single table.
    
    Parameters:
    -----------
    base_path : str
        Base directory containing world subfolders (e.g., "simul_21feb")
    worlds : list
        List of world names (e.g., ["poisson", "sporadic", "gamma", "clumpy"])
    harness_dir : str
        Subdirectory containing .pkl files
    
    Returns:
    --------
    pd.DataFrame with all worlds stacked, with world identifier column
    """
    all_results = []
    
    for world in worlds:
        pkl_folder = Path(base_path) / world / harness_dir
        
        if not pkl_folder.exists():
            print(f"WARNING: {pkl_folder} not found, skipping {world}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {world.upper()} WORLD")
        print(f"{'='*80}")
        
        # Use existing extract_simulation_metrics but modify to return df
        df = extract_world_metrics(pkl_folder, world)
        
        if df is not None and not df.empty:
            all_results.append(df)
    
    if not all_results:
        print("No results found for any world")
        return pd.DataFrame()
    
    # Stack all worlds
    stacked = pd.concat(all_results, ignore_index=True)
    
    # Display summary table
    print(f"\n{'='*100}")
    print("STACKED RESULTS: ALL 4 WORLDS")
    print(f"{'='*100}")
    
    # Sort by world and log-evidence
    stacked = stacked.sort_values(['world', 'log_evidence'], ascending=[True, False])
    print(f"\n{'World':<10} {'Model':<30} {'Log-Ev':>10} {'R²(IS)':>8} {'R²(OOS)':>8} {'OOS_RMSE':>10} {'OOS_MAE':>10} {'ARI':>6} {'p':>12} {'Status':>8}")
    print("-" * 100)

    for _, row in stacked.iterrows():
        r2_is = f"{row['r2_is_bayesian']:.3f}" if pd.notna(row.get('r2_is_bayesian')) else "N/A"
        r2_oos = f"{row['r2_oos_bayesian']:.3f}" if pd.notna(row.get('r2_oos_bayesian')) else "N/A"
        ari_str = f"{row['ari']:.3f}" if pd.notna(row['ari']) else "N/A"
        oos_str = f"{row['oos_rmse']:.2f}" if pd.notna(row['oos_rmse']) else "N/A"
        mae_str = f"{row['oos_mae']:.2f}" if pd.notna(row.get('oos_mae')) else "N/A"
        print(f"{row['world']:<10} {row['model']:<30} {row['log_evidence']:>10.2f} {r2_is:>8} {r2_oos:>8} {oos_str:>10} {mae_str:>10} {ari_str:>6} {row.get('p_ci', 'N/A'):>12} {row['status']:>8}")

    print("-" * 100)
    
    # Best by world
    print(f"\nBEST MODEL BY WORLD (by log-evidence):")
    for world in worlds:
        world_df = stacked[stacked['world'] == world]
        if not world_df.empty:
            best = world_df.loc[world_df['log_evidence'].idxmax()]
            print(f"  {world:<10}: {best['model']} (log_ev={best['log_evidence']:.2f}, OOS={best.get('oos_rmse', np.nan):.2f})")
    
    print(f"{'='*120}\n")
    
    return stacked


def extract_world_metrics(pkl_folder: Path, world_name: str) -> pd.DataFrame:
    """Extract metrics for a single world (helper function)."""
    pkl_files = sorted(pkl_folder.glob("smc_*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files in {pkl_folder}")
        return None
    
    print(f"Found {len(pkl_files)} models")
    
    rows = []
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            res = data['res']
            idata = data['idata']
            
            log_ev = res.get('log_evidence', np.nan)
            
            # Skip exploded for display but include in table
            status = "EXPLODED" if log_ev > 0 or not np.isfinite(log_ev) else "OK"
            
            # Parse filename
            fname = pkl_path.name.replace('smc_', '').replace('.pkl', '')
            parts = fname.split('_')
            
            K = [p for p in parts if p.startswith('K')][0] if any(p.startswith('K') for p in parts) else 'K?'
            model = [p for p in parts if p in ['POISSON', 'NBD', 'HURDLE', 'TWEEDIE']][0] if any(p in ['POISSON', 'NBD', 'HURDLE', 'TWEEDIE'] for p in parts) else 'UNKNOWN'
            glm_gam = 'GAM' if 'GAM' in parts else 'GLM'
            p_type = 'statep' if 'statep' in parts else ('p1.5' if 'p1.5' in parts else 'free')
            
            row = {
                'world': world_name,
                'file': pkl_path.name,
                'model': f"{K}-{model}-{glm_gam}-{p_type}",
                'K': int(K.replace('K', '')) if K != 'K?' else 0,
                'model_type': model,
                'glm_gam': glm_gam,
                'p_spec': p_type,
                'log_evidence': log_ev,
                'oos_rmse': res.get('oos_rmse', np.nan),
                'oos_mae': res.get('oos_mae', np.nan),
                'time_min': res.get('time_min', np.nan),
                'status': status
            }
            
            # ARI for K>1
            K_num = row['K']
            if K_num > 1 and status == "OK":
                # Load ground truth for ARI if available
                gt_csv = pkl_folder.parent / f"sim_{world_name}_N500_T100_seed42.csv"
                if gt_csv.exists():
                    gt = load_ground_truth(gt_csv)
                    if gt:
                        row['ari'] = compute_viterbi_ari(idata, gt['states'])
                    else:
                        row['ari'] = np.nan
                else:
                    row['ari'] = np.nan
            else:
                row['ari'] = np.nan
            
            # p-parameter CI
            if 'p' in idata.posterior and status == "OK":
                p_post = idata.posterior['p'].values
                if p_post.ndim == 2:
                    p_flat = p_post.flatten()
                    row['p_mean'] = np.mean(p_flat)
                    row['p_lo'] = np.percentile(p_flat, 2.5)
                    row['p_hi'] = np.percentile(p_flat, 97.5)
                    row['p_ci'] = f"{row['p_mean']:.2f} [{row['p_lo']:.2f}, {row['p_hi']:.2f}]"
                elif p_post.ndim == 3:
                    n_ch, n_dr, n_states = p_post.shape
                    p_flat = p_post.reshape(n_ch * n_dr, n_states)
                    p_means = np.mean(p_flat, axis=0)
                    p_lo = np.percentile(p_flat, 2.5, axis=0)
                    p_hi = np.percentile(p_flat, 97.5, axis=0)
                    row['p_mean'] = np.mean(p_means)
                    row['p_lo'] = np.min(p_lo)
                    row['p_hi'] = np.max(p_hi)
                    if n_states == 2:
                        row['p_ci'] = f"{p_means[0]:.2f}/{p_means[1]:.2f}"
                    else:
                        row['p_ci'] = f"{np.mean(p_means):.2f} [{np.min(p_lo):.2f},{np.max(p_hi):.2f}]"
            else:
                row['p_mean'] = row['p_lo'] = row['p_hi'] = np.nan
                row['p_ci'] = "N/A"
            
            rows.append(row)
            
        except Exception as e:
            print(f"  ERROR {pkl_path.name}: {str(e)[:50]}")
    
    return pd.DataFrame(rows) if rows else None

## ----

def compute_bayesian_r2(idata, y_obs, y_pred=None):
    """
    Compute Bayesian R² as Var(y_pred) / (Var(y_pred) + Var(residual))
    
    Parameters:
    -----------
    idata : InferenceData
        PyMC inference data with posterior predictive
    y_obs : np.ndarray
        Observed y values (flattened or matching y_pred shape)
    y_pred : np.ndarray, optional
        Posterior predictive mean. If None, extract from idata
    
    Returns:
    --------
    float : Bayesian R² (mean over posterior samples)
    """
    import numpy as np
    
    # Get posterior predictive if not provided
    if y_pred is None:
        if 'posterior_predictive' in idata and 'y' in idata.posterior_predictive:
            y_pred_samples = idata.posterior_predictive['y'].values
            y_pred = y_pred_samples.mean(axis=(0, 1))  # Average over chains/draws
        else:
            return np.nan
    
    # Flatten arrays
    y_obs_flat = y_obs.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaNs
    mask = ~(np.isnan(y_obs_flat) | np.isnan(y_pred_flat))
    if mask.sum() < 2:
        return np.nan
    
    y_obs_clean = y_obs_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    # Compute variances
    var_pred = np.var(y_pred_clean)
    var_resid = np.var(y_obs_clean - y_pred_clean)
    
    # Bayesian R²
    if var_pred + var_resid == 0:
        return np.nan
    
    r2 = var_pred / (var_pred + var_resid)
    
    return float(r2)


def compute_r2_from_pkl(pkl_path, in_sample=True):
    """
    Compute R² (Bayesian) from pickle file.
    
    Parameters:
    -----------
    pkl_path : Path
        Path to .pkl file
    in_sample : bool
        If True, compute on training data. If False, use OOS data
    
    Returns:
    --------
    tuple : (r2_bayesian, r2_ols) or (np.nan, np.nan) if fails
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        idata = data['idata']
        res = data['res']
        
        # Get observed y
        if in_sample:
            # Training data
            if 'y' in res:
                y_obs = res['y']
            elif 'y' in idata.observed_data:
                y_obs = idata.observed_data['y'].values
            else:
                return np.nan, np.nan
        else:
            # OOS data
            if 'y_test' in res:
                y_obs = res['y_test']
            else:
                return np.nan, np.nan
        
        # Get predicted y (posterior predictive mean)
        y_pred = None
        if 'posterior_predictive' in idata:
            pp = idata.posterior_predictive
            if 'y' in pp:
                y_pred_samples = pp['y'].values
                y_pred = y_pred_samples.mean(axis=(0, 1))
        
        if y_pred is None:
            return np.nan, np.nan
        
        # Flatten for comparison
        y_obs_flat = y_obs.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Mask valid observations
        mask = ~(np.isnan(y_obs_flat) | np.isnan(y_pred_flat))
        if mask.sum() < 2:
            return np.nan, np.nan
        
        y_obs_clean = y_obs_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        # Bayesian R²
        var_pred = np.var(y_pred_clean)
        var_resid = np.var(y_obs_clean - y_pred_clean)
        r2_bayesian = var_pred / (var_pred + var_resid) if (var_pred + var_resid) > 0 else np.nan
        
        # OLS R² (for comparison)
        ss_res = np.sum((y_obs_clean - y_pred_clean)**2)
        ss_tot = np.sum((y_obs_clean - np.mean(y_obs_clean))**2)
        r2_ols = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        return float(r2_bayesian), float(r2_ols)
        
    except Exception as e:
        print(f"  R² computation failed: {str(e)[:50]}")
        return np.nan, np.nan


## ----

def extract_parameter_recovery(pkl_folder: str, 
                               ground_truth_csv: str,
                               world_name: str = "unknown") -> pd.DataFrame:
    """
    Extract parameter recovery metrics for K>1 HMM models.
    Compares estimated parameters against ground truth.
    
    Parameters:
    -----------
    pkl_folder : str
        Path to folder containing .pkl files
    ground_truth_csv : str
        Path to ground truth CSV with true_state column
    world_name : str
        Identifier for this simulation world
    
    Returns:
    --------
    pd.DataFrame with parameter recovery metrics
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pickle
    
    folder = Path(pkl_folder)
    pkl_files = sorted(folder.glob("smc_K*_TWEEDIE_*_statep_*.pkl"))  # Only K>1 state-varying
    
    if not pkl_files:
        print(f"No K>1 state-varying Tweedie models found in {pkl_folder}")
        return pd.DataFrame()
    
    # Load ground truth
    gt = load_ground_truth(Path(ground_truth_csv))
    if gt is None:
        print("Could not load ground truth")
        return pd.DataFrame()
    
    # Extract true transition matrix from ground truth states
    true_states = gt['states']
    N, T = true_states.shape
    K_true = len(np.unique(true_states))
    
    # Compute true Gamma
    true_Gamma = np.zeros((K_true, K_true))
    for t in range(T-1):
        for i in range(N):
            s_from = int(true_states[i, t])
            s_to = int(true_states[i, t+1])
            true_Gamma[s_from, s_to] += 1
    row_sums = true_Gamma.sum(axis=1, keepdims=True)
    true_Gamma = np.divide(true_Gamma, row_sums, where=row_sums>0)
    
    print(f"\n{'='*100}")
    print(f"PARAMETER RECOVERY: {world_name.upper()} WORLD")
    print(f"{'='*100}")
    print(f"Ground truth: N={N}, T={T}, K={K_true}")
    print(f"True transition matrix (empirical):")
    for i in range(K_true):
        row_str = "  ".join([f"{true_Gamma[i,j]:.3f}" for j in range(K_true)])
        print(f"  State {i}: {row_str}")
    print(f"True persistence (diagonal): " + "  ".join([f"{true_Gamma[i,i]:.3f}" for i in range(K_true)]))
    print()
    
    results = []
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            res = data['res']
            idata = data['idata']
            
            # Skip exploded
            log_ev = res.get('log_evidence', np.nan)
            if log_ev > 0 or not np.isfinite(log_ev):
                print(f"  SKIP (exploded): {pkl_path.name}")
                continue
            
            K_est = res.get('K', 0)
            if K_est <= 1:
                continue
            
            # Parse model info
            fname = pkl_path.name
            glm_gam = 'GAM' if 'GAM' in fname else 'GLM'
            
            row = {
                'world': world_name,
                'file': pkl_path.name,
                'model': f"K{K_est}-TWEEDIE-{glm_gam}-statep",
                'K_est': K_est,
                'K_true': K_true,
                'log_evidence': log_ev,
            }
            
            # Extract estimated Gamma
            if 'Gamma' in idata.posterior:
                Gamma_post = idata.posterior['Gamma'].values  # (chains, draws, K, K)
                Gamma_est = Gamma_post.mean(axis=(0, 1))  # Mean over posterior
                Gamma_std = Gamma_post.std(axis=(0, 1))
                Gamma_lo = np.percentile(Gamma_post.reshape(-1, K_est, K_est), 2.5, axis=0)
                Gamma_hi = np.percentile(Gamma_post.reshape(-1, K_est, K_est), 97.5, axis=0)
                
                # Diagonal (persistence)
                persist_est = np.diag(Gamma_est)
                persist_lo = np.diag(Gamma_lo)
                persist_hi = np.diag(Gamma_hi)
                
                row['persistence_mean'] = np.mean(persist_est)
                row['persistence_str'] = "/".join([f"{p:.2f}" for p in persist_est])
                row['persistence_ci'] = "/".join([f"[{l:.2f},{h:.2f}]" for l, h in zip(persist_lo, persist_hi)])
                
                # Compare to true (if K matches)
                if K_est == K_true:
                    persist_true = np.diag(true_Gamma)
                    persist_rmse = np.sqrt(np.mean((persist_est - persist_true)**2))
                    row['persistence_rmse'] = persist_rmse
                else:
                    row['persistence_rmse'] = np.nan
            else:
                row['persistence_mean'] = np.nan
                row['persistence_str'] = "N/A"
            
            # Extract beta0 (state-specific intercepts)
            if 'beta0' in idata.posterior:
                beta0_post = idata.posterior['beta0'].values  # (chains, draws, K)
                beta0_est = beta0_post.mean(axis=(0, 1))
                beta0_lo = np.percentile(beta0_post.reshape(-1, K_est), 2.5, axis=0)
                beta0_hi = np.percentile(beta0_post.reshape(-1, K_est), 97.5, axis=0)
                
                row['beta0_mean'] = np.mean(beta0_est)
                row['beta0_str'] = "/".join([f"{b:.2f}" for b in beta0_est])
                row['beta0_ci'] = "/".join([f"[{l:.2f},{h:.2f}]" for l, h in zip(beta0_lo, beta0_hi)])
            else:
                row['beta0_mean'] = np.nan
                row['beta0_str'] = "N/A"
            
            # Extract p (power parameter)
            if 'p' in idata.posterior:
                p_post = idata.posterior['p'].values
                if p_post.ndim == 3:  # State-specific: (chains, draws, K)
                    p_flat = p_post.reshape(-1, K_est)
                    p_est = p_flat.mean(axis=0)
                    p_lo = np.percentile(p_flat, 2.5, axis=0)
                    p_hi = np.percentile(p_flat, 97.5, axis=0)
                    
                    row['p_mean'] = np.mean(p_est)
                    row['p_str'] = "/".join([f"{p:.2f}" for p in p_est])
                    row['p_ci'] = "/".join([f"[{l:.2f},{h:.2f}]" for l, h in zip(p_lo, p_hi)])
                else:  # Shared p
                    p_flat = p_post.flatten()
                    row['p_mean'] = np.mean(p_flat)
                    row['p_str'] = f"{row['p_mean']:.2f}"
                    row['p_ci'] = f"[{np.percentile(p_flat, 2.5):.2f},{np.percentile(p_flat, 97.5):.2f}]"
            else:
                row['p_mean'] = np.nan
                row['p_str'] = "N/A"
            
            # Extract phi (dispersion)
            if 'phi' in idata.posterior:
                phi_post = idata.posterior['phi'].values
                if phi_post.ndim == 3:  # State-specific
                    phi_flat = phi_post.reshape(-1, K_est)
                    phi_est = phi_flat.mean(axis=0)
                    phi_lo = np.percentile(phi_flat, 2.5, axis=0)
                    phi_hi = np.percentile(phi_flat, 97.5, axis=0)
                    
                    row['phi_mean'] = np.mean(phi_est)
                    row['phi_str'] = "/".join([f"{p:.2f}" for p in phi_est])
                    row['phi_ci'] = "/".join([f"[{l:.2f},{h:.2f}]" for l, h in zip(phi_lo, phi_hi)])
                else:  # Shared phi
                    phi_flat = phi_post.flatten()
                    row['phi_mean'] = np.mean(phi_flat)
                    row['phi_str'] = f"{row['phi_mean']:.2f}"
                    row['phi_ci'] = f"[{np.percentile(phi_flat, 2.5):.2f},{np.percentile(phi_flat, 97.5):.2f}]"
            else:
                row['phi_mean'] = np.nan
                row['phi_str'] = "N/A"
            
            # ARI for state recovery
            row['ari'] = compute_viterbi_ari(idata, true_states)
            
            results.append(row)
            
            # Print summary
            print(f"  ✓ {row['model']:<30s} | log_ev={log_ev:10.2f} | ARI={row['ari']:.3f}")
            print(f"      Persistence: {row['persistence_str']} (true: " + "/".join([f"{true_Gamma[i,i]:.2f}" for i in range(min(K_true, 3))]) + ")")
            print(f"      p: {row['p_str']} | phi: {row['phi_str']}")
            print(f"      beta0: {row['beta0_str']}")
            if not np.isnan(row.get('persistence_rmse', np.nan)):
                print(f"      Persistence RMSE: {row['persistence_rmse']:.4f}")
            print()
            
        except Exception as e:
            print(f"  ✗ ERROR {pkl_path.name}: {str(e)[:60]}")
    
    df = pd.DataFrame(results)
    if df.empty:
        return df
    
    print(f"{'='*100}")
    print(f"SUMMARY: {len(results)} models processed")
    print(f"{'='*100}")
    
    return df


def extract_parameter_recovery_all_worlds(base_path: str, 
                                          worlds: list = ["poisson", "sporadic", "gamma", "clumpy"]) -> pd.DataFrame:
    """
    Extract parameter recovery for all 4 worlds and stack results.
    """
    all_results = []
    
    for world in worlds:
        pkl_folder = Path(base_path) / world / "harness_n500_d800"
        gt_csv = Path(base_path) / world / f"sim_{world}_N500_T100_seed42.csv"
        
        if not pkl_folder.exists() or not gt_csv.exists():
            print(f"Skipping {world}: missing files")
            continue
        
        df = extract_parameter_recovery(str(pkl_folder), str(gt_csv), world)
        if df is not None and not df.empty:
            all_results.append(df)
    
    if not all_results:
        return pd.DataFrame()
    
    stacked = pd.concat(all_results, ignore_index=True)
    
    # Final summary table
    print(f"\n{'='*120}")
    print("STACKED PARAMETER RECOVERY: ALL WORLDS")
    print(f"{'='*120}")
    print(f"{'World':<10} {'Model':<30} {'Log-Ev':>10} {'ARI':>6} {'Persistence':>25} {'p':>20} {'phi':>15}")
    print("-" * 120)
    
    for _, row in stacked.iterrows():
        print(f"{row['world']:<10} {row['model']:<30} {row['log_evidence']:>10.2f} {row['ari']:>6.3f} {str(row.get('persistence_str', 'N/A')):>25} {str(row.get('p_str', 'N/A')):>20} {str(row.get('phi_str', 'N/A')):>15}")
    
    print(f"{'='*120}\n")
    
    return stacked


## ----

# =============================================================================
# MAIN EXECUTION BLOCK (Drop-in Replacement)
# =============================================================================
if __name__ == "__main__":
    import argparse
    import pandas as pd
    import numpy as np
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Post-process simulation results with ARI and R²')
    parser.add_argument('--pkl_folder', default=None, help='Folder containing .pkl files')
    parser.add_argument('--gt_csv', default=None, help='Ground truth CSV (for ARI/States)')
    parser.add_argument('--output', default=None, help='Output CSV path')
    parser.add_argument('--world', default='unknown', help='Simulation world name')
    parser.add_argument('--base_path', default=None, help='Base path containing world folders (e.g., ./simul_21feb)')
    
    # Mode Flags
    parser.add_argument('--extract_all_worlds', action='store_true', help='Extract metrics from all 4 worlds')
    parser.add_argument('--extract_world', action='store_true', help='Extract metrics from single world')
    parser.add_argument('--extract_recovery_all', action='store_true', help='Extract parameter recovery for all worlds')
    
    # R-Squared and OOS Recomputation
    parser.add_argument('--compute_r2', action='store_true', help='Compute Bayesian and OLS R2 metrics')
    parser.add_argument('--recompute_oos', action='store_true', help='Force recomputation of OOS metrics from CSV')
    parser.add_argument('--sim_csv', default=None, help='Simulation CSV for OOS recomputation')

    args = parser.parse_args()

    # 1. PARAMETER RECOVERY MODE (Recovery Narrative)
    if args.extract_recovery_all:
        if args.base_path is None:
            print("ERROR: --base_path required for --extract_recovery_all")
            exit(1)
        worlds = ["poisson", "sporadic", "gamma", "clumpy"]
        df = extract_parameter_recovery_all_worlds(args.base_path, worlds)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved Parameter Recovery to: {args.output}")
        exit(0)


    # Branch 3 & 4: Extract all worlds or single world (Comprehensive Metrics)
    if args.extract_all_worlds or args.extract_world:
        worlds = ["poisson", "sporadic", "gamma", "clumpy"] if args.extract_all_worlds else [args.world]
        all_dfs = []
        
        print(f"\nSTARTING COMPREHENSIVE EXTRACTION (R2={args.compute_r2})")
        
        for w in worlds:
            # 1. Resolve World Root (e.g., ./simul_21feb/sporadic)
            w_root = Path(args.base_path) / w if args.base_path else Path(args.pkl_folder)
            
            # 2. Find any .pkl files in subdirectories (handles /harness_n500_d800/ or /duel/)
            # This looks one level deeper to find the actual pkl location
            all_pkls = list(w_root.glob("**/smc_*.pkl"))
            
            if not all_pkls:
                print(f"  !! No .pkl files found in {w_root} or its subfolders. Skipping.")
                continue
                
            # Use the directory where the .pkl files actually live
            w_pkl_folder = all_pkls[0].parent
            print(f"\nProcessing {w.upper()}:")
            print(f"  -> Folder: {w_pkl_folder.relative_to(Path(args.base_path).parent if args.base_path else '.')}")

            # 3. Find the sim CSV for R2 calculation
            sim_files = list(w_root.glob(f"sim_{w}_*.csv"))
            target_gt_csv = str(sim_files[0]) if sim_files else None
            
            y_train, y_test = None, None
            if args.compute_r2 and target_gt_csv:
                print(f"  -> Data:   {Path(target_gt_csv).name}")
                # Using your N=500, T=80/20 convention
                y_train, y_test, _, _ = load_simulation_data(target_gt_csv, n_cust=500, train_frac=0.8)

            # 4. Extract
            df_w = extract_simulation_metrics(
                pkl_folder=str(w_pkl_folder),
                world_name=w,
                ground_truth_csv=target_gt_csv,
                y_train=y_train,
                y_test=y_test
            )
            all_dfs.append(df_w)

        # Final check to prevent the sort_values crash
        if not all_dfs:
            print("\nERROR: No data extracted from any folders. Check your --base_path.")
            exit(1)

        df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by world and then by Log-Evidence
        df = df.sort_values(['world', 'log_evidence'], ascending=[True, False])
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n{'='*70}\nFINAL SUMMARY SAVED TO: {args.output}\n{'='*70}")
        else:
            print(df.to_string())
        
        exit(0)


    # 3. FALLBACK: STANDARD SINGLE-FOLDER PROCESSING
    if args.pkl_folder:
        df = process_simulation_folder(args.pkl_folder, args.gt_csv, y_train=None, y_test=None)
        df.to_csv(args.output or "summary.csv", index=False)

