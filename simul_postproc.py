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


def compute_viterbi_ari(idata, ground_truth_states: np.ndarray) -> float:
    """
    Computes ARI between Viterbi path and simulation ground truth.
    """
    if 'viterbi' not in idata.posterior:
        return np.nan
    
    # Viterbi shape: (chain, draw, N, T)
    viterbi = idata.posterior['viterbi'].values
    
    # Reshape to (samples, N, T)
    n_samples = viterbi.shape[0] * viterbi.shape[1]
    v_paths = viterbi.reshape(n_samples, *viterbi.shape[2:])
    
    # Take mode across samples for each (customer, time)
    v_map = mode(v_paths, axis=0, keepdims=False).mode
    
    # Flatten and compute ARI
    return adjusted_rand_score(ground_truth_states.flatten(), v_map.flatten())


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

## ----

def process_simulation_folder(folder_path: str,
                              ground_truth_csv: Optional[str] = None,
                              metrics: List[str] = ['all'],
                              pattern: str = "smc_*.pkl") -> pd.DataFrame:
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
        results.append(metrics_dict)
    
    df = pd.DataFrame(results)
    
    # Sort by log-evidence
    if 'log_evidence' in df.columns:
        df = df.sort_values('log_evidence', ascending=False)
    
    return df


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


# Example usage
if __name__ == "__main__":
    # Example paths - modify to your setup
    pkl_folder = "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/runs_19feb/test_oos_v2"
    gt_csv = "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/runs_19feb/sim_4mode_v2_N300_T100_seed42.csv"
    
    df = process_simulation_folder(pkl_folder, gt_csv, metrics=['all'])
    display_results_table(df)
    
    # Save results
    output_csv = Path(pkl_folder) / "postproc_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved detailed results to: {output_csv}")


