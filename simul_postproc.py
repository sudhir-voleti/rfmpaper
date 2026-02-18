#!/usr/bin/env python3
"""
simul_postproc.py
=================
Post-processing for simulation study: evaluate state recovery vs ground truth

Usage:
    python simul_postproc.py --ground_truth ground_truth.pkl --results_dir ./results
"""

import numpy as np
import pandas as pd
import pickle
import glob
import argparse
from pathlib import Path
from scipy import stats


def load_ground_truth(pkl_path):
    """Load ground truth states and metadata."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_viterbi_states(result_pkl, N, T):
    """
    Extract Viterbi path from SMC result.
    Returns array of shape (N, T) or None if not available.
    """
    try:
        with open(result_pkl, 'rb') as f:
            res = pickle.load(f)
        
        idata = res['idata']
        
        if 'posterior' in dir(idata) and 'viterbi' in idata.posterior:
            viterbi = idata.posterior['viterbi']
            
            # Handle different shapes
            # Expected: (chain, draw, N, T) or (N, T)
            if 'chain' in viterbi.dims and 'draw' in viterbi.dims:
                # Take mode over chains and draws
                vals = viterbi.values
                # Reshape to combine chain and draw, then take mode
                n_chains, n_draws, n_cust, n_time = vals.shape
                vals_reshaped = vals.reshape(n_chains * n_draws, n_cust, n_time)
                viterbi_mode = stats.mode(vals_reshaped, axis=0)[0].squeeze()
                return viterbi_mode
            else:
                # Already (N, T)
                return viterbi.values
        
        return None
        
    except Exception as e:
        print(f"  Warning: Could not extract viterbi from {result_pkl}: {e}")
        return None


def compute_recovery_metrics(true_states, pred_states, true_switch_day):
    """
    Compute state recovery metrics.
    """
    metrics = {}
    
    if pred_states is None:
        return {'error': 'No predicted states'}
    
    # Ensure shapes match
    if pred_states.shape != true_states.shape:
        return {'error': f'Shape mismatch: pred {pred_states.shape} vs true {true_states.shape}'}
    
    N, T = true_states.shape
    
    # 1. State accuracy (overall)
    metrics['state_accuracy'] = (pred_states == true_states).mean()
    
    # 2. State correlation (treat as continuous 0/1)
    metrics['state_correlation'] = np.corrcoef(
        true_states.flatten(), 
        pred_states.flatten()
    )[0, 1]
    
    # 3. Per-state accuracy
    for state in [0, 1]:
        mask = true_states == state
        if mask.sum() > 0:
            metrics[f'accuracy_state_{state}'] = (pred_states[mask] == state).mean()
    
    # 4. Timing error (for customers who activated)
    # Estimate switch day from predicted states
    est_switch_day = np.full(N, -1)
    for i in range(N):
        active_times = np.where(pred_states[i, :] == 1)[0]
        if len(active_times) > 0:
            est_switch_day[i] = active_times[0]
    
    activated = true_switch_day >= 0
    if activated.sum() > 0:
        timing_errors = np.abs(est_switch_day[activated] - true_switch_day[activated])
        metrics['mean_timing_error'] = timing_errors.mean()
        metrics['median_timing_error'] = np.median(timing_errors)
        
        # Activation detection
        true_activated = activated
        pred_activated = est_switch_day >= 0
        metrics['activation_precision'] = (true_activated & pred_activated).sum() / pred_activated.sum() if pred_activated.sum() > 0 else 0
        metrics['activation_recall'] = (true_activated & pred_activated).sum() / true_activated.sum() if true_activated.sum() > 0 else 0
    else:
        metrics['mean_timing_error'] = np.nan
        metrics['activation_recall'] = 0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate simulation recovery')
    parser.add_argument('--ground_truth', type=str, default='ground_truth.pkl',
                        help='Path to ground truth pickle')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory with SMC result .pkl files')
    parser.add_argument('--output_csv', type=str, default='recovery_metrics.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SIMULATION RECOVERY EVALUATION")
    print("="*70)
    
    # Load ground truth
    print(f"\nLoading ground truth: {args.ground_truth}")
    truth = load_ground_truth(args.ground_truth)
    
    true_states = truth['states_matrix']
    true_switch_day = truth['true_switch_day']
    N, T = truth['N'], truth['T']
    
    print(f"Ground truth: {N} customers x {T} periods")
    print(f"True activation rate: {true_states.mean():.1%}")
    print(f"Mean switch day: {true_switch_day[true_switch_day >= 0].mean():.1f}")
    print(f"DGP type: {truth.get('dgp_type', 'unknown')}")
    
    # Find all result files
    result_files = glob.glob(f"{args.results_dir}/smc_*.pkl")
    print(f"\nFound {len(result_files)} result files")
    
    # Evaluate each model
    results_list = []
    
    for rf in sorted(result_files):
        model_name = Path(rf).stem
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print('='*70)
        
        # Extract model info from filename
        parts = model_name.split('_')
        try:
            K = int([p for p in parts if p.startswith('K')][0].replace('K', ''))
        except:
            K = 'unknown'
        model_type = [p for p in parts if p in ['POISSON', 'NBD', 'HURDLE', 'TWEEDIE']][0] if any(p in parts for p in ['POISSON', 'NBD', 'HURDLE', 'TWEEDIE']) else 'unknown'
        
        # Load log-evidence and timing
        try:
            with open(rf, 'rb') as f:
                res = pickle.load(f)
            log_ev = res['res']['log_evidence']
            time_min = res['res']['time_min']
            print(f"  Log-Ev: {log_ev:.2f}, Time: {time_min:.1f}min")
        except:
            log_ev = np.nan
            time_min = np.nan
            print(f"  Log-Ev: N/A")
        
        # Extract Viterbi states
        pred_states = extract_viterbi_states(rf, N, T)
        
        if pred_states is not None:
            print(f"  Viterbi shape: {pred_states.shape}")
            
            # Compute metrics
            metrics = compute_recovery_metrics(true_states, pred_states, true_switch_day)
            
            print(f"  State accuracy: {metrics.get('state_accuracy', 0):.3f}")
            print(f"  State correlation: {metrics.get('state_correlation', 0):.3f}")
            if 'mean_timing_error' in metrics and not np.isnan(metrics['mean_timing_error']):
                print(f"  Mean timing error: {metrics['mean_timing_error']:.1f} days")
            if 'activation_recall' in metrics:
                print(f"  Activation recall: {metrics['activation_recall']:.2f}")
            
            # Store results
            result_row = {
                'model': model_name,
                'K': K,
                'model_type': model_type,
                'log_evidence': log_ev,
                'time_min': time_min,
                **metrics
            }
            results_list.append(result_row)
        else:
            print(f"  ✗ Could not extract Viterbi states")
    
    # Create comparison table
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        # Sort by model type and K
        type_order = {'POISSON': 0, 'NBD': 1, 'HURDLE': 2, 'TWEEDIE': 3}
        results_df['type_order'] = results_df['model_type'].map(type_order)
        results_df = results_df.sort_values(['type_order', 'K'])
        
        print("\n" + "="*70)
        print("RECOVERY METRICS COMPARISON")
        print("="*70)
        
        # Select key columns for display
        display_cols = ['model_type', 'K', 'state_accuracy', 'state_correlation', 
                       'mean_timing_error', 'activation_recall', 'log_evidence']
        available_cols = [c for c in display_cols if c in results_df.columns]
        
        print(results_df[available_cols].to_string(index=False))
        
        # Save to CSV
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved detailed results to: {args.output_csv}")
        
        # Summary interpretation
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        
        # Find best by different metrics
        if 'state_accuracy' in results_df.columns:
            best_idx = results_df['state_accuracy'].idxmax()
            best = results_df.loc[best_idx]
            print(f"Best state accuracy: {best['model_type']} K={best['K']} ({best['state_accuracy']:.3f})")
        
        if 'log_evidence' in results_df.columns:
            best_idx = results_df['log_evidence'].idxmax()
            best = results_df.loc[best_idx]
            print(f"Best Log-Evidence: {best['model_type']} K={best['K']} ({best['log_evidence']:.2f})")
        
        # The key test
        print("\n" + "="*70)
        print("KEY TEST: Does Tweedie-HMM have better state accuracy")
        print("         but worse Log-Evidence than NBD?")
        print("="*70)
        
        tweedie_k2 = results_df[(results_df['model_type'] == 'TWEEDIE') & (results_df['K'] == 2)]
        nbd_k1 = results_df[(results_df['model_type'] == 'NBD') & (results_df['K'] == 1)]
        
        if not tweedie_k2.empty and not nbd_k1.empty:
            t_acc = tweedie_k2['state_accuracy'].values[0]
            n_acc = nbd_k1['state_accuracy'].values[0]
            t_ev = tweedie_k2['log_evidence'].values[0]
            n_ev = nbd_k1['log_evidence'].values[0]
            
            print(f"\nTweedie K=2: Acc={t_acc:.3f}, LogEv={t_ev:.2f}")
            print(f"NBD K=1:     Acc={n_acc:.3f}, LogEv={n_ev:.2f}")
            
            if t_acc > n_acc and t_ev < n_ev:
                print("\n✓✓✓ SUCCESS: Tweedie-HMM recovers states better despite worse fit!")
                print("   This proves structural insight value.")
            elif t_acc > n_acc and t_ev > n_ev:
                print("\n✓ Tweedie wins on both (unexpected but good)")
            elif t_acc < n_acc and t_ev < n_ev:
                print("\n✗ Tweedie loses on both (story fails)")
            else:
                print("\n? Mixed results (needs investigation)")
        
    else:
        print("\nNo valid results to compare")


if __name__ == "__main__":
    main()
