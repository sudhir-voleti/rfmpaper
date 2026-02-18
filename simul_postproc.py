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


def load_ground_truth(pkl_path):
    """Load ground truth states and metadata."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_posterior_states(result_pkl, N, T):
    """
    Extract posterior P(State=1) from SMC result.
    Returns array of shape (N, T) or None if not available.
    """
    try:
        with open(result_pkl, 'rb') as f:
            res = pickle.load(f)
        
        idata = res['idata']
        
        # Try to find state variable (z or states)
        if 'posterior' in dir(idata):
            post = idata.posterior
            if 'z' in post:
                # Average over chains and draws
                post_probs = post['z'].mean(dim=['chain', 'draw']).values
                return post_probs
            elif 'states' in post:
                post_probs = post['states'].mean(dim=['chain', 'draw']).values
                return post_probs
        
        # Fallback: check if stored in res dict
        if 'res' in res and 'posterior_probs' in res['res']:
            return res['res']['posterior_probs']
            
    except Exception as e:
        print(f"  Warning: Could not extract states from {result_pkl}: {e}")
    
    return None


def compute_recovery_metrics(true_states, post_probs, true_switch_day):
    """
    Compute state recovery metrics.
    
    Returns dict with:
    - state_accuracy: % correct state classification (threshold 0.5)
    - state_correlation: correlation between true and predicted probs
    - mean_timing_error: |estimated_switch - true_switch| for activators
    - brier_score: mean squared error of probabilities
    """
    metrics = {}
    
    if post_probs is None:
        return {'error': 'No posterior states found'}
    
    # Ensure shapes match
    if post_probs.shape != true_states.shape:
        return {'error': f'Shape mismatch: post_probs {post_probs.shape} vs true {true_states.shape}'}
    
    N, T = true_states.shape
    
    # 1. State accuracy (threshold 0.5)
    pred_states = (post_probs > 0.5).astype(int)
    metrics['state_accuracy'] = (pred_states == true_states).mean()
    
    # 2. State correlation
    metrics['state_correlation'] = np.corrcoef(
        true_states.flatten(), 
        post_probs.flatten()
    )[0, 1]
    
    # 3. Brier score (proper scoring rule)
    metrics['brier_score'] = np.mean((post_probs - true_states) ** 2)
    
    # 4. Timing error (for customers who activated)
    # Estimate switch day as first time P(Active) > 0.5
    est_switch_day = np.full(N, -1)
    for i in range(N):
        active_times = np.where(post_probs[i, :] > 0.5)[0]
        if len(active_times) > 0:
            est_switch_day[i] = active_times[0]
    
    activated = true_switch_day >= 0
    if activated.sum() > 0:
        timing_errors = np.abs(est_switch_day[activated] - true_switch_day[activated])
        metrics['mean_timing_error'] = timing_errors.mean()
        metrics['median_timing_error'] = np.median(timing_errors)
        
        # Activation detection accuracy
        true_activated = activated
        pred_activated = est_switch_day >= 0
        metrics['activation_precision'] = (true_activated & pred_activated).sum() / pred_activated.sum() if pred_activated.sum() > 0 else 0
        metrics['activation_recall'] = (true_activated & pred_activated).sum() / true_activated.sum() if true_activated.sum() > 0 else 0
    else:
        metrics['mean_timing_error'] = np.nan
        metrics['median_timing_error'] = np.nan
        metrics['activation_precision'] = 0
        metrics['activation_recall'] = 0
    
    return metrics

def extract_posterior_states(result_pkl, N, T):
    try:
        with open(result_pkl, 'rb') as f:
            res = pickle.load(f)
        
        idata = res['idata']
        
        # Try viterbi path first
        if 'posterior' in dir(idata):
            post = idata.posterior
            if 'viterbi' in post:
                # Use mode over chains/draws
                viterbi = post['viterbi'].mode(dim=['chain', 'draw']).values
                return viterbi
            elif 'post_probs' in post:
                probs = post['post_probs'].mean(dim=['chain', 'draw']).values
                return probs  # Shape (N, T, K)
            elif 'z' in post:
                return post['z'].mean(dim=['chain', 'draw']).values
        
    except Exception as e:
        print(f"  Error: {e}")
    
    return None
    
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
    
    # Find all result files
    result_files = glob.glob(f"{args.results_dir}/smc_*.pkl")
    print(f"\nFound {len(result_files)} result files")
    
    # Evaluate each model
    results_list = []
    
    for rf in sorted(result_files):
        model_name = Path(rf).stem
        print(f"\n{model_name}:")
        
        # Extract model info from filename
        parts = model_name.split('_')
        K = int([p for p in parts if p.startswith('K')][0].replace('K', ''))
        model_type = [p for p in parts if p in ['POISSON', 'NBD', 'HURDLE', 'TWEEDIE']][0]
        
        # Load log-evidence
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
        
        # Extract posterior states
        post_probs = extract_posterior_states(rf, N, T)
        
        if post_probs is not None:
            print(f"  Extracted posterior states: {post_probs.shape}")
            
            # Compute metrics
            metrics = compute_recovery_metrics(true_states, post_probs, true_switch_day)
            
            print(f"  State accuracy: {metrics.get('state_accuracy', 0):.3f}")
            print(f"  State correlation: {metrics.get('state_correlation', 0):.3f}")
            print(f"  Brier score: {metrics.get('brier_score', 0):.3f}")
            if 'mean_timing_error' in metrics and not np.isnan(metrics['mean_timing_error']):
                print(f"  Mean timing error: {metrics['mean_timing_error']:.1f} days")
            
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
            print(f"  ✗ Could not extract states")
    
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
                       'brier_score', 'mean_timing_error', 'log_evidence']
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
            best_accuracy = results_df.loc[results_df['state_accuracy'].idxmax()]
            print(f"Best state accuracy: {best_accuracy['model_type']} K={best_accuracy['K']} ({best_accuracy['state_accuracy']:.3f})")
        
        if 'brier_score' in results_df.columns:
            best_brier = results_df.loc[results_df['brier_score'].idxmin()]
            print(f"Best Brier score: {best_brier['model_type']} K={best_brier['K']} ({best_brier['brier_score']:.3f})")
        
        if 'log_evidence' in results_df.columns:
            best_logev = results_df.loc[results_df['log_evidence'].idxmax()]
            print(f"Best Log-Evidence: {best_logev['model_type']} K={best_logev['K']} ({best_logev['log_evidence']:.2f})")
        
        print("\nPaper story:")
        print("  If Tweedie K=2/3 has best state recovery but not best Log-Ev,")
        print("  then Tweedie-HMM provides superior structural insight")
        print("  despite not being the 'best fitting' model.")
        print("="*70)
    else:
        print("\nNo valid results to compare")


if __name__ == "__main__":
    main()
