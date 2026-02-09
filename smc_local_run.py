#!/usr/bin/env python3
"""
smc_local_run.py - Modified with dataset prefix in filenames
"""
import argparse
import pickle
import sys
from pathlib import Path
import urllib.request

# Download smc_unified if not present locally
def fetch_dependency():
    local_unified = Path("smc_unified.py")
    if not local_unified.exists():
        print("Fetching smc_unified.py from GitHub...")
        url = "https://raw.githubusercontent.com/sudhir-voleti/rfmpaper/main/smc_unified.py"
        urllib.request.urlretrieve(url, local_unified)
        print("Downloaded smc_unified.py")

fetch_dependency()

from smc_unified import load_data, build_model, run_smc

def main():
    parser = argparse.ArgumentParser(description='Run SMC for RFM-HMM-Tweedie')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['uci', 'cdnow', 'UCI', 'CDNOW'],
                       help='Dataset name (uci or cdnow)')
    parser.add_argument('-k', '--k', type=int, required=True,
                       help='Number of states (1 for static)')
    parser.add_argument('-n', '--n', type=int, default=500,
                       help='Sample size')
    parser.add_argument('--draws', type=int, default=500,
                       help='Number of SMC draws')
    parser.add_argument('-c', '--cores', type=int, default=4,
                       help='Number of cores')
    parser.add_argument('--p', type=float, default=None,
                       help='Fixed p value (e.g., 1.5). If omitted, estimates p freely.')
    parser.add_argument('--state-specific-p', action='store_true',
                       help='Use state-specific p (for K>1 only)')
    parser.add_argument('--no-gam', action='store_true',
                       help='Use GLM instead of GAM (linear only)')
    
    args = parser.parse_args()
    
    # Prepare paths
    data_dir = Path("/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/data")
    results_dir = Path("/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/results")
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading {args.dataset} data...")
    df = load_data(args.dataset, data_dir, n=args.n)
    
    # Build model
    print(f"Building model: K={args.k}, GAM={not args.no_gam}, varying_p={args.p is None}")
    model = build_model(
        df, 
        K=args.k,
        use_gam=not args.no_gam,
        fixed_p=args.p,
        state_specific_p=args.state_specific_p and args.k > 1
    )
    
    # Run SMC
    print(f"Running SMC with {args.draws} draws, {args.cores} cores...")
    idata = run_smc(model, draws=args.draws, cores=args.cores)
    
    # CRITICAL FIX: Include dataset name in filename
    dataset_prefix = args.dataset.lower()
    k_str = f"K{args.k}"
    
    if args.state_specific_p and args.k > 1:
        p_str = "statep"
    elif args.p is not None:
        p_str = f"fixedp{args.p:.2f}".replace('.', '')
    else:
        p_str = "varyingp"
    
    model_str = "GLM" if args.no_gam else "GAM"
    
    filename = f"smc_{dataset_prefix}_{k_str}_{model_str}_{p_str}_N{args.n}_D{args.draws}_C{args.cores}.pkl"
    output_path = results_dir / filename
    
    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(idata, f)
    
    print(f"\nSaved to: {output_path}")
    
    # Quick summary
    if hasattr(idata, 'sample_stats') and 'log_evidence' in idata.sample_stats:
        log_ev = float(idata.sample_stats['log_evidence'].mean())
        print(f"Log-evidence: {log_ev:.2f}")
    
    if hasattr(idata, 'posterior') and 'p' in idata.posterior:
        p_post = idata.posterior['p']
        p_mean = p_post.mean(dim=['chain', 'draw'])
        if p_mean.ndim > 0 and len(p_mean.shape) > 0:
            print(f"State-p estimates: {p_mean.values}")
        else:
            print(f"Global p estimate: {float(p_mean.values):.3f}")

if __name__ == "__main__":
    main()
