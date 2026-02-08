#!/usr/bin/env python3
"""
smc_local_run.py
================
Local runner that fetches smc_unified.py from GitHub and executes with local settings.
"""
import sys
import pathlib
import subprocess

# ---------- LOCAL CONFIGURATION ----------
ROOT = pathlib.Path("/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/results")
DATA_DIR = pathlib.Path("/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/data")

GITHUB_URL = "https://github.com/sudhir-voleti/rfmpaper/blob/main/smc_unified.py"
EXEC_GITCODE = pathlib.Path(__file__).parent / "exec_gitcode.py"
# -----------------------------------------

def run_smc_from_github(dataset, K, n_cust, draws, chains, state_specific_p=False, no_gam=False):
    """Fetch smc_unified.py from GitHub and run with local settings."""
    
    ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, str(EXEC_GITCODE), GITHUB_URL,
        "--dataset", dataset,
        "--K", str(K),
        "--n_cust", str(n_cust),
        "--draws", str(draws),
        "--chains", str(chains),
        "--data_dir", str(DATA_DIR),
        "--out_dir", str(ROOT),
    ]
    
    if state_specific_p:
        cmd.append("--state_specific_p")
    if no_gam:
        cmd.append("--no_gam")
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def k_selection(dataset=dataset):
    """Run K-selection sequence."""
    
    configs = [
        # (K, state_specific_p, no_gam, n_cust, draws, description)
        (1, False, False, 500, 500, "Static Tweedie-GAM, varying p"),
        (2, True, False, 500, 500, "HMM K=2, state-specific p, GAM"),
        (3, True, False, 500, 500, "HMM K=3, state-specific p, GAM"),
        (4, True, False, 500, 500, "HMM K=4, state-specific p, GAM"),
    ]
    
    for K, state_p, no_gam, n_cust, draws, desc in configs:
        print(f"\n{'='*70}")
        print(f"Running: {desc}")
        print(f"{'='*70}")
        
        try:
            run_smc_from_github(
                dataset=dataset,
                K=K,
                n_cust=n_cust,
                draws=draws,
                chains=4,
                state_specific_p=state_p,
                no_gam=no_gam
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed: {e}")
            continue


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Local SMC runner with GitHub fetch')
    parser.add_argument("--mode", choices=['k_selection', 'single'], default='k_selection')
    parser.add_argument("--dataset", choices=['uci', 'cdnow'], default='uci')
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--state_specific_p", action="store_true")
    parser.add_argument("--n_cust", type=int, default=500)
    parser.add_argument("--draws", type=int, default=500)
    
    args = parser.parse_args()
    
    if args.mode == 'k_selection':
        k_selection(dataset=args.dataset)
    else:
        run_smc_from_github(
            dataset=args.dataset,
            K=args.K,
            n_cust=args.n_cust,
            draws=args.draws,
            chains=4,
            state_specific_p=args.state_specific_p
        )
