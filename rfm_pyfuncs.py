# rfmpaper/rfm_funcs.py  (journal-ready, PEP-8, type-hinted)

from __future__ import annotations
import pathlib, pickle, pandas as pd, numpy as np, arviz as az, matplotlib.pyplot as plt, seaborn as sns
from scipy.special import gammaln, logsumexp
import pymc as pm
import pytensor.tensor as pt
from tqdm import tqdm

# ----------  1.  DATA LOADER  -----------------------------------------------
def load_panel_csv(csv_path: str | pathlib.Path, customer_col: str = "customer_id") -> dict:
    """
    Returns dict:  y, mask, R, F, M, N, T  (all NumPy arrays)
    """
    df = pd.read_csv(csv_path, parse_dates=["WeekStart"])
    df = df.astype({customer_col: str})
    cust = df[customer_col].unique()
    y   = df.pivot(index=customer_col, columns="WeekStart", values="WeeklySpend").loc[cust].values
    mask = ~np.isnan(y)
    R   = df.pivot(index=customer_col, columns="WeekStart", values="R_weeks").loc[cust].values
    F   = df.pivot(index=customer_col, columns="WeekStart", values="F_run").loc[cust].values
    M   = df.pivot(index=customer_col, columns="WeekStart", values="M_run").loc[cust].values
    return {"y": y, "mask": mask, "R": R, "F": F, "M": M, "N": y.shape[0], "T": y.shape[1]}

# ----------  2.  SMC RUNNER  -------------------------------------------------
def run_smc(data: dict, K: int, draws: int = 1000, chains: int = 4, seed: int = 42, out_dir: pathlib.Path | None = None) -> az.InferenceData:
    """
    Full-panel SMC for chosen K. Returns InferenceData.
    """
    import time, pathlib, os
    N, T = data["N"], data["T"]
    **old_code_block**  # paste your final SMC block here (no CLI inside)
    return idata

# ----------  3.  POST-PROCESS  ----------------------------------------------
def add_log_likelihood(idata: az.InferenceData, csv_path: str | pathlib.Path) -> az.InferenceData:
    """
    Rebuild point-wise log-likelihood if missing. Returns new InferenceData.
    """
    # paste your final helper here (down-sampled draws, pure NumPy)
    return idata

# ----------  4.  TABLES 5-9  -------------------------------------------------
def make_table5(idata: az.InferenceData, ds_name: str, out_dir: pathlib.Path | None = None) -> pd.DataFrame:
    gamma = idata.posterior["Gamma"].mean(("chain", "draw")).values
    k = gamma.shape[0]
    labs = [f"State {i}" for i in range(k)]
    df = pd.DataFrame(gamma, index=labs, columns=labs)
    df.insert(0, "From", labs)
    if out_dir:
        out_dir.mkdir(exist_ok=True)
        df.to_csv(out_dir / f"table5_{ds_name}.csv", index=False)
        plt.figure(figsize=(4, 3))
        sns.heatmap(df.set_index("From"), annot=True, fmt=".3f", cmap="Blues", cbar_kws={"label": "P(t+1|t)"})
        plt.title(f"{ds_name.upper()} – transition matrix")
        plt.tight_layout()
        plt.savefig(out_dir / f"gamma_{ds_name}.pdf")
        plt.close()
    return df

def make_table6(idata: az.InferenceData, ds_name: str, out_dir: pathlib.Path | None = None) -> pd.DataFrame:
    summ = az.summary(idata, var_names=["beta0", "phi"], hdi_prob=0.95)
    summ.insert(0, "Dataset", ds_name.upper())
    if out_dir:
        summ.to_csv(out_dir / f"table6_{ds_name}.csv", index=False)
    return summ

def make_table7(idata: az.InferenceData, ds_name: str, csv_path: str | pathlib.Path, out_dir: pathlib.Path | None = None, labels: list[str] | None = None) -> pd.DataFrame:
    # paste your final Viterbi + area-plot code here (down-sampled, padded states)
    return prop

def make_table8(idata: az.InferenceData, ds_name: str, out_dir: pathlib.Path | None = None) -> pd.DataFrame:
    loo  = az.loo(idata, pointwise=True)
    waic = az.waic(idata, pointwise=True)
    df = pd.DataFrame({
        "Dataset": [ds_name.upper()],
        "ELPD-LOO": loo.elpd_loo,
        "p_loo": loo.p_loo,
        "SE-LOO": loo.se,
        "WAIC": waic.elpd_waic,
        "SE-WAIC": waic.se,
    })
    if out_dir:
        df.to_csv(out_dir / f"table8_{ds_name}.csv", index=False)
    return df

def make_table9(idata: az.InferenceData, ds_name: str, out_dir: pathlib.Path | None = None, cost_ratio: float = 0.2, lift_pp: float = 0.05, weeks: int = 52, n_sim: int = 1000, labels: list[str] | None = None) -> pd.DataFrame:
    # paste your final ROI Monte-Carlo code here (down-sampled, Pr(ROI>0) added)
    return df

# ----------  5.  CLI ENTRY POINT  -------------------------------------------
def main():
    """
    One-line reproducibility:
    $ python -m rfmpaper
    → prompts for working folder, dataset, K-range → produces paper tables/figures
    """
    import argparse, pathlib, sys
    parser = argparse.ArgumentParser(description="Reproducible HMM-Tweedie paper pipeline")
    parser.add_argument("--folder", type=pathlib.Path, help="working folder (will create results/ inside)")
    parser.add_argument("--dataset", choices=["uci", "cdnow"], help="dataset")
    parser.add_argument("--k-start", type=int, default=2, help="min K")
    parser.add_argument("--k-end", type=int, default=6, help="max K")
    parser.add_argument("--draws", type=int, default=1000, help="SMC draws per chain")
    parser.add_argument("--chains", type=int, default=4, help="SMC chains")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    if not args.folder:
        args.folder = pathlib.Path(input("Working folder path: ")).resolve()
    if not args.dataset:
        args.dataset = input("Dataset (uci/cdnow): ").strip().lower()

    csv_path = args.folder / f"{args.dataset}_full.csv"
    data = load_panel_csv(csv_path)
    out_dir = args.folder / "results"
    out_dir.mkdir(exist_ok=True)

    results = []
    for K in range(args.k_start, args.k_end + 1):
        print(f"\n====  K = {K}  ====")
        idata = run_smc(data, K, args.draws, args.chains, args.seed, out_dir)
        idata = add_log_likelihood(idata, csv_path)
        tbl5 = make_table5(idata, args.dataset, out_dir)
        tbl6 = make_table6(idata, args.dataset, out_dir)
        tbl7 = make_table7(idata, args.dataset, csv_path, out_dir)
        tbl8 = make_table8(idata, args.dataset, out_dir)
        tbl9 = make_table9(idata, args.dataset, out_dir)
        log_ev = float(az.summary(idata, var_names=["log_marginal_likelihood"])["mean"].iloc[0])
        results.append({"K": K, "log_evidence": log_ev})
        print(f"  log-ev = {log_ev:.3f}")

    df_tbl = pd.DataFrame(results).set_index("K")
    df_tbl.to_csv(out_dir / "table4.csv")
    print("\nAll tables & figures saved to", out_dir.resolve())

if __name__ == "__main__":
    main()
