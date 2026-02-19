==================
Complete working HMM/Static models: Tweedie, Hurdle, Poisson (discrete), NBD (discrete)
Optimized for Apple Silicon (M1/M2/M3)

Usage:
    python smc_unified_new.py --dataset simulation --K 2 --model_type tweedie --no_gam
"""

# =============================================================================
# 0. APPLE SILICON OPTIMIZATION (MUST BE FIRST)
# =============================================================================
import os
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,optimizer=fast_run,openmp=True'

import pytensor
import numpy as np
import pytensor.tensor as pt

import pathlib
from pathlib import Path

print(f"PyTensor config: floatX={pytensor.config.floatX}, optimizer={pytensor.config.optimizer}")
os.environ['PYTENSOR_METAL'] = '0'

# =============================================================================
# STANDARD IMPORTS
# =============================================================================
import argparse
import time
import pathlib
import pickle
import warnings
import platform

import pandas as pd
import pymc as pm
import arviz as az
from patsy import dmatrix
from scipy.special import logsumexp

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

IS_APPLE_SILICON = platform.machine() == 'arm64' and platform.system() == 'Darwin'
if IS_APPLE_SILICON:
    print(f"Detected Apple Silicon ({platform.machine()}). Float32 optimization enabled.")


# =============================================================================
# 1. B-SPLINE BASIS FUNCTION
# =============================================================================
def create_bspline_basis(x, df=3, degree=3):
    """Create B-spline basis matrix for GAM."""
    x = np.asarray(x, dtype=np.float32).flatten()
    n_knots = df - degree + 1
    if n_knots > 1:
        knots = np.quantile(x, np.linspace(0, 1, n_knots)[1:-1]).tolist()
    else:
        knots = []
    formula = f"bs(x, knots={list(knots)}, degree={degree}, include_intercept=False)"
    basis = dmatrix(formula, {"x": x}, return_type='matrix')
    return np.asarray(basis, dtype=np.float32)

# =============================================================================
# 2. GAMMA LOG-DENSITY
# =============================================================================
def gamma_logp_det(value, mu, phi):
    """Deterministic Gamma log-density."""
    alpha = mu / phi
    beta = 1.0 / phi
    return (alpha - 1) * pt.log(value) - value * beta + alpha * pt.log(beta) - pt.gammaln(alpha)

# =============================================================================
# 3. MODEL BUILDERS
# =============================================================================

def make_model(data, K=3, state_specific_p=True, p_fixed=1.5, use_gam=True, gam_df=3):
    """Build HMM-Tweedie (K>=2) or Static Tweedie (K=1)."""
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]

    if use_gam and K >= 1:
        R_flat = R.flatten()
        F_flat = F.flatten()
        M_flat = M.flatten()
        basis_R = create_bspline_basis(R_flat, df=gam_df)
        basis_F = create_bspline_basis(F_flat, df=gam_df)
        basis_M = create_bspline_basis(M_flat, df=gam_df)
        n_basis_R = basis_R.shape[1]
        n_basis_F = basis_F.shape[1]
        n_basis_M = basis_M.shape[1]
        basis_R = basis_R.reshape(N, T, n_basis_R)
        basis_F = basis_F.reshape(N, T, n_basis_F)
        basis_M = basis_M.reshape(N, T, n_basis_M)
    else:
        n_basis_R = n_basis_F = n_basis_M = 1
        basis_R = basis_F = basis_M = None

    with pm.Model(coords={"customer": np.arange(N), "time": np.arange(T), "state": np.arange(K)}) as model:    
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
        else:
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K, dtype=np.float32), shape=(K, K))

        if K == 1:
            beta0_raw = pm.Normal("beta0_raw", 0, 1)
            beta0 = pm.Deterministic("beta0", beta0_raw)
            phi_raw = pm.Exponential("phi_raw", lam=10.0)
            phi = pm.Deterministic("phi", phi_raw)
        else:
            beta0_raw = pm.Normal("beta0_raw", 0, 1, shape=K)
            beta0 = pm.Deterministic("beta0", pt.sort(beta0_raw))
            phi_raw = pm.Exponential("phi_raw", lam=10.0, shape=K)
            #phi = pm.Deterministic("phi", pt.sort(phi_raw))
            phi = pm.Deterministic("phi", phi_raw) # Don't force sorting here

        if use_gam:
            if K == 1:
                w_R = pm.Normal("w_R", 0, 1, shape=n_basis_R)
                w_F = pm.Normal("w_F", 0, 1, shape=n_basis_F)
                w_M = pm.Normal("w_M", 0, 1, shape=n_basis_M)
            else:
                w_R = pm.Normal("w_R", 0, 1, shape=(K, n_basis_R))
                w_F = pm.Normal("w_F", 0, 1, shape=(K, n_basis_F))
                w_M = pm.Normal("w_M", 0, 1, shape=(K, n_basis_M))
        else:
            if K == 1:
                betaR = pm.Normal("betaR", 0, 1)
                betaF = pm.Normal("betaF", 0, 1)
                betaM = pm.Normal("betaM", 0, 1)
            else:
                betaR = pm.Normal("betaR", 0, 1, shape=K)
                betaF = pm.Normal("betaF", 0, 1, shape=K)
                betaM = pm.Normal("betaM", 0, 1, shape=K)

        if K == 1:
            p_raw = pm.Beta("p_raw", alpha=2, beta=2)
            p = pm.Deterministic("p", 1.1 + p_raw * 0.8)
        elif state_specific_p:
            p_raw = pm.Beta("p_raw", alpha=2, beta=2, shape=K)
            p_sorted = pt.sort(p_raw)
            p = pm.Deterministic("p", 1.1 + p_sorted * 0.8)
        else:
            p = pt.as_tensor_variable(np.array([p_fixed] * K, dtype=np.float32))

        if use_gam:
            if K == 1:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [0]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [0]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [0]))
                mu = pt.exp(beta0 + effect_R + effect_F + effect_M)
            else:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [1]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [1]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [1]))
                mu = pt.exp(beta0 + effect_R + effect_F + effect_M)
        else:
            if K == 1:
                mu = pt.exp(beta0 + betaR * R + betaF * F + betaM * M)
            else:
                mu = pt.exp(beta0 + betaR * R[..., None] + betaF * F[..., None] + betaM * M[..., None])

        mu = pt.clip(mu, 1e-3, 1e6)

        # --- 1. PARAMETER EXPANSION & STABLE EMISSION ---
        if K == 1:
            p_exp = p
            phi_exp = phi
            y_input = y
            mask_input = mask
        else:
            p_exp = p[None, None, :]
            phi_exp = phi[None, None, :]
            y_input = y[..., None]
            mask_input = mask[:, :, None]

        exponent = 2.0 - p_exp
        
        # Stable log-space calculation for Tweedie zero-mass
        # log_psi = - (mu^(2-p)) / (phi * (2-p))
        log_psi = -(pt.pow(mu, exponent) / (phi_exp * exponent))
        log_psi = pt.clip(log_psi, -100, -1e-12) # Prevents inf and stabilizes log_pos
        
        log_zero = log_psi
        # Stable log(1 - exp(x)) using log(-expm1(x))
        log_pos = pt.log(-pt.expm1(log_psi)) + gamma_logp_det(y_input, mu, phi_exp)
        
        log_emission = pt.where(y_input == 0, log_zero, log_pos)
        log_emission = pt.where(mask_input, log_emission, 0.0) 


        # --- 2. FORWARD ALGORITHM (HMM INTEGRATION) ---
        if K == 1:
            logp_cust = pt.sum(log_emission, axis=1)
            # Placeholder for K=1 consistency
            alpha_filtered = pt.ones((N, T, 1))
        else:
            # Initialize: P(S_0 | y_0)
            log_alpha = pt.log(pi0) + log_emission[:, 0, :]
            log_Gamma = pt.log(Gamma)[None, :, :]
            
            alpha_seq = [log_alpha]
            
            # Recursive pass
            for t in range(1, T):
                # temp = log P(S_{t-1} | y_{0:t-1}) + log P(S_t | S_{t-1})
                temp = log_alpha[:, :, None] + log_Gamma
                # log_alpha = log P(y_t | S_t) + log sum(temp)
                log_alpha = log_emission[:, t, :] + pt.logsumexp(temp, axis=1)
                alpha_seq.append(log_alpha)
            
            # Stack and Normalize to get P(S_t | y_{0:t})
            log_alpha_stacked = pt.stack(alpha_seq, axis=1)
            alpha_filtered = pt.exp(log_alpha_stacked - pt.logsumexp(log_alpha_stacked, axis=2, keepdims=True))
            
            # Total log-likelihood for the customer is the logsumexp of the final alpha
            logp_cust = pt.logsumexp(log_alpha, axis=1)

        # --- 3. DETERMINISTICS & POTENTIAL ---
        if K > 1:
            pm.Deterministic('alpha_filtered', alpha_filtered, dims=('customer', 'time', 'state'))

        pm.Potential('loglike', pt.sum(logp_cust))
        pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))



        # --- VITERBI BACKTRACK (MOST LIKELY STATE PATH) ---
        if K > 1:
            # log_delta[t, k] = max_{s_{1:t-1}} log P(s_{1:t-1}, s_t=k, y_{1:t})
            log_delta = pt.log(pi0) + log_emission[:, 0, :]
            log_Gamma = pt.log(Gamma)[None, :, :]
            
            # To backtrack, we need to store the argmax at each step
            psi_list = []
            
            for t in range(1, T):
                # We want: max over previous state 'j'
                # log_delta (N, j) -> log_delta[:, :, None] (N, j, 1)
                # log_Gamma (1, j, k)
                # log_emission (N, k) -> log_emission[:, t, :][:, None, :] (N, 1, k)
                
                # Probability of arriving at state k via state j
                p_matrix = log_delta[:, :, None] + log_Gamma
                
                # Which previous state j was the best for each current state k?
                best_prev_state = pt.argmax(p_matrix, axis=1) # (N, K)
                psi_list.append(best_prev_state)
                
                # Update log_delta for the next time step
                log_delta = log_emission[:, t, :] + pt.max(p_matrix, axis=1)

            # --- BACKTRACKING ---
            # We start from the end and work backwards
            viterbi_path_seq = []
            
            # The best final state for each customer
            last_state = pt.argmax(log_delta, axis=1)
            viterbi_path_seq.append(last_state)
            
            current_state = last_state
            # Walk back through the stored psi (argmax) matrices
            for t in range(T - 2, -1, -1):
                # Use advanced indexing to pick the best prev state 
                # for the current state we just found
                # psi_list[t] has shape (N, K)
                prev_state = psi_list[t][pt.arange(N), current_state]
                viterbi_path_seq.append(prev_state)
                current_state = prev_state

            # Reverse the sequence (since we walked backwards) and stack
            # Final shape: (N, T)
            viterbi_path = pt.stack(viterbi_path_seq[::-1], axis=1)
            
            pm.Deterministic('viterbi', viterbi_path, dims=('customer', 'time'))

        return model


## ---

def make_hurdle_model(data, K=3, use_gam=True, gam_df=3):
    """Hurdle-Gamma HMM model."""
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]

    if use_gam and K >= 1:
        R_flat = R.flatten()
        F_flat = F.flatten()
        M_flat = M.flatten()
        basis_R = create_bspline_basis(R_flat, df=gam_df)
        basis_F = create_bspline_basis(F_flat, df=gam_df)
        basis_M = create_bspline_basis(M_flat, df=gam_df)
        n_basis_R = basis_R.shape[1]
        n_basis_F = basis_F.shape[1]
        n_basis_M = basis_M.shape[1]
        basis_R = basis_R.reshape(N, T, n_basis_R)
        basis_F = basis_F.reshape(N, T, n_basis_F)
        basis_M = basis_M.reshape(N, T, n_basis_M)
    else:
        n_basis_R = n_basis_F = n_basis_M = 1
        basis_R = basis_F = basis_M = None

    with pm.Model(coords={"customer": np.arange(N), "time": np.arange(T), "state": np.arange(K)}) as model:
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
        else:
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K, dtype=np.float32), shape=(K, K))

        if K == 1:
            alpha0 = pm.Normal("alpha0", 0, 1)
            if use_gam:
                w_R_h = pm.Normal("w_R_h", 0, 1, shape=n_basis_R)
                w_F_h = pm.Normal("w_F_h", 0, 1, shape=n_basis_F)
                w_M_h = pm.Normal("w_M_h", 0, 1, shape=n_basis_M)
            else:
                alphaR = pm.Normal("alphaR", 0, 1)
                alphaF = pm.Normal("alphaF", 0, 1)
                alphaM = pm.Normal("alphaM", 0, 1)
        else:
            alpha0 = pm.Normal("alpha0", 0, 1, shape=K)
            if use_gam:
                w_R_h = pm.Normal("w_R_h", 0, 1, shape=(K, n_basis_R))
                w_F_h = pm.Normal("w_F_h", 0, 1, shape=(K, n_basis_F))
                w_M_h = pm.Normal("w_M_h", 0, 1, shape=(K, n_basis_M))
            else:
                alphaR = pm.Normal("alphaR", 0, 1, shape=K)
                alphaF = pm.Normal("alphaF", 0, 1, shape=K)
                alphaM = pm.Normal("alphaM", 0, 1, shape=K)

        if K == 1:
            beta0 = pm.Normal("beta0", 0, 1)
            phi = pm.Exponential("phi", lam=10.0)
            if use_gam:
                w_R = pm.Normal("w_R", 0, 1, shape=n_basis_R)
                w_F = pm.Normal("w_F", 0, 1, shape=n_basis_F)
                w_M = pm.Normal("w_M", 0, 1, shape=n_basis_M)
            else:
                betaR = pm.Normal("betaR", 0, 1)
                betaF = pm.Normal("betaF", 0, 1)
                betaM = pm.Normal("betaM", 0, 1)
        else:
            beta0 = pm.Normal("beta0", 0, 1, shape=K)
            phi = pm.Exponential("phi", lam=10.0, shape=K)
            if use_gam:
                w_R = pm.Normal("w_R", 0, 1, shape=(K, n_basis_R))
                w_F = pm.Normal("w_F", 0, 1, shape=(K, n_basis_F))
                w_M = pm.Normal("w_M", 0, 1, shape=(K, n_basis_M))
            else:
                betaR = pm.Normal("betaR", 0, 1, shape=K)
                betaF = pm.Normal("betaF", 0, 1, shape=K)
                betaM = pm.Normal("betaM", 0, 1, shape=K)

        if use_gam:
            if K == 1:
                effect_R_h = pt.tensordot(basis_R, w_R_h, axes=([2], [0]))
                effect_F_h = pt.tensordot(basis_F, w_F_h, axes=([2], [0]))
                effect_M_h = pt.tensordot(basis_M, w_M_h, axes=([2], [0]))
                logit_p_pos = alpha0 + effect_R_h + effect_F_h + effect_M_h
            else:
                effect_R_h = pt.tensordot(basis_R, w_R_h, axes=([2], [1]))
                effect_F_h = pt.tensordot(basis_F, w_F_h, axes=([2], [1]))
                effect_M_h = pt.tensordot(basis_M, w_M_h, axes=([2], [1]))
                logit_p_pos = alpha0 + effect_R_h + effect_F_h + effect_M_h
        else:
            if K == 1:
                logit_p_pos = alpha0 + alphaR * R + alphaF * F + alphaM * M
            else:
                logit_p_pos = alpha0 + alphaR * R[..., None] + alphaF * F[..., None] + alphaM * M[..., None]

        p_pos = pt.sigmoid(logit_p_pos)
        p_pos = pt.clip(p_pos, 1e-6, 1 - 1e-6)

        if use_gam:
            if K == 1:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [0]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [0]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [0]))
                mu = pt.exp(beta0 + effect_R + effect_F + effect_M)
            else:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [1]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [1]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [1]))
                mu = pt.exp(beta0 + effect_R + effect_F + effect_M)
        else:
            if K == 1:
                mu = pt.exp(beta0 + betaR * R + betaF * F + betaM * M)
            else:
                mu = pt.exp(beta0 + betaR * R[..., None] + betaF * F[..., None] + betaM * M[..., None])

        mu = pt.clip(mu, 1e-3, 1e6)

        if K == 1:
            log_zero = pt.log(1 - p_pos)
            log_pos = pt.log(p_pos) + gamma_logp_det(y, mu, phi)
            log_emission = pt.where(y == 0, log_zero, log_pos)
            log_emission = pt.where(mask, log_emission, 0.0)
        else:
            p_pos_exp = p_pos[..., None]
            mu_exp = mu[..., None]
            phi_exp = phi[None, None, :]
            log_zero = pt.log(1 - p_pos_exp)
            y_exp = y[..., None]
            log_pos = pt.log(p_pos_exp) + gamma_logp_det(y_exp, mu_exp, phi_exp)
            log_emission = pt.where(y_exp == 0, log_zero, log_pos)
            log_emission = pt.where(mask[:, :, None], log_emission, 0.0)

        if K == 1:
            logp_cust = pt.sum(log_emission, axis=1)
        else:
            log_alpha = pt.log(pi0) + log_emission[:, 0, :]
            log_Gamma = pt.log(Gamma)[None, :, :]
            for t in range(1, T):
                temp = log_alpha[:, :, None] + log_Gamma
                log_alpha = log_emission[:, t, :] + pt.logsumexp(temp, axis=1)
            logp_cust = pt.logsumexp(log_alpha, axis=1)

        pm.Potential('loglike', pt.sum(logp_cust))
        pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))

        # === ADD THIS BLOCK ===
        # Compute Viterbi path for state recovery evaluation
        if K > 1:
            # Viterbi decoding (most likely state sequence)
            log_delta = pt.log(pi0) + log_emission[:, 0, :]
            psi = pt.zeros((N, T, K), dtype='int32')
            
            for t in range(1, T):
                temp = log_delta[:, :, None] + log_Gamma + log_emission[:, t, :][:, None, :]
                max_val = pt.max(temp, axis=1)
                max_idx = pt.argmax(temp, axis=1)
                log_delta = max_val
                psi = pt.set_subtensor(psi[:, t, :], max_idx)
            
            # Backtrack to find Viterbi path
            viterbi_path = pt.zeros((N, T), dtype='int32')
            viterbi_path = pt.set_subtensor(viterbi_path[:, T-1], pt.argmax(log_delta, axis=1))
            
            # Store posterior state probabilities (marginal)
            # Approximate: use normalized forward probabilities
            #post_probs = pt.exp(log_alpha - pt.logsumexp(log_alpha, axis=1, keepdims=True))
            
            pm.Deterministic('viterbi', viterbi_path, dims=('customer', 'time'))
            #pm.Deterministic('post_probs', post_probs, dims=('customer', 'time', 'state'))
        # === END ADD ===
        return model


def make_poisson_model(data, K=3, use_gam=True, gam_df=3):
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]
    
    print(f"DEBUG: y shape={y.shape}, mask shape={mask.shape}, R shape={R.shape}")
    print(f"DEBUG: N={N}, T={T}")

    y_int = np.round(y).astype(np.int32)

    if use_gam and K >= 1:
        R_flat = R.flatten()
        F_flat = F.flatten()
        M_flat = M.flatten()
        basis_R = create_bspline_basis(R_flat, df=gam_df)
        basis_F = create_bspline_basis(F_flat, df=gam_df)
        basis_M = create_bspline_basis(M_flat, df=gam_df)
        n_basis_R = basis_R.shape[1]
        n_basis_F = basis_F.shape[1]
        n_basis_M = basis_M.shape[1]
        basis_R = basis_R.reshape(N, T, n_basis_R)
        basis_F = basis_F.reshape(N, T, n_basis_F)
        basis_M = basis_M.reshape(N, T, n_basis_M)
    else:
        n_basis_R = n_basis_F = n_basis_M = 1
        basis_R = basis_F = basis_M = None

    with pm.Model(coords={"customer": np.arange(N), "time": np.arange(T), "state": np.arange(K)}) as model:
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
        else:
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K, dtype=np.float32), shape=(K, K))

        if K == 1:
            beta0 = pm.Normal("beta0", 0, 1)
            if use_gam:
                w_R = pm.Normal("w_R", 0, 1, shape=n_basis_R)
                w_F = pm.Normal("w_F", 0, 1, shape=n_basis_F)
                w_M = pm.Normal("w_M", 0, 1, shape=n_basis_M)
            else:
                betaR = pm.Normal("betaR", 0, 1)
                betaF = pm.Normal("betaF", 0, 1)
                betaM = pm.Normal("betaM", 0, 1)
        else:
            beta0 = pm.Normal("beta0", 0, 1, shape=K)
            if use_gam:
                w_R = pm.Normal("w_R", 0, 1, shape=(K, n_basis_R))
                w_F = pm.Normal("w_F", 0, 1, shape=(K, n_basis_F))
                w_M = pm.Normal("w_M", 0, 1, shape=(K, n_basis_M))
            else:
                betaR = pm.Normal("betaR", 0, 1, shape=K)
                betaF = pm.Normal("betaF", 0, 1, shape=K)
                betaM = pm.Normal("betaM", 0, 1, shape=K)

        if use_gam:
            if K == 1:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [0]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [0]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [0]))
                log_lambda = beta0 + effect_R + effect_F + effect_M
            else:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [1]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [1]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [1]))
                log_lambda = beta0 + effect_R + effect_F + effect_M
        else:
            if K == 1:
                log_lambda = beta0 + betaR * R + betaF * F + betaM * M
            else:
                log_lambda = beta0 + betaR * R[..., None] + betaF * F[..., None] + betaM * M[..., None]

        lam = pt.exp(pt.clip(log_lambda, -10, 10))

        y_tensor = pt.as_tensor_variable(y_int)
        if K == 1:
            log_emission = y_tensor * pt.log(lam) - lam - pt.gammaln(y_tensor + 1)
            log_emission = pt.where(mask, log_emission, 0.0)
        else:
            lam_exp = lam[..., None]
            y_exp = y_tensor[..., None]
            log_emission = y_exp * pt.log(lam_exp) - lam_exp - pt.gammaln(y_exp + 1)
            log_emission = pt.where(mask[:, :, None], log_emission, 0.0)

        if K == 1:
            logp_cust = pt.sum(log_emission, axis=1)
        else:
            log_alpha = pt.log(pi0) + log_emission[:, 0, :]
            log_Gamma = pt.log(Gamma)[None, :, :]
            for t in range(1, T):
                temp = log_alpha[:, :, None] + log_Gamma
                log_alpha = log_emission[:, t, :] + pt.logsumexp(temp, axis=1)
            logp_cust = pt.logsumexp(log_alpha, axis=1)

        pm.Potential('loglike', pt.sum(logp_cust))
        pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))

        return model     
   
def make_nbd_model(data, K=3, use_gam=True, gam_df=3):
    """
    Continuous-Extension Negative Binomial HMM.
    Uses Gamma-function interpolation to allow fair comparison with 
    semi-continuous Tweedie on non-integer spend data.
    """
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]
    
    # Treat y as continuous float32 to match Tweedie precision
    y_tensor = pt.as_tensor_variable(y).astype("float32")
    eps = 1e-9 # Numerical stability epsilon

    if use_gam and K >= 1:
        R_flat, F_flat, M_flat = R.flatten(), F.flatten(), M.flatten()
        basis_R = create_bspline_basis(R_flat, df=gam_df)
        basis_F = create_bspline_basis(F_flat, df=gam_df)
        basis_M = create_bspline_basis(M_flat, df=gam_df)
        
        n_basis_R, n_basis_F, n_basis_M = basis_R.shape[1], basis_F.shape[1], basis_M.shape[1]
        
        basis_R = basis_R.reshape(N, T, n_basis_R)
        basis_F = basis_F.reshape(N, T, n_basis_F)
        basis_M = basis_M.reshape(N, T, n_basis_M)
    else:
        n_basis_R = n_basis_F = n_basis_M = 1
        basis_R = basis_F = basis_M = None

    with pm.Model(coords={"customer": np.arange(N), "time": np.arange(T), "state": np.arange(K)}) as model:
        # --- PRIORS: HMM Structure ---
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
        else:
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K, dtype=np.float32), shape=(K, K))

        # --- PRIORS: Emission Parameters ---
        beta0 = pm.Normal("beta0", 0, 1, shape=(K,) if K > 1 else None)
        alpha = pm.Exponential("alpha", lam=1.0, shape=(K,) if K > 1 else None)

        if use_gam:
            w_R = pm.Normal("w_R", 0, 1, shape=(K, n_basis_R) if K > 1 else n_basis_R)
            w_F = pm.Normal("w_F", 0, 1, shape=(K, n_basis_F) if K > 1 else n_basis_F)
            w_M = pm.Normal("w_M", 0, 1, shape=(K, n_basis_M) if K > 1 else n_basis_M)
        else:
            betaR = pm.Normal("betaR", 0, 1, shape=(K,) if K > 1 else None)
            betaF = pm.Normal("betaF", 0, 1, shape=(K,) if K > 1 else None)
            betaM = pm.Normal("betaM", 0, 1, shape=(K,) if K > 1 else None)

        # --- LOG-MU CALCULATION (Link Function) ---
        if use_gam:
            if K == 1:
                eR = pt.tensordot(basis_R, w_R, axes=([2], [0]))
                eF = pt.tensordot(basis_F, w_F, axes=([2], [0]))
                eM = pt.tensordot(basis_M, w_M, axes=([2], [0]))
                log_mu = beta0 + eR + eF + eM
            else:
                eR = pt.tensordot(basis_R, w_R, axes=([2], [1]))
                eF = pt.tensordot(basis_F, w_F, axes=([2], [1]))
                eM = pt.tensordot(basis_M, w_M, axes=([2], [1]))
                log_mu = beta0 + eR + eF + eM
        else:
            if K == 1:
                log_mu = beta0 + betaR * R + betaF * F + betaM * M
            else:
                log_mu = beta0 + betaR * R[..., None] + betaF * F[..., None] + betaM * M[..., None]

        mu = pt.exp(pt.clip(log_mu, -10, 10))

        # --- CONTINUOUS NBD LIKELIHOOD ---
        if K == 1:
            term1 = pt.gammaln(y_tensor + alpha)
            term2 = pt.gammaln(y_tensor + 1.0)
            term3 = pt.gammaln(alpha)
            # Log-space stability for p and q
            log_p = pt.log(alpha) - pt.log(mu + alpha + eps)
            log_q = pt.log(mu + eps) - pt.log(mu + alpha + eps)
            log_emission = term1 - term2 - term3 + (alpha * log_p) + (y_tensor * log_q)
            log_emission = pt.where(mask, log_emission, 0.0)
        else:
            alpha_exp = alpha[None, None, :]
            mu_exp = mu[..., None] if use_gam else mu # mu already (N,T,K) if not GAM? No, check dims
            y_exp = y_tensor[..., None]
            
            term1 = pt.gammaln(y_exp + alpha_exp)
            term2 = pt.gammaln(y_exp + 1.0)
            term3 = pt.gammaln(alpha_exp)
            log_p = pt.log(alpha_exp) - pt.log(mu_exp + alpha_exp + eps)
            log_q = pt.log(mu_exp + eps) - pt.log(mu_exp + alpha_exp + eps)
            
            log_emission = term1 - term2 - term3 + (alpha_exp * log_p) + (y_exp * log_q)
            log_emission = pt.where(mask[:, :, None], log_emission, 0.0)

        # --- FORWARD PASS (HMM INTEGRATION) ---
        if K == 1:
            logp_cust = pt.sum(log_emission, axis=1)
        else:
            log_alpha = pt.log(pi0) + log_emission[:, 0, :]
            log_Gamma = pt.log(Gamma)[None, :, :]
            for t in range(1, T):
                temp = log_alpha[:, :, None] + log_Gamma
                log_alpha = log_emission[:, t, :] + pt.logsumexp(temp, axis=1)
            logp_cust = pt.logsumexp(log_alpha, axis=1)

        pm.Potential('loglike', pt.sum(logp_cust))
        pm.Deterministic('log_likelihood', logp_cust, dims=('customer',))

    return model

# =============================================================================
# 4. DATA BUILDER
# =============================================================================

def compute_rfm_features(y, mask):
    """Compute Recency, Frequency, Monetary features from panel data."""
    N, T = y.shape
    R = np.zeros((N, T), dtype=np.float32)
    F = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)
    
    for i in range(N):
        last_purchase = -1
        cumulative_freq = 0
        cumulative_spend = 0.0
        
        for t in range(T):
            if mask[i, t]:
                if y[i, t] > 0:
                    last_purchase = t
                    cumulative_freq += 1
                    cumulative_spend += y[i, t]
                
                if last_purchase >= 0:
                    R[i, t] = t - last_purchase
                    F[i, t] = cumulative_freq
                    M[i, t] = cumulative_spend / cumulative_freq if cumulative_freq > 0 else 0.0
                else:
                    R[i, t] = t + 1  # No purchase yet
                    F[i, t] = 0
                    M[i, t] = 0.0
            else:
                R[i, t] = 0
                F[i, t] = 0
                M[i, t] = 0.0
    
    return R, F, M

## ----
	
def load_simulation_data(data_path, n_cust=None, seed=42, train_frac=1.0):
    """Load simulation from pkl or CSV file with optional train/test split."""
    import pickle
    import pandas as pd
    
    data_path = pathlib.Path(data_path)
    
    if data_path.suffix == '.pkl':
        with open(data_path, 'rb') as f:
            sim = pickle.load(f)
        N_full, T = sim['N'], sim['T']
        obs = sim['observations']
        source = 'pkl'
        
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        N_full = df['customer_id'].nunique()
        T = df['time_period'].nunique()
        obs = df.pivot(index='customer_id', columns='time_period', values='y').values
        source = 'csv'
    else:
        raise ValueError(f"Unknown format: {data_path.suffix}")
    
    # Subsample customers if requested
    if n_cust is not None and n_cust < N_full:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_full, n_cust, replace=False)
        obs = obs[idx, :]
        N = n_cust
    else:
        N = N_full
        idx = np.arange(N)
    
    # Train/test split on time dimension
    T_train = int(T * train_frac)
    
    if train_frac < 1.0:
        obs_train = obs[:, :T_train]
        obs_test = obs[:, T_train:]
    else:
        obs_train = obs
        obs_test = None
    
    # Build masks
    mask_train = np.ones((N, T_train), dtype=bool)
    
    # Compute RFM on training data only
    R_train, F_train, M_train = compute_rfm_features(obs_train, mask_train)
    
    data = {
        'N': N,
        'T': T_train,
        'y': obs_train.astype(np.float32),
        'mask': mask_train,
        'R': R_train.astype(np.float32),
        'F': F_train.astype(np.float32),
        'M': M_train.astype(np.float32),
        'customer_id': idx,
        'time': np.arange(T_train),
        'T_full': T,  # Original T
        'T_test': T - T_train if train_frac < 1.0 else 0,
    }
    
    if obs_test is not None:
        data['y_test'] = obs_test.astype(np.float32)
        data['mask_test'] = np.ones((N, T - T_train), dtype=bool)
    
    print(f"  Loaded: N={N}, T_train={T_train}, T_test={data.get('T_test', 0)}, "
          f"zeros={np.mean(obs_train==0):.1%} (from {source})")
    
    return data

## ----

def build_panel_data(df_path, customer_col='customer_id', n_cust=None, seed=RANDOM_SEED):
    """Build panel data dictionary from CSV."""
    np.random.seed(seed)
    df = pd.read_csv(df_path, parse_dates=['WeekStart'])
    df = df.astype({customer_col: str})

    if n_cust is not None:
        unique_custs = df[customer_col].unique()
        if len(unique_custs) > n_cust:
            selected = np.random.choice(unique_custs, n_cust, replace=False)
            df = df[df[customer_col].isin(selected)]

    panel_sizes = df.groupby(customer_col).size()
    
    def mat(col):
        arr = df.pivot(index=customer_col, columns='WeekStart', values=col).values
        return arr.astype(np.float32)

    return {
        'N': df[customer_col].nunique(),
        'T': panel_sizes.iloc[0],
        'y': mat('WeeklySpend'),
        'mask': ~np.isnan(mat('WeeklySpend')),
        'R': mat('R_weeks'),
        'F': mat('F_run'),
        'M': mat('M_run'),
        'p0': mat('p0_cust')
    }

# =============================================================================
# 5. SMC RUNNER
# =============================================================================

def run_smc(data, K, model_type, state_specific_p, p_fixed, use_gam, gam_df,
            draws, chains, seed, out_dir):
    """Run SMC with model_type selection."""
    
    if model_type == 'hurdle':
        model_builder = make_hurdle_model
    elif model_type == 'poisson':
        model_builder = make_poisson_model
    elif model_type == 'nbd':
        model_builder = make_nbd_model
    else:
        model_builder = make_model
    
    cores = min(chains, os.cpu_count() or 1)
    t0 = time.time()

    try:
        if model_type in ['poisson', 'nbd', 'hurdle']:
            with model_builder(data, K=K, use_gam=use_gam, gam_df=gam_df) as model:
                print(f" Model: K={K}, {model_type.upper()}-{'GAM' if use_gam else 'GLM'}")
                idata = pm.sample_smc(draws=draws, chains=chains, cores=cores,
                                     random_seed=seed, return_inferencedata=True, threshold=0.8)
        else:
            with model_builder(data, K=K, state_specific_p=state_specific_p, 
                              p_fixed=p_fixed, use_gam=use_gam, gam_df=gam_df) as model:
                print(f" Model: K={K}, {model_type.upper()}-{'GAM' if use_gam else 'GLM'}")
                idata = pm.sample_smc(draws=draws, chains=chains, cores=cores,
                                     random_seed=seed, return_inferencedata=True, threshold=0.8)

            # === ADD THIS BLOCK ===
            # Add posterior predictive for state recovery
            if K > 1:
                try:
                    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=seed)
                except:
                    pass
            # === END ADD ===        
        
        log_ev = np.nan
        try:
            lm = idata.sample_stats.log_marginal_likelihood.values
            if isinstance(lm, np.ndarray) and lm.dtype == object:
                chain_vals = []
                for c in range(lm.shape[1] if lm.ndim > 1 else 1):
                    if lm.ndim > 1:
                        chain_list = lm[-1, c] if lm.shape[0] > 0 else lm[0, c]
                    else:
                        chain_list = lm[c] if lm.ndim == 1 else lm[0]
                    if isinstance(chain_list, list):
                        valid = [float(x) for x in chain_list if isinstance(x, (int, float, np.floating)) and np.isfinite(x)]
                        if valid:
                            chain_vals.append(valid[-1])
                    elif isinstance(chain_list, (int, float, np.floating)) and np.isfinite(chain_list):
                        chain_vals.append(float(chain_list))
                log_ev = float(np.mean(chain_vals)) if chain_vals else np.nan
            else:
                flat = np.array(lm).flatten()
                valid = flat[np.isfinite(flat)]
                log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
        except:
            pass

        elapsed = (time.time() - t0) / 60


        res = {
            'K': K, 'model_type': model_type, 'N': data['N'], 'T': data['T'],
            'use_gam': use_gam, 'gam_df': gam_df if use_gam else None,
            'log_evidence': log_ev, 'draws': draws, 'chains': chains,
            'time_min': elapsed, 'timestamp': time.strftime('%Y%m%d_%H%M%S')
        }

        # Add test data if available
        if 'y_test' in data:
            res['y_test'] = data['y_test']
            res['T_test'] = data['T_test']
            res['mask_test'] = data.get('mask_test', None)

        p_tag = f"p{p_fixed}" if (model_type == 'tweedie' and not state_specific_p and K > 1) else "statep" if (state_specific_p and K > 1) else "discrete"
        pkl_path = out_dir / f"smc_K{K}_{model_type.upper()}_{'GAM' if use_gam else 'GLM'}_{p_tag}_N{data['N']}_D{draws}.pkl"

        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'res': res}, f, protocol=4)

        print(f" ✓ log_ev={log_ev:.2f}, time={elapsed:.1f}min")
        return pkl_path, res

    except Exception as e:
        print(f" ✗ CRASH: {str(e)[:60]}")
        raise
        
        elapsed = (time.time() - t0) / 60

        res = {
            'K': K, 'model_type': model_type, 'N': data['N'], 'T': data['T'],
            'use_gam': use_gam, 'gam_df': gam_df if use_gam else None,
            'log_evidence': log_ev, 'draws': draws, 'chains': chains,
            'time_min': elapsed, 'timestamp': time.strftime('%Y%m%d_%H%M%S')
        }

        p_tag = f"p{p_fixed}" if (model_type == 'tweedie' and not state_specific_p and K > 1) else "statep" if (state_specific_p and K > 1) else "discrete"
        pkl_path = out_dir / f"smc_K{K}_{model_type.upper()}_{'GAM' if use_gam else 'GLM'}_{p_tag}_N{data['N']}_D{draws}.pkl"

        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'res': res}, f, protocol=4)

        print(f" ✓ log_ev={log_ev:.2f}, time={elapsed:.1f}min")
        return pkl_path, res

    except Exception as e:
        print(f" ✗ CRASH: {str(e)[:60]}")
        raise

# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='HMM/Static Models: Tweedie, Hurdle, Poisson, NBD')
    parser.add_argument('--dataset', required=True, choices=['uci', 'cdnow', 'simulation'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--sim_path', type=str, default=None,
                       help='Path to simulation file (.csv or .pkl). Required if --dataset simulation')
    parser.add_argument('--n_cust', type=int, default=None)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--model_type', default='tweedie', choices=['tweedie', 'hurdle', 'poisson', 'nbd'])
    parser.add_argument('--state_specific_p', action='store_true')
    parser.add_argument('--p_fixed', type=float, default=1.5)
    parser.add_argument('--no_gam', action='store_true')
    parser.add_argument('--gam_df', type=int, default=3)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--train_frac', type=float, default=1.0,
                   help='Fraction of time periods for training (default: 1.0 = no split)')
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SMC Unified: {args.dataset.upper()} | {args.model_type.upper()} | K={args.K}")
    
    if args.dataset == 'simulation':
        if not args.sim_path:
            raise ValueError("--sim_path required when --dataset simulation (e.g., --sim_path /path/to/file.csv or .pkl)")
        
        sim_path = pathlib.Path(args.sim_path)
        if not sim_path.exists():
            raise FileNotFoundError(f"Simulation file not found: {sim_path}")

        data = load_simulation_data(sim_path, n_cust=args.n_cust, seed=args.seed, train_frac=args.train_frac)
    else:
        data = build_panel_data(data_dir / f"{args.dataset}_full.csv", n_cust=args.n_cust, seed=args.seed)
    
    print(f"{'='*70}\n")
    print(f"Loaded: N={data['N']}, T={data['T']}, zeros={np.mean(data['y']==0):.1%}\n")

    pkl_path, res = run_smc(data, args.K, args.model_type, args.state_specific_p,
                           args.p_fixed, not args.no_gam, args.gam_df,
                           args.draws, args.chains, args.seed, out_dir)

    print(f"\nSaved: {pkl_path}")
    print(f"Log-Ev: {res['log_evidence']:.2f}, Time: {res['time_min']:.1f}min")
    
  
if __name__ == "__main__":
    main()
