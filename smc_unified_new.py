#!/usr/bin/env python3
"""
smc_unified_new.py
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

    with pm.Model(coords={"customer": np.arange(N)}) as model:
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
            phi = pm.Deterministic("phi", pt.sort(phi_raw))

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

        if K == 1:
            p_exp = p
            phi_exp = phi
            exponent = 2.0 - p_exp
            psi = pt.exp(-pt.pow(mu, exponent) / (phi_exp * exponent))
            psi = pt.clip(psi, 1e-12, 1 - 1e-12)
            log_zero = pt.log(psi)
            log_pos = pt.log1p(-psi) + gamma_logp_det(y, mu, phi_exp)
            log_emission = pt.where(y == 0, log_zero, log_pos)
            log_emission = pt.where(mask, log_emission, 0.0)
        else:
            p_exp = p[None, None, :]
            phi_exp = phi[None, None, :]
            exponent = 2.0 - p_exp
            psi = pt.exp(-pt.pow(mu, exponent) / (phi_exp * exponent))
            psi = pt.clip(psi, 1e-12, 1 - 1e-12)
            log_zero = pt.log(psi)
            y_exp = y[..., None]
            log_pos = pt.log1p(-psi) + gamma_logp_det(y_exp, mu, phi_exp)
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
            post_probs = pt.exp(log_alpha - pt.logsumexp(log_alpha, axis=1, keepdims=True))
            
            pm.Deterministic('viterbi', viterbi_path, dims=('customer', 'time'))
            pm.Deterministic('post_probs', post_probs, dims=('customer', 'time', 'state'))
        # === END ADD ===
        return model

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

    with pm.Model(coords={"customer": np.arange(N)}) as model:
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
            post_probs = pt.exp(log_alpha - pt.logsumexp(log_alpha, axis=1, keepdims=True))
            
            pm.Deterministic('viterbi', viterbi_path, dims=('customer', 'time'))
            pm.Deterministic('post_probs', post_probs, dims=('customer', 'time', 'state'))
        # === END ADD ===
        return model



def make_poisson_model(data, K=3, use_gam=True, gam_df=3):
    """DISCRETE Poisson HMM - rounds y to integers."""
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]
    
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

    with pm.Model(coords={"customer": np.arange(N)}) as model:
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
            post_probs = pt.exp(log_alpha - pt.logsumexp(log_alpha, axis=1, keepdims=True))
            
            pm.Deterministic('viterbi', viterbi_path, dims=('customer', 'time'))
            pm.Deterministic('post_probs', post_probs, dims=('customer', 'time', 'state'))
        # === END ADD ===

        return model        

def make_nbd_model(data, K=3, use_gam=True, gam_df=3):
    """DISCRETE Negative Binomial HMM - rounds y to integers."""
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]
    
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

    with pm.Model(coords={"customer": np.arange(N)}) as model:
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
        else:
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K, dtype=np.float32), shape=(K, K))

        if K == 1:
            beta0 = pm.Normal("beta0", 0, 1)
            alpha = pm.Exponential("alpha", lam=1.0)
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
            alpha = pm.Exponential("alpha", lam=1.0, shape=K)
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
                log_mu = beta0 + effect_R + effect_F + effect_M
            else:
                effect_R = pt.tensordot(basis_R, w_R, axes=([2], [1]))
                effect_F = pt.tensordot(basis_F, w_F, axes=([2], [1]))
                effect_M = pt.tensordot(basis_M, w_M, axes=([2], [1]))
                log_mu = beta0 + effect_R + effect_F + effect_M
        else:
            if K == 1:
                log_mu = beta0 + betaR * R + betaF * F + betaM * M
            else:
                log_mu = beta0 + betaR * R[..., None] + betaF * F[..., None] + betaM * M[..., None]

        mu = pt.exp(pt.clip(log_mu, -10, 10))

        y_tensor = pt.as_tensor_variable(y_int)
        if K == 1:
            term1 = pt.gammaln(y_tensor + alpha)
            term2 = pt.gammaln(y_tensor + 1)
            term3 = pt.gammaln(alpha)
            term4 = alpha * pt.log(alpha / (mu + alpha))
            term5 = y_tensor * pt.log(mu / (mu + alpha))
            log_emission = term1 - term2 - term3 + term4 + term5
            log_emission = pt.where(mask, log_emission, 0.0)
        else:
            alpha_exp = alpha[None, None, :]
            mu_exp = mu[..., None]
            y_exp = y_tensor[..., None]
            term1 = pt.gammaln(y_exp + alpha_exp)
            term2 = pt.gammaln(y_exp + 1)
            term3 = pt.gammaln(alpha_exp)
            term4 = alpha_exp * pt.log(alpha_exp / (mu_exp + alpha_exp))
            term5 = y_exp * pt.log(mu_exp / (mu_exp + alpha_exp))
            log_emission = term1 - term2 - term3 + term4 + term5
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
            post_probs = pt.exp(log_alpha - pt.logsumexp(log_alpha, axis=1, keepdims=True))
            
            pm.Deterministic('viterbi', viterbi_path, dims=('customer', 'time'))
            pm.Deterministic('post_probs', post_probs, dims=('customer', 'time', 'state'))
        # === END ADD ===

        return model        

# =============================================================================
# 4. DATA BUILDER
# =============================================================================

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

    args = parser.parse_args()
    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SMC Unified: {args.dataset.upper()} | {args.model_type.upper()} | K={args.K}")
    print(f"{'='*70}\n")

    data = build_panel_data(data_dir / f"{args.dataset}_full.csv", n_cust=args.n_cust, seed=args.seed)
    print(f"Loaded: N={data['N']}, T={data['T']}, zeros={np.mean(data['y']==0):.1%}\n")

    pkl_path, res = run_smc(data, args.K, args.model_type, args.state_specific_p,
                           args.p_fixed, not args.no_gam, args.gam_df,
                           args.draws, args.chains, args.seed, out_dir)

    print(f"\nSaved: {pkl_path}")
    print(f"Log-Ev: {res['log_evidence']:.2f}, Time: {res['time_min']:.1f}min")

if __name__ == "__main__":
    main()
