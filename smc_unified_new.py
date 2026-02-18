COMPLETE DROP-IN MODIFICATIONS FOR smc_unified.py
======================================================================

# =============================================================================
# ADD THESE FUNCTIONS TO smc_unified.py (after make_model)
# =============================================================================

def make_hurdle_model(data, K=3, use_gam=True, gam_df=3):
    """Hurdle-Gamma HMM model (add full code from above)"""
    # [Paste the full make_hurdle_model code here]
    pass

def make_poisson_model(data, K=3, use_gam=True, gam_df=3):
    """Poisson HMM for discrete comparison"""
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]

    # Precompute GAM bases (same as before)
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
        # Latent dynamics
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
        else:
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K, dtype=np.float32), shape=(K, K))

        # Poisson params
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

        # Compute lambda
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

        # Poisson log-likelihood (continuous extension for fair comparison)
        if K == 1:
            log_emission = y * pt.log(lam) - lam - pt.gammaln(y + 1)
            log_emission = pt.where(mask, log_emission, 0.0)
        else:
            lam_exp = lam[..., None]
            y_exp = y[..., None]
            log_emission = y_exp * pt.log(lam_exp) - lam_exp - pt.gammaln(y_exp + 1)
            log_emission = pt.where(mask[:, :, None], log_emission, 0.0)

        # Forward algorithm
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
    """Negative Binomial HMM (continuous extension)"""
    N, T = data["N"], data["T"]
    y, mask = data["y"], data["mask"]
    R, F, M = data["R"], data["F"], data["M"]

    # Precompute GAM bases
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
        # Latent dynamics
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
        else:
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K, dtype=np.float32), shape=(K, K))

        # NBD params
        if K == 1:
            beta0 = pm.Normal("beta0", 0, 1)
            theta = pm.Exponential("theta", lam=1.0)  # dispersion
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
            theta = pm.Exponential("theta", lam=1.0, shape=K)
            if use_gam:
                w_R = pm.Normal("w_R", 0, 1, shape=(K, n_basis_R))
                w_F = pm.Normal("w_F", 0, 1, shape=(K, n_basis_F))
                w_M = pm.Normal("w_M", 0, 1, shape=(K, n_basis_M))
            else:
                betaR = pm.Normal("betaR", 0, 1, shape=K)
                betaF = pm.Normal("betaF", 0, 1, shape=K)
                betaM = pm.Normal("betaM", 0, 1, shape=K)

        # Compute mu
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

        # NBD log-likelihood (continuous extension via Gamma functions)
        if K == 1:
            alpha = theta
            term1 = pt.gammaln(y + alpha)
            term2 = pt.gammaln(y + 1)
            term3 = pt.gammaln(alpha)
            term4 = alpha * pt.log(alpha / (mu + alpha))
            term5 = y * pt.log(mu / (mu + alpha))
            log_emission = term1 - term2 - term3 + term4 + term5
            log_emission = pt.where(mask, log_emission, 0.0)
        else:
            alpha = theta[None, None, :]
            mu_exp = mu[..., None]
            y_exp = y[..., None]
            term1 = pt.gammaln(y_exp + alpha)
            term2 = pt.gammaln(y_exp + 1)
            term3 = pt.gammaln(alpha)
            term4 = alpha * pt.log(alpha / (mu_exp + alpha))
            term5 = y_exp * pt.log(mu_exp / (mu_exp + alpha))
            log_emission = term1 - term2 - term3 + term4 + term5
            log_emission = pt.where(mask[:, :, None], log_emission, 0.0)

        # Forward algorithm
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
# MODIFIED run_smc function signature
# =============================================================================

def run_smc(data, K, model_type, state_specific_p, p_fixed, use_gam, gam_df,
<response clipped><NOTE>Result is longer than **10000 characters**, will be **truncated**.</NOTE>
