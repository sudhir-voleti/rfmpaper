## Pretty transition matrix (reuseable helper), post smc run, from pkl

import seaborn as sns, matplotlib.pyplot as plt

def plot_transmat(idata, labels=None):
    Gamma = idata.posterior['Gamma'].mean(('chain','draw')).values
    K = Gamma.shape[0]
    if labels is None:
        labels = [f'State {k}' for k in range(K)]
    Gamma_df = pd.DataFrame(Gamma, index=labels, columns=labels)

    plt.figure(figsize=(4,3))
    sns.heatmap(Gamma_df, annot=True, fmt='.3f', cmap='Blues',
                cbar_kws={'label': 'P(t+1 | t)'})
    plt.title('Posterior-mean transition matrix')
    plt.ylabel('From'); plt.xlabel('To')
    plt.tight_layout()
    return plt.gca()

### One-liner Viterbi for all customers

def viterbi_paths_hmmlearn(idata, y, mask):
    """
    idata : InferenceData (posterior samples)
    y, mask : (N, T) float/int  (mask is True=observed)
    returns z_mat : (N, T) int  most-likely state sequence
    """
    # posterior mean parameters
    pi0   = idata.posterior['pi0'].mean(('chain','draw')).values          # (K,)
    Gamma = idata.posterior['Gamma'].mean(('chain','draw')).values      # (K,K)
    mu    = idata.posterior['mu'].mean(('chain','draw')).values         # (N,T,K)  – already clipped
    phi   = idata.posterior['phi'].mean(('chain','draw')).values        # (K,) or (N,T,K)

    N, T, K = mu.shape
    z_mat = np.empty((N, T), dtype=int)

    for i in range(N):
        # emission probs = P(y|state) under ZIG approximation
        psi = np.exp(-mu[i] / phi[i])                # (T,K)
        log_zero = np.log(psi + 1e-12)
        log_pos  = np.log1p(-psi + 1e-12) + \
                   (mu[i]/phi[i] - 1)*np.log(y[i, :, None]) - \
                   y[i, :, None]/phi[i] + \
                   (mu[i]/phi[i])*np.log(1/phi[i]) - \
                   gammaln(mu[i]/phi[i])
        log_emission = np.where(y[i, :, None] == 0, log_zero, log_pos)   # (T,K)

        # normalise to probabilities (row-stochastic)
        emission_prob = np.exp(log_emission - log_emission.max(axis=1, keepdims=True))
        emission_prob /= emission_prob.sum(axis=1, keepdims=True)

        # build hmmlearn model
        model = CategoricalHMM(n_components=K,
                               init_params='',   # we provide everything
                               params='')
        model.startprob_ = pi0
        model.transmat_  = Gamma
        model.emissionprob_ = emission_prob          # (T,K) treated as categorical probs

        # Viterbi decode
        z_mat[i] = model.predict(emission_prob)      # returns (T,) int

    return z_mat
  
### State-share plot (area chart)

def plot_state_share(z_mat, labels=None):
    N, T = z_mat.shape
    K = int(z_mat.max()) + 1
    if labels is None:
        labels = [f'State {k}' for k in range(K)]

    # proportion in each state at every week
    prop = pd.DataFrame({t: pd.Series(z_mat[:, t]).value_counts(normalize=True)
                         for t in range(T)}).T.fillna(0)
    prop = prop.reindex(columns=range(K), fill_value=0)

    prop.plot.area(stacked=True, figsize=(8,3), cmap='coolwarm',
                   title='Share of customers in each latent state')
    plt.ylabel('Proportion'); plt.xlabel('Week')
    plt.legend(labels, title='State', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    return plt.gcf()
  
