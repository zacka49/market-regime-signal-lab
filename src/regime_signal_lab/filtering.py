"""Hamilton (1989) filter for a discrete-time, finite-state Markov-switching
Gaussian model.

Model: s_t is a hidden Markov chain with transition matrix P (P[i, j] =
P(s_t = j | s_{t-1} = i)); conditional on s_t = i, r_t ~ Normal(mu_i, sigma_i).
This is exactly the regime model in simulate.py. The filter recovers
P(s_t = i | r_1, ..., r_t) recursively -- a predict/update (forward algorithm)
pair -- without ever seeing the true regime path. See
docs/mathematical_notes.md for the derivation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Solve pi = pi P, sum(pi) = 1 for the stationary distribution of a
    discrete-time Markov chain, used to initialize the filter."""
    n = transition.shape[0]
    a = np.vstack([transition.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    return pi


def hamilton_filter(
    returns: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run the Hamilton forward filter.

    Returns (filtered_probs, log_likelihood) where filtered_probs has shape
    (len(returns), n_states) and filtered_probs[t, i] = P(s_t = i | r_1..r_t).
    """
    n_states = transition.shape[0]
    n_obs = len(returns)
    filtered = np.empty((n_obs, n_states))

    xi_prev = stationary_distribution(transition)
    log_likelihood = 0.0

    for t in range(n_obs):
        # Predict: xi_{t|t-1} = P' xi_{t-1|t-1}
        xi_pred = transition.T @ xi_prev

        # Emission likelihoods f(r_t | s_t = i)
        eta = norm.pdf(returns[t], loc=means, scale=stds)

        joint = xi_pred * eta
        marginal = joint.sum()
        marginal = max(marginal, 1e-300)  # guard against float underflow
        xi_curr = joint / marginal

        filtered[t] = xi_curr
        log_likelihood += np.log(marginal)
        xi_prev = xi_curr

    return filtered, float(log_likelihood)
