"""Multi-asset factor simulation, covariance estimation (PCA, Marchenko-
Pastur denoising, Ledoit-Wolf shrinkage) and closed-form portfolio
construction. See docs/mathematical_notes.md Section 6.

Simulating returns from a KNOWN factor model (rather than only using real
data) means every covariance estimator below can be scored against the true
covariance it's trying to recover -- the same "ground truth to validate
against" idea used for regime detection in Section 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_factor_returns(
    n_assets: int,
    n_days: int,
    factor_vols: tuple[float, ...] = (0.015, 0.008),
    idio_vol: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate returns from r_t = B f_t + eps_t: len(factor_vols) independent
    factors (first one deliberately given the largest variance to play the
    role of a "market" factor) with asset loadings B drawn once and fixed,
    plus uncorrelated idiosyncratic noise. Returns (returns, true_beta,
    true_cov) where true_cov = B diag(factor_vols^2) B' + diag(idio_vol^2)
    is the exact covariance the estimators below are trying to recover.
    """
    rng = np.random.default_rng(seed)
    n_factors = len(factor_vols)

    beta = rng.uniform(0.3, 1.2, size=(n_assets, n_factors))
    beta[:, 0] = rng.uniform(0.6, 1.4, size=n_assets)  # every asset loads on the "market" factor

    factor_cov = np.diag(np.array(factor_vols) ** 2)
    factors = rng.multivariate_normal(np.zeros(n_factors), factor_cov, size=n_days)
    idio = rng.normal(0.0, idio_vol, size=(n_days, n_assets))

    returns = factors @ beta.T + idio
    true_cov = beta @ factor_cov @ beta.T + np.eye(n_assets) * idio_vol**2
    return returns, beta, true_cov


def pca_eigendecomposition(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and corresponding eigenvectors (as columns)
    of a covariance matrix -- covariance matrices are symmetric PSD, so
    eigh() (which assumes symmetry) is both faster and numerically more
    stable than a general eigensolver, and eigenvalues are guaranteed real
    and non-negative."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def explained_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    return eigenvalues / eigenvalues.sum()


def marchenko_pastur_bounds(n_assets: int, n_obs: int, variance: float = 1.0) -> tuple[float, float]:
    """Support [lambda_minus, lambda_plus] of the Marchenko-Pastur
    distribution: the eigenvalue spectrum a (n_obs x n_assets) matrix of iid
    noise (variance `variance` per entry) would produce purely from finite-
    sample randomness, with aspect ratio q = n_assets / n_obs. Any empirical
    eigenvalue inside this band is statistically indistinguishable from
    noise; only eigenvalues above lambda_plus indicate genuine structure."""
    q = n_assets / n_obs
    lambda_plus = variance * (1 + np.sqrt(q)) ** 2
    lambda_minus = variance * (1 - np.sqrt(q)) ** 2
    return lambda_minus, lambda_plus


def marchenko_pastur_pdf(x: np.ndarray, n_assets: int, n_obs: int, variance: float = 1.0) -> np.ndarray:
    q = n_assets / n_obs
    lambda_minus, lambda_plus = marchenko_pastur_bounds(n_assets, n_obs, variance)
    pdf = np.zeros_like(x)
    mask = (x >= lambda_minus) & (x <= lambda_plus) & (x > 0)
    pdf[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / (
        2 * np.pi * variance * q * x[mask]
    )
    return pdf


def denoise_covariance_mp(cov: np.ndarray, n_obs: int) -> np.ndarray:
    """Random matrix theory denoising (Laloux et al. 1999; Bouchaud &
    Potters): eigenvalues of the CORRELATION matrix that fall at or below
    the Marchenko-Pastur upper edge are statistically consistent with pure
    noise, so they are replaced by their average (preserving the trace,
    i.e. total variance, of the noise subspace) while eigenvalues above the
    edge (genuine factor structure) are left untouched. Operates on the
    correlation matrix (unit diagonal) because the MP law above assumes
    unit-variance entries; the result is rescaled back to covariance units.
    """
    n_assets = cov.shape[0]
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    eigenvalues, eigenvectors = pca_eigendecomposition(corr)
    _, lambda_plus = marchenko_pastur_bounds(n_assets, n_obs, variance=1.0)

    noise_mask = eigenvalues <= lambda_plus
    denoised_eigenvalues = eigenvalues.copy()
    if noise_mask.sum() > 0:
        denoised_eigenvalues[noise_mask] = eigenvalues[noise_mask].mean()

    denoised_corr = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T
    d = np.sqrt(np.diag(denoised_corr))
    denoised_corr = denoised_corr / np.outer(d, d)  # renormalize to a valid unit-diagonal correlation matrix
    return denoised_corr * np.outer(std, std)


def ledoit_wolf_shrinkage(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink the sample covariance toward a scaled-identity target
    (Ledoit & Wolf, 2004): S_shrunk = delta*mu*I + (1-delta)*S, where
    mu = trace(S)/n is the average variance and delta in [0, 1] is chosen to
    minimize expected Frobenius-norm loss against the (unknown) true
    covariance -- shrinking harder when the sample covariance is noisier
    (large phi_hat) relative to how much the target actually misspecifies
    the true structure (gamma_hat). Returns (shrunk_cov, delta)."""
    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    sample_cov = (x.T @ x) / t
    mu = np.trace(sample_cov) / n
    target = mu * np.eye(n)

    phi_hat = 0.0
    for row in x:
        outer = np.outer(row, row)
        phi_hat += np.sum((outer - sample_cov) ** 2)
    phi_hat /= t

    gamma_hat = np.sum((sample_cov - target) ** 2)
    kappa_hat = phi_hat / gamma_hat if gamma_hat > 0 else 0.0
    delta = float(np.clip(kappa_hat / t, 0.0, 1.0))

    shrunk = delta * target + (1 - delta) * sample_cov
    return shrunk, delta


def minimum_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Closed-form global minimum-variance weights.

    minimize w'Sigma w  subject to  1'w = 1
    Lagrangian: L = w'Sigma w - lambda*(1'w - 1)
    FOC (dL/dw = 0): 2*Sigma*w = lambda*1  =>  w proportional to Sigma^-1 1,
    normalized so the weights sum to 1.
    """
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(cov.shape[0])
    raw = inv_cov @ ones
    return raw / raw.sum()


def mean_variance_weights(
    cov: np.ndarray, expected_returns: np.ndarray, risk_aversion: float = 1.0
) -> np.ndarray:
    """Closed-form (unconstrained-leverage) mean-variance weights.

    maximize  w'mu - (risk_aversion/2) * w'Sigma w
    FOC: mu = risk_aversion * Sigma * w  =>  w = (1/risk_aversion) * Sigma^-1 mu.

    This is the direction Markowitz optimization points in; it is NOT
    normalized to any budget constraint, and (see
    scripts/run_portfolio_research.py) is extremely sensitive to estimation
    error in both Sigma^-1 and mu -- the classical "error maximization"
    problem (Michaud, 1989): the optimizer places the largest bets exactly
    where the input estimates are least reliable.
    """
    inv_cov = np.linalg.inv(cov)
    return (inv_cov @ expected_returns) / risk_aversion


def quantile_long_short_weights(signal_cross_section: np.ndarray, n_quantiles: int = 5) -> np.ndarray:
    """Dollar-neutral cross-sectional weights: equal-weight long the top
    quantile bucket of `signal_cross_section`, equal-weight short the
    bottom bucket, scaled so each leg sums to +-0.5 (net zero, gross 1)."""
    n = len(signal_cross_section)
    ranks = pd.Series(signal_cross_section).rank(method="first")
    bucket = pd.qcut(ranks, n_quantiles, labels=False)
    weights = np.zeros(n)

    top = (bucket == n_quantiles - 1).to_numpy()
    bottom = (bucket == 0).to_numpy()
    if top.sum() > 0:
        weights[top] = 0.5 / top.sum()
    if bottom.sum() > 0:
        weights[bottom] = -0.5 / bottom.sum()
    return weights
