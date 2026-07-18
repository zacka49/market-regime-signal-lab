"""Covariance estimation and portfolio construction on a simulated
multi-asset factor panel: PCA / scree, Marchenko-Pastur denoising vs
Ledoit-Wolf shrinkage vs the raw sample covariance (scored against the
TRUE covariance, which we know because we simulated it), and a direct
demonstration of naive mean-variance "error maximization" (Michaud, 1989).
"""

from __future__ import annotations

import numpy as np

from regime_signal_lab.portfolio import (
    denoise_covariance_mp,
    explained_variance_ratio,
    ledoit_wolf_shrinkage,
    marchenko_pastur_bounds,
    mean_variance_weights,
    minimum_variance_weights,
    pca_eigendecomposition,
    quantile_long_short_weights,
    simulate_factor_returns,
)

N_ASSETS = 40
N_DAYS = 150  # deliberately modest relative to N_ASSETS: q = N/T = 0.267


def realized_variance(weights: np.ndarray, true_cov: np.ndarray) -> float:
    return float(weights @ true_cov @ weights)


def main() -> None:
    returns, beta, true_cov = simulate_factor_returns(N_ASSETS, N_DAYS, seed=7)
    sample_cov = np.cov(returns, rowvar=False)

    print("=" * 70)
    print(f"1. PCA scree ({N_ASSETS} assets, {N_DAYS} days, q = {N_ASSETS / N_DAYS:.3f})")
    print("=" * 70)
    eigenvalues, eigenvectors = pca_eigendecomposition(sample_cov)
    ratios = explained_variance_ratio(eigenvalues)
    lower, upper = marchenko_pastur_bounds(N_ASSETS, N_DAYS, variance=1.0)
    print(f"MP noise band on the CORRELATION matrix: [{lower:.3f}, {upper:.3f}]")
    for i in range(8):
        flag = "genuine structure" if eigenvalues[i] > upper * np.diag(sample_cov).mean() else "~noise band"
        print(f"  eigenvalue[{i}] = {eigenvalues[i]:.6f}  ({ratios[i]:.1%} of variance)  {flag}")

    pc1 = eigenvectors[:, 0]
    market_beta = beta[:, 0]
    cos_sim = abs(np.dot(pc1, market_beta) / (np.linalg.norm(pc1) * np.linalg.norm(market_beta)))
    print(f"cos(PC1, true market-factor loadings) = {cos_sim:.4f}")

    print()
    print("=" * 70)
    print("2. Minimum-variance portfolio: raw vs denoised vs shrunk covariance")
    print("=" * 70)
    denoised_cov = denoise_covariance_mp(sample_cov, N_DAYS)
    shrunk_cov, delta = ledoit_wolf_shrinkage(returns)
    print(f"Ledoit-Wolf shrinkage intensity: {delta:.3f}")

    covariances = [
        ("true (oracle)", true_cov),
        ("raw sample", sample_cov),
        ("MP-denoised", denoised_cov),
        ("Ledoit-Wolf", shrunk_cov),
    ]
    for label, cov in covariances:
        weights = minimum_variance_weights(cov)
        risk = realized_variance(weights, true_cov)
        print(f"{label:15s} out-of-sample variance = {risk:.6f}  gross weight = {np.abs(weights).sum():.3f}")

    print()
    print("=" * 70)
    print("3. Error maximization: mean-variance weights under a noisy mean estimate")
    print("=" * 70)
    true_mean = np.zeros(N_ASSETS)  # true factor model has zero mean returns by construction
    noisy_mean_estimate = returns.mean(axis=0)  # a T=150-day sample mean is a very noisy estimate of 0
    print(
        "true mean return: 0 for all assets. Sample-mean estimate std across assets: "
        f"{noisy_mean_estimate.std():.5f}"
    )

    for label, cov in covariances[1:]:  # skip the oracle covariance here, not available in practice
        weights = mean_variance_weights(cov, noisy_mean_estimate, risk_aversion=1.0)
        realized_return = weights @ true_mean  # true expected return of this bet is exactly 0
        realized_risk = realized_variance(weights, true_cov)
        print(
            f"{label:15s} gross weight = {np.abs(weights).sum():9.2f}  "
            f"true realized risk = {realized_risk:12.4f}  true expected return = {realized_return:+.6f}"
        )
    print(
        "-> every column's TRUE expected return is exactly 0 (noisy_mean_estimate is pure sampling"
        " noise), so any nonzero gross weight is purely betting on noise. The raw-covariance"
        " optimizer takes the largest, riskiest bets on exactly that noise -- 'error maximization'"
        " (Michaud, 1989) -- while the regularized covariances are more conservative."
    )

    print()
    print("=" * 70)
    print("4. Dollar-neutral quantile long/short on realized sample returns")
    print("=" * 70)
    signal = returns.mean(axis=0)  # cross-sectional "which assets did well" signal
    ls_weights = quantile_long_short_weights(signal, n_quantiles=5)
    ls_risk = realized_variance(ls_weights, true_cov)
    print(f"gross exposure = {np.abs(ls_weights).sum():.3f}  net exposure = {ls_weights.sum():.6f}")
    print(f"true portfolio variance = {ls_risk:.6f}")


if __name__ == "__main__":
    main()
