from __future__ import annotations

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

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


def _mvp_realized_variance(cov_for_weights: np.ndarray, true_cov: np.ndarray) -> float:
    weights = minimum_variance_weights(cov_for_weights)
    return float(weights @ true_cov @ weights)


def test_simulate_factor_returns_matches_true_covariance_at_large_sample():
    returns, beta, true_cov = simulate_factor_returns(n_assets=20, n_days=20_000, seed=1)
    sample_cov = np.cov(returns, rowvar=False)
    relative_error = np.linalg.norm(sample_cov - true_cov) / np.linalg.norm(true_cov)
    assert relative_error < 0.05
    assert returns.shape == (20_000, 20)
    assert beta.shape == (20, 2)


def test_pca_recovers_the_market_factor_direction():
    _, beta, true_cov = simulate_factor_returns(n_assets=20, n_days=20_000, seed=1)
    eigenvalues, eigenvectors = pca_eigendecomposition(true_cov)

    assert np.all(np.diff(eigenvalues) <= 0)  # descending order
    market_beta = beta[:, 0]
    pc1 = eigenvectors[:, 0]
    cos_sim = abs(np.dot(pc1, market_beta) / (np.linalg.norm(pc1) * np.linalg.norm(market_beta)))
    assert cos_sim > 0.95  # PC1 should closely track the dominant (market) factor's loadings


def test_explained_variance_ratio_sums_to_one():
    eigenvalues = np.array([5.0, 3.0, 2.0])
    ratios = explained_variance_ratio(eigenvalues)
    assert ratios.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(ratios, [0.5, 0.3, 0.2])


def test_marchenko_pastur_bounds_special_case_q_equals_one():
    # for q = n_assets/n_obs = 1, MP support is [0, 4*variance] exactly
    lower, upper = marchenko_pastur_bounds(n_assets=100, n_obs=100, variance=1.0)
    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(4.0)


def test_denoising_pure_noise_covariance_moves_closer_to_true_identity():
    n_assets, n_obs, vol = 30, 120, 0.01
    rng = np.random.default_rng(2)
    noise_returns = rng.normal(0.0, vol, size=(n_obs, n_assets))
    noise_cov = np.cov(noise_returns, rowvar=False)
    true_cov = np.eye(n_assets) * vol**2

    denoised = denoise_covariance_mp(noise_cov, n_obs)

    raw_error = np.linalg.norm(noise_cov - true_cov)
    denoised_error = np.linalg.norm(denoised - true_cov)
    assert denoised_error < raw_error


def test_denoising_preserves_a_genuine_factor_above_the_mp_edge():
    n_assets, n_days = 20, 250
    returns, _, true_cov = simulate_factor_returns(n_assets, n_days, seed=3)
    sample_cov = np.cov(returns, rowvar=False)

    raw_eigenvalues, _ = pca_eigendecomposition(sample_cov)
    denoised = denoise_covariance_mp(sample_cov, n_days)
    denoised_eigenvalues, _ = pca_eigendecomposition(denoised)

    # the top eigenvalue (real market factor) should survive denoising
    # largely intact, while total variance (trace) is preserved.
    assert denoised_eigenvalues[0] == pytest.approx(raw_eigenvalues[0], rel=0.15)
    assert np.trace(denoised) == pytest.approx(np.trace(sample_cov), rel=0.05)


def test_denoising_reduces_out_of_sample_minimum_variance_portfolio_risk():
    """RMT denoising is proven to help minimum-variance PORTFOLIO RISK
    out-of-sample, not necessarily Frobenius-norm proximity to the true
    covariance matrix -- matrix inversion in minimum_variance_weights()
    amplifies whichever eigen-directions have the (possibly spuriously) low
    estimated eigenvalues, and that's exactly what denoising corrects. The
    effect is strongest when n_assets/n_days is not small, so use a modest
    sample relative to the number of assets and average over several seeds
    to avoid a single noisy draw."""
    n_assets, n_days, n_seeds = 40, 150, 15
    raw_risks, denoised_risks = [], []

    for seed in range(n_seeds):
        returns, _, true_cov = simulate_factor_returns(n_assets, n_days, seed=seed)
        sample_cov = np.cov(returns, rowvar=False)
        denoised = denoise_covariance_mp(sample_cov, n_days)

        raw_risks.append(_mvp_realized_variance(sample_cov, true_cov))
        denoised_risks.append(_mvp_realized_variance(denoised, true_cov))

    assert np.mean(denoised_risks) < np.mean(raw_risks)


def test_ledoit_wolf_shrinkage_matches_sklearn():
    n_assets, n_obs = 30, 120
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, size=(n_obs, n_assets))

    shrunk, delta = ledoit_wolf_shrinkage(returns)
    sklearn_lw = LedoitWolf().fit(returns)

    assert delta == pytest.approx(sklearn_lw.shrinkage_, rel=1e-6)
    np.testing.assert_allclose(shrunk, sklearn_lw.covariance_, rtol=1e-6)


def test_shrinkage_reduces_out_of_sample_minimum_variance_portfolio_risk():
    n_assets, n_days, n_seeds = 40, 150, 15
    raw_risks, shrunk_risks = [], []

    for seed in range(n_seeds):
        returns, _, true_cov = simulate_factor_returns(n_assets, n_days, seed=seed)
        sample_cov = np.cov(returns, rowvar=False)
        shrunk, _ = ledoit_wolf_shrinkage(returns)

        raw_risks.append(_mvp_realized_variance(sample_cov, true_cov))
        shrunk_risks.append(_mvp_realized_variance(shrunk, true_cov))

    assert np.mean(shrunk_risks) < np.mean(raw_risks)


def test_ledoit_wolf_shrinkage_intensity_is_bounded():
    n_assets, n_obs = 10, 500
    returns, _, _ = simulate_factor_returns(n_assets, n_obs, seed=5)
    _, delta = ledoit_wolf_shrinkage(returns)
    assert 0.0 <= delta <= 1.0


def test_minimum_variance_weights_are_inverse_variance_for_diagonal_covariance():
    variances = np.array([0.01, 0.04, 0.09])
    cov = np.diag(variances)
    weights = minimum_variance_weights(cov)

    expected = (1 / variances) / (1 / variances).sum()
    np.testing.assert_allclose(weights, expected)
    assert weights.sum() == pytest.approx(1.0)


def test_mean_variance_weights_proportional_to_returns_for_isotropic_covariance():
    cov = np.eye(4) * 0.02
    expected_returns = np.array([0.01, 0.02, -0.01, 0.0])
    weights = mean_variance_weights(cov, expected_returns, risk_aversion=2.0)
    np.testing.assert_allclose(weights, expected_returns / (2.0 * 0.02))


def test_quantile_long_short_weights_are_dollar_neutral_and_unit_gross():
    signal = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 0.0, 6.0, -1.0, 7.0, -2.0])
    weights = quantile_long_short_weights(signal, n_quantiles=5)

    assert weights.sum() == pytest.approx(0.0, abs=1e-10)
    assert np.abs(weights).sum() == pytest.approx(1.0)
    # the two highest-signal names should be long, the two lowest short
    top_two = np.argsort(signal)[-2:]
    bottom_two = np.argsort(signal)[:2]
    assert np.all(weights[top_two] > 0)
    assert np.all(weights[bottom_two] < 0)
