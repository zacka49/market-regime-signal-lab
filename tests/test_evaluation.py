from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_signal_lab.evaluation import (
    annualized_sharpe,
    block_bootstrap_ci,
    bootstrap_p_value,
    buy_and_hold_baseline,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    random_signal_baseline,
    stationary_bootstrap_resample,
)


def _predictions(n: int, next_returns: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "probability": np.full(n, 0.5),
            "next_return": next_returns,
            "target": (next_returns > 0).astype(int),
        }
    )


def test_buy_and_hold_baseline_holds_position_one_with_single_entry_cost():
    predictions = _predictions(5, np.array([0.01, -0.02, 0.03, 0.01, -0.01]))
    frame, metrics = buy_and_hold_baseline(predictions, transaction_cost=0.001)

    assert list(frame["position"]) == [1, 1, 1, 1, 1]
    assert list(frame["turnover"]) == [1, 0, 0, 0, 0]
    expected_return = (1.01 - 0.001) * 0.98 * 1.03 * 1.01 * 0.99 - 1.0
    assert metrics["total_return"] == pytest.approx(expected_return, rel=1e-6)


def test_random_signal_baseline_varies_with_seed():
    predictions = _predictions(500, np.random.default_rng(0).normal(0, 0.01, 500))
    _, metrics_a = random_signal_baseline(predictions, seed=1)
    _, metrics_b = random_signal_baseline(predictions, seed=2)
    assert metrics_a != metrics_b


def test_annualized_sharpe_matches_manual_calculation_and_handles_zero_std():
    returns = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = np.sqrt(252) * returns.mean() / returns.std()
    assert annualized_sharpe(returns) == pytest.approx(expected)
    assert annualized_sharpe(np.zeros(10)) == 0.0


def test_stationary_bootstrap_resample_preserves_length_and_value_set():
    rng = np.random.default_rng(0)
    series = np.arange(100, dtype=float)
    resampled = stationary_bootstrap_resample(series, mean_block_length=5, rng=rng)
    assert len(resampled) == len(series)
    assert set(resampled.tolist()).issubset(set(series.tolist()))


def test_block_bootstrap_ci_contains_point_estimate_and_has_positive_width():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0005, 0.01, 800)
    point, lower, upper = block_bootstrap_ci(
        returns, annualized_sharpe, n_bootstrap=300, mean_block_length=10, seed=1
    )
    assert lower <= point <= upper
    assert upper > lower


def test_bootstrap_p_value_distinguishes_signal_from_noise():
    rng = np.random.default_rng(4)
    strong_signal = rng.normal(0.001, 0.01, 1000)
    pure_noise = rng.normal(0.0, 0.01, 1000)

    p_signal = bootstrap_p_value(strong_signal, n_bootstrap=400, mean_block_length=10, seed=1)
    p_noise = bootstrap_p_value(pure_noise, n_bootstrap=400, mean_block_length=10, seed=2)

    assert p_signal < p_noise
    assert p_noise > 0.3  # should not falsely reject the null for pure noise


def test_expected_max_sharpe_increases_with_more_trials():
    assert expected_max_sharpe_under_null(1, 0.05) == 0.0
    low = expected_max_sharpe_under_null(5, 0.05)
    high = expected_max_sharpe_under_null(200, 0.05)
    assert 0.0 < low < high


def test_deflated_sharpe_ratio_penalizes_more_trials():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0005, 0.01, 1000)
    single_trial_sharpe = returns.mean() / returns.std(ddof=1)

    dsr_one_trial = deflated_sharpe_ratio(returns, trial_sharpes=np.array([single_trial_sharpe]))

    many_trial_sharpes = np.concatenate([[single_trial_sharpe], rng.normal(0.0, 0.03, 49)])
    dsr_many_trials = deflated_sharpe_ratio(returns, trial_sharpes=many_trial_sharpes)

    assert 0.0 <= dsr_many_trials < dsr_one_trial <= 1.0
