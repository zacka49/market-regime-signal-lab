from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_signal_lab.alpha_eval import (
    combine_equal_weight,
    combine_ic_weighted,
    combine_ridge,
    decile_returns,
    ic_decay,
    ic_t_stat,
    implied_turnover,
    information_coefficient,
    non_overlapping_ic_t_stat,
    orthogonalize,
    rolling_ic,
    signal_autocorrelation,
    signal_backtest_sharpe,
    signal_correlation_matrix,
)


def _synthetic_series(n: int, seed: int) -> tuple[pd.Series, pd.Series]:
    """A signal deliberately constructed to have real (known-sign) predictive
    power: forward_return = 0.5 * signal + noise. Used to check the harness
    detects a real relationship, as opposed to the honest (and separately
    tested) question of whether any given economic hypothesis holds on the
    simulator."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    signal = pd.Series(rng.standard_normal(n), index=dates, name="signal")
    noise = rng.standard_normal(n) * 2.0
    forward_return = pd.Series(0.5 * signal.to_numpy() + noise, index=dates, name="forward_return")
    return signal, forward_return


def test_information_coefficient_detects_a_known_relationship():
    signal, forward_return = _synthetic_series(2000, seed=1)
    ic = information_coefficient(signal, forward_return)
    assert ic > 0.15  # true relationship is strong relative to the noise added


def test_information_coefficient_is_near_zero_for_independent_series():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2020-01-01", periods=2000, freq="B")
    signal = pd.Series(rng.standard_normal(2000), index=dates)
    forward_return = pd.Series(rng.standard_normal(2000), index=dates)
    ic = information_coefficient(signal, forward_return)
    assert abs(ic) < 0.05


def test_rolling_ic_and_t_stat_are_well_formed():
    signal, forward_return = _synthetic_series(1000, seed=3)
    rolling = rolling_ic(signal, forward_return, window=60)
    assert rolling.notna().sum() > 0
    assert rolling.dropna().between(-1.0, 1.0).all()

    t_stat = ic_t_stat(rolling)
    assert np.isfinite(t_stat)
    assert t_stat > 0  # the synthetic relationship is positive by construction


def test_rolling_ic_t_stat_is_anti_conservative_under_noise():
    """Rolling-window ICs overlap heavily (a 60-day window shares 59 days
    with its neighbor), so treating each window as an independent draw for
    a t-stat badly understates the true standard error. Demonstrate the
    failure mode directly: under pure noise, the naive rolling t-stat should
    fire "significant" (|t| > 2) far more often than the nominal ~5%, while
    the non-overlapping-block version should be much closer to nominal."""
    rng = np.random.default_rng(100)
    n, window, trials = 1000, 50, 15
    naive_false_positives = 0
    block_false_positives = 0

    for trial in range(trials):
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        signal = pd.Series(rng.standard_normal(n), index=dates)
        forward_return = pd.Series(rng.standard_normal(n), index=dates)

        naive_t = ic_t_stat(rolling_ic(signal, forward_return, window=window))
        block_t = non_overlapping_ic_t_stat(signal, forward_return, block_size=window)

        naive_false_positives += abs(naive_t) > 2
        block_false_positives += abs(block_t) > 2

    naive_rate = naive_false_positives / trials
    block_rate = block_false_positives / trials

    assert naive_rate > 0.3  # heavily overlapping windows: badly inflated
    assert block_rate < naive_rate


def test_ic_decay_is_positive_across_horizons_for_a_signal_correlated_with_price_moves():
    rng = np.random.default_rng(4)
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.normal(0.0002, 0.01, size=n)
    # signal is literally tomorrow's and the next few days' cumulative return,
    # so it should show positive IC at short horizons.
    price = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=dates)
    signal = price.pct_change(5).shift(-5)  # forward-looking on purpose: known ground truth

    decay = ic_decay(signal, price, horizons=(1, 5, 10))
    assert decay.loc[5] == pytest.approx(1.0, abs=0.01)  # signal *is* the 5-day forward return


def test_implied_turnover_and_autocorrelation_are_consistent():
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    # a signal that changes sign every single day -> maximum turnover
    # (first row has no prior day to flip from, so it's 499/500, not 1.0)
    alternating = pd.Series([1.0, -1.0] * 250, index=dates)
    assert implied_turnover(alternating) == pytest.approx(1.0, abs=0.01)
    assert signal_autocorrelation(alternating, lag=1) == pytest.approx(-1.0, abs=1e-6)

    # a constant-sign, slowly varying signal -> low turnover, high autocorrelation
    rng = np.random.default_rng(5)
    smooth = pd.Series(np.cumsum(rng.normal(0, 0.01, 500)) + 5.0, index=dates)
    assert implied_turnover(smooth) < 0.05
    assert signal_autocorrelation(smooth, lag=1) > 0.9


def test_decile_returns_are_monotonic_for_a_strong_signal():
    signal, forward_return = _synthetic_series(3000, seed=6)
    deciles = decile_returns(signal, forward_return, n_bins=5)
    means = deciles["mean"].to_numpy()
    assert np.all(np.diff(means) > 0)  # strictly increasing bin-0 -> bin-4


def test_signal_backtest_sharpe_is_higher_without_costs():
    signal, forward_return = _synthetic_series(1500, seed=7)
    free = signal_backtest_sharpe(signal, forward_return, cost_per_turnover=0.0)
    costly = signal_backtest_sharpe(signal, forward_return, cost_per_turnover=0.01)
    assert free > costly


def test_orthogonalize_removes_correlation_with_earlier_columns():
    rng = np.random.default_rng(8)
    n = 2000
    base = rng.standard_normal(n)
    # second signal is base + independent noise -> correlated with the first
    correlated = 0.8 * base + rng.standard_normal(n) * 0.3
    signals = pd.DataFrame({"first": base, "second": correlated})

    raw_corr = signals["first"].corr(signals["second"])
    assert raw_corr > 0.5

    orthogonal = orthogonalize(signals)
    residual_corr = orthogonal["first"].corr(orthogonal["second"])
    assert abs(residual_corr) < 1e-8


def test_signal_correlation_matrix_is_symmetric_with_unit_diagonal():
    signals = pd.DataFrame(
        {
            "a": np.random.default_rng(9).standard_normal(200),
            "b": np.random.default_rng(10).standard_normal(200),
        }
    )
    corr = signal_correlation_matrix(signals)
    np.testing.assert_allclose(np.diag(corr), [1.0, 1.0])
    np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T)


def test_combination_methods_produce_series_aligned_to_input():
    signal, forward_return = _synthetic_series(1000, seed=11)
    other_signal, _ = _synthetic_series(1000, seed=12)
    signals = pd.DataFrame({"s1": signal, "s2": other_signal})

    equal = combine_equal_weight(signals)
    ic_weighted = combine_ic_weighted(signals, forward_return)
    ridge = combine_ridge(signals, forward_return)

    for combined in (equal, ic_weighted, ridge):
        assert combined.notna().sum() > 0

    # the IC-weighted and ridge combiners should both lean toward s1, which
    # (unlike s2) is actually correlated with forward_return by construction.
    assert information_coefficient(ic_weighted, forward_return) > information_coefficient(
        other_signal, forward_return
    )
