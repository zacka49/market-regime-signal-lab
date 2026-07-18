from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_signal_lab.sizing import (
    fractional_kelly_position,
    kelly_fraction,
    run_vol_targeted_backtest,
    vol_target_position_scale,
)


def test_kelly_fraction_matches_mean_over_variance():
    returns = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.015])
    expected = returns.mean() / returns.var(ddof=1)
    assert kelly_fraction(returns) == pytest.approx(expected)


def test_kelly_fraction_is_zero_for_zero_variance():
    assert kelly_fraction(np.array([0.01, 0.01, 0.01])) == 0.0


def test_fractional_kelly_is_a_fraction_of_full_kelly():
    returns = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.015])
    full = kelly_fraction(returns)
    half = fractional_kelly_position(returns, fraction=0.5)
    assert half == pytest.approx(0.5 * full)


def test_vol_target_position_scale_caps_at_max_leverage_and_handles_zero_vol():
    realized_vol = pd.Series([0.005, 0.01, 0.0, np.nan, 0.02])
    scale = vol_target_position_scale(realized_vol, target_vol=0.01, max_leverage=3.0)

    assert scale.iloc[0] == pytest.approx(2.0)  # 0.01/0.005 = 2, under the cap
    assert scale.iloc[1] == pytest.approx(1.0)  # 0.01/0.01 = 1
    assert scale.iloc[2] == 0.0  # zero vol -> undefined -> floored to 0
    assert scale.iloc[3] == 0.0  # NaN vol -> floored to 0
    assert scale.iloc[4] == pytest.approx(0.5)  # 0.01/0.02 = 0.5


def test_vol_target_scale_respects_the_cap():
    realized_vol = pd.Series([0.0001])  # would imply huge scale without a cap
    scale = vol_target_position_scale(realized_vol, target_vol=0.01, max_leverage=3.0)
    assert scale.iloc[0] == pytest.approx(3.0)


def _predictions(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "probability": rng.uniform(0.0, 1.0, size=n),
            "next_return": rng.normal(0.0, 0.01, size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )


def test_vol_targeted_backtest_direction_matches_threshold_logic():
    predictions = _predictions(300, seed=1)
    frame, _ = run_vol_targeted_backtest(predictions, entry_threshold=0.55, vol_window=10)

    expected_direction = np.where(
        predictions["probability"] >= 0.55, 1, np.where(predictions["probability"] <= 0.45, -1, 0)
    )
    np.testing.assert_array_equal(frame["direction"].to_numpy(), expected_direction)


def test_vol_targeted_backtest_position_magnitude_varies():
    predictions = _predictions(300, seed=2)
    frame, metrics = run_vol_targeted_backtest(predictions, entry_threshold=0.5, vol_window=10)

    nonzero_positions = frame.loc[frame["direction"] != 0, "position"].abs()
    assert nonzero_positions.nunique() > 5  # not just a constant +-1
    assert metrics["mean_gross_leverage"] >= 0.0


def test_vol_targeted_backtest_respects_max_leverage_cap():
    predictions = _predictions(300, seed=3)
    _, metrics = run_vol_targeted_backtest(predictions, max_leverage=2.0, vol_window=5)
    assert metrics["mean_gross_leverage"] <= 2.0


def test_vol_targeted_backtest_is_leak_free():
    predictions = _predictions(400, seed=4)
    baseline_frame, _ = run_vol_targeted_backtest(predictions, vol_window=15)

    cutoff = 250
    shock_start = cutoff + 5
    corrupted = predictions.copy()
    rng = np.random.default_rng(9)
    corrupted.loc[shock_start:, "next_return"] = rng.normal(0, 0.05, size=len(corrupted) - shock_start)
    shocked_frame, _ = run_vol_targeted_backtest(corrupted, vol_window=15)

    pd.testing.assert_series_equal(
        baseline_frame.loc[:cutoff, "position"],
        shocked_frame.loc[:cutoff, "position"],
    )
