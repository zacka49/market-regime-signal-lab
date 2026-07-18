from __future__ import annotations

import pandas as pd
import pytest

from regime_signal_lab.backtest import run_backtest


def _predictions(probabilities, next_returns):
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=len(probabilities), freq="B"),
            "probability": probabilities,
            "next_return": next_returns,
            "target": [int(r > 0) for r in next_returns],
        }
    )


def test_positions_follow_probability_thresholds():
    # threshold 0.55 -> long if p >= 0.55, short if p <= 0.45, else flat.
    predictions = _predictions(
        probabilities=[0.9, 0.5, 0.1, 0.6, 0.4],
        next_returns=[0.01, 0.01, 0.01, 0.01, 0.01],
    )
    frame, _ = run_backtest(predictions, entry_threshold=0.55, transaction_cost=0.0)

    assert list(frame["position"]) == [1, 0, -1, 1, -1]


def test_turnover_and_costs_reduce_returns():
    predictions = _predictions(
        probabilities=[0.9, 0.9, 0.1, 0.9],
        next_returns=[0.01, 0.01, 0.01, 0.01],
    )
    free, free_metrics = run_backtest(predictions, entry_threshold=0.55, transaction_cost=0.0)
    costly, costly_metrics = run_backtest(predictions, entry_threshold=0.55, transaction_cost=0.01)

    # Position flips 1 -> 1 -> -1 -> 1: turnover on rows 0, 2, 3 (row 1 unchanged).
    assert list(free["turnover"]) == [1, 0, 2, 2]
    assert costly_metrics["total_return"] < free_metrics["total_return"]


def test_equity_curve_matches_manual_compounding():
    predictions = _predictions(
        probabilities=[0.9, 0.9, 0.9],
        next_returns=[0.01, -0.02, 0.03],
    )
    frame, metrics = run_backtest(predictions, entry_threshold=0.55, transaction_cost=0.0)

    expected_equity = (1.01) * (0.98) * (1.03)
    assert frame["equity"].iloc[-1] == pytest.approx(expected_equity)
    assert metrics["total_return"] == pytest.approx(expected_equity - 1.0)


def test_zero_variance_strategy_has_zero_sharpe():
    predictions = _predictions(
        probabilities=[0.5, 0.5, 0.5],
        next_returns=[0.0, 0.0, 0.0],
    )
    _, metrics = run_backtest(predictions, entry_threshold=0.6, transaction_cost=0.0)
    assert metrics["annualised_sharpe"] == 0.0
