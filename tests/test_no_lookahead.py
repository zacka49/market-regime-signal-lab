from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_signal_lab.features import build_features, feature_columns
from regime_signal_lab.simulate import simulate_market


def test_features_are_unchanged_by_shocking_the_future():
    """Feature columns at time t must depend only on data up to and including t.

    We build features once, then corrupt returns/prices strictly after a
    cutoff date and rebuild. If any feature at or before the cutoff moves,
    it must be reading data from the future -- a leakage bug.
    """
    raw = simulate_market(n_days=600, seed=11)
    baseline = build_features(raw)
    cols = feature_columns(baseline)

    cutoff_idx = 400
    shock_start = cutoff_idx + 5  # buffer past the largest rolling window (60)
    corrupted_raw = raw.copy()
    rng = np.random.default_rng(99)
    shocked_returns = corrupted_raw["return"].to_numpy(copy=True)
    shocked_returns[shock_start:] = rng.normal(0, 0.05, size=len(shocked_returns) - shock_start)
    corrupted_raw["return"] = shocked_returns
    corrupted_raw["price"] = 100.0 * np.exp(np.cumsum(shocked_returns))

    corrupted = build_features(corrupted_raw)

    merged = baseline.merge(corrupted, on="date", suffixes=("_base", "_shock"))
    cutoff_date = raw.loc[cutoff_idx, "date"]
    past = merged[merged["date"] <= cutoff_date]

    assert len(past) > 50, "expected a healthy number of rows before the cutoff"
    for col in cols:
        pd.testing.assert_series_equal(
            past[f"{col}_base"].reset_index(drop=True),
            past[f"{col}_shock"].reset_index(drop=True),
            check_names=False,
            obj=col,
        )


def test_target_is_the_only_forward_looking_column():
    """target/next_return are labels, not features -- they must look forward
    by exactly one day, and feature_columns() must never include them."""
    raw = simulate_market(n_days=300, seed=3)
    frame = build_features(raw)
    cols = feature_columns(frame)

    assert "target" not in cols
    assert "next_return" not in cols

    raw_indexed = raw.set_index("date")
    for _, row in frame.sample(20, random_state=0).iterrows():
        next_return = raw_indexed.loc[raw_indexed.index > row["date"], "return"].iloc[0]
        assert row["next_return"] == pytest.approx(next_return)
        assert row["target"] == int(next_return > 0)


def test_no_nan_or_inf_in_feature_columns():
    raw = simulate_market(n_days=400, seed=5)
    frame = build_features(raw)
    cols = feature_columns(frame)

    assert not frame[cols].isna().any().any()
    assert np.isfinite(frame[cols].to_numpy()).all()
