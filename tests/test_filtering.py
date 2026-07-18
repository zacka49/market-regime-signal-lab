from __future__ import annotations

import numpy as np
import pytest

from regime_signal_lab.filtering import hamilton_filter, stationary_distribution
from regime_signal_lab.simulate import (
    REGIME_DRIFT,
    REGIME_TRANSITION_MATRIX,
    REGIME_VOLATILITY,
    simulate_market,
)


def test_stationary_distribution_is_a_fixed_point():
    pi = stationary_distribution(REGIME_TRANSITION_MATRIX)
    np.testing.assert_allclose(pi @ REGIME_TRANSITION_MATRIX, pi, atol=1e-6)
    assert pi.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(pi >= -1e-9)


def test_hamilton_filter_recovers_true_regime_better_than_chance():
    raw = simulate_market(n_days=3000, seed=21)
    filtered, log_likelihood = hamilton_filter(
        raw["return"].to_numpy(),
        REGIME_TRANSITION_MATRIX,
        REGIME_DRIFT,
        REGIME_VOLATILITY,
    )

    inferred_regime = filtered.argmax(axis=1)
    accuracy = (inferred_regime == raw["regime"].to_numpy()).mean()

    assert np.isfinite(log_likelihood)
    assert filtered.shape == (3000, 3)
    np.testing.assert_allclose(filtered.sum(axis=1), np.ones(3000), atol=1e-8)
    # regimes are persistent (diagonal transition probs 0.90-0.96) and have
    # distinct drift/vol, so the filter should clear chance (1/3) by a wide margin.
    assert accuracy > 0.55
