from __future__ import annotations

import numpy as np

from regime_signal_lab.simulate import REGIME_TRANSITION_MATRIX, simulate_market


def test_transition_matrix_rows_are_valid_probability_distributions():
    assert REGIME_TRANSITION_MATRIX.shape == (3, 3)
    assert np.all(REGIME_TRANSITION_MATRIX >= 0)
    row_sums = REGIME_TRANSITION_MATRIX.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(3))


def test_simulate_market_shape_and_columns():
    frame = simulate_market(n_days=250, seed=1)
    assert len(frame) == 250
    assert set(frame.columns) == {"date", "price", "return", "regime"}
    assert frame["regime"].isin([0, 1, 2]).all()
    assert (frame["price"] > 0).all()


def test_simulate_market_is_deterministic_given_a_seed():
    first = simulate_market(n_days=500, seed=42)
    second = simulate_market(n_days=500, seed=42)
    import pandas as pd

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_paths():
    first = simulate_market(n_days=500, seed=1)
    second = simulate_market(n_days=500, seed=2)
    assert not first["return"].equals(second["return"])
