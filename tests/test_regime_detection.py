from __future__ import annotations

import numpy as np
import pandas as pd

from regime_signal_lab.features import build_features, feature_columns
from regime_signal_lab.model import candidate_models
from regime_signal_lab.regime_detection import (
    detect_regimes,
    walk_forward_regime_conditioned_validate,
    walk_forward_regime_posteriors,
)
from regime_signal_lab.simulate import simulate_market


def test_hmm_recovers_ground_truth_regimes_better_than_chance():
    raw = simulate_market(n_days=3000, seed=13)
    result = detect_regimes(raw["return"].to_numpy(), raw["regime"].to_numpy(), n_states=3, seed=0)

    assert result.confusion.sum() == 3000
    assert np.trace(result.confusion) / result.confusion.sum() == result.accuracy
    assert result.accuracy > 0.5  # chance level is 1/3
    assert result.mean_detection_lag >= 0.0
    assert np.isfinite(result.mean_detection_lag)
    assert result.posterior.shape == (3000, 3)
    np.testing.assert_allclose(result.posterior.sum(axis=1), np.ones(3000), atol=1e-6)


def test_regime_posteriors_are_valid_probabilities_and_leak_free():
    raw = simulate_market(n_days=1200, seed=17)
    posteriors = walk_forward_regime_posteriors(raw, min_train_size=500, test_size=200, n_states=3, seed=1)

    prob_cols = [c for c in posteriors.columns if c.startswith("regime_prob_")]
    assert len(prob_cols) == 3
    np.testing.assert_allclose(posteriors[prob_cols].sum(axis=1), np.ones(len(posteriors)), atol=1e-6)
    assert (posteriors[prob_cols] >= -1e-9).all().all()
    assert (posteriors[prob_cols] <= 1 + 1e-9).all().all()


def test_regime_posteriors_unaffected_by_shocking_returns_after_each_folds_test_window():
    """Each fold's HMM is fit on returns[:start] and filtered through
    returns[:start+test_size] -- shocking returns strictly after a fold's
    test window must not change that fold's posterior."""
    raw = simulate_market(n_days=1200, seed=19)
    baseline = walk_forward_regime_posteriors(raw, min_train_size=500, test_size=200, n_states=3, seed=2)

    corrupted_raw = raw.copy()
    rng = np.random.default_rng(5)
    shock_from = 900  # strictly after the first fold's test window [500, 700)
    corrupted_raw.loc[shock_from:, "return"] = rng.normal(0, 0.05, size=len(corrupted_raw) - shock_from)

    corrupted = walk_forward_regime_posteriors(
        corrupted_raw, min_train_size=500, test_size=200, n_states=3, seed=2
    )

    first_fold_dates = raw["date"].iloc[500:700]
    base_first_fold = baseline[baseline["date"].isin(first_fold_dates)]
    shock_first_fold = corrupted[corrupted["date"].isin(first_fold_dates)]

    pd.testing.assert_frame_equal(
        base_first_fold.reset_index(drop=True), shock_first_fold.reset_index(drop=True)
    )


def test_regime_conditioned_validate_produces_valid_predictions():
    raw = simulate_market(n_days=1600, seed=23)
    frame = build_features(raw)
    features = feature_columns(frame)
    model_template = candidate_models()["logistic_regression"]

    result = walk_forward_regime_conditioned_validate(
        frame, features, "regime_conditioned", model_template,
        min_train_size=600, test_size=200, n_states=3, seed=4,
    )

    assert 0.0 <= result.auc <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.predictions["probability"].between(0.0, 1.0).all()
    assert not result.predictions["probability"].isna().any()
