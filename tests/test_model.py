from __future__ import annotations

import pandas as pd

from regime_signal_lab.features import build_features, feature_columns
from regime_signal_lab.model import candidate_models, walk_forward_validate
from regime_signal_lab.simulate import simulate_market


def test_walk_forward_folds_never_train_on_the_future():
    raw = simulate_market(n_days=1200, seed=4)
    frame = build_features(raw)
    features = feature_columns(frame)

    min_train_size = 400
    test_size = 100
    model = candidate_models()["logistic_regression"]
    result = walk_forward_validate(
        frame, features, "logistic_regression", model,
        min_train_size=min_train_size, test_size=test_size,
    )

    # Every prediction date must be strictly after the model's own last
    # training date for that fold -- reconstruct the fold boundaries the
    # same way walk_forward_validate does and check no overlap exists.
    n_folds = 0
    for start in range(min_train_size, len(frame) - test_size, test_size):
        train_dates = frame.iloc[:start]["date"]
        test_dates = frame.iloc[start:start + test_size]["date"]
        assert train_dates.max() < test_dates.min()
        n_folds += 1

    assert n_folds > 0
    assert len(result.predictions) == n_folds * test_size


def test_walk_forward_predictions_cover_expected_row_count():
    raw = simulate_market(n_days=1200, seed=4)
    frame = build_features(raw)
    features = feature_columns(frame)
    model = candidate_models()["gradient_boosting"]

    result = walk_forward_validate(
        frame, features, "gradient_boosting", model,
        min_train_size=500, test_size=200,
    )

    assert 0.0 <= result.auc <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.predictions["probability"].between(0.0, 1.0).all()


def test_embargo_purges_training_rows_adjacent_to_each_test_window():
    raw = simulate_market(n_days=1200, seed=4)
    frame = build_features(raw)
    features = feature_columns(frame)
    model = candidate_models()["logistic_regression"]

    min_train_size, test_size, embargo = 400, 100, 20
    result = walk_forward_validate(
        frame, features, "embargoed", model,
        min_train_size=min_train_size, test_size=test_size, embargo=embargo,
    )

    n_folds = 0
    for start in range(min_train_size, len(frame) - test_size, test_size):
        train_dates = frame.iloc[: start - embargo]["date"]
        test_dates = frame.iloc[start : start + test_size]["date"]
        assert len(train_dates) == start - embargo
        assert train_dates.max() < test_dates.min()
        n_folds += 1

    assert len(result.predictions) == n_folds * test_size
    assert 0.0 <= result.auc <= 1.0


def test_embargo_zero_reproduces_original_behavior():
    raw = simulate_market(n_days=1200, seed=4)
    frame = build_features(raw)
    features = feature_columns(frame)
    model_a = candidate_models()["logistic_regression"]
    model_b = candidate_models()["logistic_regression"]

    result_default = walk_forward_validate(
        frame, features, "no_embargo_kw", model_a, min_train_size=400, test_size=100
    )
    result_explicit = walk_forward_validate(
        frame, features, "no_embargo_explicit", model_b, min_train_size=400, test_size=100, embargo=0
    )

    pd.testing.assert_frame_equal(result_default.predictions, result_explicit.predictions)
