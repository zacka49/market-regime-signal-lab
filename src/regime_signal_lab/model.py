from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ClassifierEstimator(Protocol):
    """Structural type for the sklearn-style classifiers used in this module."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ClassifierEstimator: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class ValidationResult:
    name: str
    auc: float
    accuracy: float
    predictions: pd.DataFrame


def candidate_models() -> dict[str, ClassifierEstimator]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, C=0.3)),
            ]
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.04,
            max_depth=2,
            random_state=3,
        ),
    }


def walk_forward_validate(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    model: ClassifierEstimator,
    min_train_size: int = 800,
    test_size: int = 125,
    embargo: int = 0,
) -> ValidationResult:
    """Expanding-window walk-forward validation.

    `embargo` drops the most recent `embargo` rows of each fold's training
    window (purging), leaving a gap before the test window starts. In this
    pipeline specifically, leakage is already avoided by construction --
    labels are single-day-ahead (no overlapping label horizons to purge) and
    features only look backward (see tests/test_no_lookahead.py) -- so an
    embargo is not required for correctness here. It's included anyway as
    the standard extra margin of safety against near-boundary serial
    correlation (Lopez de Prado, "Advances in Financial Machine Learning",
    ch. 7) and to demonstrate the technique; embargo=0 (the default)
    reproduces the original behavior exactly.
    """
    rows = []

    for start in range(min_train_size, len(frame) - test_size, test_size):
        train = frame.iloc[: start - embargo]
        test = frame.iloc[start : start + test_size]

        fitted = model.fit(train[features], train["target"])
        probabilities = fitted.predict_proba(test[features])[:, 1]

        fold = test[["date", "next_return", "target"]].copy()
        fold["probability"] = probabilities
        rows.append(fold)

    predictions = pd.concat(rows, ignore_index=True)
    predicted_class = (predictions["probability"] >= 0.5).astype(int)
    auc = roc_auc_score(predictions["target"], predictions["probability"])
    accuracy = accuracy_score(predictions["target"], predicted_class)

    return ValidationResult(
        name=model_name, auc=float(auc), accuracy=float(accuracy), predictions=predictions
    )


def choose_best(results: list[ValidationResult]) -> ValidationResult:
    return max(results, key=lambda result: (result.auc, result.accuracy))
