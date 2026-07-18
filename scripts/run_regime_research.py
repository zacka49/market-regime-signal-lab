"""Regime detection quality vs. ground truth, and whether inferred regimes
help the downstream trading model.

Three questions, in order:
1. How well does an unsupervised Gaussian HMM recover the simulator's true
   hidden regimes? (accuracy, confusion matrix, detection lag)
2. Does adding leak-free regime-probability features help the classifier?
3. Does fitting a separate classifier per inferred regime help further?

This is deliberately not a hard pass/fail script: (2) and (3) are open
research questions and the honest answer may be "no, and here is why."
"""

from __future__ import annotations

import numpy as np

from regime_signal_lab.backtest import run_backtest
from regime_signal_lab.features import build_features, feature_columns
from regime_signal_lab.model import candidate_models, walk_forward_validate
from regime_signal_lab.regime_detection import (
    detect_regimes,
    walk_forward_regime_conditioned_validate,
    walk_forward_regime_posteriors,
)
from regime_signal_lab.simulate import simulate_market

MIN_TRAIN_SIZE = 800
TEST_SIZE = 125


def print_confusion_matrix(confusion: np.ndarray) -> None:
    header = "true\\pred  " + "  ".join(f"regime_{j}" for j in range(confusion.shape[1]))
    print(header)
    for i, row in enumerate(confusion):
        print(f"regime_{i}   " + "  ".join(f"{v:8d}" for v in row))


def main() -> None:
    raw = simulate_market(n_days=4000, seed=7)

    print("=" * 70)
    print("1. Regime detection quality (Gaussian HMM vs ground truth)")
    print("=" * 70)
    detection = detect_regimes(raw["return"].to_numpy(), raw["regime"].to_numpy(), n_states=3, seed=0)
    print(f"accuracy            {detection.accuracy:.3f}   (chance level = 0.333)")
    print(f"mean detection lag  {detection.mean_detection_lag:.1f} trading days")
    print()
    print_confusion_matrix(detection.confusion)

    frame = build_features(raw)
    features = feature_columns(frame)
    model_template = candidate_models()["logistic_regression"]

    print()
    print("=" * 70)
    print("2. Downstream model: regime-agnostic vs regime-aware")
    print("=" * 70)

    baseline = walk_forward_validate(
        frame, features, "baseline", model_template,
        min_train_size=MIN_TRAIN_SIZE, test_size=TEST_SIZE,
    )

    regime_posteriors = walk_forward_regime_posteriors(
        frame, min_train_size=MIN_TRAIN_SIZE, test_size=TEST_SIZE, n_states=3, seed=0,
    )
    augmented_frame = frame.merge(regime_posteriors, on="date", how="inner")
    regime_prob_cols = [c for c in augmented_frame.columns if c.startswith("regime_prob_")]
    augmented_features = features + regime_prob_cols

    augmented = walk_forward_validate(
        augmented_frame, augmented_features, "regime_feature", model_template,
        min_train_size=MIN_TRAIN_SIZE, test_size=TEST_SIZE,
    )

    conditioned = walk_forward_regime_conditioned_validate(
        frame, features, "regime_conditioned", model_template,
        min_train_size=MIN_TRAIN_SIZE, test_size=TEST_SIZE, n_states=3, seed=0,
    )

    for result in [baseline, augmented, conditioned]:
        print(f"{result.name:20s} auc={result.auc:.3f} accuracy={result.accuracy:.3f}")

    print()
    print("=" * 70)
    print("3. Backtest impact")
    print("=" * 70)
    for result in [baseline, augmented, conditioned]:
        _, metrics = run_backtest(result.predictions)
        summary = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"{result.name:20s} {summary}")


if __name__ == "__main__":
    main()
