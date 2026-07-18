"""Unsupervised regime detection, scored against the simulator's ground truth.

`simulate_market` gives us hidden regime labels that a real market never
would -- so instead of just fitting an HMM and hoping it's sensible, we can
directly measure how well it recovers the truth: confusion matrix, accuracy,
and detection lag. `walk_forward_regime_posteriors` then reuses the fitted
HMM's parameters inside the causal `hamilton_filter` (see filtering.py) to
build a leak-free regime-probability feature for the downstream trading
model, refit on each walk-forward fold exactly like `model.walk_forward_validate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score

from .filtering import hamilton_filter
from .model import ClassifierEstimator, ValidationResult


def fit_gaussian_hmm(returns: np.ndarray, n_states: int = 3, seed: int = 0) -> GaussianHMM:
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=seed)
    model.fit(returns.reshape(-1, 1))
    return model


def hmm_gaussian_params(model: GaussianHMM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (transition matrix, per-state means, per-state stds) from a
    fitted 1-D GaussianHMM, in the shape hamilton_filter() expects."""
    n_states = model.n_components
    transition = model.transmat_
    means = model.means_.reshape(n_states)
    stds = np.sqrt(model.covars_).reshape(n_states)
    return transition, means, stds


def _align_states(true_regimes: np.ndarray, inferred_states: np.ndarray, n_states: int) -> dict[int, int]:
    """HMM state indices are arbitrary/unlabeled. Find the one-to-one mapping
    from inferred state -> true regime that maximizes agreement (Hungarian
    algorithm on the confusion matrix) so label-based scoring is meaningful."""
    confusion = np.zeros((n_states, n_states), dtype=int)
    for inferred, true in zip(inferred_states, true_regimes):
        confusion[inferred, true] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    return dict(zip(row_ind.tolist(), col_ind.tolist()))


@dataclass(frozen=True)
class RegimeDetectionResult:
    model: GaussianHMM
    aligned_states: np.ndarray
    posterior: np.ndarray  # (T, n_states), columns ordered to match true regime ids
    label_map: dict[int, int]
    accuracy: float
    confusion: np.ndarray
    mean_detection_lag: float


def detect_regimes(
    returns: np.ndarray, true_regimes: np.ndarray, n_states: int = 3, seed: int = 0
) -> RegimeDetectionResult:
    model = fit_gaussian_hmm(returns, n_states=n_states, seed=seed)
    posterior_raw = model.predict_proba(returns.reshape(-1, 1))
    inferred_states = posterior_raw.argmax(axis=1)

    label_map = _align_states(true_regimes, inferred_states, n_states)
    aligned_states = np.array([label_map[s] for s in inferred_states])

    posterior_aligned = np.zeros_like(posterior_raw)
    for hmm_state, true_regime in label_map.items():
        posterior_aligned[:, true_regime] = posterior_raw[:, hmm_state]

    confusion = np.zeros((n_states, n_states), dtype=int)
    for true, pred in zip(true_regimes, aligned_states):
        confusion[true, pred] += 1
    accuracy = float(np.trace(confusion) / confusion.sum())

    lag = _detection_lag(true_regimes, aligned_states)

    return RegimeDetectionResult(
        model=model,
        aligned_states=aligned_states,
        posterior=posterior_aligned,
        label_map=label_map,
        accuracy=accuracy,
        confusion=confusion,
        mean_detection_lag=lag,
    )


def _detection_lag(true_regimes: np.ndarray, aligned_states: np.ndarray, max_lag: int = 30) -> float:
    """Average number of days between a true regime change and the first
    subsequent day the detector reports the new regime. Censored at max_lag
    for changes the detector never confirms within the window."""
    change_points = np.where(np.diff(true_regimes) != 0)[0] + 1
    lags = []
    for cp in change_points:
        new_regime = true_regimes[cp]
        window_end = min(cp + max_lag, len(aligned_states))
        matches = np.where(aligned_states[cp:window_end] == new_regime)[0]
        lags.append(int(matches[0]) if len(matches) > 0 else max_lag)
    return float(np.mean(lags)) if lags else float("nan")


def _fit_and_filter_fold(
    returns: np.ndarray, start: int, test_size: int, n_states: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian HMM on returns[:start] only, then run the causal
    Hamilton filter over returns[:start+test_size] using those fitted
    parameters. Returns (filtered_probs, order) where filtered_probs has
    shape (start+test_size, n_states) and `order` sorts states by ascending
    fitted mean return -- HMM state indices are otherwise arbitrary and can
    flip between refits, so every caller must relabel through `order`.
    """
    hmm = fit_gaussian_hmm(returns[:start], n_states=n_states, seed=seed)
    transition, means, stds = hmm_gaussian_params(hmm)
    filtered, _ = hamilton_filter(returns[: start + test_size], transition, means, stds)
    order = np.argsort(means)
    return filtered[:, order], order


def walk_forward_regime_posteriors(
    frame: pd.DataFrame,
    min_train_size: int,
    test_size: int,
    n_states: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Leak-free regime-probability features, refit each walk-forward fold.

    For every fold [0, start) -> [start, start+test_size), a fresh HMM is
    fit on returns[:start] only, then the causal Hamilton filter is run over
    returns[:start+test_size] using that fold's fitted parameters, and only
    the test-window filtered probabilities are kept. States are relabeled by
    ascending fitted mean return (so "regime_prob_0" consistently means
    "lowest-drift state") since HMM state indices are otherwise arbitrary
    and can flip between refits.
    """
    returns = frame["return"].to_numpy()
    rows = []

    for start in range(min_train_size, len(frame) - test_size, test_size):
        filtered, _ = _fit_and_filter_fold(returns, start, test_size, n_states, seed)
        fold_posterior = filtered[start : start + test_size]

        fold = frame.iloc[start : start + test_size][["date"]].copy()
        for i in range(n_states):
            fold[f"regime_prob_{i}"] = fold_posterior[:, i]
        rows.append(fold)

    return pd.concat(rows, ignore_index=True)


def walk_forward_regime_conditioned_validate(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    model_template: ClassifierEstimator,
    min_train_size: int,
    test_size: int,
    n_states: int = 3,
    seed: int = 0,
    min_rows_per_regime: int = 50,
) -> ValidationResult:
    """Walk-forward validation with one classifier fit per inferred regime.

    Each fold fits an HMM on the training window only (see
    `_fit_and_filter_fold`), uses it to causally label both the training
    rows (to partition training data by regime) and the test rows (to route
    each test row to its regime's model). A pooled model trained on all of
    that fold's training data is the fallback whenever a regime has fewer
    than `min_rows_per_regime` training rows, so the comparison to
    `model.walk_forward_validate`'s single pooled model is apples-to-apples
    everywhere the split is actually well-supported by data.
    """
    returns = frame["return"].to_numpy()
    rows = []

    for start in range(min_train_size, len(frame) - test_size, test_size):
        train = frame.iloc[:start]
        test = frame.iloc[start : start + test_size]

        filtered, _ = _fit_and_filter_fold(returns, start, test_size, n_states, seed)
        state_labels = filtered.argmax(axis=1)
        train_states = state_labels[:start]
        test_states = state_labels[start : start + test_size]

        pooled = clone(model_template).fit(train[features], train["target"])
        probabilities = pooled.predict_proba(test[features])[:, 1]

        for state in range(n_states):
            train_mask = train_states == state
            test_mask = test_states == state
            if train_mask.sum() < min_rows_per_regime or test_mask.sum() == 0:
                continue
            sub_model = clone(model_template)
            sub_model.fit(train[features].iloc[train_mask], train["target"].iloc[train_mask])
            probabilities[test_mask] = sub_model.predict_proba(test[features].iloc[test_mask])[:, 1]

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
