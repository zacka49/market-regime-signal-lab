from __future__ import annotations

from regime_signal_lab.backtest import run_backtest
from regime_signal_lab.features import build_features, feature_columns
from regime_signal_lab.model import candidate_models, choose_best, walk_forward_validate
from regime_signal_lab.simulate import simulate_market


def _run_pipeline(seed: int) -> dict[str, float]:
    raw = simulate_market(n_days=1200, seed=seed)
    frame = build_features(raw)
    features = feature_columns(frame)
    results = [
        walk_forward_validate(frame, features, name, model, min_train_size=400, test_size=150)
        for name, model in candidate_models().items()
    ]
    best = choose_best(results)
    _, metrics = run_backtest(best.predictions)
    return metrics


def test_full_pipeline_is_deterministic_for_a_fixed_seed():
    first = _run_pipeline(seed=7)
    second = _run_pipeline(seed=7)
    assert first == second


def test_full_pipeline_changes_with_a_different_seed():
    baseline = _run_pipeline(seed=7)
    other = _run_pipeline(seed=8)
    assert baseline != other
