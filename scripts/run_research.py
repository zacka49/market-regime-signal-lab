from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_signal_lab.backtest import run_backtest
from regime_signal_lab.features import build_features, feature_columns
from regime_signal_lab.model import candidate_models, choose_best, walk_forward_validate
from regime_signal_lab.simulate import simulate_market


def main() -> None:
    raw = simulate_market()
    frame = build_features(raw)
    features = feature_columns(frame)

    results = [
        walk_forward_validate(frame, features, name, model)
        for name, model in candidate_models().items()
    ]
    best = choose_best(results)
    _, metrics = run_backtest(best.predictions)

    print("Model comparison")
    for result in results:
        print(f"{result.name:20s} auc={result.auc:.3f} accuracy={result.accuracy:.3f}")

    print("\nSelected model")
    print(best.name)

    print("\nBacktest")
    for key, value in metrics.items():
        print(f"{key:18s} {value:.3f}")


if __name__ == "__main__":
    main()
