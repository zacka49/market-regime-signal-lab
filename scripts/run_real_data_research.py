"""Real-data robustness pass: run the same pipeline validated on synthetic
data against real ETFs, sweep cost/threshold and signal-lookback parameters
for stability (a narrow spike vs. a broad plateau), and compare binary +-1
position sizing against Kelly-derived and volatility-targeted sizing.

Expectation, stated up front: the synthetic simulator has genuine
autoregressive/regime structure by construction, so a model finding weak
signal there is not surprising. Real daily ETF returns are close to a
random walk after costs. If the numbers below come back weak or negative,
that is the finding, not a bug -- see docs/research_report.md.
"""

from __future__ import annotations

import numpy as np

from regime_signal_lab.alpha_eval import signal_backtest_sharpe
from regime_signal_lab.alphas import momentum_signal
from regime_signal_lab.backtest import run_backtest
from regime_signal_lab.data import load_real
from regime_signal_lab.features import build_features, feature_columns
from regime_signal_lab.model import candidate_models, walk_forward_validate
from regime_signal_lab.sizing import fractional_kelly_position, kelly_fraction, run_vol_targeted_backtest

TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLK"]


def main() -> None:
    print("=" * 70)
    print("1. Pipeline on real ETFs (walk-forward, same features/model as synthetic)")
    print("=" * 70)
    real_results = {}
    for ticker in TICKERS:
        raw = load_real(ticker, start="2010-01-01")
        frame = build_features(raw)
        features = feature_columns(frame)
        if len(frame) < 1000:
            print(f"{ticker:6s} insufficient history after feature warmup ({len(frame)} rows), skipping")
            continue

        model = candidate_models()["logistic_regression"]
        result = walk_forward_validate(
            frame, features, ticker, model, min_train_size=800, test_size=125
        )
        _, metrics = run_backtest(result.predictions)
        real_results[ticker] = (result, metrics)
        print(
            f"{ticker:6s} n_days={len(raw):5d}  auc={result.auc:.4f}  "
            f"sharpe={metrics['annualised_sharpe']:+.3f}  total_return={metrics['total_return']:+.3f}"
        )

    print()
    print("=" * 70)
    print("2. Cost x threshold sensitivity (SPY)")
    print("=" * 70)
    if "SPY" in real_results:
        spy_result, _ = real_results["SPY"]
        thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]
        costs_bps = [0.0, 2.0, 5.0, 10.0]
        header = "threshold\\cost_bps  " + "  ".join(f"{c:>7.1f}" for c in costs_bps)
        print(header)
        for threshold in thresholds:
            row = []
            for cost_bps in costs_bps:
                _, metrics = run_backtest(
                    spy_result.predictions, entry_threshold=threshold, transaction_cost=cost_bps / 10_000
                )
                row.append(metrics["annualised_sharpe"])
            print(f"{threshold:>18.2f}  " + "  ".join(f"{v:7.3f}" for v in row))

    print()
    print("=" * 70)
    print("3. Parameter stability: momentum lookback x transaction cost (SPY)")
    print("=" * 70)
    if "SPY" in real_results:
        raw_spy = load_real("SPY", start="2010-01-01")
        forward_return = raw_spy["return"].shift(-1)
        lookbacks = [10, 20, 40, 60, 90, 120]
        costs_bps = [0.0, 5.0, 10.0]
        header = "lookback\\cost_bps  " + "  ".join(f"{c:>7.1f}" for c in costs_bps)
        print(header)
        for lookback in lookbacks:
            signal = momentum_signal(raw_spy, lookback=lookback)
            row = [
                signal_backtest_sharpe(signal, forward_return, cost_per_turnover=cost_bps / 10_000)
                for cost_bps in costs_bps
            ]
            print(f"{lookback:>17d}  " + "  ".join(f"{v:7.3f}" for v in row))

    print()
    print("=" * 70)
    print("4. Position sizing: binary +-1 vs Kelly vs volatility targeting")
    print("=" * 70)
    if "SPY" in real_results:
        spy_result, binary_metrics = real_results["SPY"]
        strategy_returns = (
            np.sign(spy_result.predictions["probability"] - 0.5) * spy_result.predictions["next_return"]
        ).to_numpy()
        full_kelly = kelly_fraction(strategy_returns)
        half_kelly = fractional_kelly_position(strategy_returns, fraction=0.5)
        print(f"full Kelly fraction (mu/sigma^2): {full_kelly:.2f}x leverage")
        print("  ^ NOT a recommendation -- see docs/mathematical_notes.md Section 6.1")
        print(f"half Kelly fraction:               {half_kelly:.2f}x leverage")

        _, vol_target_metrics = run_vol_targeted_backtest(
            spy_result.predictions, target_vol=0.01, vol_window=20, max_leverage=3.0
        )
        print()
        columns = f"{'sizing':20s} {'sharpe':>10s} {'total_return':>14s} "
        columns += f"{'max_drawdown':>14s} {'mean_leverage':>14s}"
        print(columns)
        print(
            f"{'binary +-1':20s} {binary_metrics['annualised_sharpe']:10.3f} "
            f"{binary_metrics['total_return']:14.3f} {binary_metrics['max_drawdown']:14.3f} {1.0:14.3f}"
        )
        print(
            f"{'vol-targeted':20s} {vol_target_metrics['annualised_sharpe']:10.3f} "
            f"{vol_target_metrics['total_return']:14.3f} {vol_target_metrics['max_drawdown']:14.3f} "
            f"{vol_target_metrics['mean_gross_leverage']:14.3f}"
        )


if __name__ == "__main__":
    main()
