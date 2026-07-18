"""Statistical rigor pass on the selected strategy: baselines, block
bootstrap confidence intervals, a bootstrap significance test, an embargo
robustness check, and a deflated Sharpe ratio that honestly accounts for
every configuration tried across this project before reporting one.
"""

from __future__ import annotations

import numpy as np

from regime_signal_lab.alpha_eval import signal_backtest_sharpe
from regime_signal_lab.alphas import (
    mean_reversion_signal,
    momentum_signal,
    placebo_signal,
    vol_managed_signal,
)
from regime_signal_lab.backtest import run_backtest
from regime_signal_lab.evaluation import (
    annualized_sharpe,
    block_bootstrap_ci,
    bootstrap_p_value,
    buy_and_hold_baseline,
    deflated_sharpe_ratio,
    random_signal_baseline,
)
from regime_signal_lab.features import build_features, feature_columns
from regime_signal_lab.model import candidate_models, choose_best, walk_forward_validate
from regime_signal_lab.simulate import simulate_market

MIN_TRAIN_SIZE = 800
TEST_SIZE = 125


def main() -> None:
    raw = simulate_market(n_days=4000, seed=7)
    frame = build_features(raw)
    features = feature_columns(frame)

    print("=" * 70)
    print("1. Model selection (this itself is one of the 'trials' below)")
    print("=" * 70)
    results = [
        walk_forward_validate(
            frame, features, name, model, min_train_size=MIN_TRAIN_SIZE, test_size=TEST_SIZE
        )
        for name, model in candidate_models().items()
    ]
    best = choose_best(results)
    strategy_frame, strategy_metrics = run_backtest(best.predictions)
    print(f"selected model: {best.name}  auc={best.auc:.3f}")
    for k, v in strategy_metrics.items():
        print(f"  {k:18s} {v:.4f}")

    print()
    print("=" * 70)
    print("2. Baselines")
    print("=" * 70)
    _, bh_metrics = buy_and_hold_baseline(best.predictions)
    _, rand_metrics = random_signal_baseline(best.predictions, seed=0)
    print(f"{'strategy':18s} {'total_return':>14s} {'sharpe':>10s}")
    baselines = [
        ("strategy", strategy_metrics),
        ("buy_and_hold", bh_metrics),
        ("random_signal", rand_metrics),
    ]
    for label, m in baselines:
        print(f"{label:18s} {m['total_return']:14.4f} {m['annualised_sharpe']:10.4f}")

    print()
    print("=" * 70)
    print("3. Block bootstrap: confidence intervals + significance")
    print("=" * 70)
    strategy_returns = strategy_frame["strategy_return"].to_numpy()
    sharpe_pt, sharpe_lo, sharpe_hi = block_bootstrap_ci(
        strategy_returns, annualized_sharpe, n_bootstrap=1000, mean_block_length=20, seed=0
    )
    print(f"Sharpe:       {sharpe_pt:+.3f}  90% CI [{sharpe_lo:+.3f}, {sharpe_hi:+.3f}]")

    def total_return_fn(r: np.ndarray) -> float:
        return float(np.prod(1.0 + r) - 1.0)

    ret_pt, ret_lo, ret_hi = block_bootstrap_ci(
        strategy_returns, total_return_fn, n_bootstrap=1000, mean_block_length=20, seed=1
    )
    print(f"Total return: {ret_pt:+.3f}  90% CI [{ret_lo:+.3f}, {ret_hi:+.3f}]")

    p_value = bootstrap_p_value(strategy_returns, n_bootstrap=1000, mean_block_length=20, seed=2)
    print(f"Bootstrap p-value (mean daily return == 0): {p_value:.3f}")
    if sharpe_lo <= 0.0 <= sharpe_hi:
        print("-> Sharpe's 90% CI includes 0: cannot rule out zero true skill at this confidence level.")
    else:
        print("-> Sharpe's 90% CI excludes 0.")

    print()
    print("=" * 70)
    print("4. Embargo robustness check")
    print("=" * 70)
    for embargo in [0, 10, 30]:
        result = walk_forward_validate(
            frame, features, f"embargo_{embargo}", candidate_models()[best.name],
            min_train_size=MIN_TRAIN_SIZE, test_size=TEST_SIZE, embargo=embargo,
        )
        print(f"embargo={embargo:3d}  auc={result.auc:.4f}  accuracy={result.accuracy:.4f}")

    print()
    print("=" * 70)
    print("5. Deflated Sharpe ratio: accounting for everything tried")
    print("=" * 70)
    trial_sharpes = []
    for result in results:
        _, m = run_backtest(result.predictions)
        trial_sharpes.append(m["annualised_sharpe"] / np.sqrt(252))  # de-annualize to per-period

    for signal_fn in [mean_reversion_signal, momentum_signal, vol_managed_signal]:
        signal = signal_fn(raw)
        forward_return = raw["return"].shift(-1)
        sharpe = signal_backtest_sharpe(signal, forward_return) / np.sqrt(252)
        trial_sharpes.append(sharpe)
    placebo = placebo_signal(raw, seed=123)
    trial_sharpes.append(signal_backtest_sharpe(placebo, raw["return"].shift(-1)) / np.sqrt(252))

    trial_sharpes_arr = np.array(trial_sharpes)
    dsr = deflated_sharpe_ratio(strategy_returns, trial_sharpes=trial_sharpes_arr)
    print(f"Number of configurations considered: {len(trial_sharpes_arr)}")
    print(f"Spread of trial Sharpes (per-period): std={trial_sharpes_arr.std(ddof=1):.4f}")
    print(f"Deflated Sharpe ratio (P[true skill > 0]): {dsr:.3f}")


if __name__ == "__main__":
    main()
