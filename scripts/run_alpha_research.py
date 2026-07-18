"""Alpha research report: evaluate a small signal zoo (each with an explicit
economic hypothesis, see alphas.py) against the standard IC-based harness,
including a placebo signal that should show ~zero predictive power. Then
look at redundancy between signals and how a few combination rules compare.

All IC/decile/Sharpe numbers here are in-sample diagnostics over the full
history (standard practice for characterizing a candidate signal's raw
properties) -- NOT a leak-free backtest. That happens separately when a
signal is fed into model.walk_forward_validate as a feature column.
"""

from __future__ import annotations

import pandas as pd

from regime_signal_lab.alpha_eval import (
    AlphaReport,
    build_alpha_report,
    combine_equal_weight,
    combine_ic_weighted,
    combine_ridge,
    information_coefficient,
    orthogonalize,
    signal_correlation_matrix,
)
from regime_signal_lab.alphas import (
    mean_reversion_signal,
    momentum_signal,
    placebo_signal,
    regime_transition_signal,
    vol_managed_signal,
)
from regime_signal_lab.regime_detection import walk_forward_regime_posteriors
from regime_signal_lab.simulate import simulate_market


def print_report(report: AlphaReport) -> None:
    print(f"\n--- {report.name} ---")
    print(f"IC (full sample)          {report.ic:+.4f}")
    print(f"IC t-stat (rolling, naive) {report.ic_t_stat:+.2f}  <- anti-conservative, see docstring")
    print(f"IC t-stat (non-overlapping) {report.non_overlapping_ic_t_stat:+.2f}")
    print(f"autocorrelation(1d)  {report.autocorr_lag1:+.3f}")
    print(f"implied turnover     {report.implied_turnover:.3f}")
    print(f"after-cost Sharpe    {report.after_cost_sharpe:+.3f}")
    print("IC by horizon:")
    print(report.ic_decay.to_string())
    print("decile forward returns:")
    print(report.decile_returns.to_string())


def _align_to_raw(signal: pd.Series, signal_dates: pd.Series, raw: pd.DataFrame) -> pd.Series:
    """walk_forward_regime_posteriors (and anything derived from it) is
    concatenated across folds with a fresh 0..n index, which does NOT line
    up positionally with `raw`. Realign by date instead, left-joining onto
    raw's full date range so pre-min_train_size / trailing rows become NaN
    (correctly excluded downstream) rather than silently misaligned."""
    lookup = pd.DataFrame({"date": signal_dates, "value": signal.to_numpy()})
    merged = raw[["date"]].merge(lookup, on="date", how="left")
    return pd.Series(merged["value"].to_numpy(), index=raw.index)


def main() -> None:
    raw = simulate_market(n_days=4000, seed=7)
    posteriors = walk_forward_regime_posteriors(raw, min_train_size=800, test_size=125, n_states=3, seed=0)
    regime_transition = _align_to_raw(regime_transition_signal(posteriors), posteriors["date"], raw)

    signals = {
        "mean_reversion": mean_reversion_signal(raw, lookback=5),
        "momentum": momentum_signal(raw, lookback=60),
        "vol_managed": vol_managed_signal(raw, vol_window=20),
        "regime_transition": regime_transition,
        "placebo": placebo_signal(raw, seed=123),
    }

    print("=" * 70)
    print("1. Per-signal alpha report")
    print("=" * 70)
    reports = {}
    for name, signal in signals.items():
        aligned_signal = signal.reindex(raw.index)
        report = build_alpha_report(name, aligned_signal, raw)
        reports[name] = report
        print_report(report)

    print()
    print("=" * 70)
    print("2. Redundancy: signal correlation matrix")
    print("=" * 70)
    signal_frame = pd.DataFrame({name: s.reindex(raw.index) for name, s in signals.items()}).dropna()
    print(signal_correlation_matrix(signal_frame).round(3).to_string())

    print()
    print("=" * 70)
    print("3. Orthogonalized correlation (order: economic signals first, placebo last)")
    print("=" * 70)
    order = ["mean_reversion", "momentum", "vol_managed", "regime_transition", "placebo"]
    orthogonal = orthogonalize(signal_frame[order])
    print(signal_correlation_matrix(orthogonal).round(3).to_string())

    print()
    print("=" * 70)
    print("4. Combination methods (in-sample diagnostic, see module docstring)")
    print("=" * 70)
    forward_return = raw["return"].shift(-1).reindex(signal_frame.index)
    combined_equal = combine_equal_weight(signal_frame)
    combined_ic = combine_ic_weighted(signal_frame, forward_return)
    combined_ridge = combine_ridge(signal_frame, forward_return)

    for name, combined in [
        ("equal_weight", combined_equal),
        ("ic_weighted", combined_ic),
        ("ridge", combined_ridge),
    ]:
        ic = information_coefficient(combined, forward_return)
        print(f"{name:15s} IC={ic:+.4f}")


if __name__ == "__main__":
    main()
