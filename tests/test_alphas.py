from __future__ import annotations

import numpy as np
import pandas as pd

from regime_signal_lab.alphas import (
    mean_reversion_signal,
    momentum_signal,
    placebo_signal,
    regime_transition_signal,
    vol_managed_signal,
)
from regime_signal_lab.regime_detection import walk_forward_regime_posteriors
from regime_signal_lab.simulate import simulate_market

PRICE_SIGNAL_FUNCS = {
    "mean_reversion": lambda frame: mean_reversion_signal(frame, lookback=5, zscore_window=100),
    "momentum": lambda frame: momentum_signal(frame, lookback=20, zscore_window=100),
    "vol_managed": lambda frame: vol_managed_signal(frame, vol_window=20, zscore_window=100),
}


def test_price_based_signals_are_unaffected_by_shocking_the_future():
    raw = simulate_market(n_days=600, seed=31)

    cutoff_idx = 400
    shock_start = cutoff_idx + 5
    corrupted_raw = raw.copy()
    rng = np.random.default_rng(7)
    shocked_returns = corrupted_raw["return"].to_numpy(copy=True)
    shocked_returns[shock_start:] = rng.normal(0, 0.05, size=len(shocked_returns) - shock_start)
    corrupted_raw["return"] = shocked_returns
    corrupted_raw["price"] = 100.0 * np.exp(np.cumsum(shocked_returns))

    cutoff_date = raw.loc[cutoff_idx, "date"]

    for name, signal_fn in PRICE_SIGNAL_FUNCS.items():
        baseline_signal = signal_fn(raw)
        corrupted_signal = signal_fn(corrupted_raw)

        merged = pd.concat(
            [raw["date"], baseline_signal.rename("base"), corrupted_signal.rename("shock")], axis=1
        )
        past = merged[merged["date"] <= cutoff_date]
        pd.testing.assert_series_equal(
            past["base"].reset_index(drop=True),
            past["shock"].reset_index(drop=True),
            check_names=False,
            obj=name,
        )


def test_regime_transition_signal_is_leak_free():
    raw = simulate_market(n_days=1200, seed=33)
    posteriors = walk_forward_regime_posteriors(raw, min_train_size=500, test_size=200, n_states=3, seed=1)
    signal = regime_transition_signal(posteriors)

    assert signal.name == "alpha_regime_transition"
    # the underlying posteriors are already shown leak-free in
    # test_regime_detection.py; here just check the shift(1) is applied
    # (first non-NaN value corresponds to the second available row).
    non_na = signal.dropna()
    assert len(non_na) == len(posteriors) - 1


def test_placebo_signal_has_near_zero_correlation_with_returns():
    raw = simulate_market(n_days=2000, seed=41)
    signal = placebo_signal(raw, seed=99)

    from scipy.stats import spearmanr

    forward_return = raw["return"].shift(-1)
    aligned = pd.concat([signal, forward_return], axis=1).dropna()
    corr, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    # under the null, std(IC) ~= 1/sqrt(n): a fixed-seed placebo should sit
    # well within a few standard errors of zero.
    assert abs(corr) < 0.05


def test_signals_have_expected_names_and_alignment():
    raw = simulate_market(n_days=400, seed=2)
    mr = mean_reversion_signal(raw)
    mom = momentum_signal(raw)
    vm = vol_managed_signal(raw)

    for signal in (mr, mom, vm):
        assert len(signal) == len(raw)
        assert signal.index.equals(raw.index)
