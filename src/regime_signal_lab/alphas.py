"""A small library of candidate alpha signals, each with an explicit
economic hypothesis stated in its docstring.

Every signal is computed causally: the value at row t depends only on data
available through row t-1, matching the `.shift(1)` convention already used
in `features.py` (see `tests/test_no_lookahead.py` and
`tests/test_alphas.py` for the leakage checks). Signals are expressed as
trailing z-scores (`_rolling_zscore`) so they are on a comparable scale and
can be combined (`alpha_eval.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_zscore(series: pd.Series, window: int = 250, min_periods: int = 60) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0.0, np.nan)


def mean_reversion_signal(frame: pd.DataFrame, lookback: int = 5, zscore_window: int = 250) -> pd.Series:
    """Hypothesis: short-horizon moves partly revert toward a local mean
    (the OU dynamics of Section 1.3) -- the recent cumulative-return
    z-score, NEGATED so a large recent gain gives a negative (short) signal.
    """
    recent_return = frame["price"].pct_change(lookback)
    signal = -_rolling_zscore(recent_return, window=zscore_window)
    return signal.shift(1).rename("alpha_mean_reversion")


def momentum_signal(frame: pd.DataFrame, lookback: int = 60, zscore_window: int = 250) -> pd.Series:
    """Hypothesis: medium-horizon trends continue (time-series momentum) --
    the same construction as mean_reversion_signal but WITHOUT the sign
    flip, and over a longer lookback."""
    recent_return = frame["price"].pct_change(lookback)
    signal = _rolling_zscore(recent_return, window=zscore_window)
    return signal.shift(1).rename("alpha_momentum")


def vol_managed_signal(frame: pd.DataFrame, vol_window: int = 20, zscore_window: int = 250) -> pd.Series:
    """Hypothesis: low realized volatility relative to its own trailing
    history predicts better risk-adjusted forward returns (volatility
    clustering / volatility-managed portfolios, cf. Moreira & Muir 2017) --
    the signal is HIGH (long-biased) when vol is LOW relative to its
    trailing distribution, and negative when vol is elevated."""
    realized_vol = frame["return"].rolling(vol_window).std()
    signal = -_rolling_zscore(realized_vol, window=zscore_window)
    return signal.shift(1).rename("alpha_vol_managed")


def regime_transition_signal(regime_posteriors: pd.DataFrame) -> pd.Series:
    """Hypothesis: the HMM's posterior lean toward the highest- vs lowest-
    drift inferred regime carries directional information (Section 3).
    Takes the `regime_prob_i` columns from
    `regime_detection.walk_forward_regime_posteriors` (already leak-free;
    states are relabeled there by ascending fitted mean, so column 0 is the
    lowest-drift state and the last column the highest). Shifted by one
    further day to match the information lag of the other signals here.
    """
    prob_cols = sorted(c for c in regime_posteriors.columns if c.startswith("regime_prob_"))
    signal = regime_posteriors[prob_cols[-1]] - regime_posteriors[prob_cols[0]]
    return signal.shift(1).rename("alpha_regime_transition")


def placebo_signal(frame: pd.DataFrame, seed: int = 0) -> pd.Series:
    """No hypothesis -- pure iid noise, independent of the data. Included so
    alpha_eval.py's harness can be checked against a signal known to carry
    zero information (IC statistically indistinguishable from zero)."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(len(frame))
    return pd.Series(values, index=frame.index, name="alpha_placebo")
