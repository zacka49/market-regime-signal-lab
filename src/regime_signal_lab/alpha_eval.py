"""Standard alpha evaluation report: information coefficient (IC), IC decay
across horizons, signal autocorrelation / implied turnover, decile forward
returns, and a quick after-cost Sharpe screen. Also signal combination
(orthogonalization, equal-weight / IC-weighted / ridge) and a correlation
matrix for judging redundancy between signals.

IC and decile/Sharpe diagnostics here are computed IN-SAMPLE over the full
history -- standard practice for characterizing a *candidate* signal's raw
statistical properties. This is NOT the same as a leak-free backtest: to
actually use a signal (or a combination) in the trading model, feed it as a
feature column into `model.walk_forward_validate`, which only ever fits on
each fold's training window. `combine_ic_weighted` and `combine_ridge` below
carry the same in-sample caveat explicitly in their docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


def information_coefficient(signal: pd.Series, forward_return: pd.Series) -> float:
    """Spearman rank correlation between a signal and the forward return it
    is meant to predict -- the standard 'IC' for a candidate alpha. Rank
    correlation (not Pearson) so the score is robust to the signal's exact
    scale and to outliers; only the ranking matters."""
    aligned = pd.concat([signal.rename("signal"), forward_return.rename("forward_return")], axis=1).dropna()
    if len(aligned) < 10:
        return float("nan")
    corr, _ = spearmanr(aligned["signal"], aligned["forward_return"])
    return float(corr)


def rolling_ic(signal: pd.Series, forward_return: pd.Series, window: int = 60) -> pd.Series:
    """IC computed on trailing windows, showing whether predictive power is
    stable through time or concentrated in a few episodes."""
    aligned = pd.concat([signal.rename("signal"), forward_return.rename("forward_return")], axis=1).dropna()
    sig = aligned["signal"].to_numpy()
    fwd = aligned["forward_return"].to_numpy()
    ic_values = np.full(len(aligned), np.nan)
    for i in range(window - 1, len(aligned)):
        corr, _ = spearmanr(sig[i - window + 1 : i + 1], fwd[i - window + 1 : i + 1])
        ic_values[i] = corr
    return pd.Series(ic_values, index=aligned.index, name="rolling_ic")


def ic_t_stat(ic_series: pd.Series) -> float:
    """t-stat for whether the mean IC is nonzero: mean(IC) / (std(IC)/sqrt(n)).

    CAVEAT: when `ic_series` comes from rolling_ic(), consecutive windows
    overlap heavily (a 60-day window shares 59 days with the next one), so
    treating each entry as an independent observation badly understates the
    true standard error. Empirically, under pure noise this formula reports
    |t| > 2 roughly 65% of the time instead of the nominal ~5%
    (tests/test_alpha_eval.py::test_rolling_ic_t_stat_is_anti_conservative_under_noise).
    Treat this as an optimistic, rough diagnostic only -- use
    non_overlapping_ic_t_stat for a much less misleading (if lower-power)
    number, and see docs/mathematical_notes.md Section 4 and Phase 5's
    block-bootstrap significance test for the rigorous version.
    """
    clean = ic_series.dropna()
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        return float("nan")
    return float(clean.mean() / (clean.std(ddof=1) / np.sqrt(len(clean))))


def non_overlapping_ic_t_stat(signal: pd.Series, forward_return: pd.Series, block_size: int = 60) -> float:
    """A much less misleading alternative to ic_t_stat(rolling_ic(...)):
    compute IC on NON-overlapping blocks so each block-IC is a nearly
    independent draw (rather than sharing block_size-1 days with its
    neighbor), at the cost of a far smaller effective sample size (and
    therefore lower power). Still an approximation -- the rigorous version
    is the block-bootstrap test in Phase 5 -- but meaningfully more honest
    than the naive rolling t-stat above."""
    aligned = pd.concat([signal.rename("signal"), forward_return.rename("forward_return")], axis=1).dropna()
    n_blocks = len(aligned) // block_size
    if n_blocks < 2:
        return float("nan")

    block_ics = []
    for i in range(n_blocks):
        block = aligned.iloc[i * block_size : (i + 1) * block_size]
        corr, _ = spearmanr(block["signal"], block["forward_return"])
        if np.isfinite(corr):
            block_ics.append(corr)

    block_ic_array = np.array(block_ics)
    if len(block_ic_array) < 2 or block_ic_array.std(ddof=1) == 0:
        return float("nan")
    return float(block_ic_array.mean() / (block_ic_array.std(ddof=1) / np.sqrt(len(block_ic_array))))


def ic_decay(signal: pd.Series, price: pd.Series, horizons: tuple[int, ...] = (1, 5, 10, 20)) -> pd.Series:
    """IC of the signal against forward returns at multiple horizons, to see
    how quickly (or slowly) predictive content decays."""
    results = {}
    for h in horizons:
        forward = price.pct_change(h).shift(-h)
        results[h] = information_coefficient(signal, forward)
    return pd.Series(results, name="ic_by_horizon")


def signal_autocorrelation(signal: pd.Series, lag: int = 1) -> float:
    return float(signal.dropna().autocorr(lag=lag))


def implied_turnover(signal: pd.Series) -> float:
    """Fraction of days a sign(signal)-based position flips. Higher
    autocorrelation in the signal implies lower turnover, and vice versa."""
    position = np.sign(signal.dropna())
    return float((position.diff().abs() > 0).mean())


def decile_returns(signal: pd.Series, forward_return: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    """Mean forward return by signal bin -- a genuinely informative signal
    should show forward returns that increase roughly monotonically from the
    lowest to the highest bin."""
    aligned = pd.concat([signal.rename("signal"), forward_return.rename("forward_return")], axis=1).dropna()
    aligned["bin"] = pd.qcut(aligned["signal"], n_bins, labels=False, duplicates="drop")
    summary = aligned.groupby("bin")["forward_return"].agg(["mean", "count"])
    summary.index = [f"bin_{i}" for i in summary.index]
    return summary


def signal_backtest_sharpe(
    signal: pd.Series, forward_return: pd.Series, cost_per_turnover: float = 0.0002
) -> float:
    """Quick sign(signal)-based long/short Sharpe screen, net of a flat
    per-turnover cost. A coarser tool than backtest.run_backtest (no
    probability threshold, no walk-forward split) -- useful for ranking
    candidate signals before deciding which are worth feeding into the model.
    """
    aligned = pd.concat([signal.rename("signal"), forward_return.rename("forward_return")], axis=1).dropna()
    position = np.sign(aligned["signal"])
    turnover = position.diff().abs().fillna(position.abs())
    strategy_return = position * aligned["forward_return"] - turnover * cost_per_turnover
    if strategy_return.std() == 0:
        return 0.0
    return float(np.sqrt(252) * strategy_return.mean() / strategy_return.std())


@dataclass(frozen=True)
class AlphaReport:
    name: str
    ic: float
    ic_t_stat: float
    non_overlapping_ic_t_stat: float
    ic_decay: pd.Series
    autocorr_lag1: float
    implied_turnover: float
    decile_returns: pd.DataFrame
    after_cost_sharpe: float


def build_alpha_report(
    name: str, signal: pd.Series, frame: pd.DataFrame, cost_per_turnover: float = 0.0002
) -> AlphaReport:
    forward_return = frame["return"].shift(-1)
    rolling = rolling_ic(signal, forward_return, window=60)
    return AlphaReport(
        name=name,
        ic=information_coefficient(signal, forward_return),
        ic_t_stat=ic_t_stat(rolling),
        non_overlapping_ic_t_stat=non_overlapping_ic_t_stat(signal, forward_return, block_size=60),
        ic_decay=ic_decay(signal, frame["price"]),
        autocorr_lag1=signal_autocorrelation(signal),
        implied_turnover=implied_turnover(signal),
        decile_returns=decile_returns(signal, forward_return),
        after_cost_sharpe=signal_backtest_sharpe(signal, forward_return, cost_per_turnover),
    )


def signal_correlation_matrix(signals: pd.DataFrame) -> pd.DataFrame:
    return signals.corr()


def orthogonalize(signals: pd.DataFrame) -> pd.DataFrame:
    """Sequential (Gram-Schmidt-style) orthogonalization: regress each
    signal, in column order, on all signals before it and keep the
    residual -- so a later signal only contributes information NOT already
    explained by earlier ones. Order-dependent by construction: pass columns
    in the order you want prioritized (e.g. most economically fundamental
    signal first)."""
    clean = signals.dropna()
    orthogonal = pd.DataFrame(index=clean.index)
    for i, col in enumerate(clean.columns):
        if i == 0:
            orthogonal[col] = clean[col]
            continue
        design = np.column_stack([np.ones(len(clean)), orthogonal.iloc[:, :i].to_numpy()])
        target = clean[col].to_numpy()
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        orthogonal[col] = target - design @ beta
    return orthogonal


def _zscore_columns(signals: pd.DataFrame) -> pd.DataFrame:
    return (signals - signals.mean()) / signals.std()


def combine_equal_weight(signals: pd.DataFrame) -> pd.Series:
    return _zscore_columns(signals).mean(axis=1).rename("combined_equal_weight")


def combine_ic_weighted(signals: pd.DataFrame, forward_return: pd.Series) -> pd.Series:
    """In-sample diagnostic combiner: weight each z-scored signal by its
    full-sample IC (sign and rough magnitude). NOT a leak-free production
    combination rule -- the weights are estimated on the whole sample. Use
    to see which signals a combiner would lean on; for an actual strategy,
    feed the raw signal columns into model.walk_forward_validate instead."""
    ics = pd.Series({col: information_coefficient(signals[col], forward_return) for col in signals.columns})
    weights = ics / ics.abs().sum()
    return (_zscore_columns(signals) * weights).sum(axis=1).rename("combined_ic_weighted")


def combine_ridge(signals: pd.DataFrame, forward_return: pd.Series, alpha: float = 1.0) -> pd.Series:
    """In-sample diagnostic combiner (same look-ahead caveat as
    combine_ic_weighted) via ridge regression of forward returns on the
    z-scored signals. Ridge rather than OLS because correlated signals
    (see signal_correlation_matrix) make OLS coefficients unstable."""
    aligned = pd.concat([signals, forward_return.rename("forward_return")], axis=1).dropna()
    x = _zscore_columns(aligned[signals.columns])
    y = aligned["forward_return"]
    model = Ridge(alpha=alpha)
    model.fit(x, y)
    combined = pd.Series(x.to_numpy() @ model.coef_, index=x.index, name="combined_ridge")
    return combined
