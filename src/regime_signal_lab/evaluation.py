"""Statistical rigor tools: baselines every metric should be compared
against, a stationary block bootstrap for confidence intervals and p-values
that respect serial dependence, and the deflated Sharpe ratio for honest
multiple-testing adjustment. See docs/mathematical_notes.md Section 5.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest import run_backtest


def buy_and_hold_baseline(
    predictions: pd.DataFrame, transaction_cost: float = 0.0002
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Baseline: hold a constant +1 position throughout (a single entry cost,
    then never trade again) -- economically identical to buy-and-hold."""
    always_long = predictions.copy()
    always_long["probability"] = 1.0
    return run_backtest(always_long, entry_threshold=0.5, transaction_cost=transaction_cost)


def random_signal_baseline(
    predictions: pd.DataFrame, transaction_cost: float = 0.0002, seed: int = 0
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Baseline: uniform-random probabilities pushed through the SAME
    threshold/position/cost machinery as the real strategy, so the
    comparison isolates "does the model know something" from "does the
    trading rule itself look good on paper"."""
    rng = np.random.default_rng(seed)
    random_predictions = predictions.copy()
    random_predictions["probability"] = rng.uniform(0.0, 1.0, size=len(predictions))
    return run_backtest(random_predictions, transaction_cost=transaction_cost)


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    returns = np.asarray(returns, dtype=float)
    std = returns.std()
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def stationary_bootstrap_resample(
    series: np.ndarray, mean_block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap: build a resampled
    series of the same length by stringing together blocks whose length is
    drawn from a Geometric distribution with mean `mean_block_length`
    (wrapping around the end of the series when a block runs past it).
    Unlike an iid bootstrap, this preserves local (short-range) serial
    dependence -- unlike a fixed-length block bootstrap, block boundaries
    aren't pinned to any particular calendar structure, avoiding edge
    artifacts from always cutting at the same spacing."""
    n = len(series)
    p = 1.0 / mean_block_length
    indices = np.empty(n, dtype=int)
    idx = int(rng.integers(0, n))
    for i in range(n):
        if i == 0 or rng.random() < p:
            idx = int(rng.integers(0, n))
        else:
            idx = (idx + 1) % n
        indices[i] = idx
    return series[indices]


def block_bootstrap_ci(
    returns: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 2000,
    mean_block_length: float = 20.0,
    ci: float = 0.90,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Stationary block bootstrap confidence interval for an arbitrary
    statistic of a return series (e.g. annualized_sharpe, total return).
    Respects the series' own serial dependence rather than assuming iid
    returns -- an iid bootstrap would understate the true uncertainty for
    autocorrelated returns, the same failure mode documented for the naive
    overlapping-window IC t-stat in alpha_eval.ic_t_stat."""
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    point_estimate = statistic_fn(returns)

    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resampled = stationary_bootstrap_resample(returns, mean_block_length, rng)
        samples[i] = statistic_fn(resampled)

    alpha = 1.0 - ci
    lower, upper = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(point_estimate), float(lower), float(upper)


def bootstrap_p_value(
    returns: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float] = lambda r: float(np.mean(r)),
    n_bootstrap: int = 2000,
    mean_block_length: float = 20.0,
    seed: int = 0,
) -> float:
    """Two-sided bootstrap p-value for the null statistic_fn(true process) == 0.
    Centers the series to impose the null (preserving higher moments and
    dependence structure), resamples via the stationary block bootstrap, and
    reports how often the null-centered bootstrap statistic is at least as
    extreme as the one actually observed."""
    returns = np.asarray(returns, dtype=float)
    observed = statistic_fn(returns)

    centered = returns - returns.mean()
    rng = np.random.default_rng(seed)
    null_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resampled = stationary_bootstrap_resample(centered, mean_block_length, rng)
        null_samples[i] = statistic_fn(resampled)

    return float(np.mean(np.abs(null_samples) >= np.abs(observed)))


def expected_max_sharpe_under_null(n_trials: int, sharpe_std_across_trials: float) -> float:
    """Expected maximum (non-annualized, per-period) Sharpe ratio across
    `n_trials` independent strategies/configurations if NONE of them had any
    true skill (Bailey & Lopez de Prado, 2014) -- the benchmark the deflated
    Sharpe ratio tests the observed Sharpe against, instead of testing
    against 0. `sharpe_std_across_trials` is the standard deviation of the
    per-period Sharpe ratios actually observed across those trials."""
    if n_trials <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return sharpe_std_across_trials * ((1 - euler_mascheroni) * z1 + euler_mascheroni * z2)


def deflated_sharpe_ratio(period_returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """Probability the strategy has genuine skill (true Sharpe > 0), after
    deflating for having selected the best of len(trial_sharpes) candidate
    configurations (Bailey & Lopez de Prado, 2014). `trial_sharpes` should
    be the (non-annualized, per-period) Sharpe ratios actually observed
    across every configuration considered during selection -- including the
    one ultimately reported -- so the multiple-testing penalty reflects the
    real search that was done, not a hypothetical one. Returns a probability
    in [0, 1]; e.g. 0.55 means "55% confidence of genuine skill even after
    accounting for how many things were tried."

    Formula (Bailey & Lopez de Prado, 2014): the Sharpe ratio estimator's
    standard error depends on the return distribution's skewness and
    kurtosis (Mertens, 2002 / Lo, 2002); deflation replaces the usual null
    of SR=0 with the expected max Sharpe achievable by pure luck across
    n_trials attempts.
    """
    returns = np.asarray(period_returns, dtype=float)
    n_obs = len(returns)
    sr = returns.mean() / returns.std(ddof=1)

    skew = float(pd.Series(returns).skew())
    # pandas' .kurtosis() is EXCESS kurtosis (Gaussian == 0); the formula
    # below uses kurtosis on the Gaussian == 3 convention.
    kurtosis = float(pd.Series(returns).kurtosis()) + 3.0

    se = np.sqrt(max(1 - skew * sr + (kurtosis - 1) / 4 * sr**2, 1e-12) / (n_obs - 1))

    n_trials = len(trial_sharpes)
    sharpe_std_across_trials = float(np.std(trial_sharpes, ddof=1)) if n_trials > 1 else se
    sr_benchmark = expected_max_sharpe_under_null(n_trials, sharpe_std_across_trials)

    z = (sr - sr_benchmark) / se
    return float(norm.cdf(z))
