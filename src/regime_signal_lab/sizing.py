"""Position sizing beyond binary +-1: the Kelly criterion and volatility
targeting. See docs/mathematical_notes.md Section 6.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(returns: np.ndarray) -> float:
    """Kelly-optimal leverage f* maximizing long-run geometric growth
    E[log(1 + f*r)], via the standard small-return approximation
    E[log(1+f*r)] ~= f*mu - 0.5*f^2*sigma^2 (second-order Taylor expansion),
    maximized at f* = mu / sigma^2. See docs/mathematical_notes.md Section
    6.1 for the derivation and for why using this at full strength on an
    ESTIMATED mu is reckless in practice."""
    returns = np.asarray(returns, dtype=float)
    variance = returns.var(ddof=1)
    if variance == 0:
        return 0.0
    return float(returns.mean() / variance)


def fractional_kelly_position(returns: np.ndarray, fraction: float = 0.5) -> float:
    """A fraction (e.g. 0.5 = half-Kelly) of the full Kelly leverage -- the
    standard practical mitigation for estimation error in mu (the mean
    return is far harder to estimate reliably than the variance, and full
    Kelly's leverage is linear in mu but only needs mu's *sign* to be wrong
    for full Kelly to be a bad bet)."""
    return fraction * kelly_fraction(returns)


def vol_target_position_scale(
    realized_vol: pd.Series, target_vol: float, max_leverage: float = 3.0
) -> pd.Series:
    """Per-row position-size multiplier: target_vol / realized_vol, capped
    at max_leverage and floored at 0 for undefined (NaN/zero) vol -- this
    only scales MAGNITUDE; direction still comes from the model's
    probability threshold, exactly as in backtest.run_backtest."""
    scale = target_vol / realized_vol.replace(0.0, np.nan)
    return scale.clip(upper=max_leverage).fillna(0.0)


def run_vol_targeted_backtest(
    predictions: pd.DataFrame,
    entry_threshold: float = 0.54,
    transaction_cost: float = 0.0002,
    target_vol: float = 0.01,
    vol_window: int = 20,
    max_leverage: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Same directional logic as backtest.run_backtest (long/short/flat via
    entry_threshold on `probability`), but the POSITION MAGNITUDE is scaled
    to target a constant realized volatility instead of always being
    exactly +-1: smaller bets after volatile periods, larger (up to
    max_leverage) after calm ones.

    Realized vol at row t is estimated from `next_return.shift(1)` -- since
    `next_return` at row t-1 equals the return actually realized on row t's
    date, shifting by 1 aligns each day's already-realized return to the
    row where it became known, so the rolling window only ever uses returns
    genuinely available before predicting row t's next_return (see
    tests/test_sizing.py for the corresponding no-lookahead check).
    """
    frame = predictions.copy()
    frame["direction"] = 0
    frame.loc[frame["probability"] >= entry_threshold, "direction"] = 1
    frame.loc[frame["probability"] <= 1.0 - entry_threshold, "direction"] = -1

    realized_vol = frame["next_return"].shift(1).rolling(vol_window).std()
    scale = vol_target_position_scale(realized_vol, target_vol, max_leverage)
    frame["position"] = frame["direction"] * scale

    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["strategy_return"] = frame["position"] * frame["next_return"] - frame["turnover"] * transaction_cost
    frame["equity"] = (1.0 + frame["strategy_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0

    daily_mean = frame["strategy_return"].mean()
    daily_std = frame["strategy_return"].std()
    sharpe = 0.0 if daily_std == 0 else np.sqrt(252) * daily_mean / daily_std

    metrics = {
        "total_return": float(frame["equity"].iloc[-1] - 1.0),
        "annualised_sharpe": float(sharpe),
        "max_drawdown": float(frame["drawdown"].min()),
        "mean_turnover": float(frame["turnover"].mean()),
        "active_days": float((frame["direction"] != 0).mean()),
        "mean_gross_leverage": float(frame["position"].abs().mean()),
    }
    return frame, metrics
