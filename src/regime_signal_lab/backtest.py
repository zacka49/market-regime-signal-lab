from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(predictions: pd.DataFrame, entry_threshold: float = 0.54, transaction_cost: float = 0.0002) -> tuple[pd.DataFrame, dict[str, float]]:
    """Convert predicted probabilities into a simple cost-aware strategy."""
    frame = predictions.copy()
    frame["position"] = 0
    frame.loc[frame["probability"] >= entry_threshold, "position"] = 1
    frame.loc[frame["probability"] <= 1.0 - entry_threshold, "position"] = -1

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
        "active_days": float((frame["position"] != 0).mean()),
    }
    return frame, metrics
