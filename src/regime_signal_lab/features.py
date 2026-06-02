from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create leak-free features for next-day return direction."""
    frame = data.copy()

    for lag in [1, 2, 3, 5, 10]:
        frame[f"return_lag_{lag}"] = frame["return"].shift(lag)

    for window in [5, 10, 20, 60]:
        frame[f"momentum_{window}"] = frame["price"].pct_change(window).shift(1)
        frame[f"volatility_{window}"] = frame["return"].rolling(window).std().shift(1)
        rolling_high = frame["price"].rolling(window).max().shift(1)
        frame[f"drawdown_{window}"] = frame["price"] / rolling_high - 1.0

    frame["target"] = (frame["return"].shift(-1) > 0).astype(int)
    frame["next_return"] = frame["return"].shift(-1)

    feature_columns = [column for column in frame.columns if column.startswith(("return_lag", "momentum", "volatility", "drawdown"))]
    clean = frame.dropna(subset=feature_columns + ["target", "next_return"]).reset_index(drop=True)
    clean[feature_columns] = clean[feature_columns].replace([np.inf, -np.inf], np.nan)
    return clean.dropna(subset=feature_columns).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(("return_lag", "momentum", "volatility", "drawdown"))]
