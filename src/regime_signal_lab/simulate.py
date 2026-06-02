from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_market(n_days: int = 2500, seed: int = 7) -> pd.DataFrame:
    """Simulate returns with hidden trend and volatility regimes."""
    rng = np.random.default_rng(seed)

    regimes = np.zeros(n_days, dtype=int)
    transition = np.array(
        [
            [0.96, 0.03, 0.01],
            [0.05, 0.90, 0.05],
            [0.03, 0.07, 0.90],
        ]
    )

    for idx in range(1, n_days):
        regimes[idx] = rng.choice([0, 1, 2], p=transition[regimes[idx - 1]])

    drift = np.choose(regimes, [0.0002, 0.0008, -0.0005])
    volatility = np.choose(regimes, [0.006, 0.012, 0.020])

    noise = rng.normal(0.0, volatility)
    autoregressive_component = np.zeros(n_days)
    for idx in range(1, n_days):
        autoregressive_component[idx] = 0.12 * autoregressive_component[idx - 1] + noise[idx]

    returns = drift + autoregressive_component
    price = 100.0 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=n_days, freq="B"),
            "price": price,
            "return": returns,
            "regime": regimes,
        }
    )
