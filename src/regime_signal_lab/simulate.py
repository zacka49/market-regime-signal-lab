from __future__ import annotations

import numpy as np
import pandas as pd

# Regime order: 0 = calm/low-drift, 1 = trending/bull, 2 = stressed/high-vol.
# Row i = transition probabilities out of regime i; each row is a categorical
# distribution and must sum to 1.
REGIME_TRANSITION_MATRIX = np.array(
    [
        [0.96, 0.03, 0.01],
        [0.05, 0.90, 0.05],
        [0.03, 0.07, 0.90],
    ]
)
REGIME_DRIFT = np.array([0.0002, 0.0008, -0.0005])
REGIME_VOLATILITY = np.array([0.006, 0.012, 0.020])


def simulate_market(n_days: int = 2500, seed: int = 7) -> pd.DataFrame:
    """Simulate returns with hidden trend and volatility regimes."""
    rng = np.random.default_rng(seed)

    regimes = np.zeros(n_days, dtype=int)
    for idx in range(1, n_days):
        regimes[idx] = rng.choice([0, 1, 2], p=REGIME_TRANSITION_MATRIX[regimes[idx - 1]])

    drift = np.choose(regimes, REGIME_DRIFT)
    volatility = np.choose(regimes, REGIME_VOLATILITY)

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
