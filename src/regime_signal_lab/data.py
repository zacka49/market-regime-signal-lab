"""Data loading: synthetic (default, via simulate.py) and real market data
via yfinance, both exposed through the same date/price/return interface so
everything downstream (features/model/backtest/alphas) works unchanged
regardless of which one is used. Real data is cached locally so repeated
runs -- and anyone without network access -- don't need to hit the network
every time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .simulate import simulate_market

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"


def load_synthetic(n_days: int = 2500, seed: int = 7) -> pd.DataFrame:
    """Thin wrapper around simulate_market so callers can switch between
    synthetic and real data through one function without changing anything
    downstream."""
    return simulate_market(n_days=n_days, seed=seed)


def _download_ticker(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"no data returned for ticker {ticker!r}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance sometimes returns MultiIndex columns
        close = close.iloc[:, 0]

    return pd.DataFrame({"date": pd.to_datetime(raw.index), "price": close.to_numpy()})


def load_real(
    ticker: str, start: str = "2015-01-01", end: str | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Load daily close prices for `ticker`, cached to data/{ticker}.csv.
    Returns a DataFrame with date/price/return columns matching
    simulate_market's output, so it's a drop-in replacement everywhere
    downstream."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}.csv"

    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        frame = _download_ticker(ticker, start, end)
        frame.to_csv(cache_path, index=False)

    frame = frame.sort_values("date").reset_index(drop=True)
    frame["return"] = frame["price"].pct_change()
    return frame.dropna(subset=["return"]).reset_index(drop=True)


def load_real_panel(
    tickers: list[str], start: str = "2015-01-01", end: str | None = None, refresh: bool = False
) -> dict[str, pd.DataFrame]:
    return {ticker: load_real(ticker, start=start, end=end, refresh=refresh) for ticker in tickers}
