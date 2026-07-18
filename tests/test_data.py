from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_signal_lab.data import load_real, load_synthetic
from regime_signal_lab.simulate import simulate_market


def test_load_synthetic_matches_simulate_market_directly():
    from_data_module = load_synthetic(n_days=200, seed=3)
    from_simulate = simulate_market(n_days=200, seed=3)
    pd.testing.assert_frame_equal(from_data_module, from_simulate)


def _fake_yf_download(ticker, start, end, progress, auto_adjust):
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    prices = 100.0 * np.exp(np.cumsum(np.full(50, 0.0005)))
    # mimic yfinance's MultiIndex columns (Price, Ticker)
    columns = pd.MultiIndex.from_product([["Close", "High", "Low", "Open", "Volume"], [ticker]])
    data = np.column_stack([prices, prices, prices, prices, np.full(50, 1000.0)])
    return pd.DataFrame(data, index=dates, columns=columns)


def test_load_real_downloads_caches_and_computes_returns(tmp_path, monkeypatch):
    import regime_signal_lab.data as data_module

    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)

    import yfinance as yf

    monkeypatch.setattr(yf, "download", _fake_yf_download)

    frame = load_real("FAKE", start="2020-01-01")

    assert list(frame.columns) == ["date", "price", "return"]
    assert (tmp_path / "FAKE.csv").exists()
    assert len(frame) == 49  # first row dropped (pct_change NaN)
    np.testing.assert_allclose(frame["return"].to_numpy(), 0.0005, atol=1e-6)


def test_load_real_uses_cache_on_second_call_without_hitting_network(tmp_path, monkeypatch):
    import regime_signal_lab.data as data_module

    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)

    import yfinance as yf

    call_count = {"n": 0}

    def counting_download(ticker, start, end, progress, auto_adjust):
        call_count["n"] += 1
        return _fake_yf_download(ticker, start, end, progress, auto_adjust)

    monkeypatch.setattr(yf, "download", counting_download)

    first = load_real("FAKE2", start="2020-01-01")
    second = load_real("FAKE2", start="2020-01-01")

    assert call_count["n"] == 1  # second call served from cache
    pd.testing.assert_frame_equal(first, second)


def test_load_real_refresh_forces_a_new_download(tmp_path, monkeypatch):
    import regime_signal_lab.data as data_module

    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)

    import yfinance as yf

    call_count = {"n": 0}

    def counting_download(ticker, start, end, progress, auto_adjust):
        call_count["n"] += 1
        return _fake_yf_download(ticker, start, end, progress, auto_adjust)

    monkeypatch.setattr(yf, "download", counting_download)

    load_real("FAKE3", start="2020-01-01")
    load_real("FAKE3", start="2020-01-01", refresh=True)

    assert call_count["n"] == 2


def test_load_real_raises_on_empty_response(tmp_path, monkeypatch):
    import regime_signal_lab.data as data_module

    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)

    import yfinance as yf

    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())

    with pytest.raises(ValueError):
        load_real("EMPTY", start="2020-01-01")
