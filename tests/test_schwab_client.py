from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from btc_trend_bot.schwab_client import fetch_schwab_ohlcv


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.text = "" if status_code == 200 else "error"
        self._payload = payload

    def json(self):
        return self._payload


class FakeSchwabClient:
    """In-process test double for SchwabClientProtocol. Returns synthetic candles
    built from a dict[str, DataFrame] keyed by symbol, or an error status for
    symbols listed in `fail_symbols`/`empty_symbols`."""

    def __init__(self, frames: dict[str, pd.DataFrame], fail_symbols=(), empty_symbols=()):
        self.frames = frames
        self.fail_symbols = set(fail_symbols)
        self.empty_symbols = set(empty_symbols)
        self.calls: list[dict] = []

    def get_price_history_every_day(self, symbol, *, start_datetime=None, end_datetime=None,
                                     need_extended_hours_data=None, need_previous_close=None):
        self.calls.append({"symbol": symbol, "start_datetime": start_datetime, "end_datetime": end_datetime})
        if symbol in self.fail_symbols:
            return _FakeResponse(500, {})
        if symbol in self.empty_symbols:
            return _FakeResponse(200, {"candles": [], "symbol": symbol, "empty": True})
        frame = self.frames[symbol]
        candles = [
            {"datetime": int(idx.timestamp() * 1000), "open": float(row.open), "high": float(row.high),
             "low": float(row.low), "close": float(row.close), "volume": float(row.volume)}
            for idx, row in frame.iterrows()
        ]
        return _FakeResponse(200, {"candles": candles, "symbol": symbol, "empty": False})


def _synthetic_frame(n=30, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    op = np.r_[close[0], close[:-1]]
    high = np.maximum(op, close) * 1.01
    low = np.minimum(op, close) * 0.99
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close,
                          "volume": rng.integers(1_000_000, 5_000_000, n)}, index=idx)


def _normalize_stub(frame, symbol):
    out = frame.copy()
    out["symbol"] = symbol
    return out.sort_index()


def test_fetch_schwab_ohlcv_matches_normalize_output_shape():
    frames = {"AAPL": _synthetic_frame(seed=1), "MSFT": _synthetic_frame(seed=2)}
    client = FakeSchwabClient(frames)
    result = fetch_schwab_ohlcv(client, ["AAPL", "MSFT"], lookback_days=60, normalize_fn=_normalize_stub)
    assert set(result) == {"AAPL", "MSFT"}
    for symbol, frame in result.items():
        assert list(frame.columns) == ["open", "high", "low", "close", "volume", "symbol"]
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert (frame["symbol"] == symbol).all()

def test_lookback_days_bounds_request_window():
    frames = {"AAPL": _synthetic_frame(seed=1)}
    client = FakeSchwabClient(frames)
    fetch_schwab_ohlcv(client, ["AAPL"], lookback_days=90, normalize_fn=_normalize_stub)
    call = client.calls[0]
    span = call["end_datetime"] - call["start_datetime"]
    assert 89 <= span.days <= 90

def test_non_200_response_raises_with_symbol_and_status():
    frames = {"AAPL": _synthetic_frame(seed=1)}
    client = FakeSchwabClient(frames, fail_symbols=["AAPL"])
    with pytest.raises(RuntimeError, match="AAPL.*500"):
        fetch_schwab_ohlcv(client, ["AAPL"], lookback_days=60, normalize_fn=_normalize_stub)

def test_empty_candles_response_raises_with_symbol():
    frames = {"AAPL": _synthetic_frame(seed=1)}
    client = FakeSchwabClient(frames, empty_symbols=["AAPL"])
    with pytest.raises(RuntimeError, match="AAPL"):
        fetch_schwab_ohlcv(client, ["AAPL"], lookback_days=60, normalize_fn=_normalize_stub)
