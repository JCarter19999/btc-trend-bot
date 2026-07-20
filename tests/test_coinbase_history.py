from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pandas as pd

from btc_trend_bot.coinbase_history import (
    fetch_coinbase_hourly,
    resample_hourly_ohlcv,
)


def fake_coinbase_fetch(url: str, timeout: float):
    del timeout
    query = parse_qs(urlparse(url).query)
    start = pd.Timestamp(query["start"][0])
    end = pd.Timestamp(query["end"][0])
    timestamps = pd.date_range(start, end, freq="1h", inclusive="left")
    rows = []
    for i, ts in enumerate(timestamps):
        close = 100.0 + i
        rows.append([int(ts.timestamp()), close - 2, close + 2, close - 1, close, 10.0])
    return list(reversed(rows))


def test_fetch_coinbase_hourly_paginates_at_300():
    frame, report = fetch_coinbase_hourly(
        start="2024-01-01T00:00:00Z",
        end="2024-01-13T13:00:00Z",  # 301 hours
        pause_seconds=0,
        fetch_json=fake_coinbase_fetch,
    )
    assert report.requests == 2
    assert len(frame) == 301
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["timestamp"].nunique() == 301


def test_resample_hourly_to_four_hours():
    hourly, _ = fetch_coinbase_hourly(
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T08:00:00Z",
        pause_seconds=0,
        fetch_json=fake_coinbase_fetch,
    )
    bars = resample_hourly_ohlcv(hourly, timeframe="4h")
    assert len(bars) == 2
    assert bars.iloc[0]["open"] == 99.0
    assert bars.iloc[0]["high"] == 105.0
    assert bars.iloc[0]["low"] == 98.0
    assert bars.iloc[0]["close"] == 103.0
    assert bars.iloc[0]["volume"] == 40.0


def test_resample_drops_incomplete_source_group():
    hourly, _ = fetch_coinbase_hourly(
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T08:00:00Z",
        pause_seconds=0,
        fetch_json=fake_coinbase_fetch,
    )
    hourly = hourly.drop(index=2).reset_index(drop=True)
    bars = resample_hourly_ohlcv(hourly, timeframe="4h")
    assert len(bars) == 1
    assert bars.iloc[0]["timestamp"] == pd.Timestamp("2024-01-01T04:00:00Z")
