import pandas as pd
import pytest

from btc_trend_bot.data import drop_incomplete_last_bar, normalize_ohlcv


def frame():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC"),
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [10, 11, 12],
        }
    )


def test_normalize_reports_no_gaps():
    _, report = normalize_ohlcv(frame(), "4h")
    assert report.missing_intervals == 0
    assert report.rows == 3


def test_duplicate_timestamps_rejected():
    data = pd.concat([frame(), frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        normalize_ohlcv(data, "4h")


def test_gap_is_counted():
    data = frame().drop(index=1)
    _, report = normalize_ohlcv(data, "4h")
    assert report.missing_intervals == 1


def test_incomplete_last_bar_is_dropped():
    data = frame()
    now = pd.Timestamp("2024-01-01T09:00:00Z")
    trimmed = drop_incomplete_last_bar(data, "4h", now=now)
    assert len(trimmed) == 2
