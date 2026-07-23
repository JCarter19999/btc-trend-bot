from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from btc_trend_bot.data import merge_ohlcv_frames, timeframe_to_minutes
from btc_trend_bot.kraken_archive import (
    find_kraken_archive_member,
    load_kraken_ohlcvt_archive,
)


def archive_rows() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            0: [int(ts.timestamp()) for ts in timestamps],
            1: [100.0, 101.0, 102.0],
            2: [102.0, 103.0, 104.0],
            3: [99.0, 100.0, 101.0],
            4: [101.0, 102.0, 103.0],
            5: [10.0, 11.0, 12.0],
            6: [5, 6, 7],
        }
    )


def test_timeframe_to_minutes():
    assert timeframe_to_minutes("4h") == 240
    assert timeframe_to_minutes("1d") == 1440


def test_archive_member_matches_btc_alias():
    member = find_kraken_archive_member(
        ["nested/XBTUSD_60.csv", "nested/XBTUSD_240.csv"],
        symbol="BTC/USD",
        timeframe="4h",
    )
    assert member.endswith("XBTUSD_240.csv")


def test_archive_member_reports_available_intervals():
    with pytest.raises(FileNotFoundError, match="60"):
        find_kraken_archive_member(
            ["XBTUSD_60.csv"], symbol="BTC/USD", timeframe="4h"
        )


def test_load_kraken_zip_without_extracting(tmp_path):
    archive_path = tmp_path / "Kraken_OHLCVT.zip"
    payload = archive_rows().to_csv(index=False, header=False).encode("utf-8")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Kraken_OHLCVT/XBTUSD_240.csv", payload)
        archive.writestr("Kraken_OHLCVT/ETHUSD_240.csv", payload)

    frame, member = load_kraken_ohlcvt_archive(
        archive_path, symbol="BTC/USD", timeframe="4h"
    )

    assert member == "Kraken_OHLCVT/XBTUSD_240.csv"
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(frame) == 3
    assert frame["timestamp"].dt.tz is not None
    assert frame["close"].tolist() == [101.0, 102.0, 103.0]


def test_merge_prefers_later_frame_on_overlap():
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC"),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 11.0],
        }
    )
    recent = base.iloc[[1]].copy()
    recent.loc[:, "close"] = 102.5
    recent.loc[:, "high"] = 103.5

    merged, report = merge_ohlcv_frames([base, recent], timeframe="4h")

    assert report.rows == 2
    assert merged.loc[1, "close"] == 102.5
