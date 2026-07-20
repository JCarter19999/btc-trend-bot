from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from btc_trend_bot.data import (
    REQUIRED_COLUMNS,
    drop_incomplete_last_bar,
    timeframe_to_timedelta,
)

COINBASE_EXCHANGE_CANDLES_URL = (
    "https://api.exchange.coinbase.com/products/{product_id}/candles"
)
COINBASE_MAX_CANDLES = 300


@dataclass(frozen=True)
class CoinbaseDownloadReport:
    requests: int
    hourly_rows: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _iso_z(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_fetch_json(url: str, timeout_seconds: float) -> list[list[float]]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "btc-trend-bot/0.3.0",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - fixed HTTPS endpoint
        return json.loads(response.read().decode("utf-8"))


def fetch_coinbase_hourly(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    product_id: str = "BTC-USD",
    pause_seconds: float = 0.35,
    timeout_seconds: float = 30.0,
    max_retries: int = 4,
    fetch_json: Callable[[str, float], list[list[float]]] | None = None,
    progress: Callable[[int, pd.Timestamp, pd.Timestamp], None] | None = None,
) -> tuple[pd.DataFrame, CoinbaseDownloadReport]:
    """Download public Coinbase Exchange 1-hour candles in <=300-row windows.

    Coinbase's Exchange candles endpoint supports at most 300 candles per request.
    The function advances in non-overlapping 300-hour windows, filters any rows the
    endpoint returns outside the requested window, and de-duplicates timestamps.
    """
    start_ts = _utc_timestamp(start).floor("h")
    end_ts = _utc_timestamp(end).floor("h")
    if end_ts <= start_ts:
        raise ValueError("end must be later than start")
    if pause_seconds < 0:
        raise ValueError("pause_seconds cannot be negative")

    fetch = fetch_json or _default_fetch_json
    cursor = start_ts
    rows: list[list[float]] = []
    request_count = 0
    window = pd.to_timedelta(COINBASE_MAX_CANDLES, unit="h")

    while cursor < end_ts:
        window_end = min(cursor + window, end_ts)
        params = urlencode(
            {
                "start": _iso_z(cursor),
                "end": _iso_z(window_end),
                "granularity": 3600,
            }
        )
        url = COINBASE_EXCHANGE_CANDLES_URL.format(product_id=product_id) + "?" + params

        payload: list[list[float]] | None = None
        for attempt in range(max_retries + 1):
            try:
                payload = fetch(url, timeout_seconds)
                break
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= max_retries:
                    raise RuntimeError(
                        f"Coinbase candle request failed with HTTP {exc.code}: {url}"
                    ) from exc
            except URLError as exc:
                if attempt >= max_retries:
                    raise RuntimeError(f"Coinbase candle request failed: {exc}") from exc
            time.sleep(min(2**attempt, 8))

        request_count += 1
        if progress is not None:
            progress(request_count, cursor, window_end)
        if payload:
            for item in payload:
                if len(item) < 6:
                    raise ValueError(f"Unexpected Coinbase candle row: {item!r}")
                # Coinbase schema: [time, low, high, open, close, volume]
                rows.append([item[0], item[3], item[2], item[1], item[4], item[5]])

        cursor = window_end
        if pause_seconds and cursor < end_ts:
            time.sleep(pause_seconds)

    if not rows:
        raise RuntimeError("Coinbase returned no hourly candles for the requested range.")

    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = (
        frame.loc[(frame["timestamp"] >= start_ts) & (frame["timestamp"] < end_ts)]
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise RuntimeError("Coinbase rows did not overlap the requested range.")

    report = CoinbaseDownloadReport(
        requests=request_count,
        hourly_rows=len(frame),
        first_timestamp=frame["timestamp"].iloc[0],
        last_timestamp=frame["timestamp"].iloc[-1],
    )
    return frame, report


def resample_hourly_ohlcv(
    hourly: pd.DataFrame,
    timeframe: str = "4h",
    require_complete_source_bars: bool = True,
) -> pd.DataFrame:
    """Resample hourly OHLCV into a larger fixed interval aligned to UTC epoch."""
    if timeframe == "1h":
        return hourly[REQUIRED_COLUMNS].copy().reset_index(drop=True)
    step = timeframe_to_timedelta(timeframe)
    source_step = pd.to_timedelta(1, unit="h")
    ratio = step / source_step
    if not float(ratio).is_integer() or ratio < 1:
        raise ValueError(f"Timeframe must be a whole number of hours: {timeframe}")
    expected_count = int(ratio)

    frame = hourly[REQUIRED_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").set_index("timestamp")

    rule = timeframe.lower()
    grouped = frame.resample(rule, origin="epoch", label="left", closed="left")
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    counts = grouped["close"].count()
    out = out.loc[counts > 0]
    if require_complete_source_bars:
        out = out.loc[counts == expected_count]
    out = out.dropna().reset_index()
    return out[REQUIRED_COLUMNS]


def download_coinbase_history(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    product_id: str = "BTC-USD",
    timeframe: str = "4h",
    pause_seconds: float = 0.35,
    fetch_json: Callable[[str, float], list[list[float]]] | None = None,
    progress: Callable[[int, pd.Timestamp, pd.Timestamp], None] | None = None,
) -> tuple[pd.DataFrame, CoinbaseDownloadReport]:
    end_ts = (
        pd.Timestamp.now(tz="UTC").floor("h")
        if end is None
        else _utc_timestamp(end).floor("h")
    )
    hourly, report = fetch_coinbase_hourly(
        start=start,
        end=end_ts,
        product_id=product_id,
        pause_seconds=pause_seconds,
        fetch_json=fetch_json,
        progress=progress,
    )
    resampled = resample_hourly_ohlcv(hourly, timeframe=timeframe)
    resampled = drop_incomplete_last_bar(resampled, timeframe=timeframe, now=end_ts)
    if resampled.empty:
        raise RuntimeError("No complete resampled candles were produced.")
    return resampled.reset_index(drop=True), report
