from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class CoverageReport:
    rows: int
    duplicate_timestamps: int
    missing_intervals: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    units = {"m": "min", "h": "h", "d": "d"}
    if len(timeframe) < 2 or timeframe[-1] not in units:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    value = int(timeframe[:-1])
    if value <= 0:
        raise ValueError(f"Timeframe value must be positive: {timeframe}")
    return pd.to_timedelta(value, unit=units[timeframe[-1]])


def timeframe_to_minutes(timeframe: str) -> int:
    step = timeframe_to_timedelta(timeframe)
    minutes = step.total_seconds() / 60.0
    if not minutes.is_integer():
        raise ValueError(f"Timeframe does not resolve to whole minutes: {timeframe}")
    return int(minutes)


def normalize_ohlcv(frame: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, CoverageReport]:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {missing}")

    out = frame[REQUIRED_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for column in REQUIRED_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if out.isna().any().any():
        bad = out[out.isna().any(axis=1)].head(5)
        raise ValueError(f"OHLCV contains invalid or missing values. Examples:\n{bad}")

    out = out.sort_values("timestamp").reset_index(drop=True)
    duplicate_count = int(out["timestamp"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"OHLCV contains {duplicate_count} duplicate timestamps.")

    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be strictly positive.")
    if (out["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")
    if (out["high"] < out[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("At least one high is below another OHLC value.")
    if (out["low"] > out[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("At least one low is above another OHLC value.")

    step = timeframe_to_timedelta(timeframe)
    deltas = out["timestamp"].diff().dropna()
    missing_intervals = int(((deltas / step).round().astype(int) - 1).clip(lower=0).sum())
    report = CoverageReport(
        rows=len(out),
        duplicate_timestamps=duplicate_count,
        missing_intervals=missing_intervals,
        first_timestamp=out["timestamp"].iloc[0],
        last_timestamp=out["timestamp"].iloc[-1],
    )
    return out, report


def drop_incomplete_last_bar(frame: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    step = timeframe_to_timedelta(timeframe)
    last_open = pd.Timestamp(frame["timestamp"].iloc[-1])
    if last_open + step > now:
        return frame.iloc[:-1].copy()
    return frame.copy()


def load_ohlcv_csv(path: str | Path, timeframe: str) -> tuple[pd.DataFrame, CoverageReport]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {file_path}")
    return normalize_ohlcv(pd.read_csv(file_path), timeframe=timeframe)


def download_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    start: str,
    max_bars: int,
) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError as exc:
        raise RuntimeError("CCXT is not installed. Run: pip install -r requirements.txt") from exc

    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown CCXT exchange: {exchange_id}")

    exchange_class: Any = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    if not exchange.has.get("fetchOHLCV"):
        raise RuntimeError(f"{exchange_id} does not report fetchOHLCV support through CCXT.")

    exchange.load_markets()
    if symbol not in exchange.markets:
        examples = list(exchange.markets)[:10]
        raise ValueError(f"Symbol {symbol!r} not found on {exchange_id}. Examples: {examples}")

    since = exchange.parse8601(start)
    if since is None:
        raise ValueError(f"Could not parse market.start: {start}")

    rows: list[list[float]] = []
    last_seen: int | None = None
    while len(rows) < max_bars:
        limit = min(720, max_bars - len(rows))
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        newest = int(batch[-1][0])
        if last_seen is not None and newest <= last_seen:
            break
        last_seen = newest
        since = newest + 1
        step_ms = int(timeframe_to_timedelta(timeframe).total_seconds() * 1000)
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        if newest + step_ms >= now_ms:
            break

    if not rows:
        raise RuntimeError("No OHLCV rows were downloaded.")

    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    frame = drop_incomplete_last_bar(frame, timeframe=timeframe)
    return frame.reset_index(drop=True)


def save_ohlcv(frame: pd.DataFrame, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(file_path, index=False)
    return file_path


def merge_ohlcv_frames(
    frames: list[pd.DataFrame],
    timeframe: str,
) -> tuple[pd.DataFrame, CoverageReport]:
    """Merge OHLCV sources, preferring values from later frames on overlap."""
    non_empty = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        raise ValueError("At least one non-empty OHLCV frame is required.")
    combined = pd.concat(non_empty, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = (
        combined.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    return normalize_ohlcv(combined, timeframe=timeframe)


def update_ohlcv_file(
    existing_path: str | Path,
    recent_frame: pd.DataFrame,
    timeframe: str,
) -> tuple[pd.DataFrame, CoverageReport]:
    path = Path(existing_path)
    frames: list[pd.DataFrame] = []
    if path.exists():
        existing, _ = load_ohlcv_csv(path, timeframe=timeframe)
        frames.append(existing)
    frames.append(recent_frame)
    merged, report = merge_ohlcv_frames(frames, timeframe=timeframe)
    save_ohlcv(merged, path)
    return merged, report
