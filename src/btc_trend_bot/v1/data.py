from pathlib import Path
import pandas as pd
from .binance import BinanceClient

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]

def klines_to_frame(rows: list[list[object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if frame.empty:
        return frame
    numeric = ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="raise").astype("int64")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    return frame.drop(columns=["ignore"]).sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

def validate_candles(frame: pd.DataFrame, interval: str = "5min") -> dict[str, int]:
    if frame.empty:
        return {"rows": 0, "duplicates": 0, "gaps": 0, "invalid_ohlc": 0}
    invalid = ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))).sum()
    return {"rows": len(frame), "duplicates": int(frame["open_time"].duplicated().sum()), "gaps": int((frame["open_time"].diff().dropna() != pd.Timedelta(interval)).sum()), "invalid_ohlc": int(invalid)}

def download_history(client: BinanceClient, symbol: str, interval: str, start: str, end: str | None, output: str | Path) -> Path:
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(end) if end else None
    if end_ts is not None and end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    rows = list(client.iter_klines(symbol, interval, int(start_ts.timestamp() * 1000), None if end_ts is None else int(end_ts.timestamp() * 1000)))
    frame = klines_to_frame(rows)
    report = validate_candles(frame)
    if report["duplicates"] or report["invalid_ohlc"]:
        raise ValueError(f"Invalid Binance candle data: {report}")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path

def resample_15m(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index("open_time")
    result = indexed.resample("15min", label="left", closed="left").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), quote_volume=("quote_volume", "sum"), trade_count=("trade_count", "sum"))
    return result.dropna().reset_index()
