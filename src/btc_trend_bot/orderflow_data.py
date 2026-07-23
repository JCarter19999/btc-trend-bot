"""Real historical order-flow data acquisition: trade tape, funding rate,
open interest -- Tier 1 of the "engines besides candlesticks" option space
(see EQUITY_... no, this is BTC research, not equity -- see
BTC_ORDERFLOW_STUDY.md). Not order-book depth (that needs live collection or
a paid vendor -- see that doc's scoping section); this is all free, real,
historical data pulled directly from Binance via CCXT.

Empirically confirmed limits (checked live against Binance, not assumed):
  - Trade tape: ~541,000 trades/day for BTC/USDT spot. Full-resolution tick
    history is available arbitrarily far back via paginated `fetch_trades`,
    but volume makes a long pull slow/large -- scope the window deliberately.
  - Funding rate history (BTC/USDT:USDT perpetual): available at least 2
    years back, small dataset (3 observations/day), no practical limit hit.
  - Open interest history: Binance's public endpoint hard-caps at ~29 days
    back (confirmed: 29d succeeds, 30d+ fails with "startTime is invalid").
    This bounds any OI-based backtest to a ~29-day recent window regardless
    of how much tick/funding history is pulled.

Downloads are checkpointed (resumable) and written incrementally to parquet
so an interrupted run doesn't lose progress and re-running is idempotent.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


def _get_binance():
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def download_trades_range(
    symbol: str,
    start: str,
    end: str,
    output_path: str | Path,
    batch_limit: int = 1000,
    checkpoint_every: int = 50,
) -> Path:
    """Paginated historical trade download, deduplicated by trade id,
    checkpointed to parquet every `checkpoint_every` batches so a long pull
    can be safely interrupted and resumed (re-running picks up from the
    last saved trade's timestamp)."""
    exchange = _get_binance()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_ms = exchange.parse8601(start)
    end_ms = exchange.parse8601(end)

    existing = pd.DataFrame()
    since = start_ms
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if len(existing):
            since = int(existing["timestamp"].max()) + 1
            print(f"Resuming from checkpoint: {len(existing):,} trades already saved, "
                  f"continuing from {pd.Timestamp(since, unit='ms', tz='UTC')}")

    rows: list[dict] = []
    seen_ids: set = set(existing["id"].tolist()) if len(existing) and "id" in existing.columns else set()
    batch_count = 0
    total_new = 0

    while since < end_ms:
        batch = exchange.fetch_trades(symbol, since=since, limit=batch_limit)
        if not batch:
            break
        new_rows = [
            {"id": t["id"], "timestamp": t["timestamp"], "datetime": t["datetime"],
             "price": t["price"], "amount": t["amount"], "side": t["side"],
             "cost": t.get("cost", t["price"] * t["amount"])}
            for t in batch if t["id"] not in seen_ids and t["timestamp"] < end_ms
        ]
        for r in new_rows:
            seen_ids.add(r["id"])
        rows.extend(new_rows)
        total_new += len(new_rows)

        last_ts = batch[-1]["timestamp"]
        if last_ts <= since:
            break  # no progress; avoid infinite loop
        since = last_ts + 1
        batch_count += 1

        if batch_count % checkpoint_every == 0:
            _flush(rows, existing, output_path)
            existing = pd.read_parquet(output_path)
            rows = []
            print(f"  checkpoint: {len(existing):,} total trades saved, "
                  f"at {pd.Timestamp(since, unit='ms', tz='UTC')}")

    if rows:
        _flush(rows, existing, output_path)

    final = pd.read_parquet(output_path)
    print(f"Done: {len(final):,} trades saved to {output_path} "
          f"({total_new:,} new this run)")
    return output_path


def _flush(new_rows: list[dict], existing: pd.DataFrame, path: Path) -> None:
    if not new_rows:
        return
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if len(existing) else pd.DataFrame(new_rows)
    combined = combined.drop_duplicates(subset="id").sort_values("timestamp").reset_index(drop=True)
    combined.to_parquet(path, index=False)


def download_funding_rate_history(symbol: str, start: str, output_path: str | Path) -> Path:
    exchange = _get_binance()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    since = exchange.parse8601(start)
    rows: list[dict] = []
    while True:
        batch = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not batch:
            break
        rows.extend({"timestamp": f["timestamp"], "datetime": f["datetime"], "funding_rate": f["fundingRate"]}
                     for f in batch)
        last_ts = batch[-1]["timestamp"]
        if last_ts <= since:
            break
        since = last_ts + 1
        if since >= exchange.milliseconds():
            break
    frame = pd.DataFrame(rows).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    frame.to_parquet(output_path, index=False)
    print(f"Funding rate: {len(frame):,} observations saved to {output_path}")
    return output_path


def download_open_interest_history(symbol: str, output_path: str | Path, days_back: int = 29) -> Path:
    """Binance's free OI history endpoint hard-caps at ~29 days back --
    confirmed empirically, not a chosen default. Requesting further back
    raises a BadRequest, not a silently-truncated result."""
    exchange = _get_binance()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    since = exchange.milliseconds() - days_back * 24 * 3600 * 1000
    rows: list[dict] = []
    while True:
        batch = exchange.fetch_open_interest_history(symbol, timeframe="1h", since=since, limit=500)
        if not batch:
            break
        rows.extend({"timestamp": o["timestamp"], "datetime": o["datetime"],
                     "open_interest_amount": o["openInterestAmount"], "open_interest_value": o["openInterestValue"]}
                    for o in batch)
        last_ts = batch[-1]["timestamp"]
        if last_ts <= since:
            break
        since = last_ts + 1
        if since >= exchange.milliseconds():
            break
    frame = pd.DataFrame(rows).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    frame.to_parquet(output_path, index=False)
    print(f"Open interest: {len(frame):,} observations saved to {output_path}")
    return output_path
