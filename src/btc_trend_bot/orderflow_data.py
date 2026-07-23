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
    """Paginated historical trade download, checkpointed so a long pull can
    be safely interrupted and resumed (re-running picks up from the last
    saved trade's timestamp).

    Writes each checkpoint's NEW rows to its own small part file
    (`{stem}_parts/part_NNNNNN.parquet`) instead of repeatedly reading the
    ENTIRE accumulated dataset back into memory to concat and rewrite --
    that was the original design, and it OOM-killed a 60-day pull on this
    VM's 1.9GB RAM at ~8.6M rows / 190MB (confirmed via dmesg, 2026-07-23):
    every checkpoint reloaded the whole growing file, so memory pressure
    scaled with total downloaded size, not batch size, and grew every
    checkpoint for the life of the run. Resuming only reads the LAST part
    file (to find the resume timestamp), never the full history.

    Returns the parts directory. `orderflow_features.load_trades()` reads
    a directory of parquet files as one dataset natively (pyarrow), so
    downstream code doesn't need a separate consolidation step -- point it
    at the parts directory instead of a single file.
    """
    exchange = _get_binance()
    output_path = Path(output_path)
    parts_dir = output_path.parent / f"{output_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    start_ms = exchange.parse8601(start)
    end_ms = exchange.parse8601(end)

    since = start_ms
    existing_parts = sorted(parts_dir.glob("part_*.parquet"))
    total_existing = 0
    part_index = 0
    if existing_parts:
        last_part = pd.read_parquet(existing_parts[-1])
        if len(last_part):
            since = int(last_part["timestamp"].max()) + 1
        total_existing = sum(len(pd.read_parquet(p, columns=["id"])) for p in existing_parts)
        part_index = len(existing_parts)
        print(f"Resuming from checkpoint: {total_existing:,} trades already saved across "
              f"{len(existing_parts)} part files, continuing from {pd.Timestamp(since, unit='ms', tz='UTC')}")
    elif output_path.exists():
        # One-time migration from the old single-growing-file layout.
        print(f"Migrating legacy single-file checkpoint {output_path} to part-file layout...")
        legacy = pd.read_parquet(output_path)
        if len(legacy):
            legacy.to_parquet(parts_dir / "part_000000.parquet", index=False)
            since = int(legacy["timestamp"].max()) + 1
            total_existing = len(legacy)
            part_index = 1
        del legacy

    rows: list[dict] = []
    batch_count = 0
    total_new = 0

    while since < end_ms:
        batch = exchange.fetch_trades(symbol, since=since, limit=batch_limit)
        if not batch:
            break
        seen_in_batch: set = set()
        new_rows = []
        for t in batch:
            if t["timestamp"] >= end_ms or t["id"] in seen_in_batch:
                continue
            seen_in_batch.add(t["id"])
            new_rows.append({"id": t["id"], "timestamp": t["timestamp"], "datetime": t["datetime"],
                              "price": t["price"], "amount": t["amount"], "side": t["side"],
                              "cost": t.get("cost", t["price"] * t["amount"])})
        rows.extend(new_rows)
        total_new += len(new_rows)

        last_ts = batch[-1]["timestamp"]
        if last_ts <= since:
            break  # no progress; avoid infinite loop
        since = last_ts + 1
        batch_count += 1

        if batch_count % checkpoint_every == 0 and rows:
            part_path = parts_dir / f"part_{part_index:06d}.parquet"
            pd.DataFrame(rows).to_parquet(part_path, index=False)
            part_index += 1
            print(f"  checkpoint: +{len(rows):,} trades -> {part_path.name}, "
                  f"at {pd.Timestamp(since, unit='ms', tz='UTC')} ({total_existing + total_new:,} total)")
            rows = []

    if rows:
        part_path = parts_dir / f"part_{part_index:06d}.parquet"
        pd.DataFrame(rows).to_parquet(part_path, index=False)

    print(f"Done: {total_existing + total_new:,} trades saved across part files in {parts_dir} "
          f"({total_new:,} new this run)")
    return parts_dir


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
