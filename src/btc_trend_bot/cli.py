from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from btc_trend_bot.coinbase_history import download_coinbase_history
from btc_trend_bot.config import load_config
from btc_trend_bot.data import (
    download_ohlcv,
    load_ohlcv_csv,
    merge_ohlcv_frames,
    save_ohlcv,
    timeframe_to_timedelta,
    update_ohlcv_file,
)
from btc_trend_bot.gaps import build_gap_report
from btc_trend_bot.kraken_archive import load_kraken_ohlcvt_archive
from btc_trend_bot.paper import run_paper_step
from btc_trend_bot.pipeline import print_metrics, run_research, save_outputs
from btc_trend_bot.synthetic import generate_synthetic_ohlcv


def _print_coverage(prefix: str, report) -> None:
    print(
        f"{prefix}: {report.rows:,} candles from "
        f"{report.first_timestamp.isoformat()} to {report.last_timestamp.isoformat()}"
    )
    print(f"Coverage: {report.missing_intervals:,} missing intervals; duplicates: {report.duplicate_timestamps}")


def command_download(config_path: str) -> None:
    """Download recent REST candles and merge them into the historical file."""
    cfg = load_config(config_path)
    market = cfg["market"]
    timeframe = str(market["timeframe"])
    data_path = Path(str(market["data_path"]))
    configured_start = str(market["start"])
    start = configured_start

    if data_path.exists():
        existing, existing_report = load_ohlcv_csv(data_path, timeframe=timeframe)
        # Include one overlap bar so revised latest values replace the old row.
        start = (existing_report.last_timestamp - timeframe_to_timedelta(timeframe)).isoformat()
        print(f"Updating existing history after {existing_report.last_timestamp.isoformat()}.")

    recent = download_ohlcv(
        exchange_id=str(market["exchange"]),
        symbol=str(market["symbol"]),
        timeframe=timeframe,
        start=start,
        max_bars=int(market["max_download_bars"]),
    )
    merged, report = update_ohlcv_file(data_path, recent, timeframe=timeframe)
    print(f"Downloaded {len(recent):,} recent completed candles.")
    _print_coverage(f"Saved merged data to {data_path}", report)

    requested = pd.Timestamp(configured_start)
    if requested.tzinfo is None:
        requested = requested.tz_localize("UTC")
    else:
        requested = requested.tz_convert("UTC")
    if not data_path.exists() or len(merged) <= len(recent):
        if recent["timestamp"].iloc[0] > requested + timeframe_to_timedelta(timeframe):
            print(
                "Warning: the REST endpoint did not return the configured historical start. "
                "Import Kraken_OHLCVT.zip with the import-kraken-history command for full history."
            )


def command_import_kraken_history(
    config_path: str,
    archive_path: str,
    pair: str | None,
    replace: bool,
) -> None:
    cfg = load_config(config_path)
    market = cfg["market"]
    timeframe = str(market["timeframe"])
    data_path = Path(str(market["data_path"]))

    archive_frame, member = load_kraken_ohlcvt_archive(
        archive_path=archive_path,
        symbol=str(market["symbol"]),
        timeframe=timeframe,
        pair_override=pair,
    )

    start = pd.Timestamp(str(market["start"]))
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    archive_frame = archive_frame.loc[archive_frame["timestamp"] >= start].reset_index(drop=True)
    if archive_frame.empty:
        raise ValueError(
            f"Archive member {member!r} contains no rows on or after configured start {start.isoformat()}."
        )

    frames = [archive_frame]
    if data_path.exists() and not replace:
        existing, _ = load_ohlcv_csv(data_path, timeframe=timeframe)
        # Existing data is later in the list, so recent REST rows win on overlap.
        frames.append(existing)
    merged, report = merge_ohlcv_frames(frames, timeframe=timeframe)
    save_ohlcv(merged, data_path)

    print(f"Imported {len(archive_frame):,} candles from archive member {member!r}.")
    _print_coverage(f"Saved merged history to {data_path}", report)
    if report.missing_intervals:
        print(
            "Note: Kraken archives omit intervals with no trades. Large or unexpected gaps "
            "should still be investigated before backtesting."
        )



def command_download_coinbase_history(
    config_path: str,
    output_path: str | None,
    product_id: str,
    start_override: str | None,
    end_override: str | None,
    pause_seconds: float,
) -> None:
    cfg = load_config(config_path)
    market = cfg["market"]
    timeframe = str(market["timeframe"])
    start = start_override or str(market["start"])
    output = Path(output_path or "data/btc_usd_4h_coinbase.csv")

    def progress(request_number: int, cursor: pd.Timestamp, window_end: pd.Timestamp) -> None:
        if request_number == 1 or request_number % 10 == 0:
            print(
                f"  Coinbase request {request_number}: "
                f"{cursor.date()} through {window_end.date()}"
            )

    frame, download_report = download_coinbase_history(
        start=start,
        end=end_override,
        product_id=product_id,
        timeframe=timeframe,
        pause_seconds=pause_seconds,
        progress=progress,
    )
    normalized, report = merge_ohlcv_frames([frame], timeframe=timeframe)
    save_ohlcv(normalized, output)
    print(
        f"Downloaded {download_report.hourly_rows:,} hourly candles in "
        f"{download_report.requests:,} requests and resampled to {len(normalized):,} {timeframe} candles."
    )
    _print_coverage(f"Saved Coinbase history to {output}", report)


def command_diagnose_gaps(config_path: str, data_path: str | None, limit: int) -> None:
    cfg = load_config(config_path)
    market = cfg["market"]
    timeframe = str(market["timeframe"])
    resolved = Path(data_path or str(market["data_path"]))
    frame, _ = load_ohlcv_csv(resolved, timeframe=timeframe)
    report = build_gap_report(frame["timestamp"], timeframe=timeframe)

    print(f"Observed bars: {report.observed_bars:,}")
    print(f"Expected bars: {report.expected_bars:,}")
    print(f"Missing bars:  {report.missing_bars:,}")
    print(f"Missing rate:  {report.missing_rate:.2%}")
    print(f"Largest gap:  {report.largest_gap_bars:,} bars")
    print(f"First bar:     {frame['timestamp'].iloc[0]}")
    print(f"Last bar:      {frame['timestamp'].iloc[-1]}")

    if report.runs.empty:
        print("\nNo missing intervals.")
        return
    print("\nLargest gap runs:")
    print(
        report.runs.sort_values("missing_bars", ascending=False)
        .head(limit)
        .to_string(index=False)
    )
    missing_series = pd.Series(report.missing_timestamps).dt.tz_localize(None)
    print("\nMissing bars by year:")
    print(missing_series.groupby(missing_series.dt.year).size().to_string())
    print("\nWorst months:")
    months = missing_series.groupby(missing_series.dt.to_period("M").astype(str)).size()
    print(months.sort_values(ascending=False).head(limit).to_string())

def command_data_status(config_path: str, data_path: str | None) -> None:
    cfg = load_config(config_path)
    market = cfg["market"]
    resolved = Path(data_path or str(market["data_path"]))
    _, report = load_ohlcv_csv(resolved, timeframe=str(market["timeframe"]))
    _print_coverage(str(resolved), report)


def command_backtest(config_path: str, data_path: str | None, allow_data_gaps: bool) -> None:
    result, metrics = run_research(
        config_path=config_path, data_path=data_path, allow_data_gaps=allow_data_gaps
    )
    save_outputs(result, metrics)
    print_metrics(metrics)
    print("\nSaved outputs under outputs/.")


def command_demo(config_path: str) -> None:
    frame = generate_synthetic_ohlcv()
    result, metrics = run_research(config_path=config_path, frame=frame)
    save_outputs(result, metrics)
    print_metrics(metrics)
    print("\nSynthetic demo complete; saved outputs under outputs/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC trend research and paper bot")
    parser.add_argument("--config", default="config/settings.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download recent public OHLCV and merge with local history")

    coinbase = subparsers.add_parser(
        "download-coinbase-history",
        help="Download paginated Coinbase BTC-USD hourly history and resample it",
    )
    coinbase.add_argument("--output", default=None, help="Output CSV; default data/btc_usd_4h_coinbase.csv")
    coinbase.add_argument("--product", default="BTC-USD")
    coinbase.add_argument("--start", default=None, help="Override configured start timestamp")
    coinbase.add_argument("--end", default=None, help="Optional exclusive end timestamp")
    coinbase.add_argument("--pause", type=float, default=0.35, help="Pause between API requests")

    importer = subparsers.add_parser(
        "import-kraken-history",
        help="Import Kraken's downloadable OHLCVT ZIP, CSV, or extracted directory",
    )
    importer.add_argument("--archive", required=True, help="Path to Kraken_OHLCVT.zip or extracted data")
    importer.add_argument("--pair", default=None, help="Optional archive pair override, e.g. XBTUSD")
    importer.add_argument(
        "--replace",
        action="store_true",
        help="Replace the local candle file instead of preserving existing REST rows",
    )

    status = subparsers.add_parser("data-status", help="Show local candle coverage and date range")
    status.add_argument("--data", default=None, help="Override market.data_path")

    gaps = subparsers.add_parser("diagnose-gaps", help="Show missing-candle runs")
    gaps.add_argument("--data", default=None, help="Override market.data_path")
    gaps.add_argument("--limit", type=int, default=25)

    backtest = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest.add_argument("--data", default=None, help="Override market.data_path")
    backtest.add_argument(
        "--allow-data-gaps",
        action="store_true",
        help="Bypass the data-quality gate for diagnostic runs only",
    )
    subparsers.add_parser("demo", help="Run against generated synthetic data")
    subparsers.add_parser("paper-step", help="Advance the local paper account once")

    args = parser.parse_args()
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("paper").mkdir(exist_ok=True)

    if args.command == "download":
        command_download(args.config)
    elif args.command == "download-coinbase-history":
        command_download_coinbase_history(
            args.config, args.output, args.product, args.start, args.end, args.pause
        )
    elif args.command == "import-kraken-history":
        command_import_kraken_history(args.config, args.archive, args.pair, args.replace)
    elif args.command == "data-status":
        command_data_status(args.config, args.data)
    elif args.command == "diagnose-gaps":
        command_diagnose_gaps(args.config, args.data, args.limit)
    elif args.command == "backtest":
        command_backtest(args.config, args.data, args.allow_data_gaps)
    elif args.command == "demo":
        command_demo(args.config)
    elif args.command == "paper-step":
        run_paper_step(args.config)


if __name__ == "__main__":
    main()
