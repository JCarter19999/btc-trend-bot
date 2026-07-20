from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from btc_trend_bot.config import load_config
from btc_trend_bot.data import (
    download_ohlcv,
    load_ohlcv_csv,
    merge_ohlcv_frames,
    save_ohlcv,
    timeframe_to_timedelta,
    update_ohlcv_file,
)
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


def command_data_status(config_path: str, data_path: str | None) -> None:
    cfg = load_config(config_path)
    market = cfg["market"]
    resolved = Path(data_path or str(market["data_path"]))
    _, report = load_ohlcv_csv(resolved, timeframe=str(market["timeframe"]))
    _print_coverage(str(resolved), report)


def command_backtest(config_path: str, data_path: str | None) -> None:
    result, metrics = run_research(config_path=config_path, data_path=data_path)
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

    backtest = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest.add_argument("--data", default=None, help="Override market.data_path")
    subparsers.add_parser("demo", help="Run against generated synthetic data")
    subparsers.add_parser("paper-step", help="Advance the local paper account once")

    args = parser.parse_args()
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("paper").mkdir(exist_ok=True)

    if args.command == "download":
        command_download(args.config)
    elif args.command == "import-kraken-history":
        command_import_kraken_history(args.config, args.archive, args.pair, args.replace)
    elif args.command == "data-status":
        command_data_status(args.config, args.data)
    elif args.command == "backtest":
        command_backtest(args.config, args.data)
    elif args.command == "demo":
        command_demo(args.config)
    elif args.command == "paper-step":
        run_paper_step(args.config)


if __name__ == "__main__":
    main()
