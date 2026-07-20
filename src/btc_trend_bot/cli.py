from __future__ import annotations

import argparse
from pathlib import Path

from btc_trend_bot.config import load_config
from btc_trend_bot.data import download_ohlcv, save_ohlcv
from btc_trend_bot.paper import run_paper_step
from btc_trend_bot.pipeline import print_metrics, run_research, save_outputs
from btc_trend_bot.synthetic import generate_synthetic_ohlcv


def command_download(config_path: str) -> None:
    cfg = load_config(config_path)
    market = cfg["market"]
    frame = download_ohlcv(
        exchange_id=str(market["exchange"]),
        symbol=str(market["symbol"]),
        timeframe=str(market["timeframe"]),
        start=str(market["start"]),
        max_bars=int(market["max_download_bars"]),
    )
    path = save_ohlcv(frame, str(market["data_path"]))
    print(f"Saved {len(frame):,} completed candles to {path}")


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

    subparsers.add_parser("download", help="Download public OHLCV through CCXT")
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
    elif args.command == "backtest":
        command_backtest(args.config, args.data)
    elif args.command == "demo":
        command_demo(args.config)
    elif args.command == "paper-step":
        run_paper_step(args.config)


if __name__ == "__main__":
    main()
