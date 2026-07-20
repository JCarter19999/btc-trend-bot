from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from btc_trend_bot.backtest import BacktestResult, run_backtest
from btc_trend_bot.config import load_config
from btc_trend_bot.data import load_ohlcv_csv, normalize_ohlcv
from btc_trend_bot.features import add_features
from btc_trend_bot.metrics import moving_block_bootstrap_mean_ci, summarize_backtest
from btc_trend_bot.strategy import build_target_positions


def prepare_strategy_frame(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    bars_per_year = int(cfg["backtest"]["bars_per_year"])
    featured = add_features(frame, cfg["strategy"], bars_per_year=bars_per_year)
    return build_target_positions(featured, cfg["strategy"])


def run_research(
    config_path: str = "config/settings.yaml",
    data_path: str | None = None,
    frame: pd.DataFrame | None = None,
) -> tuple[BacktestResult, dict]:
    cfg = load_config(config_path)
    timeframe = str(cfg["market"]["timeframe"])

    if frame is None:
        resolved_path = data_path or str(cfg["market"]["data_path"])
        normalized, report = load_ohlcv_csv(resolved_path, timeframe=timeframe)
    else:
        normalized, report = normalize_ohlcv(frame, timeframe=timeframe)

    if report.missing_intervals:
        print(f"Warning: detected approximately {report.missing_intervals} missing {timeframe} intervals.")

    strategy_frame = prepare_strategy_frame(normalized, cfg)
    result = run_backtest(strategy_frame, cfg["backtest"])

    fixed_frame = strategy_frame.copy()
    fixed_max = float(cfg["strategy"].get("max_position", 1.0))
    fixed_frame["target_position"] = fixed_frame["direction"] * fixed_max
    fixed_result = run_backtest(fixed_frame, cfg["backtest"])
    result.bars["fixed_trend_equity"] = fixed_result.bars["equity"]
    result.bars["fixed_trend_return"] = fixed_result.bars["strategy_return"]

    bars_per_year = int(cfg["backtest"]["bars_per_year"])
    metrics = summarize_backtest(result.bars, bars_per_year=bars_per_year)
    fixed_metrics = summarize_backtest(fixed_result.bars, bars_per_year=bars_per_year)["strategy"]
    metrics["fixed_size_trend"] = fixed_metrics
    metrics["block_bootstrap"] = moving_block_bootstrap_mean_ci(
        result.bars["strategy_return"],
        n_samples=int(cfg["backtest"]["bootstrap_samples"]),
        block_bars=int(cfg["backtest"]["bootstrap_block_bars"]),
        seed=int(cfg["backtest"]["random_seed"]),
    )
    metrics["data"] = {
        "rows": report.rows,
        "missing_intervals": report.missing_intervals,
        "first_timestamp": report.first_timestamp.isoformat(),
        "last_timestamp": report.last_timestamp.isoformat(),
        "breaker_timestamp": result.breaker_timestamp.isoformat() if result.breaker_timestamp else None,
    }
    return result, metrics


def save_outputs(result: BacktestResult, metrics: dict, output_dir: str = "outputs") -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result.bars.to_csv(directory / "backtest_bars.csv", index=False)
    with (directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(result.bars["timestamp"], result.bars["equity"], label="Vol-scaled trend")
    if "fixed_trend_equity" in result.bars:
        ax.plot(result.bars["timestamp"], result.bars["fixed_trend_equity"], label="Fixed-size trend")
    ax.plot(result.bars["timestamp"], result.bars["benchmark_equity"], label="BTC buy-and-hold")
    ax.set_title("BTC Trend Strategy vs Buy-and-Hold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "equity_curve.png", dpi=150)
    plt.close(fig)


def print_metrics(metrics: dict) -> None:
    for name in ("strategy", "fixed_size_trend", "buy_and_hold"):
        section = metrics[name]
        print(f"\n{name.replace('_', ' ').title()}")
        for key in (
            "ending_equity",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "exposure",
            "total_turnover",
        ):
            print(f"  {key}: {section[key]:.6f}")
    print("\nBlock bootstrap")
    for key, value in metrics["block_bootstrap"].items():
        print(f"  {key}: {value}")
