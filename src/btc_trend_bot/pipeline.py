from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from btc_trend_bot.analytics import extract_trades, market_capture, summarize_trades, yearly_performance
from btc_trend_bot.backtest import BacktestResult, run_backtest
from btc_trend_bot.config import load_config
from btc_trend_bot.data import load_ohlcv_csv, normalize_ohlcv, timeframe_to_timedelta
from btc_trend_bot.features import add_features
from btc_trend_bot.gaps import build_gap_report
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
    allow_data_gaps: bool = False,
) -> tuple[BacktestResult, dict]:
    cfg = load_config(config_path)
    timeframe = str(cfg["market"]["timeframe"])

    if frame is None:
        resolved_path = data_path or str(cfg["market"]["data_path"])
        normalized, report = load_ohlcv_csv(resolved_path, timeframe=timeframe)
    else:
        normalized, report = normalize_ohlcv(frame, timeframe=timeframe)

    gap_report = build_gap_report(normalized["timestamp"], timeframe=timeframe)
    quality = cfg.get("data_quality", {})
    requested_start = pd.Timestamp(str(cfg["market"]["start"]))
    if requested_start.tzinfo is None:
        requested_start = requested_start.tz_localize("UTC")
    else:
        requested_start = requested_start.tz_convert("UTC")
    step = timeframe_to_timedelta(timeframe)

    quality_failures: list[str] = []
    if (
        frame is None
        and bool(quality.get("require_configured_start", True))
        and report.first_timestamp > requested_start + step
    ):
        quality_failures.append(
            f"first candle is {report.first_timestamp.isoformat()}, after configured start "
            f"{requested_start.isoformat()}"
        )
    max_missing_rate = float(quality.get("max_missing_rate", 0.005))
    if gap_report.missing_rate > max_missing_rate:
        quality_failures.append(
            f"missing rate {gap_report.missing_rate:.2%} exceeds {max_missing_rate:.2%}"
        )
    max_gap = int(quality.get("max_single_gap_bars", 6))
    if gap_report.largest_gap_bars > max_gap:
        quality_failures.append(
            f"largest gap {gap_report.largest_gap_bars} bars exceeds {max_gap}"
        )

    if quality_failures and not allow_data_gaps:
        details = "; ".join(quality_failures)
        raise ValueError(
            "Data quality gate failed: " + details + ". "
            "Use a complete history source or pass --allow-data-gaps for diagnostics only."
        )
    if quality_failures:
        print("Warning: data quality override enabled: " + "; ".join(quality_failures))

    strategy_frame = prepare_strategy_frame(normalized, cfg)
    result = run_backtest(strategy_frame, cfg["backtest"])

    fixed_frame = strategy_frame.copy()
    fixed_frame["target_position"] = fixed_frame["fixed_target_position"]
    fixed_result = run_backtest(fixed_frame, cfg["backtest"])
    result.bars["fixed_trend_equity"] = fixed_result.bars["equity"]
    result.bars["fixed_trend_return"] = fixed_result.bars["strategy_return"]
    result.bars["fixed_trend_position"] = fixed_result.bars["held_position"]
    result.bars["fixed_trend_turnover"] = fixed_result.bars["turnover"]

    bars_per_year = int(cfg["backtest"]["bars_per_year"])
    metrics = summarize_backtest(result.bars, bars_per_year=bars_per_year)
    fixed_metrics = summarize_backtest(fixed_result.bars, bars_per_year=bars_per_year)["strategy"]
    metrics["fixed_size_trend"] = fixed_metrics

    bootstrap_kwargs = {
        "n_samples": int(cfg["backtest"]["bootstrap_samples"]),
        "block_bars": int(cfg["backtest"]["bootstrap_block_bars"]),
        "seed": int(cfg["backtest"]["random_seed"]),
    }
    metrics["block_bootstrap"] = moving_block_bootstrap_mean_ci(
        result.bars["strategy_return"], **bootstrap_kwargs
    )
    metrics["bootstraps"] = {
        "vol_scaled": moving_block_bootstrap_mean_ci(
            result.bars["strategy_return"], **bootstrap_kwargs
        ),
        "fixed_size": moving_block_bootstrap_mean_ci(
            result.bars["fixed_trend_return"], **bootstrap_kwargs
        ),
        "fixed_minus_buy_and_hold": moving_block_bootstrap_mean_ci(
            result.bars["fixed_trend_return"] - result.bars["benchmark_return"],
            **bootstrap_kwargs,
        ),
        "fixed_minus_vol_scaled": moving_block_bootstrap_mean_ci(
            result.bars["fixed_trend_return"] - result.bars["strategy_return"],
            **bootstrap_kwargs,
        ),
    }

    vol_trades = extract_trades(
        result.bars, "held_position", "strategy_return", "equity", "vol_scaled"
    )
    fixed_trades = extract_trades(
        result.bars,
        "fixed_trend_position",
        "fixed_trend_return",
        "fixed_trend_equity",
        "fixed_size",
    )
    metrics["trade_summary"] = {
        "vol_scaled": summarize_trades(vol_trades),
        "fixed_size": summarize_trades(fixed_trades),
    }
    metrics["market_capture"] = {
        "vol_scaled": market_capture(
            result.bars["strategy_return"], result.bars["benchmark_return"], bars_per_year
        ),
        "fixed_size": market_capture(
            result.bars["fixed_trend_return"], result.bars["benchmark_return"], bars_per_year
        ),
    }
    metrics["data"] = {
        "rows": report.rows,
        "missing_intervals": report.missing_intervals,
        "expected_intervals": gap_report.expected_bars,
        "missing_rate": gap_report.missing_rate,
        "largest_gap_bars": gap_report.largest_gap_bars,
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

    vol_trades = extract_trades(
        result.bars, "held_position", "strategy_return", "equity", "vol_scaled"
    )
    fixed_trades = extract_trades(
        result.bars,
        "fixed_trend_position",
        "fixed_trend_return",
        "fixed_trend_equity",
        "fixed_size",
    )
    pd.concat([vol_trades, fixed_trades], ignore_index=True).to_csv(
        directory / "trades.csv", index=False
    )
    yearly_performance(
        result.bars,
        {
            "vol_scaled": "strategy_return",
            "fixed_size": "fixed_trend_return",
            "buy_and_hold": "benchmark_return",
        },
        bars_per_year=2190,
    ).to_csv(directory / "yearly_performance.csv", index=False)

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

    print("\nBootstrap tests")
    for name, section in metrics.get("bootstraps", {"vol_scaled": metrics["block_bootstrap"]}).items():
        print(f"  {name}:")
        for key, value in section.items():
            print(f"    {key}: {value}")

    print("\nTrade summary")
    for name, section in metrics.get("trade_summary", {}).items():
        print(f"  {name}:")
        for key, value in section.items():
            print(f"    {key}: {value}")

    print("\nMarket capture")
    for name, section in metrics.get("market_capture", {}).items():
        print(f"  {name}:")
        for key, value in section.items():
            print(f"    {key}: {value}")
