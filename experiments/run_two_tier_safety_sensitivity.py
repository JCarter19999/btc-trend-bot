"""Parameter sensitivity sweep for the two-tier safety layer (see
run_two_tier_safety_backtest.py) -- one-at-a-time from the first-pass defaults,
holding every other knob fixed, matching this project's own "nearby settings"
promotion-gate habit rather than a full combinatorial grid.

Also reports the buy-and-hold BTC benchmark alongside both variants: neither
risk-control variant should be read in isolation from "what if you'd just
held."

Research-only. Does not modify backtest.py or any config.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from btc_trend_bot.backtest import run_backtest  # noqa: E402
from btc_trend_bot.config import load_config  # noqa: E402
from btc_trend_bot.data import load_ohlcv_csv  # noqa: E402
from btc_trend_bot.metrics import summarize_backtest  # noqa: E402
from btc_trend_bot.pipeline import prepare_strategy_frame  # noqa: E402
from run_two_tier_safety_backtest import SafetyConfig, run_backtest_with_two_tier_safety  # noqa: E402

SWEEP = {
    "drawdown_pause": [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25],
    "consecutive_loss_limit_bars": [4, 6, 8, 10, 12, 16, 20],
    "cooldown_bars": [10, 20, 30, 45, 60, 90, 120],
    "hard_shutdown_drawdown": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
}


def _metrics_row(label: str, param: str, value, metrics: dict, reason_counts: dict, hard_ts) -> dict:
    return {
        "param": param,
        "value": value,
        "label": label,
        "total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe": metrics["sharpe"],
        "calmar": metrics["calmar"],
        "exposure": metrics["exposure"],
        "drawdown_pause_bars": reason_counts.get("drawdown_pause", 0),
        "loss_cooldown_bars": reason_counts.get("loss_cooldown", 0),
        "hard_shutdown_ever": hard_ts is not None,
    }


def main() -> None:
    cfg = load_config(str(ROOT / "config" / "settings_production.yaml"))
    normalized, _ = load_ohlcv_csv(cfg["market"]["data_path"], timeframe=cfg["market"]["timeframe"])
    frame = prepare_strategy_frame(normalized, cfg)
    bars_per_year = int(cfg["backtest"]["bars_per_year"])

    baseline_result = run_backtest(frame, cfg["backtest"])
    baseline_summary = summarize_backtest(baseline_result.bars, bars_per_year)
    baseline_metrics = baseline_summary["strategy"]
    buy_and_hold_metrics = baseline_summary["buy_and_hold"]

    default_safety = SafetyConfig()
    rows = [_metrics_row("baseline_no_breaker", "-", None, baseline_metrics, {}, None)]
    rows.append({
        "param": "-", "value": None, "label": "buy_and_hold_btc",
        "total_return": buy_and_hold_metrics["total_return"],
        "max_drawdown": buy_and_hold_metrics["max_drawdown"],
        "sharpe": buy_and_hold_metrics["sharpe"],
        "calmar": buy_and_hold_metrics["calmar"],
        "exposure": buy_and_hold_metrics["exposure"],
        "drawdown_pause_bars": 0, "loss_cooldown_bars": 0, "hard_shutdown_ever": False,
    })

    default_result = run_backtest_with_two_tier_safety(frame, cfg["backtest"], default_safety)
    default_metrics = summarize_backtest(default_result.bars, bars_per_year)["strategy"]
    default_reasons = default_result.bars["safety_reason"].value_counts().to_dict()
    rows.append(_metrics_row("two_tier_safety_default", "-", None, default_metrics, default_reasons,
                              default_result.breaker_timestamp))

    for param, values in SWEEP.items():
        for value in values:
            safety = replace(default_safety, **{param: value})
            result = run_backtest_with_two_tier_safety(frame, cfg["backtest"], safety)
            metrics = summarize_backtest(result.bars, bars_per_year)["strategy"]
            reasons = result.bars["safety_reason"].value_counts().to_dict()
            rows.append(_metrics_row(f"{param}={value}", param, value, metrics, reasons,
                                      result.breaker_timestamp))

    sweep_frame = pd.DataFrame(rows)
    out_dir = ROOT / "outputs" / "two_tier_safety_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_frame.to_csv(out_dir / "sensitivity_results.csv", index=False)
    (out_dir / "default_safety_config.json").write_text(json.dumps(asdict(default_safety), indent=2))

    with pd.option_context("display.max_rows", None, "display.width", 160,
                            "display.float_format", lambda v: f"{v:,.4f}"):
        print(sweep_frame.to_string(index=False))


if __name__ == "__main__":
    main()
