"""Research-only comparison: does the equity project's two-tier capital-safety
layer improve on this project's frozen production risk control?

See equity_v2_4/experiments/run_equity_real_data_walkforward.py::simulate_capital
for the source design: a soft drawdown pause that recovers after a cooldown, a
separate consecutive-loss cooldown, a hard permanent shutdown, and a minimum-
equity floor. This project's frozen production config has only a single
all-or-nothing drawdown breaker (btc_trend_bot.backtest.run_backtest,
max_drawdown_breaker), and it is currently DISABLED (0.0) in
config/settings_production.yaml -- the README lists a breaker as a headline
feature, but production runs without one.

This script does not modify backtest.py or any config. It is a new hypothesis,
not a promotion: nothing here is wired into cli.py, production.py, or the
frozen production config.

Unit-adaptation notes (BTC's continuous per-bar vol-targeted exposure differs
from equity's discrete ATR-stop trade model, so parameters are re-derived, not
literally copied):
  - drawdown_pause / hard_shutdown_drawdown are percentages of equity -> ported
    unchanged (0.15 / 0.35, same as configs/real_data.yaml's safety block).
  - minimum_equity is a percentage-of-initial-capital floor in both -> ported
    as 1% of initial_cash (equity's $25 / $2,500 base).
  - consecutive_loss_limit / cooldown_trades count discrete TRADES in equity
    (each trade can span up to max_hold_bars=10 daily bars); this project has
    no discrete trade unit -- it rebalances continuous exposure every 4h bar.
    Re-denominated in BARS, at a scale keeping the same spirit (a short
    whipsaw-losing streak trips the loss-cooldown; a pause is a multi-day
    sit-out): 8 consecutive losing 4h bars (~1.3 days) trips a 60-bar
    (~10-day) cooldown. These are first-pass, not separately calibrated --
    see the printed report's caveats.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.backtest import BacktestResult, run_backtest  # noqa: E402
from btc_trend_bot.config import load_config  # noqa: E402
from btc_trend_bot.data import load_ohlcv_csv  # noqa: E402
from btc_trend_bot.metrics import summarize_backtest  # noqa: E402
from btc_trend_bot.pipeline import prepare_strategy_frame  # noqa: E402


@dataclass(frozen=True)
class SafetyConfig:
    drawdown_pause: float = 0.15
    hard_shutdown_drawdown: float = 0.35
    minimum_equity_fraction: float = 0.01
    consecutive_loss_limit_bars: int = 8
    cooldown_bars: int = 60


def run_backtest_with_two_tier_safety(frame: pd.DataFrame, cfg: dict, safety: SafetyConfig) -> BacktestResult:
    """Identical cost/turnover mechanics to btc_trend_bot.backtest.run_backtest;
    only the risk-control layer differs (see module docstring)."""
    required = {"timestamp", "close", "simple_return", "target_position"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Backtest input missing columns: {sorted(missing)}")

    initial_cash = float(cfg["initial_cash"])
    fee_rate = float(cfg["fee_bps_per_turnover"]) / 10_000.0
    slippage_rate = float(cfg["slippage_bps_per_turnover"]) / 10_000.0
    minimum_equity = initial_cash * safety.minimum_equity_fraction

    out = frame.copy().reset_index(drop=True)
    desired_held_position = out["target_position"].shift(1).fillna(0.0)

    equity = initial_cash
    peak = initial_cash
    previous_position = 0.0
    hard_halted = False
    hard_halt_timestamp: pd.Timestamp | None = None
    loss_cooldown_remaining = 0
    dd_cooldown_remaining = 0
    consecutive_losses = 0
    records: list[dict] = []

    for index, row in out.iterrows():
        # Position for this bar is decided from state carried over from the
        # PRIOR bar only -- no lookahead into this bar's own outcome.
        if hard_halted:
            position, reason = 0.0, "hard_shutdown"
        elif loss_cooldown_remaining > 0:
            position, reason = 0.0, "loss_cooldown"
            loss_cooldown_remaining -= 1
        elif dd_cooldown_remaining > 0:
            position, reason = 0.0, "drawdown_pause"
            dd_cooldown_remaining -= 1
            if dd_cooldown_remaining == 0:
                peak = equity  # re-anchor so the pause doesn't instantly re-trip
        else:
            position, reason = float(desired_held_position.iloc[index]), ""

        asset_return = 0.0 if pd.isna(row["simple_return"]) else float(row["simple_return"])
        turnover = abs(position - previous_position)
        execution_cost = turnover * (fee_rate + slippage_rate)
        funding_rate = float(row.get("funding_rate", 0.0) or 0.0)
        funding_cost = position * funding_rate
        gross_return = position * asset_return
        net_return = gross_return - execution_cost - funding_cost
        if net_return <= -1.0:
            raise RuntimeError("A bar produced a return <= -100%; inspect data and leverage assumptions.")

        equity *= 1.0 + net_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0

        if net_return < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        # Trigger checks for the NEXT bar, evaluated after this bar's outcome.
        if not hard_halted:
            if equity < minimum_equity or drawdown <= -safety.hard_shutdown_drawdown:
                hard_halted = True
                hard_halt_timestamp = pd.Timestamp(row["timestamp"])
            elif loss_cooldown_remaining == 0 and consecutive_losses >= safety.consecutive_loss_limit_bars:
                loss_cooldown_remaining = safety.cooldown_bars
                consecutive_losses = 0
            elif dd_cooldown_remaining == 0 and drawdown <= -safety.drawdown_pause:
                dd_cooldown_remaining = safety.cooldown_bars

        records.append({
            "held_position": position,
            "turnover": turnover,
            "gross_strategy_return": gross_return,
            "execution_cost_return": execution_cost,
            "funding_cost_return": funding_cost,
            "strategy_return": net_return,
            "equity": equity,
            "drawdown": drawdown,
            "breaker_active": hard_halted,
            "safety_reason": reason,
        })
        previous_position = position

    record_frame = pd.DataFrame(records)
    out = pd.concat([out, record_frame], axis=1)
    out["benchmark_return"] = out["simple_return"].fillna(0.0)
    out["benchmark_equity"] = initial_cash * (1.0 + out["benchmark_return"]).cumprod()
    return BacktestResult(bars=out, breaker_timestamp=hard_halt_timestamp)


def main() -> None:
    config_path = str(ROOT / "config" / "settings_production.yaml")
    cfg = load_config(config_path)
    normalized, _ = load_ohlcv_csv(cfg["market"]["data_path"], timeframe=cfg["market"]["timeframe"])
    strategy_frame = prepare_strategy_frame(normalized, cfg)
    bars_per_year = int(cfg["backtest"]["bars_per_year"])

    baseline_result = run_backtest(strategy_frame, cfg["backtest"])
    baseline_metrics = summarize_backtest(baseline_result.bars, bars_per_year)["strategy"]

    safety = SafetyConfig()
    safety_result = run_backtest_with_two_tier_safety(strategy_frame, cfg["backtest"], safety)
    safety_metrics = summarize_backtest(safety_result.bars, bars_per_year)["strategy"]

    yearly = []
    for year, group in safety_result.bars.assign(
            year=pd.to_datetime(safety_result.bars["timestamp"]).dt.year).groupby("year"):
        baseline_year = baseline_result.bars.loc[group.index]
        yearly.append({
            "year": int(year),
            "baseline_return": float(baseline_year["equity"].iloc[-1] / baseline_year["equity"].iloc[0] - 1),
            "safety_return": float(group["equity"].iloc[-1] / group["equity"].iloc[0] - 1),
            "safety_max_drawdown": float(group["drawdown"].min()),
            "baseline_max_drawdown": float(baseline_year["drawdown"].min()),
        })

    report = {
        "safety_config": asdict(safety),
        "baseline_no_breaker": baseline_metrics,
        "two_tier_safety": safety_metrics,
        "safety_reason_bar_counts": safety_result.bars["safety_reason"].value_counts().to_dict(),
        "hard_shutdown_timestamp": str(safety_result.breaker_timestamp) if safety_result.breaker_timestamp else None,
        "yearly_comparison": yearly,
    }
    out_dir = ROOT / "outputs" / "two_tier_safety_backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=str))
    baseline_result.bars.to_csv(out_dir / "baseline_bars.csv", index=False)
    safety_result.bars.to_csv(out_dir / "safety_bars.csv", index=False)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
