"""Out-of-sample validation for the two-tier safety layer.

The prior sensitivity sweep (run_two_tier_safety_sensitivity.py) was run over
the FULL 2018-2026 history -- exactly the kind of "pick the best backtest
point" trap this project's own promotion-gate discipline warns against. This
script splits history into a TRAIN period (used only to pick a parameter
"plateau," never the single best point) and a TEST period the parameters never
see, then checks whether the chosen setting still does its job out of sample.

Split: train = 2018-01-01..2021-12-31 (includes the 2018 crash, NOT the 2022
crash), test = 2022-01-01..present (includes the 2022 bear + recovery + recent
data). This is deliberately the harder split: the plateau is chosen without
ever seeing the 2022 crash, then evaluated on exactly the period this whole
safety-layer idea was motivated by.

The strategy's own features (EMA/vol/Donchian) are rolling/causal and computed
once over the full frame before slicing -- no separate "fit" step exists for
them, so slicing after feature computation does not leak information; only the
safety-layer thresholds are chosen from train and frozen before touching test.

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

TRAIN_END = pd.Timestamp("2022-01-01", tz="UTC")  # exclusive
TRAIN_SWEEP = {
    "drawdown_pause": [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25],
    "consecutive_loss_limit_bars": [4, 6, 8, 10, 12, 16, 20],
    "cooldown_bars": [10, 20, 30, 45, 60, 90, 120],
}


def _row(label: str, param: str, value, metrics: dict, reasons: dict) -> dict:
    return {
        "label": label, "param": param, "value": value,
        "total_return": metrics["total_return"], "max_drawdown": metrics["max_drawdown"],
        "sharpe": metrics["sharpe"], "calmar": metrics["calmar"], "exposure": metrics["exposure"],
        "drawdown_pause_bars": reasons.get("drawdown_pause", 0),
        "loss_cooldown_bars": reasons.get("loss_cooldown", 0),
    }


def run_period(frame: pd.DataFrame, cfg: dict, safety: SafetyConfig | None, bars_per_year: int) -> tuple[dict, dict]:
    if safety is None:
        result = run_backtest(frame, cfg["backtest"])
    else:
        result = run_backtest_with_two_tier_safety(frame, cfg["backtest"], safety)
    summary = summarize_backtest(result.bars, bars_per_year)
    reasons = result.bars["safety_reason"].value_counts().to_dict() if "safety_reason" in result.bars else {}
    return summary, reasons


def main() -> None:
    cfg = load_config(str(ROOT / "config" / "settings_production.yaml"))
    normalized, _ = load_ohlcv_csv(cfg["market"]["data_path"], timeframe=cfg["market"]["timeframe"])
    full_frame = prepare_strategy_frame(normalized, cfg)
    bars_per_year = int(cfg["backtest"]["bars_per_year"])

    timestamps = pd.to_datetime(full_frame["timestamp"], utc=True)
    train_frame = full_frame.loc[timestamps < TRAIN_END].reset_index(drop=True)
    test_frame = full_frame.loc[timestamps >= TRAIN_END].reset_index(drop=True)
    print(f"train: {train_frame['timestamp'].iloc[0]} .. {train_frame['timestamp'].iloc[-1]} "
          f"({len(train_frame)} bars)")
    print(f"test:  {test_frame['timestamp'].iloc[0]} .. {test_frame['timestamp'].iloc[-1]} "
          f"({len(test_frame)} bars)")

    # --- Step 1: sensitivity sweep on TRAIN only, to pick a plateau -------- #
    default_safety = SafetyConfig()
    train_rows = []
    for param, values in TRAIN_SWEEP.items():
        for value in values:
            safety = replace(default_safety, **{param: value})
            summary, reasons = run_period(train_frame, cfg, safety, bars_per_year)
            train_rows.append(_row(f"{param}={value}", param, value, summary["strategy"], reasons))
    train_sweep_frame = pd.DataFrame(train_rows)

    out_dir = ROOT / "outputs" / "two_tier_safety_oos_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_sweep_frame.to_csv(out_dir / "train_sweep.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.width", 160,
                            "display.float_format", lambda v: f"{v:,.4f}"):
        print("\n=== TRAIN-only sensitivity sweep (2018-2021, no 2022 crash) ===")
        print(train_sweep_frame.to_string(index=False))

    # --- Step 2: pick a plateau config from TRAIN results only ------------ #
    # Chosen by inspection of the train-only sweep printed above, for a
    # *stable neighborhood* rather than the single best point:
    #   drawdown_pause=0.15: 0.125/0.15/0.2 all cluster at total_return
    #     2.87-3.23x, calmar 1.71-1.84 -- no sharp peak to chase, and it
    #     avoids the 0.10 landmine seen in the full-period sweep.
    #   consecutive_loss_limit_bars=8: 4 is a clear landmine (1.09x, strategy
    #     nearly paused into the ground); 10+ is inert (identical results,
    #     trigger stops firing); 8 sits in the narrow window where the trigger
    #     is live but not choking the strategy.
    #   cooldown_bars=30: 20/30/45/60 all cluster tightly (3.21-3.29x); 90/120
    #     start degrading (missed-recovery cost). 30 is the plateau's middle.
    # These three values are frozen BEFORE the test period below is touched.
    chosen_safety = SafetyConfig(
        drawdown_pause=0.15,
        consecutive_loss_limit_bars=8,
        cooldown_bars=30,
        hard_shutdown_drawdown=0.35,
        minimum_equity_fraction=0.01,
    )
    (out_dir / "chosen_safety_config.json").write_text(json.dumps(asdict(chosen_safety), indent=2))

    # --- Step 3: evaluate baseline / chosen-safety / buy-and-hold on BOTH - #
    results = {}
    for period_name, frame in (("train", train_frame), ("test", test_frame)):
        baseline_summary, _ = run_period(frame, cfg, None, bars_per_year)
        safety_summary, safety_reasons = run_period(frame, cfg, chosen_safety, bars_per_year)
        results[period_name] = {
            "baseline_no_breaker": baseline_summary["strategy"],
            "buy_and_hold_btc": baseline_summary["buy_and_hold"],
            "two_tier_safety_chosen": safety_summary["strategy"],
            "safety_reason_bar_counts": safety_reasons,
        }

    (out_dir / "oos_results.json").write_text(json.dumps(results, indent=2, default=str))

    print("\n=== Chosen plateau config (frozen before looking at test) ===")
    print(json.dumps(asdict(chosen_safety), indent=2))

    for period_name in ("train", "test"):
        print(f"\n=== {period_name.upper()} period ===")
        for label in ("buy_and_hold_btc", "baseline_no_breaker", "two_tier_safety_chosen"):
            m = results[period_name][label]
            print(f"  {label:26s} total_return={m['total_return']:+8.2%}  "
                  f"max_drawdown={m['max_drawdown']:8.2%}  sharpe={m['sharpe']:6.2f}  "
                  f"calmar={m['calmar']:6.2f}")
        print(f"  safety_reason_bar_counts: {results[period_name]['safety_reason_bar_counts']}")


if __name__ == "__main__":
    main()
