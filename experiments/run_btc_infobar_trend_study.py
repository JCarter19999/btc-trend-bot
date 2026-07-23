"""BTC information-driven-bar trend study: can dollar-bar trend-following
beat the deployed simple_trend equity benchmark (243.3 bps/trade, ~24.3
bps/day over its 10-day fixed hold -- see EQUITY_EXIT_REGIME_SIMPLE_TREND.md
on the research branch / config/simple_trend_exit_regime_strategy.yaml on
main)?

This is a different question from the Tier-1 order-flow study
(BTC_ORDERFLOW_STUDY.md), which tested a sub-second flow-imbalance
*predictive* signal and found it real but ~76-100x too small for costs.
Here the hypothesis is narrower: does sampling bars by dollar turnover
instead of wall-clock time (so a bar always represents roughly the same
amount of "stuff happening," per Lopez de Prado) make a plain trend-
continuation strategy -- the exact "widen the candidate net, hold to a
fixed exit, don't manage it early" recipe that won for equities --
work better single-asset on BTC, which trades 24/7 with zero commission on
Binance spot (only spread+slippage: 12-16bps round trip, same cost floor
established in the latency study).

Honesty caveat, up front: only 11.4 days of tick data are downloaded (a
longer pull is just a bigger download, not a blocker, but wasn't run yet).
That is a far shorter window and far fewer independent regimes than the
equity study (2018-2026, bull and bear both). Whatever this finds is a
"worth extending the data pull for" signal, not a promotion-grade result on
its own -- treated with the same skepticism the equity pool got before
label-shuffling was replaced with a real random-selection control.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.orderflow_features import build_dollar_bars, load_trades
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position

SYMBOL = "BTCUSDT"
N_RANDOM_SEEDS = 50
ROUND_TRIP_COST_BPS_LOW = 12.0
ROUND_TRIP_COST_BPS_HIGH = 16.0
SIMPLE_TREND_BPS_PER_TRADE = 243.3
SIMPLE_TREND_HOLD_DAYS = 10.0
SIMPLE_TREND_BPS_PER_DAY = SIMPLE_TREND_BPS_PER_TRADE / SIMPLE_TREND_HOLD_DAYS


@dataclass
class Cfg:
    initial_capital: float = 2500.0
    minimum_equity: float = 25.0
    hard_shutdown_drawdown: float = 1.0  # single-asset toy sim; don't let this bug (see portfolio_sim docstring) truncate the run
    safety_enabled: bool = False
    drawdown_pause: float = 1.0
    cooldown_trades: int = 0
    consecutive_loss_limit: int = 10_000


SIZING = SizingMode("fixed_notional_2500", "fixed_notional", 2500.0)


def add_trend_features(bars: pd.DataFrame, ema_window: int, slope_lag: int) -> pd.DataFrame:
    bars = bars.copy()
    bars["ema"] = bars["close"].ewm(span=ema_window, adjust=False).mean()
    bars["ema_slope"] = bars["ema"] - bars["ema"].shift(slope_lag)
    bars["trend_up"] = (bars["close"] > bars["ema"]) & (bars["ema_slope"] > 0)
    return bars


def build_pool(bars: pd.DataFrame, hold_bars: int, safety_valve: int) -> pd.DataFrame:
    """Candidate = every bar where trend_up holds, entering at the next bar's
    open and exiting `hold_bars` bars later at close -- fixed hold, no early
    stop/target, mirroring exactly the exit mechanics that won for equities."""
    rows = []
    n = len(bars)
    for i in range(n - safety_valve - 1):
        if not bool(bars.iloc[i].trend_up):
            continue
        entry_i = i + 1
        exit_i = min(entry_i + hold_bars - 1, n - 1)
        entry_row, exit_row = bars.iloc[entry_i], bars.iloc[exit_i]
        gross_return = float(exit_row.close) / float(entry_row.open) - 1
        rows.append({
            "signal_time": bars.iloc[i].close_time, "symbol": SYMBOL,
            "entry_time": entry_row.open_time, "exit_time": exit_row.close_time,
            "gross_return": gross_return,
            "hold_hours": (exit_row.close_time - entry_row.open_time).total_seconds() / 3600.0,
        })
    return pd.DataFrame(rows)


def select_random_timing(bars: pd.DataFrame, n_signals: int, hold_bars: int, safety_valve: int, seed: int) -> pd.DataFrame:
    """Control: same hold duration & trade count, but entries fire at random
    bars instead of only when trend_up holds -- isolates whether the trend
    filter's *timing* adds anything over just being in the market that often."""
    n = len(bars)
    eligible = n - safety_valve - 1
    if eligible <= 0 or n_signals <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    idx = rng.choice(eligible, size=min(n_signals, eligible), replace=False)
    rows = []
    for i in idx:
        entry_i = i + 1
        exit_i = min(entry_i + hold_bars - 1, n - 1)
        entry_row, exit_row = bars.iloc[entry_i], bars.iloc[exit_i]
        gross_return = float(exit_row.close) / float(entry_row.open) - 1
        rows.append({
            "signal_time": bars.iloc[i].close_time, "symbol": SYMBOL,
            "entry_time": entry_row.open_time, "exit_time": exit_row.close_time,
            "gross_return": gross_return,
        })
    return pd.DataFrame(rows).sort_values("signal_time").reset_index(drop=True)


def evaluate(trades: pd.DataFrame, cost_bps: float, cfg: Cfg) -> dict:
    if trades.empty:
        return {"trades_taken": 0}
    trades = trades.copy()
    trades["net_return"] = trades["gross_return"] - cost_bps / 10000.0
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    trades["signal_time"] = pd.to_datetime(trades["signal_time"], utc=True)
    path, summary = simulate_single_position(trades, cfg, SIZING)
    taken = path[path.trade_taken] if len(path) else path
    if not len(taken):
        return summary
    merged = taken.merge(trades[["signal_time", "symbol", "entry_time", "exit_time"]],
                          on=["signal_time", "symbol"], how="left")
    hold_days = (pd.to_datetime(merged.exit_time, utc=True) - pd.to_datetime(merged.entry_time, utc=True)).dt.total_seconds() / 86400.0
    avg_hold_days = float(hold_days.mean()) if len(hold_days) else None
    returns = taken["trade_return"]
    sharpe_like = float(returns.mean() / returns.std()) if returns.std() > 0 else None
    summary["avg_hold_days"] = avg_hold_days
    summary["bps_per_day"] = (summary["expectancy_bps"] / avg_hold_days) if avg_hold_days else None
    summary["sharpe_like_per_trade"] = sharpe_like
    return summary


def main() -> None:
    trades_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/orderflow/btcusdt_trades_14d.parquet")
    print(f"Loading {trades_path} ...")
    trades = load_trades(trades_path)
    span_days = (trades.timestamp.max() - trades.timestamp.min()).total_seconds() / 86400
    print(f"{len(trades):,} trades, {span_days:.2f} days span")

    buy_hold_return = float(trades.iloc[-1].price) / float(trades.iloc[0].price) - 1
    print(f"Buy-and-hold over window: {buy_hold_return*100:.2f}% "
          f"({buy_hold_return*10000/span_days:.1f} bps/day) -- context, not a strategy result")

    cfg = Cfg()
    dollar_thresholds = [5_000_000, 10_000_000, 20_000_000]
    ema_windows = [20, 50, 100]
    hold_bars_options = [10, 20, 40, 80]
    safety_valve = max(hold_bars_options) + 5

    results = []
    for thresh in dollar_thresholds:
        bars = build_dollar_bars(trades, thresh)
        print(f"\n=== dollar_bar_threshold=${thresh:,} -> {len(bars)} bars "
              f"(median duration {(bars.close_time-bars.open_time).dt.total_seconds().median():.0f}s) ===")
        for ema_w in ema_windows:
            fbars = add_trend_features(bars, ema_w, slope_lag=max(3, ema_w // 5))
            for hold in hold_bars_options:
                pool = build_pool(fbars, hold, safety_valve)
                n_signals = len(pool)
                if n_signals < 5:
                    continue
                trend_summary = evaluate(pool, ROUND_TRIP_COST_BPS_HIGH, cfg)
                if trend_summary.get("trades_taken", 0) == 0:
                    continue
                random_summaries = [
                    evaluate(select_random_timing(fbars, n_signals, hold, safety_valve, seed), ROUND_TRIP_COST_BPS_HIGH, cfg)
                    for seed in range(N_RANDOM_SEEDS)
                ]
                random_bps = [s["expectancy_bps"] for s in random_summaries if s.get("trades_taken", 0) > 0]
                pct = (float((np.array(random_bps) < trend_summary["expectancy_bps"]).mean() * 100)
                       if random_bps else None)
                row = {
                    "dollar_threshold": thresh, "ema_window": ema_w, "hold_bars": hold,
                    "n_signals": n_signals, "trades_taken": trend_summary.get("trades_taken"),
                    "win_rate": trend_summary.get("win_rate"),
                    "expectancy_bps_net16": trend_summary.get("expectancy_bps"),
                    "avg_hold_days": trend_summary.get("avg_hold_days"),
                    "bps_per_day_net16": trend_summary.get("bps_per_day"),
                    "sharpe_like": trend_summary.get("sharpe_like_per_trade"),
                    "max_drawdown": trend_summary.get("max_drawdown"),
                    "profit_factor": trend_summary.get("profit_factor"),
                    "random_mean_bps": float(np.mean(random_bps)) if random_bps else None,
                    "random_std_bps": float(np.std(random_bps)) if random_bps else None,
                    "percentile_vs_random": pct,
                }
                results.append(row)

    res = pd.DataFrame(results).sort_values("bps_per_day_net16", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Top 15 configs by net bps/day (16bps round-trip cost) ===")
    print(res.head(15).to_string(index=False))

    print(f"\nBenchmark to beat: simple_trend live deployment = "
          f"{SIMPLE_TREND_BPS_PER_TRADE} bps/trade over a {SIMPLE_TREND_HOLD_DAYS:.0f}-day hold "
          f"= {SIMPLE_TREND_BPS_PER_DAY:.2f} bps/day.")
    beats = res[(res.bps_per_day_net16 > SIMPLE_TREND_BPS_PER_DAY) & (res.percentile_vs_random.fillna(0) >= 95) & (res.trades_taken >= 15)]
    print(f"\nConfigs beating that bps/day AND >=95th percentile vs random-timing control AND >=15 trades taken: {len(beats)}")
    if len(beats):
        print(beats.to_string(index=False))

    out = ROOT / "outputs" / "btc_infobar_trend_study"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "sweep_results.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "trades_span_days": span_days, "buy_hold_return": buy_hold_return,
        "n_trades": len(trades), "cost_bps_used": ROUND_TRIP_COST_BPS_HIGH,
    }, indent=2))
    print(f"\nFull sweep written to {out / 'sweep_results.csv'}")


if __name__ == "__main__":
    main()
