"""Systematic exit-strategy comparison, holding selection fixed at
simple_trend (the currently-deployed live strategy's selector -- see
config/simple_trend_exit_regime_strategy.yaml on main).

The trap this study is explicitly built to avoid: raw expectancy increases
almost monotonically with hold duration in this bull-trending basket (see
EQUITY_EXIT_REGIME_SIMPLE_TREND.md's hold-duration sensitivity check), so an
unconstrained sweep would just "discover" that holding forever is best. Every
candidate here is scored on duration-normalized and risk-adjusted terms
(bps captured per day held, a Sharpe-like ratio of trade returns, max
drawdown, profit factor) specifically so "hold longer" can't win by default,
and against a same-exit-rule random-selection control so a good score can't
be selection-method noise either.

Families compared:
  - fixed_hold: N in {5,10,15,20,30,45,60} (extends the earlier sensitivity
    check with duration-normalized scoring instead of raw bps)
  - atr_stop_target: the original mechanism at a few stop/target multiples,
    for reference (this is what's now disabled in the live deployment)
  - trailing_stop: highest-high-since-entry minus k*ATR, no fixed target,
    capped by a generous max hold (60 days) as a safety valve
  - momentum_reversal: exit when relative_strength_20 (the same signal
    simple_trend selects on) turns negative, after a minimum hold; capped
    by the same 60-day safety valve
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position

from run_equity_real_data_walkforward import (
    FEATURES,
    add_features,
    load_config,
    load_csv_dir,
)

SAFETY_VALVE_HOLD = 60  # generous cap so flexible exits can't run forever
N_RANDOM_SEEDS = 50
SIZING = SizingMode("fixed_notional_2500", "fixed_notional", 2500.0)


# --------------------------------------------------------------------------- #
# Exit-rule family implementations. Each takes (frame, i, cfg, **params) and
# returns the same trade dict shape as the base runner's simulate_trade.
# --------------------------------------------------------------------------- #

def exit_fixed_hold(frame, i, cfg, hold_bars):
    entry_i = i + 1
    if entry_i >= len(frame):
        return None
    entry = float(frame.iloc[entry_i].open) * (1 + cfg.slippage_bps_each_side / 10000)
    atr = float(frame.iloc[i].atr)
    if not np.isfinite(atr) or atr <= 0:
        return None
    exit_i = min(entry_i + hold_bars - 1, len(frame) - 1)
    exit_price = float(frame.iloc[exit_i].close)
    exit_fill = exit_price * (1 - cfg.slippage_bps_each_side / 10000)
    return {"entry_time": frame.index[entry_i], "exit_time": frame.index[exit_i], "entry_price": entry,
            "exit_price": exit_fill, "net_return": exit_fill / entry - 1, "exit_reason": "time_exit",
            "bars_held": exit_i - entry_i + 1}


def exit_atr_stop_target(frame, i, cfg, stop_atr, target_atr, max_hold_bars):
    entry_i = i + 1
    if entry_i >= len(frame):
        return None
    entry = float(frame.iloc[entry_i].open) * (1 + cfg.slippage_bps_each_side / 10000)
    atr = float(frame.iloc[i].atr)
    if not np.isfinite(atr) or atr <= 0:
        return None
    stop = entry - stop_atr * atr
    target = entry + target_atr * atr
    exit_i = min(entry_i + max_hold_bars - 1, len(frame) - 1)
    exit_price = float(frame.iloc[exit_i].close)
    reason = "time_exit"
    for j in range(entry_i, exit_i + 1):
        row = frame.iloc[j]
        low, high = float(row.low), float(row.high)
        if low <= stop and high >= target:
            exit_price, exit_i, reason = stop, j, "ambiguous_stop_first"
            break
        if low <= stop:
            exit_price, exit_i, reason = stop, j, "stop"
            break
        if high >= target:
            exit_price, exit_i, reason = target, j, "target"
            break
    exit_fill = exit_price * (1 - cfg.slippage_bps_each_side / 10000)
    return {"entry_time": frame.index[entry_i], "exit_time": frame.index[exit_i], "entry_price": entry,
            "exit_price": exit_fill, "net_return": exit_fill / entry - 1, "exit_reason": reason,
            "bars_held": exit_i - entry_i + 1}


def exit_trailing_stop(frame, i, cfg, trail_atr, max_hold_bars, min_hold_bars=2):
    entry_i = i + 1
    if entry_i >= len(frame):
        return None
    entry = float(frame.iloc[entry_i].open) * (1 + cfg.slippage_bps_each_side / 10000)
    atr = float(frame.iloc[i].atr)
    if not np.isfinite(atr) or atr <= 0:
        return None
    exit_i = min(entry_i + max_hold_bars - 1, len(frame) - 1)
    exit_price = float(frame.iloc[exit_i].close)
    reason = "time_exit"
    highest = entry
    for j in range(entry_i, exit_i + 1):
        row = frame.iloc[j]
        highest = max(highest, float(row.high))
        stop = highest - trail_atr * atr
        bars_held = j - entry_i + 1
        if bars_held >= min_hold_bars and float(row.low) <= stop:
            exit_price, exit_i, reason = max(stop, 0.0), j, "trailing_stop"
            break
    exit_fill = exit_price * (1 - cfg.slippage_bps_each_side / 10000)
    return {"entry_time": frame.index[entry_i], "exit_time": frame.index[exit_i], "entry_price": entry,
            "exit_price": exit_fill, "net_return": exit_fill / entry - 1, "exit_reason": reason,
            "bars_held": exit_i - entry_i + 1}


def exit_momentum_reversal(frame, i, cfg, max_hold_bars, min_hold_bars=3):
    entry_i = i + 1
    if entry_i >= len(frame):
        return None
    entry = float(frame.iloc[entry_i].open) * (1 + cfg.slippage_bps_each_side / 10000)
    atr = float(frame.iloc[i].atr)
    if not np.isfinite(atr) or atr <= 0:
        return None
    exit_i = min(entry_i + max_hold_bars - 1, len(frame) - 1)
    exit_price = float(frame.iloc[exit_i].close)
    reason = "time_exit"
    for j in range(entry_i, exit_i + 1):
        row = frame.iloc[j]
        bars_held = j - entry_i + 1
        rs = row.get("relative_strength_20", np.nan)
        if bars_held >= min_hold_bars and np.isfinite(rs) and rs < 0:
            exit_price, exit_i, reason = float(row.close), j, "momentum_reversal"
            break
    exit_fill = exit_price * (1 - cfg.slippage_bps_each_side / 10000)
    return {"entry_time": frame.index[entry_i], "exit_time": frame.index[exit_i], "entry_price": entry,
            "exit_price": exit_fill, "net_return": exit_fill / entry - 1, "exit_reason": reason,
            "bars_held": exit_i - entry_i + 1}


# --------------------------------------------------------------------------- #
# Pool construction: mask-filtered candidates (matching build_candidates'
# mask exactly), simple_trend selection, given exit_fn. Cap loop bound by
# SAFETY_VALVE_HOLD so flexible exits always have enough forward data.
# --------------------------------------------------------------------------- #

def build_pool(frames, benchmark_symbol, cfg, exit_fn):
    bench = frames[benchmark_symbol]
    rows = []
    for symbol in cfg.symbols:
        f = add_features(frames[symbol], bench)
        for i in range(60, len(f) - SAFETY_VALVE_HOLD - 1):
            r = f.iloc[i]
            mask = (r.relative_volume > 0.6 and r.atr_pct > 0 and abs(r.ema_spread_atr) < 8 and r.return_20 > -0.25)
            if not mask:
                continue
            trade = exit_fn(f, i, cfg)
            if trade is None:
                continue
            record = {"signal_time": f.index[i], "symbol": symbol, **trade}
            record.update({c: float(r[c]) for c in FEATURES if c in r and pd.notna(r[c])})
            if all(c in record and np.isfinite(record[c]) for c in FEATURES):
                rows.append(record)
    if not rows:
        return pd.DataFrame(columns=["signal_time", "symbol", *FEATURES, "net_return"])
    return pd.DataFrame(rows).sort_values(["signal_time", "symbol"]).reset_index(drop=True)


def select_simple_trend(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pool
    eligible = pool[pool.market_above_ema50 >= 1] if "market_above_ema50" in pool else pool
    if eligible.empty:
        eligible = pool
    return (
        eligible.sort_values(["signal_time", "relative_strength_20"], ascending=[True, False])
        .groupby("signal_time", as_index=False).head(1).sort_values("signal_time")
    )


def select_random(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    if pool.empty:
        return pool
    rng = np.random.default_rng(seed)
    return (
        pool.assign(_r=rng.random(len(pool)))
        .sort_values(["signal_time", "_r"])
        .groupby("signal_time", as_index=False).head(1)
        .drop(columns="_r").sort_values("signal_time")
    )


def evaluate(trades: pd.DataFrame, cfg) -> dict:
    if trades.empty:
        return {"trades_taken": 0}
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    path, summary = simulate_single_position(trades, cfg, SIZING)
    taken = path[path.trade_taken] if len(path) else path
    if not len(taken):
        return summary
    merged = taken.merge(trades[["signal_time", "symbol", "bars_held"]], on=["signal_time", "symbol"], how="left")
    avg_hold_bars = float(merged.bars_held.mean()) if len(merged) else None
    returns = taken["trade_return"]
    sharpe_like = float(returns.mean() / returns.std()) if returns.std() > 0 else None
    summary["avg_hold_bars"] = avg_hold_bars
    summary["bps_per_day"] = (summary["expectancy_bps"] / avg_hold_bars) if avg_hold_bars else None
    summary["sharpe_like_per_trade"] = sharpe_like
    return summary


def main() -> None:
    cfg = load_config(ROOT / "configs/real_data.yaml")
    symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(symbols, ROOT / "data/real")

    candidates_families = {}
    print("Building pools per exit family (this takes a bit)...")

    for hold in (5, 10, 15, 20, 30, 45, 60):
        candidates_families[f"fixed_hold_{hold}"] = build_pool(
            frames, cfg.benchmark, cfg, lambda f, i, cfg, hb=hold: exit_fixed_hold(f, i, cfg, hb))

    for stop, target in ((1.35, 2.15), (1.0, 2.0), (2.0, 2.0), (1.5, 4.0)):
        candidates_families[f"atr_stop{stop}_target{target}"] = build_pool(
            frames, cfg.benchmark, cfg,
            lambda f, i, cfg, s=stop, t=target: exit_atr_stop_target(f, i, cfg, s, t, 10))

    for trail in (1.5, 2.0, 2.5, 3.0, 4.0):
        candidates_families[f"trailing_stop_{trail}atr"] = build_pool(
            frames, cfg.benchmark, cfg,
            lambda f, i, cfg, tr=trail: exit_trailing_stop(f, i, cfg, tr, SAFETY_VALVE_HOLD))

    candidates_families["momentum_reversal"] = build_pool(
        frames, cfg.benchmark, cfg,
        lambda f, i, cfg: exit_momentum_reversal(f, i, cfg, SAFETY_VALVE_HOLD))

    results = {}
    for name, pool in candidates_families.items():
        pool = pool.copy()
        pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True)
        pool["exit_time"] = pd.to_datetime(pool["exit_time"], utc=True)
        st_trades = select_simple_trend(pool)
        st_summary = evaluate(st_trades, cfg)
        random_summaries = [evaluate(select_random(pool, seed), cfg) for seed in range(N_RANDOM_SEEDS)]
        random_bps = [s["expectancy_bps"] for s in random_summaries if s.get("trades_taken", 0) > 0]
        pct = float((np.array(random_bps) < st_summary.get("expectancy_bps", -1e9)).mean() * 100) if random_bps else None
        results[name] = {
            "simple_trend": st_summary,
            "random_mean_bps": float(np.mean(random_bps)) if random_bps else None,
            "random_std_bps": float(np.std(random_bps)) if random_bps else None,
            "simple_trend_percentile_in_random": pct,
        }
        s = st_summary
        print(f"{name:28s} trades={s.get('trades_taken',0):4d} win={s.get('win_rate',float('nan')):.3f} "
              f"expectancy={s.get('expectancy_bps',float('nan')):7.1f}bps "
              f"bps/day={s.get('bps_per_day') or float('nan'):6.1f} "
              f"sharpe={s.get('sharpe_like_per_trade') or float('nan'):.3f} "
              f"maxDD={s.get('max_drawdown',float('nan')):.3f} "
              f"PF={s.get('profit_factor') or float('nan'):.2f} "
              f"pctile_vs_random={pct}")

    out = ROOT / "outputs" / "equity_exit_strategy_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nFull results written to", out / "summary.json")


if __name__ == "__main__":
    main()
