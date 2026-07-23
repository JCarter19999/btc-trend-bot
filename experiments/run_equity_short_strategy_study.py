"""Short-selling research: mirror image of the long simple_trend strategy.

Not options/puts (that still needs real historical option-chain data --
see EQUITY_EXIT_REGIME_SIMPLE_TREND.md's options section). This shorts the
underlying stock directly, which is testable with the same real daily OHLCV
already in data/real/ -- no new data dependency.

Mechanics, mirrored from the long side:
  - Regime gate: market_above_ema50 < 1 (benchmark below its own 50-day
    trend) instead of >= 1 -- risk-off regime, mirroring simple_trend's
    risk-on gate.
  - Selection: lowest (most negative) relative_strength_20 instead of
    highest -- short the candidate underperforming the benchmark the most,
    instead of buying the one outperforming it the most.
  - Entry: next-bar open, slippage works against the short (worse fill both
    ways: sell short slightly below quoted open, buy to cover slightly
    above quoted price).
  - Stop/target flipped: stop = entry + stop_atr*atr (price rising hurts a
    short), target = entry - target_atr*atr (price falling profits a
    short).
  - Borrow cost: NOT real historical borrow-fee data (not available) -- a
    flat, deliberately conservative annualized rate prorated by hold
    duration, subtracted from net_return. Flagged as a modeling assumption
    throughout, same honesty standard as the options-overlay caveat.
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

from run_equity_real_data_walkforward import FEATURES, add_features, load_config, load_csv_dir

SIZING = SizingMode("fixed_notional_2500", "fixed_notional", 2500.0)
N_RANDOM_SEEDS = 50
BORROW_RATE_ANNUAL = 0.02  # conservative flat assumption -- see module docstring


def short_fixed_hold(frame, i, cfg, hold_bars):
    entry_i = i + 1
    if entry_i >= len(frame):
        return None
    entry = float(frame.iloc[entry_i].open) * (1 - cfg.slippage_bps_each_side / 10000)  # worse fill selling short
    atr = float(frame.iloc[i].atr)
    if not np.isfinite(atr) or atr <= 0:
        return None
    exit_i = min(entry_i + hold_bars - 1, len(frame) - 1)
    cover_price = float(frame.iloc[exit_i].close)
    cover_fill = cover_price * (1 + cfg.slippage_bps_each_side / 10000)  # worse fill buying to cover
    bars_held = exit_i - entry_i + 1
    gross_return = 1 - cover_fill / entry  # profit when price fell
    borrow_cost = BORROW_RATE_ANNUAL * (bars_held / 252)
    net_return = gross_return - borrow_cost
    return {"entry_time": frame.index[entry_i], "exit_time": frame.index[exit_i], "entry_price": entry,
            "exit_price": cover_fill, "net_return": net_return, "gross_return": gross_return,
            "borrow_cost": borrow_cost, "exit_reason": "time_exit", "bars_held": bars_held}


def short_atr_stop_target(frame, i, cfg, stop_atr, target_atr, max_hold_bars):
    entry_i = i + 1
    if entry_i >= len(frame):
        return None
    entry = float(frame.iloc[entry_i].open) * (1 - cfg.slippage_bps_each_side / 10000)
    atr = float(frame.iloc[i].atr)
    if not np.isfinite(atr) or atr <= 0:
        return None
    stop = entry + stop_atr * atr    # price rising hurts a short
    target = entry - target_atr * atr  # price falling profits a short
    exit_i = min(entry_i + max_hold_bars - 1, len(frame) - 1)
    cover_price = float(frame.iloc[exit_i].close)
    reason = "time_exit"
    for j in range(entry_i, exit_i + 1):
        row = frame.iloc[j]
        low, high = float(row.low), float(row.high)
        if high >= stop and low <= target:
            cover_price, exit_i, reason = stop, j, "ambiguous_stop_first"
            break
        if high >= stop:
            cover_price, exit_i, reason = stop, j, "stop"
            break
        if low <= target:
            cover_price, exit_i, reason = target, j, "target"
            break
    cover_fill = cover_price * (1 + cfg.slippage_bps_each_side / 10000)
    bars_held = exit_i - entry_i + 1
    gross_return = 1 - cover_fill / entry
    borrow_cost = BORROW_RATE_ANNUAL * (bars_held / 252)
    net_return = gross_return - borrow_cost
    return {"entry_time": frame.index[entry_i], "exit_time": frame.index[exit_i], "entry_price": entry,
            "exit_price": cover_fill, "net_return": net_return, "gross_return": gross_return,
            "borrow_cost": borrow_cost, "exit_reason": reason, "bars_held": bars_held}


def build_short_pool(frames, benchmark_symbol, cfg, exit_fn, regime_gated: bool, max_hold_cap: int):
    bench = frames[benchmark_symbol]
    rows = []
    for symbol in cfg.symbols:
        f = add_features(frames[symbol], bench)
        for i in range(60, len(f) - max_hold_cap - 1):
            r = f.iloc[i]
            # Same liquidity/sanity mask as the long side, minus the return_20
            # floor (a short candidate is EXPECTED to have negative recent
            # returns -- that floor would exclude the exact candidates we want).
            mask = (r.relative_volume > 0.6 and r.atr_pct > 0 and abs(r.ema_spread_atr) < 8)
            if not mask:
                continue
            if regime_gated and r.get("market_above_ema50", 1.0) >= 1:
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


def select_weakest_momentum(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pool
    return (
        pool.sort_values(["signal_time", "relative_strength_20"], ascending=[True, True])
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
    if len(taken):
        returns = taken["trade_return"]
        summary["sharpe_like_per_trade"] = float(returns.mean() / returns.std()) if returns.std() > 0 else None
    return summary


def main() -> None:
    cfg = load_config(ROOT / "configs/real_data.yaml")
    symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(symbols, ROOT / "data/real")
    max_hold_cap = 30

    print("=== Diagnostic: raw unselected short-candidate pool (no selection at all) ===")
    raw_pool = build_short_pool(frames, cfg.benchmark, cfg,
                                  lambda f, i, cfg: short_fixed_hold(f, i, cfg, 10),
                                  regime_gated=False, max_hold_cap=max_hold_cap)
    print(f"n={len(raw_pool)} win_rate={(raw_pool.net_return > 0).mean():.4f} "
          f"mean_net_return={raw_pool.net_return.mean()*100:.4f}% "
          f"mean_gross_return={raw_pool.gross_return.mean()*100:.4f}% "
          f"(borrow cost alone: {raw_pool.borrow_cost.mean()*100:.4f}%/trade)")
    print("Per symbol:")
    print(raw_pool.groupby("symbol").net_return.agg(count="count", mean="mean", win_rate=lambda s: (s > 0).mean()).round(4))

    results = {}
    families = {}
    for hold in (5, 10, 20):
        families[f"short_fixed_hold_{hold}_regime_gated"] = (
            lambda f, i, cfg, hb=hold: short_fixed_hold(f, i, cfg, hb), True)
        families[f"short_fixed_hold_{hold}_ungated"] = (
            lambda f, i, cfg, hb=hold: short_fixed_hold(f, i, cfg, hb), False)
    families["short_atr_stop2.0_target3.0_regime_gated"] = (
        lambda f, i, cfg: short_atr_stop_target(f, i, cfg, 2.0, 3.0, max_hold_cap), True)

    print("\n=== Selected (weakest-momentum) vs random control, per exit family ===")
    for name, (exit_fn, gated) in families.items():
        pool = build_short_pool(frames, cfg.benchmark, cfg, exit_fn, regime_gated=gated, max_hold_cap=max_hold_cap)
        if pool.empty:
            print(f"{name:42s} EMPTY POOL")
            continue
        pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True)
        pool["exit_time"] = pd.to_datetime(pool["exit_time"], utc=True)
        selected = select_weakest_momentum(pool)
        sel_summary = evaluate(selected, cfg)
        random_summaries = [evaluate(select_random(pool, seed), cfg) for seed in range(N_RANDOM_SEEDS)]
        random_bps = [s["expectancy_bps"] for s in random_summaries if s.get("trades_taken", 0) > 0]
        pct = float((np.array(random_bps) < sel_summary.get("expectancy_bps", -1e9)).mean() * 100) if random_bps else None
        results[name] = {"selected": sel_summary, "random_mean_bps": float(np.mean(random_bps)) if random_bps else None,
                          "random_std_bps": float(np.std(random_bps)) if random_bps else None,
                          "selected_percentile_in_random": pct, "pool_size": len(pool)}
        s = sel_summary
        print(f"{name:42s} pool={len(pool):4d} trades={s.get('trades_taken',0):4d} "
              f"win={s.get('win_rate',float('nan')):.3f} expectancy={s.get('expectancy_bps',float('nan')):7.1f}bps "
              f"PF={s.get('profit_factor') or float('nan'):.2f} maxDD={s.get('max_drawdown',float('nan')):.3f} "
              f"sharpe={s.get('sharpe_like_per_trade') or float('nan'):.3f} pctile_vs_random={pct}")

    out = ROOT / "outputs" / "equity_short_strategy_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nFull results written to", out / "summary.json")


if __name__ == "__main__":
    main()
