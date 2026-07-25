"""Crash-overlay test for simple_trend: actively exit an open position when
the market-wide trend classifier flips to "not trending" (crash proxy) and
re-enter once it recovers, instead of letting positions run to their natural
10-day exit regardless of conditions (the current live behavior). Different
question from the earlier regime-rotation backtest, which only gated NEW
entries and never force-closed an open position.

Crash signal: regime_classifier.compute_trend_flag_market_wide (SPY,
return_26 > -1% and ema_slope_atr > -0.20) -- already-validated market-wide
trend proxy. FALSE = crash state, entered only on that state's day close and
acted on next-bar-open (no lookahead), same convention as every trade entry
elsewhere in this project.

Selection on re-entry: same simple_trend rule (top relative_strength_20 among
symbols with market_above_ema50 >= 1) via _select_fold_winners.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_equity_real_data_walkforward import (
    load_config, load_csv_dir, build_candidates, add_features, _select_fold_winners,
)
from btc_trend_bot.regime_classifier import compute_trend_flag_market_wide

SLIPPAGE = 2.0 / 10000
INITIAL_CAPITAL = 2500.0
MAX_HOLD = 10


def load_everything():
    cfg = load_config(ROOT / "configs" / "real_data.yaml")
    cfg = replace(cfg, stop_atr=100.0, target_atr=100.0)  # matches live deployed convention
    all_symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(all_symbols, ROOT / "data" / "real")
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    winners = _select_fold_winners(candidates, candidates, cfg, 0, np.random.default_rng(0), False, "simple_trend")
    winners = winners.copy()
    winners["signal_date"] = pd.to_datetime(winners["signal_time"]).dt.tz_localize(None).dt.normalize()
    trending = compute_trend_flag_market_wide(frames, cfg.benchmark)
    return cfg, frames, winners, trending


def run_crash_overlay(winners: pd.DataFrame, frames: dict, trending: pd.Series):
    """Day-by-day single-position sim. Enter tomorrow's open on today's
    winner IF today is trending; force-exit at tomorrow's open if today's
    trend flips to non-trending mid-hold; else natural exit at MAX_HOLD."""
    trending = trending.sort_index()
    dates = trending.index
    winners_by_date = winners.set_index("signal_date")

    price = {sym: frames[sym].copy() for sym in frames}
    for sym in price:
        price[sym].index = pd.DatetimeIndex(price[sym].index).tz_localize(None).normalize()

    equity = INITIAL_CAPITAL
    position = None  # dict(symbol, entry_price, entry_date_idx)
    rows = []

    for i, d in enumerate(dates):
        if i + 1 >= len(dates):
            break
        next_d = dates[i + 1]
        is_trending_today = bool(trending.loc[d]) if d in trending.index else True

        if position is not None:
            days_held = i - position["entry_idx"]
            force_crash_exit = not is_trending_today
            natural_exit = days_held >= MAX_HOLD
            if force_crash_exit or natural_exit:
                sym = position["symbol"]
                if next_d not in price[sym].index:
                    continue
                exit_open = float(price[sym].loc[next_d, "open"])
                exit_fill = exit_open * (1 - SLIPPAGE)
                net_return = exit_fill / position["entry_price"] - 1
                pnl = INITIAL_CAPITAL * net_return  # fixed notional, no compounding, matches project convention
                equity += pnl
                rows.append({
                    "entry_date": position["entry_date"], "exit_date": next_d, "symbol": sym,
                    "net_return": net_return, "pnl": pnl, "equity_after": equity,
                    "exit_reason": "crash" if force_crash_exit else "time_exit", "days_held": days_held + 1,
                })
                position = None
                continue  # don't also re-enter same day we just exited

        if position is None and is_trending_today and d in winners_by_date.index:
            w = winners_by_date.loc[d]
            if isinstance(w, pd.DataFrame):
                w = w.iloc[0]
            sym = str(w["symbol"])
            if next_d not in price[sym].index:
                continue
            entry_open = float(price[sym].loc[next_d, "open"])
            entry_fill = entry_open * (1 + SLIPPAGE)
            position = {"symbol": sym, "entry_price": entry_fill, "entry_idx": i, "entry_date": next_d}

    trades = pd.DataFrame(rows)
    return trades


def summarize(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"label": label, "trades": 0}
    eq = INITIAL_CAPITAL + trades["pnl"].cumsum()
    dd = 1 - eq / eq.cummax()
    return {
        "label": label,
        "trades": len(trades),
        "win_rate": float((trades.net_return > 0).mean()),
        "mean_return": float(trades.net_return.mean()),
        "total_return_pct": float((eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100),
        "max_drawdown_pct": float(dd.max() * 100),
        "final_equity": float(eq.iloc[-1]),
        "crash_exits": int((trades.exit_reason == "crash").sum()),
        "time_exits": int((trades.exit_reason == "time_exit").sum()),
    }


def buy_and_hold_equal_weight(frames: dict, symbols: tuple, start: str, end: str) -> dict:
    closes = {}
    for sym in symbols:
        f = frames[sym].copy()
        f.index = pd.DatetimeIndex(f.index).tz_localize(None).normalize()
        f = f[(f.index >= start) & (f.index <= end)]
        closes[sym] = f["close"]
    df = pd.DataFrame(closes).dropna()
    norm = df / df.iloc[0]
    equal_weight = norm.mean(axis=1)
    total_return = (equal_weight.iloc[-1] / equal_weight.iloc[0] - 1) * 100
    dd = 1 - equal_weight / equal_weight.cummax()
    daily_ret = equal_weight.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else float("nan")
    return {"total_return_pct": float(total_return), "max_drawdown_pct": float(dd.max() * 100), "sharpe": float(sharpe)}


def sharpe_from_trades(trades: pd.DataFrame, dates_index: pd.DatetimeIndex) -> float:
    if trades.empty:
        return float("nan")
    eq = pd.Series(INITIAL_CAPITAL, index=dates_index)
    running = INITIAL_CAPITAL
    trades_sorted = trades.sort_values("exit_date")
    for _, t in trades_sorted.iterrows():
        running += t.pnl
        eq.loc[eq.index >= t.exit_date] = running
    daily_ret = eq.pct_change().dropna()
    daily_ret = daily_ret[daily_ret != 0]
    if daily_ret.std() == 0 or len(daily_ret) < 2:
        return float("nan")
    return float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))


def check_known_crash_episodes(trending: pd.Series, trades: pd.DataFrame):
    episodes = {
        "2018 Q4 selloff": ("2018-10-01", "2018-12-26"),
        "2020 COVID crash": ("2020-02-15", "2020-04-15"),
        "2022 bear market": ("2022-01-01", "2022-10-15"),
    }
    print("\n=== Known crash episodes: did the signal fire, and when? ===")
    for name, (start, end) in episodes.items():
        window = trending[(trending.index >= start) & (trending.index <= end)]
        if window.empty:
            print(f"{name}: no data in window")
            continue
        first_crash = window[window == False]
        first_crash_date = first_crash.index[0] if len(first_crash) else None
        crash_days = int((~window).sum())
        print(f"{name} ({start} to {end}): {crash_days}/{len(window)} days flagged crash; "
              f"first crash-flag date: {first_crash_date}")
        overlap = trades[(trades.exit_date >= start) & (trades.exit_date <= end) & (trades.exit_reason == "crash")]
        if len(overlap):
            print(f"  -> {len(overlap)} crash-triggered exit(s) in window: "
                  f"{overlap[['exit_date','symbol','net_return']].to_string(index=False)}")
        else:
            print("  -> no crash-triggered exit logged in this specific window")


def random_timing_control(cfg, frames, winners, trending, real_trades, seeds=200):
    """Same exit-count/duration distribution as the real crash-triggered
    exits, but pick RANDOM days to be 'crash' instead of the real signal."""
    n_crash_days = int((~trending).sum())
    total_days = len(trending)
    real_total_return = summarize(real_trades, "real")["total_return_pct"]
    rng = np.random.default_rng(42)
    null_returns = []
    for seed in range(seeds):
        r = np.random.default_rng(seed)
        shuffled_trending = pd.Series(True, index=trending.index)
        crash_idx = r.choice(len(trending), size=n_crash_days, replace=False)
        shuffled_trending.iloc[crash_idx] = False
        t = run_crash_overlay(winners, frames, shuffled_trending)
        s = summarize(t, f"random_{seed}")
        null_returns.append(s.get("total_return_pct", np.nan))
    null_returns = np.array([x for x in null_returns if np.isfinite(x)])
    pct = float((null_returns < real_total_return).mean() * 100)
    return real_total_return, float(np.mean(null_returns)), float(np.std(null_returns)), pct


def main():
    cfg, frames, winners, trending = load_everything()

    print("=== Baseline check: crash overlay with real trend signal ===")
    trades = run_crash_overlay(winners, frames, trending)
    baseline = summarize(trades, "crash_overlay")
    dates_index = pd.DatetimeIndex(sorted(trending.index))
    baseline_sharpe = sharpe_from_trades(trades, dates_index)
    print(baseline, "sharpe=", baseline_sharpe)

    print("\n=== Comparison: fixed 10-day hold baseline (validated) ===")
    print("188 trades, 58.0% win rate, PF 2.042, +457.4% total return, 22.1% max drawdown (from EQUITY_EXIT_REGIME_SIMPLE_TREND.md)")

    print("\n=== Buy-and-hold, equal-weighted 4-stock basket, same window ===")
    bh = buy_and_hold_equal_weight(frames, cfg.symbols, str(trades.exit_date.min().date()) if len(trades) else "2018-04-01",
                                    str(trades.exit_date.max().date()) if len(trades) else "2026-06-30")
    print(bh)

    check_known_crash_episodes(trending, trades)

    print("\n=== OOS split ===")
    mid = trades.exit_date.median() if len(trades) else pd.Timestamp("2022-01-01")
    first_half = trades[trades.exit_date <= mid]
    second_half = trades[trades.exit_date > mid]
    print("first half:", summarize(first_half, "first_half"))
    print("second half:", summarize(second_half, "second_half"))

    print("\n=== Random-timing control (200 seeds) ===")
    real_ret, null_mean, null_std, pct = random_timing_control(cfg, frames, winners, trending, trades)
    print(f"real total return: {real_ret:.1f}%  null mean: {null_mean:.1f}%  null std: {null_std:.1f}%  percentile: {pct:.1f}")

    print("\n=== Cost stress (5bps) ===")
    global SLIPPAGE
    SLIPPAGE = 5.0 / 10000
    trades_stress = run_crash_overlay(winners, frames, trending)
    print(summarize(trades_stress, "cost_stress_5bp"))

    out_dir = ROOT / "outputs" / "crash_overlay_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "crash_overlay_trades.csv", index=False)
    print(f"\nWrote {out_dir / 'crash_overlay_trades.csv'}")


if __name__ == "__main__":
    main()
