"""Options "nail in the coffin" deep dive -- Track A/B follow-up to Joey's
original question (does a 10-day stock thesis justify a 30-day call?) and
the synthetic-hybrid-optimizer's earlier negative finding. Two changes from
that earlier test: (1) real underlying daily price data instead of synthetic
30-minute bars, (2) a genuinely more volatile universe (TSLA/COIN/MSTR/PLTR/
GME instead of AAPL/MSFT/NVDA/TSLA), since options economics depend heavily
on volatility and the original universe was unusually low-vol mega-caps.

Still NOT a real options backtest -- see options_pricing.py's docstring for
the full honesty framing. Short version: prices are Black-Scholes off
trailing REALIZED volatility (real data), which systematically UNDERPRICES
real options (volatility risk premium), making this a deliberately generous
test in the options' favor. A negative result here is real evidence; a
positive one is weak evidence pending real data.

Compares, on the identical simple_trend-selected signal (same entry dates/
symbols the stock-only strategy uses):
  - stock-only (fixed 10-day hold, matching the live deployment)
  - long call, 30 DTE at entry, sold (not exercised) at the same 10-day exit
  - long put on the mirror-image weakest-momentum signal (vs. short-selling
    the same signal, tested earlier and found structurally negative on the
    old universe -- does a *capped-risk* put change that on a universe with
    real drawdown periods?)
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

from btc_trend_bot.options_pricing import realized_vol, synthetic_call_trade, synthetic_put_trade
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position

from run_equity_real_data_walkforward import (
    BacktestConfig, add_features, load_config, load_csv_dir,
)

VOL_WINDOW = 20
DTE_AT_ENTRY = 30
HOLD_DAYS = 10
STOCK_SIZING = SizingMode("fixed_notional_2500", "fixed_notional", 2500.0)
CALL_PREMIUM_BUDGET = 250.0  # 10% of the stock notional -- a defined-risk premium allocation, not notional-matched
CALL_SIZING = SizingMode("fixed_premium_250", "fixed_notional", CALL_PREMIUM_BUDGET)


def build_signal_pool(frames: dict, benchmark_symbol: str, cfg: BacktestConfig, weakest: bool) -> pd.DataFrame:
    """Mirrors build_candidates/select_simple_trend from the equity pipeline,
    but attaches realized_vol (needed for option pricing) per symbol, and can
    select either strongest (long thesis) or weakest (short/put thesis)
    relative-momentum candidate."""
    bench = frames[benchmark_symbol]
    rows = []
    for symbol in cfg.symbols:
        f = add_features(frames[symbol], bench)
        f = f.copy()
        f["realized_vol"] = realized_vol(f["close"].to_numpy(), VOL_WINDOW)
        for i in range(60, len(f) - HOLD_DAYS - 1):
            r = f.iloc[i]
            mask = (r.relative_volume > 0.6 and r.atr_pct > 0 and abs(r.ema_spread_atr) < 8)
            if not weakest:
                mask = mask and r.return_20 > -0.25
            if not mask:
                continue
            if not np.isfinite(r.get("realized_vol", np.nan)) or r.realized_vol <= 0:
                continue
            entry_i = i + 1
            exit_i = min(entry_i + HOLD_DAYS - 1, len(f) - 1)
            entry_price = float(f.iloc[entry_i].open)
            exit_price = float(f.iloc[exit_i].close)
            exit_vol = float(f.iloc[exit_i].realized_vol) if np.isfinite(f.iloc[exit_i].realized_vol) else float(r.realized_vol)
            stock_return = exit_price / entry_price - 1
            rows.append({
                "signal_time": f.index[i], "symbol": symbol,
                "entry_time": f.index[entry_i], "exit_time": f.index[exit_i],
                "entry_price": entry_price, "exit_price": exit_price,
                "entry_vol": float(r.realized_vol), "exit_vol": exit_vol,
                "stock_net_return": stock_return,
                "relative_strength_20": float(r.relative_strength_20) if np.isfinite(r.relative_strength_20) else np.nan,
                "market_above_ema50": float(r.market_above_ema50) if "market_above_ema50" in r else np.nan,
            })
    return pd.DataFrame(rows)


def select(pool: pd.DataFrame, weakest: bool) -> pd.DataFrame:
    if pool.empty:
        return pool
    if not weakest:
        eligible = pool[pool.market_above_ema50 >= 1] if "market_above_ema50" in pool else pool
        ascending = False
    else:
        eligible = pool[pool.market_above_ema50 < 1] if "market_above_ema50" in pool else pool
        ascending = True
    if eligible.empty:
        eligible = pool
    return (
        eligible.sort_values(["signal_time", "relative_strength_20"], ascending=[True, ascending])
        .groupby("signal_time", as_index=False).head(1).sort_values("signal_time")
    )


def price_options(trades: pd.DataFrame, kind: str, strike_moneyness: float, spread: float) -> pd.DataFrame:
    out = trades.copy()
    results = []
    for _, t in out.iterrows():
        fn = synthetic_call_trade if kind == "call" else synthetic_put_trade
        r = fn(t.entry_price, t.entry_vol, t.exit_price, t.exit_vol, DTE_AT_ENTRY, HOLD_DAYS,
               strike_moneyness=strike_moneyness, spread_frac_of_premium=spread)
        results.append(r)
    priced = pd.DataFrame(results)
    out["net_return"] = priced["net_return"].to_numpy()
    out["entry_premium"] = priced["entry_premium"].to_numpy()
    return out


def summarize(trades: pd.DataFrame, label: str, sizing: SizingMode, cfg: BacktestConfig) -> dict:
    if trades.empty:
        return {"label": label, "trades": 0}
    t = trades.dropna(subset=["net_return"]).copy()
    _, summary = simulate_single_position(t, cfg, sizing)
    summary["label"] = label
    return summary


def main() -> None:
    cfg_base = load_config(ROOT / "configs/real_data.yaml")
    symbols = ("TSLA", "COIN", "MSTR", "PLTR", "GME")
    import dataclasses
    # safety_enabled=False AND hard_shutdown_drawdown=1.0: this universe's
    # volatility (62-127% annualized vs. the ~30-60% the safety thresholds
    # were calibrated against) makes the $2,500-fixed-notional hard-shutdown
    # trip almost immediately on the stock leg -- and simulate_single_position
    # (like the original simulate_capital it mirrors) checks
    # hard_shutdown_drawdown UNCONDITIONALLY, not gated by safety_enabled, so
    # disabling safety_enabled alone doesn't stop it from permanently halting
    # the sim after a handful of early bad trades. Both are neutralized here
    # because this study is testing expression economics, not safety-layer
    # robustness on a mismatched-volatility universe.
    cfg = dataclasses.replace(cfg_base, symbols=symbols, stop_atr=100.0, target_atr=100.0, max_hold_bars=HOLD_DAYS,
                               safety_enabled=False, hard_shutdown_drawdown=1.0, minimum_equity=0.0)
    frames = load_csv_dir((*symbols, cfg.benchmark), ROOT / "data/real")

    print("=== Universe realized volatility (full history, annualized) ===")
    for s in symbols:
        rv = frames[s].close.pct_change().std() * np.sqrt(252)
        print(f"  {s}: {rv*100:.1f}%")

    results = {}

    print("\n=== LONG side: stock vs. call, simple_trend selection ===")
    long_pool = build_signal_pool(frames, cfg.benchmark, cfg, weakest=False)
    long_trades = select(long_pool, weakest=False)
    long_trades["signal_time"] = pd.to_datetime(long_trades["signal_time"], utc=True)
    long_trades["entry_time"] = pd.to_datetime(long_trades["entry_time"], utc=True)
    long_trades["exit_time"] = pd.to_datetime(long_trades["exit_time"], utc=True)
    print(f"Signal count: {len(long_trades)}")

    stock_trades = long_trades.copy()
    stock_trades["net_return"] = stock_trades["stock_net_return"]
    results["stock_only"] = summarize(stock_trades, "stock_only", STOCK_SIZING, cfg)

    for moneyness, spread in ((1.0, 0.05), (1.0, 0.10), (1.05, 0.05)):
        call_trades = price_options(long_trades, "call", moneyness, spread)
        key = f"call_moneyness{moneyness}_spread{spread}"
        results[key] = summarize(call_trades, key, CALL_SIZING, cfg)

    print("\n=== SHORT side: put vs. short-stock (context only -- prior short study), weakest-momentum selection ===")
    short_pool = build_signal_pool(frames, cfg.benchmark, cfg, weakest=True)
    short_trades = select(short_pool, weakest=True)
    short_trades["signal_time"] = pd.to_datetime(short_trades["signal_time"], utc=True)
    short_trades["entry_time"] = pd.to_datetime(short_trades["entry_time"], utc=True)
    short_trades["exit_time"] = pd.to_datetime(short_trades["exit_time"], utc=True)
    print(f"Signal count: {len(short_trades)}")

    for moneyness, spread in ((1.0, 0.05), (1.0, 0.10)):
        put_trades = price_options(short_trades, "put", moneyness, spread)
        key = f"put_moneyness{moneyness}_spread{spread}"
        results[key] = summarize(put_trades, key, CALL_SIZING, cfg)

    print("\n=== Results ===")
    for key, s in results.items():
        if s.get("trades_taken", 0) == 0:
            print(f"{key:32s} NO TRADES")
            continue
        print(f"{key:32s} trades={s['trades_taken']:4d} win={s.get('win_rate',float('nan')):.3f} "
              f"expectancy={s.get('expectancy_bps',float('nan')):8.1f}bps "
              f"PF={s.get('profit_factor') or float('nan'):.2f} maxDD={s.get('max_drawdown',float('nan')):.3f} "
              f"total_return={s.get('total_return',float('nan'))*100:7.1f}%")

    out = ROOT / "outputs" / "equity_options_deep_dive"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
