"""Re-tests the original calls deep dive (EQUITY_OPTIONS_DEEP_DIVE.md) with
REAL ThetaData historical bid/ask instead of Black-Scholes-off-realized-vol
-- the actual resolution to that doc's own verdict: "not a nail in the
coffin for calls -- an honest inconclusive without real spread data...
the highest-leverage next real-data purchase is realistic historical
bid-ask spread data."

Same signal (simple_trend, same TSLA/COIN/MSTR/PLTR/GME universe, same
mask), same 30-DTE target / 10-day hold as the original -- only the
pricing changes: real ask at entry, real bid at exit, real listed
strikes/expirations (nearest available, not an idealized exact value).

Scope: signals from 2021-06-01 onward only -- confirmed empirically (not
assumed) that this account's data tier returns real quotes from 2021
onward and "no data" for 2020 and earlier. This cuts the original 2018+
sample roughly in half, stated plainly, not hidden in a footnote.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402
from btc_trend_bot.thetadata_pricing import real_call_trade  # noqa: E402
from run_equity_options_deep_dive import build_signal_pool, select  # noqa: E402
from run_equity_real_data_walkforward import load_config, load_csv_dir  # noqa: E402

REAL_DATA_START = pd.Timestamp("2021-06-01", tz="UTC")
DTE_AT_ENTRY = 30
HOLD_DAYS = 10
STOCK_SIZING = SizingMode("fixed_notional_2500", "fixed_notional", 2500.0)
CALL_SIZING = SizingMode("fixed_premium_250", "fixed_notional", 250.0)


def price_real_calls(trades: pd.DataFrame, moneyness: float) -> pd.DataFrame:
    out = trades.copy()
    rows = []
    n_priced, n_skipped = 0, 0
    t0 = time.time()
    for i, t in enumerate(out.itertuples()):
        signal_date = t.signal_time.date()
        entry_date = t.entry_time.date()
        exit_date = t.exit_time.date()
        r = real_call_trade(t.symbol, signal_date, entry_date, exit_date, t.entry_price, DTE_AT_ENTRY, moneyness)
        if r is None:
            rows.append({"net_return": np.nan})
            n_skipped += 1
        else:
            rows.append(r)
            n_priced += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(out)} priced ({n_priced} ok, {n_skipped} skipped), {elapsed:.0f}s elapsed")
    priced = pd.DataFrame(rows)
    out["net_return"] = priced["net_return"].to_numpy()
    print(f"Moneyness {moneyness}: {n_priced} priced, {n_skipped} skipped (no real contract/quote available)")
    return out


def summarize(trades: pd.DataFrame, label: str, sizing: SizingMode, cfg) -> dict:
    if trades.empty:
        return {"label": label, "trades_taken": 0}
    t = trades.dropna(subset=["net_return"]).copy()
    _, summary = simulate_single_position(t, cfg, sizing)
    summary["label"] = label
    return summary


def main() -> None:
    cfg_base = load_config(ROOT / "configs/real_data.yaml")
    symbols = ("TSLA", "COIN", "MSTR", "PLTR", "GME")
    cfg = dataclasses.replace(cfg_base, symbols=symbols, stop_atr=100.0, target_atr=100.0, max_hold_bars=HOLD_DAYS,
                               safety_enabled=False, hard_shutdown_drawdown=1.0, minimum_equity=0.0)
    frames = load_csv_dir((*symbols, cfg.benchmark), ROOT / "data/real")

    print("Building signal pool (same simple_trend signal as the original synthetic study)...")
    pool = build_signal_pool(frames, cfg.benchmark, cfg, weakest=False)
    trades = select(pool, weakest=False)
    for col in ("signal_time", "entry_time", "exit_time"):
        trades[col] = pd.to_datetime(trades[col], utc=True)
    trades = trades[trades.signal_time >= REAL_DATA_START].reset_index(drop=True)
    print(f"{len(trades)} signals from {REAL_DATA_START.date()} onward (real-data-confirmed window)")

    stock_trades = trades.copy()
    stock_trades["net_return"] = stock_trades["stock_net_return"]
    stock_summary = summarize(stock_trades, "stock_only_real_window", STOCK_SIZING, cfg)
    print(f"\nStock-only (same window): trades={stock_summary.get('trades_taken')} "
          f"total_return={stock_summary.get('total_return',float('nan'))*100:.1f}% "
          f"PF={stock_summary.get('profit_factor') or float('nan'):.2f}")

    results = {"stock_only": stock_summary}
    for moneyness in (1.0, 1.05):
        print(f"\n=== Pricing real calls, moneyness={moneyness} ===")
        priced = price_real_calls(trades, moneyness)
        key = f"real_call_moneyness{moneyness}"
        results[key] = summarize(priced, key, CALL_SIZING, cfg)
        s = results[key]
        print(f"{key}: trades={s.get('trades_taken',0)} win={s.get('win_rate',float('nan')):.3f} "
              f"expectancy={s.get('expectancy_bps',float('nan')):.1f}bps PF={s.get('profit_factor') or float('nan'):.2f} "
              f"maxDD={s.get('max_drawdown',float('nan'))*100:.1f}% total_return={s.get('total_return',float('nan'))*100:.1f}%")

    out = ROOT / "outputs" / "equity_options_real_data_retest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")
    print("\nOriginal synthetic study (full 2018+ sample, for reference, NOT the same window):")
    print("  stock_only: +980.0% total return, PF 1.93")
    print("  call_5pctOTM_5pctspread: +975.3% total return, PF 2.32")
    print("  call_5pctOTM_10pctspread (more conservative): +380.4%, PF 1.52")


if __name__ == "__main__":
    main()
