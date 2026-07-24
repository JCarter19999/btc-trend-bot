"""Re-tests the ORIGINAL weakest-momentum put/short thesis
(EQUITY_OPTIONS_DEEP_DIVE.md's SHORT side, and the underlying short-stock
diagnostic before it) with real ThetaData quotes. Different from the
European-signal puts already re-tested (which outperformed calls) --
this is the older, separately-discredited idea: short/put the WEAKEST-
momentum candidate each day, mirroring the long side's simple_trend logic
but inverted.

Important expectation, stated up front rather than only after the
result: this thesis's problem was never established to be a pricing/data
issue. The underlying short-STOCK diagnostic (no options involved) was
already negative (-1.64% mean return, every symbol negative) -- the
signal itself lacks edge, unlike the original calls story where the
uncertainty was specifically about spread/IV assumptions. Real data is
not expected to rescue a selector with no underlying edge, and if it
doesn't, that is a confirming result, not a null one.

Same universe (TSLA/COIN/MSTR/PLTR/GME), same 2021-06-01+ real-data
window, same 30-DTE/10-day-hold framing as the calls re-test, weakest
relative-momentum candidate selected each day instead of strongest.
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
from btc_trend_bot.thetadata_pricing import real_put_trade  # noqa: E402
from run_equity_options_deep_dive import build_signal_pool, select  # noqa: E402
from run_equity_options_real_data_retest import REAL_DATA_START  # noqa: E402
from run_equity_real_data_walkforward import load_config, load_csv_dir  # noqa: E402

DTE_AT_ENTRY = 30
HOLD_DAYS = 10
STOCK_SIZING = SizingMode("fixed_notional_2500", "fixed_notional", 2500.0)
CALL_SIZING = SizingMode("fixed_premium_250", "fixed_notional", 250.0)


def price_real_puts(trades: pd.DataFrame, moneyness: float) -> pd.DataFrame:
    out = trades.copy()
    rows = []
    n_priced, n_skipped = 0, 0
    t0 = time.time()
    for i, t in enumerate(out.itertuples()):
        r = real_put_trade(t.symbol, t.signal_time.date(), t.entry_time.date(), t.exit_time.date(),
                            t.entry_price, DTE_AT_ENTRY, moneyness)
        if r is None:
            rows.append({"net_return": np.nan})
            n_skipped += 1
        else:
            rows.append(r)
            n_priced += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(out)} priced ({n_priced} ok, {n_skipped} skipped), {time.time()-t0:.0f}s elapsed", flush=True)
    priced = pd.DataFrame(rows)
    out["net_return"] = priced["net_return"].to_numpy()
    print(f"Moneyness {moneyness}: {n_priced} priced, {n_skipped} skipped", flush=True)
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

    print("Building WEAKEST-momentum signal pool (short/put thesis)...", flush=True)
    pool = build_signal_pool(frames, cfg.benchmark, cfg, weakest=True)
    trades = select(pool, weakest=True)
    for col in ("signal_time", "entry_time", "exit_time"):
        trades[col] = pd.to_datetime(trades[col], utc=True)
    trades = trades[trades.signal_time >= REAL_DATA_START].reset_index(drop=True)
    print(f"{len(trades)} weakest-momentum signals from {REAL_DATA_START.date()} onward", flush=True)

    short_stock_trades = trades.copy()
    short_stock_trades["net_return"] = -short_stock_trades["stock_net_return"]  # short = inverse of stock return
    short_stock_summary = summarize(short_stock_trades, "short_stock_real_window", STOCK_SIZING, cfg)
    print(f"\nShort-stock (same window, no options): trades={short_stock_summary.get('trades_taken')} "
          f"total_return={short_stock_summary.get('total_return',float('nan'))*100:.1f}% "
          f"PF={short_stock_summary.get('profit_factor') or float('nan'):.2f}", flush=True)

    results = {"short_stock": short_stock_summary}
    for moneyness in (1.0, 0.95):
        print(f"\n=== Pricing real puts, moneyness={moneyness} ===", flush=True)
        priced = price_real_puts(trades, moneyness)
        key = f"real_put_moneyness{moneyness}"
        results[key] = summarize(priced, key, CALL_SIZING, cfg)
        s = results[key]
        print(f"{key}: trades={s.get('trades_taken',0)} win={s.get('win_rate',float('nan')):.3f} "
              f"expectancy={s.get('expectancy_bps',float('nan')):.1f}bps PF={s.get('profit_factor') or float('nan'):.2f} "
              f"total_return={s.get('total_return',float('nan'))*100:.1f}%", flush=True)

    out = ROOT / "outputs" / "equity_puts_real_data_retest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")
    print("\nOriginal synthetic study (full universe/window, for reference):")
    print("  put_ATM_5pctspread: win 28.9%, total_return +1.7%, PF 1.01")
    print("  put_ATM_10pctspread: win 27.7%, total_return -77.9%, PF 0.75")


if __name__ == "__main__":
    main()
