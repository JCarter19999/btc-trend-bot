"""Isolates how much of the original calls deep dive's "not a clean nail,
inconclusive without real spread data" verdict was a DATA problem
(Black-Scholes-off-realized-vol mispricing) vs. a STRATEGY problem (the
signal/structure itself doesn't work well expressed as options) --
Joey's direct question, 2026-07-23.

Method: price the EXACT SAME signal pool (same symbols, same dates, same
2021-06-01+ window used in the real-data re-test) TWO ways -- synthetic
Black-Scholes-off-realized-vol (the old method) and real ThetaData quotes
(the new one, from run_equity_options_real_data_retest.py's output). Only
the pricing methodology differs; everything else (signal, universe,
dates, hold period, target DTE/moneyness) is held identical. Any
difference in outcome is then attributable to data quality, not strategy,
by construction -- this is the whole point of holding everything else fixed.

Depends on run_equity_options_real_data_retest.py having already run (reads
its real-data trades) -- run that first.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.options_pricing import synthetic_call_trade  # noqa: E402
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402
from run_equity_options_deep_dive import build_signal_pool, select  # noqa: E402
from run_equity_options_real_data_retest import REAL_DATA_START, DTE_AT_ENTRY  # noqa: E402
from run_equity_real_data_walkforward import load_config, load_csv_dir  # noqa: E402

CALL_SIZING = SizingMode("fixed_premium_250", "fixed_notional", 250.0)


def summarize(trades: pd.DataFrame, label: str, cfg) -> dict:
    if trades.empty:
        return {"label": label, "trades_taken": 0}
    t = trades.dropna(subset=["net_return"]).copy()
    _, summary = simulate_single_position(t, cfg, CALL_SIZING)
    summary["label"] = label
    return summary


def main() -> None:
    cfg_base = load_config(ROOT / "configs/real_data.yaml")
    symbols = ("TSLA", "COIN", "MSTR", "PLTR", "GME")
    cfg = dataclasses.replace(cfg_base, symbols=symbols, stop_atr=100.0, target_atr=100.0, max_hold_bars=10,
                               safety_enabled=False, hard_shutdown_drawdown=1.0, minimum_equity=0.0)
    frames = load_csv_dir((*symbols, cfg.benchmark), ROOT / "data/real")

    print("Rebuilding the EXACT SAME signal pool used by the real-data re-test...")
    pool = build_signal_pool(frames, cfg.benchmark, cfg, weakest=False)
    trades = select(pool, weakest=False)
    for col in ("signal_time", "entry_time", "exit_time"):
        trades[col] = pd.to_datetime(trades[col], utc=True)
    trades = trades[trades.signal_time >= REAL_DATA_START].reset_index(drop=True)
    print(f"{len(trades)} signals, same window as the real-data re-test ({REAL_DATA_START.date()}+)")

    real_summary_path = ROOT / "outputs" / "equity_options_real_data_retest" / "summary.json"
    real_results = json.loads(real_summary_path.read_text()) if real_summary_path.exists() else None
    if real_results is None:
        print("WARNING: real-data re-test hasn't produced summary.json yet -- run "
              "run_equity_options_real_data_retest.py first. Showing synthetic-only results for now.")

    print("\n=== Synthetic Black-Scholes-off-realized-vol, SAME signals/window ===")
    synthetic_results = {}
    for moneyness, spread in ((1.0, 0.05), (1.05, 0.05)):
        priced = trades.copy()
        rows = [synthetic_call_trade(t.entry_price, t.entry_vol, t.exit_price, t.exit_vol, DTE_AT_ENTRY, 10,
                                      strike_moneyness=moneyness, spread_frac_of_premium=spread)
                for t in priced.itertuples()]
        priced["net_return"] = [r["net_return"] for r in rows]
        key = f"synthetic_call_moneyness{moneyness}_spread{spread}"
        synthetic_results[key] = summarize(priced, key, cfg)
        s = synthetic_results[key]
        print(f"{key}: trades={s.get('trades_taken',0)} win={s.get('win_rate',float('nan')):.3f} "
              f"expectancy={s.get('expectancy_bps',float('nan')):.1f}bps PF={s.get('profit_factor') or float('nan'):.2f} "
              f"total_return={s.get('total_return',float('nan'))*100:.1f}%")

    if real_results:
        print("\n=== SIDE BY SIDE: same signals, same window, only pricing method differs ===")
        print(f"{'Metric':<20} {'Synthetic (1.0/5%)':<22} {'Real (1.0)':<18} {'Synthetic (1.05/5%)':<22} {'Real (1.05)':<18}")
        syn_atm = synthetic_results.get("synthetic_call_moneyness1.0_spread0.05", {})
        syn_otm = synthetic_results.get("synthetic_call_moneyness1.05_spread0.05", {})
        real_atm = real_results.get("real_call_moneyness1.0", {})
        real_otm = real_results.get("real_call_moneyness1.05", {})
        for metric, fmt in [("trades_taken", "{:.0f}"), ("win_rate", "{:.3f}"), ("expectancy_bps", "{:.1f}"),
                             ("profit_factor", "{:.2f}"), ("total_return", "{:.1%}"), ("max_drawdown", "{:.1%}")]:
            def g(d, m):
                v = d.get(m)
                return fmt.format(v) if v is not None else "—"
            print(f"{metric:<20} {g(syn_atm,metric):<22} {g(real_atm,metric):<18} {g(syn_otm,metric):<22} {g(real_otm,metric):<18}")

    out = ROOT / "outputs" / "equity_options_real_vs_synthetic"
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_same_window.json").write_text(json.dumps(synthetic_results, indent=2, default=str))
    print(f"\nWritten to {out / 'synthetic_same_window.json'}")


if __name__ == "__main__":
    main()
