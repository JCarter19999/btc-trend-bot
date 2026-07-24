"""Instrument-selection experiment, per Joey's reframing (2026-07-24):
not "should I trade options" but "given this validated edge, which
instrument produces the best risk-adjusted outcome." Applies to the
European lead signal (DAX-top-quartile direction, 116 real trading days,
2023-09 to 2026-07) as the concrete demonstration -- shares and ATM
0DTE/1DTE already priced with real quotes in
run_european_signal_options_real_data_retest.py; this adds the two
missing instruments (0.40-delta 1DTE, vertical spread) and produces one
unified ranking.

Mini futures: explicitly out of scope, not silently skipped -- no data
access for that asset class at this subscription tier or in this
project's existing data pipeline.
"""

from __future__ import annotations

import json
import sys
import time as time_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.thetadata_intraday_pricing import (  # noqa: E402
    real_delta_targeted_trade, real_vertical_spread_first_hour_trade,
)
from run_equity_options_real_data_retest import summarize as _unused  # noqa: E402,F401
from run_european_signal_options_real_data_retest import rigor_report  # noqa: E402
from run_european_lead_us_first_hour_backtest import build_dataset  # noqa: E402
from run_european_lead_us_first_hour_study import download_hourly  # noqa: E402


def price_instrument(df: pd.DataFrame, spy_open: pd.Series, fn, label: str, **kwargs) -> pd.DataFrame:
    rows = []
    n_priced, n_skipped = 0, 0
    t0 = time_module.time()
    for i, (dt, row) in enumerate(df.iterrows()):
        trade_date = dt.date()
        spot = spy_open.get(dt.normalize())
        if spot is None or not np.isfinite(spot):
            rows.append({"net_return": np.nan})
            n_skipped += 1
            continue
        r = fn("SPY", trade_date, int(row["direction"]), **kwargs, spot_at_entry=float(spot))
        if r is None:
            rows.append({"net_return": np.nan})
            n_skipped += 1
        else:
            rows.append(r)
            n_priced += 1
        if (i + 1) % 20 == 0:
            print(f"  [{label}] {i+1}/{len(df)} priced ({n_priced} ok, {n_skipped} skipped), "
                  f"{time_module.time()-t0:.0f}s elapsed", flush=True)
    priced = pd.DataFrame(rows, index=df.index)
    print(f"[{label}] {n_priced} priced, {n_skipped} skipped", flush=True)
    return priced


def main() -> None:
    print("Building DAX-top-quartile signal dataset...", flush=True)
    df = build_dataset()
    abs_eu = df["eu_pre_open_return"].abs()
    top_quartile = df[abs_eu >= abs_eu.quantile(0.75)].copy()
    if top_quartile.index.tz is None:
        top_quartile.index = top_quartile.index.tz_localize("UTC")
    print(f"{len(top_quartile)} top-quartile days", flush=True)

    spy_hourly = download_hourly("SPY")
    spy_open = spy_hourly[spy_hourly.index.time == pd.Timestamp("13:30:00").time()]["open"]
    spy_open.index = spy_open.index.normalize()

    out = ROOT / "outputs" / "instrument_selection_experiment"
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    # Already have shares + ATM 0DTE/1DTE from the prior re-test's saved raw trades
    prior_dir = ROOT / "outputs" / "european_signal_options_real_data_retest"
    for dte in (0, 1):
        raw = pd.read_parquet(prior_dir / f"raw_trades_{dte}dte.parquet")
        results[f"spy_atm_{dte}dte"] = rigor_report(raw, f"spy_atm_{dte}dte")

    delta_path = out / "raw_trades_delta040_1dte.parquet"
    if delta_path.exists():
        print(f"\n=== 0.40-delta 1DTE (reusing saved {delta_path.name}) ===", flush=True)
        delta_priced = pd.read_parquet(delta_path)
    else:
        print("\n=== 0.40-delta 1DTE ===", flush=True)
        delta_priced = price_instrument(top_quartile, spy_open, real_delta_targeted_trade, "0.40delta_1dte",
                                         dte_days=1, target_delta=0.40)
        delta_priced.to_parquet(delta_path)
    results["spy_040delta_1dte"] = rigor_report(delta_priced, "spy_040delta_1dte")

    print("\n=== Vertical spread 1DTE ===", flush=True)
    spread_priced = price_instrument(top_quartile, spy_open, real_vertical_spread_first_hour_trade, "vertical_spread_1dte",
                                      dte_days=1)
    spread_priced.to_parquet(out / "raw_trades_vertical_spread_1dte.parquet")
    results["spy_vertical_spread_1dte"] = rigor_report(spread_priced, "spy_vertical_spread_1dte")

    print("\n=== RANKING (by Sharpe) ===", flush=True)
    # Shares reference from the validated backtest (not re-priced here -- already real, no options involved)
    results["spy_shares"] = {
        "label": "spy_shares", "trades_taken": 116, "win_rate": 0.612, "total_return": 0.129,
        "sharpe_annualized": 3.76, "max_drawdown": None, "note": "from EUROPEAN_LEAD_US_FIRST_HOUR_BACKTEST.txt, not re-run here",
    }
    ranked = sorted(
        [(k, v) for k, v in results.items() if v.get("trades_taken", 0) > 0],
        key=lambda kv: kv[1].get("sharpe_annualized") or -999, reverse=True)
    for name, r in ranked:
        print(f"{name:30s} trades={r.get('trades_taken'):4} win={r.get('win_rate',float('nan')):.3f} "
              f"sharpe={r.get('sharpe_annualized') or float('nan'):6.2f} maxDD={r.get('max_drawdown') if r.get('max_drawdown') is not None else float('nan'):.1%} "
              f"total_return={r.get('total_return',float('nan')):+.1%}", flush=True)

    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")
    print("\nNote: mini futures explicitly out of scope -- no data access for that asset class.")


if __name__ == "__main__":
    main()
