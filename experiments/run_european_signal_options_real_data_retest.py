"""Re-tests the European lead signal (DAX-top-quartile direction) expressed
via SPY options (0DTE and 1DTE ATM, real quotes) instead of shares --
Joey's priority #1 item from his ThetaData task list. Same signal dates
as the validated backtest (EUROPEAN_LEAD_US_FIRST_HOUR_BACKTEST.txt,
top-quartile-by-|DAX move| filter, static cutoff over the full sample --
matching what was actually out-of-sample tested, not the live shadow
deployment's expanding-percentile variant), same entry (~9:30 ET) and
exit (~10:30 ET) timestamps. Real intraday 1-minute quotes, ask-in/bid-out.

Data window: SPY options confirmed available back to the full 2023-2026
hourly-bar backtest window (unlike the 2021-cutoff volatile-universe
equities -- SPY/QQQ are far more liquid and have deep chain history at
this subscription tier).
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

from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402
from btc_trend_bot.thetadata_intraday_pricing import real_first_hour_option_trade  # noqa: E402
from run_european_lead_us_first_hour_backtest import build_dataset  # noqa: E402
from run_european_lead_us_first_hour_study import download_hourly  # noqa: E402

PREMIUM_SIZING = SizingMode("fixed_premium_250", "fixed_notional", 250.0)


def price_options(df: pd.DataFrame, spy_open: pd.Series, dte_days: int, instrument: str) -> pd.DataFrame:
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
        r = real_first_hour_option_trade(instrument, trade_date, int(row["direction"]), dte_days, float(spot))
        if r is None:
            rows.append({"net_return": np.nan})
            n_skipped += 1
        else:
            rows.append(r)
            n_priced += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} priced ({n_priced} ok, {n_skipped} skipped), {time_module.time()-t0:.0f}s elapsed", flush=True)
    priced = pd.DataFrame(rows, index=df.index)
    print(f"{instrument} {dte_days}DTE: {n_priced} priced, {n_skipped} skipped", flush=True)
    return priced


def summarize(net_returns: pd.Series, label: str) -> dict:
    trades = pd.DataFrame({
        "signal_time": net_returns.index, "symbol": "SPY",
        "entry_time": net_returns.index, "exit_time": net_returns.index,
        "net_return": net_returns.to_numpy(),
    }).dropna(subset=["net_return"])
    if trades.empty:
        return {"label": label, "trades_taken": 0}

    class Cfg:
        initial_capital = 2500.0
        minimum_equity = 0.0
        hard_shutdown_drawdown = 1.0
        safety_enabled = False
        drawdown_pause = 1.0
        cooldown_trades = 0
        consecutive_loss_limit = 10_000

    _, summary = simulate_single_position(trades, Cfg(), PREMIUM_SIZING)
    summary["label"] = label
    return summary


def main() -> None:
    print("Building DAX-top-quartile signal dataset (same as the validated backtest)...", flush=True)
    df = build_dataset()
    abs_eu = df["eu_pre_open_return"].abs()
    top_quartile = df[abs_eu >= abs_eu.quantile(0.75)].copy()
    if top_quartile.index.tz is None:
        top_quartile.index = top_quartile.index.tz_localize("UTC")
    print(f"{len(df)} total days, {len(top_quartile)} top-quartile-by-|DAX move| days", flush=True)

    spy_hourly = download_hourly("SPY")
    spy_open = spy_hourly[spy_hourly.index.time == pd.Timestamp("13:30:00").time()]["open"]
    spy_open.index = spy_open.index.normalize()

    results = {}
    for dte in (0, 1):
        print(f"\n=== SPY ATM {dte}DTE options on top-quartile DAX signal ===", flush=True)
        priced = price_options(top_quartile, spy_open, dte, "SPY")
        key = f"spy_atm_{dte}dte"
        results[key] = summarize(priced["net_return"], key)
        s = results[key]
        print(f"{key}: trades={s.get('trades_taken',0)} win={s.get('win_rate',float('nan')):.3f} "
              f"expectancy={s.get('expectancy_bps',float('nan')):.1f}bps PF={s.get('profit_factor') or float('nan'):.2f} "
              f"total_return={s.get('total_return',float('nan'))*100:.1f}%", flush=True)

    out = ROOT / "outputs" / "european_signal_options_real_data_retest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")
    print("\nReference: SPY shares (top-quartile arm, 2bp cost) from the validated backtest:")
    print("  win 61.2%, mean 9.57bps/trade, total_return +12.9%, Sharpe 3.76")


if __name__ == "__main__":
    main()
