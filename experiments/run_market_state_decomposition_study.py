"""Why does DAX predict SPY -- market-state decomposition, per Joey's
vertical-understanding pivot (2026-07-24): does the already-validated,
already-gated DAX top-quartile signal strengthen or weaken under specific
market conditions (elevated VIX, large overnight gap, large overnight
range)? All three state variables use only information available before
signal_date's US open -- no lookahead.

- VIX regime: PRIOR trading day's ^VIX close (not same-day -- VIX at
  13:30 UTC today isn't settled/known before the US open decision).
- Gap size: |today's US open - yesterday's US close| / yesterday's close,
  computed from the same SPY 60m bars already in use everywhere else.
- Overnight range: DAX's own session |high-low|/open that same morning --
  already known before 13:30 UTC signal time, a direct proxy for "how
  much overnight/European-session volatility was there."

Splits each state variable at its median (not quartiles -- with 116
gated trades, quartile bucketing leaves too few trades per bucket to
read anything from) and compares the DAX-gated trade's Sharpe/win/return
above vs. below median.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_european_lead_us_first_hour_backtest import build_dataset, evaluate  # noqa: E402
from run_european_lead_us_first_hour_study import download_hourly  # noqa: E402

US_OPEN_UTC = pd.Timestamp("13:30:00").time()


def main() -> None:
    print("Building DAX signal + state variables...", flush=True)
    df = build_dataset()
    abs_dax = df["eu_pre_open_return"].abs()
    gated_direction = df["direction"].where(abs_dax >= abs_dax.quantile(0.75), 0)
    gated_dates = df.index[gated_direction != 0]

    # VIX: prior trading day's close
    import yfinance as yf
    vix = yf.download("^VIX", start="2023-08-01", progress=False, auto_adjust=True)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix_prior_close = vix["Close"].shift(1)
    vix_state = vix_prior_close.reindex(df.index, method="ffill")

    # Gap size: today's US open vs. yesterday's US close (SPY 60m bars)
    spy = download_hourly("SPY")
    spy_daily = spy.groupby(spy.index.date).agg(open=("open", "first"), close=("close", "last"))
    spy_daily.index = pd.to_datetime(spy_daily.index)
    gap = (spy_daily["open"] - spy_daily["close"].shift(1)).abs() / spy_daily["close"].shift(1)
    gap_state = gap.reindex(df.index)

    # Overnight range: DAX's own session range that morning
    dax = download_hourly("^GDAXI")
    dax["date"] = dax.index.date
    dax["time"] = dax.index.time
    pre_open = dax[dax["time"] < US_OPEN_UTC]
    dax_range = pre_open.groupby("date").apply(lambda d: (d["high"].max() - d["low"].min()) / d["open"].iloc[0])
    dax_range.index = pd.to_datetime(dax_range.index)
    range_state = dax_range.reindex(df.index)

    state_vars = {"prior_day_VIX_close": vix_state, "overnight_gap_pct": gap_state, "dax_session_range_pct": range_state}

    print(f"Coverage on {len(gated_dates)} gated trade days: "
          f"VIX={vix_state.reindex(gated_dates).notna().sum()}, "
          f"gap={gap_state.reindex(gated_dates).notna().sum()}, "
          f"range={range_state.reindex(gated_dates).notna().sum()}", flush=True)

    all_results = {}
    for state_name, state_series in state_vars.items():
        gated_state = state_series.reindex(gated_dates).dropna()
        median = gated_state.median()
        above = gated_direction.copy()
        above.loc[~above.index.isin(gated_state[gated_state >= median].index)] = 0
        below = gated_direction.copy()
        below.loc[~below.index.isin(gated_state[gated_state < median].index)] = 0

        r_above = evaluate(df, above, 1.0, f"{state_name}_above_median")
        r_below = evaluate(df, below, 1.0, f"{state_name}_below_median")
        all_results[state_name] = {"median": float(median), "above": r_above, "below": r_below}

        print(f"\n{state_name} (median={median:.4f}):")
        print(f"  ABOVE median: trades={r_above.get('trades',0):3d} win={r_above.get('win_rate',float('nan')):.3f} "
              f"sharpe={r_above.get('sharpe_annualized') or float('nan'):6.2f} total={r_above.get('total_return_compounded',float('nan'))*100:7.2f}%")
        print(f"  BELOW median: trades={r_below.get('trades',0):3d} win={r_below.get('win_rate',float('nan')):.3f} "
              f"sharpe={r_below.get('sharpe_annualized') or float('nan'):6.2f} total={r_below.get('total_return_compounded',float('nan'))*100:7.2f}%")

    out = ROOT / "outputs" / "market_state_decomposition_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
