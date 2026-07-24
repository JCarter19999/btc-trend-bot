"""Adds futures (ES=F, continuous front-month E-mini S&P) as a sixth
instrument to the instrument-selection framework, per Joey's flag that
"mini futures" was the vehicle the platform hadn't tested yet. No new
data purchase needed -- yfinance serves free 60m continuous-futures bars
(confirmed: 'ES=F' returns real hourly OHLCV, same endpoint already used
for SPY/^GDAXI everywhere else in this project), so this reuses the exact
same DAX top-quartile signal, same 13:30-14:30 UTC first-hour window, same
directional rule as the SPY-shares backtest -- only the traded instrument
changes.

What this does NOT model: real futures accounts trade on margin (roughly
15-50x notional leverage depending on broker/day-trading buying power),
so the honest number to report is the RAW index-point return (directly
comparable to the "shares" row -- no leverage assumed, same footing as
trading 1 share of SPY), with a plain note that a levered account would
scale both the return AND the drawdown by whatever margin ratio it
actually uses. No fabricated leverage multiplier -- that number depends on
a specific broker's margin schedule this project has no data on.

Also skips modeling overnight roll cost/basis (ES is cash-settled
quarterly, continuous contract has small roll gaps) since every trade
here opens and closes same-day within a single contract period -- no
roll occurs inside any one trade.

Window caveat, found and fixed after the first run produced only 1
usable trade: ES=F trades ~24/5, so yfinance's free 60m bars for it are
aligned to the top of the hour (13:00, 14:00, 15:00 UTC...), NOT the
:30 grid SPY/DAX get (whose 60m bars start at the 13:30 UTC cash open).
There is no 13:30-14:30 UTC bar to look up at all for ES=F at this
granularity -- the exact SPY first-hour window literally doesn't exist
in this feed. 30m bars WOULD align to :30, but yfinance caps 30m history
at ~60 days, useless against a signal spanning 2023-09 to 2026-07 (fewer
than 116/730 of the actual signal days would even be in range). The
fix used here: approximate the window as 13:00-15:00 UTC (open of the
13:00 bar to close of the 14:00 bar) -- a 2-hour window that fully
contains the true 13:30-14:30 hour, on the full ~2.4-year 60m history.
Stated plainly: this is a wider, shifted window, not the literal same
signal target as the SPY backtest -- a real proxy, not an exact match.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_european_lead_us_first_hour_backtest import build_dataset  # noqa: E402
from run_european_lead_us_first_hour_study import download_hourly  # noqa: E402

ROUND_TRIP_COSTS_BPS = (1.0, 2.0, 5.0)  # ES bid/ask is typically tighter than SPY in point terms; same range for comparability


def futures_two_hour_proxy_return(fut_df: pd.DataFrame) -> pd.Series:
    """13:00-15:00 UTC window (open of the 13:00 bar to close of the 14:00
    bar) -- the closest available proxy on ES=F's :00-aligned 60m grid to
    SPY's true 13:30-14:30 UTC first hour. See module docstring."""
    d = fut_df.copy()
    d["date"] = d.index.date
    d["time"] = d.index.time
    t1300, t1400 = pd.Timestamp("13:00:00").time(), pd.Timestamp("14:00:00").time()
    opens = d[d["time"] == t1300].set_index("date")["open"]
    closes = d[d["time"] == t1400].set_index("date")["close"]
    joined = pd.concat([opens, closes], axis=1, keys=["open", "close"]).dropna()
    return pd.Series(
        (joined["close"] / joined["open"] - 1).to_dict(), name="futures_2hr_proxy_return")


def evaluate(direction: pd.Series, target: pd.Series, cost_bps: float, label: str) -> dict:
    traded = direction != 0
    idx = direction[traded].index.intersection(target.index)
    gross = direction.loc[idx] * target.loc[idx]
    net = gross - cost_bps / 10000.0
    if len(net) == 0:
        return {"label": label, "trades": 0}
    equity = (1 + net).cumprod()
    drawdown = 1 - equity / equity.cummax()
    sharpe = float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else None
    return {
        "label": label, "cost_bps_round_trip": cost_bps, "trades": int(len(net)),
        "win_rate": float((net > 0).mean()),
        "mean_return_bps": float(net.mean() * 10000),
        "total_return_compounded": float(equity.iloc[-1] - 1),
        "sharpe_annualized": sharpe,
        "max_drawdown": float(drawdown.max()),
    }


def main() -> None:
    print("Building DAX signal + top-quartile filter (same as instrument-selection framework)...", flush=True)
    df = build_dataset()
    abs_eu = df["eu_pre_open_return"].abs()
    top_quartile = df[abs_eu >= abs_eu.quantile(0.75)].copy()
    print(f"{len(top_quartile)} top-quartile signal days", flush=True)

    print("Downloading ES=F 60m bars (yfinance, free, ~2.4yr lookback observed)...", flush=True)
    es = download_hourly("ES=F")
    print(f"ES=F 60m coverage: {es.index.min().date()} to {es.index.max().date()}", flush=True)
    es_first_hour = futures_two_hour_proxy_return(es)
    es_first_hour.index = pd.to_datetime(es_first_hour.index)
    if es_first_hour.index.tz is None:
        es_first_hour.index = es_first_hour.index.tz_localize("UTC")

    direction = top_quartile["direction"]
    direction.index = pd.to_datetime(direction.index)
    if direction.index.tz is None:
        direction.index = direction.index.tz_localize("UTC")
    direction.index = direction.index.normalize()
    es_first_hour.index = es_first_hour.index.normalize()

    overlap = direction.index.intersection(es_first_hour.index)
    print(f"{len(overlap)} of {len(direction)} top-quartile days have ES=F 60m coverage "
          f"(yfinance 60m history caps at 730 days back, so older signal days are dropped here, not mispriced)", flush=True)

    out = ROOT / "outputs" / "futures_instrument_experiment"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for cost_bps in ROUND_TRIP_COSTS_BPS:
        r = evaluate(direction, es_first_hour, cost_bps, "es_futures_2hr_proxy")
        results[cost_bps] = r
        print(f"cost={cost_bps}bps trades={r.get('trades',0)} win={r.get('win_rate',float('nan')):.3f} "
              f"mean={r.get('mean_return_bps',float('nan')):6.2f}bps "
              f"total={r.get('total_return_compounded',float('nan'))*100:7.2f}% "
              f"sharpe={r.get('sharpe_annualized') or float('nan'):.2f} "
              f"maxDD={r.get('max_drawdown',float('nan'))*100:.2f}%", flush=True)

    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")
    print("\nReference row from EQUITY_INSTRUMENT_SELECTION_FRAMEWORK.md (SPY shares, same signal, no leverage):")
    print("  spy_shares  trades=116  win=0.612  sharpe=3.76  maxDD=2.3%  total_return=+11.6%")
    print("\nNote: raw index-point return, no margin leverage applied -- see module docstring.")


if __name__ == "__main__":
    main()
