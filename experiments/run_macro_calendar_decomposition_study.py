"""Why does DAX predict SPY -- macro-calendar decomposition, the third
dimension of Joey's vertical-understanding pivot (sector and market-state
already done in EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md). Tests
whether the edge concentrates around scheduled macro events (Possibility
1: "Europe processes scheduled information first") or works broadly
regardless of the calendar (Possibility 2: "general risk-transmission
effect") -- distinguishing these was explicitly flagged as needing "a
reliable historical event calendar and careful timestamp handling," which
this project didn't have before (the original FOMC check in
EUROPEAN_LEAD_US_FIRST_HOUR_BACKTEST.txt logged its own honest gap:
"ECB meeting dates weren't checked with the same precision -- couldn't
get a complete verified date list").

Event dates sourced this session via direct web lookups against
authoritative/primary sources, not guessed or approximated:
- FOMC: federalreserve.gov official calendar. Full, high-confidence
  coverage 2023-2026 (decision = second day of each 2-day meeting).
- ECB: cross-referenced against ecb.europa.eu press-release URLs.
  High-confidence coverage for 2023-2024 only (single Thursday decision
  day each). 2025-2026 explicitly EXCLUDED, not guessed -- available
  sources were incomplete/inconsistent (a March 2025 meeting couldn't be
  confirmed, a "June 3-5" 2025 entry didn't match the normal single-day
  pattern, 2026 Jan-Jul dates weren't found at all). Same honest-gap
  discipline as the pre-existing FOMC note, now with much better (but
  still not 100%) coverage.
- Earnings weeks: real historical reported-earnings dates for AAPL/MSFT/
  NVDA via the Alpha Vantage EARNINGS endpoint (already validated in this
  project for real data, free tier), 2023-08 through 2026-05, 12
  quarters each. "Earnings week" = the ISO calendar week (Mon-Sun)
  containing any of these 36 real dates.
- Payrolls: US Employment Situation report, computed as the first Friday
  of each month -- the standard, publicly documented BLS convention.
  Not individually verified against BLS's calendar for rare
  holiday-driven exceptions; treated as a structural approximation, not
  an exact-verified list like FOMC/ECB/earnings.
- CPI: NOT included. Could not source a complete, reliably verified
  2023-2026 release calendar within reasonable effort (checked BLS,
  usinflationcalculator, investing.com, FRED/ALFRED, tradingeconomics --
  each either blocked the fetch or only covered a partial window). A
  real, disclosed gap, not a fabricated list.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_european_lead_us_first_hour_backtest import build_dataset, evaluate  # noqa: E402

FOMC_DECISION_DATES = [
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
]

# 2023-2024 only -- see module docstring for why 2025-2026 are excluded.
ECB_DECISION_DATES = [
    "2023-02-02", "2023-03-16", "2023-05-04", "2023-06-15", "2023-07-27", "2023-09-14", "2023-10-26", "2023-12-14",
    "2024-01-25", "2024-03-07", "2024-04-11", "2024-06-06", "2024-07-18", "2024-09-12", "2024-10-17", "2024-12-12",
]

EARNINGS_DATES = [
    # AAPL
    "2023-08-03", "2023-11-02", "2024-02-01", "2024-05-02", "2024-08-01", "2024-10-31",
    "2025-01-30", "2025-05-01", "2025-07-31", "2025-10-30", "2026-01-29", "2026-04-30",
    # MSFT
    "2023-07-25", "2023-10-24", "2024-01-30", "2024-04-25", "2024-07-30", "2024-10-30",
    "2025-01-29", "2025-04-30", "2025-07-30", "2025-10-29", "2026-01-28", "2026-04-29",
    # NVDA
    "2023-08-23", "2023-11-21", "2024-02-21", "2024-05-22", "2024-08-28", "2024-11-20",
    "2025-02-26", "2025-05-28", "2025-08-27", "2025-11-19", "2026-02-25", "2026-05-20",
]


def compare(df: pd.DataFrame, direction: pd.Series, event_mask: pd.Series, label: str) -> dict:
    on_event = direction.where(event_mask, 0)
    off_event = direction.where(~event_mask, 0)
    r_on = evaluate(df, on_event, 1.0, f"{label}_ON")
    r_off = evaluate(df, off_event, 1.0, f"{label}_OFF")
    print(f"\n{label}:")
    print(f"  ON  ({r_on.get('trades',0):3d} trades): win={r_on.get('win_rate',float('nan')):.3f} "
          f"mean={r_on.get('mean_return_bps',float('nan')):6.2f}bps sharpe={r_on.get('sharpe_annualized') or float('nan'):.2f}")
    print(f"  OFF ({r_off.get('trades',0):3d} trades): win={r_off.get('win_rate',float('nan')):.3f} "
          f"mean={r_off.get('mean_return_bps',float('nan')):6.2f}bps sharpe={r_off.get('sharpe_annualized') or float('nan'):.2f}")
    return {"on": r_on, "off": r_off}


def main() -> None:
    print("Building DAX signal...", flush=True)
    df = build_dataset()

    # Deliberately the UNGATED daily direction signal here (not the
    # top-quartile arm) -- matches the exact methodology of the original
    # FOMC check in EUROPEAN_LEAD_US_FIRST_HOUR_BACKTEST.txt ("FOMC
    # decision days (n=15)... Non-FOMC days (n=447)"), and necessary for
    # a usable sample size: gating first would leave only 1-16 trades in
    # the ON bucket for every event type here (tried it, discarded --
    # noise, not a result), since scheduled-event days and top-quartile-
    # move days are two independently rare conditions whose intersection
    # is too sparse to read anything from in a 2.7-year window.
    gated_direction = df["direction"]

    dates_idx = df.index

    fomc_set = set(pd.to_datetime(FOMC_DECISION_DATES))
    fomc_mask = pd.Series(dates_idx.isin(fomc_set), index=dates_idx)

    ecb_set = set(pd.to_datetime(ECB_DECISION_DATES))
    ecb_mask = pd.Series(dates_idx.isin(ecb_set), index=dates_idx)
    # 2025-2026 excluded from ECB dates -- restrict the comparison window
    # to 2023-2024 so "OFF" days aren't contaminated by an unchecked period.
    ecb_window = (dates_idx >= pd.Timestamp("2023-09-01")) & (dates_idx < pd.Timestamp("2025-01-01"))

    earnings_weeks = {pd.Timestamp(d).to_period("W-SUN") for d in EARNINGS_DATES}
    earnings_mask = pd.Series([dates_idx[i].to_period("W-SUN") in earnings_weeks for i in range(len(dates_idx))], index=dates_idx)

    payroll_dates = pd.date_range(dates_idx.min(), dates_idx.max() + pd.Timedelta(days=31), freq="MS")
    payroll_fridays = set()
    for month_start in payroll_dates:
        month_days = pd.date_range(month_start, month_start + pd.Timedelta(days=6), freq="D")
        fridays = [d for d in month_days if d.weekday() == 4]
        if fridays:
            payroll_fridays.add(fridays[0])
    payroll_mask = pd.Series(dates_idx.isin(payroll_fridays), index=dates_idx)

    print(f"Coverage: FOMC days in sample={fomc_mask.sum()}, ECB days in sample (2023-2024 window)="
          f"{(ecb_mask & pd.Series(ecb_window, index=dates_idx)).sum()}, earnings-week days={earnings_mask.sum()}, "
          f"payroll-Friday days={payroll_mask.sum()}", flush=True)

    results = {}
    results["fomc"] = compare(df, gated_direction, fomc_mask, "FOMC decision days")

    ecb_df = df[ecb_window]
    ecb_direction = gated_direction[ecb_window]
    ecb_mask_windowed = ecb_mask[ecb_window]
    results["ecb"] = compare(ecb_df, ecb_direction, ecb_mask_windowed, "ECB decision days (2023-2024 only, see docstring)")

    results["earnings_weeks"] = compare(df, gated_direction, earnings_mask, "Mega-cap (AAPL/MSFT/NVDA) earnings weeks")
    results["payroll_fridays"] = compare(df, gated_direction, payroll_mask, "Nonfarm payroll Fridays")

    out = ROOT / "outputs" / "macro_calendar_decomposition_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")
    print("\nCPI not tested -- could not source a reliable complete release calendar (see module docstring).")


if __name__ == "__main__":
    main()
