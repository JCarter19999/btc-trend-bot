"""Falsification test for the Short Strangle mechanism card
(MECHANISM_CARDS.md): mechanism claimed is "implied volatility priced into
the option overstates what subsequently realizes"; failure mode is "a real
catalyst materializes that the option wasn't overpricing after all." If the
worst-loss trades in the two completed strangle backtests cluster on
identifiable scheduled-catalyst days (FOMC, ECB, mega-cap earnings) rather
than being spread uniformly through the sample, that's direct, concrete
evidence the mechanism's own predicted failure mode is what's actually
firing -- not just "bad luck" -- and argues for a real macro-calendar
pre-trade exclusion filter as a next step.

Reuses the verified FOMC/ECB/earnings date lists already sourced in
`run_macro_calendar_decomposition_study.py` (MACRO_CALENDAR_DECOMPOSITION.md)
rather than re-deriving them. Read-only against the two completed strangle
backtest output directories -- does not touch the strangle backtest script,
implied_greeks.py, or thetadata_pricing.py.

Coverage caveat, stated plainly: FOMC dates are only verified from
2023-02-01 onward; ECB from 2023-2024 only. Both strangle backtests' trades
start 2021-06-17 (real ThetaData's own data floor), so a meaningful chunk of
early trades (2021-06 through 2023-01) cannot be checked against FOMC/ECB at
all -- reported as a real gap, not silently ignored or backfilled with a
guess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_macro_calendar_decomposition_study import ECB_DECISION_DATES, EARNINGS_DATES, FOMC_DECISION_DATES  # noqa: E402

FOMC_COVERAGE_START = pd.Timestamp("2023-02-01", tz="UTC")
ECB_COVERAGE_START = pd.Timestamp("2023-02-01", tz="UTC")
ECB_COVERAGE_END = pd.Timestamp("2024-12-31", tz="UTC")


def load_trades(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signal_time", "entry_time", "exit_time"])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["variant"] = label
    return df


def event_dates_index(dates: list[str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dates])


def touches_event(entry: pd.Timestamp, exit_: pd.Timestamp, events: pd.DatetimeIndex) -> bool:
    return bool(((events >= entry) & (events <= exit_)).any())


def coverage_ok(entry: pd.Timestamp, exit_: pd.Timestamp, cov_start: pd.Timestamp, cov_end: pd.Timestamp | None = None) -> bool:
    if entry < cov_start:
        return False
    if cov_end is not None and exit_ > cov_end:
        return False
    return True


def main() -> None:
    trend_slope = load_trades(ROOT / "outputs" / "short_strangle_chop_backtest" / "chop_gated_trades.csv", "trend_slope")
    realized_vol = load_trades(ROOT / "outputs" / "short_strangle_chop_backtest_realized_vol" / "chop_gated_trades.csv", "realized_vol")
    all_trades = pd.concat([trend_slope, realized_vol], ignore_index=True)

    print(f"Loaded {len(trend_slope)} trend_slope trades + {len(realized_vol)} realized_vol trades "
          f"= {len(all_trades)} total priced strangle trades")
    print(f"Date range: {all_trades['entry_time'].min().date()} to {all_trades['exit_time'].max().date()}")

    fomc_dates = event_dates_index(FOMC_DECISION_DATES)
    ecb_dates = event_dates_index(ECB_DECISION_DATES)
    earnings_dates = event_dates_index(EARNINGS_DATES)

    all_trades["fomc_coverage_ok"] = all_trades.apply(
        lambda r: coverage_ok(r["entry_time"], r["exit_time"], FOMC_COVERAGE_START), axis=1)
    all_trades["ecb_coverage_ok"] = all_trades.apply(
        lambda r: coverage_ok(r["entry_time"], r["exit_time"], ECB_COVERAGE_START, ECB_COVERAGE_END), axis=1)

    all_trades["touches_fomc"] = all_trades.apply(
        lambda r: touches_event(r["entry_time"], r["exit_time"], fomc_dates) if r["fomc_coverage_ok"] else None, axis=1)
    all_trades["touches_ecb"] = all_trades.apply(
        lambda r: touches_event(r["entry_time"], r["exit_time"], ecb_dates) if r["ecb_coverage_ok"] else None, axis=1)
    all_trades["touches_earnings"] = all_trades.apply(
        lambda r: touches_event(r["entry_time"], r["exit_time"], earnings_dates), axis=1)

    n_fomc_checkable = int(all_trades["fomc_coverage_ok"].sum())
    n_ecb_checkable = int(all_trades["ecb_coverage_ok"].sum())
    print(f"\nFOMC-checkable trades (entry >= 2023-02-01): {n_fomc_checkable}/{len(all_trades)}")
    print(f"ECB-checkable trades (2023-02-01 to 2024-12-31 window): {n_ecb_checkable}/{len(all_trades)}")

    worst5 = all_trades.nsmallest(5, "net_return")[
        ["variant", "signal_time", "entry_time", "exit_time", "net_return",
         "touches_fomc", "touches_ecb", "touches_earnings"]
    ]
    print("\n=== 5 worst-loss trades across both completed variants ===")
    print(worst5.to_string(index=False))

    worst10 = all_trades.nsmallest(10, "net_return")
    checkable_worst10_fomc = worst10[worst10["fomc_coverage_ok"]]
    rate_worst10_fomc = float(checkable_worst10_fomc["touches_fomc"].mean()) if len(checkable_worst10_fomc) else None
    checkable_all_fomc = all_trades[all_trades["fomc_coverage_ok"]]
    base_rate_fomc = float(checkable_all_fomc["touches_fomc"].mean()) if len(checkable_all_fomc) else None

    checkable_worst10_earn = worst10  # earnings has full coverage from 2021 on (list starts 2023-08, so still partial -- checked below)
    print(f"\nBase rate: fraction of ALL FOMC-checkable trades whose hold window touches an FOMC date: "
          f"{base_rate_fomc:.1%}" if base_rate_fomc is not None else "\nBase rate: N/A (no checkable trades)")
    print(f"Fraction of worst-10-loss trades (that are FOMC-checkable, n={len(checkable_worst10_fomc)}) "
          f"touching FOMC: {rate_worst10_fomc:.1%}" if rate_worst10_fomc is not None else "N/A")

    earnings_coverage_start = pd.Timestamp(min(EARNINGS_DATES), tz="UTC")
    all_trades["earnings_coverage_ok"] = all_trades["entry_time"] >= earnings_coverage_start
    checkable_all_earn = all_trades[all_trades["earnings_coverage_ok"]]
    base_rate_earn = float(checkable_all_earn["touches_earnings"].mean()) if len(checkable_all_earn) else None
    worst10_earn_checkable = worst10[worst10["entry_time"] >= earnings_coverage_start]
    rate_worst10_earn = float(worst10_earn_checkable["touches_earnings"].mean()) if len(worst10_earn_checkable) else None
    print(f"\nBase rate: fraction of ALL earnings-checkable trades touching a mega-cap earnings date: "
          f"{base_rate_earn:.1%}" if base_rate_earn is not None else "N/A")
    print(f"Fraction of worst-10-loss trades (earnings-checkable, n={len(worst10_earn_checkable)}) "
          f"touching earnings: {rate_worst10_earn:.1%}" if rate_worst10_earn is not None else "N/A")

    n_uncheckable_pre2023 = int((all_trades["entry_time"] < FOMC_COVERAGE_START).sum())
    print(f"\nTrades with entry before 2023-02-01 (uncheckable against FOMC/ECB at all): {n_uncheckable_pre2023}/{len(all_trades)}")

    out_dir = ROOT / "outputs" / "strangle_event_day_clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_trades.to_csv(out_dir / "trades_with_event_flags.csv", index=False)
    summary = {
        "n_total_trades": len(all_trades),
        "n_fomc_checkable": n_fomc_checkable,
        "n_ecb_checkable": n_ecb_checkable,
        "n_uncheckable_pre_2023_02": n_uncheckable_pre2023,
        "fomc_base_rate_all_checkable": base_rate_fomc,
        "fomc_rate_worst10": rate_worst10_fomc,
        "n_worst10_fomc_checkable": len(checkable_worst10_fomc),
        "earnings_base_rate_all_checkable": base_rate_earn,
        "earnings_rate_worst10": rate_worst10_earn,
        "n_worst10_earnings_checkable": len(worst10_earn_checkable),
        "worst5_trades": worst5.to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nWrote {out_dir / 'summary.json'} and {out_dir / 'trades_with_event_flags.csv'}")


if __name__ == "__main__":
    main()
