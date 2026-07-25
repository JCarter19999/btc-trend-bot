"""Falsification test for the DAX-Lead mechanism card (MECHANISM_CARDS.md):
mechanism claimed is delayed international information transmission (DAX
pre-open move predicts SPY's first-hour reaction before the US market has
processed the same information). If that delay is closing over time (more
algorithmic cross-market arbitrage, faster information flow), the RAW
statistical relationship itself -- not just the options P&L already shown to
decay out-of-sample -- should shrink across sub-periods.

This is deliberately independent of EQUITY_OPTIONS_REAL_DATA_RETEST.md's
out-of-sample split, which only covers 2021-06+ (real ThetaData's own data
floor) and measures options P&L (entangled with option pricing/liquidity
changes, not just the underlying statistical edge). This test uses the full
`build_dataset()` history (no ThetaData dependency) and looks at the
correlation/direction-accuracy of the raw signal itself, split into
sub-periods -- distinguishing "the mechanism itself is decaying" from "the
options market got more efficient at pricing it" as two different possible
explanations for the options-side decay already documented.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_european_lead_us_first_hour_backtest import build_dataset, evaluate, random_control  # noqa: E402

N_RANDOM_SEEDS = 1000


def half_split_stats(df: pd.DataFrame) -> dict:
    abs_eu = df["eu_pre_open_return"].abs()
    top_quartile_cutoff = abs_eu.quantile(0.75)
    gated = df["direction"].where(abs_eu >= top_quartile_cutoff, 0)
    gated_df = df.assign(direction=gated)
    traded = gated_df[gated_df["direction"] != 0].copy()

    n = len(traded)
    half = n // 2
    first_half = traded.iloc[:half]
    second_half = traded.iloc[half:]

    def stats(sub: pd.DataFrame, label: str) -> dict:
        corr = float(np.corrcoef(sub["eu_pre_open_return"], sub["us_first_hour_return"])[0, 1])
        direction_acc = float((np.sign(sub["eu_pre_open_return"]) == np.sign(sub["us_first_hour_return"])).mean())
        return {
            "label": label, "n": len(sub),
            "date_range": f"{sub.index.min().date()} to {sub.index.max().date()}",
            "raw_corr_eu_vs_us_return": corr,
            "direction_accuracy": direction_acc,
            "mean_us_first_hour_return_bps": float(sub["us_first_hour_return"].mean() * 10000),
        }

    return {
        "full_gated_n": n,
        "first_half": stats(first_half, "first_half_top_quartile"),
        "second_half": stats(second_half, "second_half_top_quartile"),
        "full_ungated_corr": float(np.corrcoef(df["eu_pre_open_return"], df["us_first_hour_return"])[0, 1]),
    }


def year_by_year(df: pd.DataFrame) -> list[dict]:
    abs_eu = df["eu_pre_open_return"].abs()
    top_quartile_cutoff = abs_eu.quantile(0.75)
    gated = df["direction"].where(abs_eu >= top_quartile_cutoff, 0)
    gated_df = df.assign(direction=gated)
    traded = gated_df[gated_df["direction"] != 0].copy()
    traded["year"] = traded.index.year

    rows = []
    for yr, sub in traded.groupby("year"):
        if len(sub) < 5:
            continue
        corr = float(np.corrcoef(sub["eu_pre_open_return"], sub["us_first_hour_return"])[0, 1]) if len(sub) > 2 else None
        direction_acc = float((np.sign(sub["eu_pre_open_return"]) == np.sign(sub["us_first_hour_return"])).mean())
        rows.append({
            "year": int(yr), "n": len(sub),
            "raw_corr": corr, "direction_accuracy": direction_acc,
            "mean_us_first_hour_return_bps": float(sub["us_first_hour_return"].mean() * 10000),
        })
    return rows


def main() -> None:
    print("Building dataset (DAX + SPY hourly bars, full available history)...")
    df = build_dataset()
    print(f"Full dataset: {len(df)} days, {df.index.min().date()} to {df.index.max().date()}")

    half = half_split_stats(df)
    print("\n=== Half-split: raw signal (top-quartile gated), independent of options P&L ===")
    print(json.dumps(half, indent=2, default=str))

    yearly = year_by_year(df)
    print("\n=== Year-by-year: raw signal (top-quartile gated) ===")
    for row in yearly:
        print(f"{row['year']}: n={row['n']:3d} corr={row['raw_corr']:+.3f} "
              f"dir_acc={row['direction_accuracy']:.3f} mean_ret={row['mean_us_first_hour_return_bps']:+.1f}bps")

    out_dir = ROOT / "outputs" / "dax_lag_decay_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump({"half_split": half, "year_by_year": yearly,
                    "full_history_range": f"{df.index.min().date()} to {df.index.max().date()}"},
                   fh, indent=2, default=str)
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
