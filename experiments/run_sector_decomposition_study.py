"""Why does DAX predict SPY -- sector decomposition, per Joey's vertical-
understanding pivot (2026-07-24): instead of hunting more predictor
markets, understand which part of the US market the DAX signal's
information actually transmits to. Same validated signal (DAX pre-US-open
return, top-quartile |move| gate -- exactly what's live in the shadow
deployment), same first-hour target window, only the traded instrument
changes across the 10 SPDR sector ETFs (XLK/XLF/XLI/XLE/XLY/XLP/XLV/XLB/
XLU/XLC). Reuses `analyze()` (correlation + shuffle significance) and
`evaluate()` (Sharpe/win/return backtest) unchanged from the validated
SPY study/backtest -- this is instrument substitution, not a new model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_european_lead_us_first_hour_backtest import build_dataset, evaluate  # noqa: E402
from run_european_lead_us_first_hour_study import download_hourly, us_first_hour_return  # noqa: E402
from run_taiwan_semiconductor_lead_lag_study import analyze  # noqa: E402

SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLI": "Industrials", "XLE": "Energy",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLV": "Health Care",
    "XLB": "Materials", "XLU": "Utilities", "XLC": "Communication Services",
}


def main() -> None:
    print("Building DAX signal (validated, same as live shadow deployment)...", flush=True)
    df = build_dataset()
    abs_dax = df["eu_pre_open_return"].abs()
    gated_direction = df["direction"].where(abs_dax >= abs_dax.quantile(0.75), 0)

    corr_results = {}
    backtest_results = {}
    for ticker, name in SECTORS.items():
        print(f"  {ticker} ({name})...", flush=True)
        sector_df = download_hourly(ticker)
        sector_target = us_first_hour_return(sector_df)
        sector_target.index = pd.to_datetime(sector_target.index)

        corr_results[ticker] = analyze(df["eu_pre_open_return"], sector_target, ticker)

        joined = pd.concat([sector_target.rename("us_first_hour_return")], axis=1)
        joined = joined.reindex(df.index)
        temp_df = df[["eu_pre_open_return"]].copy()
        temp_df["us_first_hour_return"] = joined["us_first_hour_return"]
        r = evaluate(temp_df, gated_direction, 1.0, f"{ticker}_DAX_top_quartile_gate")
        backtest_results[ticker] = r

    print("\n" + "=" * 90)
    print("CORRELATION (all 462 days, direction/magnitude vs. shuffle)")
    print("=" * 90)
    for ticker, r in sorted(corr_results.items(), key=lambda kv: -(kv[1].get("directional_correlation") or -99)):
        print(f"{ticker:5s} ({SECTORS[ticker]:24s}) dir_corr={r.get('directional_correlation'):+.3f} "
              f"(pctile={r.get('directional_corr_percentile_vs_shuffled'):5.1f})  "
              f"mag_corr={r.get('magnitude_correlation'):+.3f} (pctile={r.get('magnitude_corr_percentile_vs_shuffled'):5.1f})")

    print("\n" + "=" * 90)
    print("BACKTEST: DAX top-quartile-gated direction trade, applied per sector (1bp cost)")
    print("=" * 90)
    for ticker, r in sorted(backtest_results.items(), key=lambda kv: -(kv[1].get("sharpe_annualized") or -99)):
        print(f"{ticker:5s} ({SECTORS[ticker]:24s}) trades={r.get('trades',0):4d} win={r.get('win_rate',float('nan')):.3f} "
              f"sharpe={r.get('sharpe_annualized') or float('nan'):6.2f} "
              f"total={r.get('total_return_compounded',float('nan'))*100:7.2f}% "
              f"maxDD={r.get('max_drawdown',float('nan'))*100:.2f}%")

    print(f"\nReference (SPY, the original validated target): sharpe=4.16, total=+12.9% (from prior backtest)")

    out = ROOT / "outputs" / "sector_decomposition_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"correlation": corr_results, "backtest": backtest_results}, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
