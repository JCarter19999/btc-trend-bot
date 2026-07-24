"""Turns tonight's recurring pattern -- every cross-market pair tested
(DAX, Taiwan, EURUSD, USDJPY) shows a magnitude correlation with SPY's
first-hour move at the 100th percentile vs. shuffled control, while
DIRECTIONAL correlation is real only for DAX and inconsistent/weak
everywhere else -- into an actual, testable strategy modification, not
just an observation.

Joey's framing: the magnitude-clustering pattern isn't edge by itself
(it doesn't say which way to trade) -- it's a candidate FILTER on the one
directional edge already validated (DAX's direction). This mirrors
exactly the shape of the one thing in this project that already worked
this way: DAX-direction gated by Asia's magnitude improved Sharpe from
1.76 to 2.56 (EUROPEAN_LEAD_ASIAN_MARKETS_AND_OPTIONS_OVERLAY.txt). This
generalizes that from one gating market (Asia) to a composite of every
independent magnitude signal found tonight (DAX's own move, Taiwan,
EURUSD, USDJPY), and tests whether the broader composite beats both the
no-gate baseline AND the DAX-only-quartile gate already in the live
shadow deployment.

Composite construction: full-sample percentile rank of each market's
|pre-US-open move| (consistent with how build_dataset()'s own DAX top-
quartile filter is computed -- this is a backtest, not the live shadow
system, which separately uses an expanding/no-lookahead percentile
bootstrap for the same reason results should be viewed as a backtest,
not a live-tradeable number yet). Missing days (a market's data doesn't
cover that date) are skipped per-column, not zero-filled, so the
composite is the mean of whichever signals are actually available that
day.

Also tests a composite EXCLUDING DAX's own magnitude (Taiwan+EURUSD+
USDJPY only) -- isolates whether genuinely independent markets add
anything beyond what DAX's own move size already implies, since DAX's
own magnitude gate is already a known-good baseline on its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_european_lead_us_first_hour_backtest import (  # noqa: E402
    ROUND_TRIP_COSTS_BPS, N_RANDOM_SEEDS, build_dataset, evaluate, random_control,
)
from run_european_lead_us_first_hour_study import (  # noqa: E402
    download_hourly, european_pre_us_open_return,
)


def magnitude_series(ticker: str) -> pd.Series:
    s = european_pre_us_open_return(download_hourly(ticker)).abs()
    s.index = pd.to_datetime(s.index)
    return s


def main() -> None:
    print("Building DAX signal + SPY target (validated baseline)...", flush=True)
    df = build_dataset()
    abs_dax = df["eu_pre_open_return"].abs()

    print("Building Taiwan/EURUSD/USDJPY magnitude series...", flush=True)
    twii_mag = magnitude_series("^TWII")
    eurusd_mag = magnitude_series("EURUSD=X")
    usdjpy_mag = magnitude_series("USDJPY=X")

    combo = pd.DataFrame({
        "dax": abs_dax,
        "twii": twii_mag.reindex(df.index),
        "eurusd": eurusd_mag.reindex(df.index),
        "usdjpy": usdjpy_mag.reindex(df.index),
    })
    ranks = combo.rank(pct=True)  # full-sample percentile rank per column, NaN-safe
    full_composite = ranks.mean(axis=1, skipna=True)
    non_dax_composite = ranks[["twii", "eurusd", "usdjpy"]].mean(axis=1, skipna=True)

    print(f"Coverage: DAX={combo['dax'].notna().sum()}, TWII={combo['twii'].notna().sum()}, "
          f"EURUSD={combo['eurusd'].notna().sum()}, USDJPY={combo['usdjpy'].notna().sum()}, "
          f"full_composite={full_composite.notna().sum()}, non_dax_composite={non_dax_composite.notna().sum()}", flush=True)

    variants = {
        "baseline_trade_every_day": df["direction"],
        "baseline_DAX_top_quartile_gate": df["direction"].where(abs_dax >= abs_dax.quantile(0.75), 0),
        "NEW_multi_market_composite_top_quartile_gate": df["direction"].where(
            full_composite >= full_composite.quantile(0.75), 0),
        "NEW_non_DAX_composite_top_quartile_gate": df["direction"].where(
            non_dax_composite >= non_dax_composite.quantile(0.75), 0),
    }

    all_results = {}
    for cost_bps in ROUND_TRIP_COSTS_BPS:
        print(f"\n{'='*70}\ncost = {cost_bps} bps round-trip\n{'='*70}")
        for name, direction in variants.items():
            r = evaluate(df, direction, cost_bps, name)
            print(f"{name:48s} trades={r.get('trades',0):4d} win={r.get('win_rate',float('nan')):.3f} "
                  f"mean={r.get('mean_return_bps',float('nan')):6.2f}bps "
                  f"total={r.get('total_return_compounded',float('nan'))*100:7.2f}% "
                  f"sharpe={r.get('sharpe_annualized') or float('nan'):.2f} "
                  f"maxDD={r.get('max_drawdown',float('nan'))*100:.2f}%")
            all_results.setdefault(name, {})[cost_bps] = r

        real_mean = all_results["NEW_multi_market_composite_top_quartile_gate"][cost_bps]["mean_return_bps"]
        gated_dates = df.loc[variants["NEW_multi_market_composite_top_quartile_gate"] != 0].index
        random_means = random_control(df, df.loc[gated_dates, "direction"], cost_bps, N_RANDOM_SEEDS)
        pct = float((np.array(random_means) < real_mean).mean() * 100)
        print(f"\nMulti-market-gated mean ({real_mean:.2f}bps) vs. {N_RANDOM_SEEDS} random-direction seeds "
              f"(same trade count/dates, mean={np.mean(random_means):.2f}bps): percentile={pct:.1f}")
        all_results.setdefault("random_control_percentile_multi_market_gate", {})[cost_bps] = pct

    out = ROOT / "outputs" / "multi_market_magnitude_gate_backtest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
