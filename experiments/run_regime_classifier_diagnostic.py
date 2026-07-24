"""Validation harness for src/btc_trend_bot/regime_classifier.py -- the first
piece of a planned strategy-rotation system (rotate between independently-
validated strategies depending on market regime). This is a scaffold
diagnostic, not a rotation backtest: it only checks that the classifier's
two wired-in regime flags (TRENDING, DAX_SIGNAL_DAY) actually track what
they claim to, using real historical data already cached in this repo.
CHOPPY_SIDELINED is left as an unvalidated placeholder pending the separate
short-strangle chop backtest.

Checks performed:
1. Base-rate coverage per bucket (TRENDING / DAX_SIGNAL_DAY / overlap /
   CHOPPY_SIDELINED), noting DAX_SIGNAL_DAY's shorter data window.
2. Does TRENDING actually track simple_trend's real per-trade performance?
   Runs the real walk-forward simple_trend selection over 2018-2026 cached
   daily data, joins each selected trade's signal date to the classifier's
   trending flag, and compares mean net_return trending vs. non-trending --
   checked against a shuffled-label null (permute the flag across dates),
   same discipline as every other finding in this project.
3. Does DAX_SIGNAL_DAY's narrower (quartile + calm) subset behave as
   expected relative to the full validated quartile-gated 116-trade sample?
   Reports trade counts and performance for both, explicit that the narrower
   subset's profitability is NOT independently re-validated here (see the
   module docstring's caveat) -- only that its construction is a proper,
   no-lookahead subset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.regime_classifier import classify_regimes  # noqa: E402

from run_equity_real_data_walkforward import (  # noqa: E402
    build_candidates, load_config, load_csv_dir, walk_forward,
)
from run_european_lead_us_first_hour_backtest import build_dataset, evaluate  # noqa: E402

N_NULL_SEEDS = 1000


def shuffled_null_percentile(labels: np.ndarray, values: np.ndarray, real_stat: float, seeds: int = N_NULL_SEEDS):
    rng_stats = []
    n = len(labels)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(labels)
        rng_stats.append(values[shuffled].mean() - values[~shuffled].mean() if shuffled.sum() not in (0, n) else np.nan)
    rng_stats = np.array([s for s in rng_stats if np.isfinite(s)])
    pct = float((rng_stats < real_stat).mean() * 100) if len(rng_stats) else float("nan")
    return float(np.nanmean(rng_stats)), float(np.nanstd(rng_stats)), pct


def check_trending_flag(regimes: pd.DataFrame) -> dict:
    print("\n=== Check 1: does TRENDING track simple_trend's real performance? ===")
    cfg = load_config(ROOT / "configs" / "real_data.yaml")
    all_symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(all_symbols, ROOT / "data" / "real")
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    _, winners = walk_forward(candidates, cfg, selection="simple_trend")
    winners = winners.copy()
    winners["signal_date"] = pd.to_datetime(winners["signal_time"]).dt.tz_localize(None).dt.normalize()

    trending_by_date = regimes["trending"]
    winners["trending"] = winners["signal_date"].map(trending_by_date).fillna(False)

    n_trending = int(winners["trending"].sum())
    n_not = int((~winners["trending"]).sum())
    mean_trending = float(winners.loc[winners["trending"], "net_return"].mean()) if n_trending else float("nan")
    mean_not = float(winners.loc[~winners["trending"], "net_return"].mean()) if n_not else float("nan")

    real_diff = mean_trending - mean_not
    null_mean, null_std, pct = shuffled_null_percentile(
        winners["trending"].to_numpy(), winners["net_return"].to_numpy(), real_diff
    )

    print(f"simple_trend trades total: {len(winners)}  (trending={n_trending}, non-trending={n_not})")
    print(f"mean net_return | TRENDING=True:  {mean_trending*100:6.2f}%")
    print(f"mean net_return | TRENDING=False: {mean_not*100:6.2f}%")
    print(f"real diff: {real_diff*100:+.2f}pp  null_mean={null_mean*100:+.2f}pp null_std={null_std*100:.2f}pp "
          f"percentile={pct:.0f}")

    return {
        "n_trades": len(winners), "n_trending": n_trending, "n_not_trending": n_not,
        "mean_return_trending": mean_trending, "mean_return_not_trending": mean_not,
        "real_diff": real_diff, "null_mean": null_mean, "null_std": null_std, "null_percentile": pct,
    }


def check_dax_signal_flag(regimes: pd.DataFrame, dax_diag: dict) -> dict:
    print("\n=== Check 2: DAX_SIGNAL_DAY (quartile+calm) vs. full validated quartile gate ===")
    df = build_dataset()
    abs_dax = df["eu_pre_open_return"].abs()
    full_gate_direction = df["direction"].where(abs_dax >= abs_dax.quantile(0.75), 0)
    full_gate_dates = set(df.index[full_gate_direction != 0])

    narrow_dates = set(pd.DatetimeIndex(regimes.index[regimes["dax_signal_day"]]))
    subset_ok = narrow_dates.issubset(full_gate_dates)
    print(f"Full validated quartile-gate days: {len(full_gate_dates)}")
    print(f"DAX_SIGNAL_DAY (quartile+calm) days: {len(narrow_dates)}")
    print(f"DAX_SIGNAL_DAY is a proper subset of the validated gate: {subset_ok}")

    narrow_direction = df["direction"].where(df.index.isin(narrow_dates), 0)
    r_full = evaluate(df, full_gate_direction, 2.0, "full_quartile_gate")
    r_narrow = evaluate(df, narrow_direction, 2.0, "quartile_and_calm_subset")
    print(f"Full gate   : trades={r_full.get('trades')} win={r_full.get('win_rate'):.3f} "
          f"sharpe={r_full.get('sharpe_annualized'):.2f} total={r_full.get('total_return_compounded')*100:.1f}%")
    print(f"Narrow (DAX_SIGNAL_DAY): trades={r_narrow.get('trades')} win={r_narrow.get('win_rate', float('nan')):.3f} "
          f"sharpe={r_narrow.get('sharpe_annualized') or float('nan'):.2f} "
          f"total={r_narrow.get('total_return_compounded', float('nan'))*100:.1f}%")
    print("NOTE: this is the underlying SPY-first-hour STOCK signal, not the real-ThetaData options "
          "trade -- the narrow subset's options profitability has not been independently re-tested "
          "with real quotes (see module docstring caveat).")

    return {
        "dax_coverage": dax_diag,
        "full_gate_days": len(full_gate_dates),
        "narrow_days": len(narrow_dates),
        "narrow_is_proper_subset": subset_ok,
        "full_gate_result": r_full,
        "narrow_result": r_narrow,
    }


def main() -> None:
    cfg = load_config(ROOT / "configs" / "real_data.yaml")
    frames = load_csv_dir(cfg.symbols, ROOT / "data" / "real")
    # benchmark (SPY) must also be present for add_features
    if cfg.benchmark not in frames:
        frames = {**frames, cfg.benchmark: load_csv_dir((cfg.benchmark,), ROOT / "data" / "real")[cfg.benchmark]}

    print("Classifying regimes over cached real daily data (2018-2026 for TRENDING; "
          "~2023-09+ for DAX_SIGNAL_DAY, limited by yfinance intraday history)...")
    regimes, dax_diag = classify_regimes(frames, cfg.benchmark)

    print("\n=== Base-rate coverage ===")
    print(regimes["primary_bucket"].value_counts())
    print(f"\nTotal classified days: {len(regimes)}")
    print(f"Days with DAX_SIGNAL_DAY data coverage: {int(regimes['dax_coverage'].sum())} "
          f"({regimes['dax_coverage'].mean()*100:.1f}% of total)")
    print("\nWithin DAX-data-covered window only:")
    covered = regimes[regimes["dax_coverage"]]
    print(covered["primary_bucket"].value_counts())

    trend_check = check_trending_flag(regimes)
    dax_check = check_dax_signal_flag(regimes, dax_diag)

    out_dir = ROOT / "outputs" / "regime_classifier_scaffold"
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes.to_csv(out_dir / "classified_days.csv")
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(
            {
                "bucket_counts_full_history": regimes["primary_bucket"].value_counts().to_dict(),
                "bucket_counts_dax_covered_window": covered["primary_bucket"].value_counts().to_dict(),
                "trend_flag_check": trend_check,
                "dax_signal_flag_check": dax_check,
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"\nWrote {out_dir / 'classified_days.csv'} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
