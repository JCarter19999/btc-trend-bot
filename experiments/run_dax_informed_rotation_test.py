"""Does the already-validated DAX cross-market signal sharpen simple_trend's
stock rotation? Two versions tested:
(1) joint-signal-day effect: does simple_trend perform better on days its
    signal_time also had a DAX top-quartile move?
(2) does DAX's direction that morning correlate with the leader stock's
    subsequent performance, at a larger sample (every trading day, not just
    top-quartile days)?

No-lookahead check, stated explicitly: DAX's pre-open reading is known by
13:30 UTC (US market open) on date T. simple_trend's own signal for date T
is generated from date T's CLOSE (~21:00 UTC), entered at T+1's open. So
DAX's same-day (date T) reading is legitimately known hours before
simple_trend's own signal-generation moment on that same date -- joining on
the same calendar date T is not lookahead.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from run_equity_real_data_walkforward import (  # noqa: E402
    build_candidates, load_config, load_csv_dir, _select_fold_winners,
)
from run_european_lead_us_first_hour_backtest import build_dataset  # noqa: E402
from btc_trend_bot.portfolio_sim import simulate_single_position, SizingMode  # noqa: E402

N_NULL_SEEDS = 1000


def shuffled_null(labels: np.ndarray, values: np.ndarray, real_stat: float, seeds: int = N_NULL_SEEDS):
    stats = []
    n = len(labels)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(labels)
        if shuffled.sum() in (0, n):
            continue
        stats.append(values[shuffled].mean() - values[~shuffled].mean())
    stats = np.array(stats)
    pct = float((stats < real_stat).mean() * 100) if len(stats) else float("nan")
    return float(stats.mean()), float(stats.std()), pct


def main() -> None:
    cfg = load_config(ROOT / "configs" / "real_data.yaml")
    cfg = dataclasses.replace(cfg, stop_atr=100.0, target_atr=100.0)
    all_symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(all_symbols, ROOT / "data" / "real")
    candidates = build_candidates(frames, cfg.benchmark, cfg)

    rng = np.random.default_rng(0)
    winners = _select_fold_winners(candidates, candidates, cfg, 0, rng, False, "simple_trend")
    winners = winners.dropna(subset=["net_return"]).sort_values("signal_time").reset_index(drop=True)
    sizing = SizingMode(name="fixed_notional_2500", kind="fixed_notional", value=cfg.initial_capital)
    trade_path, summary = simulate_single_position(winners, cfg, sizing)
    taken = trade_path[trade_path.trade_taken == True].copy()  # noqa: E712
    print(f"Baseline reconstruction: {len(taken)} trades, win_rate={ (taken.trade_return>0).mean():.3f}, "
          f"total_return={summary['total_return']*100:.1f}%")
    assert len(taken) == 188, f"Expected 188 trades, got {len(taken)} -- reconstruction methodology mismatch"
    print("Confirmed exact match to documented 188-trade baseline before proceeding.\n")

    taken["signal_date"] = pd.to_datetime(taken["signal_time"]).dt.tz_localize(None).dt.normalize()

    print("Building DAX dataset (real ThetaData-validated cross-market signal, ~2023-09 to 2026-07 coverage)...")
    dax = build_dataset()
    dax = dax.copy()
    dax.index = pd.to_datetime(dax.index).tz_localize(None).normalize()
    abs_eu = dax["eu_pre_open_return"].abs()
    dax["top_quartile"] = abs_eu >= abs_eu.quantile(0.75)
    print(f"DAX dataset: {len(dax)} days, {dax.index.min().date()} to {dax.index.max().date()}, "
          f"{dax['top_quartile'].sum()} top-quartile days\n")

    # ---- Hypothesis 1: joint-signal-day effect on simple_trend's real trades ----
    print("=" * 70)
    print("HYPOTHESIS 1: does simple_trend perform better on DAX top-quartile days?")
    print("=" * 70)
    merged = taken.merge(dax[["top_quartile", "direction"]], left_on="signal_date", right_index=True, how="left")
    covered = merged[merged["top_quartile"].notna()].copy()
    print(f"simple_trend trades within DAX-data-covered window: {len(covered)} of {len(taken)} total "
          f"({len(covered)/len(taken)*100:.1f}%) -- DAX intraday data only goes back to "
          f"{dax.index.min().date()}, well after simple_trend's 2018 start.")

    n_joint = int(covered["top_quartile"].sum())
    n_other = len(covered) - n_joint
    print(f"Joint (DAX top-quartile) days: {n_joint}   Other covered days: {n_other}")
    if n_joint < 10:
        print("TOO THIN TO TRUST (n<10) -- stating plainly rather than forcing a conclusion. "
              "This mirrors the regime classifier's earlier ~25-27-day joint-bucket thinness.")
        h1_result = {"n_joint": n_joint, "n_other": n_other, "verdict": "too_thin_to_trust"}
    else:
        mean_joint = covered.loc[covered["top_quartile"], "trade_return"].mean()
        mean_other = covered.loc[~covered["top_quartile"], "trade_return"].mean()
        diff = mean_joint - mean_other
        null_mean, null_std, pct = shuffled_null(
            covered["top_quartile"].to_numpy(), covered["trade_return"].to_numpy(), diff)
        print(f"mean return | joint=True: {mean_joint*100:.2f}%   | joint=False: {mean_other*100:.2f}%")
        print(f"real diff: {diff*100:+.2f}pp   null_mean={null_mean*100:+.2f}pp null_std={null_std*100:.2f}pp "
              f"percentile={pct:.1f}")
        h1_result = {"n_joint": n_joint, "n_other": n_other, "mean_joint": mean_joint, "mean_other": mean_other,
                     "diff": diff, "null_mean": null_mean, "null_std": null_std, "null_percentile": pct}

    # ---- Hypothesis 2: does DAX direction correlate with the leader's subsequent return? ----
    # Larger sample: every trading day with a DAX reading, not just top-quartile days,
    # joined against the RAW candidate pool (every day's simple_trend pick and its
    # net_return), not just the 188 single-position-filtered trades -- more statistical
    # power for this specific question, at the cost of the single-position realism
    # (stated explicitly, same caveat pattern used elsewhere this session).
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: does DAX direction correlate with the leader's subsequent return?")
    print("=" * 70)
    raw_winners = _select_fold_winners(candidates, candidates, cfg, 0, rng, False, "simple_trend")
    raw_winners = raw_winners.dropna(subset=["net_return"]).copy()
    raw_winners["signal_date"] = pd.to_datetime(raw_winners["signal_time"]).dt.tz_localize(None).dt.normalize()
    raw_merged = raw_winners.merge(dax[["direction", "eu_pre_open_return"]], left_on="signal_date",
                                    right_index=True, how="left")
    raw_covered = raw_merged[raw_merged["direction"].notna()].copy()
    print(f"Raw daily candidates within DAX-covered window: {len(raw_covered)} of {len(raw_winners)} total")

    bullish = raw_covered[raw_covered["direction"] > 0]
    bearish = raw_covered[raw_covered["direction"] < 0]
    print(f"DAX-bullish mornings: n={len(bullish)}, mean simple_trend leader return={bullish['net_return'].mean()*100:.3f}%")
    print(f"DAX-bearish mornings: n={len(bearish)}, mean simple_trend leader return={bearish['net_return'].mean()*100:.3f}%")
    diff2 = bullish["net_return"].mean() - bearish["net_return"].mean()
    labels2 = (raw_covered["direction"] > 0).to_numpy()
    null_mean2, null_std2, pct2 = shuffled_null(labels2, raw_covered["net_return"].to_numpy(), diff2)
    print(f"real diff (bullish - bearish): {diff2*100:+.2f}pp   null_mean={null_mean2*100:+.2f}pp "
          f"null_std={null_std2*100:.2f}pp percentile={pct2:.1f}")

    # OOS split for hypothesis 2 (larger sample permits it)
    mid = raw_covered["signal_date"].median()
    first = raw_covered[raw_covered["signal_date"] <= mid]
    second = raw_covered[raw_covered["signal_date"] > mid]
    for half_name, half in (("first half", first), ("second half", second)):
        b = half[half["direction"] > 0]["net_return"]
        be = half[half["direction"] < 0]["net_return"]
        d = b.mean() - be.mean() if len(b) and len(be) else float("nan")
        print(f"  {half_name}: n={len(half)}, bullish_mean={b.mean()*100 if len(b) else float('nan'):.2f}%, "
              f"bearish_mean={be.mean()*100 if len(be) else float('nan'):.2f}%, diff={d*100:+.2f}pp")

    # Actionable check: does filtering simple_trend's real 188 trades to DAX-bullish-or-no-signal
    # mornings only (dropping DAX-bearish-morning entries) change total return/drawdown?
    print("\n--- Actionable filter check: drop simple_trend entries on DAX-bearish mornings ---")
    covered_all = taken.merge(dax[["direction"]], left_on="signal_date", right_index=True, how="left")
    bearish_days = covered_all[covered_all["direction"] < 0]
    print(f"Of 188 real trades, {len(bearish_days)} fell on a DAX-bearish morning "
          f"({len(bearish_days)/len(taken)*100:.1f}% of trades, within the DAX-covered subset)")
    if len(bearish_days) >= 3:
        filtered = taken[~taken.index.isin(bearish_days.index)]
        # Fixed-notional sizing (not compounding): total_return = sum(trade_return),
        # confirmed to match simulate_single_position's own summary['total_return']
        # exactly (both 4.574... for the full 188). NOT (1+r).prod()-1, which would
        # be the compounding convention this project does NOT use for this baseline.
        orig_total = taken["trade_return"].sum()
        filt_total = filtered["trade_return"].sum()
        print(f"Original 188-trade total_return (fixed-notional, sum): {orig_total*100:.1f}%")
        print(f"Filtered ({len(filtered)}-trade) total_return (fixed-notional, sum): {filt_total*100:.1f}%")

    out_dir = ROOT / "outputs" / "dax_informed_rotation_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    taken.to_csv(out_dir / "simple_trend_188_with_dax.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "hypothesis_1_joint_signal_day": h1_result,
            "hypothesis_2": {
                "n_covered": len(raw_covered), "n_bullish": len(bullish), "n_bearish": len(bearish),
                "diff": diff2, "null_mean": null_mean2, "null_std": null_std2, "null_percentile": pct2,
            },
        }, f, indent=2, default=str)
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
