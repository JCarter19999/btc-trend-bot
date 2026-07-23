"""Turns the DAX/FTSE -> SPY first-hour lead-lag finding
(EUROPEAN_LEAD_US_FIRST_HOUR_STUDY.txt) into an actual backtest: real
entry/exit rules, real cost sensitivity, and the same random/shuffle
controls used everywhere else in this project -- the raw correlation study
measured a statistical relationship, not a strategy.

Strategy: at US open (13:30 UTC), go long SPY if that morning's DAX
pre-open return was positive, short if negative, flat if exactly zero.
Exit at the end of the first hour (14:30 UTC) -- matching exactly the bar
the lead-lag study measured, so this backtest can't quietly redefine the
signal to look better. No lookahead: the DAX signal only uses DAX bars
that closed before 13:30 UTC.

Unlike this project's multi-day equity holds (which had a real overlap
bug fixed in portfolio_sim.py), each trade here opens and closes the same
day -- structurally impossible to overlap, so a plain daily return series
and honest compounding equity curve are both valid without needing that
machinery.

Cost/spread caveat, stated once here: no real historical SPY quote data
(bid/ask) is available for free -- same constraint as every other cost
assumption in this project. SPY is one of the most liquid instruments in
the world, so a tight assumption is defensible, but tested across a range
(1/2/5 bps round-trip) rather than picking one number, and the open print
specifically tends to have a wider effective spread than mid-day (price
discovery), which the wider end of the range is meant to cover.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_european_lead_us_first_hour_study import (  # noqa: E402
    download_hourly, european_pre_us_open_return, us_first_hour_return,
)

ROUND_TRIP_COSTS_BPS = (1.0, 2.0, 5.0)
N_RANDOM_SEEDS = 1000


def build_dataset() -> pd.DataFrame:
    dax = download_hourly("^GDAXI")
    spy = download_hourly("SPY")
    eu_signal = european_pre_us_open_return(dax)
    us_target = us_first_hour_return(spy)
    df = pd.concat([eu_signal, us_target], axis=1).dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df["direction"] = np.sign(df["eu_pre_open_return"])
    return df


def evaluate(df: pd.DataFrame, direction: pd.Series, cost_bps: float, label: str) -> dict:
    """direction: +1/-1/0 per day (0 = no trade). Fixed-notional per-trade
    stats (comparable to every other study in this project) plus an honest
    compounding equity curve (valid here since trades never overlap)."""
    traded = direction != 0
    gross = direction[traded] * df.loc[traded, "us_first_hour_return"]
    net = gross - cost_bps / 10000.0  # cost charged whenever a trade is taken
    if len(net) == 0:
        return {"label": label, "trades": 0}

    equity = (1 + net).cumprod()
    running_max = equity.cummax()
    drawdown = 1 - equity / running_max
    sharpe = float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else None

    return {
        "label": label, "cost_bps_round_trip": cost_bps, "trades": int(len(net)),
        "win_rate": float((net > 0).mean()),
        "mean_return_bps": float(net.mean() * 10000),
        "total_return_compounded": float(equity.iloc[-1] - 1),
        "sharpe_annualized": sharpe,
        "max_drawdown": float(drawdown.max()),
    }


def random_control(df: pd.DataFrame, n_trades_like: pd.Series, cost_bps: float, seed_count: int) -> list[float]:
    """Same trade count and timing, random +1/-1 direction instead of the
    DAX signal -- isolates whether the SIGNAL adds value beyond just
    'SPY's first hour has some average drift you'd capture by trading
    every day regardless of direction.'"""
    traded_dates = n_trades_like.index
    results = []
    rng = np.random.default_rng(0)
    for _ in range(seed_count):
        random_dir = pd.Series(rng.choice([-1, 1], size=len(traded_dates)), index=traded_dates)
        gross = random_dir * df.loc[traded_dates, "us_first_hour_return"]
        net = gross - cost_bps / 10000.0
        results.append(float(net.mean() * 10000))
    return results


def main() -> None:
    print("Building dataset (DAX signal, SPY first-hour target)...")
    df = build_dataset()
    print(f"{len(df)} trading days, {df.index.min().date()} to {df.index.max().date()}")

    abs_eu = df["eu_pre_open_return"].abs()
    top_quartile_cutoff = abs_eu.quantile(0.75)
    median_cutoff = abs_eu.median()

    variants = {
        "trade_every_day": df["direction"],
        "trade_top_half_by_|DAX_move|": df["direction"].where(abs_eu >= median_cutoff, 0),
        "trade_top_quartile_by_|DAX_move|": df["direction"].where(abs_eu >= top_quartile_cutoff, 0),
        "always_long_context_only": pd.Series(1, index=df.index),  # benchmark: SPY's raw first-hour drift, no signal
    }

    all_results = {}
    for cost_bps in ROUND_TRIP_COSTS_BPS:
        print(f"\n{'='*70}\ncost = {cost_bps} bps round-trip\n{'='*70}")
        for name, direction in variants.items():
            r = evaluate(df, direction, cost_bps, name)
            print(f"{name:38s} trades={r.get('trades',0):4d} win={r.get('win_rate',float('nan')):.3f} "
                  f"mean={r.get('mean_return_bps',float('nan')):6.2f}bps "
                  f"total={r.get('total_return_compounded',float('nan'))*100:7.2f}% "
                  f"sharpe={r.get('sharpe_annualized') or float('nan'):.2f} "
                  f"maxDD={r.get('max_drawdown',float('nan'))*100:.2f}%")
            all_results.setdefault(name, {})[cost_bps] = r

        # Random control at this cost level, sized to the "trade every day" variant
        real_mean = all_results["trade_every_day"][cost_bps]["mean_return_bps"]
        random_means = random_control(df, df.loc[df["direction"] != 0, "direction"], cost_bps, N_RANDOM_SEEDS)
        pct = float((np.array(random_means) < real_mean).mean() * 100)
        print(f"\nDAX-signal mean ({real_mean:.2f}bps) vs. {N_RANDOM_SEEDS} random-direction seeds "
              f"(mean={np.mean(random_means):.2f}bps, std={np.std(random_means):.2f}bps): "
              f"percentile = {pct:.1f}")
        all_results.setdefault("random_control_percentile", {})[cost_bps] = pct

    out = ROOT / "outputs" / "european_lead_us_first_hour_backtest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
