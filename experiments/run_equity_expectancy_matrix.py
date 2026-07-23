"""Decisive selector-quality comparison: fixed sizing modes (no compounding
path-dependence) + single-position-correct timing (see portfolio_sim.py),
across selectors, with a many-seed random distribution to rank real-label
Ridge against chance.

This directly implements the redesign: separate expectancy (does the
*selector* pick better trades?) from compounding (which inflates almost any
positive-expectancy, high-frequency, long-only strategy in a bull-trending
universe regardless of skill -- see the raw-candidate-pool audit in
EQUITY_KALMAN_ONLINE_REGRESSION.md, where even the UNSELECTED candidate pool
already has +0.57%/trade mean expectancy).

Selectors compared: real-label Ridge, shuffled-label Ridge (x3 seeds),
simple_trend, Kalman-online baseline, and random selection (N seeds -- the
distribution real Ridge is ranked against).

Sizing modes: fixed_notional ($2,500/trade, no compounding -- the cleanest
comparison), fixed_fractional (10% of current equity), vol_targeted (0.5%
of equity risked per trade, sized off the ATR stop distance).
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

from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position

from run_equity_real_data_walkforward import (
    build_candidates,
    load_config,
    load_csv_dir,
    walk_forward,
)
from run_equity_kalman_online import select_kalman_winners

N_RANDOM_SEEDS = 50
SIZINGS = [
    SizingMode("fixed_notional_2500", "fixed_notional", 2500.0),
    SizingMode("fixed_fractional_10pct", "fixed_fractional", 0.10),
    SizingMode("vol_targeted_0.5pct_risk", "vol_targeted", 0.005),
]


def run_all_sizings(trades: pd.DataFrame, cfg) -> dict:
    out = {}
    for sizing in SIZINGS:
        if trades.empty:
            out[sizing.name] = {"sizing": sizing.name, "trades_taken": 0}
            continue
        _, summary = simulate_single_position(trades, cfg, sizing)
        out[sizing.name] = summary
    return out


def main() -> None:
    cfg = load_config(ROOT / "configs/real_data.yaml")
    symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(symbols, ROOT / "data/real")
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    candidates["signal_time"] = pd.to_datetime(candidates["signal_time"], utc=True)
    candidates["exit_time"] = pd.to_datetime(candidates["exit_time"], utc=True)

    results: dict = {}

    _, ridge_real = walk_forward(candidates, cfg, shuffle_labels=False, seed=0, selection="ridge")
    ridge_real["entry_time"] = pd.to_datetime(ridge_real["entry_time"], utc=True)
    ridge_real["exit_time"] = pd.to_datetime(ridge_real["exit_time"], utc=True)
    results["ridge_real"] = run_all_sizings(ridge_real, cfg)

    for seed in (0, 1, 2):
        _, ridge_shuf = walk_forward(candidates, cfg, shuffle_labels=True, seed=seed, selection="ridge")
        ridge_shuf["entry_time"] = pd.to_datetime(ridge_shuf["entry_time"], utc=True)
        ridge_shuf["exit_time"] = pd.to_datetime(ridge_shuf["exit_time"], utc=True)
        results[f"ridge_shuffled_seed{seed}"] = run_all_sizings(ridge_shuf, cfg)

    _, simple_trend = walk_forward(candidates, cfg, shuffle_labels=False, seed=0, selection="simple_trend")
    simple_trend["entry_time"] = pd.to_datetime(simple_trend["entry_time"], utc=True)
    simple_trend["exit_time"] = pd.to_datetime(simple_trend["exit_time"], utc=True)
    results["simple_trend"] = run_all_sizings(simple_trend, cfg)

    kalman = select_kalman_winners(candidates, cfg, process_var=1e-5, obs_var=1.0, prior_var=1.0,
                                    warmup_updates=200, shuffle_labels=False, seed=0)
    results["kalman_online"] = run_all_sizings(kalman, cfg)

    random_runs = {sizing.name: [] for sizing in SIZINGS}
    for seed in range(N_RANDOM_SEEDS):
        _, r = walk_forward(candidates, cfg, shuffle_labels=False, seed=seed, selection="random")
        r["entry_time"] = pd.to_datetime(r["entry_time"], utc=True)
        r["exit_time"] = pd.to_datetime(r["exit_time"], utc=True)
        per_sizing = run_all_sizings(r, cfg)
        for name, summary in per_sizing.items():
            random_runs[name].append(summary)
    results["random_distribution_n"] = N_RANDOM_SEEDS

    percentile_rank = {}
    for sizing in SIZINGS:
        dist = [row["expectancy_bps"] for row in random_runs[sizing.name] if row.get("trades_taken", 0) > 0]
        ridge_val = results["ridge_real"][sizing.name].get("expectancy_bps", np.nan)
        if dist and np.isfinite(ridge_val):
            pct = float((np.array(dist) < ridge_val).mean() * 100)
        else:
            pct = None
        percentile_rank[sizing.name] = {
            "ridge_real_expectancy_bps": ridge_val,
            "random_expectancy_bps_mean": float(np.mean(dist)) if dist else None,
            "random_expectancy_bps_std": float(np.std(dist)) if dist else None,
            "random_expectancy_bps_min": float(np.min(dist)) if dist else None,
            "random_expectancy_bps_max": float(np.max(dist)) if dist else None,
            "ridge_percentile_within_random": pct,
        }
    results["ridge_vs_random_percentile"] = percentile_rank
    results["random_runs_raw"] = random_runs

    out = ROOT / "outputs" / "equity_expectancy_matrix"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))

    for name in ("ridge_real", "ridge_shuffled_seed0", "ridge_shuffled_seed1", "ridge_shuffled_seed2",
                 "simple_trend", "kalman_online"):
        print(f"\n=== {name} ===")
        for sizing_name, summary in results[name].items():
            print(f"  {sizing_name}: trades_taken={summary.get('trades_taken')} "
                  f"win_rate={summary.get('win_rate')} expectancy_bps={summary.get('expectancy_bps')} "
                  f"profit_factor={summary.get('profit_factor')} total_return={summary.get('total_return')}")

    print(f"\n=== Ridge vs. {N_RANDOM_SEEDS}-seed random distribution (expectancy_bps) ===")
    print(json.dumps(percentile_rank, indent=2, default=str))


if __name__ == "__main__":
    main()
