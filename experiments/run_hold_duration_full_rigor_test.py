"""Full-rigor test of longer hold durations (15/20/30 days) for simple_trend,
extending EQUITY_EXIT_REGIME_SIMPLE_TREND.md's single-run hold-duration
sweep (5/10/15/20d: 106.6/243.3/272.3/452.7 bps/trade, no drawdown/OOS/
random-control ever reported for 15/20d, and 30d never tested at all).

Reuses the exact reconciliation method already validated this session for
the 10-day baseline: _select_fold_winners against the FULL candidate set
(bypassing walk_forward's 756-bar warmup, which simple_trend doesn't need),
then portfolio_sim.simulate_single_position (genuinely single-position,
unlike simulate_capital which allows overlap) with fixed-notional $2,500.

For each hold duration: candidates must be rebuilt with that max_hold_bars
(it's baked into simulate_trade's exit_i at candidate-generation time, not
a post-hoc simulation parameter) -- so this reruns build_candidates fresh
per duration, not just re-slicing one candidate set.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_equity_real_data_walkforward import (
    build_candidates, load_config, load_csv_dir, _select_fold_winners,
)
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position

LIVE_CONFIG = Path("/home/joey/equity_v2_4/config/simple_trend_exit_regime_strategy.yaml")
HOLD_DAYS = (10, 15, 20, 30)
N_RANDOM_SEEDS = 50
OUT_DIR = ROOT / "outputs" / "hold_duration_full_rigor_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_variant(frames, cfg, hold_days: int, sizing: SizingMode):
    variant_cfg = dataclasses.replace(cfg, max_hold_bars=hold_days)
    candidates = build_candidates(frames, cfg.benchmark, variant_cfg)

    rng = np.random.default_rng(0)
    winners_real = _select_fold_winners(candidates, candidates, variant_cfg, 0, rng, False, "simple_trend")
    path_real, summary_real = simulate_single_position(winners_real, variant_cfg, sizing)

    # Random-selection control: same candidate pool, random pick per signal_time.
    random_totals = []
    for seed in range(N_RANDOM_SEEDS):
        rrng = np.random.default_rng(seed + 1000)
        winners_rand = _select_fold_winners(candidates, candidates, variant_cfg, 0, rrng, False, "random")
        _, summary_rand = simulate_single_position(winners_rand, variant_cfg, sizing)
        random_totals.append(summary_rand["total_return"])
    random_totals = np.array(random_totals)
    real_total = summary_real["total_return"]
    z = (real_total - random_totals.mean()) / random_totals.std() if random_totals.std() > 0 else float("nan")
    pctile = float((random_totals < real_total).mean() * 100)

    # OOS split: chronological midpoint of the full candidate date range.
    mid = candidates.signal_time.min() + (candidates.signal_time.max() - candidates.signal_time.min()) / 2
    oos_results = {}
    for half_name, half_candidates in (
        ("first_half", candidates[candidates.signal_time < mid]),
        ("second_half", candidates[candidates.signal_time >= mid]),
    ):
        if half_candidates.empty:
            oos_results[half_name] = None
            continue
        hrng = np.random.default_rng(0)
        half_winners = _select_fold_winners(half_candidates, half_candidates, variant_cfg, 0, hrng, False, "simple_trend")
        half_path, half_summary = simulate_single_position(half_winners, variant_cfg, sizing)
        oos_results[half_name] = half_summary

    return {
        "hold_days": hold_days,
        "trades_taken": summary_real["trades_taken"],
        "win_rate": summary_real["win_rate"],
        "total_return": summary_real["total_return"],
        "max_drawdown": summary_real["max_drawdown"],
        "mean_return_per_trade": summary_real["mean_return_per_trade"],
        "random_control_mean": float(random_totals.mean()),
        "random_control_std": float(random_totals.std()),
        "random_control_z": float(z),
        "random_control_percentile": pctile,
        "oos_first_half_total_return": oos_results["first_half"]["total_return"] if oos_results["first_half"] else None,
        "oos_second_half_total_return": oos_results["second_half"]["total_return"] if oos_results["second_half"] else None,
    }, path_real


def buy_and_hold_equal_weight(frames, symbols, start, end) -> float:
    rets = []
    for sym in symbols:
        f = frames[sym]
        f = f[(f.index >= start) & (f.index <= end)]
        if len(f) < 2:
            continue
        rets.append(f["close"].iloc[-1] / f["close"].iloc[0] - 1.0)
    return float(np.mean(rets))


def main():
    cfg = load_config(LIVE_CONFIG)
    all_symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(all_symbols, ROOT / "data" / "real")
    sizing = SizingMode(name="fixed_notional_2500", kind="fixed_notional", value=2500.0)

    results = []
    baseline_path = None
    for hold_days in HOLD_DAYS:
        res, path = run_variant(frames, cfg, hold_days, sizing)
        if hold_days == 10:
            assert res["trades_taken"] == 188, f"10-day reconciliation FAILED: {res['trades_taken']} != 188 trades"
            assert abs(res["win_rate"] - 0.580) < 0.005, f"10-day win_rate mismatch: {res['win_rate']}"
            assert abs(res["total_return"] - 4.574) < 0.02, f"10-day total_return mismatch: {res['total_return']}"
            assert abs(res["max_drawdown"] - 0.221) < 0.01, f"10-day max_drawdown mismatch: {res['max_drawdown']}"
            print("10-day baseline reconciliation: CONFIRMED EXACT MATCH to documented characterization.")
            baseline_path = path
        results.append(res)
        print(f"hold={hold_days}d: trades={res['trades_taken']} win_rate={res['win_rate']*100:.1f}% "
              f"total_return={res['total_return']*100:+.1f}% max_dd={res['max_drawdown']*100:.1f}% "
              f"mean/trade={res['mean_return_per_trade']*10000:.1f}bps "
              f"random_ctrl_pctile={res['random_control_percentile']:.1f} "
              f"oos_first={res['oos_first_half_total_return']*100 if res['oos_first_half_total_return'] is not None else float('nan'):+.1f}% "
              f"oos_second={res['oos_second_half_total_return']*100 if res['oos_second_half_total_return'] is not None else float('nan'):+.1f}%")

    # Buy-and-hold equal-weight comparison over the full window each variant covers
    # (use the full candidate date range from the 10-day run as the reference window,
    # since all variants share the same underlying price history / start date).
    full_start = baseline_path[baseline_path.trade_taken]["signal_time"].min() if baseline_path is not None else None
    full_end = baseline_path[baseline_path.trade_taken]["signal_time"].max() if baseline_path is not None else None
    bh_return = buy_and_hold_equal_weight(frames, cfg.symbols, full_start, full_end) if full_start is not None else None
    print(f"\nBuy-and-hold equal-weight (4 stocks, no rotation), same window: {bh_return*100:+.1f}%" if bh_return else "")

    for r in results:
        r["buy_and_hold_equal_weight_same_window"] = bh_return
        r["gap_to_buy_and_hold_pp"] = (r["total_return"] - bh_return) * 100 if bh_return is not None else None

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "hold_duration_full_rigor_results.csv", index=False)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR / 'hold_duration_full_rigor_results.csv'} and summary.json")


if __name__ == "__main__":
    main()
