"""Does simple_trend's momentum-rotation edge generalize to non-tech stocks,
or is it specific to the current AI/tech secular bull market? Joey's framing:
"want to see if it holds up outside of an artificially inflated market."

Basket: JPM (financials), UNH (healthcare), CAT (industrials), XOM (energy)
-- four distinct traditional-economy sectors, deliberately avoiding any
tech/semiconductor/AI-adjacent name. Same exact mechanics as the validated
tech-basket baseline (AAPL/MSFT/NVDA/TSLA): relative_strength_20 selection
among symbols above their 50-day benchmark (SPY) trend, fixed 10-day hold,
stop_atr/target_atr disabled, $2,500 fixed-notional, 2018-2026. Only the
stock universe changes -- isolates whether the MECHANISM works outside tech,
not whether some other parameter also needs retuning (retuning here would be
a new hypothesis, not a generalization test).

Reuses the exact reconciliation method already validated this session for
the 10-day tech baseline: _select_fold_winners against the FULL candidate
set (bypassing walk_forward's 756-bar warmup, which simple_trend doesn't
need), then portfolio_sim.simulate_single_position (genuinely single-
position, unlike simulate_capital which allows overlap).
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
NONTECH_SYMBOLS = ("JPM", "UNH", "CAT", "XOM")
N_RANDOM_SEEDS = 50
OUT_DIR = ROOT / "outputs" / "nontech_basket_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_variant(frames, cfg, symbols: tuple[str, ...], sizing: SizingMode, label: str):
    variant_cfg = dataclasses.replace(cfg, symbols=symbols)
    candidates = build_candidates(frames, cfg.benchmark, variant_cfg)

    rng = np.random.default_rng(0)
    winners_real = _select_fold_winners(candidates, candidates, variant_cfg, 0, rng, False, "simple_trend")
    path_real, summary_real = simulate_single_position(winners_real, variant_cfg, sizing)

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
        _, half_summary = simulate_single_position(half_winners, variant_cfg, sizing)
        oos_results[half_name] = half_summary

    return {
        "label": label,
        "symbols": symbols,
        "trades_taken": summary_real["trades_taken"],
        "win_rate": summary_real["win_rate"],
        "total_return": summary_real["total_return"],
        "max_drawdown": summary_real["max_drawdown"],
        "mean_return_per_trade": summary_real["mean_return_per_trade"],
        "profit_factor": summary_real.get("profit_factor"),
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
    tech_symbols = cfg.symbols
    all_symbols = tuple(dict.fromkeys((*tech_symbols, *NONTECH_SYMBOLS, cfg.benchmark)))

    frames = {}
    tech_dir = ROOT / "data" / "real"
    sp100_dir = ROOT / "data" / "real_sp100"
    for sym in all_symbols:
        if (tech_dir / f"{sym}.csv").exists():
            frames.update(load_csv_dir((sym,), tech_dir))
        elif (sp100_dir / f"{sym}.csv").exists():
            frames.update(load_csv_dir((sym,), sp100_dir))
        else:
            raise FileNotFoundError(f"No cached real data for {sym} in data/real or data/real_sp100")

    sizing = SizingMode(name="fixed_notional_2500", kind="fixed_notional", value=2500.0)

    print(f"Tech baseline symbols: {tech_symbols}")
    print(f"Non-tech basket symbols: {NONTECH_SYMBOLS}\n")

    # 1) Reconcile tech baseline exactly before trusting anything else.
    tech_res, tech_path = run_variant(frames, cfg, tech_symbols, sizing, "tech_baseline")
    assert tech_res["trades_taken"] == 188, f"RECONCILIATION FAILED: {tech_res['trades_taken']} != 188 trades"
    assert abs(tech_res["win_rate"] - 0.580) < 0.005, f"win_rate mismatch: {tech_res['win_rate']}"
    assert abs(tech_res["total_return"] - 4.574) < 0.02, f"total_return mismatch: {tech_res['total_return']}"
    assert abs(tech_res["max_drawdown"] - 0.221) < 0.01, f"max_drawdown mismatch: {tech_res['max_drawdown']}"
    print("Tech baseline reconciliation: CONFIRMED EXACT MATCH to documented characterization "
          f"(188 trades, {tech_res['win_rate']*100:.1f}% win, {tech_res['total_return']*100:+.1f}% return, "
          f"{tech_res['max_drawdown']*100:.1f}% max DD).\n")

    # 2) Run identical mechanics on the non-tech basket.
    nontech_res, nontech_path = run_variant(frames, cfg, NONTECH_SYMBOLS, sizing, "nontech_basket")
    print(f"Non-tech basket: trades={nontech_res['trades_taken']} win_rate={nontech_res['win_rate']*100:.1f}% "
          f"total_return={nontech_res['total_return']*100:+.1f}% max_dd={nontech_res['max_drawdown']*100:.1f}% "
          f"mean/trade={nontech_res['mean_return_per_trade']*10000:.1f}bps "
          f"random_ctrl_pctile={nontech_res['random_control_percentile']:.1f} "
          f"oos_first={nontech_res['oos_first_half_total_return']*100:+.1f}% "
          f"oos_second={nontech_res['oos_second_half_total_return']*100:+.1f}%\n")

    # 3) Buy-and-hold comparisons, same window each basket's own candidates cover.
    tech_start = tech_path[tech_path.trade_taken]["signal_time"].min()
    tech_end = tech_path[tech_path.trade_taken]["signal_time"].max()
    nontech_start = nontech_path[nontech_path.trade_taken]["signal_time"].min()
    nontech_end = nontech_path[nontech_path.trade_taken]["signal_time"].max()

    bh_tech = buy_and_hold_equal_weight(frames, tech_symbols, tech_start, tech_end)
    bh_nontech = buy_and_hold_equal_weight(frames, NONTECH_SYMBOLS, nontech_start, nontech_end)

    print(f"Buy-and-hold equal-weight, tech basket, same window: {bh_tech*100:+.1f}%")
    print(f"Buy-and-hold equal-weight, non-tech basket, same window: {bh_nontech*100:+.1f}%")

    tech_res["buy_and_hold_equal_weight_same_window"] = bh_tech
    nontech_res["buy_and_hold_equal_weight_same_window"] = bh_nontech
    tech_res["gap_to_buy_and_hold_pp"] = (tech_res["total_return"] - bh_tech) * 100
    nontech_res["gap_to_buy_and_hold_pp"] = (nontech_res["total_return"] - bh_nontech) * 100

    results = [tech_res, nontech_res]
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "nontech_basket_results.csv", index=False)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR / 'nontech_basket_results.csv'} and summary.json")


if __name__ == "__main__":
    main()
