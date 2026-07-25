"""Full-rigor re-test of simple_trend's hold-duration sweep at 15/20/30
trading days (the original EQUITY_EXIT_REGIME_SIMPLE_TREND.md sweep was a
single run each, no drawdown/OOS/random-control). Question: does the
monotonic "longer hold = better expectancy" pattern keep holding up under
real rigor, or does it converge into indistinguishable-from-buy-and-hold?
"""
from __future__ import annotations
import sys, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
from run_equity_real_data_walkforward import load_config, load_csv_dir, build_candidates, _select_fold_winners
from btc_trend_bot.portfolio_sim import simulate_single_position, SizingMode

HOLD_LENGTHS = [10, 15, 20, 30]  # 10 is the sanity-check anchor


def run_one(frames, cfg, hold_bars, seed=0):
    cfg = replace(cfg, max_hold_bars=hold_bars, stop_atr=100.0, target_atr=100.0)
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    rng = np.random.default_rng(seed)
    winners = _select_fold_winners(candidates, candidates, cfg, 0, rng, False, "simple_trend")
    sizing = SizingMode(name="fixed_2500", kind="fixed_notional", value=2500.0)
    capital_path, stats = simulate_single_position(winners, cfg, sizing)
    taken = capital_path[capital_path.trade_taken == True].copy()
    return taken, stats, candidates, winners


def random_control(candidates, cfg, hold_bars, n_seeds=200):
    cfg = replace(cfg, max_hold_bars=hold_bars, stop_atr=100.0, target_atr=100.0)
    sizing = SizingMode(name="fixed_2500", kind="fixed_notional", value=2500.0)
    totals = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        winners = _select_fold_winners(candidates, candidates, cfg, 0, rng, False, "random")
        _, stats = simulate_single_position(winners, cfg, sizing)
        totals.append(stats["total_return"])
    return np.array(totals)


def buy_and_hold(frames, cfg, start, end):
    rets = []
    for sym in cfg.symbols:
        f = frames[sym]
        f = f[(f.index >= start) & (f.index <= end)]
        if len(f) < 2:
            continue
        rets.append(f["close"].iloc[-1] / f["close"].iloc[0] - 1)
    return float(np.mean(rets))


def main():
    cfg = load_config(ROOT / "configs" / "real_data.yaml")
    all_symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(all_symbols, ROOT / "data" / "real")

    out_dir = ROOT / "outputs" / "hold_duration_rigor_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for hold in HOLD_LENGTHS:
        taken, stats, candidates, winners = run_one(frames, cfg, hold)
        n = len(taken)
        win_rate = float((taken.trade_return > 0).mean()) if n else float("nan")
        mean_ret = float(taken.trade_return.mean()) if n else float("nan")
        total_return = stats["total_return"]
        max_dd = float(taken.drawdown.max()) if n else 0.0
        start, end = taken.signal_time.min(), taken.signal_time.max()
        bh = buy_and_hold(frames, cfg, start, end) if n else float("nan")

        # OOS split
        mid_date = candidates.signal_time.quantile(0.5)
        first_c = candidates[candidates.signal_time <= mid_date]
        second_c = candidates[candidates.signal_time > mid_date]
        rng0 = np.random.default_rng(0)
        w1 = _select_fold_winners(first_c, first_c, cfg, 0, rng0, False, "simple_trend")
        w2 = _select_fold_winners(second_c, second_c, cfg, 0, rng0, False, "simple_trend")
        sizing = SizingMode(name="fixed_2500", kind="fixed_notional", value=2500.0)
        _, stats1 = simulate_single_position(w1, cfg, sizing)
        _, stats2 = simulate_single_position(w2, cfg, sizing)

        rc = random_control(candidates, cfg, hold, n_seeds=200)
        pct = float((rc < total_return).mean() * 100)

        results[hold] = {
            "trades": n, "win_rate": win_rate, "mean_return": mean_ret,
            "total_return": total_return, "max_drawdown": max_dd,
            "buy_and_hold_same_window": bh,
            "strategy_minus_buyhold": total_return - bh if np.isfinite(bh) else None,
            "oos_first_half": stats1["total_return"], "oos_second_half": stats2["total_return"],
            "random_control_mean": float(rc.mean()), "random_control_std": float(rc.std()),
            "random_control_percentile": pct,
            "window_start": str(start), "window_end": str(end),
        }
        print(f"hold={hold}: trades={n} win={win_rate*100:.1f}% mean={mean_ret*10000:.1f}bps "
              f"total={total_return*100:+.1f}% dd={max_dd*100:.1f}% bh={bh*100:+.1f}% "
              f"edge_over_bh={(total_return-bh)*100:+.1f}pp "
              f"oos=[{stats1['total_return']*100:+.1f}%,{stats2['total_return']*100:+.1f}%] "
              f"random_pctile={pct:.1f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
