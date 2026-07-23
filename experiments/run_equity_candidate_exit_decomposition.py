"""Decompose where the candidate pool's positive expectancy actually comes
from: the mask in build_candidates (candidate edge) vs. the ATR stop/target
mechanics (exit edge) vs. neither (i.e. it's just this bull-trending
universe). Uses the same corrected single-position simulator as
run_equity_expectancy_matrix.py.

Candidate edge: mask-filtered candidates vs. an unrestricted pool (every
day/symbol with a valid ATR, no mask condition at all), both with random
selection over many seeds.

Exit edge: holds the exact same (symbol, signal_time) selections fixed
across four exit regimes -- current ATR stop/target, fixed 1-day hold
(buy next open, sell next close), fixed 5-day hold, fixed 10-day hold (same
as max_hold_bars but with the stop/target checks disabled) -- by
recomputing each exit variant against the identical iteration order used to
build the mask-filtered candidate pool, so a given random seed selects the
same underlying (symbol, day) across every exit variant. Only the outcome of
that trade differs.
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
    FEATURES,
    add_features,
    load_config,
    load_csv_dir,
)

N_SEEDS = 100
SIZING = SizingMode("fixed_notional_2500", "fixed_notional", 2500.0)


def simulate_trade_fixed_hold(frame: pd.DataFrame, i: int, cfg, hold_bars: int) -> dict | None:
    entry_i = i + 1
    if entry_i >= len(frame):
        return None
    entry = float(frame.iloc[entry_i].open) * (1 + cfg.slippage_bps_each_side / 10000)
    atr = float(frame.iloc[i].atr)
    if not np.isfinite(atr) or atr <= 0:
        return None
    exit_i = min(entry_i + hold_bars - 1, len(frame) - 1)
    exit_price = float(frame.iloc[exit_i].close)
    exit_fill = exit_price * (1 - cfg.slippage_bps_each_side / 10000)
    return {"entry_time": frame.index[entry_i], "exit_time": frame.index[exit_i], "entry_price": entry,
            "exit_price": exit_fill, "net_return": exit_fill / entry - 1, "exit_reason": f"fixed_hold_{hold_bars}",
            "bars_held": exit_i - entry_i + 1}


def build_pool(frames: dict, benchmark_symbol: str, cfg, exit_fn, apply_mask: bool) -> pd.DataFrame:
    """Same iteration order for every (exit_fn, apply_mask) combination given
    the same frames/cfg, so a fixed random seed selects identical
    (symbol, i) pairs across exit variants when apply_mask is held fixed."""
    bench = frames[benchmark_symbol]
    rows = []
    for symbol in cfg.symbols:
        f = add_features(frames[symbol], bench)
        for i in range(60, len(f) - cfg.max_hold_bars - 1):
            r = f.iloc[i]
            if apply_mask:
                mask = (r.relative_volume > 0.6 and r.atr_pct > 0 and abs(r.ema_spread_atr) < 8 and r.return_20 > -0.25)
                if not mask:
                    continue
            trade = exit_fn(f, i, cfg)
            if trade is None:
                continue
            record = {"signal_time": f.index[i], "symbol": symbol, **trade}
            record.update({c: float(r[c]) for c in FEATURES if c in r and pd.notna(r[c])})
            if all(c in record and np.isfinite(record[c]) for c in FEATURES):
                rows.append(record)
    if not rows:
        return pd.DataFrame(columns=["signal_time", "symbol", *FEATURES, "net_return"])
    return pd.DataFrame(rows).sort_values(["signal_time", "symbol"]).reset_index(drop=True)


def random_selection_runs(pool: pd.DataFrame, cfg, n_seeds: int) -> list[dict]:
    out = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        picks = (
            pool.assign(_r=rng.random(len(pool)))
            .sort_values(["signal_time", "_r"])
            .groupby("signal_time", as_index=False).head(1)
            .drop(columns="_r").sort_values("signal_time")
        )
        if picks.empty:
            continue
        picks = picks.copy()
        picks["entry_time"] = pd.to_datetime(picks["entry_time"], utc=True)
        picks["exit_time"] = pd.to_datetime(picks["exit_time"], utc=True)
        _, summary = simulate_single_position(picks, cfg, SIZING)
        out.append(summary)
    return out


def agg(summaries: list[dict]) -> dict:
    vals = [s["expectancy_bps"] for s in summaries if s.get("trades_taken", 0) > 0]
    return {
        "n_runs": len(vals),
        "mean_expectancy_bps": float(np.mean(vals)) if vals else None,
        "std_expectancy_bps": float(np.std(vals)) if vals else None,
        "min_expectancy_bps": float(np.min(vals)) if vals else None,
        "max_expectancy_bps": float(np.max(vals)) if vals else None,
        "mean_win_rate": float(np.mean([s["win_rate"] for s in summaries if s.get("trades_taken", 0) > 0])) if vals else None,
    }


def main() -> None:
    cfg = load_config(ROOT / "configs/real_data.yaml")
    symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(symbols, ROOT / "data/real")

    from run_equity_real_data_walkforward import simulate_trade as atr_exit_fn

    results: dict = {}

    print("Building pools...")
    pool_masked_atr = build_pool(frames, cfg.benchmark, cfg, atr_exit_fn, apply_mask=True)
    pool_unrestricted_atr = build_pool(frames, cfg.benchmark, cfg, atr_exit_fn, apply_mask=False)

    print("=== Candidate edge: mask-filtered vs. unrestricted pool (both ATR exits, random selection) ===")
    results["candidate_edge_masked"] = agg(random_selection_runs(pool_masked_atr, cfg, N_SEEDS))
    results["candidate_edge_unrestricted"] = agg(random_selection_runs(pool_unrestricted_atr, cfg, N_SEEDS))
    print(json.dumps({"masked": results["candidate_edge_masked"],
                       "unrestricted": results["candidate_edge_unrestricted"]}, indent=2))

    print("\n=== Exit edge: mask-filtered pool, ATR exit vs. fixed-hold exits (random selection) ===")
    results["exit_edge_atr"] = results["candidate_edge_masked"]  # same pool, reused
    for hold_bars in (1, 5, 10):
        pool_fixed = build_pool(
            frames, cfg.benchmark, cfg,
            lambda f, i, cfg, hb=hold_bars: simulate_trade_fixed_hold(f, i, cfg, hb),
            apply_mask=True,
        )
        results[f"exit_edge_fixed_hold_{hold_bars}"] = agg(random_selection_runs(pool_fixed, cfg, N_SEEDS))
        print(f"fixed_hold_{hold_bars}: {json.dumps(results[f'exit_edge_fixed_hold_{hold_bars}'], indent=2)}")

    out = ROOT / "outputs" / "equity_candidate_exit_decomposition"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nFull summary written to", out / "summary.json")


if __name__ == "__main__":
    main()
