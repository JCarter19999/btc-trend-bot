"""Remaining promotion-gate checks for the online Kalman-regression candidate
(see EQUITY_KALMAN_ONLINE_REGRESSION.md): a random-selection control, a
process-noise sensitivity sweep, and a head-to-head vs. Ridge restricted to
their overlapping tradable date window (both models start trading at
different points -- Ridge needs a full train_bars history before its first
fold, Kalman needs warmup_updates observations -- so a fair comparison must
use only dates both models are actually predicting for).

Writes one JSON summary; does not touch the live deployment or main branch.
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

from btc_trend_bot.kalman_regression import online_predict_and_update

from run_equity_real_data_walkforward import (
    FEATURES,
    build_candidates,
    load_config,
    load_csv_dir,
    simulate_capital,
    walk_forward,
)
from run_equity_kalman_online import select_kalman_winners


def summarize(trades: pd.DataFrame, cfg, label: str) -> dict:
    if trades.empty:
        return {"label": label, "trades_selected": 0}
    path, summary = simulate_capital(trades, cfg)
    return {
        "label": label,
        "trades_selected": int(len(trades)),
        "trades_taken": summary["trades_taken"],
        "trades_skipped_by_safety_layer": summary["trades_skipped"],
        # win_rate is computed over trades actually TAKEN (excludes trades
        # skipped by the drawdown-pause/cooldown safety layer) -- matching
        # simulate_capital's own definition, so this is comparable to the
        # baseline run in EQUITY_KALMAN_ONLINE_REGRESSION.md.
        "win_rate": round(summary["win_rate"], 4) if summary["trades_taken"] else None,
        "total_return_pct": round(summary["total_return"] * 100, 1),
        "max_drawdown": round(summary["max_drawdown"], 4),
        "date_range": [str(trades.signal_time.min()), str(trades.signal_time.max())],
    }


def year_slice(trades: pd.DataFrame, year: int) -> pd.DataFrame:
    return trades[trades.signal_time.dt.year == year]


def main() -> None:
    cfg = load_config(ROOT / "configs/real_data.yaml")
    symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(symbols, ROOT / "data/real")
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    candidates["signal_time"] = pd.to_datetime(candidates["signal_time"], utc=True)
    candidates["exit_time"] = pd.to_datetime(candidates["exit_time"], utc=True)
    work = candidates.dropna(subset=[*FEATURES, "net_return"]).reset_index(drop=True)

    results: dict = {}

    # --- 1. Baseline Kalman (process_var=1e-5, the value used in the initial write-up) ---
    baseline_winners = select_kalman_winners(
        candidates, cfg, process_var=1e-5, obs_var=1.0, prior_var=1.0,
        warmup_updates=200, shuffle_labels=False, seed=0,
    )
    results["kalman_baseline"] = summarize(baseline_winners, cfg, "kalman_baseline")
    results["kalman_baseline_2022"] = summarize(year_slice(baseline_winners, 2022), cfg, "kalman_baseline_2022")
    tradable_start = baseline_winners.signal_time.min()

    # --- 2. Random-selection control (uninformed but real candidate pool, same
    # tradable window as the baseline Kalman run so it's an apples-to-apples
    # comparison, not just "random from day 1"). ---
    random_pool = work[work.signal_time >= tradable_start]
    random_summaries = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        picks = (
            random_pool.assign(_r=rng.random(len(random_pool)))
            .sort_values(["signal_time", "_r"])
            .groupby("signal_time", as_index=False).head(1)
            .drop(columns="_r").sort_values("signal_time")
        )
        random_summaries.append(summarize(picks, cfg, f"random_seed{seed}"))
    results["random_control"] = random_summaries

    # --- 3. process_var sensitivity sweep (obs_var/prior_var held at baseline) ---
    sweep_rows = []
    for process_var in (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2):
        predictions = online_predict_and_update(
            work, features=list(FEATURES), label_col="net_return",
            signal_time_col="signal_time", ready_time_col="exit_time",
            process_var=process_var, obs_var=1.0, prior_var=1.0, warmup_updates=200,
        )
        w2 = work.copy()
        w2["predicted_return"] = predictions
        qualified = w2.dropna(subset=["predicted_return"])
        qualified = qualified[qualified.predicted_return >= cfg.return_threshold_bps / 10000]
        winners = (
            qualified.sort_values(["signal_time", "predicted_return"], ascending=[True, False])
            .groupby("signal_time", as_index=False).head(1).sort_values("signal_time")
        )
        overall = summarize(winners, cfg, f"process_var={process_var:g}")
        y2022 = summarize(year_slice(winners, 2022), cfg, f"process_var={process_var:g}_2022")
        sweep_rows.append({"process_var": process_var, "overall": overall, "y2022": y2022})
    results["process_var_sweep"] = sweep_rows

    # --- 4. Head-to-head vs Ridge, restricted to the overlapping tradable window ---
    ridge_folds, ridge_trades = walk_forward(candidates, cfg, shuffle_labels=False, seed=0, selection="ridge")
    ridge_trades["signal_time"] = pd.to_datetime(ridge_trades["signal_time"], utc=True)
    overlap_start = max(baseline_winners.signal_time.min(), ridge_trades.signal_time.min())
    overlap_end = min(baseline_winners.signal_time.max(), ridge_trades.signal_time.max())
    kalman_overlap = baseline_winners[
        (baseline_winners.signal_time >= overlap_start) & (baseline_winners.signal_time <= overlap_end)
    ]
    ridge_overlap = ridge_trades[
        (ridge_trades.signal_time >= overlap_start) & (ridge_trades.signal_time <= overlap_end)
    ]
    results["head_to_head_overlap_window"] = [str(overlap_start), str(overlap_end)]
    results["head_to_head_kalman"] = summarize(kalman_overlap, cfg, "kalman_overlap")
    results["head_to_head_ridge"] = summarize(ridge_overlap, cfg, "ridge_overlap")

    out = ROOT / "outputs" / "equity_kalman_promotion_checks"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
