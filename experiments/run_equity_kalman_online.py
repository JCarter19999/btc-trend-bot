"""Track-B candidate: online Kalman-filter regression in place of fold-retrained Ridge.

Reuses the equity v3.0 pipeline (data loading, feature engineering, ATR
entry/exit simulation, capital safety layer) unchanged from
`run_equity_real_data_walkforward.py` -- only the candidate-selection step
changes. Instead of refitting a fresh `StandardScaler -> Ridge` every
`step_bars` on a trailing window (see `walk_forward`/`_select_fold_winners`
in the base runner), this predicts and updates a single coefficient state
continuously, one candidate at a time, in chronological order -- see
`btc_trend_bot.kalman_regression` for the causality argument (a label is only
folded into the filter once its trade's exit_time has passed).

This is a hypothesis, not a validated replacement: run it with
--shuffle-labels first (should look like noise) before trusting a positive
result, same discipline as the Ridge runner's --shuffle-labels flag.

Usage (from cached CSVs, matching the frozen data snapshot):
    python experiments/run_equity_kalman_online.py \
        --config configs/real_data.yaml --provider csv \
        --output outputs/equity_kalman_online
"""

from __future__ import annotations

import argparse
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
    BacktestConfig,
    benchmark_summary,
    build_candidates,
    download_yfinance,
    load_config,
    load_csv_dir,
    simulate_capital,
)


def select_kalman_winners(
    candidates: pd.DataFrame,
    cfg: BacktestConfig,
    process_var: float,
    obs_var: float,
    prior_var: float,
    warmup_updates: int,
    shuffle_labels: bool,
    seed: int,
) -> pd.DataFrame:
    work = candidates.dropna(subset=[*FEATURES, "net_return"]).reset_index(drop=True)
    if shuffle_labels:
        # Sanity check: breaks any real feature/outcome relationship globally.
        # A still-profitable result means the filter isn't tracking real signal.
        rng = np.random.default_rng(seed)
        work["net_return"] = rng.permutation(work["net_return"].to_numpy())

    predictions = online_predict_and_update(
        work,
        features=list(FEATURES),
        label_col="net_return",
        signal_time_col="signal_time",
        ready_time_col="exit_time",
        process_var=process_var,
        obs_var=obs_var,
        prior_var=prior_var,
        warmup_updates=warmup_updates,
    )
    work["predicted_return"] = predictions
    qualified = work.dropna(subset=["predicted_return"])
    qualified = qualified[qualified.predicted_return >= cfg.return_threshold_bps / 10000]
    winners = (
        qualified.sort_values(["signal_time", "predicted_return"], ascending=[True, False])
        .groupby("signal_time", as_index=False)
        .head(1)
        .sort_values("signal_time")
        .reset_index(drop=True)
    )
    return winners


def main() -> None:
    p = argparse.ArgumentParser(description="Online Kalman-filter regression candidate (Track B)")
    p.add_argument("--config", default="configs/real_data.yaml")
    p.add_argument("--provider", choices=["yfinance", "csv"], default="csv")
    p.add_argument("--csv-dir", default="data/real")
    p.add_argument("--output", default="outputs/equity_kalman_online")
    p.add_argument("--process-var", type=float, default=1e-5, help="Coefficient random-walk variance (Q diag). 0 = plain recursive least squares.")
    p.add_argument("--obs-var", type=float, default=1.0, help="Observation-noise variance (r).")
    p.add_argument("--prior-var", type=float, default=1.0, help="Initial coefficient covariance (P0 diag); larger = weaker prior.")
    p.add_argument("--warmup", type=int, default=200, help="Minimum updates before the filter is trusted to trade (mirrors the len(train)<200 Ridge guard).")
    p.add_argument("--shuffle-labels", action="store_true", help="Sanity check: permute labels before the online pass; a still-profitable result is not trustworthy.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config))
    symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = (
        download_yfinance(symbols, cfg.start, cfg.end, cfg.interval)
        if args.provider == "yfinance"
        else load_csv_dir(symbols, ROOT / args.csv_dir)
    )

    candidates = build_candidates(frames, cfg.benchmark, cfg)
    if candidates.empty:
        raise RuntimeError("No candidates generated")
    candidates["signal_time"] = pd.to_datetime(candidates["signal_time"], utc=True)
    candidates["exit_time"] = pd.to_datetime(candidates["exit_time"], utc=True)

    winners = select_kalman_winners(
        candidates, cfg, args.process_var, args.obs_var, args.prior_var,
        args.warmup, args.shuffle_labels, args.seed,
    )
    if winners.empty:
        raise RuntimeError("No trades selected; lower --warmup, raise --prior-var, or check the threshold")

    path, summary = simulate_capital(winners, cfg)
    start = pd.Timestamp(winners.signal_time.min())
    end = pd.Timestamp(winners.exit_time.max())
    benchmark = benchmark_summary(frames[cfg.benchmark], start, end, cfg.initial_capital)

    out = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out / "candidates.parquet", index=False)
    winners.to_csv(out / "selected_trades.csv", index=False)
    path.to_csv(out / "capital_path.csv", index=False)
    report = {
        "model": "kalman_online",
        "configuration": {
            **cfg.__dict__,
            "process_var": args.process_var,
            "obs_var": args.obs_var,
            "prior_var": args.prior_var,
            "warmup_updates": args.warmup,
            "shuffle_labels": args.shuffle_labels,
            "seed": args.seed,
        },
        "strategy": summary,
        "benchmark": benchmark,
        "options_backtest": "disabled: requires historical option-chain data",
    }
    (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
