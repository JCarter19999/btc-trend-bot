"""Does a non-linear model find real predictive signal where Ridge and the
Kalman-filter (both linear/linear-adaptive) didn't? Both prior ML
candidates in this project were killed against a proper random-selection
control -- Ridge lands 2nd-10th percentile, Kalman is statistically
indistinguishable from random (EQUITY_EXPECTANCY_MATRIX_FINDINGS.md,
EQUITY_KALMAN_ONLINE_REGRESSION.md). This tests whether that's a
statement about linear models specifically, or about the whole
"these features predict forward returns" premise -- using the EXACT same
candidate pool, features, threshold-based selection rule, chronological
folds, and purging as the Ridge walk-forward, swapping only the model
class, so a difference in result is attributable to the model, not a
methodology change.

Two non-linear families (sklearn only -- no xgboost/lightgbm installed):
HistGradientBoostingRegressor (histogram-based gradient boosting, the
built-in analog to LightGBM) and RandomForestRegressor (bagged trees,
a structurally different non-linear family, included so a positive
result isn't just "one particular boosting implementation got lucky").

Same rigor as every other model candidate in this project: label-shuffle
sanity check (does fitting on permuted labels still look profitable? if
so, don't trust the real result) AND the stronger random-selection
control (50-seed uniform-random candidate choice, same pool, same exit
mechanics) -- label-shuffling alone was already shown too weak a control
for the Kalman candidate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402
from run_equity_real_data_walkforward import (  # noqa: E402
    FEATURES, build_candidates, load_config, load_csv_dir, _select_fold_winners,
)

SIZING = SizingMode("fixed_notional_2500", "fixed_notional", 2500.0)
N_RANDOM_SEEDS = 100

MODEL_FACTORIES = {
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.05, random_state=0),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=0, n_jobs=-1),
}


def _select_fold_winners_ml(train: pd.DataFrame, test: pd.DataFrame, cfg, fold: int,
                             rng: np.random.Generator, shuffle_labels: bool, model_factory) -> pd.DataFrame:
    """Mirrors _select_fold_winners's ridge branch exactly -- same
    StandardScaler pre-step, same return_threshold_bps qualification, same
    per-signal_time best-prediction selection -- only the regressor differs."""
    test = test.copy()
    test["fold"] = fold
    train_labels = rng.permutation(train.net_return.to_numpy()) if shuffle_labels else train.net_return
    model = Pipeline([("scale", StandardScaler()), ("model", model_factory())])
    model.fit(train[FEATURES], train_labels)
    test["predicted_return"] = model.predict(test[FEATURES])
    qualified = test[test.predicted_return >= cfg.return_threshold_bps / 10000]
    return (qualified.sort_values(["signal_time", "predicted_return"], ascending=[True, False])
            .groupby("signal_time", as_index=False).head(1).sort_values("signal_time"))


def walk_forward_ml(candidates: pd.DataFrame, cfg, model_factory, shuffle_labels: bool = False, seed: int = 0) -> pd.DataFrame:
    dates = np.array(sorted(candidates.signal_time.dt.normalize().unique()))
    selected = []
    start = cfg.train_bars
    rng = np.random.default_rng(seed)
    while start + cfg.test_bars <= len(dates):
        train_start = max(0, start - cfg.train_bars)
        train_end = max(train_start, start - cfg.purge_bars)
        train_dates = dates[train_start:train_end]
        test_dates = dates[start:start + cfg.test_bars]
        train = candidates[candidates.signal_time.dt.normalize().isin(train_dates)].dropna(subset=FEATURES + ["net_return"])
        test = candidates[candidates.signal_time.dt.normalize().isin(test_dates)].dropna(subset=FEATURES)
        if len(train) < 200 or test.empty:
            start += cfg.step_bars
            continue
        winners = _select_fold_winners_ml(train, test, cfg, len(selected), rng, shuffle_labels, model_factory)
        selected.append(winners)
        start += cfg.step_bars
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def walk_forward_random_control(candidates: pd.DataFrame, cfg, seed: int) -> pd.DataFrame:
    """Same fold/purge structure, but selection is uniform-random from the
    same candidate pool -- no model in the loop. Stronger control than
    label-shuffling; this is what actually killed Ridge and Kalman."""
    dates = np.array(sorted(candidates.signal_time.dt.normalize().unique()))
    selected = []
    start = cfg.train_bars
    rng = np.random.default_rng(seed)
    while start + cfg.test_bars <= len(dates):
        test_dates = dates[start:start + cfg.test_bars]
        test = candidates[candidates.signal_time.dt.normalize().isin(test_dates)].dropna(subset=FEATURES)
        if test.empty:
            start += cfg.step_bars
            continue
        winners = _select_fold_winners(pd.DataFrame(), test, cfg, len(selected), rng, False, "random")
        selected.append(winners)
        start += cfg.step_bars
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def summarize(trades: pd.DataFrame, label: str, cfg) -> dict:
    if trades.empty:
        return {"label": label, "trades_taken": 0}
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    _, summary = simulate_single_position(t, cfg, SIZING)
    summary["label"] = label
    return summary


def main() -> None:
    cfg = load_config(ROOT / "configs/real_data.yaml")
    symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(symbols, ROOT / "data/real")

    print("Building candidate pool (same mask/features as Ridge)...")
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    candidates["signal_time"] = pd.to_datetime(candidates["signal_time"], utc=True)
    print(f"{len(candidates)} candidate signals")

    results = {}
    for model_name, factory in MODEL_FACTORIES.items():
        print(f"\n=== {model_name} ===")

        real_trades = walk_forward_ml(candidates, cfg, factory, shuffle_labels=False, seed=0)
        real_summary = summarize(real_trades, f"{model_name}_real", cfg)
        print(f"real labels: trades={real_summary.get('trades_taken',0)} "
              f"expectancy={real_summary.get('expectancy_bps',float('nan')):.1f}bps "
              f"win={real_summary.get('win_rate',float('nan')):.3f}")

        shuffled_trades = walk_forward_ml(candidates, cfg, factory, shuffle_labels=True, seed=0)
        shuffled_summary = summarize(shuffled_trades, f"{model_name}_shuffled", cfg)
        print(f"shuffled labels: trades={shuffled_summary.get('trades_taken',0)} "
              f"expectancy={shuffled_summary.get('expectancy_bps',float('nan')):.1f}bps")

        print(f"Running {N_RANDOM_SEEDS}-seed random-selection control...")
        random_summaries = [summarize(walk_forward_random_control(candidates, cfg, seed), f"random_{seed}", cfg)
                             for seed in range(N_RANDOM_SEEDS)]
        random_bps = [s["expectancy_bps"] for s in random_summaries if s.get("trades_taken", 0) > 0]
        pct = float((np.array(random_bps) < real_summary.get("expectancy_bps", -1e9)).mean() * 100) if random_bps else None

        print(f"real vs random: {real_summary.get('expectancy_bps', float('nan')):.1f}bps vs "
              f"mean={np.mean(random_bps):.1f}bps (std={np.std(random_bps):.1f}) -> percentile {pct}")

        results[model_name] = {
            "real": real_summary, "shuffled": shuffled_summary,
            "random_mean_bps": float(np.mean(random_bps)) if random_bps else None,
            "random_std_bps": float(np.std(random_bps)) if random_bps else None,
            "percentile_vs_random": pct,
        }

    out = ROOT / "outputs" / "equity_nonlinear_walkforward"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")
    print("\nReference points: Ridge = 2nd-10th percentile (killed). simple_trend = 99th percentile (deployed).")


if __name__ == "__main__":
    main()
