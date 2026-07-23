"""Fast, research-only exit hyperparameter laboratory.

The module intentionally keeps the v0.8 15-minute strict entry template fixed.
It uses deterministic random search plus successive halving across chronological
train/validation/test partitions.  The test partition is never used to rank the
full trial population; it is opened only for the small finalist set.

This is not an auto-deployer and contains no authenticated exchange path.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from btc_trend_bot.data import download_ohlcv, load_ohlcv_csv, timeframe_to_timedelta
from btc_trend_bot import exit_lab


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Hyperparameter-lab config must be a mapping")
    for section in ("market", "features", "costs", "research", "entry_template", "baseline", "search_space"):
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")
    return cfg


def _sample_value(spec: Any, rng: random.Random) -> Any:
    if isinstance(spec, list):
        if not spec:
            raise ValueError("Search-space lists cannot be empty")
        return copy.deepcopy(rng.choice(spec))
    if not isinstance(spec, dict):
        return copy.deepcopy(spec)
    kind = str(spec.get("type", "choice"))
    if kind == "choice":
        values = spec.get("values", [])
        if not values:
            raise ValueError("choice requires non-empty values")
        return copy.deepcopy(rng.choice(values))
    if kind == "int":
        low, high = int(spec["low"]), int(spec["high"])
        step = int(spec.get("step", 1))
        values = list(range(low, high + 1, step))
        return rng.choice(values)
    if kind == "float":
        low, high = float(spec["low"]), float(spec["high"])
        step = spec.get("step")
        if step is None:
            return rng.uniform(low, high)
        n = int(round((high - low) / float(step)))
        return low + rng.randint(0, n) * float(step)
    raise ValueError(f"Unknown search-space type: {kind}")


def sample_trial(trial_id: int, cfg: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    strategy = {
        "id": f"trial_{trial_id:04d}",
        "type": "exit_variant",
    }
    for key, spec in cfg["search_space"].items():
        strategy[key] = _sample_value(spec, rng)

    mode = str(strategy["exit_mode"])
    # Remove parameters irrelevant to the chosen mode.  This keeps exported
    # champion configs readable and prevents accidental hidden interactions.
    allowed = {
        "time_capture": {"target_bps", "stop_bps", "maximum_hold_bars_5m", "minimum_hold_bars_5m"},
        "breakeven_lock": {"target_bps", "stop_bps", "maximum_hold_bars_5m", "minimum_hold_bars_5m", "lock_activation_bps", "lock_bps"},
        "activated_trailing": {"stop_bps", "maximum_hold_bars_5m", "minimum_hold_bars_5m", "activation_bps", "trail_bps"},
        "fixed_target": {"target_bps", "stop_bps", "maximum_hold_bars_5m", "minimum_hold_bars_5m"},
    }
    if mode not in allowed:
        raise ValueError(f"Unsupported tuned exit mode: {mode}")
    for key in list(strategy):
        if key not in {"id", "type", "exit_mode"} | allowed[mode]:
            strategy.pop(key)
    return strategy


def _single_strategy_cfg(cfg: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    run_cfg = copy.deepcopy(cfg)
    run_cfg["strategies"] = [copy.deepcopy(strategy)]
    return run_cfg


def _split_feature_frame(features: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    valid = features[features["feature_valid"]].copy().reset_index(drop=True)
    ratios = cfg["research"].get("split_ratios", [0.60, 0.20, 0.20])
    if len(ratios) != 3 or not math.isclose(sum(map(float, ratios)), 1.0, abs_tol=1e-9):
        raise ValueError("research.split_ratios must contain three values summing to 1")
    n = len(valid)
    a = int(n * float(ratios[0]))
    b = a + int(n * float(ratios[1]))
    if min(a, b - a, n - b) < 500:
        raise ValueError("Not enough scored rows for train/validation/test tuning")
    return {
        "train": valid.iloc[:a].copy().reset_index(drop=True),
        "validation": valid.iloc[a:b].copy().reset_index(drop=True),
        "test": valid.iloc[b:].copy().reset_index(drop=True),
        "full": valid,
    }


def _metric_score(metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    objective = cfg["research"].get("objective", {})
    break_even = float(metrics.get("gross_break_even_all_in_bps_per_side") or 0.0)
    avg_gross = float(metrics.get("average_realized_gross_bps") or -999.0)
    net_return = float(metrics.get("net_return_pct") or 0.0)
    drawdown = abs(float(metrics.get("max_drawdown") or 0.0))
    trades = int(metrics.get("round_trips_closed") or 0)
    concentration = metrics.get("best_five_trade_gross_share")
    concentration = float(concentration) if concentration is not None and math.isfinite(float(concentration)) else 1.0

    score = (
        float(objective.get("break_even_weight", 1.0)) * break_even
        + float(objective.get("average_gross_weight", 0.20)) * avg_gross
        + float(objective.get("net_return_weight", 25.0)) * net_return
        - float(objective.get("drawdown_weight", 15.0)) * drawdown
    )
    min_trades = int(objective.get("minimum_trades", 12))
    if trades < min_trades:
        score -= float(objective.get("missing_trade_penalty", 1.0)) * (min_trades - trades)
    max_concentration = float(objective.get("maximum_best_five_share", 0.70))
    if concentration > max_concentration:
        score -= float(objective.get("concentration_penalty", 5.0)) * (concentration - max_concentration)
    return score


def evaluate(strategy: dict[str, Any], frame: pd.DataFrame, cfg: dict[str, Any], split: str) -> dict[str, Any]:
    run_cfg = _single_strategy_cfg(cfg, strategy)
    result = exit_lab.simulate(frame, run_cfg)
    metrics = result.summary[str(strategy["id"])]
    row = {
        "trial_id": str(strategy["id"]),
        "split": split,
        "score": _metric_score(metrics, cfg),
        **{k: v for k, v in strategy.items() if k not in ("id", "type")},
        **metrics,
    }
    return row


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: float(r["score"]), reverse=True)


def _download(cfg: dict[str, Any], bars: int, data_path: str | None) -> pd.DataFrame:
    timeframe = str(cfg["market"]["timeframe"])
    if data_path:
        raw, _ = load_ohlcv_csv(data_path, timeframe=timeframe)
        return raw.tail(bars).reset_index(drop=True)
    step = timeframe_to_timedelta(timeframe)
    start = (pd.Timestamp.now(tz="UTC") - step * (bars + 400)).isoformat()
    return download_ohlcv(
        exchange_id=str(cfg["market"]["exchange"]),
        symbol=str(cfg["market"]["symbol"]),
        timeframe=timeframe,
        start=start,
        max_bars=bars,
    )


def _params_from_row(row: dict[str, Any], trials: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return copy.deepcopy(trials[str(row["trial_id"])])


def run_tuning(config_path: str, bars: int, output_dir: str, data_path: str | None = None, trials_override: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw = _download(cfg, bars, data_path)
    features = exit_lab.build_feature_frame(raw, cfg)
    splits = _split_feature_frame(features, cfg)

    research = cfg["research"]
    n_trials = int(trials_override or research.get("trials", 120))
    validation_candidates = min(int(research.get("validation_candidates", 24)), n_trials)
    test_candidates = min(int(research.get("test_candidates", 6)), validation_candidates)
    rng = random.Random(int(research.get("seed", 42)))

    trial_map: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    while len(trial_map) < n_trials:
        candidate = sample_trial(len(trial_map) + 1, cfg, rng)
        signature = json.dumps({k: v for k, v in candidate.items() if k not in ("id", "type")}, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        trial_map[str(candidate["id"])] = candidate

    baseline = copy.deepcopy(cfg["baseline"])
    baseline.setdefault("id", "original_goal_time144_t25_s65")
    baseline.setdefault("type", "exit_variant")

    train_rows = [evaluate(s, splits["train"], cfg, "train") for s in trial_map.values()]
    baseline_rows = [evaluate(baseline, splits[name], cfg, name) for name in ("train", "validation", "test", "full")]

    promoted_validation = _rank(train_rows)[:validation_candidates]
    validation_rows = [evaluate(_params_from_row(r, trial_map), splits["validation"], cfg, "validation") for r in promoted_validation]
    val_by_id = {r["trial_id"]: r for r in validation_rows}

    # Rank finalists primarily by unseen validation performance, with a small
    # stability contribution from train to avoid selecting a single lucky fold.
    finalist_ranking: list[dict[str, Any]] = []
    train_by_id = {r["trial_id"]: r for r in train_rows}
    for row in validation_rows:
        combined = copy.deepcopy(row)
        combined["selection_score"] = float(row["score"]) + 0.20 * float(train_by_id[row["trial_id"]]["score"])
        finalist_ranking.append(combined)
    finalist_ranking.sort(key=lambda r: float(r["selection_score"]), reverse=True)
    finalists = finalist_ranking[:test_candidates]

    test_rows = [evaluate(_params_from_row(r, trial_map), splits["test"], cfg, "test") for r in finalists]
    full_rows = [evaluate(_params_from_row(r, trial_map), splits["full"], cfg, "full") for r in finalists]

    all_rows = train_rows + validation_rows + test_rows + full_rows + baseline_rows
    leaderboard = pd.DataFrame(all_rows)
    leaderboard.to_csv(out / "leaderboard.csv", index=False)

    test_by_id = {r["trial_id"]: r for r in test_rows}
    full_by_id = {r["trial_id"]: r for r in full_rows}
    selected = finalists[0]
    selected_id = str(selected["trial_id"])
    champion_strategy = trial_map[selected_id]

    champion = {
        "selection_rule": "highest validation score plus 20% train stability; test opened only after selection",
        "strategy": champion_strategy,
        "train": train_by_id[selected_id],
        "validation": val_by_id[selected_id],
        "test": test_by_id[selected_id],
        "full": full_by_id[selected_id],
        "baseline": {r["split"]: r for r in baseline_rows},
    }
    (out / "champion.json").write_text(json.dumps(champion, indent=2, allow_nan=False), encoding="utf-8")

    # The report makes overfitting visible rather than hiding it: compare the
    # best train trial, the validation-selected champion, and the fixed baseline.
    best_train = _rank(train_rows)[0]
    best_train_id = str(best_train["trial_id"])
    if best_train_id in val_by_id:
        best_train_validation = val_by_id[best_train_id]
    else:
        best_train_validation = evaluate(trial_map[best_train_id], splits["validation"], cfg, "validation")
    best_train_test = evaluate(trial_map[best_train_id], splits["test"], cfg, "test")
    overfit_report = {
        "best_train_trial": {
            "strategy": trial_map[best_train_id],
            "train": best_train,
            "validation": best_train_validation,
            "test": best_train_test,
            "score_decay_train_to_test": float(best_train["score"]) - float(best_train_test["score"]),
        },
        "validation_selected_champion": champion,
        "interpretation": "Large train-to-validation/test decay is direct evidence of selection overfit. Test results must not be used for another tuning round.",
    }
    (out / "overfit_report.json").write_text(json.dumps(overfit_report, indent=2, allow_nan=False), encoding="utf-8")

    # Ready-to-run config for the chosen candidate, while preserving the
    # original goal candidate alongside it for honest comparison.
    deploy_cfg = copy.deepcopy(cfg)
    deploy_cfg["strategies"] = [
        {"id": "cash_5m", "type": "cash"},
        {"id": "buy_hold_5m", "type": "buy_hold"},
        {**copy.deepcopy(baseline), "id": "original_goal_time144_t25_s65"},
        {**copy.deepcopy(champion_strategy), "id": "tuned_validation_champion"},
    ]
    for key in ("baseline", "search_space"):
        deploy_cfg.pop(key, None)
    deploy_cfg["research"].pop("objective", None)
    (out / "champion_exit_lab.yaml").write_text(yaml.safe_dump(deploy_cfg, sort_keys=False), encoding="utf-8")

    summary = {
        "downloaded_bars": len(raw),
        "scored_rows": len(splits["full"]),
        "split_rows": {name: len(frame) for name, frame in splits.items() if name != "full"},
        "trial_count": n_trials,
        "validation_candidates": validation_candidates,
        "test_candidates": test_candidates,
        "seed": int(research.get("seed", 42)),
        "elapsed_seconds": time.time() - started,
        "original_goal": baseline,
        "champion": champion,
        "outputs": {
            "leaderboard": "leaderboard.csv",
            "champion": "champion.json",
            "overfit_report": "overfit_report.json",
            "champion_exit_lab_config": "champion_exit_lab.yaml",
        },
    }
    (out / "tuning_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Successive-halving exit hyperparameter laboratory")
    parser.add_argument("--config", default="config/settings_hyperparameter_lab.yaml")
    parser.add_argument("--bars", type=int, default=50_000)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--output", default="outputs/hyperparameter_lab_50000")
    parser.add_argument("--data", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_tuning(args.config, args.bars, args.output, args.data, args.trials)


if __name__ == "__main__":
    main()
