"""Research-only fixed-entry exit laboratory for selective RSI mean reversion.

All active variants use the same v0.8 15-minute strict entry state machine.
Only exits differ, allowing causal evaluation of profit capture without entry drift.
No authenticated exchange or live-order path exists here.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from btc_trend_bot import popular_matrix as base
from btc_trend_bot import selective_reversion_matrix as selective
from btc_trend_bot.data import download_ohlcv, load_ohlcv_csv, timeframe_to_timedelta


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Exit-lab config must be a mapping")
    for section in ("market", "features", "costs", "research", "entry_template", "strategies"):
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")
    ids = [str(s["id"]) for s in cfg["strategies"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Strategy IDs must be unique")
    return cfg


def _expanded_strategies(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in cfg["strategies"]:
        strategy = copy.deepcopy(raw)
        if strategy.get("type") == "exit_variant":
            merged = copy.deepcopy(cfg["entry_template"])
            merged.update(strategy)
            merged["type"] = "selective_rsi2"
            strategy = merged
        result.append(strategy)
    return result


def build_feature_frame(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    selective_cfg = copy.deepcopy(cfg)
    selective_cfg["strategies"] = _expanded_strategies(cfg)
    return selective.build_feature_frame(frame, selective_cfg)


def _finite(value: Any, default: float = 0.0) -> float:
    return base._finite(value, default)


def _exit_decision(strategy: dict[str, Any], row: pd.Series, state: base.StrategyState, index: int) -> base.Decision:
    sid = str(strategy["id"])
    held = base._bars_held(state, index)
    close = float(row["close"])
    entry = _finite(state.entry_mark, close)
    high = max(_finite(state.highest_high, close), float(row["high"]))
    return_bps = (close / entry - 1.0) * 10_000.0 if entry > 0 else 0.0
    high_bps = (high / entry - 1.0) * 10_000.0 if entry > 0 else 0.0
    rsi = _finite(row["m15_rsi2"], 50.0)
    mode = str(strategy.get("exit_mode", "control"))
    minhold = int(strategy.get("minimum_hold_bars_5m", 24))
    maxhold = int(strategy.get("maximum_hold_bars_5m", 288))

    if held >= maxhold:
        return base.Decision(sid, 0.0, "exit_time", "maximum hold")

    if mode == "control":
        atr = _finite(row["m15_atr"])
        if held >= minhold and atr > 0 and close <= high - float(strategy.get("trailing_stop_atr", 3.25)) * atr:
            return base.Decision(sid, 0.0, "exit_trailing", "control ATR trailing stop")
        regime_fail = (
            _finite(row["h4_trend_bps"]) <= float(strategy.get("exit_h4_trend_bps", -25))
            or _finite(row["d1_momentum_bps"]) <= float(strategy.get("exit_d1_momentum_bps", -250))
        )
        if held >= minhold and regime_fail:
            return base.Decision(sid, 0.0, "exit_regime", "control regime failure")
        if held >= minhold and rsi >= float(strategy.get("exit_rsi", 80)):
            return base.Decision(sid, 0.0, "exit_recovery", "control RSI recovery")

    elif mode == "fixed_target":
        if return_bps <= -float(strategy["stop_bps"]):
            return base.Decision(sid, 0.0, "exit_stop", "fixed protective stop")
        if return_bps >= float(strategy["target_bps"]):
            return base.Decision(sid, 0.0, "exit_target", "fixed profit target")

    elif mode == "activated_trailing":
        if return_bps <= -float(strategy["stop_bps"]):
            return base.Decision(sid, 0.0, "exit_stop", "initial protective stop")
        activation = float(strategy["activation_bps"])
        if high_bps >= activation:
            trail_price = high * (1.0 - float(strategy["trail_bps"]) / 10_000.0)
            if close <= trail_price:
                return base.Decision(sid, 0.0, "exit_activated_trail", "activated price trail")

    elif mode == "rsi_recovery":
        if return_bps <= -float(strategy.get("stop_bps", 50)):
            return base.Decision(sid, 0.0, "exit_stop", "RSI variant protective stop")
        if held >= minhold and rsi >= float(strategy["exit_rsi"]):
            return base.Decision(sid, 0.0, "exit_recovery", "RSI recovery target")

    elif mode == "time_capture":
        if return_bps <= -float(strategy.get("stop_bps", 50)):
            return base.Decision(sid, 0.0, "exit_stop", "time variant protective stop")
        target = float(strategy.get("target_bps", 25))
        if return_bps >= target:
            return base.Decision(sid, 0.0, "exit_target", "time variant profit target")

    elif mode == "breakeven_lock":
        # Preserve the wide initial stop needed by mean reversion, but once the
        # trade has shown enough favorable excursion, refuse to surrender the
        # entire rebound.  All checks are causal and use only the current close
        # plus the highest high observed so far.
        if return_bps <= -float(strategy.get("stop_bps", 65)):
            return base.Decision(sid, 0.0, "exit_stop", "breakeven-lock protective stop")
        target = float(strategy.get("target_bps", 30))
        if return_bps >= target:
            return base.Decision(sid, 0.0, "exit_target", "breakeven-lock profit target")
        activation = float(strategy.get("lock_activation_bps", 20))
        lock_bps = float(strategy.get("lock_bps", 0))
        if high_bps >= activation and return_bps <= lock_bps:
            return base.Decision(sid, 0.0, "exit_profit_lock", "activated profit lock")

    else:
        raise ValueError(f"Unknown exit_mode: {mode}")

    return base.Decision(sid, 1.0, "hold_btc", f"exit lab hold: {mode}")


def decide(
    strategy: dict[str, Any],
    row: pd.Series,
    state: base.StrategyState,
    index: int,
    costs: dict[str, Any],
    setup: selective.SelectiveSetupState | None,
    diagnostics: dict[tuple[str, str], int] | None,
) -> base.Decision:
    typ = str(strategy["type"])
    if typ in ("cash", "buy_hold"):
        return selective.decide(strategy, row, state, index, costs, setup, diagnostics)
    if state.target_position <= 0:
        return selective.decide(strategy, row, state, index, costs, setup, diagnostics)
    if setup is not None:
        setup.clear()
    return _exit_decision(strategy, row, state, index)


def _augment_summary(summary: dict[str, dict[str, Any]], episodes: pd.DataFrame) -> None:
    for sid, metrics in summary.items():
        closed = episodes[(episodes["strategy_id"] == sid) & (~episodes["is_open"])]
        if closed.empty:
            metrics.update({
                "average_realized_gross_bps": None,
                "average_mfe_bps": None,
                "mfe_capture_ratio": None,
                "best_five_trade_gross_share": None,
            })
            continue
        realized = closed["gross_price_return_pct"].astype(float)
        mfe = closed["maximum_favorable_excursion_pct"].astype(float)
        realized_sum = float(realized.sum())
        best_five = float(realized.nlargest(min(5, len(realized))).sum())
        avg_realized = float(realized.mean())
        avg_mfe = float(mfe.mean())
        metrics["average_realized_gross_bps"] = avg_realized * 10_000.0
        metrics["average_mfe_bps"] = avg_mfe * 10_000.0
        metrics["mfe_capture_ratio"] = avg_realized / avg_mfe if avg_mfe > 0 else None
        metrics["best_five_trade_gross_share"] = best_five / realized_sum if realized_sum > 0 else None


def simulate(feature_frame: pd.DataFrame, cfg: dict[str, Any], costs_override: dict[str, Any] | None = None) -> selective.SelectiveSimulationResult:
    costs = copy.deepcopy(costs_override if costs_override is not None else cfg["costs"])
    initial = float(cfg["research"].get("initial_cash", 500.0))
    strategies = _expanded_strategies(cfg)
    states = {
        str(s["id"]): base.StrategyState(str(s["id"]), initial, initial, 0.0, initial, 0.0, peak_equity=initial)
        for s in strategies
    }
    setups = {str(s["id"]): selective.SelectiveSetupState() for s in strategies if str(s["type"]) == "selective_rsi2"}
    diagnostics: dict[tuple[str, str], int] = {}
    valid = feature_frame.index[feature_frame["feature_valid"]].tolist()
    if not valid:
        raise RuntimeError("No rows contain a complete feature history")
    snapshots: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for index in range(valid[0], len(feature_frame) - 1):
        row = feature_frame.iloc[index]
        next_row = feature_frame.iloc[index + 1]
        if not bool(row["feature_valid"]):
            continue
        for strategy in strategies:
            sid = str(strategy["id"])
            state = states[sid]
            if state.target_position > 0:
                state.highest_high = max(_finite(state.highest_high), float(row["high"]))
            decision = decide(strategy, row, state, index, costs, setups.get(sid), diagnostics)
            snapshot, trade = base.execute_decision(state, decision, row, next_row, index, costs)
            snapshots.append(snapshot)
            if trade is not None:
                trades.append(asdict(trade))
    sf = pd.DataFrame(snapshots)
    tf = pd.DataFrame(trades)
    episodes = base.build_trade_episodes(sf, initial, 5)
    days = max((pd.Timestamp(sf["execution_timestamp"].max()) - pd.Timestamp(sf["execution_timestamp"].min())).total_seconds() / 86400, 1e-9)
    summary = base._summarize(sf, tf, episodes, initial, days, costs)
    _augment_summary(summary, episodes)
    gates = pd.DataFrame([{"strategy_id": sid, "gate": gate, "count": count} for (sid, gate), count in sorted(diagnostics.items())])
    return selective.SelectiveSimulationResult(sf, tf, episodes, summary, gates)


def cost_grid(feature_frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bps in cfg["research"].get("cost_sensitivity_all_in_bps_per_side", [3, 5, 8, 10, 12]):
        override = {
            "fee_bps_per_side": float(bps),
            "slippage_bps_per_side": 0.0,
            "assumed_spread_bps_per_side": 0.0,
            "min_notional": cfg["costs"].get("min_notional", 10),
            "rebalance_tolerance_bps": cfg["costs"].get("rebalance_tolerance_bps", 1),
        }
        result = simulate(feature_frame, cfg, override)
        for sid, m in result.summary.items():
            rows.append({
                "strategy_id": sid,
                "all_in_bps_per_side": bps,
                "net_return_pct": m["net_return_pct"],
                "gross_return_pct": m["gross_return_pct"],
                "max_drawdown": m["max_drawdown"],
                "round_trips_closed": m["round_trips_closed"],
                "average_realized_gross_bps": m["average_realized_gross_bps"],
                "mfe_capture_ratio": m["mfe_capture_ratio"],
            })
    return pd.DataFrame(rows)


def chronological_folds(feature_frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    fold_count = int(cfg["research"].get("chronological_folds", 5))
    valid = feature_frame[feature_frame["feature_valid"]].copy().reset_index(drop=True)
    if fold_count < 2 or len(valid) < fold_count * 100:
        return pd.DataFrame()
    boundaries = np.linspace(0, len(valid), fold_count + 1, dtype=int)
    rows: list[dict[str, Any]] = []
    history = int(cfg["research"].get("fold_history_rows", 1000))
    for fold in range(fold_count):
        score_start, score_end = boundaries[fold], boundaries[fold + 1]
        history_start = max(0, score_start - history)
        frame = valid.iloc[history_start:score_end].copy().reset_index(drop=True)
        prior = score_start - history_start
        if prior > 0:
            frame.loc[: prior - 1, "feature_valid"] = False
        result = simulate(frame, cfg)
        for sid, m in result.summary.items():
            rows.append({
                "fold": fold + 1,
                "strategy_id": sid,
                "gross_return_pct": m["gross_return_pct"],
                "net_return_pct": m["net_return_pct"],
                "max_drawdown": m["max_drawdown"],
                "round_trips_closed": m["round_trips_closed"],
                "average_realized_gross_bps": m["average_realized_gross_bps"],
            })
    return pd.DataFrame(rows)


def run_research(config_path: str, bars_requested: int, output_dir: str, data_path: str | None = None, skip_cost_grid: bool = False, compact: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path)
    market = cfg["market"]
    timeframe = str(market["timeframe"])
    if data_path:
        raw, _ = load_ohlcv_csv(data_path, timeframe=timeframe)
        raw = raw.tail(bars_requested).reset_index(drop=True)
    else:
        step = timeframe_to_timedelta(timeframe)
        start = (pd.Timestamp.now(tz="UTC") - step * (bars_requested + 400)).isoformat()
        raw = download_ohlcv(exchange_id=str(market["exchange"]), symbol=str(market["symbol"]), timeframe=timeframe, start=start, max_bars=bars_requested)
    features = build_feature_frame(raw, cfg)
    result = simulate(features, cfg)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not compact:
        features.to_csv(out / "feature_frame.csv", index=False)
        result.snapshots.to_csv(out / "strategy_equity.csv", index=False)
    result.transactions.to_csv(out / "transactions.csv", index=False)
    result.episodes.to_csv(out / "trade_episodes.csv", index=False)
    comparison = pd.DataFrame.from_dict(result.summary, orient="index").rename_axis("strategy_id").reset_index()
    comparison.to_csv(out / "strategy_comparison.csv", index=False)
    result.gate_counts.to_csv(out / "gate_counts.csv", index=False)
    folds = chronological_folds(features, cfg)
    folds.to_csv(out / "chronological_folds.csv", index=False)
    if not skip_cost_grid:
        cost_grid(features, cfg).to_csv(out / "cost_sensitivity.csv", index=False)
    valid = features[features["feature_valid"]]
    first, last = pd.Timestamp(valid["timestamp"].iloc[0]), pd.Timestamp(valid["timestamp"].iloc[-1])
    summary = {
        "market": market,
        "downloaded_bars": len(raw),
        "scored_bars": max(len(valid) - 1, 0),
        "first_scored_bar": first.isoformat(),
        "last_scored_bar": last.isoformat(),
        "sample_days": max((last - first).total_seconds() / 86400, 1e-9),
        "fixed_entry": "v0.8 rsi2_selective_15m_strict",
        "compact_outputs": compact,
        "cost_assumptions": cfg["costs"],
        "strategies": result.summary,
        "outputs": {
            "strategy_comparison": "strategy_comparison.csv",
            "cost_sensitivity": None if skip_cost_grid else "cost_sensitivity.csv",
            "chronological_folds": "chronological_folds.csv",
            "trade_episodes": "trade_episodes.csv",
            "transactions": "transactions.csv",
            "gate_counts": "gate_counts.csv",
        },
    }
    (out / "research_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fixed-entry selective RSI exit laboratory")
    parser.add_argument("--config", default="config/settings_exit_lab.yaml")
    parser.add_argument("--bars", type=int, default=10_000)
    parser.add_argument("--output", default="outputs/exit_lab_10000")
    parser.add_argument("--data", default=None)
    parser.add_argument("--skip-cost-grid", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_research(args.config, args.bars, args.output, args.data, args.skip_cost_grid, args.compact)


if __name__ == "__main__":
    main()
