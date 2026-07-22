"""Research-only intraday BTC strategy matrix.

This module compares several five-minute strategy families under one common,
causal, cost-aware simulation. It intentionally contains no live-order or
exchange-authentication path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from btc_trend_bot.data import (
    download_ohlcv,
    load_ohlcv_csv,
    normalize_ohlcv,
    timeframe_to_timedelta,
)


@dataclass
class StrategyState:
    strategy_id: str
    initial_cash: float
    cash: float
    btc: float
    gross_cash: float
    gross_btc: float
    target_position: float = 0.0
    peak_equity: float | None = None
    total_fees: float = 0.0
    total_spread: float = 0.0
    total_slippage: float = 0.0
    total_turnover: float = 0.0
    trade_count: int = 0
    last_trade_index: int | None = None
    entry_index: int | None = None
    entry_mark: float | None = None
    highest_high: float | None = None
    entry_reference: float | None = None
    pending_entry_index: int | None = None


@dataclass(frozen=True)
class Decision:
    strategy_id: str
    target_position: float
    signal: str
    reason: str
    entry_reference: float | None = None


@dataclass(frozen=True)
class SimulatedTrade:
    strategy_id: str
    signal_timestamp: str
    execution_timestamp: str
    side: str
    mark_price: float
    reference_price: float
    fill_price: float
    btc_delta: float
    gross_notional: float
    fee: float
    spread_cost: float
    slippage_cost: float
    cash_after: float
    btc_after: float
    equity_after: float
    signal: str
    reason: str


@dataclass(frozen=True)
class SimulationResult:
    snapshots: pd.DataFrame
    transactions: pd.DataFrame
    episodes: pd.DataFrame
    summary: dict[str, dict[str, Any]]


def load_matrix_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Intraday matrix config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Intraday matrix config must contain a YAML mapping.")
    for section in ("market", "features", "costs", "research", "strategies"):
        if section not in raw:
            raise ValueError(f"Missing intraday matrix config section: {section}")
    strategies = raw["strategies"]
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("strategies must be a non-empty list.")
    ids = [str(item["id"]) for item in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError("Strategy IDs must be unique.")
    return raw


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _ema(series: pd.Series, bars: int) -> pd.Series:
    if bars <= 0:
        raise ValueError("EMA bars must be positive.")
    return series.ewm(span=bars, adjust=False, min_periods=bars).mean()


def _completed_resample(
    frame: pd.DataFrame,
    base_step: pd.Timedelta,
    rule: str,
    expected_source_bars: int,
) -> pd.DataFrame:
    """Build only complete right-labeled higher-timeframe candles.

    The source timestamps are candle opens. Converting them to candle-end times
    makes the right-labeled aggregate available exactly when its final five-minute
    source candle has completed, preventing higher-timeframe lookahead.
    """
    work = frame.copy()
    work["bar_end"] = pd.to_datetime(work["timestamp"], utc=True) + base_step
    indexed = work.set_index("bar_end")
    grouped = indexed.resample(
        rule,
        origin="epoch",
        label="right",
        closed="right",
    )
    aggregated = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped["close"].count()
    aggregated = aggregated.loc[counts == expected_source_bars]
    aggregated = aggregated.dropna().reset_index()
    return aggregated


def build_feature_frame(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    market = cfg["market"]
    feature_cfg = cfg["features"]
    timeframe = str(market["timeframe"])
    step = timeframe_to_timedelta(timeframe)
    if step != pd.Timedelta(minutes=5):
        raise ValueError("The intraday strategy matrix currently requires 5m data.")

    normalized, _ = normalize_ohlcv(frame, timeframe=timeframe)
    out = normalized.copy().reset_index(drop=True)
    out["bar_end"] = pd.to_datetime(out["timestamp"], utc=True) + step

    previous_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - previous_close).abs(),
            (out["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_bars = int(feature_cfg.get("atr_bars_5m", 24))
    out["atr_5m"] = true_range.rolling(atr_bars, min_periods=atr_bars).mean()
    out["range_atr_ratio"] = (
        (out["high"] - out["low"]) / out["atr_5m"].replace(0, np.nan)
    )

    breakout_bars = int(feature_cfg.get("breakout_lookback_bars_5m", 24))
    out["prior_breakout_high"] = (
        out["high"].shift(1).rolling(breakout_bars, min_periods=breakout_bars).max()
    )
    out["prior_breakout_low"] = (
        out["low"].shift(1).rolling(breakout_bars, min_periods=breakout_bars).min()
    )

    volume_window = int(feature_cfg.get("volume_window_bars_5m", 48))
    previous_volume_median = (
        out["volume"]
        .shift(1)
        .rolling(volume_window, min_periods=volume_window)
        .median()
    )
    out["relative_volume"] = out["volume"] / previous_volume_median.replace(0, np.nan)

    out["candle_return_bps"] = (out["close"] / out["open"] - 1.0) * 10_000.0
    out["close_return_bps"] = out["close"].pct_change() * 10_000.0
    out["previous_close_return_bps"] = out["close_return_bps"].shift(1)
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_fraction"] = ((out["close"] - out["open"]).abs() / candle_range).fillna(0.0)

    doji_threshold = float(feature_cfg.get("doji_threshold_bps", 1.0))
    direction = pd.Series(0, index=out.index, dtype="int64")
    direction.loc[out["candle_return_bps"] > doji_threshold] = 1
    direction.loc[out["candle_return_bps"] < -doji_threshold] = -1
    group_change = direction.ne(direction.shift(1)) | direction.eq(0)
    run_group = group_change.cumsum()
    streak = direction.groupby(run_group).cumcount() + 1
    streak = streak.where(direction.ne(0), 0)
    run_open = out["open"].groupby(run_group).transform("first")
    out["streak_direction"] = direction
    out["streak_length"] = streak.astype(int)
    out["run_return_bps"] = (
        ((out["close"] / run_open - 1.0) * 10_000.0).where(direction.ne(0), 0.0)
    )

    ema_bars = int(feature_cfg.get("execution_ema_bars_5m", 12))
    out["execution_ema_5m"] = _ema(out["close"], ema_bars)
    out["execution_ema_deviation_bps"] = (
        (out["close"] / out["execution_ema_5m"] - 1.0) * 10_000.0
    )
    out["recovery_cross_up"] = (
        (out["close"] > out["execution_ema_5m"])
        & (out["close"].shift(1) <= out["execution_ema_5m"].shift(1))
        & (out["candle_return_bps"] > 0.0)
    )
    pullback_window = int(feature_cfg.get("pullback_window_bars_5m", 12))
    out["recent_pullback_bps"] = (
        out["execution_ema_deviation_bps"]
        .rolling(pullback_window, min_periods=pullback_window)
        .min()
    )

    vwap_window = int(feature_cfg.get("vwap_window_bars_5m", 96))
    typical_price = (out["high"] + out["low"] + out["close"]) / 3.0
    rolling_volume = out["volume"].rolling(vwap_window, min_periods=vwap_window).sum()
    rolling_pv = (typical_price * out["volume"]).rolling(
        vwap_window,
        min_periods=vwap_window,
    ).sum()
    out["rolling_vwap"] = rolling_pv / rolling_volume.replace(0, np.nan)
    vwap_deviation = out["close"] - out["rolling_vwap"]
    vwap_std_window = int(feature_cfg.get("vwap_std_window_bars_5m", vwap_window))
    vwap_deviation_std = vwap_deviation.rolling(
        vwap_std_window,
        min_periods=vwap_std_window,
    ).std(ddof=0)
    out["vwap_zscore"] = vwap_deviation / vwap_deviation_std.replace(0, np.nan)
    out["selling_pressure_easing"] = (
        (out["candle_return_bps"] > 0.0)
        & (out["close_return_bps"] > out["previous_close_return_bps"])
    )

    one_hour = _completed_resample(out, step, "1h", expected_source_bars=12)
    one_hour_fast_bars = int(feature_cfg.get("one_hour_fast_ema_bars", 24))
    one_hour_slow_bars = int(feature_cfg.get("one_hour_slow_ema_bars", 72))
    if one_hour_fast_bars >= one_hour_slow_bars:
        raise ValueError("one_hour_fast_ema_bars must be smaller than one_hour_slow_ema_bars")
    one_hour["one_hour_fast_ema"] = _ema(one_hour["close"], one_hour_fast_bars)
    one_hour["one_hour_slow_ema"] = _ema(one_hour["close"], one_hour_slow_bars)
    one_hour["one_hour_ema_spread_bps"] = (
        (one_hour["one_hour_fast_ema"] / one_hour["one_hour_slow_ema"] - 1.0)
        * 10_000.0
    )
    one_hour_momentum_bars = int(feature_cfg.get("one_hour_momentum_bars", 24))
    one_hour["one_hour_momentum_bps"] = (
        (one_hour["close"] / one_hour["close"].shift(one_hour_momentum_bars) - 1.0)
        * 10_000.0
    )
    one_hour = one_hour[
        ["bar_end", "one_hour_ema_spread_bps", "one_hour_momentum_bps"]
    ]

    four_hour = _completed_resample(out, step, "4h", expected_source_bars=48)
    four_hour_momentum_bars = int(feature_cfg.get("four_hour_momentum_bars", 1))
    four_hour["four_hour_momentum_bps"] = (
        (four_hour["close"] / four_hour["close"].shift(four_hour_momentum_bars) - 1.0)
        * 10_000.0
    )
    four_hour = four_hour[["bar_end", "four_hour_momentum_bps"]]

    out = pd.merge_asof(
        out.sort_values("bar_end"),
        one_hour.sort_values("bar_end"),
        on="bar_end",
        direction="backward",
        allow_exact_matches=True,
    )
    out = pd.merge_asof(
        out.sort_values("bar_end"),
        four_hour.sort_values("bar_end"),
        on="bar_end",
        direction="backward",
        allow_exact_matches=True,
    )

    required = [
        "atr_5m",
        "range_atr_ratio",
        "prior_breakout_high",
        "relative_volume",
        "execution_ema_5m",
        "recent_pullback_bps",
        "rolling_vwap",
        "vwap_zscore",
        "one_hour_ema_spread_bps",
        "one_hour_momentum_bps",
        "four_hour_momentum_bps",
    ]
    out["feature_valid"] = out[required].notna().all(axis=1)
    return out.reset_index(drop=True)


def _bars_since_trade(state: StrategyState, index: int) -> int:
    if state.last_trade_index is None:
        return 1_000_000_000
    return max(0, index - state.last_trade_index)


def _bars_held(state: StrategyState, index: int) -> int:
    if state.entry_index is None:
        return 0
    return max(0, index - state.entry_index + 1)


def _one_hour_bullish(row: pd.Series, strategy: dict[str, Any]) -> bool:
    return (
        _finite(row["one_hour_ema_spread_bps"]) >= float(strategy.get("entry_1h_ema_spread_bps", 0.0))
        and _finite(row["one_hour_momentum_bps"]) >= float(strategy.get("entry_1h_momentum_bps", 0.0))
    )


def _one_hour_bearish(row: pd.Series, strategy: dict[str, Any]) -> bool:
    spread_weak = _finite(row["one_hour_ema_spread_bps"]) <= float(
        strategy.get("exit_1h_ema_spread_bps", -10.0)
    )
    momentum_weak = _finite(row["one_hour_momentum_bps"]) <= float(
        strategy.get("exit_1h_momentum_bps", -10.0)
    )
    required = int(strategy.get("exit_confirmations_required", 2))
    if required not in (1, 2):
        raise ValueError("exit_confirmations_required must be 1 or 2")
    return int(spread_weak) + int(momentum_weak) >= required


def decide_strategy(
    strategy: dict[str, Any],
    row: pd.Series,
    state: StrategyState,
    index: int,
    costs: dict[str, Any],
) -> Decision:
    strategy_id = str(strategy["id"])
    strategy_type = str(strategy["type"])
    previous_target = state.target_position

    if strategy_type == "cash":
        return Decision(strategy_id, 0.0, "cash", "cash benchmark")
    if strategy_type == "buy_hold":
        return Decision(strategy_id, 1.0, "long", "buy-and-hold benchmark")

    if strategy_type == "candle_run_control":
        entry_bars = int(strategy.get("entry_run_bars", 2))
        exit_bars = int(strategy.get("exit_run_bars", 2))
        if int(row["streak_direction"]) < 0 and int(row["streak_length"]) >= exit_bars:
            return Decision(strategy_id, 0.0, "exit", f"{int(row['streak_length'])}-bar downward run")
        if int(row["streak_direction"]) > 0 and int(row["streak_length"]) >= entry_bars:
            failures: list[str] = []
            min_run = float(strategy.get("min_run_return_bps", 0.0))
            if _finite(row["run_return_bps"]) < min_run:
                failures.append(f"run below {min_run:.1f}bps")
            if _finite(row["relative_volume"]) < float(strategy.get("min_relative_volume", 0.0)):
                failures.append("relative volume filter")
            if _finite(row["body_fraction"]) < float(strategy.get("min_body_fraction", 0.0)):
                failures.append("body fraction filter")
            if bool(strategy.get("require_above_execution_ema", True)) and float(row["close"]) <= float(row["execution_ema_5m"]):
                failures.append("below execution EMA")
            if failures:
                return Decision(strategy_id, previous_target, "hold", "; ".join(failures))
            return Decision(strategy_id, 1.0, "enter", f"{int(row['streak_length'])}-bar upward run")
        return Decision(strategy_id, previous_target, "hold", "no opposite run")

    if strategy_type == "volatility_breakout":
        if previous_target <= 0.0:
            state.pending_entry_index = None
            cooldown = int(strategy.get("cooldown_bars_5m", 12))
            failures: list[str] = []
            if _bars_since_trade(state, index) < cooldown:
                failures.append("cooldown active")
            breakout_level = _finite(row["prior_breakout_high"], float("inf"))
            if float(row["close"]) <= breakout_level:
                failures.append("no prior-range breakout")
            if _finite(row["range_atr_ratio"]) < float(strategy.get("minimum_range_atr", 1.25)):
                failures.append("range below ATR threshold")
            if _finite(row["relative_volume"]) < float(strategy.get("minimum_relative_volume", 1.5)):
                failures.append("volume below threshold")
            if not _one_hour_bullish(row, strategy):
                failures.append("1h regime not bullish")
            if _finite(row["four_hour_momentum_bps"]) < float(strategy.get("minimum_4h_momentum_bps", 0.0)):
                failures.append("4h momentum negative")
            if failures:
                return Decision(strategy_id, 0.0, "hold_cash", "; ".join(failures))
            return Decision(
                strategy_id,
                1.0,
                "enter_breakout",
                "range breakout with volatility, volume, and slow-regime confirmation",
                entry_reference=breakout_level,
            )

        held = _bars_held(state, index)
        min_hold = int(strategy.get("minimum_hold_bars_5m", 12))
        max_hold = int(strategy.get("maximum_hold_bars_5m", 144))
        current_high_water = max(_finite(state.highest_high), float(row["high"]))
        trailing_stop = current_high_water - float(strategy.get("trailing_stop_atr", 2.5)) * _finite(row["atr_5m"])
        if held >= max_hold:
            return Decision(strategy_id, 0.0, "exit_time", f"maximum hold reached ({held} bars)")
        if float(row["close"]) <= trailing_stop:
            return Decision(strategy_id, 0.0, "exit_trailing", "ATR trailing stop")
        if held >= min_hold:
            if _one_hour_bearish(row, strategy):
                return Decision(strategy_id, 0.0, "exit_regime", "1h regime turned bearish")
            if state.entry_reference is not None and float(row["close"]) < state.entry_reference:
                return Decision(strategy_id, 0.0, "exit_failed_breakout", "close fell below breakout reference")
        return Decision(strategy_id, 1.0, "hold_btc", f"breakout position held {held} bars")

    if strategy_type == "vwap_reversion":
        if previous_target <= 0.0:
            state.pending_entry_index = None
            cooldown = int(strategy.get("cooldown_bars_5m", 12))
            failures: list[str] = []
            if _bars_since_trade(state, index) < cooldown:
                failures.append("cooldown active")
            if _finite(row["vwap_zscore"], 999.0) > float(strategy.get("entry_z_score", -2.0)):
                failures.append("VWAP deviation not oversold")
            if _finite(row["one_hour_ema_spread_bps"]) < float(strategy.get("minimum_1h_ema_spread_bps", -20.0)):
                failures.append("1h regime too bearish")
            if bool(strategy.get("require_selling_pressure_easing", True)) and not bool(row["selling_pressure_easing"]):
                failures.append("selling pressure not easing")
            if failures:
                return Decision(strategy_id, 0.0, "hold_cash", "; ".join(failures))
            return Decision(strategy_id, 1.0, "enter_reversion", "oversold VWAP deviation began recovering")

        held = _bars_held(state, index)
        minimum_hold = int(strategy.get("minimum_hold_bars_5m", 3))
        maximum_hold = int(strategy.get("maximum_hold_bars_5m", 72))
        zscore = _finite(row["vwap_zscore"])
        if zscore <= float(strategy.get("stop_z_score", -3.25)):
            return Decision(strategy_id, 0.0, "exit_stop", f"VWAP z-score stop at {zscore:.2f}")
        if held >= maximum_hold:
            return Decision(strategy_id, 0.0, "exit_time", f"maximum hold reached ({held} bars)")
        if held >= minimum_hold and zscore >= float(strategy.get("exit_z_score", -0.25)):
            return Decision(strategy_id, 0.0, "exit_vwap", f"VWAP mean reversion completed at z={zscore:.2f}")
        if held >= minimum_hold and _one_hour_bearish(row, strategy):
            return Decision(strategy_id, 0.0, "exit_regime", "1h regime became bearish")
        return Decision(strategy_id, 1.0, "hold_btc", f"waiting for VWAP reversion, z={zscore:.2f}")

    if strategy_type == "momentum_1h":
        if previous_target <= 0.0:
            state.pending_entry_index = None
            cooldown = int(strategy.get("cooldown_bars_5m", 12))
            if _bars_since_trade(state, index) < cooldown:
                return Decision(strategy_id, 0.0, "hold_cash", "cooldown active")
            if _one_hour_bullish(row, strategy):
                return Decision(strategy_id, 1.0, "enter_momentum", "1h momentum regime bullish")
            return Decision(strategy_id, 0.0, "hold_cash", "1h momentum regime not bullish")
        held = _bars_held(state, index)
        if held >= int(strategy.get("minimum_hold_bars_5m", 12)) and _one_hour_bearish(row, strategy):
            return Decision(strategy_id, 0.0, "exit_momentum", "1h momentum regime bearish")
        return Decision(strategy_id, 1.0, "hold_btc", "1h momentum regime remains acceptable")

    if strategy_type == "momentum_1h_5m_entry":
        if previous_target > 0.0:
            state.pending_entry_index = None
            held = _bars_held(state, index)
            if held >= int(strategy.get("minimum_hold_bars_5m", 12)) and _one_hour_bearish(row, strategy):
                return Decision(strategy_id, 0.0, "exit_momentum", "same 1h exit as immediate strategy")
            return Decision(strategy_id, 1.0, "hold_btc", "same 1h regime remains acceptable")

        cooldown = int(strategy.get("cooldown_bars_5m", 12))
        if _bars_since_trade(state, index) < cooldown:
            state.pending_entry_index = None
            return Decision(strategy_id, 0.0, "hold_cash", "cooldown active")
        if not _one_hour_bullish(row, strategy):
            state.pending_entry_index = None
            return Decision(strategy_id, 0.0, "hold_cash", "1h momentum regime not bullish")

        if state.pending_entry_index is None:
            state.pending_entry_index = index
        pending_age = index - state.pending_entry_index
        pullback_threshold = float(strategy.get("minimum_pullback_bps", -5.0))
        pullback_seen = _finite(row["recent_pullback_bps"], 999.0) <= pullback_threshold
        recovery = bool(row["recovery_cross_up"])
        if pullback_seen and recovery:
            state.pending_entry_index = None
            return Decision(strategy_id, 1.0, "enter_5m_recovery", "1h bullish; 5m pullback recovered")
        max_wait = int(strategy.get("maximum_entry_wait_bars_5m", 12))
        if pending_age >= max_wait:
            state.pending_entry_index = None
            return Decision(strategy_id, 1.0, "enter_timeout", f"1h bullish; 5m wait expired after {pending_age} bars")
        return Decision(strategy_id, 0.0, "wait_5m_entry", f"waiting for 5m pullback recovery ({pending_age}/{max_wait})")

    raise ValueError(f"Unknown intraday strategy type: {strategy_type}")


def _equity(state: StrategyState, mark: float) -> float:
    return state.cash + state.btc * mark


def _gross_equity(state: StrategyState, mark: float) -> float:
    return state.gross_cash + state.gross_btc * mark


def execute_decision(
    state: StrategyState,
    decision: Decision,
    signal_row: pd.Series,
    next_row: pd.Series,
    signal_index: int,
    costs: dict[str, Any],
) -> tuple[dict[str, Any], SimulatedTrade | None]:
    mark = float(next_row["open"])
    fee_rate = float(costs.get("fee_bps_per_side", 0.0)) / 10_000.0
    spread_rate = float(costs.get("assumed_spread_bps_per_side", 0.0)) / 10_000.0
    slippage_rate = float(costs.get("slippage_bps_per_side", 0.0)) / 10_000.0
    target = min(1.0, max(0.0, float(decision.target_position)))

    equity_before = _equity(state, mark)
    gross_equity_before = _gross_equity(state, mark)
    desired_btc = equity_before * target / mark
    delta_btc = desired_btc - state.btc
    min_notional = float(costs.get("min_notional", 10.0))
    tolerance = equity_before * float(costs.get("rebalance_tolerance_bps", 1.0)) / 10_000.0

    transaction: SimulatedTrade | None = None
    execution_index = signal_index + 1
    if abs(delta_btc) * mark >= max(min_notional, tolerance):
        if delta_btc > 0.0:
            side = "buy"
            reference = mark * (1.0 + spread_rate)
            fill = reference * (1.0 + slippage_rate)
            affordable = state.cash / (fill * (1.0 + fee_rate))
            executed_btc = min(delta_btc, affordable)
        else:
            side = "sell"
            reference = mark * (1.0 - spread_rate)
            fill = reference * (1.0 - slippage_rate)
            executed_btc = -min(abs(delta_btc), state.btc)

        notional = abs(executed_btc) * fill
        if notional >= min_notional and abs(executed_btc) > 1e-15:
            fee = notional * fee_rate
            spread_cost = abs(executed_btc) * abs(reference - mark)
            slippage_cost = abs(executed_btc) * abs(fill - reference)
            if executed_btc > 0.0:
                state.cash -= notional + fee
                state.btc += executed_btc
                state.entry_index = execution_index
                state.entry_mark = mark
                state.highest_high = mark
                state.entry_reference = decision.entry_reference or mark
            else:
                state.cash += notional - fee
                state.btc += executed_btc
                if abs(state.btc) < 1e-12:
                    state.btc = 0.0
                state.entry_index = None
                state.entry_mark = None
                state.highest_high = None
                state.entry_reference = None
                state.pending_entry_index = None
            state.total_fees += fee
            state.total_spread += spread_cost
            state.total_slippage += slippage_cost
            state.total_turnover += notional
            state.trade_count += 1
            state.last_trade_index = execution_index
            transaction = SimulatedTrade(
                strategy_id=state.strategy_id,
                signal_timestamp=pd.Timestamp(signal_row["timestamp"]).isoformat(),
                execution_timestamp=pd.Timestamp(next_row["timestamp"]).isoformat(),
                side=side,
                mark_price=mark,
                reference_price=reference,
                fill_price=fill,
                btc_delta=executed_btc,
                gross_notional=notional,
                fee=fee,
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
                cash_after=state.cash,
                btc_after=state.btc,
                equity_after=_equity(state, mark),
                signal=decision.signal,
                reason=decision.reason,
            )

    desired_gross_btc = gross_equity_before * target / mark
    gross_delta = desired_gross_btc - state.gross_btc
    state.gross_cash -= gross_delta * mark
    state.gross_btc += gross_delta
    state.target_position = target

    equity = _equity(state, mark)
    gross_equity = _gross_equity(state, mark)
    state.peak_equity = max(float(state.peak_equity or state.initial_cash), equity)
    drawdown = equity / state.peak_equity - 1.0

    snapshot = {
        "strategy_id": state.strategy_id,
        "bar_timestamp": pd.Timestamp(signal_row["timestamp"]).isoformat(),
        "execution_timestamp": pd.Timestamp(next_row["timestamp"]).isoformat(),
        "mark_price": mark,
        "signal_close": float(signal_row["close"]),
        "signal": decision.signal,
        "reason": decision.reason,
        "target_position": target,
        "cash": state.cash,
        "btc": state.btc,
        "equity": equity,
        "gross_equity": gross_equity,
        "return_pct": equity / state.initial_cash - 1.0,
        "gross_return_pct": gross_equity / state.initial_cash - 1.0,
        "drawdown": drawdown,
        "cost_drag": gross_equity - equity,
        "trade_count": state.trade_count,
        "total_fees": state.total_fees,
        "total_spread": state.total_spread,
        "total_slippage": state.total_slippage,
        "total_turnover": state.total_turnover,
        "bars_held": _bars_held(state, execution_index),
        "streak_direction": int(signal_row["streak_direction"]),
        "streak_length": int(signal_row["streak_length"]),
        "run_return_bps": _finite(signal_row["run_return_bps"]),
        "range_atr_ratio": _finite(signal_row["range_atr_ratio"]),
        "relative_volume": _finite(signal_row["relative_volume"]),
        "vwap_zscore": _finite(signal_row["vwap_zscore"]),
        "one_hour_ema_spread_bps": _finite(signal_row["one_hour_ema_spread_bps"]),
        "one_hour_momentum_bps": _finite(signal_row["one_hour_momentum_bps"]),
        "four_hour_momentum_bps": _finite(signal_row["four_hour_momentum_bps"]),
    }
    return snapshot, transaction


def build_trade_episodes(
    snapshots: pd.DataFrame,
    initial_cash: float,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    columns = [
        "strategy_id",
        "entry_timestamp",
        "exit_timestamp",
        "is_open",
        "holding_bars",
        "holding_minutes",
        "entry_mark",
        "exit_mark",
        "gross_price_return_pct",
        "net_portfolio_return_pct",
        "maximum_adverse_excursion_pct",
        "maximum_favorable_excursion_pct",
    ]
    if snapshots.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for strategy_id, group in snapshots.groupby("strategy_id", sort=False):
        group = group.sort_values("execution_timestamp").reset_index(drop=True)
        in_position = False
        entry_timestamp: str | None = None
        entry_mark = 0.0
        equity_before_entry = initial_cash
        min_mark = 0.0
        max_mark = 0.0
        holding_bars = 0
        previous_equity = initial_cash

        for record in group.to_dict("records"):
            target = float(record["target_position"])
            mark = float(record["mark_price"])
            if not in_position and target > 0.0:
                in_position = True
                entry_timestamp = str(record["execution_timestamp"])
                entry_mark = mark
                equity_before_entry = previous_equity
                min_mark = mark
                max_mark = mark
                holding_bars = 1
            elif in_position and target > 0.0:
                holding_bars += 1
                min_mark = min(min_mark, mark)
                max_mark = max(max_mark, mark)
            elif in_position and target <= 0.0:
                min_mark = min(min_mark, mark)
                max_mark = max(max_mark, mark)
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "entry_timestamp": entry_timestamp,
                        "exit_timestamp": str(record["execution_timestamp"]),
                        "is_open": False,
                        "holding_bars": holding_bars,
                        "holding_minutes": holding_bars * timeframe_minutes,
                        "entry_mark": entry_mark,
                        "exit_mark": mark,
                        "gross_price_return_pct": mark / entry_mark - 1.0,
                        "net_portfolio_return_pct": float(record["equity"]) / equity_before_entry - 1.0,
                        "maximum_adverse_excursion_pct": min_mark / entry_mark - 1.0,
                        "maximum_favorable_excursion_pct": max_mark / entry_mark - 1.0,
                    }
                )
                in_position = False
                entry_timestamp = None
                holding_bars = 0
            previous_equity = float(record["equity"])

        if in_position:
            final = group.iloc[-1]
            final_mark = float(final["mark_price"])
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "entry_timestamp": entry_timestamp,
                    "exit_timestamp": None,
                    "is_open": True,
                    "holding_bars": holding_bars,
                    "holding_minutes": holding_bars * timeframe_minutes,
                    "entry_mark": entry_mark,
                    "exit_mark": final_mark,
                    "gross_price_return_pct": final_mark / entry_mark - 1.0,
                    "net_portfolio_return_pct": float(final["equity"]) / equity_before_entry - 1.0,
                    "maximum_adverse_excursion_pct": min_mark / entry_mark - 1.0,
                    "maximum_favorable_excursion_pct": max_mark / entry_mark - 1.0,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _summarize(
    snapshots: pd.DataFrame,
    transactions: pd.DataFrame,
    episodes: pd.DataFrame,
    initial_cash: float,
    sample_days: float,
    costs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    if snapshots.empty:
        return summary

    all_in = (
        float(costs.get("fee_bps_per_side", 0.0))
        + float(costs.get("slippage_bps_per_side", 0.0))
        + float(costs.get("assumed_spread_bps_per_side", 0.0))
    )
    for strategy_id, group in snapshots.groupby("strategy_id"):
        group = group.sort_values("execution_timestamp")
        last = group.iloc[-1]
        turnover = float(last["total_turnover"])
        gross_profit = float(last["gross_equity"]) - initial_cash
        break_even = max(0.0, gross_profit) / turnover * 10_000.0 if turnover > 0.0 else 0.0
        strategy_episodes = episodes[episodes["strategy_id"] == strategy_id]
        closed = strategy_episodes[~strategy_episodes["is_open"]]
        strategy_transactions = transactions[transactions["strategy_id"] == strategy_id] if not transactions.empty else pd.DataFrame()
        summary[strategy_id] = {
            "ending_equity": float(last["equity"]),
            "ending_gross_equity": float(last["gross_equity"]),
            "net_return_pct": float(last["return_pct"]),
            "gross_return_pct": float(last["gross_return_pct"]),
            "max_drawdown": float(group["drawdown"].min()),
            "transaction_count": int(last["trade_count"]),
            "round_trips_closed": int(len(closed)),
            "transactions_per_day": float(last["trade_count"]) / sample_days,
            "total_fees": float(last["total_fees"]),
            "total_spread": float(last["total_spread"]),
            "total_slippage": float(last["total_slippage"]),
            "total_turnover": turnover,
            "turnover_multiple": turnover / initial_cash,
            "cost_drag": float(last["cost_drag"]),
            "gross_break_even_all_in_bps_per_side": break_even,
            "assumed_all_in_bps_per_side": all_in,
            "cost_to_break_even_multiple": all_in / break_even if break_even > 0.0 else None,
            "average_holding_bars": float(closed["holding_bars"].mean()) if not closed.empty else None,
            "median_holding_bars": float(closed["holding_bars"].median()) if not closed.empty else None,
            "average_gross_return_per_round_trip": float(closed["gross_price_return_pct"].mean()) if not closed.empty else None,
            "average_net_return_per_round_trip": float(closed["net_portfolio_return_pct"].mean()) if not closed.empty else None,
            "round_trip_win_rate": float((closed["net_portfolio_return_pct"] > 0.0).mean()) if not closed.empty else None,
            "average_mae_pct": float(closed["maximum_adverse_excursion_pct"].mean()) if not closed.empty else None,
            "average_mfe_pct": float(closed["maximum_favorable_excursion_pct"].mean()) if not closed.empty else None,
            "buy_transactions": int((strategy_transactions.get("side", pd.Series(dtype=str)) == "buy").sum()) if not strategy_transactions.empty else 0,
            "sell_transactions": int((strategy_transactions.get("side", pd.Series(dtype=str)) == "sell").sum()) if not strategy_transactions.empty else 0,
        }

    buy_hold_return = summary.get("buy_hold_5m", {}).get("net_return_pct")
    if buy_hold_return is not None:
        for metrics in summary.values():
            metrics["net_excess_return_vs_buy_hold"] = metrics["net_return_pct"] - buy_hold_return
    return summary


def simulate_matrix(
    feature_frame: pd.DataFrame,
    cfg: dict[str, Any],
    costs_override: dict[str, Any] | None = None,
) -> SimulationResult:
    costs = copy.deepcopy(costs_override if costs_override is not None else cfg["costs"])
    strategies = cfg["strategies"]
    initial_cash = float(cfg["research"].get("initial_cash", 500.0))
    states = {
        str(strategy["id"]): StrategyState(
            strategy_id=str(strategy["id"]),
            initial_cash=initial_cash,
            cash=initial_cash,
            btc=0.0,
            gross_cash=initial_cash,
            gross_btc=0.0,
            peak_equity=initial_cash,
        )
        for strategy in strategies
    }

    valid_indices = feature_frame.index[feature_frame["feature_valid"]].tolist()
    if not valid_indices:
        raise RuntimeError("No rows contain a complete feature history.")
    start_index = valid_indices[0]
    snapshots: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []

    for index in range(start_index, len(feature_frame) - 1):
        row = feature_frame.iloc[index]
        next_row = feature_frame.iloc[index + 1]
        if not bool(row["feature_valid"]):
            continue
        for strategy in strategies:
            strategy_id = str(strategy["id"])
            state = states[strategy_id]
            if state.target_position > 0.0:
                state.highest_high = max(_finite(state.highest_high), float(row["high"]))
            decision = decide_strategy(strategy, row, state, index, costs)
            snapshot, transaction = execute_decision(
                state,
                decision,
                row,
                next_row,
                index,
                costs,
            )
            snapshots.append(snapshot)
            if transaction is not None:
                transactions.append(asdict(transaction))

    snapshot_frame = pd.DataFrame(snapshots)
    transaction_frame = pd.DataFrame(transactions)
    timeframe_minutes = int(timeframe_to_timedelta(str(cfg["market"]["timeframe"])).total_seconds() // 60)
    episodes = build_trade_episodes(snapshot_frame, initial_cash, timeframe_minutes)

    first_execution = pd.Timestamp(snapshot_frame["execution_timestamp"].min())
    last_execution = pd.Timestamp(snapshot_frame["execution_timestamp"].max())
    sample_days = max((last_execution - first_execution).total_seconds() / 86_400.0, 1e-9)
    summary = _summarize(
        snapshot_frame,
        transaction_frame,
        episodes,
        initial_cash,
        sample_days,
        costs,
    )
    return SimulationResult(snapshot_frame, transaction_frame, episodes, summary)


def _cost_sensitivity(
    feature_frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    grid = cfg["research"].get("cost_sensitivity_all_in_bps_per_side", [3, 5, 8, 10, 12])
    rows: list[dict[str, Any]] = []
    for all_in in grid:
        override = {
            "fee_bps_per_side": float(all_in),
            "slippage_bps_per_side": 0.0,
            "assumed_spread_bps_per_side": 0.0,
            "min_notional": float(cfg["costs"].get("min_notional", 10.0)),
            "rebalance_tolerance_bps": float(cfg["costs"].get("rebalance_tolerance_bps", 1.0)),
        }
        result = simulate_matrix(feature_frame, cfg, costs_override=override)
        for strategy_id, metrics in result.summary.items():
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "all_in_bps_per_side": float(all_in),
                    "ending_equity": metrics["ending_equity"],
                    "net_return_pct": metrics["net_return_pct"],
                    "max_drawdown": metrics["max_drawdown"],
                    "transaction_count": metrics["transaction_count"],
                    "round_trips_closed": metrics["round_trips_closed"],
                    "turnover_multiple": metrics["turnover_multiple"],
                    "average_holding_bars": metrics["average_holding_bars"],
                }
            )
    return pd.DataFrame(rows)


def _period_slices(
    feature_frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    fold_count = int(cfg["research"].get("chronological_folds", 5))
    if fold_count <= 1:
        return pd.DataFrame()
    valid = feature_frame[feature_frame["feature_valid"]].copy().reset_index(drop=True)
    if len(valid) < fold_count * 100:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    boundaries = np.linspace(0, len(valid), fold_count + 1, dtype=int)
    history_rows = int(cfg["research"].get("fold_history_rows", 1000))
    for fold in range(fold_count):
        score_start = boundaries[fold]
        score_end = boundaries[fold + 1]
        history_start = max(0, score_start - history_rows)
        fold_frame = valid.iloc[history_start:score_end].copy().reset_index(drop=True)
        # Only score the fold itself. Prior rows remain as feature history but are
        # marked invalid so the simulator starts with fresh capital at the fold.
        fold_frame.loc[: score_start - history_start - 1, "feature_valid"] = False
        if int(fold_frame["feature_valid"].sum()) < 2:
            continue
        result = simulate_matrix(fold_frame, cfg)
        score_rows = fold_frame[fold_frame["feature_valid"]]
        for strategy_id, metrics in result.summary.items():
            rows.append(
                {
                    "fold": fold + 1,
                    "strategy_id": strategy_id,
                    "first_signal_bar": pd.Timestamp(score_rows["timestamp"].iloc[0]).isoformat(),
                    "last_signal_bar": pd.Timestamp(score_rows["timestamp"].iloc[-1]).isoformat(),
                    "bars": int(len(score_rows)),
                    "net_return_pct": metrics["net_return_pct"],
                    "gross_return_pct": metrics["gross_return_pct"],
                    "max_drawdown": metrics["max_drawdown"],
                    "transaction_count": metrics["transaction_count"],
                    "turnover_multiple": metrics["turnover_multiple"],
                }
            )
    return pd.DataFrame(rows)


def _save_charts(snapshots: pd.DataFrame, output: Path) -> None:
    equity = snapshots.pivot_table(
        index="execution_timestamp",
        columns="strategy_id",
        values="equity",
        aggfunc="last",
    )
    equity.index = pd.to_datetime(equity.index, utc=True)
    figure, axis = plt.subplots(figsize=(12, 7))
    equity.plot(ax=axis)
    axis.set_title("Intraday strategy matrix: net equity")
    axis.set_xlabel("Execution time (UTC)")
    axis.set_ylabel("Portfolio value")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "equity_curve.png", dpi=150)
    plt.close(figure)

    drawdown = snapshots.pivot_table(
        index="execution_timestamp",
        columns="strategy_id",
        values="drawdown",
        aggfunc="last",
    )
    drawdown.index = pd.to_datetime(drawdown.index, utc=True)
    figure, axis = plt.subplots(figsize=(12, 7))
    drawdown.plot(ax=axis)
    axis.set_title("Intraday strategy matrix: drawdown")
    axis.set_xlabel("Execution time (UTC)")
    axis.set_ylabel("Drawdown")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "drawdown_curve.png", dpi=150)
    plt.close(figure)


def run_research(
    config_path: str,
    bars_requested: int,
    output_dir: str,
    data_path: str | None = None,
    skip_cost_grid: bool = False,
) -> dict[str, Any]:
    cfg = load_matrix_config(config_path)
    market = cfg["market"]
    timeframe = str(market["timeframe"])
    if data_path:
        raw, _ = load_ohlcv_csv(data_path, timeframe=timeframe)
        raw = raw.tail(bars_requested).reset_index(drop=True)
    else:
        step = timeframe_to_timedelta(timeframe)
        start = (pd.Timestamp.now(tz="UTC") - step * (bars_requested + 50)).isoformat()
        raw = download_ohlcv(
            exchange_id=str(market["exchange"]),
            symbol=str(market["symbol"]),
            timeframe=timeframe,
            start=start,
            max_bars=bars_requested,
        )

    feature_frame = build_feature_frame(raw, cfg)
    result = simulate_matrix(feature_frame, cfg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feature_frame.to_csv(output / "feature_frame.csv", index=False)
    result.snapshots.to_csv(output / "strategy_equity.csv", index=False)
    result.transactions.to_csv(output / "transactions.csv", index=False)
    result.episodes.to_csv(output / "trade_episodes.csv", index=False)

    signal_counts = (
        result.snapshots.groupby(["strategy_id", "signal"])
        .size()
        .rename("count")
        .reset_index()
    )
    signal_counts.to_csv(output / "signal_counts.csv", index=False)

    comparison = pd.DataFrame.from_dict(result.summary, orient="index")
    comparison.index.name = "strategy_id"
    comparison.reset_index().to_csv(output / "strategy_comparison.csv", index=False)

    if not skip_cost_grid:
        _cost_sensitivity(feature_frame, cfg).to_csv(output / "cost_sensitivity.csv", index=False)
    period_slices = _period_slices(feature_frame, cfg)
    if not period_slices.empty:
        period_slices.to_csv(output / "chronological_folds.csv", index=False)

    _save_charts(result.snapshots, output)

    first_bar = pd.Timestamp(raw["timestamp"].iloc[0])
    last_bar = pd.Timestamp(raw["timestamp"].iloc[-1])
    valid = feature_frame[feature_frame["feature_valid"]]
    first_scored = pd.Timestamp(valid["timestamp"].iloc[0])
    last_scored = pd.Timestamp(valid["timestamp"].iloc[-1])
    sample_days = max((last_scored - first_scored).total_seconds() / 86_400.0, 1e-9)
    summary = {
        "market": {
            "exchange": str(market["exchange"]),
            "symbol": str(market["symbol"]),
            "timeframe": timeframe,
        },
        "downloaded_bars": int(len(raw)),
        "first_downloaded_bar": first_bar.isoformat(),
        "last_downloaded_bar": last_bar.isoformat(),
        "scored_bars": int(len(valid) - 1),
        "first_scored_bar": first_scored.isoformat(),
        "last_scored_bar": last_scored.isoformat(),
        "sample_days": sample_days,
        "cost_assumptions": {
            "fee_bps_per_side": float(cfg["costs"].get("fee_bps_per_side", 0.0)),
            "slippage_bps_per_side": float(cfg["costs"].get("slippage_bps_per_side", 0.0)),
            "assumed_spread_bps_per_side": float(cfg["costs"].get("assumed_spread_bps_per_side", 0.0)),
            "all_in_bps_per_side": (
                float(cfg["costs"].get("fee_bps_per_side", 0.0))
                + float(cfg["costs"].get("slippage_bps_per_side", 0.0))
                + float(cfg["costs"].get("assumed_spread_bps_per_side", 0.0))
            ),
        },
        "strategies": result.summary,
        "outputs": {
            "strategy_comparison": "strategy_comparison.csv",
            "cost_sensitivity": None if skip_cost_grid else "cost_sensitivity.csv",
            "chronological_folds": "chronological_folds.csv" if not period_slices.empty else None,
            "trade_episodes": "trade_episodes.csv",
            "transactions": "transactions.csv",
            "equity_curve": "equity_curve.png",
            "drawdown_curve": "drawdown_curve.png",
        },
    }
    (output / "research_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Research-only BTC intraday strategy matrix"
    )
    parser.add_argument(
        "--config",
        default="config/settings_intraday_matrix.yaml",
    )
    parser.add_argument("--bars", type=int, default=10_000)
    parser.add_argument("--output", default="outputs/intraday_matrix_10000")
    parser.add_argument("--data", default=None, help="Optional normalized OHLCV CSV")
    parser.add_argument("--skip-cost-grid", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_research(
        config_path=args.config,
        bars_requested=args.bars,
        output_dir=args.output,
        data_path=args.data,
        skip_cost_grid=args.skip_cost_grid,
    )


if __name__ == "__main__":
    main()
