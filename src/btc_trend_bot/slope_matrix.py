"""Research-only rolling-slope inversion strategy matrix.

The module estimates a causal, trailing local-polynomial derivative of log price
on complete candles, then tests whether persistent slope threshold crossings can
identify trend inversions at several high-tempo signal horizons.

It contains no authenticated exchange path and no live order submission code.
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
from numpy.lib.stride_tricks import sliding_window_view

from btc_trend_bot.data import (
    download_ohlcv,
    load_ohlcv_csv,
    normalize_ohlcv,
    timeframe_to_timedelta,
)
from btc_trend_bot.intraday_matrix import (
    Decision,
    SimulatedTrade,
    StrategyState,
    _bars_held,
    _bars_since_trade,
    _equity,
    _finite,
    _gross_equity,
    _summarize,
    build_trade_episodes,
)


@dataclass
class SlopeStrategyState(StrategyState):
    entry_confirmations: int = 0
    exit_confirmations: int = 0
    last_signal_bar_end: str | None = None


@dataclass(frozen=True)
class SlopeSimulationResult:
    snapshots: pd.DataFrame
    transactions: pd.DataFrame
    episodes: pd.DataFrame
    summary: dict[str, dict[str, Any]]


def load_slope_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Slope matrix config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Slope matrix config must contain a YAML mapping.")
    for section in ("market", "costs", "research", "strategies"):
        if section not in raw:
            raise ValueError(f"Missing slope matrix config section: {section}")
    strategies = raw["strategies"]
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("strategies must be a non-empty list.")
    ids = [str(item["id"]) for item in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError("Strategy IDs must be unique.")
    for strategy in strategies:
        strategy_type = str(strategy["type"])
        if strategy_type not in {"cash", "buy_hold", "rolling_slope"}:
            raise ValueError(f"Unknown slope strategy type: {strategy_type}")
        if strategy_type == "rolling_slope":
            _validate_slope_strategy(strategy)
    return raw


def _validate_slope_strategy(strategy: dict[str, Any]) -> None:
    timeframe = str(strategy.get("signal_timeframe", ""))
    if timeframe not in {"5m", "15m", "30m", "1h"}:
        raise ValueError("signal_timeframe must be one of 5m, 15m, 30m, or 1h")
    lookback = int(strategy.get("lookback_bars", 0))
    degree = int(strategy.get("polynomial_degree", 1))
    if lookback < 6:
        raise ValueError("lookback_bars must be at least 6")
    if degree not in (1, 2):
        raise ValueError("polynomial_degree must be 1 or 2")
    if lookback <= degree + 2:
        raise ValueError("lookback_bars is too small for the polynomial degree")
    if int(strategy.get("entry_confirmation_ticks", 2)) < 1:
        raise ValueError("entry_confirmation_ticks must be positive")
    if int(strategy.get("exit_confirmation_ticks", 2)) < 1:
        raise ValueError("exit_confirmation_ticks must be positive")
    entry = float(strategy.get("entry_slope_score", 0.0))
    exit_ = float(strategy.get("exit_slope_score", 0.0))
    if entry <= exit_:
        raise ValueError("entry_slope_score must exceed exit_slope_score for hysteresis")


def _timeframe_source_bars(base_timeframe: str, target_timeframe: str) -> int:
    base = timeframe_to_timedelta(base_timeframe)
    target = timeframe_to_timedelta(target_timeframe)
    ratio = target / base
    if ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"{target_timeframe} is not an integer multiple of {base_timeframe}")
    return int(round(ratio))


def _complete_signal_candles(
    base: pd.DataFrame,
    base_timeframe: str,
    signal_timeframe: str,
) -> pd.DataFrame:
    base_step = timeframe_to_timedelta(base_timeframe)
    expected = _timeframe_source_bars(base_timeframe, signal_timeframe)
    work = base.copy()
    work["bar_end"] = pd.to_datetime(work["timestamp"], utc=True) + base_step
    if expected == 1:
        return work[["bar_end", "open", "high", "low", "close", "volume"]].copy()

    indexed = work.set_index("bar_end")
    pandas_rule = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}[signal_timeframe]
    grouped = indexed.resample(
        pandas_rule,
        origin="epoch",
        label="right",
        closed="right",
    )
    candles = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped["close"].count()
    candles = candles.loc[counts == expected].dropna().reset_index()
    return candles


def trailing_local_polynomial(
    close: pd.Series,
    lookback_bars: int,
    polynomial_degree: int,
    volatility_window_bars: int | None = None,
) -> pd.DataFrame:
    """Estimate a causal endpoint derivative from trailing log prices.

    The polynomial is fitted only to the current and preceding closes. For a
    quadratic fit, the reported slope is the derivative at the right endpoint,
    not the center of the window. This avoids centered-filter lookahead.
    """
    values = np.asarray(close, dtype=float)
    n = int(lookback_bars)
    degree = int(polynomial_degree)
    if len(values) < n:
        return pd.DataFrame(
            {
                "slope_bps_per_bar": np.full(len(values), np.nan),
                "slope_score": np.full(len(values), np.nan),
                "slope_tstat": np.full(len(values), np.nan),
                "fit_r2": np.full(len(values), np.nan),
                "acceleration_score": np.full(len(values), np.nan),
            }
        )
    if np.any(values <= 0.0):
        raise ValueError("Close prices must be positive for log-slope estimation.")

    y = np.log(values)
    windows = sliding_window_view(y, window_shape=n)
    x = np.arange(n, dtype=float)
    design = np.vander(x, N=degree + 1, increasing=True)
    pseudo_inverse = np.linalg.pinv(design)
    coefficients = windows @ pseudo_inverse.T
    fitted = coefficients @ design.T
    residual = windows - fitted
    sse = np.sum(residual * residual, axis=1)
    centered = windows - windows.mean(axis=1, keepdims=True)
    sst = np.sum(centered * centered, axis=1)
    r2 = np.where(sst > 0.0, np.maximum(0.0, 1.0 - sse / sst), 0.0)

    x_end = float(n - 1)
    if degree == 1:
        endpoint_slope = coefficients[:, 1]
        endpoint_curvature = np.full(len(endpoint_slope), np.nan)
        derivative_design = np.array([0.0, 1.0])
    else:
        endpoint_slope = coefficients[:, 1] + 2.0 * coefficients[:, 2] * x_end
        endpoint_curvature = 2.0 * coefficients[:, 2]
        derivative_design = np.array([0.0, 1.0, 2.0 * x_end])

    dof = n - (degree + 1)
    xtx_inverse = np.linalg.inv(design.T @ design)
    derivative_variance_multiplier = float(derivative_design @ xtx_inverse @ derivative_design)
    residual_variance = np.where(dof > 0, sse / dof, np.nan)
    derivative_se = np.sqrt(np.maximum(0.0, residual_variance * derivative_variance_multiplier))
    tstat = np.divide(
        endpoint_slope,
        derivative_se,
        out=np.full_like(endpoint_slope, np.nan),
        where=derivative_se > 0.0,
    )

    return_vol_window = int(volatility_window_bars or n)
    log_returns = pd.Series(y).diff()
    realized_vol = log_returns.rolling(
        return_vol_window,
        min_periods=return_vol_window,
    ).std(ddof=0).to_numpy()
    endpoint_vol = realized_vol[n - 1 :]
    slope_score = np.divide(
        endpoint_slope,
        endpoint_vol,
        out=np.full_like(endpoint_slope, np.nan),
        where=endpoint_vol > 0.0,
    )

    if degree == 2:
        acceleration_score = np.divide(
            endpoint_curvature,
            endpoint_vol,
            out=np.full_like(endpoint_curvature, np.nan),
            where=endpoint_vol > 0.0,
        )
    else:
        slope_change = np.r_[np.nan, np.diff(endpoint_slope)]
        acceleration_score = np.divide(
            slope_change,
            endpoint_vol,
            out=np.full_like(slope_change, np.nan),
            where=endpoint_vol > 0.0,
        )

    pad = n - 1
    output = pd.DataFrame(
        {
            "slope_bps_per_bar": np.r_[np.full(pad, np.nan), endpoint_slope * 10_000.0],
            "slope_score": np.r_[np.full(pad, np.nan), slope_score],
            "slope_tstat": np.r_[np.full(pad, np.nan), tstat],
            "fit_r2": np.r_[np.full(pad, np.nan), r2],
            "acceleration_score": np.r_[np.full(pad, np.nan), acceleration_score],
        }
    )
    return output


def _feature_prefix(strategy_id: str) -> str:
    return f"slope__{strategy_id}__"


def build_slope_feature_frame(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    market = cfg["market"]
    base_timeframe = str(market["timeframe"])
    if base_timeframe != "5m":
        raise ValueError("The slope matrix currently requires 5m base data.")

    normalized, _ = normalize_ohlcv(frame, timeframe=base_timeframe)
    out = normalized.copy().reset_index(drop=True)
    base_step = timeframe_to_timedelta(base_timeframe)
    out["bar_end"] = pd.to_datetime(out["timestamp"], utc=True) + base_step

    required_columns: list[str] = []
    for strategy in cfg["strategies"]:
        if str(strategy["type"]) != "rolling_slope":
            continue
        strategy_id = str(strategy["id"])
        prefix = _feature_prefix(strategy_id)
        signal_timeframe = str(strategy["signal_timeframe"])
        signal = _complete_signal_candles(out, base_timeframe, signal_timeframe)
        estimates = trailing_local_polynomial(
            signal["close"],
            lookback_bars=int(strategy["lookback_bars"]),
            polynomial_degree=int(strategy.get("polynomial_degree", 1)),
            volatility_window_bars=int(
                strategy.get("volatility_window_bars", strategy["lookback_bars"])
            ),
        )
        signal = pd.concat([signal.reset_index(drop=True), estimates], axis=1)
        renamed = signal.rename(
            columns={
                "bar_end": f"{prefix}bar_end",
                "close": f"{prefix}signal_close",
                "slope_bps_per_bar": f"{prefix}slope_bps_per_bar",
                "slope_score": f"{prefix}slope_score",
                "slope_tstat": f"{prefix}slope_tstat",
                "fit_r2": f"{prefix}fit_r2",
                "acceleration_score": f"{prefix}acceleration_score",
            }
        )[
            [
                f"{prefix}bar_end",
                f"{prefix}signal_close",
                f"{prefix}slope_bps_per_bar",
                f"{prefix}slope_score",
                f"{prefix}slope_tstat",
                f"{prefix}fit_r2",
                f"{prefix}acceleration_score",
            ]
        ]
        out = pd.merge_asof(
            out.sort_values("bar_end"),
            renamed.sort_values(f"{prefix}bar_end"),
            left_on="bar_end",
            right_on=f"{prefix}bar_end",
            direction="backward",
            allow_exact_matches=True,
        )
        out[f"{prefix}signal_update"] = out[f"{prefix}bar_end"].ne(
            out[f"{prefix}bar_end"].shift(1)
        )
        required_columns.extend(
            [
                f"{prefix}slope_bps_per_bar",
                f"{prefix}slope_score",
                f"{prefix}slope_tstat",
                f"{prefix}fit_r2",
                f"{prefix}acceleration_score",
            ]
        )

    if not required_columns:
        raise ValueError("At least one rolling_slope strategy is required.")
    out["feature_valid"] = out[required_columns].notna().all(axis=1)
    return out.reset_index(drop=True)


def _signal_bars_to_base(strategy: dict[str, Any], signal_bars: int) -> int:
    return _timeframe_source_bars("5m", str(strategy["signal_timeframe"])) * int(signal_bars)


def _slope_values(strategy: dict[str, Any], row: pd.Series) -> dict[str, float | bool | str]:
    prefix = _feature_prefix(str(strategy["id"]))
    return {
        "bar_end": str(row[f"{prefix}bar_end"]),
        "signal_update": bool(row[f"{prefix}signal_update"]),
        "signal_close": _finite(row[f"{prefix}signal_close"], float(row["close"])),
        "slope_bps_per_bar": _finite(row[f"{prefix}slope_bps_per_bar"]),
        "slope_score": _finite(row[f"{prefix}slope_score"]),
        "slope_tstat": _finite(row[f"{prefix}slope_tstat"]),
        "fit_r2": _finite(row[f"{prefix}fit_r2"]),
        "acceleration_score": _finite(row[f"{prefix}acceleration_score"]),
    }


def _all_in_bps_per_side(costs: dict[str, Any]) -> float:
    return (
        float(costs.get("fee_bps_per_side", 0.0))
        + float(costs.get("slippage_bps_per_side", 0.0))
        + float(costs.get("assumed_spread_bps_per_side", 0.0))
    )


def _entry_condition(
    strategy: dict[str, Any],
    values: dict[str, float | bool | str],
    costs: dict[str, Any],
) -> tuple[bool, list[str], float]:
    failures: list[str] = []
    slope_score = float(values["slope_score"])
    slope_bps = float(values["slope_bps_per_bar"])
    r2 = float(values["fit_r2"])
    tstat = float(values["slope_tstat"])
    acceleration = float(values["acceleration_score"])

    if slope_score < float(strategy.get("entry_slope_score", 0.15)):
        failures.append("slope score below entry threshold")
    if r2 < float(strategy.get("minimum_fit_r2", 0.15)):
        failures.append("fit R2 below threshold")
    if tstat < float(strategy.get("minimum_entry_tstat", 0.0)):
        failures.append("slope t-stat below threshold")
    if bool(strategy.get("require_positive_acceleration", False)) and acceleration < float(
        strategy.get("minimum_entry_acceleration_score", 0.0)
    ):
        failures.append("acceleration not positive enough")

    expected_hold = int(strategy.get("cost_hurdle_expected_hold_signal_bars", 0))
    implied_move_bps = slope_bps * expected_hold
    if bool(strategy.get("require_cost_hurdle", True)):
        hurdle = (
            2.0
            * _all_in_bps_per_side(costs)
            * float(strategy.get("cost_hurdle_multiple", 1.25))
        )
        if implied_move_bps < hurdle:
            failures.append(f"implied move {implied_move_bps:.1f}bps below {hurdle:.1f}bps hurdle")
    return not failures, failures, implied_move_bps


def _exit_condition(
    strategy: dict[str, Any],
    values: dict[str, float | bool | str],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if float(values["slope_score"]) > float(strategy.get("exit_slope_score", -0.08)):
        failures.append("slope score above exit threshold")
    if float(values["fit_r2"]) < float(strategy.get("minimum_exit_fit_r2", 0.0)):
        failures.append("exit fit R2 below threshold")
    if bool(strategy.get("require_negative_acceleration", False)) and float(
        values["acceleration_score"]
    ) > float(strategy.get("maximum_exit_acceleration_score", 0.0)):
        failures.append("acceleration not negative enough")
    return not failures, failures


def decide_slope_strategy(
    strategy: dict[str, Any],
    row: pd.Series,
    state: SlopeStrategyState,
    index: int,
    costs: dict[str, Any],
) -> Decision:
    strategy_id = str(strategy["id"])
    strategy_type = str(strategy["type"])
    if strategy_type == "cash":
        return Decision(strategy_id, 0.0, "cash", "cash benchmark")
    if strategy_type == "buy_hold":
        return Decision(strategy_id, 1.0, "long", "buy-and-hold benchmark")

    values = _slope_values(strategy, row)
    if not bool(values["signal_update"]):
        return Decision(strategy_id, state.target_position, "hold_between_ticks", "no new signal candle")

    state.last_signal_bar_end = str(values["bar_end"])
    entry_ok, entry_failures, implied_move_bps = _entry_condition(strategy, values, costs)
    exit_ok, exit_failures = _exit_condition(strategy, values)

    if state.target_position <= 0.0:
        state.exit_confirmations = 0
        state.entry_confirmations = state.entry_confirmations + 1 if entry_ok else 0
        cooldown_base = _signal_bars_to_base(
            strategy,
            int(strategy.get("cooldown_signal_bars", 1)),
        )
        if _bars_since_trade(state, index) < cooldown_base:
            return Decision(
                strategy_id,
                0.0,
                "hold_cash",
                f"cooldown active; entry confirmations={state.entry_confirmations}",
            )
        required = int(strategy.get("entry_confirmation_ticks", 2))
        if state.entry_confirmations >= required:
            state.entry_confirmations = 0
            return Decision(
                strategy_id,
                1.0,
                "enter_slope_inversion",
                (
                    f"slope={float(values['slope_score']):.3f}, "
                    f"R2={float(values['fit_r2']):.2f}, "
                    f"t={float(values['slope_tstat']):.2f}, "
                    f"implied={implied_move_bps:.1f}bps persisted {required} ticks"
                ),
            )
        reason = "; ".join(entry_failures) if entry_failures else "entry slope confirmation building"
        return Decision(
            strategy_id,
            0.0,
            "hold_cash",
            f"{reason}; confirmations={state.entry_confirmations}/{required}",
        )

    state.entry_confirmations = 0
    emergency_stop = float(strategy.get("emergency_stop_pct", 0.0))
    if emergency_stop > 0.0 and state.entry_mark is not None:
        if float(row["close"]) / float(state.entry_mark) - 1.0 <= -emergency_stop:
            state.exit_confirmations = 0
            return Decision(strategy_id, 0.0, "exit_emergency_stop", "emergency price stop")

    state.exit_confirmations = state.exit_confirmations + 1 if exit_ok else 0
    minimum_hold_base = _signal_bars_to_base(
        strategy,
        int(strategy.get("minimum_hold_signal_bars", 1)),
    )
    held = _bars_held(state, index)
    max_hold_signal = int(strategy.get("maximum_hold_signal_bars", 0))
    if max_hold_signal > 0 and held >= _signal_bars_to_base(strategy, max_hold_signal):
        state.exit_confirmations = 0
        return Decision(strategy_id, 0.0, "exit_time_stop", "maximum holding period reached")

    required = int(strategy.get("exit_confirmation_ticks", 2))
    if held >= minimum_hold_base and state.exit_confirmations >= required:
        state.exit_confirmations = 0
        return Decision(
            strategy_id,
            0.0,
            "exit_slope_inversion",
            (
                f"slope={float(values['slope_score']):.3f}, "
                f"R2={float(values['fit_r2']):.2f}, "
                f"accel={float(values['acceleration_score']):.3f} persisted {required} ticks"
            ),
        )
    reason = "; ".join(exit_failures) if exit_failures else "exit slope confirmation building"
    return Decision(
        strategy_id,
        1.0,
        "hold_btc",
        f"{reason}; confirmations={state.exit_confirmations}/{required}; held={held}",
    )


def execute_slope_decision(
    state: SlopeStrategyState,
    decision: Decision,
    strategy: dict[str, Any],
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
                state.entry_reference = mark
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

    values: dict[str, float | bool | str]
    if str(strategy["type"]) == "rolling_slope":
        values = _slope_values(strategy, signal_row)
    else:
        values = {
            "bar_end": str(signal_row["bar_end"]),
            "signal_update": True,
            "signal_close": float(signal_row["close"]),
            "slope_bps_per_bar": 0.0,
            "slope_score": 0.0,
            "slope_tstat": 0.0,
            "fit_r2": 0.0,
            "acceleration_score": 0.0,
        }

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
        "signal_timeframe": str(strategy.get("signal_timeframe", "benchmark")),
        "slope_signal_bar_end": str(values["bar_end"]),
        "slope_signal_update": bool(values["signal_update"]),
        "slope_bps_per_bar": float(values["slope_bps_per_bar"]),
        "slope_score": float(values["slope_score"]),
        "slope_tstat": float(values["slope_tstat"]),
        "fit_r2": float(values["fit_r2"]),
        "acceleration_score": float(values["acceleration_score"]),
        "entry_confirmations": state.entry_confirmations,
        "exit_confirmations": state.exit_confirmations,
    }
    return snapshot, transaction


def simulate_slope_matrix(
    feature_frame: pd.DataFrame,
    cfg: dict[str, Any],
    costs_override: dict[str, Any] | None = None,
) -> SlopeSimulationResult:
    costs = copy.deepcopy(costs_override if costs_override is not None else cfg["costs"])
    strategies = cfg["strategies"]
    initial_cash = float(cfg["research"].get("initial_cash", 500.0))
    states = {
        str(strategy["id"]): SlopeStrategyState(
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
        raise RuntimeError("No rows contain complete slope feature history.")
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
            decision = decide_slope_strategy(strategy, row, state, index, costs)
            snapshot, transaction = execute_slope_decision(
                state,
                decision,
                strategy,
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
    episodes = build_trade_episodes(snapshot_frame, initial_cash, timeframe_minutes=5)
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
    for strategy in strategies:
        strategy_id = str(strategy["id"])
        metrics = summary[strategy_id]
        metrics["signal_timeframe"] = str(strategy.get("signal_timeframe", "benchmark"))
        metrics["average_holding_hours"] = (
            metrics["average_holding_bars"] * 5.0 / 60.0
            if metrics["average_holding_bars"] is not None
            else None
        )
        strategy_snapshots = snapshot_frame[snapshot_frame["strategy_id"] == strategy_id]
        metrics["time_in_market_pct"] = float(
            (strategy_snapshots["target_position"] > 0.0).mean()
        )
    return SlopeSimulationResult(snapshot_frame, transaction_frame, episodes, summary)


def cost_sensitivity(feature_frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
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
        result = simulate_slope_matrix(feature_frame, cfg, costs_override=override)
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
                    "average_holding_hours": metrics["average_holding_hours"],
                    "time_in_market_pct": metrics["time_in_market_pct"],
                }
            )
    return pd.DataFrame(rows)


def chronological_folds(feature_frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    fold_count = int(cfg["research"].get("chronological_folds", 5))
    if fold_count <= 1:
        return pd.DataFrame()
    valid = feature_frame[feature_frame["feature_valid"]].copy().reset_index(drop=True)
    if len(valid) < fold_count * 100:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    boundaries = np.linspace(0, len(valid), fold_count + 1, dtype=int)
    history_rows = int(cfg["research"].get("fold_history_rows", 1200))
    for fold in range(fold_count):
        score_start = boundaries[fold]
        score_end = boundaries[fold + 1]
        history_start = max(0, score_start - history_rows)
        fold_frame = valid.iloc[history_start:score_end].copy().reset_index(drop=True)
        prior_rows = score_start - history_start
        if prior_rows > 0:
            fold_frame.loc[: prior_rows - 1, "feature_valid"] = False
        if int(fold_frame["feature_valid"].sum()) < 2:
            continue
        result = simulate_slope_matrix(fold_frame, cfg)
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
                    "round_trips_closed": metrics["round_trips_closed"],
                    "turnover_multiple": metrics["turnover_multiple"],
                }
            )
    return pd.DataFrame(rows)


def inversion_forward_returns(feature_frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    horizons = [int(value) for value in cfg["research"].get("diagnostic_horizons_signal_bars", [1, 2, 4, 8, 16])]
    rows: list[dict[str, Any]] = []
    for strategy in cfg["strategies"]:
        if str(strategy["type"]) != "rolling_slope":
            continue
        strategy_id = str(strategy["id"])
        prefix = _feature_prefix(strategy_id)
        updates = feature_frame[
            feature_frame[f"{prefix}signal_update"] & feature_frame["feature_valid"]
        ][
            [
                f"{prefix}signal_close",
                f"{prefix}slope_score",
                f"{prefix}slope_tstat",
                f"{prefix}fit_r2",
                f"{prefix}acceleration_score",
            ]
        ].copy()
        updates.columns = ["close", "slope_score", "slope_tstat", "fit_r2", "acceleration_score"]
        if updates.empty:
            continue
        entry_raw = (
            (updates["slope_score"] >= float(strategy.get("entry_slope_score", 0.15)))
            & (updates["fit_r2"] >= float(strategy.get("minimum_fit_r2", 0.15)))
            & (updates["slope_tstat"] >= float(strategy.get("minimum_entry_tstat", 0.0)))
        )
        if bool(strategy.get("require_positive_acceleration", False)):
            entry_raw &= updates["acceleration_score"] >= float(
                strategy.get("minimum_entry_acceleration_score", 0.0)
            )
        exit_raw = updates["slope_score"] <= float(strategy.get("exit_slope_score", -0.08))
        if bool(strategy.get("require_negative_acceleration", False)):
            exit_raw &= updates["acceleration_score"] <= float(
                strategy.get("maximum_exit_acceleration_score", 0.0)
            )

        entry_n = int(strategy.get("entry_confirmation_ticks", 2))
        exit_n = int(strategy.get("exit_confirmation_ticks", 2))
        entry_confirmed = entry_raw.rolling(entry_n, min_periods=entry_n).sum().eq(entry_n)
        exit_confirmed = exit_raw.rolling(exit_n, min_periods=exit_n).sum().eq(exit_n)
        entry_events = entry_confirmed & ~entry_confirmed.shift(1, fill_value=False)
        exit_events = exit_confirmed & ~exit_confirmed.shift(1, fill_value=False)

        for direction, events in (("up", entry_events), ("down", exit_events)):
            for horizon in horizons:
                future_return = updates["close"].shift(-horizon) / updates["close"] - 1.0
                sample = future_return[events].dropna()
                if sample.empty:
                    continue
                success = sample > 0.0 if direction == "up" else sample < 0.0
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "signal_timeframe": str(strategy["signal_timeframe"]),
                        "direction": direction,
                        "horizon_signal_bars": horizon,
                        "horizon_minutes": int(
                            timeframe_to_timedelta(str(strategy["signal_timeframe"])).total_seconds()
                            // 60
                        )
                        * horizon,
                        "observations": int(len(sample)),
                        "directional_success_probability": float(success.mean()),
                        "mean_future_return_bps": float(sample.mean() * 10_000.0),
                        "median_future_return_bps": float(sample.median() * 10_000.0),
                        "future_return_std_bps": float(sample.std(ddof=0) * 10_000.0),
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
    axis.set_title("Rolling-slope inversion matrix: net equity")
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
    axis.set_title("Rolling-slope inversion matrix: drawdown")
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
    cfg = load_slope_config(config_path)
    market = cfg["market"]
    timeframe = str(market["timeframe"])
    if data_path:
        raw, _ = load_ohlcv_csv(data_path, timeframe=timeframe)
        raw = raw.tail(bars_requested).reset_index(drop=True)
    else:
        step = timeframe_to_timedelta(timeframe)
        start = (pd.Timestamp.now(tz="UTC") - step * (bars_requested + 100)).isoformat()
        raw = download_ohlcv(
            exchange_id=str(market["exchange"]),
            symbol=str(market["symbol"]),
            timeframe=timeframe,
            start=start,
            max_bars=bars_requested,
        )

    feature_frame = build_slope_feature_frame(raw, cfg)
    result = simulate_slope_matrix(feature_frame, cfg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feature_frame.to_csv(output / "slope_feature_frame.csv", index=False)
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

    cost_frame = pd.DataFrame()
    if not skip_cost_grid:
        cost_frame = cost_sensitivity(feature_frame, cfg)
        cost_frame.to_csv(output / "cost_sensitivity.csv", index=False)
    folds = chronological_folds(feature_frame, cfg)
    if not folds.empty:
        folds.to_csv(output / "chronological_folds.csv", index=False)
    diagnostics = inversion_forward_returns(feature_frame, cfg)
    diagnostics.to_csv(output / "inversion_forward_returns.csv", index=False)
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
            "all_in_bps_per_side": _all_in_bps_per_side(cfg["costs"]),
        },
        "strategies": result.summary,
        "outputs": {
            "strategy_comparison": "strategy_comparison.csv",
            "cost_sensitivity": None if skip_cost_grid else "cost_sensitivity.csv",
            "chronological_folds": "chronological_folds.csv" if not folds.empty else None,
            "inversion_forward_returns": "inversion_forward_returns.csv",
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
        description="Research-only rolling-slope inversion strategy matrix"
    )
    parser.add_argument("--config", default="config/settings_slope_matrix.yaml")
    parser.add_argument("--bars", type=int, default=10_000)
    parser.add_argument("--output", default="outputs/slope_matrix_10000")
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
