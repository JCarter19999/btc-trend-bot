from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import math
import pandas as pd
from .types import Candidate
from .labeling import round_trip_cost_return

@dataclass(frozen=True)
class ExitEngineConfig:
    max_hold_bars: int = 144
    collision_policy: str = "stop_first"
    breakeven_enabled: bool = True
    breakeven_trigger_atr: float = 1.0
    breakeven_offset_bps: float = 2.0
    trailing_enabled: bool = True
    trailing_activation_atr: float = 1.5
    trailing_distance_atr: float = 1.0
    chandelier_lookback_bars: int = 24
    momentum_exit_enabled: bool = True
    momentum_lookback_bars: int = 6
    momentum_exit_bps: float = -12.0
    trend_exit_enabled: bool = True
    trend_fast_bars: int = 6
    trend_slow_bars: int = 18
    volatility_exit_enabled: bool = True
    volatility_range_atr: float = 3.5
    min_hold_before_soft_exit_bars: int = 3
    volatility_adaptive_enabled: bool = False
    volatility_lookback_bars: int = 12
    volatility_factor_min: float = 1.0
    volatility_factor_max: float = 2.5
    volatility_reference_atr: float = 1.0
    volatility_atr_pct_reference: float = 0.0015
    trailing_volatility_exponent: float = 0.75
    soft_exit_delay_volatility_exponent: float = 1.0
    momentum_volatility_exponent: float = 0.75

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], max_hold_bars: int | None = None) -> "ExitEngineConfig":
        adaptive = cfg.get("adaptive", {})
        values = {name: adaptive.get(name, getattr(cls(), name)) for name in cls.__dataclass_fields__}
        values["max_hold_bars"] = int(max_hold_bars if max_hold_bars is not None else cfg.get("max_hold_bars", values["max_hold_bars"]))
        values["collision_policy"] = cfg.get("collision_policy", "stop_first")
        return cls(**values)


def _ema(values: list[float], span: int) -> float:
    if not values:
        return math.nan
    return float(pd.Series(values).ewm(span=span, adjust=False).mean().iloc[-1])


def run_exit_engine(candidate: Candidate, future: pd.DataFrame, cost_cfg: dict, engine_cfg: ExitEngineConfig) -> dict[str, Any]:
    rows = future[future["open_time"] > pd.Timestamp(candidate.timestamp)].head(engine_cfg.max_hold_bars)
    if rows.empty:
        return {**asdict(candidate), "label_status": "PENDING"}

    stop = float(candidate.stop_price)
    target = float(candidate.target_price)
    dynamic_stop = stop
    highest = candidate.entry_price
    closes: list[float] = []
    exit_price = float(rows.iloc[-1]["close"])
    reason = "time_exit"
    ambiguous = False
    maturity = rows.iloc[-1]["open_time"]
    mfe = 0.0
    mae = 0.0
    bars_held = 0
    breakeven_armed = False
    trailing_armed = False
    volatility_factor = 1.0
    recent_ranges: list[float] = []

    for bars_held, (_, row) in enumerate(rows.iterrows(), start=1):
        high, low, close = float(row.high), float(row.low), float(row.close)
        highest = max(highest, high)
        closes.append(close)
        recent_ranges.append(high-low)
        if engine_cfg.volatility_adaptive_enabled:
            lookback = recent_ranges[-max(1, engine_cfg.volatility_lookback_bars):]
            observed = float(pd.Series(lookback).median()) / max(candidate.atr * engine_cfg.volatility_reference_atr, 1e-12)
            entry_atr_pct = float(candidate.features.get("atr_pct", candidate.atr / max(candidate.entry_price, 1e-12)))
            regime_factor = entry_atr_pct / max(engine_cfg.volatility_atr_pct_reference, 1e-12)
            volatility_factor = min(engine_cfg.volatility_factor_max, max(engine_cfg.volatility_factor_min, observed, regime_factor))
        else:
            volatility_factor = 1.0
        mfe = max(mfe, high / candidate.entry_price - 1.0)
        mae = min(mae, low / candidate.entry_price - 1.0)

        hard_stop_hit = low <= dynamic_stop
        target_hit = high >= target
        if hard_stop_hit and target_hit:
            if engine_cfg.collision_policy != "stop_first":
                raise ValueError("Only stop_first is supported in v1")
            exit_price, reason, ambiguous, maturity = dynamic_stop, "stop_loss", True, row.open_time
            break
        if hard_stop_hit:
            exit_price, reason, maturity = dynamic_stop, "stop_loss" if dynamic_stop <= stop + 1e-12 else "adaptive_stop", row.open_time
            break
        if target_hit:
            exit_price, reason, maturity = target, "profit_target", row.open_time
            break

        favorable_atr = (highest - candidate.entry_price) / max(candidate.atr, 1e-12)
        if engine_cfg.breakeven_enabled and favorable_atr >= engine_cfg.breakeven_trigger_atr:
            breakeven_armed = True
            be = candidate.entry_price * (1.0 + engine_cfg.breakeven_offset_bps / 10000.0)
            dynamic_stop = max(dynamic_stop, be)

        if engine_cfg.trailing_enabled and favorable_atr >= engine_cfg.trailing_activation_atr:
            trailing_armed = True
            recent_high = highest
            if engine_cfg.chandelier_lookback_bars > 0:
                recent = rows.iloc[max(0, bars_held-engine_cfg.chandelier_lookback_bars):bars_held]
                recent_high = float(recent.high.max())
            trail_distance = engine_cfg.trailing_distance_atr * (volatility_factor ** engine_cfg.trailing_volatility_exponent)
            dynamic_stop = max(dynamic_stop, recent_high - trail_distance * candidate.atr)

        soft_exit_min_bars = int(math.ceil(engine_cfg.min_hold_before_soft_exit_bars * (volatility_factor ** engine_cfg.soft_exit_delay_volatility_exponent)))
        if bars_held >= soft_exit_min_bars:
            candle_range = high - low
            if engine_cfg.volatility_exit_enabled and candle_range >= engine_cfg.volatility_range_atr * volatility_factor * candidate.atr and close < candidate.entry_price:
                exit_price, reason, maturity = close, "volatility_shock", row.open_time
                break
            if engine_cfg.momentum_exit_enabled and len(closes) >= engine_cfg.momentum_lookback_bars:
                mom_bps = (closes[-1] / closes[-engine_cfg.momentum_lookback_bars] - 1.0) * 10000.0
                adjusted_momentum_exit = engine_cfg.momentum_exit_bps * (volatility_factor ** engine_cfg.momentum_volatility_exponent)
                if mom_bps <= adjusted_momentum_exit:
                    exit_price, reason, maturity = close, "momentum_decay", row.open_time
                    break
            if engine_cfg.trend_exit_enabled and len(closes) >= engine_cfg.trend_slow_bars:
                if _ema(closes, engine_cfg.trend_fast_bars) < _ema(closes, engine_cfg.trend_slow_bars):
                    exit_price, reason, maturity = close, "trend_reversal", row.open_time
                    break

    gross = exit_price / candidate.entry_price - 1.0
    net = gross - round_trip_cost_return(cost_cfg)
    return {
        **asdict(candidate), "label_status": "MATURED", "maturity_timestamp": maturity,
        "exit_price": exit_price, "exit_reason": reason, "ambiguous_same_bar": ambiguous,
        "target_hit_first": reason == "profit_target", "gross_return": gross, "net_return": net,
        "positive_net": int(net > 0), "bars_held": bars_held, "mfe_return": mfe, "mae_return": mae,
        "initial_stop_price": stop, "final_dynamic_stop": dynamic_stop,
        "breakeven_armed": breakeven_armed, "trailing_armed": trailing_armed,
        "final_volatility_factor": volatility_factor,
    }
