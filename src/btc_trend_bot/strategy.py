from __future__ import annotations

import numpy as np
import pandas as pd


def _side_size(direction: pd.Series, long_size: float, short_size: float) -> pd.Series:
    return direction.where(direction >= 0, direction.abs() * -short_size).where(direction <= 0, direction * long_size)


def apply_short_trailing_stop(
    frame: pd.DataFrame,
    position_column: str,
    atr_multiple: float,
) -> pd.Series:
    """Stop shorts after a rebound from the lowest close since entry.

    Once stopped, the short remains blocked until the underlying signal exits the
    short state. This prevents an immediate re-entry on the next bar.
    """
    base = frame[position_column].fillna(0.0)
    if atr_multiple <= 0:
        return base.copy()

    adjusted: list[float] = []
    short_active = False
    blocked = False
    lowest_close = np.inf

    for idx, desired in base.items():
        close = float(frame.at[idx, "close"])
        atr = frame.at[idx, "atr"]
        atr_value = float(atr) if pd.notna(atr) else np.nan

        if desired >= 0:
            short_active = False
            blocked = False
            lowest_close = np.inf
            adjusted.append(float(desired))
            continue

        if blocked:
            adjusted.append(0.0)
            continue

        if not short_active:
            short_active = True
            lowest_close = close
        else:
            lowest_close = min(lowest_close, close)

        stop_level = lowest_close + atr_multiple * atr_value if np.isfinite(atr_value) else np.inf
        if close > stop_level:
            short_active = False
            blocked = True
            adjusted.append(0.0)
        else:
            adjusted.append(float(desired))

    return pd.Series(adjusted, index=frame.index, name=position_column)


def build_target_positions(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = frame.copy()
    mode = str(cfg.get("mode", "long_flat"))
    threshold = float(cfg.get("trend_strength_atr", 0.35))

    ema_up = out["ema_fast"] > out["ema_slow"]
    ema_down = out["ema_fast"] < out["ema_slow"]
    strong_up = out["trend_strength"] >= threshold
    strong_down = out["trend_strength"] <= -threshold

    direction = pd.Series(0.0, index=out.index, name="direction")
    direction.loc[ema_up & (strong_up | out["up_breakout"])] = 1.0

    if mode in {"long_short", "selective_short"}:
        if mode == "long_short":
            short_signal = ema_down & (strong_down | out["down_breakout"])
        else:
            short_signal = ema_down & strong_down & out["down_breakout"] & (out["close"] < out["ema_slow"])
            if bool(cfg.get("short_regime_filter", False)):
                short_signal &= out["close"] < out["ema_regime"]
                short_signal &= out["ema_regime"] < out["ema_regime"].shift(1)
        direction.loc[short_signal] = -1.0
    elif mode != "long_flat":
        raise ValueError("strategy.mode must be 'long_flat', 'long_short', or 'selective_short'.")

    long_size = float(cfg.get("long_position_size", cfg.get("max_position", 1.0)))
    short_size = float(cfg.get("short_position_size", cfg.get("max_position", 1.0)))
    maximum = float(cfg.get("max_position", 1.0))
    minimum = float(cfg.get("min_position", 0.0))

    target_vol = float(cfg["target_annual_vol"])
    raw_scalar = target_vol / out["rv_short"].replace(0, np.nan)
    vol_scalar = raw_scalar.clip(lower=minimum, upper=maximum)

    shock_threshold = float(cfg.get("vol_shock_ratio", np.inf))
    shock_multiplier = float(cfg.get("vol_shock_multiplier", 1.0))
    shock = out["vol_shock_ratio"] > shock_threshold
    vol_scalar = vol_scalar.where(~shock, vol_scalar * shock_multiplier)

    side_sizes = _side_size(direction, long_size=long_size, short_size=short_size)
    out["direction"] = direction
    out["vol_scalar"] = vol_scalar
    out["fixed_target_position"] = side_sizes.fillna(0.0)
    out["target_position"] = (side_sizes * vol_scalar).fillna(0.0).clip(-short_size, long_size)
    out["vol_shock"] = shock.fillna(False)

    trailing_atr = float(cfg.get("short_trailing_stop_atr", 0.0))
    if trailing_atr > 0:
        out["target_position"] = apply_short_trailing_stop(out, "target_position", trailing_atr)
        out["fixed_target_position"] = apply_short_trailing_stop(
            out, "fixed_target_position", trailing_atr
        )
        out["direction"] = np.sign(out["fixed_target_position"])
    return out
