from __future__ import annotations

import numpy as np
import pandas as pd


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
    if mode == "long_short":
        direction.loc[ema_down & (strong_down | out["down_breakout"])] = -1.0
    elif mode != "long_flat":
        raise ValueError("strategy.mode must be 'long_flat' or 'long_short'.")

    target_vol = float(cfg["target_annual_vol"])
    minimum = float(cfg.get("min_position", 0.0))
    maximum = float(cfg.get("max_position", 1.0))
    raw_scalar = target_vol / out["rv_short"].replace(0, np.nan)
    vol_scalar = raw_scalar.clip(lower=minimum, upper=maximum)

    shock_threshold = float(cfg.get("vol_shock_ratio", np.inf))
    shock_multiplier = float(cfg.get("vol_shock_multiplier", 1.0))
    shock = out["vol_shock_ratio"] > shock_threshold
    vol_scalar = vol_scalar.where(~shock, vol_scalar * shock_multiplier)

    out["direction"] = direction
    out["vol_scalar"] = vol_scalar
    out["target_position"] = (direction * vol_scalar).fillna(0.0).clip(-maximum, maximum)
    out["vol_shock"] = shock.fillna(False)
    return out
