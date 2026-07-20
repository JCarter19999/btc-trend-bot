from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    components = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return components.max(axis=1).rename("true_range")


def realized_volatility(log_returns: pd.Series, window: int, bars_per_year: int) -> pd.Series:
    variance = log_returns.pow(2).rolling(window, min_periods=window).mean()
    return np.sqrt(variance * bars_per_year).rename(f"rv_{window}")


def add_features(frame: pd.DataFrame, strategy_cfg: dict, bars_per_year: int) -> pd.DataFrame:
    out = frame.copy()
    fast = int(strategy_cfg["fast_ema_bars"])
    slow = int(strategy_cfg["slow_ema_bars"])
    breakout = int(strategy_cfg["breakout_bars"])
    atr_bars = int(strategy_cfg["atr_bars"])
    rv_bars = int(strategy_cfg["realized_vol_bars"])
    long_vol_bars = int(strategy_cfg["long_vol_bars"])

    out["simple_return"] = out["close"].pct_change()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["ema_fast"] = out["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    out["ema_slow"] = out["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    out["true_range"] = true_range(out)
    out["atr"] = out["true_range"].ewm(span=atr_bars, adjust=False, min_periods=atr_bars).mean()
    out["atr_pct"] = out["atr"] / out["close"]
    out["trend_strength"] = (out["ema_fast"] - out["ema_slow"]) / out["atr"].replace(0, np.nan)

    out["prior_breakout_high"] = out["high"].rolling(breakout, min_periods=breakout).max().shift(1)
    out["prior_breakout_low"] = out["low"].rolling(breakout, min_periods=breakout).min().shift(1)
    out["up_breakout"] = out["close"] > out["prior_breakout_high"]
    out["down_breakout"] = out["close"] < out["prior_breakout_low"]

    out["rv_short"] = realized_volatility(out["log_return"], rv_bars, bars_per_year)
    out["rv_long"] = realized_volatility(out["log_return"], long_vol_bars, bars_per_year)
    out["vol_shock_ratio"] = out["rv_short"] / out["rv_long"].replace(0, np.nan)

    volume_mean = out["volume"].rolling(long_vol_bars, min_periods=long_vol_bars).mean()
    volume_std = out["volume"].rolling(long_vol_bars, min_periods=long_vol_bars).std(ddof=0)
    out["volume_zscore"] = (out["volume"] - volume_mean) / volume_std.replace(0, np.nan)
    return out
