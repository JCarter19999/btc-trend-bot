import numpy as np
import pandas as pd
from btc_trend_bot.features import true_range
from .candlestick_geometry import add_candlestick_geometry, GEOMETRY_COLUMNS

BASE_FEATURE_COLUMNS = ["return_1", "return_3", "return_6", "return_12", "ema_spread_atr", "ema_fast_slope_atr", "atr_pct", "volatility_12", "volume_zscore", "body_fraction", "upper_wick_fraction", "lower_wick_fraction", "regime_spread_bps", "momentum_bps", "hour_sin", "hour_cos", "weekend", "entry_run_return_bps", "relative_volume"]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + GEOMETRY_COLUMNS

def build_features(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = frame.copy().sort_values("open_time").reset_index(drop=True)
    strategy = cfg["strategy"]
    close = out["close"]
    for bars in (1, 3, 6, 12):
        out[f"return_{bars}"] = close.pct_change(bars)
    out["atr"] = true_range(out).ewm(span=int(strategy["atr_bars"]), adjust=False, min_periods=int(strategy["atr_bars"])).mean()
    fast = close.ewm(span=int(strategy["regime_fast_ema_bars"]), adjust=False).mean()
    slow = close.ewm(span=int(strategy["regime_slow_ema_bars"]), adjust=False).mean()
    out["ema_spread_atr"] = (fast - slow) / out["atr"].replace(0, np.nan)
    out["ema_fast_slope_atr"] = fast.diff() / out["atr"].replace(0, np.nan)
    out["atr_pct"] = out["atr"] / close
    out["volatility_12"] = out["return_1"].rolling(12).std(ddof=0)
    volume_mean = out["volume"].rolling(48).mean()
    volume_std = out["volume"].rolling(48).std(ddof=0)
    out["volume_zscore"] = (out["volume"] - volume_mean) / volume_std.replace(0, np.nan)
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_fraction"] = (out["close"] - out["open"]).abs() / candle_range
    out["upper_wick_fraction"] = (out["high"] - out[["open", "close"]].max(axis=1)) / candle_range
    out["lower_wick_fraction"] = (out[["open", "close"]].min(axis=1) - out["low"]) / candle_range
    out["regime_spread_bps"] = (fast / slow - 1) * 10000
    out["momentum_bps"] = (close / close.shift(int(strategy["momentum_bars"])) - 1) * 10000
    hour = out["open_time"].dt.hour + out["open_time"].dt.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekend"] = (out["open_time"].dt.dayofweek >= 5).astype(float)
    run_bars = int(strategy["entry_run_bars"])
    out["entry_run_return_bps"] = (close / close.shift(run_bars) - 1) * 10000
    out["relative_volume"] = out["volume"] / volume_mean.replace(0, np.nan)
    out = add_candlestick_geometry(out)
    out["feature_valid"] = out[FEATURE_COLUMNS + ["atr"]].notna().all(axis=1)
    return out
