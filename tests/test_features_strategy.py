import numpy as np
import pandas as pd

from btc_trend_bot.features import add_features, realized_volatility
from btc_trend_bot.strategy import build_target_positions
from btc_trend_bot.synthetic import generate_synthetic_ohlcv


CFG = {
    "mode": "long_flat",
    "fast_ema_bars": 10,
    "slow_ema_bars": 30,
    "breakout_bars": 20,
    "atr_bars": 10,
    "realized_vol_bars": 10,
    "long_vol_bars": 30,
    "target_annual_vol": 0.35,
    "min_position": 0.0,
    "max_position": 1.0,
    "trend_strength_atr": 0.1,
    "vol_shock_ratio": 2.0,
    "vol_shock_multiplier": 0.5,
}


def test_realized_volatility_math():
    returns = pd.Series([0.01] * 20)
    rv = realized_volatility(returns, window=10, bars_per_year=100)
    assert np.isclose(rv.iloc[-1], 0.1)


def test_position_is_bounded_and_long_only():
    frame = generate_synthetic_ohlcv(rows=300)
    featured = add_features(frame, CFG, bars_per_year=2190)
    positioned = build_target_positions(featured, CFG)
    assert positioned["target_position"].min() >= 0
    assert positioned["target_position"].max() <= 1


def test_future_prices_do_not_change_prior_signal():
    frame = generate_synthetic_ohlcv(rows=400)
    original = build_target_positions(add_features(frame, CFG, 2190), CFG)
    modified = frame.copy()
    modified.loc[350:, ["open", "high", "low", "close"]] *= 3
    changed = build_target_positions(add_features(modified, CFG, 2190), CFG)
    pd.testing.assert_series_equal(
        original.loc[:349, "target_position"],
        changed.loc[:349, "target_position"],
        check_names=False,
    )
