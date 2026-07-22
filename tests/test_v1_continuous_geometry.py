import pandas as pd
from btc_trend_bot.v1.candlestick_geometry import CONTINUOUS_GEOMETRY_COLUMNS, add_candlestick_geometry


def test_geometry_contains_no_named_pattern_flags():
    assert all(name not in CONTINUOUS_GEOMETRY_COLUMNS for name in [
        "bullish_engulfing", "morning_star", "three_white_soldiers", "hammer"
    ])


def test_continuous_geometry_is_finite_after_warmup():
    frame = pd.DataFrame({
        "open": [100, 101, 100.5, 102],
        "high": [102, 102, 103, 104],
        "low": [99, 100, 100, 101],
        "close": [101, 100.5, 102, 103.5],
        "atr": [2.0, 2.0, 2.0, 2.0],
    })
    result = add_candlestick_geometry(frame)
    assert set(CONTINUOUS_GEOMETRY_COLUMNS).issubset(result.columns)
    assert result.loc[3, "close_location"] > 0.0
