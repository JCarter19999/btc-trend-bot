import pandas as pd
from btc_trend_bot.v1.candlestick_geometry import add_candlestick_geometry


def frame(rows):
    return pd.DataFrame(rows, columns=['open','high','low','close']).assign(atr=1.0)


def test_engulfing_strength_is_continuous_and_signed():
    df=frame([(10,10.2,8.8,9.0),(8.9,10.3,8.7,10.2)])
    out=add_candlestick_geometry(df)
    assert out.iloc[-1].engulfing_strength_signed > 1.0


def test_hammer_geometry_score():
    df=frame([(10,10.2,8.0,10.1)])
    out=add_candlestick_geometry(df)
    assert out.iloc[-1].hammer_score > out.iloc[-1].inverted_hammer_score
