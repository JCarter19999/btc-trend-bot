import numpy as np
import pandas as pd

from btc_trend_bot.analytics import extract_trades, market_capture, summarize_trades, yearly_performance


def make_bars():
    returns = pd.Series([0.0, 0.10, -0.02, -0.001, 0.0, -0.05, 0.02, -0.001])
    equity = 10000 * (1 + returns).cumprod()
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC"),
            "close": [100, 110, 108, 108, 108, 103, 105, 105],
            "position": [0, 1, 1, 0, 0, -1, -1, 0],
            "return": returns,
            "equity": equity,
            "benchmark": [0.0, 0.10, -0.02, 0.0, 0.0, -0.05, 0.02, 0.0],
        }
    )


def test_extract_trades_finds_long_and_short_episodes():
    trades = extract_trades(make_bars(), "position", "return", "equity", "test")
    assert len(trades) == 2
    assert trades["side"].tolist() == ["long", "short"]
    assert trades["active_bars"].tolist() == [2, 2]
    assert trades.iloc[0]["trade_return"] < 0.10


def test_trade_summary_reports_concentration():
    trades = extract_trades(make_bars(), "position", "return", "equity", "test")
    summary = summarize_trades(trades)
    assert summary["trades"] == 2
    assert 0 <= summary["win_rate"] <= 1


def test_yearly_performance_has_each_strategy_year():
    bars = make_bars()
    result = yearly_performance(bars, {"strategy": "return", "benchmark": "benchmark"}, 2190)
    assert set(result["strategy"]) == {"strategy", "benchmark"}
    assert set(result["year"]) == {2024}


def test_market_capture_beta_is_finite():
    bars = make_bars()
    capture = market_capture(bars["return"], bars["benchmark"], 2190)
    assert np.isfinite(capture["beta"])
    assert np.isfinite(capture["correlation"])
