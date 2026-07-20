import numpy as np
import pandas as pd

from btc_trend_bot.backtest import run_backtest


BASE_CFG = {
    "initial_cash": 10000,
    "fee_bps_per_turnover": 0,
    "slippage_bps_per_turnover": 0,
    "max_drawdown_breaker": 0.25,
}


def make_frame(returns, targets):
    close = 100 * np.cumprod(1 + np.array(returns))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(returns), freq="4h", tz="UTC"),
            "close": close,
            "simple_return": returns,
            "target_position": targets,
        }
    )


def test_signal_is_lagged_one_bar():
    frame = make_frame([0.0, 0.10, 0.10], [1.0, 1.0, 1.0])
    result = run_backtest(frame, BASE_CFG).bars
    assert result.loc[0, "held_position"] == 0.0
    assert result.loc[1, "held_position"] == 1.0
    assert np.isclose(result.loc[1, "strategy_return"], 0.10)


def test_costs_are_charged_on_turnover():
    cfg = dict(BASE_CFG, fee_bps_per_turnover=10, slippage_bps_per_turnover=0)
    frame = make_frame([0.0, 0.0], [1.0, 1.0])
    result = run_backtest(frame, cfg).bars
    assert np.isclose(result.loc[1, "strategy_return"], -0.001)


def test_drawdown_breaker_halts_future_positions():
    cfg = dict(BASE_CFG, max_drawdown_breaker=0.10)
    frame = make_frame([0.0, -0.20, 0.10, 0.10], [1.0, 1.0, 1.0, 1.0])
    result = run_backtest(frame, cfg)
    assert result.breaker_timestamp is not None
    assert result.bars.loc[2, "held_position"] == 0.0
