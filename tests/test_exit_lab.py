import pandas as pd

from btc_trend_bot import exit_lab
from btc_trend_bot import popular_matrix as base
from btc_trend_bot import selective_reversion_matrix as selective


def _state(entry=100.0):
    return base.StrategyState(
        strategy_id="x", initial_cash=500, cash=0, btc=5,
        gross_cash=0, gross_btc=5, target_position=1,
        entry_index=0, entry_mark=entry, highest_high=entry,
    )


def _row(close, high=None, rsi=50):
    return pd.Series({
        "close": close, "high": close if high is None else high,
        "m15_rsi2": rsi, "m15_atr": 1.0,
        "h4_trend_bps": 100, "d1_momentum_bps": 100,
    })


def test_fixed_target_exits_at_target():
    s = {"id": "x", "exit_mode": "fixed_target", "target_bps": 20, "stop_bps": 50, "maximum_hold_bars_5m": 100}
    d = exit_lab._exit_decision(s, _row(100.25), _state(), 5)
    assert d.signal == "exit_target"


def test_activated_trail_requires_activation_then_drawdown():
    s = {"id": "x", "exit_mode": "activated_trailing", "activation_bps": 25, "trail_bps": 15, "stop_bps": 50, "maximum_hold_bars_5m": 100}
    st = _state(); st.highest_high = 100.40
    d = exit_lab._exit_decision(s, _row(100.20, high=100.40), st, 5)
    assert d.signal == "exit_activated_trail"


def test_expanded_variants_share_entry_template():
    cfg = {
        "entry_template": {"entry_rsi": 3, "signal_source": "15m"},
        "strategies": [{"id": "v", "type": "exit_variant", "exit_mode": "fixed_target", "target_bps": 20, "stop_bps": 35}],
    }
    s = exit_lab._expanded_strategies(cfg)[0]
    assert s["type"] == "selective_rsi2"
    assert s["entry_rsi"] == 3
    assert s["exit_mode"] == "fixed_target"
