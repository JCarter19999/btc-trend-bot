import pandas as pd

from btc_trend_bot.selective_reversion_matrix import (
    SelectiveSetupState,
    decide,
    load_config,
)
from btc_trend_bot.popular_matrix import StrategyState

COSTS = {
    "fee_bps_per_side": 2,
    "slippage_bps_per_side": 5,
    "assumed_spread_bps_per_side": 1,
}


def _row(rsi: float, *, recovery_2: bool = False) -> pd.Series:
    return pd.Series(
        {
            "rsi2_5m": rsi,
            "rsi2_5m_prev": rsi - 1,
            "rsi5_recovery_1": recovery_2,
            "rsi5_recovery_2": recovery_2,
            "m15_rsi2": rsi,
            "m15_rsi_recovery_1": recovery_2,
            "m15_rsi_recovery_2": recovery_2,
            "h4_trend_bps": 10,
            "d1_trend_bps": 10,
            "d1_momentum_bps": 10,
            "m15_atr_ratio_to_median": 1,
            "m15_atr_bps": 30,
            "m15_atr": 100,
            "high": 100,
            "close": 100,
        }
    )


def _strategy() -> dict:
    return {
        "id": "x",
        "type": "selective_rsi2",
        "signal_source": "5m",
        "entry_rsi": 5,
        "recovery_ticks": 2,
        "setup_expiry_bars_5m": 12,
        "min_h4_trend_bps": 0,
        "min_d1_trend_bps": -1,
        "min_d1_momentum_bps": -1,
        "min_atr_ratio": 0,
        "max_atr_ratio": 2,
        "expected_move_atr_multiple": 2,
        "cost_hurdle_multiple": 1,
        "cooldown_bars_5m": 0,
    }


def test_config_loads():
    cfg = load_config("config/settings_selective_reversion.yaml")
    assert len(cfg["strategies"]) == 7
    control = next(s for s in cfg["strategies"] if s["id"] == "rsi2_v06_control")
    assert control["type"] == "rsi2_v06_control"


def test_selective_setup_arms_then_enters_after_recovery():
    strategy = _strategy()
    state = StrategyState("x", 500, 500, 0, 500, 0, peak_equity=500)
    setup = SelectiveSetupState()
    diagnostics = {}

    armed = decide(strategy, _row(2), state, 100, COSTS, setup, diagnostics)
    assert armed.target_position == 0
    assert armed.signal == "arm_oversold"
    assert setup.armed

    # The recovery candle no longer needs to remain below the oversold threshold.
    entered = decide(strategy, _row(15, recovery_2=True), state, 101, COSTS, setup, diagnostics)
    assert entered.target_position == 1
    assert entered.signal == "enter_selective_rsi"
    assert not setup.armed


def test_economic_hurdle_is_diagnostic_not_entry_blocker():
    strategy = _strategy()
    strategy["expected_move_atr_multiple"] = 0
    state = StrategyState("x", 500, 500, 0, 500, 0, peak_equity=500)
    setup = SelectiveSetupState()
    diagnostics = {}

    decide(strategy, _row(2), state, 100, COSTS, setup, diagnostics)
    entered = decide(strategy, _row(15, recovery_2=True), state, 101, COSTS, setup, diagnostics)
    assert entered.target_position == 1
    assert diagnostics[("x", "economic_hurdle_fail")] == 1


def test_control_routes_to_original_v06_logic():
    strategy = {
        "id": "control",
        "type": "rsi2_v06_control",
        "entry_rsi": 8,
        "min_h4_trend_bps": 0,
        "cooldown_bars_5m": 0,
    }
    state = StrategyState("control", 500, 500, 0, 500, 0, peak_equity=500)
    decision = decide(strategy, _row(3, recovery_2=False), state, 100, COSTS)
    assert decision.target_position == 1
    assert decision.signal == "enter_rsi2"
