import numpy as np
import pandas as pd

from btc_trend_bot.strategy import apply_short_trailing_stop, build_target_positions


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "close": [100.0, 99.0, 98.0, 97.0],
        "ema_fast": [99.0] * 4,
        "ema_slow": [101.0] * 4,
        "ema_regime": [102.0] * 4,
        "trend_strength": [-0.5] * 4,
        "up_breakout": [False] * 4,
        "down_breakout": [True] * 4,
        "rv_short": [0.2] * 4,
        "vol_shock_ratio": [1.0] * 4,
        "atr": [2.0] * 4,
    })


def _cfg(**overrides):
    cfg = {
        "mode": "selective_short",
        "trend_strength_atr": 0.35,
        "target_annual_vol": 0.35,
        "min_position": 0.0,
        "max_position": 1.0,
        "long_position_size": 1.0,
        "short_position_size": 0.5,
        "vol_shock_ratio": 1.8,
        "vol_shock_multiplier": 0.5,
        "short_regime_filter": False,
        "short_trailing_stop_atr": 0.0,
    }
    cfg.update(overrides)
    return cfg


def test_selective_short_uses_half_size():
    out = build_target_positions(_frame(), _cfg())
    assert (out["fixed_target_position"] == -0.5).all()


def test_regime_filter_blocks_short_above_regime_ema():
    frame = _frame()
    frame["close"] = 103.0
    out = build_target_positions(frame, _cfg(short_regime_filter=True))
    assert (out["fixed_target_position"] == 0.0).all()


def test_trailing_stop_blocks_reentry_until_signal_resets():
    frame = pd.DataFrame({
        "close": [100.0, 95.0, 101.0, 99.0, 98.0, 97.0],
        "atr": [2.0] * 6,
        "position": [-0.5, -0.5, -0.5, -0.5, 0.0, -0.5],
    })
    result = apply_short_trailing_stop(frame, "position", 2.5)
    assert np.allclose(result.to_numpy(), [-0.5, -0.5, 0.0, 0.0, 0.0, -0.5])
