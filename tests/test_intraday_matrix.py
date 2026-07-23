from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from btc_trend_bot.intraday_matrix import (
    Decision,
    StrategyState,
    _completed_resample,
    build_feature_frame,
    build_trade_episodes,
    decide_strategy,
    execute_decision,
    load_matrix_config,
    simulate_matrix,
)


COSTS = {
    "fee_bps_per_side": 2.0,
    "slippage_bps_per_side": 5.0,
    "assumed_spread_bps_per_side": 1.0,
    "min_notional": 1.0,
    "rebalance_tolerance_bps": 0.0,
}


def state(strategy_id: str = "test") -> StrategyState:
    return StrategyState(
        strategy_id=strategy_id,
        initial_cash=500.0,
        cash=500.0,
        btc=0.0,
        gross_cash=500.0,
        gross_btc=0.0,
        peak_equity=500.0,
    )


def row(**overrides) -> pd.Series:
    values = {
        "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.5,
        "volume": 20.0,
        "atr_5m": 1.0,
        "range_atr_ratio": 3.0,
        "prior_breakout_high": 101.0,
        "relative_volume": 2.0,
        "body_fraction": 0.7,
        "streak_direction": 1,
        "streak_length": 2,
        "run_return_bps": 25.0,
        "execution_ema_5m": 100.0,
        "recent_pullback_bps": -10.0,
        "recovery_cross_up": True,
        "vwap_zscore": -2.2,
        "selling_pressure_easing": True,
        "one_hour_ema_spread_bps": 15.0,
        "one_hour_momentum_bps": 20.0,
        "four_hour_momentum_bps": 10.0,
        "feature_valid": True,
    }
    values.update(overrides)
    return pd.Series(values)


def next_row(open_price: float = 102.0) -> pd.Series:
    return pd.Series(
        {
            "timestamp": pd.Timestamp("2026-01-01T00:05:00Z"),
            "open": open_price,
            "high": open_price + 1.0,
            "low": open_price - 1.0,
            "close": open_price,
            "volume": 20.0,
        }
    )


def matrix_config() -> dict:
    return load_matrix_config("config/settings_intraday_matrix.yaml")


def synthetic_ohlcv(rows: int = 1200) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    trend = np.linspace(100.0, 120.0, rows)
    wave = np.sin(np.arange(rows) / 15.0) * 0.5
    close = trend + wave
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.4
    low = np.minimum(open_, close) - 0.4
    volume = 10.0 + (np.arange(rows) % 12)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_completed_hour_is_not_available_before_twelfth_five_minute_bar():
    frame = synthetic_ohlcv(12)
    first_eleven = _completed_resample(
        frame.iloc[:11],
        pd.Timedelta(minutes=5),
        "1h",
        12,
    )
    full_hour = _completed_resample(
        frame,
        pd.Timedelta(minutes=5),
        "1h",
        12,
    )
    assert first_eleven.empty
    assert len(full_hour) == 1
    assert full_hour.iloc[0]["bar_end"] == pd.Timestamp("2026-01-01T01:00:00Z")


def test_feature_frame_has_causal_higher_timeframe_features():
    cfg = matrix_config()
    features = build_feature_frame(synthetic_ohlcv(), cfg)
    assert features["feature_valid"].any()
    first_valid = features.index[features["feature_valid"]][0]
    assert first_valid >= 800
    assert pd.notna(features.loc[first_valid, "one_hour_ema_spread_bps"])
    assert pd.notna(features.loc[first_valid, "four_hour_momentum_bps"])


def test_breakout_enters_only_with_all_confirmations():
    strategy = {
        "id": "breakout",
        "type": "volatility_breakout",
        "minimum_range_atr": 1.25,
        "minimum_relative_volume": 1.5,
        "entry_1h_ema_spread_bps": 0.0,
        "entry_1h_momentum_bps": 0.0,
        "minimum_4h_momentum_bps": 0.0,
        "cooldown_bars_5m": 12,
    }
    decision = decide_strategy(strategy, row(), state("breakout"), 100, COSTS)
    assert decision.target_position == 1.0
    assert decision.signal == "enter_breakout"
    assert decision.entry_reference == 101.0

    failed = decide_strategy(
        strategy,
        row(relative_volume=1.0),
        state("breakout"),
        100,
        COSTS,
    )
    assert failed.target_position == 0.0
    assert "volume" in failed.reason


def test_breakout_trailing_stop_exits():
    strategy = {
        "id": "breakout",
        "type": "volatility_breakout",
        "minimum_hold_bars_5m": 12,
        "maximum_hold_bars_5m": 144,
        "trailing_stop_atr": 2.5,
        "exit_1h_ema_spread_bps": -10.0,
        "exit_1h_momentum_bps": -10.0,
        "exit_confirmations_required": 2,
    }
    account = state("breakout")
    account.target_position = 1.0
    account.entry_index = 10
    account.entry_reference = 100.0
    account.highest_high = 110.0
    decision = decide_strategy(
        strategy,
        row(close=106.0, high=109.0, atr_5m=1.0),
        account,
        30,
        COSTS,
    )
    assert decision.target_position == 0.0
    assert decision.signal == "exit_trailing"


def test_vwap_reversion_entry_and_exit():
    strategy = {
        "id": "vwap",
        "type": "vwap_reversion",
        "entry_z_score": -2.0,
        "exit_z_score": -0.25,
        "stop_z_score": -3.25,
        "minimum_1h_ema_spread_bps": -20.0,
        "require_selling_pressure_easing": True,
        "minimum_hold_bars_5m": 3,
        "maximum_hold_bars_5m": 72,
        "cooldown_bars_5m": 12,
        "exit_1h_ema_spread_bps": -25.0,
        "exit_1h_momentum_bps": -25.0,
        "exit_confirmations_required": 2,
    }
    enter = decide_strategy(strategy, row(), state("vwap"), 100, COSTS)
    assert enter.target_position == 1.0
    assert enter.signal == "enter_reversion"

    account = state("vwap")
    account.target_position = 1.0
    account.entry_index = 90
    exit_decision = decide_strategy(
        strategy,
        row(vwap_zscore=-0.1),
        account,
        100,
        COSTS,
    )
    assert exit_decision.target_position == 0.0
    assert exit_decision.signal == "exit_vwap"


def test_momentum_immediate_uses_one_hour_regime():
    strategy = {
        "id": "momentum",
        "type": "momentum_1h",
        "entry_1h_ema_spread_bps": 0.0,
        "entry_1h_momentum_bps": 0.0,
        "exit_1h_ema_spread_bps": -10.0,
        "exit_1h_momentum_bps": -10.0,
        "exit_confirmations_required": 2,
        "minimum_hold_bars_5m": 12,
        "cooldown_bars_5m": 12,
    }
    enter = decide_strategy(strategy, row(), state("momentum"), 100, COSTS)
    assert enter.signal == "enter_momentum"

    account = state("momentum")
    account.target_position = 1.0
    account.entry_index = 50
    exit_decision = decide_strategy(
        strategy,
        row(one_hour_ema_spread_bps=-20.0, one_hour_momentum_bps=-30.0),
        account,
        100,
        COSTS,
    )
    assert exit_decision.signal == "exit_momentum"


def test_five_minute_timed_momentum_waits_then_recovers():
    strategy = {
        "id": "timed",
        "type": "momentum_1h_5m_entry",
        "entry_1h_ema_spread_bps": 0.0,
        "entry_1h_momentum_bps": 0.0,
        "exit_1h_ema_spread_bps": -10.0,
        "exit_1h_momentum_bps": -10.0,
        "exit_confirmations_required": 2,
        "minimum_pullback_bps": -5.0,
        "maximum_entry_wait_bars_5m": 12,
        "minimum_hold_bars_5m": 12,
        "cooldown_bars_5m": 12,
    }
    account = state("timed")
    wait = decide_strategy(
        strategy,
        row(recovery_cross_up=False),
        account,
        100,
        COSTS,
    )
    assert wait.signal == "wait_5m_entry"
    assert account.pending_entry_index == 100

    enter = decide_strategy(strategy, row(recovery_cross_up=True), account, 101, COSTS)
    assert enter.target_position == 1.0
    assert enter.signal == "enter_5m_recovery"
    assert account.pending_entry_index is None


def test_execution_applies_all_three_cost_components():
    account = state("buy")
    decision = Decision("buy", 1.0, "enter", "test")
    snapshot, transaction = execute_decision(
        account,
        decision,
        row(),
        next_row(100.0),
        10,
        COSTS,
    )
    assert transaction is not None
    assert transaction.fee > 0.0
    assert transaction.spread_cost > 0.0
    assert transaction.slippage_cost > 0.0
    assert snapshot["gross_equity"] > snapshot["equity"]


def test_trade_episode_metrics_pair_entry_and_exit():
    snapshots = pd.DataFrame(
        [
            {
                "strategy_id": "x",
                "execution_timestamp": "2026-01-01T00:05:00+00:00",
                "target_position": 1.0,
                "mark_price": 100.0,
                "equity": 499.0,
            },
            {
                "strategy_id": "x",
                "execution_timestamp": "2026-01-01T00:10:00+00:00",
                "target_position": 1.0,
                "mark_price": 105.0,
                "equity": 524.0,
            },
            {
                "strategy_id": "x",
                "execution_timestamp": "2026-01-01T00:15:00+00:00",
                "target_position": 0.0,
                "mark_price": 103.0,
                "equity": 513.0,
            },
        ]
    )
    episodes = build_trade_episodes(snapshots, initial_cash=500.0)
    assert len(episodes) == 1
    episode = episodes.iloc[0]
    assert not bool(episode["is_open"])
    assert episode["holding_bars"] == 2
    assert episode["gross_price_return_pct"] == pytest.approx(0.03)
    assert episode["maximum_favorable_excursion_pct"] == pytest.approx(0.05)


def test_small_end_to_end_simulation_contains_all_strategies():
    cfg = matrix_config()
    cfg = copy.deepcopy(cfg)
    cfg["research"]["cost_sensitivity_all_in_bps_per_side"] = [8]
    features = build_feature_frame(synthetic_ohlcv(1400), cfg)
    result = simulate_matrix(features, cfg)
    assert set(result.summary) == {item["id"] for item in cfg["strategies"]}
    assert "buy_hold_5m" in result.summary
    assert result.summary["buy_hold_5m"]["transaction_count"] == 1
