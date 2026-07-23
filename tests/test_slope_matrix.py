from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from btc_trend_bot.slope_matrix import (
    SlopeStrategyState,
    _complete_signal_candles,
    _feature_prefix,
    build_slope_feature_frame,
    decide_slope_strategy,
    load_slope_config,
    simulate_slope_matrix,
    trailing_local_polynomial,
)

COSTS = {
    "fee_bps_per_side": 2.0,
    "slippage_bps_per_side": 5.0,
    "assumed_spread_bps_per_side": 1.0,
    "min_notional": 1.0,
    "rebalance_tolerance_bps": 0.0,
}


def synthetic_ohlcv(rows: int = 2500) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    x = np.arange(rows, dtype=float)
    close = 100.0 * np.exp(0.00008 * x + 0.003 * np.sin(x / 25.0))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = 10.0 + (x % 20)
    return pd.DataFrame({"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def state(strategy_id: str) -> SlopeStrategyState:
    return SlopeStrategyState(strategy_id=strategy_id, initial_cash=500.0, cash=500.0, btc=0.0, gross_cash=500.0, gross_btc=0.0, peak_equity=500.0)


def slope_row(strategy_id: str, **overrides) -> pd.Series:
    prefix = _feature_prefix(strategy_id)
    values = {
        "timestamp": pd.Timestamp("2026-01-01T01:00:00Z"),
        "bar_end": pd.Timestamp("2026-01-01T01:05:00Z"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
        f"{prefix}bar_end": pd.Timestamp("2026-01-01T01:00:00Z"),
        f"{prefix}signal_update": True,
        f"{prefix}signal_close": 100.5,
        f"{prefix}slope_bps_per_bar": 3.0,
        f"{prefix}slope_score": 0.25,
        f"{prefix}slope_tstat": 2.0,
        f"{prefix}fit_r2": 0.6,
        f"{prefix}acceleration_score": 0.1,
        "feature_valid": True,
    }
    values.update(overrides)
    return pd.Series(values)


def strategy(strategy_id: str = "slope") -> dict:
    return {
        "id": strategy_id,
        "type": "rolling_slope",
        "signal_timeframe": "15m",
        "lookback_bars": 16,
        "polynomial_degree": 1,
        "entry_slope_score": 0.15,
        "exit_slope_score": -0.08,
        "minimum_fit_r2": 0.15,
        "minimum_exit_fit_r2": 0.05,
        "minimum_entry_tstat": 0.75,
        "entry_confirmation_ticks": 2,
        "exit_confirmation_ticks": 2,
        "minimum_hold_signal_bars": 1,
        "maximum_hold_signal_bars": 48,
        "cooldown_signal_bars": 0,
        "require_cost_hurdle": True,
        "cost_hurdle_expected_hold_signal_bars": 8,
        "cost_hurdle_multiple": 1.25,
        "emergency_stop_pct": 0.03,
    }


def test_linear_local_polynomial_recovers_exact_log_slope():
    n = 18
    x = np.arange(80, dtype=float)
    close = pd.Series(np.exp(4.0 + 0.0015 * x))
    result = trailing_local_polynomial(close, n, 1, volatility_window_bars=n)
    assert abs(result["slope_bps_per_bar"].iloc[-1] - 15.0) < 1e-8
    assert result["fit_r2"].iloc[-1] > 0.999999


def test_quadratic_uses_right_endpoint_derivative():
    n = 16
    x = np.arange(60, dtype=float)
    log_close = 4.0 + 0.0002 * x + 0.00001 * x * x
    result = trailing_local_polynomial(pd.Series(np.exp(log_close)), n, 2, volatility_window_bars=n)
    expected = (0.0002 + 2.0 * 0.00001 * (len(x) - 1)) * 10_000.0
    assert abs(result["slope_bps_per_bar"].iloc[-1] - expected) < 1e-7
    assert result["acceleration_score"].iloc[-1] > 0.0


def test_resample_emits_only_complete_15m_candle():
    frame = synthetic_ohlcv(3)
    assert _complete_signal_candles(frame.iloc[:2], "5m", "15m").empty
    full = _complete_signal_candles(frame, "5m", "15m")
    assert len(full) == 1
    assert full.iloc[0]["bar_end"] == pd.Timestamp("2026-01-01T00:15:00Z")


def test_feature_updates_only_on_new_signal_candle():
    cfg = load_slope_config("config/settings_slope_matrix.yaml")
    cfg = copy.deepcopy(cfg)
    chosen = next(s for s in cfg["strategies"] if s["id"] == "slope_15m_fast_2tick")
    cfg["strategies"] = [cfg["strategies"][0], cfg["strategies"][1], chosen]
    features = build_slope_feature_frame(synthetic_ohlcv(800), cfg)
    prefix = _feature_prefix("slope_15m_fast_2tick")
    updates = pd.to_datetime(features.loc[features[f"{prefix}signal_update"], f"{prefix}bar_end"], utc=True)
    assert (updates.diff().dropna() == pd.Timedelta(minutes=15)).all()


def test_two_signal_ticks_required_for_entry():
    strat = strategy()
    account = state("slope")
    first = decide_slope_strategy(strat, slope_row("slope"), account, 100, COSTS)
    assert first.target_position == 0.0
    assert account.entry_confirmations == 1
    prefix = _feature_prefix("slope")
    second = slope_row("slope", **{f"{prefix}bar_end": pd.Timestamp("2026-01-01T01:15:00Z")})
    decision = decide_slope_strategy(strat, second, account, 103, COSTS)
    assert decision.target_position == 1.0
    assert decision.signal == "enter_slope_inversion"


def test_confirmation_does_not_increment_between_signal_ticks():
    strat = strategy()
    account = state("slope")
    prefix = _feature_prefix("slope")
    row = slope_row("slope", **{f"{prefix}signal_update": False})
    decision = decide_slope_strategy(strat, row, account, 101, COSTS)
    assert decision.signal == "hold_between_ticks"
    assert account.entry_confirmations == 0


def test_cost_hurdle_blocks_weak_slope():
    strat = strategy()
    account = state("slope")
    prefix = _feature_prefix("slope")
    weak = slope_row("slope", **{f"{prefix}slope_bps_per_bar": 1.0})
    decide_slope_strategy(strat, weak, account, 100, COSTS)
    assert account.entry_confirmations == 0


def test_exit_requires_persistence_and_negative_hysteresis():
    strat = strategy()
    account = state("slope")
    account.target_position = 1.0
    account.entry_index = 50
    account.entry_mark = 100.0
    prefix = _feature_prefix("slope")
    negative = slope_row(
        "slope",
        **{
            f"{prefix}slope_score": -0.20,
            f"{prefix}slope_bps_per_bar": -3.0,
            f"{prefix}slope_tstat": -2.0,
            f"{prefix}acceleration_score": -0.1,
        },
    )
    first = decide_slope_strategy(strat, negative, account, 100, COSTS)
    assert first.target_position == 1.0
    second = decide_slope_strategy(strat, negative, account, 103, COSTS)
    assert second.target_position == 0.0
    assert second.signal == "exit_slope_inversion"


def test_small_end_to_end_simulation_contains_all_strategies():
    cfg = load_slope_config("config/settings_slope_matrix.yaml")
    features = build_slope_feature_frame(synthetic_ohlcv(2500), cfg)
    result = simulate_slope_matrix(features, cfg)
    expected = {str(item["id"]) for item in cfg["strategies"]}
    assert set(result.summary) == expected
    assert not result.snapshots.empty
    assert result.summary["buy_hold_5m"]["transaction_count"] == 1
