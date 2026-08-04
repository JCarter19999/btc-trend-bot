"""Regime classifier unit tests (spec sections 3, 11) -- plain synthetic
feature dicts, no fixtures, matching house style (see test_intraday_matrix.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.options_router.regimes import classify_regime, regime_persistence

REGIME_CFG = {
    "regimes": {
        "bull_return_threshold": 0.0015,
        "bear_return_threshold": -0.0015,
        "moderate_return_threshold": 0.0005,
        "clv_bull_threshold": 0.70,
        "clv_bear_threshold": 0.30,
        "efficiency_ratio_trend_threshold": 0.55,
        "efficiency_ratio_chop_threshold": 0.35,
        "trend_r2_threshold": 0.35,
        "expansion_range_multiple": 1.5,
        "expansion_vix_change_threshold": 0.03,
        "vwap_crossing_chop_threshold": 3,
        "min_confidence": 0.4,
    }
}


def base_row(**overrides) -> dict:
    row = {
        "first_hour_return": 0.0,
        "close_location_value": 0.5,
        "efficiency_ratio": 0.5,
        "trend_slope": 0.0,
        "trend_r2": 0.5,
        "first_hour_range_vs_trailing_avg": 1.0,
        "vix_first_hour_change": 0.0,
        "vwap_crossings": 1,
    }
    row.update(overrides)
    return row


def test_strong_bull_trend():
    row = base_row(first_hour_return=0.004, close_location_value=0.85, efficiency_ratio=0.8,
                    trend_slope=1.0, trend_r2=0.6)
    label, conf, _ = classify_regime(row, REGIME_CFG)
    assert label == "BULL_TREND"
    assert conf > 0


def test_strong_bear_trend():
    row = base_row(first_hour_return=-0.004, close_location_value=0.15, efficiency_ratio=0.8,
                    trend_slope=-1.0, trend_r2=0.6)
    label, conf, _ = classify_regime(row, REGIME_CFG)
    assert label == "BEAR_TREND"


def test_volatility_expansion_when_direction_uncertain_and_range_wide():
    row = base_row(first_hour_return=0.0002, close_location_value=0.5, efficiency_ratio=0.2,
                    trend_slope=0.0, trend_r2=0.1, first_hour_range_vs_trailing_avg=2.0)
    label, conf, _ = classify_regime(row, REGIME_CFG)
    assert label == "VOLATILITY_EXPANSION"


def test_moderate_bull_when_mild_positive_return_and_not_strong():
    row = base_row(first_hour_return=0.0008, close_location_value=0.6, efficiency_ratio=0.45,
                    trend_slope=0.2, trend_r2=0.2)
    label, conf, _ = classify_regime(row, REGIME_CFG)
    assert label == "MODERATE_BULL"


def test_moderate_bear_when_mild_negative_return_and_not_strong():
    row = base_row(first_hour_return=-0.0008, close_location_value=0.4, efficiency_ratio=0.45,
                    trend_slope=-0.2, trend_r2=0.2)
    label, conf, _ = classify_regime(row, REGIME_CFG)
    assert label == "MODERATE_BEAR"


def test_chop_range_on_low_efficiency_ratio():
    row = base_row(first_hour_return=0.0001, close_location_value=0.5, efficiency_ratio=0.1,
                    trend_slope=0.0, trend_r2=0.05, first_hour_range_vs_trailing_avg=0.9)
    label, conf, _ = classify_regime(row, REGIME_CFG)
    assert label == "CHOP_RANGE"


def test_no_trade_on_missing_features():
    row = {"first_hour_return": float("nan"), "close_location_value": 0.5,
           "efficiency_ratio": 0.5, "trend_slope": 0.0, "trend_r2": 0.5}
    label, conf, diag = classify_regime(row, REGIME_CFG)
    assert label == "NO_TRADE"
    assert conf == 0.0
    assert "missing" in diag["reason"]


def _synthetic_day_bars(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "bar_of_day": np.arange(n),
        "open": [closes[0]] + closes[:-1],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": np.zeros(n),
    })


def test_persistence_detects_reversal_at_90_minutes():
    # 12 bars (first hour) trending up cleanly, then a sharp reversal for
    # the remaining bars through +90 minutes (18 more 5-minute bars).
    up = list(np.linspace(5000, 5020, 12))
    down = list(np.linspace(5020, 4950, 18))
    bars = _synthetic_day_bars(up + down)
    base_row_dict = base_row(first_hour_return=0.004, close_location_value=0.9, efficiency_ratio=0.9,
                              trend_slope=1.0, trend_r2=0.9)
    base_label, _, _ = classify_regime(base_row_dict, REGIME_CFG)
    assert base_label == "BULL_TREND"

    persistence = regime_persistence(bars, base_label, REGIME_CFG, checkpoints=(15, 30, 60, 90))
    assert persistence[90]["available"]
    assert persistence[90]["matches_base"] is False


def test_persistence_holds_on_clean_continuation():
    up = list(np.linspace(5000, 5060, 30))
    bars = _synthetic_day_bars(up)
    base_row_dict = base_row(first_hour_return=0.004, close_location_value=0.9, efficiency_ratio=0.9,
                              trend_slope=1.0, trend_r2=0.9)
    base_label, _, _ = classify_regime(base_row_dict, REGIME_CFG)
    persistence = regime_persistence(bars, base_label, REGIME_CFG, checkpoints=(15, 30, 60, 90))
    assert persistence[90]["available"]
    assert persistence[90]["matches_base"] is True
