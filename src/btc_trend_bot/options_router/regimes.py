"""Deterministic CPU-only regime classifier (spec sections 3, 11) plus a
free-data regime-persistence check (spec section 9's non-options-path
question: does the first-hour regime label still hold at 15/30/60/90
minutes?).

All thresholds come from cfg["regimes"] -- never hardcoded here -- so
walk-forward threshold sweeps don't require code changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FIRST_HOUR_BARS, _efficiency_ratio, _linreg_slope_r2, _vwap_series

REGIME_LABELS = [
    "BULL_TREND", "BEAR_TREND", "VOLATILITY_EXPANSION",
    "MODERATE_BULL", "MODERATE_BEAR", "CHOP_RANGE", "NO_TRADE",
]


def classify_regime(features: dict, cfg: dict) -> tuple[str, float, dict]:
    """features: a row (dict-like) from build_first_hour_features, or the
    lighter dict produced by _window_metrics for a persistence checkpoint.

    Returns (label, confidence in [0,1], diagnostics dict)."""
    rc = cfg["regimes"]
    required = ["first_hour_return", "close_location_value", "efficiency_ratio",
                "trend_slope", "trend_r2"]
    missing = [k for k in required if k not in features or _is_nan(features[k])]
    if missing:
        return "NO_TRADE", 0.0, {"reason": f"missing/NaN features: {missing}"}

    ret = features["first_hour_return"]
    clv = features["close_location_value"]
    er = features["efficiency_ratio"]
    slope = features["trend_slope"]
    r2 = features["trend_r2"]
    range_vs_avg = features.get("first_hour_range_vs_trailing_avg", float("nan"))
    vix_change = features.get("vix_first_hour_change", float("nan"))

    strong_bull = (
        ret > rc["bull_return_threshold"]
        and clv > rc["clv_bull_threshold"]
        and er > rc["efficiency_ratio_trend_threshold"]
        and slope > 0
        and r2 > rc["trend_r2_threshold"]
    )
    strong_bear = (
        ret < rc["bear_return_threshold"]
        and clv < rc["clv_bear_threshold"]
        and er > rc["efficiency_ratio_trend_threshold"]
        and slope < 0
        and r2 > rc["trend_r2_threshold"]
    )
    expansion_signal = (not _is_nan(range_vs_avg) and range_vs_avg > rc["expansion_range_multiple"]) or (
        not _is_nan(vix_change) and vix_change > rc["expansion_vix_change_threshold"]
    )
    direction_uncertain = er < rc["efficiency_ratio_trend_threshold"]
    expansion = expansion_signal and direction_uncertain and not strong_bull and not strong_bear

    moderate_bull = (not strong_bull and not expansion and ret > rc["moderate_return_threshold"])
    moderate_bear = (not strong_bear and not expansion and ret < -rc["moderate_return_threshold"])

    chop = er < rc["efficiency_ratio_chop_threshold"] or (
        features.get("vwap_crossings", 0) >= rc["vwap_crossing_chop_threshold"]
    )

    diagnostics = {
        "first_hour_return": ret, "close_location_value": clv, "efficiency_ratio": er,
        "trend_slope": slope, "trend_r2": r2, "range_vs_avg": range_vs_avg, "vix_change": vix_change,
    }

    if strong_bull:
        label, conf = "BULL_TREND", _trend_confidence(er, r2, rc)
    elif strong_bear:
        label, conf = "BEAR_TREND", _trend_confidence(er, r2, rc)
    elif expansion:
        label, conf = "VOLATILITY_EXPANSION", min(1.0, (range_vs_avg / rc["expansion_range_multiple"]) if not _is_nan(range_vs_avg) else 0.5)
    elif moderate_bull:
        label, conf = "MODERATE_BULL", _moderate_confidence(ret, rc)
    elif moderate_bear:
        label, conf = "MODERATE_BEAR", _moderate_confidence(-ret, rc)
    elif chop:
        label, conf = "CHOP_RANGE", 1.0 - min(1.0, er / rc["efficiency_ratio_chop_threshold"]) if rc["efficiency_ratio_chop_threshold"] > 0 else 0.5
    else:
        label, conf = "NO_TRADE", 0.2

    if conf < rc["min_confidence"] and label not in ("CHOP_RANGE", "NO_TRADE"):
        diagnostics["reason"] = "confidence below min_confidence"
        return "NO_TRADE", conf, diagnostics

    return label, float(conf), diagnostics


def _trend_confidence(er: float, r2: float, rc: dict) -> float:
    er_margin = (er - rc["efficiency_ratio_trend_threshold"]) / max(1e-9, 1.0 - rc["efficiency_ratio_trend_threshold"])
    r2_margin = (r2 - rc["trend_r2_threshold"]) / max(1e-9, 1.0 - rc["trend_r2_threshold"])
    return float(np.clip((er_margin + r2_margin) / 2.0, 0.0, 1.0))


def _moderate_confidence(signed_ret: float, rc: dict) -> float:
    """Scales from 0.5 at the moderate threshold up to 1.0 at the strong
    (BULL/BEAR_TREND) threshold, instead of a flat 0.5 for every moderate
    day regardless of how close it is to a real trend. A flat 0.5 combined
    with a credit spread's max_loss > max_profit made the expectancy proxy
    reject every moderate-regime day by construction."""
    span = rc["bull_return_threshold"] - rc["moderate_return_threshold"]
    margin = (signed_ret - rc["moderate_return_threshold"]) / max(1e-9, span)
    return float(np.clip(0.5 + 0.5 * margin, 0.5, 1.0))


def _is_nan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


def _window_metrics(day_bars: pd.DataFrame, end_bar_of_day: int, window_bars: int) -> dict:
    """Trend/chop metrics over the trailing `window_bars` bars ending at
    `end_bar_of_day`, with VWAP anchored at the session open (bar 0) through
    `end_bar_of_day`, matching how a real VWAP is computed intraday."""
    day_bars = day_bars.sort_values("bar_of_day")
    window = day_bars[(day_bars["bar_of_day"] > end_bar_of_day - window_bars) & (day_bars["bar_of_day"] <= end_bar_of_day)]
    if len(window) < window_bars:
        return {}
    closes = window["close"].to_numpy()
    highs = window["high"].to_numpy()
    lows = window["low"].to_numpy()
    fh_range = float(highs.max() - lows.min())
    clv = float((closes[-1] - lows.min()) / fh_range) if fh_range > 0 else 0.5
    slope, r2 = _linreg_slope_r2(closes)
    metrics = {
        "first_hour_return": float(closes[-1] / window["open"].iloc[0] - 1.0),
        "close_location_value": clv,
        "efficiency_ratio": _efficiency_ratio(closes),
        "trend_slope": slope,
        "trend_r2": r2,
    }
    if "volume" in day_bars.columns:
        session = day_bars[day_bars["bar_of_day"] <= end_bar_of_day]
        if session["volume"].sum() > 0:
            vwap = _vwap_series(session)
            vwap_last = float(vwap.iloc[-1])
            if not np.isnan(vwap_last) and vwap_last:
                metrics["distance_from_vwap"] = float(closes[-1] / vwap_last - 1.0)
    return metrics


def regime_persistence(spx_day_bars: pd.DataFrame, base_label: str, cfg: dict,
                        checkpoints: tuple[int, ...] = (15, 30, 60, 90)) -> dict:
    """Re-classify a trailing FIRST_HOUR_BARS-length window ending at
    10:30 + each checkpoint (in minutes) and report whether the label still
    matches the 10:30 label. Uses only the free, already-dense 5-minute SPX
    path -- no options data needed."""
    out: dict = {}
    for m in checkpoints:
        bars_ahead = m // 5
        end_idx = (FIRST_HOUR_BARS - 1) + bars_ahead
        metrics = _window_metrics(spx_day_bars, end_idx, FIRST_HOUR_BARS)
        if not metrics:
            out[m] = {"available": False}
            continue
        label, conf, _ = classify_regime(metrics, cfg)
        out[m] = {"available": True, "label": label, "confidence": conf, "matches_base": label == base_label}
    return out
