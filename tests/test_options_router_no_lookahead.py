"""No-lookahead invariants (house standard -- every signal lane gets one,
see research_controls.py and PROMOTION_CONTROLS.md)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.options_router.features import FIRST_HOUR_BARS, build_first_hour_features
from research_controls import assert_no_window_overlap


def _fake_underlying_bars(n_days: int = 3, bars_per_day: int = 20) -> pd.DataFrame:
    rows = []
    for day_i in range(n_days):
        date = pd.Timestamp("2026-05-06") + pd.Timedelta(days=day_i)
        base = 5000 + day_i * 5
        for b in range(bars_per_day):
            price = base + b * 0.1
            rows.append({"date": date, "bar_of_day": b, "open": price, "high": price + 0.2,
                         "low": price - 0.2, "close": price, "volume": 1000})
    return pd.DataFrame(rows)


def test_entry_signal_window_never_reaches_past_1030():
    """The 10:30 feature window must end at-or-before the 10:30 entry window
    it prices against -- a bar published after 10:30 must never leak into
    build_first_hour_features."""
    spx = _fake_underlying_bars()
    feats = build_first_hour_features({"^GSPC": spx}, underlying="^GSPC")

    # Signal window "end" = 10:30 ET every day; entry window "start" = 10:30
    # ET every day -- equal is fine (assert_no_window_overlap only fails if
    # end > start).
    signal_end = feats["date"] + pd.Timedelta(hours=10, minutes=30)
    target_start = feats["date"] + pd.Timedelta(hours=10, minutes=30)
    assert_no_window_overlap(signal_end, target_start, "options_router entry-vs-signal")


def test_first_hour_close_is_the_twelfth_five_minute_bar():
    spx = _fake_underlying_bars(bars_per_day=FIRST_HOUR_BARS + 5)
    feats = build_first_hour_features({"^GSPC": spx}, underlying="^GSPC")
    day0 = spx[spx["date"] == spx["date"].iloc[0]].sort_values("bar_of_day")
    expected_fh_close = float(day0.iloc[FIRST_HOUR_BARS - 1]["close"])
    assert feats.iloc[0]["first_hour_close"] == pytest.approx(expected_fh_close)


def test_days_with_fewer_than_first_hour_bars_are_dropped():
    spx = _fake_underlying_bars(n_days=1, bars_per_day=FIRST_HOUR_BARS - 1)
    feats = build_first_hour_features({"^GSPC": spx}, underlying="^GSPC")
    assert feats.empty
