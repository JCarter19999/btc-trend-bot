from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from btc_trend_bot.data import timeframe_to_timedelta


@dataclass(frozen=True)
class GapReport:
    observed_bars: int
    expected_bars: int
    missing_bars: int
    missing_rate: float
    largest_gap_bars: int
    runs: pd.DataFrame
    missing_timestamps: pd.DatetimeIndex


def build_gap_report(timestamps: pd.Series, timeframe: str) -> GapReport:
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).sort_values().drop_duplicates()
    if len(ts) == 0:
        raise ValueError("At least one timestamp is required.")
    step = timeframe_to_timedelta(timeframe)
    expected = pd.date_range(start=ts[0], end=ts[-1], freq=step, tz="UTC")
    missing = expected.difference(ts)

    if len(missing):
        missing_series = pd.Series(missing, name="timestamp")
        groups = missing_series.diff().ne(step).cumsum()
        runs = (
            missing_series.groupby(groups)
            .agg(start="min", end="max", missing_bars="size")
            .reset_index(drop=True)
        )
        runs["missing_hours"] = runs["missing_bars"] * step.total_seconds() / 3600.0
        runs["missing_days"] = runs["missing_hours"] / 24.0
        largest = int(runs["missing_bars"].max())
    else:
        runs = pd.DataFrame(
            columns=["start", "end", "missing_bars", "missing_hours", "missing_days"]
        )
        largest = 0

    return GapReport(
        observed_bars=len(ts),
        expected_bars=len(expected),
        missing_bars=len(missing),
        missing_rate=(len(missing) / len(expected)) if len(expected) else 0.0,
        largest_gap_bars=largest,
        runs=runs,
        missing_timestamps=missing,
    )
