import pandas as pd

from btc_trend_bot.gaps import build_gap_report


def test_gap_report_groups_consecutive_missing_bars():
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01T00:00:00Z",
                "2024-01-01T04:00:00Z",
                "2024-01-01T16:00:00Z",
                "2024-01-01T20:00:00Z",
                "2024-01-02T04:00:00Z",
            ],
            utc=True,
        )
    )
    report = build_gap_report(timestamps, "4h")
    assert report.missing_bars == 3
    assert report.largest_gap_bars == 2
    assert list(report.runs["missing_bars"]) == [2, 1]
