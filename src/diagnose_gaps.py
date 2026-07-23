from pathlib import Path

import pandas as pd

path = Path("data/btc_usd_4h.csv")
df = pd.read_csv(path)

timestamps = (
    pd.to_datetime(df["timestamp"], utc=True)
    .sort_values()
    .drop_duplicates()
)

expected = pd.date_range(
    start=timestamps.iloc[0],
    end=timestamps.iloc[-1],
    freq="4h",
    tz="UTC",
)

missing = expected.difference(pd.DatetimeIndex(timestamps))

print(f"Observed bars: {len(timestamps):,}")
print(f"Expected bars: {len(expected):,}")
print(f"Missing bars:  {len(missing):,}")
print(f"Missing rate:  {len(missing) / len(expected):.2%}")
print(f"First bar:     {timestamps.iloc[0]}")
print(f"Last bar:      {timestamps.iloc[-1]}")
print()

if len(missing) == 0:
    print("No missing intervals.")
    raise SystemExit(0)

missing_series = pd.Series(missing, name="timestamp")

# Consecutive missing timestamps belong to the same gap.
gap_group = (
    missing_series.diff().ne(pd.Timedelta(hours=4))
).cumsum()

runs = (
    missing_series.groupby(gap_group)
    .agg(start="min", end="max", missing_bars="size")
    .reset_index(drop=True)
)

runs["missing_hours"] = runs["missing_bars"] * 4
runs["missing_days"] = runs["missing_hours"] / 24

print("Largest gap runs:")
print(
    runs.sort_values("missing_bars", ascending=False)
    .head(25)
    .to_string(index=False)
)
print()

by_year = missing_series.groupby(missing_series.dt.year).size()
print("Missing bars by year:")
print(by_year.to_string())
print()

by_month = missing_series.groupby(
    missing_series.dt.to_period("M").astype(str)
).size()

print("Worst months:")
print(by_month.sort_values(ascending=False).head(25).to_string())