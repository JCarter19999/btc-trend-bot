"""Does the European session (which trades ~6.5 hours before the US opens)
carry real, no-lookahead-usable information about the US market's first
hour? Follow-up to the volatility-breakout straddle thread -- same theme
(can we detect an unusually large move coming before it happens), applied
to a genuinely different information source: cross-market lead-lag instead
of within-market vol compression.

Data: yfinance hourly (60m) bars, ^GDAXI (DAX) and ^FTSE (FTSE 100) as
European proxies, SPY as the US target. 60m is a deliberate choice over
finer granularity -- 5m/15m bars cap at 60 days of history via yfinance (a
real API limit, not a preference), which isn't enough sample for a
credible test; 60m bars go back ~2 years (~700+ trading days) AND their
first daily bar happens to align exactly with each market's session open
(DAX/FTSE 07:00 UTC, SPY 13:30 UTC = 9:30am ET), so the first US bar IS
the first hour without needing finer resampling.

Signal (no lookahead): European cumulative return from session open through
the last European bar that closes before 13:30 UTC (US open) -- everything
used is fully known before a US trader could act on it.
Target: SPY's first-hour bar return (13:30-14:30 UTC), tested both as
direction (does EU up-morning predict US up first hour?) and magnitude
(does a big EU morning move predict a big US first-hour move -- the actual
vol-breakout-shaped question).

Includes a random-shuffle control: real correlation vs. the same
correlation computed on randomly reshuffled date-pairings, many times --
guards against both markets simply sharing a volatility regime over the
window (e.g. both calmer in some months, both choppier in others) creating
a spurious correlation that has nothing to do with day-to-day lead-lag.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "intraday"
US_OPEN_UTC = pd.Timestamp("13:30:00").time()
N_SHUFFLE = 1000


def download_hourly(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    path = DATA_DIR / f"{symbol.replace('^', '')}_60m.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    df = yf.download(symbol, interval="60m", period="730d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def european_pre_us_open_return(eu_df: pd.DataFrame) -> pd.Series:
    """Per calendar date: cumulative return from that day's first EU bar's
    open through the close of the last EU bar strictly before 13:30 UTC."""
    eu_df = eu_df.copy()
    eu_df["date"] = eu_df.index.date
    eu_df["time"] = eu_df.index.time
    pre_open = eu_df[eu_df["time"] < US_OPEN_UTC]
    out = {}
    for date, day in pre_open.groupby("date"):
        day = day.sort_index()
        if len(day) < 2:
            continue
        out[date] = float(day["close"].iloc[-1] / day["open"].iloc[0] - 1)
    return pd.Series(out, name="eu_pre_open_return")


def us_first_hour_return(us_df: pd.DataFrame) -> pd.Series:
    us_df = us_df.copy()
    us_df["date"] = us_df.index.date
    us_df["time"] = us_df.index.time
    first_bars = us_df[us_df["time"] == US_OPEN_UTC]
    return pd.Series(
        {row.date: float(row.close / row.open - 1) for row in first_bars.itertuples()},
        name="us_first_hour_return",
    )


def analyze(eu_signal: pd.Series, us_target: pd.Series, eu_label: str) -> dict:
    joined = pd.concat([eu_signal, us_target], axis=1).dropna()
    if len(joined) < 30:
        return {"eu_label": eu_label, "n": len(joined), "note": "too few overlapping days"}

    eu, us = joined["eu_pre_open_return"], joined["us_first_hour_return"]
    dir_corr = float(np.corrcoef(eu, us)[0, 1])
    mag_corr = float(np.corrcoef(eu.abs(), us.abs())[0, 1])

    rng = np.random.default_rng(0)
    shuffled_dir = np.array([np.corrcoef(eu.to_numpy(), rng.permutation(us.to_numpy()))[0, 1] for _ in range(N_SHUFFLE)])
    shuffled_mag = np.array([np.corrcoef(eu.abs().to_numpy(), rng.permutation(us.abs().to_numpy()))[0, 1] for _ in range(N_SHUFFLE)])
    dir_percentile = float((shuffled_dir < dir_corr).mean() * 100)
    mag_percentile = float((shuffled_mag < mag_corr).mean() * 100)

    # Quantile check for the vol-breakout framing: split by EU move magnitude
    # quartile, look at mean US first-hour move magnitude per quartile.
    quartiles = pd.qcut(eu.abs(), 4, labels=["Q1_smallest_EU_move", "Q2", "Q3", "Q4_biggest_EU_move"])
    by_quartile = us.abs().groupby(quartiles, observed=True).mean().to_dict()

    return {
        "eu_label": eu_label, "n_days": len(joined),
        "date_range": f"{min(joined.index)} to {max(joined.index)}",
        "directional_correlation": dir_corr,
        "directional_corr_percentile_vs_shuffled": dir_percentile,
        "magnitude_correlation": mag_corr,
        "magnitude_corr_percentile_vs_shuffled": mag_percentile,
        "mean_us_first_hour_abs_move_by_eu_move_quartile": {k: float(v) for k, v in by_quartile.items()},
        "mean_us_first_hour_abs_move_overall": float(us.abs().mean()),
    }


def main() -> None:
    print("Downloading/loading hourly bars (DAX, FTSE, SPY)...")
    dax = download_hourly("^GDAXI")
    ftse = download_hourly("^FTSE")
    spy = download_hourly("SPY")

    us_target = us_first_hour_return(spy)
    print(f"US first-hour observations: {len(us_target)}")

    results = []
    for eu_df, label in [(dax, "DAX"), (ftse, "FTSE_100")]:
        eu_signal = european_pre_us_open_return(eu_df)
        print(f"{label} pre-US-open observations: {len(eu_signal)}")
        results.append(analyze(eu_signal, us_target, label))

    # Combined: average of DAX + FTSE pre-open returns, as a broader "Europe" signal
    dax_sig = european_pre_us_open_return(dax).rename("dax")
    ftse_sig = european_pre_us_open_return(ftse).rename("ftse")
    combined = pd.concat([dax_sig, ftse_sig], axis=1).dropna()
    combined_signal = combined.mean(axis=1).rename("eu_pre_open_return")
    results.append(analyze(combined_signal, us_target, "DAX+FTSE_average"))

    for r in results:
        print("\n" + "=" * 70)
        for k, v in r.items():
            print(f"{k}: {v}")

    out_dir = ROOT / "outputs" / "european_lead_us_first_hour_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
