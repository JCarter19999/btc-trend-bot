"""Post-earnings-announcement drift (PEAD) test using real Alpha Vantage
earnings-surprise data (already-configured API key at
/home/joey/.config/btc-trend-bot/alphavantage.env, previously used for a
different purpose in this project) -- this resolves the data-access gap
flagged in MECHANISM_CARDS.md, where yfinance's earnings_dates endpoint
(~25 quarters, 2020+) was too short to test PEAD over the full 2018-2026
window this project's equity research uses.

No lookahead: entry is always the trading day AFTER the report date
(regardless of reportTime pre/post-market -- conservative, matching this
project's signal-at-t/entry-at-t+1 convention used everywhere else).
Forward returns measured from that entry close/open over N=5/10/20 trading
days. Real surprise% from Alpha Vantage's EARNINGS endpoint, not restated
or estimated after the fact -- reportedEPS/estimatedEPS/surprisePercentage
are exactly what Alpha Vantage's API returns for each historical quarter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA"]
HORIZONS = (5, 10, 20)
N_NULL_SEEDS = 1000
COST_BPS_ROUND_TRIP = 2.0


def load_earnings(symbol: str) -> pd.DataFrame:
    path = Path(f"/tmp/{symbol}_earnings.json")
    d = json.loads(path.read_text())
    q = pd.DataFrame(d["quarterlyEarnings"])
    q["reportedDate"] = pd.to_datetime(q["reportedDate"])
    q["surprisePercentage"] = pd.to_numeric(q["surprisePercentage"], errors="coerce")
    q["reportedEPS"] = pd.to_numeric(q["reportedEPS"], errors="coerce")
    q["estimatedEPS"] = pd.to_numeric(q["estimatedEPS"], errors="coerce")
    q = q.dropna(subset=["surprisePercentage"])
    q["symbol"] = symbol
    return q[["symbol", "reportedDate", "reportTime", "estimatedEPS", "reportedEPS", "surprisePercentage"]]


def load_prices(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "real" / f"{symbol}.csv", parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)


def shuffled_null_percentile(x: np.ndarray, y: np.ndarray, real_corr: float, seeds: int = N_NULL_SEEDS):
    null_corrs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(y)
        null_corrs.append(np.corrcoef(x, shuffled)[0, 1])
    null_corrs = np.array(null_corrs)
    pct = float((null_corrs < real_corr).mean() * 100)
    return float(null_corrs.mean()), float(null_corrs.std()), pct


def main() -> None:
    all_events = []
    for symbol in SYMBOLS:
        earnings = load_earnings(symbol)
        prices = load_prices(symbol)
        prices = prices.set_index("date")

        for _, row in earnings.iterrows():
            report_date = row["reportedDate"]
            # No lookahead: entry always the first trading day strictly AFTER the report date.
            future_dates = prices.index[prices.index > report_date]
            if len(future_dates) < max(HORIZONS) + 1:
                continue
            entry_date = future_dates[0]
            entry_idx = prices.index.get_loc(entry_date)
            entry_price = prices["close"].iloc[entry_idx]

            event = {
                "symbol": symbol,
                "report_date": report_date,
                "entry_date": entry_date,
                "surprise_pct": row["surprisePercentage"],
            }
            for h in HORIZONS:
                if entry_idx + h < len(prices):
                    exit_price = prices["close"].iloc[entry_idx + h]
                    event[f"fwd_ret_{h}d"] = exit_price / entry_price - 1.0
            all_events.append(event)

    events = pd.DataFrame(all_events)
    events = events[(events["report_date"] >= "2018-01-01") & (events["report_date"] <= "2026-07-01")]
    print(f"Total earnings events in 2018-2026 window, all 4 symbols: {len(events)}")
    print(f"Surprise% distribution: mean={events['surprise_pct'].mean():.2f}, "
          f"median={events['surprise_pct'].median():.2f}, std={events['surprise_pct'].std():.2f}")

    results = {}
    for h in HORIZONS:
        col = f"fwd_ret_{h}d"
        valid = events[["surprise_pct", col]].dropna()
        if len(valid) < 10:
            continue
        corr = valid["surprise_pct"].corr(valid[col])
        null_mean, null_std, pct = shuffled_null_percentile(
            valid["surprise_pct"].to_numpy(), valid[col].to_numpy(), corr)

        # Directional check: does top-tercile surprise outperform bottom-tercile?
        valid = valid.copy()
        valid["tercile"] = pd.qcut(valid["surprise_pct"], 3, labels=["bottom", "mid", "top"], duplicates="drop")
        top_mean = valid[valid["tercile"] == "top"][col].mean()
        bottom_mean = valid[valid["tercile"] == "bottom"][col].mean()

        print(f"\n=== {h}-day forward return ===")
        print(f"n={len(valid)}  pearson_corr(surprise%, fwd_ret)={corr:+.4f}  "
              f"null_mean={null_mean:+.4f} null_std={null_std:.4f} percentile={pct:.0f}")
        print(f"Top-tercile surprise mean fwd return: {top_mean*100:+.2f}%  "
              f"Bottom-tercile: {bottom_mean*100:+.2f}%  Spread: {(top_mean-bottom_mean)*100:+.2f}pp")

        cost = COST_BPS_ROUND_TRIP / 10000.0
        print(f"Spread net of {COST_BPS_ROUND_TRIP}bp round-trip cost: {(top_mean-bottom_mean-cost)*100:+.2f}pp")

        # OOS split
        valid_dated = events[["report_date", "surprise_pct", col]].dropna()
        mid = valid_dated["report_date"].median()
        first = valid_dated[valid_dated["report_date"] <= mid]
        second = valid_dated[valid_dated["report_date"] > mid]
        corr_first = first["surprise_pct"].corr(first[col])
        corr_second = second["surprise_pct"].corr(second[col])
        print(f"OOS split: first half (n={len(first)}) corr={corr_first:+.4f}  "
              f"second half (n={len(second)}) corr={corr_second:+.4f}")

        results[h] = {
            "n": len(valid), "pearson_corr": corr, "null_mean": null_mean, "null_std": null_std,
            "null_percentile": pct, "top_tercile_mean": top_mean, "bottom_tercile_mean": bottom_mean,
            "spread_pp": (top_mean - bottom_mean) * 100, "spread_net_cost_pp": (top_mean - bottom_mean - cost) * 100,
            "oos_first_half_corr": corr_first, "oos_second_half_corr": corr_second,
        }

    out_dir = ROOT / "outputs" / "pead_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "earnings_events.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_dir / 'earnings_events.csv'} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
