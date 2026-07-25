"""Data-provenance check: does ThetaData's real stock EOD/intraday history
agree with the yfinance daily bars every walk-forward/simple_trend/regime-
classifier result in this project has been built on? Read-only -- never
modifies data/real/ or configs/real_data.yaml.

Coverage boundary, confirmed empirically (not assumed) before running any
comparison: the current ThetaData subscription is "Options Value" -- it
does NOT include a paid stock-data tier. Bisected directly against
stock_history_eod:

  - pre-2021ish   : requires STANDARD subscription (blocked)
  - 2021-01 .. 2023-05-31 : requires VALUE subscription (blocked)
  - 2023-06-01 onward     : accessible under the account's FREE stock tier
    (exact boundary day pinned: 2023-05-31 blocked, 2023-06-01 OK)

stock_history_ohlc (intraday), stock_history_trade, stock_history_quote,
stock_snapshot_ohlc are ALL blocked unconditionally regardless of date --
no free-tier carve-out the way EOD has one. Confirmed at three different
dates spanning 2023-2026, always "requires a value/standard subscription".

So: this project's cached 2018-01-01+ history (the window every backtest
here depends on) is 78% outside what this subscription can verify
(2018-01-01 .. 2023-05-31 is unreachable). Only the most recent ~3 years
(2023-06-01 onward) can actually be cross-checked, and intraday equity
data cannot be cross-checked or sourced from ThetaData at all under this
subscription -- it does NOT resolve the "yfinance intraday history is
limited" constraint CLAUDE.md flags, contrary to the working assumption
that going in.

Also note: max 365 days per stock_history_eod call (INVALID_ARGUMENT
above that) -- chunked accordingly, unlike option_history_eod which had no
such limit hit in prior work.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.thetadata_pricing import get_client  # noqa: E402

SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
FREE_TIER_START = date(2023, 6, 1)  # empirically pinned boundary, see module docstring
MAX_DAYS_PER_CALL = 365

# Publicly known stock splits in this universe, 2018-2026 (for adjustment-
# methodology spot checks around split dates within the accessible window).
KNOWN_SPLITS_IN_ACCESSIBLE_WINDOW: dict[str, list[str]] = {
    # NVDA 10:1 split 2024-06-10 is the only split from this universe that
    # falls inside the FREE-tier-accessible 2023-06-01+ window.
    "NVDA": ["2024-06-10"],
}


def probe_subscription_boundary(client) -> dict:
    """Confirm (not assume) the tier boundaries this run depends on, so a
    future re-run under a different subscription doesn't silently reuse a
    stale assumption."""
    probes = [date(2018, 1, 2), date(2020, 1, 2), date(2021, 1, 4), date(2022, 1, 3),
              date(2023, 1, 3), date(2023, 5, 31), date(2023, 6, 1), date(2024, 1, 3)]
    results = {}
    for d in probes:
        try:
            _call_with_retry(client.stock_history_eod, "AAPL", start_date=d, end_date=d)
            results[str(d)] = "OK"
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            tier = "UNKNOWN"
            for tok in ("STANDARD", "VALUE"):
                if tok in msg:
                    tier = tok
                    break
            results[str(d)] = f"BLOCKED (requires {tier})"

    intraday_blocked = {}
    for d in [date(2023, 6, 5), date(2024, 1, 3), date(2026, 7, 1)]:
        try:
            _call_with_retry(client.stock_history_ohlc, "SPY", interval="1h", start_date=d, end_date=d)
            intraday_blocked[str(d)] = "OK"
        except Exception as e:  # noqa: BLE001
            intraday_blocked[str(d)] = "BLOCKED"

    return {"eod_boundary_probes": results, "intraday_probes_all_dates": intraday_blocked}


def _call_with_retry(fn, *args, retries: int = 5, **kwargs):
    """ThetaData enforces a single concurrent session per API key --
    multiple forks in this project hitting the same key at once (confirmed
    directly: this call failed once with UNAUTHENTICATED "Invalid session
    ID... more than one terminal is running", then succeeded seconds later
    with no code change) causes transient session-eviction errors, not a
    real auth failure. Retry with backoff rather than treating it as fatal."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if "UNAUTHENTICATED" in str(e) or "session" in str(e).lower():
                last_exc = e
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise last_exc


def fetch_thetadata_eod(client, symbol: str, start: date, end: date) -> pd.DataFrame:
    """Chunked pull respecting the 365-day-per-call limit (confirmed via
    INVALID_ARGUMENT, not assumed to match option_history_eod's lack of
    this limit)."""
    frames = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=MAX_DAYS_PER_CALL - 1), end)
        df = _call_with_retry(client.stock_history_eod, symbol, start_date=cur, end_date=chunk_end)
        if df is not None and not df.empty:
            frames.append(df)
        cur = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["created"]).dt.tz_localize(None).dt.normalize()
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_yfinance_cached(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "real" / f"{symbol}.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df


def compare_symbol(client, symbol: str) -> dict:
    yf = load_yfinance_cached(symbol)
    yf_window = yf[(yf["date"] >= pd.Timestamp(FREE_TIER_START)) & (yf["date"] <= yf["date"].max())]
    end = yf["date"].max().date()

    td = fetch_thetadata_eod(client, symbol, FREE_TIER_START, end)

    merged = pd.merge(
        yf_window[["date", "close"]].rename(columns={"close": "yf_close"}),
        td[["date", "close"]].rename(columns={"close": "td_close"}) if not td.empty else pd.DataFrame(columns=["date", "td_close"]),
        on="date", how="outer", indicator=True,
    )
    both = merged[merged["_merge"] == "both"].copy()
    only_yf = merged[merged["_merge"] == "left_only"]
    only_td = merged[merged["_merge"] == "right_only"]

    if len(both):
        both["abs_diff"] = (both["yf_close"] - both["td_close"]).abs()
        both["pct_diff"] = both["abs_diff"] / both["yf_close"]
        exact_match_rate = float((both["abs_diff"] < 1e-6).mean())
        mean_pct_diff = float(both["pct_diff"].mean())
        max_pct_diff = float(both["pct_diff"].max())
        max_diff_row = both.loc[both["pct_diff"].idxmax()]
    else:
        exact_match_rate = mean_pct_diff = max_pct_diff = float("nan")
        max_diff_row = None

    split_check = {}
    for split_date_str in KNOWN_SPLITS_IN_ACCESSIBLE_WINDOW.get(symbol, []):
        sd = pd.Timestamp(split_date_str)
        window = both[(both["date"] >= sd - pd.Timedelta(days=5)) & (both["date"] <= sd + pd.Timedelta(days=5))]
        split_check[split_date_str] = window[["date", "yf_close", "td_close", "pct_diff"]].to_dict(orient="records")

    return {
        "symbol": symbol,
        "yf_rows_in_window": int(len(yf_window)),
        "td_rows_fetched": int(len(td)),
        "matched_dates": int(len(both)),
        "dates_only_in_yfinance": int(len(only_yf)),
        "dates_only_in_thetadata": int(len(only_td)),
        "only_in_yfinance_sample": only_yf["date"].dt.strftime("%Y-%m-%d").tolist()[:10],
        "only_in_thetadata_sample": only_td["date"].dt.strftime("%Y-%m-%d").tolist()[:10],
        "close_exact_match_rate": exact_match_rate,
        "close_mean_pct_diff": mean_pct_diff,
        "close_max_pct_diff": max_pct_diff,
        "close_max_diff_date": str(max_diff_row["date"].date()) if max_diff_row is not None else None,
        "split_window_check": split_check,
    }


def main() -> None:
    client = get_client()

    print("=== Step 0: confirm subscription-tier boundary (not assumed) ===")
    boundary = probe_subscription_boundary(client)
    for d, r in boundary["eod_boundary_probes"].items():
        print(f"  EOD {d}: {r}")
    for d, r in boundary["intraday_probes_all_dates"].items():
        print(f"  Intraday-OHLC {d}: {r}")

    print(f"\n=== Step 1-3: EOD close-price comparison, {FREE_TIER_START}+ only "
          f"(older history is subscription-blocked, see above) ===")
    results = []
    for sym in SYMBOLS:
        print(f"\n--- {sym} ---")
        r = compare_symbol(client, sym)
        results.append(r)
        print(f"  yfinance rows in window: {r['yf_rows_in_window']}  "
              f"ThetaData rows fetched: {r['td_rows_fetched']}  matched dates: {r['matched_dates']}")
        print(f"  dates only in yfinance: {r['dates_only_in_yfinance']} (sample: {r['only_in_yfinance_sample']})")
        print(f"  dates only in ThetaData: {r['dates_only_in_thetadata']} (sample: {r['only_in_thetadata_sample']})")
        print(f"  close exact-match rate: {r['close_exact_match_rate']*100:.1f}%  "
              f"mean pct diff: {r['close_mean_pct_diff']*100:.4f}%  "
              f"max pct diff: {r['close_max_pct_diff']*100:.4f}% (on {r['close_max_diff_date']})")
        if r["split_window_check"]:
            print(f"  split-window check: {r['split_window_check']}")

    out_dir = ROOT / "outputs" / "thetadata_stock_crosscheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump({"subscription_boundary": boundary, "per_symbol": results}, fh, indent=2, default=str)
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
