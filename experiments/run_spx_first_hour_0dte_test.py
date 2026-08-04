"""SPX first-hour direction -> 0DTE call/put test.

Question: does the sign of SPX's first-hour move (9:30-10:30am ET) predict
the direction of the rest of the day well enough to profitably buy a 0DTE
ATM call (first hour up) or put (first hour down) at 10:30am and hold to
the cash-settled close?

Data:
- Signal + payoff reference: SPX index level (^GSPC) via free yfinance 5m
  bars, 63 trading days, 2026-05-06 - 2026-07-31 (yfinance's entire free
  intraday history for this project -- same ceiling documented in
  EQUITY_OPENING_RANGE_BREAKOUT_TEST.md). Small-sample caveat applies here
  exactly as it did there.
- Entry cost: REAL SPXW 0DTE option bid/ask quotes bought from Databento
  OPRA.PILLAR (cbbo-1m, 14:30-14:35 UTC = 10:30am ET window each day),
  fetched by fetch_spx_0dte_options.py into data/opra_spx/. ATM strike is
  the 0DTE contract (same-day expiry) whose strike is closest to the SPX
  level at 10:30am; entry price = real ask. Exit is NOT a second option
  quote -- 0DTE cash-settles, so the payoff at the close is just the
  intrinsic value max(close-strike,0) [call] / max(strike-close,0) [put],
  computed from the same yfinance close used for the signal. This means
  the only real friction modeled is the entry spread, which is exactly
  right for a buy-and-hold-to-expiry 0DTE trade.

No lookahead: the first-hour return and the 10:30 ATM strike selection use
only bars/quotes at or before 10:30am; the exit is the day's actual final
close, known only at end of day.

Baselines: always-call, always-put, random-direction (shuffled-null,
many draws) -- same standard as every other signal test in this project.
"""

from __future__ import annotations

import glob
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OPRA_DIR = ROOT / "data" / "opra_spx"
OUT_DIR = ROOT / "outputs" / "spx_first_hour_0dte_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIRST_HOUR_BARS = 12  # 12 x 5min = 9:30 -> 10:30 ET
N_SHUFFLES = 2000
RNG_SEED = 20260803

SYMBOL_RE = re.compile(r"^SPXW\s+(\d{6})([CP])(\d{8})$")


def fetch_spx_5m() -> pd.DataFrame:
    df = yf.download("^GSPC", interval="5m", period="60d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df["date"] = df.index.tz_convert("America/New_York").normalize()
    df["bar_of_day"] = df.groupby("date").cumcount()
    return df


def build_day_signal(spx: pd.DataFrame) -> pd.DataFrame:
    """One row per day: open, first-hour close (bar index 11 = 12th bar),
    day close, and the resulting signal sign. No lookahead: only bars
    strictly at-or-before the first-hour close are used for the signal."""
    rows = []
    for d, day in spx.groupby("date"):
        day = day.sort_values("bar_of_day")
        if len(day) <= FIRST_HOUR_BARS:
            continue
        open_px = float(day["open"].iloc[0])
        fh_close = float(day.iloc[FIRST_HOUR_BARS - 1]["close"])
        day_close = float(day["close"].iloc[-1])
        fh_ret = fh_close / open_px - 1.0
        rows.append({
            "date": d.tz_localize(None).normalize(),
            "open": open_px,
            "first_hour_close": fh_close,
            "day_close": day_close,
            "first_hour_return": fh_ret,
            "signal": "call" if fh_ret > 0 else ("put" if fh_ret < 0 else "flat"),
        })
    return pd.DataFrame(rows)


def parse_spxw_symbol(sym: str):
    m = SYMBOL_RE.match(sym.strip() if isinstance(sym, str) else "")
    if not m:
        return None
    yymmdd, cp, strike8 = m.groups()
    expiry = pd.to_datetime(yymmdd, format="%y%m%d")
    strike = int(strike8) / 1000.0
    return expiry.normalize(), cp, strike


def pick_0dte_quote(parquet_path: Path, trade_date: pd.Timestamp, spot: float, want_cp: str):
    """From the 5-minute quote snapshot, keep only same-day-expiry (0DTE)
    contracts of the wanted type, take each contract's LAST quote in the
    window (closest to the 10:30 decision instant), and return the one
    whose strike is closest to spot (ATM)."""
    df = pd.read_parquet(parquet_path)
    parsed = df["symbol"].apply(parse_spxw_symbol)
    df = df[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    df["expiry"] = [p[0] for p in parsed]
    df["cp"] = [p[1] for p in parsed]
    df["strike"] = [p[2] for p in parsed]
    df = df[(df["expiry"] == trade_date) & (df["cp"] == want_cp)]
    df = df[(df["ask_px_00"] > 0) & (df["bid_px_00"] > 0)]
    if df.empty:
        return None
    df = df.sort_values("ts_event").groupby("strike", as_index=False).last()
    df["dist"] = (df["strike"] - spot).abs()
    row = df.sort_values("dist").iloc[0]
    return {
        "strike": float(row["strike"]),
        "bid": float(row["bid_px_00"]),
        "ask": float(row["ask_px_00"]),
    }


def main() -> int:
    spx = fetch_spx_5m()
    signal_df = build_day_signal(spx)
    signal_df = signal_df[signal_df["signal"] != "flat"].reset_index(drop=True)

    files = {pd.to_datetime(Path(f).stem): Path(f) for f in glob.glob(str(OPRA_DIR / "*.parquet"))}

    trades = []
    for _, r in signal_df.iterrows():
        d = pd.Timestamp(r["date"])
        if d not in files:
            continue
        cp = "C" if r["signal"] == "call" else "P"
        quote = pick_0dte_quote(files[d], d, r["first_hour_close"], cp)
        if quote is None:
            continue
        entry_cost = quote["ask"]  # buy at real ask
        if entry_cost <= 0:
            continue
        if cp == "C":
            payoff = max(r["day_close"] - quote["strike"], 0.0)
        else:
            payoff = max(quote["strike"] - r["day_close"], 0.0)
        ret = (payoff - entry_cost) / entry_cost
        trades.append({
            "date": d.strftime("%Y-%m-%d"),
            "signal": r["signal"],
            "first_hour_return": r["first_hour_return"],
            "strike": quote["strike"],
            "entry_ask": entry_cost,
            "day_close": r["day_close"],
            "payoff": payoff,
            "return": ret,
            "win": payoff > entry_cost,
        })

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(OUT_DIR / "trades.csv", index=False)

    def summarize(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"n": 0}
        return {
            "n": int(len(df)),
            "win_rate": float(df["win"].mean()),
            "mean_return": float(df["return"].mean()),
            "median_return": float(df["return"].median()),
            "total_return_equal_weight": float(df["return"].sum() / len(df)),
        }

    signal_summary = summarize(trades_df)

    # Baseline: always buy the call regardless of first-hour signal.
    always_call = []
    for _, r in signal_df.iterrows():
        d = pd.Timestamp(r["date"])
        if d not in files:
            continue
        quote = pick_0dte_quote(files[d], d, r["first_hour_close"], "C")
        if quote is None or quote["ask"] <= 0:
            continue
        payoff = max(r["day_close"] - quote["strike"], 0.0)
        ret = (payoff - quote["ask"]) / quote["ask"]
        always_call.append({"return": ret, "win": payoff > quote["ask"]})
    always_call_df = pd.DataFrame(always_call)

    always_put = []
    for _, r in signal_df.iterrows():
        d = pd.Timestamp(r["date"])
        if d not in files:
            continue
        quote = pick_0dte_quote(files[d], d, r["first_hour_close"], "P")
        if quote is None or quote["ask"] <= 0:
            continue
        payoff = max(quote["strike"] - r["day_close"], 0.0)
        ret = (payoff - quote["ask"]) / quote["ask"]
        always_put.append({"return": ret, "win": payoff > quote["ask"]})
    always_put_df = pd.DataFrame(always_put)

    # Shuffled-null: randomly relabel which days got "call" vs "put" signal
    # (keeping the same count of each), re-derive returns from the SAME
    # per-day call/put quotes already fetched above, many draws.
    rng = np.random.default_rng(RNG_SEED)
    per_day = {}
    for _, r in signal_df.iterrows():
        d = pd.Timestamp(r["date"])
        if d not in files:
            continue
        cq = pick_0dte_quote(files[d], d, r["first_hour_close"], "C")
        pq = pick_0dte_quote(files[d], d, r["first_hour_close"], "P")
        if cq is None or pq is None or cq["ask"] <= 0 or pq["ask"] <= 0:
            continue
        call_ret = (max(r["day_close"] - cq["strike"], 0.0) - cq["ask"]) / cq["ask"]
        put_ret = (max(pq["strike"] - r["day_close"], 0.0) - pq["ask"]) / pq["ask"]
        per_day[d] = (call_ret, put_ret)

    real_signal_map = dict(zip(signal_df["date"].astype(str), signal_df["signal"]))
    days = list(per_day.keys())
    n_call_actual = sum(1 for d in days if real_signal_map.get(d.strftime("%Y-%m-%d")) == "call")

    real_mean = float(np.mean([per_day[d][0] if real_signal_map.get(d.strftime("%Y-%m-%d")) == "call" else per_day[d][1] for d in days])) if days else float("nan")

    null_means = []
    for _ in range(N_SHUFFLES):
        picks = rng.permutation(len(days)) < n_call_actual
        rets = [per_day[d][0] if pick else per_day[d][1] for d, pick in zip(days, picks)]
        null_means.append(np.mean(rets))
    null_means = np.array(null_means)
    percentile = float((null_means < real_mean).mean() * 100) if days else float("nan")

    summary = {
        "n_signal_days": int(len(signal_df)),
        "n_days_with_option_data": int(len(days)),
        "date_range": [str(spx["date"].min().date()), str(spx["date"].max().date())],
        "signal_strategy": signal_summary,
        "always_call_baseline": summarize(always_call_df),
        "always_put_baseline": summarize(always_put_df),
        "shuffled_null": {
            "n_shuffles": N_SHUFFLES,
            "real_signal_mean_return": real_mean,
            "null_mean_of_means": float(null_means.mean()) if days else float("nan"),
            "null_std_of_means": float(null_means.std()) if days else float("nan"),
            "percentile_of_real_vs_null": percentile,
        },
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
