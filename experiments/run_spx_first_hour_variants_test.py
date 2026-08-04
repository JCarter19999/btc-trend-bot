"""SPX first-hour signal: variant matrix.

Extends run_spx_first_hour_0dte_test.py (naked ATM 0DTE, hold to close,
all days) along three axes the user asked about:

1. INSTRUMENT: naked ATM long option  vs.  vertical debit spread (buy ATM,
   sell the strike nearest spot +/- SPREAD_WIDTH_POINTS, same expiry/side --
   5-10 SPX points OTM per the user, strikes are 5-wide near the money).
   The spread trades premium for a capped payoff -- since the base test
   showed a median trade return of -100% (most naked 0DTE contracts expire
   worthless) driven by a small number of big winners, a cheaper capped
   structure is the natural next question.

2. EXIT: hold to cash-settled close  vs.  sell at the real bid at the
   noon-ET snapshot (9am PST -- same UTC offset as ET all through this
   EDT-only date range, so 9am PT = 12pm ET = 16:00 UTC). Requires a
   second OPRA pull (data/opra_spx_exit/, fetched via
   fetch_spx_0dte_options.py --window-start 16:00 --window-end 16:05).
   For the spread, "exit" = sell the long leg at its bid, buy back the
   short leg at its ask (both real quotes from the same snapshot).

3. REGIME: all days  vs.  choppy days only. Chop is measured RETROSPECTIVELY
   per day via Kaufman's efficiency ratio on the 5-minute path:
       ER = |close - open| / sum(|bar_t - bar_{t-1}|)
   Low ER = lots of back-and-forth with little net progress = choppy.
   Days below the median ER are "choppy". This is a same-day, after-the-
   fact classification -- it answers "does the signal work conditional on
   today turning out choppy," not "can chop be predicted before 10:30am."
   That distinction is called out again in the report.

No lookahead beyond the regime-filter caveat above: entry signal and ATM
strike use only 9:30-10:30 bars/quotes; the noon exit uses only the noon
snapshot; the close exit uses only the actual final close.
"""

from __future__ import annotations

import glob
import itertools
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OPRA_ENTRY_DIR = ROOT / "data" / "opra_spx"
OPRA_EXIT_DIR = ROOT / "data" / "opra_spx_exit"
OUT_DIR = ROOT / "outputs" / "spx_first_hour_variants_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIRST_HOUR_BARS = 12  # 12 x 5min = 9:30 -> 10:30 ET
N_SHUFFLES = 2000
RNG_SEED = 20260803
SPREAD_WIDTH_POINTS = 7.5  # short leg 5-10 SPX points OTM from spot (per user)

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


def efficiency_ratio(day: pd.DataFrame) -> float:
    closes = day.sort_values("bar_of_day")["close"].to_numpy()
    net = abs(closes[-1] - closes[0])
    path = np.abs(np.diff(closes)).sum()
    return float(net / path) if path > 0 else float("nan")


def build_day_signal(spx: pd.DataFrame) -> pd.DataFrame:
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
            "efficiency_ratio": efficiency_ratio(day),
        })
    df = pd.DataFrame(rows)
    df["choppy"] = df["efficiency_ratio"] < df["efficiency_ratio"].median()
    return df


def parse_spxw_symbol(sym: str):
    m = SYMBOL_RE.match(sym.strip() if isinstance(sym, str) else "")
    if not m:
        return None
    yymmdd, cp, strike8 = m.groups()
    expiry = pd.to_datetime(yymmdd, format="%y%m%d")
    strike = int(strike8) / 1000.0
    return expiry.normalize(), cp, strike


def load_chain(parquet_path: Path, trade_date: pd.Timestamp, want_cp: str) -> pd.DataFrame:
    """Last quote per strike in the snapshot window, for one side (C/P), 0DTE only."""
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
        return df
    return df.sort_values("ts_event").groupby("strike", as_index=False).last()


def nearest_strike_row(chain: pd.DataFrame, target: float) -> dict | None:
    if chain.empty:
        return None
    row = chain.iloc[(chain["strike"] - target).abs().argsort().iloc[0]]
    return {"strike": float(row["strike"]), "bid": float(row["bid_px_00"]), "ask": float(row["ask_px_00"])}


def main() -> int:
    spx = fetch_spx_5m()
    signal_df = build_day_signal(spx)
    signal_df = signal_df[signal_df["signal"] != "flat"].reset_index(drop=True)

    entry_files = {pd.to_datetime(Path(f).stem): Path(f) for f in glob.glob(str(OPRA_ENTRY_DIR / "*.parquet"))}
    exit_files = {pd.to_datetime(Path(f).stem): Path(f) for f in glob.glob(str(OPRA_EXIT_DIR / "*.parquet"))}

    records = []
    for _, r in signal_df.iterrows():
        d = pd.Timestamp(r["date"])
        if d not in entry_files:
            continue
        cp = "C" if r["signal"] == "call" else "P"
        spot = r["first_hour_close"]
        entry_chain = load_chain(entry_files[d], d, cp)
        long_leg = nearest_strike_row(entry_chain, spot)
        if long_leg is None:
            continue
        short_target = spot + SPREAD_WIDTH_POINTS if cp == "C" else spot - SPREAD_WIDTH_POINTS
        short_leg = nearest_strike_row(entry_chain, short_target)
        has_spread = short_leg is not None and short_leg["strike"] != long_leg["strike"]

        exit_chain = load_chain(exit_files[d], d, cp) if d in exit_files else pd.DataFrame()
        long_exit = None
        short_exit = None
        if not exit_chain.empty:
            m = exit_chain[exit_chain["strike"] == long_leg["strike"]]
            if not m.empty:
                long_exit = {"bid": float(m.iloc[0]["bid_px_00"]), "ask": float(m.iloc[0]["ask_px_00"])}
            if has_spread:
                m2 = exit_chain[exit_chain["strike"] == short_leg["strike"]]
                if not m2.empty:
                    short_exit = {"bid": float(m2.iloc[0]["bid_px_00"]), "ask": float(m2.iloc[0]["ask_px_00"])}

        def intrinsic(strike, is_call, close):
            return max(close - strike, 0.0) if is_call else max(strike - close, 0.0)

        is_call = cp == "C"
        close = r["day_close"]

        # naked, hold to close
        naked_close_cost = long_leg["ask"]
        naked_close_payoff = intrinsic(long_leg["strike"], is_call, close)
        naked_close_ret = (naked_close_payoff - naked_close_cost) / naked_close_cost if naked_close_cost > 0 else np.nan

        # naked, exit at noon (sell at bid)
        naked_noon_ret = np.nan
        if long_exit is not None and long_exit["bid"] > 0:
            naked_noon_ret = (long_exit["bid"] - naked_close_cost) / naked_close_cost

        # spread, hold to close
        spread_close_ret = np.nan
        if has_spread:
            debit = long_leg["ask"] - short_leg["bid"]
            width = abs(short_leg["strike"] - long_leg["strike"])
            if debit > 0:
                gross = intrinsic(long_leg["strike"], is_call, close) - intrinsic(short_leg["strike"], is_call, close)
                gross = min(gross, width)
                spread_close_ret = (gross - debit) / debit

        # spread, exit at noon
        spread_noon_ret = np.nan
        if has_spread and long_exit is not None and short_exit is not None:
            debit = long_leg["ask"] - short_leg["bid"]
            exit_credit = long_exit["bid"] - short_exit["ask"]
            if debit > 0:
                spread_noon_ret = (exit_credit - debit) / debit

        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "signal": r["signal"],
            "choppy": bool(r["choppy"]),
            "efficiency_ratio": r["efficiency_ratio"],
            "naked_close_ret": naked_close_ret,
            "naked_noon_ret": naked_noon_ret,
            "spread_close_ret": spread_close_ret,
            "spread_noon_ret": spread_noon_ret,
        })

    trades = pd.DataFrame(records)
    trades.to_csv(OUT_DIR / "trades.csv", index=False)

    def summarize(s: pd.Series) -> dict:
        s = s.dropna()
        if s.empty:
            return {"n": 0}
        return {
            "n": int(len(s)),
            "win_rate": float((s > 0).mean()),
            "mean_return": float(s.mean()),
            "median_return": float(s.median()),
        }

    rng = np.random.default_rng(RNG_SEED)

    def bootstrap_significance(col: str, subset: pd.DataFrame) -> dict:
        """Day-resampled (with replacement) bootstrap of the mean return.
        Reports the 95% CI and the percentile rank of zero within the
        bootstrap distribution -- e.g. if zero sits at the 3rd percentile,
        the mean is significantly positive at roughly the 94% two-sided
        confidence level; if zero sits in the middle, there's no signal."""
        s = subset[col].dropna()
        if s.empty:
            return {"n": 0}
        real_mean = float(s.mean())
        boot = rng.choice(s.to_numpy(), size=(N_SHUFFLES, len(s)), replace=True).mean(axis=1)
        zero_percentile = float((boot < 0).mean() * 100)
        return {
            "n": int(len(s)),
            "real_mean": real_mean,
            "bootstrap_ci_low": float(np.percentile(boot, 2.5)),
            "bootstrap_ci_high": float(np.percentile(boot, 97.5)),
            "zero_percentile_in_bootstrap": zero_percentile,
        }

    combos = list(itertools.product(
        ["naked_close_ret", "naked_noon_ret", "spread_close_ret", "spread_noon_ret"],
        [("all", trades), ("choppy_only", trades[trades["choppy"]])],
    ))

    results = {}
    for col, (label, subset) in combos:
        key = f"{col}__{label}"
        results[key] = {
            "summary": summarize(subset[col]),
            "significance": bootstrap_significance(col, subset),
        }

    summary = {
        "n_signal_days": int(len(signal_df)),
        "n_days_with_option_data": int(trades["naked_close_ret"].notna().sum()),
        "n_choppy_days": int(trades["choppy"].sum()),
        "spread_width_points": SPREAD_WIDTH_POINTS,
        "note": "regime filter is retrospective (full-day efficiency ratio), see script docstring",
        "results": results,
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
