"""First-hour (9:30-10:30 ET) feature engineering, free data only.

Every feature here is computed from bars at-or-before the first-hour close
(bar_of_day index FIRST_HOUR_BARS-1) -- no lookahead. Confirmation symbols
(SPY/QQQ/IWM/VIX) are fetched alongside the SPX signal because SPX itself
carries no volume, so VWAP and relative-volume features use SPY as a proxy
(documented inline wherever that substitution happens).

Not computed here (see OPTIONS_ROUTER_PHASE1.md for why): implied move
(needs the options chain, computed downstream once a chain is loaded),
overnight realized volatility (no overnight tick data in this feed -- only
the open/close gap is available), breadth/advance-decline/equal-vs-cap-weight
(no vendor for this yet).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FIRST_HOUR_BARS = 12  # 12 x 5min = 9:30 -> 10:30 ET
TRAILING_WINDOW_DAYS = 5  # per user: keep the "recent baseline" short and intraday-focused


def _tag_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df["date"] = df.index.tz_convert("America/New_York").normalize()
    df["bar_of_day"] = df.groupby("date").cumcount()
    return df


def fetch_confirmation_data(symbols: list[str], underlying: str = "^GSPC",
                             period: str = "60d") -> dict[str, pd.DataFrame]:
    """5-minute bars for the signal underlying plus confirmation symbols."""
    out: dict[str, pd.DataFrame] = {}
    for sym in [underlying, *symbols]:
        raw = yf.download(sym, interval="5m", period=period, progress=False, auto_adjust=True)
        out[sym] = _tag_bars(raw)
    return out


def _efficiency_ratio(closes: np.ndarray) -> float:
    net = abs(closes[-1] - closes[0])
    path = np.abs(np.diff(closes)).sum()
    return float(net / path) if path > 0 else float("nan")


def _linreg_slope_r2(y: np.ndarray) -> tuple[float, float]:
    x = np.arange(len(y), dtype=float)
    if len(y) < 2 or np.all(y == y[0]):
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r2)


def _hh_hl_lh_ll(high: np.ndarray, low: np.ndarray) -> dict:
    hh = int((high[1:] > high[:-1]).sum())
    hl = int((low[1:] > low[:-1]).sum())
    lh = int((high[1:] < high[:-1]).sum())
    ll = int((low[1:] < low[:-1]).sum())
    return {"higher_highs": hh, "higher_lows": hl, "lower_highs": lh, "lower_lows": ll}


def _max_consecutive_run(closes: np.ndarray) -> int:
    diffs = np.sign(np.diff(closes))
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0
    best = cur = 1
    for i in range(1, len(diffs)):
        cur = cur + 1 if diffs[i] == diffs[i - 1] else 1
        best = max(best, cur)
    return int(best)


def _vwap_series(day: pd.DataFrame) -> pd.Series:
    typical = (day["high"] + day["low"] + day["close"]) / 3.0
    cum_pv = (typical * day["volume"]).cumsum()
    cum_v = day["volume"].cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def _vwap_crossings(closes: np.ndarray, vwap: np.ndarray) -> int:
    sign = np.sign(closes - vwap)
    sign = sign[~np.isnan(sign)]
    if len(sign) < 2:
        return 0
    return int((np.diff(sign) != 0).sum())


def build_first_hour_features(bars_by_symbol: dict[str, pd.DataFrame],
                               underlying: str = "^GSPC",
                               trailing_window_days: int = TRAILING_WINDOW_DAYS) -> pd.DataFrame:
    """One row per trading day, using only bars at-or-before the first-hour
    close (bar_of_day index FIRST_HOUR_BARS-1 = 12th 5-minute bar).

    Trailing-average features (first_hour_range_vs_trailing_avg, SPY relative
    volume) use a short ROLLING window of `trailing_window_days` prior days,
    not an expanding all-history average -- the point is to compare today's
    first hour against a recent, intraday-relevant baseline, not a stale
    multi-month one."""
    spx = bars_by_symbol[underlying]
    spy = bars_by_symbol.get("SPY")
    qqq = bars_by_symbol.get("QQQ")
    iwm = bars_by_symbol.get("IWM")
    vix = bars_by_symbol.get("^VIX")

    dates = sorted(d for d in spx["date"].unique())
    prev_close = None
    fh_range_history: deque[float] = deque(maxlen=trailing_window_days)
    daily_range_history: deque[float] = deque(maxlen=trailing_window_days)
    fh_vol_history: deque[float] = deque(maxlen=trailing_window_days)
    rows = []

    for d in dates:
        day = spx[spx["date"] == d].sort_values("bar_of_day")
        if len(day) <= FIRST_HOUR_BARS:
            continue
        fh = day.iloc[:FIRST_HOUR_BARS]
        full_day_close = float(day["close"].iloc[-1])

        open_px = float(fh["open"].iloc[0])
        fh_close = float(fh["close"].iloc[-1])
        fh_high = float(fh["high"].max())
        fh_low = float(fh["low"].min())
        fh_range = fh_high - fh_low
        clv = (fh_close - fh_low) / fh_range if fh_range > 0 else 0.5

        ret_5m = float(fh["close"].iloc[0] / open_px - 1.0)
        ret_15m = float(fh["close"].iloc[2] / open_px - 1.0) if len(fh) > 2 else float("nan")
        ret_30m = float(fh["close"].iloc[5] / open_px - 1.0) if len(fh) > 5 else float("nan")
        ret_60m = float(fh_close / open_px - 1.0)

        er = _efficiency_ratio(fh["close"].to_numpy())
        slope, r2 = _linreg_slope_r2(fh["close"].to_numpy())
        hh_hl = _hh_hl_lh_ll(fh["high"].to_numpy(), fh["low"].to_numpy())
        consec = _max_consecutive_run(fh["close"].to_numpy())

        overnight_gap = float(open_px / prev_close - 1.0) if prev_close else float("nan")
        price_rel_prev_close = float(fh_close / prev_close - 1.0) if prev_close else float("nan")

        trailing_fh_range_avg = float(np.mean(fh_range_history)) if fh_range_history else float("nan")
        fh_range_vs_avg = float(fh_range / trailing_fh_range_avg) if trailing_fh_range_avg and trailing_fh_range_avg > 0 else float("nan")
        prev_day_range = daily_range_history[-1] if daily_range_history else float("nan")

        row = {
            "date": d.tz_localize(None).normalize(),
            "open": open_px,
            "first_hour_close": fh_close,
            "first_hour_high": fh_high,
            "first_hour_low": fh_low,
            "day_close": full_day_close,
            "overnight_gap": overnight_gap,
            "return_5m": ret_5m,
            "return_15m": ret_15m,
            "return_30m": ret_30m,
            "return_60m": ret_60m,
            "first_hour_return": ret_60m,
            "first_hour_range": fh_range,
            "distance_from_fh_high": float((fh_high - fh_close) / fh_high) if fh_high else float("nan"),
            "distance_from_fh_low": float((fh_close - fh_low) / fh_low) if fh_low else float("nan"),
            "close_location_value": clv,
            "price_rel_open": ret_60m,
            "price_rel_prev_close": price_rel_prev_close,
            "efficiency_ratio": er,
            "trend_slope": slope,
            "trend_r2": r2,
            "consecutive_run": consec,
            "first_hour_range_vs_trailing_avg": fh_range_vs_avg,
            "prev_day_range": prev_day_range,
            **hh_hl,
        }

        for sym_key, sym_df, prefix in [("SPY", spy, "spy"), ("QQQ", qqq, "qqq"), ("IWM", iwm, "iwm")]:
            if sym_df is None:
                continue
            sday = sym_df[sym_df["date"] == d].sort_values("bar_of_day")
            if len(sday) <= FIRST_HOUR_BARS:
                continue
            sfh = sday.iloc[:FIRST_HOUR_BARS]
            s_open = float(sfh["open"].iloc[0])
            s_close = float(sfh["close"].iloc[-1])
            row[f"{prefix}_first_hour_return"] = float(s_close / s_open - 1.0)
            if sym_key == "SPY":
                vwap = _vwap_series(sfh)
                vwap_last = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else float("nan")
                row["spy_vwap"] = vwap_last
                row["distance_from_vwap"] = float(s_close / vwap_last - 1.0) if vwap_last and not np.isnan(vwap_last) else float("nan")
                row["vwap_crossings"] = _vwap_crossings(sfh["close"].to_numpy(), vwap.to_numpy())
                cum_vol = float(sfh["volume"].sum())
                fh_vol_history_avg = float(np.mean(fh_vol_history)) if fh_vol_history else float("nan")
                row["spy_relative_volume"] = float(cum_vol / fh_vol_history_avg) if fh_vol_history_avg and fh_vol_history_avg > 0 else float("nan")
                fh_vol_history.append(cum_vol)
            else:
                row[f"{prefix}_relative_return"] = row[f"{prefix}_first_hour_return"] - ret_60m

        if vix is not None:
            vday = vix[vix["date"] == d].sort_values("bar_of_day")
            if len(vday) > FIRST_HOUR_BARS:
                vfh = vday.iloc[:FIRST_HOUR_BARS]
                v_open = float(vfh["open"].iloc[0])
                v_close = float(vfh["close"].iloc[-1])
                row["vix_level"] = v_close
                row["vix_first_hour_change"] = float(v_close / v_open - 1.0)

        rows.append(row)
        prev_close = full_day_close
        fh_range_history.append(fh_range)
        daily_range_history.append(float(day["high"].max() - day["low"].min()))

    return pd.DataFrame(rows)
