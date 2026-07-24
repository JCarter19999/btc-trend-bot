"""Real intraday option pricing for the European lead signal's SPY/QQQ
0DTE and 1DTE overlay -- entry at the real ask near market open (9:30 ET
/ 13:30 UTC, matching the signal's actual entry time), exit at the real
bid near 10:30 ET / 14:30 UTC, one hour later. Uses 1-minute intraday
quotes (`option_history_quote`), not end-of-day EOD data -- the European
signal's hold period is a single hour, EOD granularity can't resolve it.
"""

from __future__ import annotations

from datetime import date, time

import pandas as pd

from .thetadata_pricing import get_client, find_nearest_expiration, find_nearest_strike

ENTRY_TIME = time(9, 30)
EXIT_TIME = time(10, 30)


def _nearest_quote(quotes: pd.DataFrame, target_time: time) -> dict | None:
    """Nearest-by-time quote among rows with an actual valid bid/ask --
    NaN comparisons (`nan <= 0`) are always False in Python, so a naive
    `<= 0` check silently lets NaN rows through as if they were valid
    (found this the hard way testing one trade before scaling up: entry
    quote was NaN but passed the old check). Filter to valid rows FIRST,
    then pick nearest-by-time among those, rather than picking
    nearest-by-time-regardless-of-validity and checking after (the
    nearest-by-time row is very often the very first row in the window,
    which ThetaData frequently returns as NaN before the first real print)."""
    if quotes is None or quotes.empty:
        return None
    valid = quotes[quotes["bid"].notna() & quotes["ask"].notna() & ((quotes["bid"] > 0) | (quotes["ask"] > 0))]
    if valid.empty:
        return None
    ts = pd.to_datetime(valid["timestamp"]).dt.time
    diffs = valid.assign(_t=ts)["_t"].apply(lambda t: abs((t.hour * 3600 + t.minute * 60 + t.second) -
                                                            (target_time.hour * 3600 + target_time.minute * 60)))
    row = valid.loc[diffs.idxmin()]
    return {"bid": float(row["bid"]), "ask": float(row["ask"])}


def real_first_hour_option_trade(symbol: str, trade_date: date, direction: int, dte_days: int,
                                  spot_at_entry: float) -> dict | None:
    """direction +1 -> buy a call (expressing long SPY), -1 -> buy a put.
    dte_days: 0 for same-day expiry, 1 for next trading day. ATM strike
    selected using the REAL spot price at entry (from this project's own
    already-downloaded SPY/QQQ hourly bars, data/intraday/*.csv -- not
    guessed from the strike list's median, which isn't reliably centered
    on the current price)."""
    client = get_client()
    right = "call" if direction > 0 else "put"

    expirations = client.option_list_expirations(symbol=symbol)
    exp_dates = pd.to_datetime(expirations["expiration"]).dt.date
    future_exps = sorted(d for d in exp_dates.unique() if d >= trade_date)
    if len(future_exps) <= dte_days:
        return None
    expiration = future_exps[dte_days]

    strike = find_nearest_strike(symbol, expiration, spot_at_entry)
    if strike is None:
        return None

    entry_quotes = client.option_history_quote(
        symbol=symbol, expiration=expiration, strike=f"{strike:.2f}", right=right,
        date=trade_date, interval="1m", start_time="09:29:00", end_time="09:35:00")
    entry_q = _nearest_quote(entry_quotes, ENTRY_TIME)
    if entry_q is None or entry_q["ask"] <= 0:
        return None

    exit_quotes = client.option_history_quote(
        symbol=symbol, expiration=expiration, strike=f"{strike:.2f}", right=right,
        date=trade_date, interval="1m", start_time="10:25:00", end_time="10:35:00")
    exit_q = _nearest_quote(exit_quotes, EXIT_TIME)
    if exit_q is None:
        return None

    entry_fill = entry_q["ask"]
    exit_fill = max(exit_q["bid"], 0.0)
    return {
        "expiration": expiration, "strike": strike, "right": right,
        "entry_ask": entry_fill, "exit_bid": exit_fill, "net_return": exit_fill / entry_fill - 1,
    }


def real_first_hour_option_trade_with_entry_delays(symbol: str, trade_date: date, direction: int, dte_days: int,
                                                     spot_at_entry: float, delays_seconds: tuple[int, ...] = (0, 5, 30, 60)
                                                     ) -> dict | None:
    """Same trade, but returns net_return at SEVERAL entry-delay scenarios
    from a single quote pull (the 09:29-09:35 window already covers up to
    5 minutes of delay) -- avoids re-pulling the API once per delay
    scenario. Exit is always at the fixed ~10:30 target regardless of
    entry delay (the signal's exit rule doesn't shift with entry timing)."""
    client = get_client()
    right = "call" if direction > 0 else "put"

    expirations = client.option_list_expirations(symbol=symbol)
    exp_dates = pd.to_datetime(expirations["expiration"]).dt.date
    future_exps = sorted(d for d in exp_dates.unique() if d >= trade_date)
    if len(future_exps) <= dte_days:
        return None
    expiration = future_exps[dte_days]

    strike = find_nearest_strike(symbol, expiration, spot_at_entry)
    if strike is None:
        return None

    entry_quotes = client.option_history_quote(
        symbol=symbol, expiration=expiration, strike=f"{strike:.2f}", right=right,
        date=trade_date, interval="1m", start_time="09:29:00", end_time="09:35:00")
    exit_quotes = client.option_history_quote(
        symbol=symbol, expiration=expiration, strike=f"{strike:.2f}", right=right,
        date=trade_date, interval="1m", start_time="10:25:00", end_time="10:35:00")
    exit_q = _nearest_quote(exit_quotes, EXIT_TIME)
    if exit_q is None:
        return None
    exit_fill = max(exit_q["bid"], 0.0)

    out = {"expiration": expiration, "strike": strike, "right": right, "exit_bid": exit_fill}
    for delay in delays_seconds:
        total_seconds = ENTRY_TIME.hour * 3600 + ENTRY_TIME.minute * 60 + delay
        target = time(total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60)
        entry_q = _nearest_quote(entry_quotes, target)
        if entry_q is None or entry_q["ask"] <= 0:
            out[f"net_return_delay{delay}s"] = None
        else:
            out[f"entry_ask_delay{delay}s"] = entry_q["ask"]
            out[f"net_return_delay{delay}s"] = exit_fill / entry_q["ask"] - 1
    return out
