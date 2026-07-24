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

from .implied_greeks import bs_delta, implied_volatility
from .thetadata_pricing import get_client, find_nearest_expiration, find_nearest_strike

ENTRY_TIME = time(9, 30)
EXIT_TIME = time(10, 30)


def _entry_time_seconds() -> int:
    return ENTRY_TIME.hour * 3600 + ENTRY_TIME.minute * 60


def _safe_quote(client, **kwargs) -> pd.DataFrame | None:
    """Every direct client call in this module MUST go through this --
    found the hard way (2026-07-24) that this file, unlike
    thetadata_pricing.py, called client.option_history_quote directly
    with no exception handling, which crashed an entire batch run on the
    first date/strike/right combination with genuinely no data
    (NoDataFoundError). The earlier 0DTE/1DTE runs "succeeding" with 0
    skipped was luck (never hit a missing combination in 116 real calls),
    not correctness -- this fixes it everywhere in the file, not just the
    one call site that happened to crash first."""
    try:
        return client.option_history_quote(**kwargs)
    except Exception:
        return None


def _safe_expirations(client, symbol: str) -> pd.DataFrame | None:
    try:
        return client.option_list_expirations(symbol=symbol)
    except Exception:
        return None


def find_delta_targeted_strike(symbol: str, expiration: date, trade_date: date, right: str,
                                spot_at_entry: float, target_delta: float, dte_years: float) -> tuple[float, float] | None:
    """Pulls ALL strikes' quotes near market open in a single call
    (`strike='*'`), computes real implied vol + delta for each from the
    actual quoted price (not assumed), and returns the strike whose delta
    is closest to `target_delta`. Real Greeks derived from real prices --
    see implied_greeks.py's module docstring for why this stays within
    the $40/mo tier instead of paying for direct Greeks access."""
    client = get_client()
    all_quotes = _safe_quote(client, symbol=symbol, expiration=expiration, strike="*", right=right,
                              date=trade_date, interval="1m", start_time="09:29:00", end_time="09:31:00")
    if all_quotes is None or all_quotes.empty:
        return None

    ts = pd.to_datetime(all_quotes["timestamp"]).dt.time
    near_open = all_quotes[ts.apply(lambda t: abs((t.hour * 3600 + t.minute * 60) - _entry_time_seconds()) <= 90)]
    near_open = near_open[near_open["ask"].notna() & (near_open["ask"] > 0.05)]
    if near_open.empty:
        return None

    best_strike, best_diff, best_ask = None, None, None
    for strike, group in near_open.groupby("strike"):
        ask = float(group["ask"].iloc[-1])
        iv = implied_volatility(ask, spot_at_entry, float(strike), dte_years, right)
        if iv is None:
            continue
        delta = bs_delta(spot_at_entry, float(strike), dte_years, iv, right)
        if delta is None:
            continue
        diff = abs(abs(delta) - target_delta)
        if best_diff is None or diff < best_diff:
            best_strike, best_diff, best_ask = float(strike), diff, ask
    if best_strike is None:
        return None
    return best_strike, best_ask


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

    expirations = _safe_expirations(client, symbol)
    if expirations is None or expirations.empty:
        return None
    exp_dates = pd.to_datetime(expirations["expiration"]).dt.date
    future_exps = sorted(d for d in exp_dates.unique() if d >= trade_date)
    if len(future_exps) <= dte_days:
        return None
    expiration = future_exps[dte_days]

    strike = find_nearest_strike(symbol, expiration, spot_at_entry)
    if strike is None:
        return None

    entry_quotes = _safe_quote(client,
        symbol=symbol, expiration=expiration, strike=f"{strike:.2f}", right=right,
        date=trade_date, interval="1m", start_time="09:29:00", end_time="09:35:00")
    entry_q = _nearest_quote(entry_quotes, ENTRY_TIME)
    if entry_q is None or entry_q["ask"] <= 0:
        return None

    exit_quotes = _safe_quote(client,
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

    expirations = _safe_expirations(client, symbol)
    if expirations is None or expirations.empty:
        return None
    exp_dates = pd.to_datetime(expirations["expiration"]).dt.date
    future_exps = sorted(d for d in exp_dates.unique() if d >= trade_date)
    if len(future_exps) <= dte_days:
        return None
    expiration = future_exps[dte_days]

    strike = find_nearest_strike(symbol, expiration, spot_at_entry)
    if strike is None:
        return None

    entry_quotes = _safe_quote(client,
        symbol=symbol, expiration=expiration, strike=f"{strike:.2f}", right=right,
        date=trade_date, interval="1m", start_time="09:29:00", end_time="09:35:00")
    exit_quotes = _safe_quote(client,
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


def real_delta_targeted_trade(symbol: str, trade_date: date, direction: int, dte_days: int,
                               spot_at_entry: float, target_delta: float = 0.40) -> dict | None:
    """Same first-hour structure as real_first_hour_option_trade, but the
    strike is chosen by real delta (computed from real prices) instead of
    moneyness/ATM."""
    client = get_client()
    right = "call" if direction > 0 else "put"

    expirations = _safe_expirations(client, symbol)
    if expirations is None or expirations.empty:
        return None
    exp_dates = pd.to_datetime(expirations["expiration"]).dt.date
    future_exps = sorted(d for d in exp_dates.unique() if d >= trade_date)
    if len(future_exps) <= dte_days:
        return None
    expiration = future_exps[dte_days]
    dte_years = max((expiration - trade_date).days, 1) / 365

    found = find_delta_targeted_strike(symbol, expiration, trade_date, right, spot_at_entry, target_delta, dte_years)
    if found is None:
        return None
    strike, _ = found

    entry_quotes = _safe_quote(client,
        symbol=symbol, expiration=expiration, strike=f"{strike:.2f}", right=right,
        date=trade_date, interval="1m", start_time="09:29:00", end_time="09:35:00")
    entry_q = _nearest_quote(entry_quotes, ENTRY_TIME)
    if entry_q is None or entry_q["ask"] <= 0:
        return None

    exit_quotes = _safe_quote(client,
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


def real_vertical_spread_first_hour_trade(symbol: str, trade_date: date, direction: int, dte_days: int,
                                           spot_at_entry: float, width_pct: float = 0.01) -> dict | None:
    """Vertical (debit) spread version of the same first-hour trade: long
    the ATM leg, short a strike `width_pct` further out-of-the-money
    (same direction as the long leg's moneyside), same expiry. Both legs
    real ask-in/bid-out; net_return is on the net debit paid."""
    client = get_client()
    right = "call" if direction > 0 else "put"

    expirations = _safe_expirations(client, symbol)
    if expirations is None or expirations.empty:
        return None
    exp_dates = pd.to_datetime(expirations["expiration"]).dt.date
    future_exps = sorted(d for d in exp_dates.unique() if d >= trade_date)
    if len(future_exps) <= dte_days:
        return None
    expiration = future_exps[dte_days]

    long_strike = find_nearest_strike(symbol, expiration, spot_at_entry)
    if long_strike is None:
        return None
    far_target = spot_at_entry * (1 + width_pct) if direction > 0 else spot_at_entry * (1 - width_pct)
    short_strike = find_nearest_strike(symbol, expiration, far_target)
    if short_strike is None or short_strike == long_strike:
        return None

    def _leg_quotes(strike: float, start: str, end: str) -> pd.DataFrame | None:
        return _safe_quote(client, symbol=symbol, expiration=expiration, strike=f"{strike:.2f}",
                            right=right, date=trade_date, interval="1m", start_time=start, end_time=end)

    long_entry = _nearest_quote(_leg_quotes(long_strike, "09:29:00", "09:35:00"), ENTRY_TIME)
    short_entry = _nearest_quote(_leg_quotes(short_strike, "09:29:00", "09:35:00"), ENTRY_TIME)
    if long_entry is None or short_entry is None or long_entry["ask"] <= 0:
        return None
    net_debit = long_entry["ask"] - short_entry["bid"]
    if net_debit <= 0.03:  # same MIN_TRADABLE_PREMIUM-style floor, adapted for a net debit
        return None

    long_exit = _nearest_quote(_leg_quotes(long_strike, "10:25:00", "10:35:00"), EXIT_TIME)
    short_exit = _nearest_quote(_leg_quotes(short_strike, "10:25:00", "10:35:00"), EXIT_TIME)
    if long_exit is None or short_exit is None:
        return None
    net_credit = max(long_exit["bid"], 0.0) - short_exit["ask"]

    return {
        "expiration": expiration, "long_strike": long_strike, "short_strike": short_strike, "right": right,
        "net_debit": net_debit, "net_credit": net_credit,
        "net_return": max(net_credit / net_debit - 1, -1.0),  # same defined-risk floor as the daily-DTE debit spread
    }
