"""Real historical options pricing via ThetaData -- replaces the
Black-Scholes-off-realized-vol synthetic pricing in options_pricing.py for
symbols/dates where real data is available. This is the actual resolution
to every "not promotion-grade, needs real spread data" caveat in this
project's options docs.

Coverage note, confirmed empirically (not assumed): real 2020 TSLA data
returned "No data found" under the $40/mo Options Value tier; 2021 onward
returned real strikes/quotes correctly. Older history needs a higher tier.
Every symbol/date combination here is checked against what's actually
available, not assumed -- `find_real_contract` returns None (not a
fabricated fallback) when nothing usable exists, and callers must treat
that as "no trade," not paper over it with a synthetic price.

Execution convention: buy at the real ask, sell at the real bid (not
midpoint) -- matching what every prior options doc in this project called
for and could only assume before now.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

_client = None


def get_client():
    global _client
    if _client is None:
        import os
        if "THETADATA_API_KEY" not in os.environ:
            # Don't depend on the launching shell having sourced this --
            # a background run launched without it fails opaquely with
            # AuthenticationError deep inside ThetaClient.__init__ (bit us
            # once: tail hedge re-test launched from a fresh shell after a
            # context reset had no THETADATA_API_KEY exported).
            env_path = "/home/joey/.config/btc-trend-bot/thetadata.env"
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            os.environ.setdefault(key.strip(), value.strip())
        from thetadata import ThetaClient
        _client = ThetaClient(dataframe_type="pandas")
    return _client


def find_nearest_expiration(symbol: str, signal_date: date, target_dte: int) -> date | None:
    client = get_client()
    try:
        exps = client.option_list_expirations(symbol=symbol)
    except Exception:
        return None
    if exps is None or exps.empty:
        return None
    exp_dates = pd.to_datetime(exps["expiration"]).dt.date
    target_date = signal_date + timedelta(days=target_dte)
    future_exps = exp_dates[exp_dates > signal_date]
    if future_exps.empty:
        return None
    diffs = future_exps.apply(lambda d: abs((d - target_date).days))
    return future_exps.loc[diffs.idxmin()]


def find_nearest_strike(symbol: str, expiration: date, target_strike: float) -> float | None:
    client = get_client()
    try:
        strikes = client.option_list_strikes(symbol=symbol, expiration=expiration)
    except Exception:
        return None
    if strikes is None or strikes.empty:
        return None
    diffs = (strikes["strike"] - target_strike).abs()
    return float(strikes.loc[diffs.idxmin(), "strike"])


def real_quote_on_date(symbol: str, expiration: date, strike: float, right: str, on_date: date) -> dict | None:
    """Real bid/ask for a specific contract on a specific date -- the
    building block every trade below is priced from. Returns None (not a
    fabricated value) if no data exists for that exact date."""
    client = get_client()
    try:
        eod = client.option_history_eod(
            start_date=on_date, end_date=on_date, symbol=symbol,
            expiration=expiration, strike=f"{strike:.2f}", right=right.upper())
    except Exception:
        return None
    if eod is None or eod.empty:
        return None
    row = eod.iloc[0]
    if row["bid"] <= 0 and row["ask"] <= 0:
        return None
    return {"bid": float(row["bid"]), "ask": float(row["ask"]), "volume": float(row.get("volume", 0))}


def real_option_leg_trade(symbol: str, signal_date: date, entry_date: date, exit_date: date,
                           spot_at_signal: float, target_dte: int, strike_moneyness: float, right: str) -> dict | None:
    """Real single-leg option trade (call or put): contract selected using
    only information known at signal_date (spot price, target DTE/
    moneyness), buy at real ask on entry_date, sell at real bid on
    exit_date. Returns None if the contract or either quote isn't
    available -- caller must treat as no-trade, not interpolate or
    substitute a synthetic price."""
    expiration = find_nearest_expiration(symbol, signal_date, target_dte)
    if expiration is None or expiration <= exit_date:
        return None
    target_strike = spot_at_signal * strike_moneyness
    strike = find_nearest_strike(symbol, expiration, target_strike)
    if strike is None:
        return None

    entry_q = real_quote_on_date(symbol, expiration, strike, right, entry_date)
    if entry_q is None or entry_q["ask"] <= 0:
        return None
    exit_q = real_quote_on_date(symbol, expiration, strike, right, exit_date)
    if exit_q is None:
        return None

    entry_fill = entry_q["ask"]
    exit_fill = max(exit_q["bid"], 0.0)  # a real bid of 0 (expired worthless / no bid) is a real -100%, not missing data
    net_return = exit_fill / entry_fill - 1
    return {
        "expiration": expiration, "strike": strike, "right": right,
        "entry_ask": entry_fill, "exit_bid": exit_fill,
        "entry_volume": entry_q["volume"], "net_return": net_return,
    }


def real_call_trade(symbol: str, signal_date: date, entry_date: date, exit_date: date,
                     spot_at_signal: float, target_dte: int, strike_moneyness: float) -> dict | None:
    return real_option_leg_trade(symbol, signal_date, entry_date, exit_date, spot_at_signal,
                                  target_dte, strike_moneyness, "call")


def real_put_trade(symbol: str, signal_date: date, entry_date: date, exit_date: date,
                    spot_at_signal: float, target_dte: int, strike_moneyness: float) -> dict | None:
    return real_option_leg_trade(symbol, signal_date, entry_date, exit_date, spot_at_signal,
                                  target_dte, strike_moneyness, "put")


def real_straddle_trade(symbol: str, signal_date: date, entry_date: date, exit_date: date,
                         spot_at_signal: float, target_dte: int, strike_moneyness: float = 1.0) -> dict | None:
    """Real long straddle: call + put at the SAME strike/expiration (found
    once, shared by both legs, not independently nearest-matched -- a real
    straddle only makes sense on one strike). Both legs priced at real
    ask-in/bid-out; net_return is on the combined premium (call+put)."""
    expiration = find_nearest_expiration(symbol, signal_date, target_dte)
    if expiration is None or expiration <= exit_date:
        return None
    target_strike = spot_at_signal * strike_moneyness
    strike = find_nearest_strike(symbol, expiration, target_strike)
    if strike is None:
        return None

    call_entry = real_quote_on_date(symbol, expiration, strike, "call", entry_date)
    put_entry = real_quote_on_date(symbol, expiration, strike, "put", entry_date)
    if call_entry is None or put_entry is None or (call_entry["ask"] + put_entry["ask"]) <= 0:
        return None
    call_exit = real_quote_on_date(symbol, expiration, strike, "call", exit_date)
    put_exit = real_quote_on_date(symbol, expiration, strike, "put", exit_date)
    if call_exit is None or put_exit is None:
        return None

    entry_fill = call_entry["ask"] + put_entry["ask"]
    exit_fill = max(call_exit["bid"], 0.0) + max(put_exit["bid"], 0.0)
    net_return = exit_fill / entry_fill - 1
    return {
        "expiration": expiration, "strike": strike,
        "entry_premium": entry_fill, "exit_premium": exit_fill,
        "call_entry_ask": call_entry["ask"], "put_entry_ask": put_entry["ask"],
        "net_return": net_return,
    }
