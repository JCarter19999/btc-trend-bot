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


def reset_client() -> None:
    """Force the next get_client() call to build a fresh ThetaClient.
    Needed because a single long-running process making many hundred
    sequential option_history_eod calls was found to degrade over time and
    eventually fail 100% of subsequent calls (confirmed directly: a
    background run resolved ~35% of dates through its first ~550 calls,
    then ZERO of the next ~740, while every one of those same failing
    dates succeeded immediately when re-tried in a fresh process/client).
    Root cause not further isolated (thetadata client/terminal connection
    or resource exhaustion, not a per-date data issue) -- this is a
    pragmatic mitigation (periodic + on-failure reset), not a fix of the
    underlying library behavior."""
    global _client
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


def build_option_chain(symbol: str, on_date: date, max_dte: int | None = None,
                        two_sided_only: bool = True) -> pd.DataFrame:
    """Full real options chain for `symbol` as it traded on `on_date` -- every
    expiration, every strike, both rights, in one ThetaData call rather than
    looping expiration_list x strike_list x quote calls. Returns an empty
    frame (not a fabricated one) if nothing is available for that date.

    `two_sided_only` drops quotes ThetaData returned with bid<=0 and ask<=0
    (no real market that day for that contract) -- same "don't paper over
    missing data" convention as real_quote_on_date above.
    """
    client = get_client()
    try:
        chain = client.option_history_eod(
            start_date=on_date, end_date=on_date, symbol=symbol,
            expiration="*", strike="*", right="both", max_dte=max_dte)
    except Exception:
        return pd.DataFrame()
    if chain is None or chain.empty:
        return pd.DataFrame()
    if two_sided_only:
        chain = chain[(chain["bid"] > 0) | (chain["ask"] > 0)]
    chain = chain.sort_values(["expiration", "strike", "right"]).reset_index(drop=True)
    chain["dte"] = (pd.to_datetime(chain["expiration"]).dt.date - on_date).apply(lambda d: d.days)
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2.0
    return chain


def real_atm_iv_on_date(symbol: str, on_date: date, target_dte: int) -> float | None:
    """Real ATM implied vol on a given date, solved from real quoted mid
    prices via implied_greeks.implied_volatility (inverts Black-Scholes
    against the real market price -- not a realized-vol proxy). One
    option-chain fetch per date.

    Deliberately takes NO external spot argument. This project's daily
    OHLCV (data/real/*.csv) is dividend/split-ADJUSTED (auto_adjust=True,
    per CLAUDE.md's convention for feature consistency) -- but real option
    strikes are fixed, unadjusted nominal dollar levels. Feeding an
    adjusted close into a Black-Scholes solve as "spot" silently mismatches
    strike-vs-spot moneyness (confirmed directly: on 2021-06-01, the
    adjusted close was $390.96 while strikes priced consistent with a true
    spot near $420 -- a ~7% error that would make a nominally "ATM" pick
    actually deep ITM relative to true spot and badly bias the IV solve, or
    fail the solve outright). Fixed via PUT-CALL PARITY instead: for the
    expiration nearest `target_dte`, the strike where the real call mid and
    put mid are closest together is (by parity, C-P=S-K*exp(-rT)~=S-K at
    short maturities) the true at-the-money strike, and true spot ~= that
    strike + (call_mid - put_mid) -- both derived entirely from real
    quotes, no external/adjusted price series involved at all.

    Returns None (not a fabricated value) if the chain isn't available for
    that date or no solvable IV exists in it."""
    from .implied_greeks import implied_volatility

    chain = build_option_chain(symbol, on_date, max_dte=target_dte + 15)
    if chain.empty:
        return None
    dtes = chain["dte"].unique()
    if len(dtes) == 0:
        return None
    nearest_dte = min(dtes, key=lambda d: abs(d - target_dte))
    exp_chain = chain[chain["dte"] == nearest_dte]
    calls = exp_chain[exp_chain["right"].str.upper() == "CALL"].drop_duplicates("strike").set_index("strike")
    puts = exp_chain[exp_chain["right"].str.upper() == "PUT"].drop_duplicates("strike").set_index("strike")
    common_strikes = calls.index.intersection(puts.index)
    common_strikes = [k for k in common_strikes if calls.loc[k, "mid"] > 0 and puts.loc[k, "mid"] > 0]
    if not common_strikes:
        return None

    gaps = {k: abs(calls.loc[k, "mid"] - puts.loc[k, "mid"]) for k in common_strikes}
    atm_strike = min(gaps, key=gaps.get)
    call_mid, put_mid = float(calls.loc[atm_strike, "mid"]), float(puts.loc[atm_strike, "mid"])
    implied_spot = atm_strike + (call_mid - put_mid)  # put-call parity, r*T~0 approximation at this DTE

    years_to_expiry = nearest_dte / 365.0
    ivs = []
    call_iv = implied_volatility(call_mid, implied_spot, atm_strike, years_to_expiry, "call")
    put_iv = implied_volatility(put_mid, implied_spot, atm_strike, years_to_expiry, "put")
    if call_iv is not None:
        ivs.append(call_iv)
    if put_iv is not None:
        ivs.append(put_iv)
    return float(np.mean(ivs)) if ivs else None


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


def real_short_strangle_trade(symbol: str, signal_date: date, entry_date: date, exit_date: date,
                               spot_at_signal: float, target_dte: int,
                               call_moneyness: float, put_moneyness: float,
                               tick_stress: float = 0.0, bps_stress: float = 0.0) -> dict | None:
    """Real SHORT strangle: sell an OTM call + OTM put at two independent
    strikes, same expiration (each strike found on its own target moneyness
    -- unlike a straddle, a strangle's two legs are never the same strike).
    Opposite fill convention from every long structure in this module: we
    SELL at the real bid (both legs) to open, and BUY BACK at the real ask
    (both legs) to close -- entry credit collected up front, exit debit paid
    to close. net_return is on the credit collected: +1.0 (=+100%) means
    both legs expired/closed worthless (max gain, credit fully kept);
    net_return is NOT floored at -1.0 the way a long option's is -- a
    strangle that moves deep ITM on either leg before exit can cost several
    multiples of the credit collected, and that should show up here as
    net_return << -1.0, not be silently capped.

    tick_stress / bps_stress apply an extra fill-quality haircut on TOP of
    the real quotes (sell-side fills reduced, buy-side fills increased) --
    used for the cost-stress check, not the baseline result.
    """
    expiration = find_nearest_expiration(symbol, signal_date, target_dte)
    if expiration is None or expiration <= exit_date:
        return None
    call_strike = find_nearest_strike(symbol, expiration, spot_at_signal * call_moneyness)
    put_strike = find_nearest_strike(symbol, expiration, spot_at_signal * put_moneyness)
    if call_strike is None or put_strike is None or call_strike == put_strike:
        return None

    call_entry = real_quote_on_date(symbol, expiration, call_strike, "call", entry_date)
    put_entry = real_quote_on_date(symbol, expiration, put_strike, "put", entry_date)
    if call_entry is None or put_entry is None:
        return None
    call_exit = real_quote_on_date(symbol, expiration, call_strike, "call", exit_date)
    put_exit = real_quote_on_date(symbol, expiration, put_strike, "put", exit_date)
    if call_exit is None or put_exit is None:
        return None

    call_entry_bid = max(call_entry["bid"] - tick_stress, 0.0) * (1 - bps_stress)
    put_entry_bid = max(put_entry["bid"] - tick_stress, 0.0) * (1 - bps_stress)
    entry_credit = call_entry_bid + put_entry_bid
    if entry_credit <= 0:
        return None

    call_exit_ask = (call_exit["ask"] + tick_stress) * (1 + bps_stress)
    put_exit_ask = (put_exit["ask"] + tick_stress) * (1 + bps_stress)
    exit_debit = call_exit_ask + put_exit_ask

    net_return = 1.0 - exit_debit / entry_credit
    return {
        "expiration": expiration, "call_strike": call_strike, "put_strike": put_strike,
        "entry_credit": entry_credit, "exit_debit": exit_debit,
        "call_entry_bid": call_entry_bid, "put_entry_bid": put_entry_bid,
        "call_exit_ask": call_exit_ask, "put_exit_ask": put_exit_ask,
        "call_entry_price": call_entry["ask"], "put_entry_price": put_entry["ask"],
        "call_exit_price": call_exit["bid"], "put_exit_price": put_exit["bid"],
        "net_return": net_return,
    }
