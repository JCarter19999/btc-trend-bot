"""Black-Scholes option pricing calibrated to REAL underlying price data, for
a "nail in the coffin" options deep-dive on real (not synthetic) daily OHLCV.

This is explicitly NOT a historical options backtest -- no real historical
option chain data is available for free (checked: yfinance only exposes
current/live chains, not expired historical contracts; comprehensive
vendors like ORATS/IVolatility/EODHD are paid -- see BTC_ORDERFLOW_STUDY.md's
sibling doc, EQUITY_OPTIONS_DEEP_DIVE.md, for the source check). This module
prices synthetic contracts on real underlying price paths using trailing
REALIZED volatility (computed from real daily returns) as the implied-vol
proxy.

Deliberate, honest bias, stated once here and referenced everywhere this
module is used: real implied volatility is almost always higher than
trailing realized volatility (the volatility risk premium -- options are
structurally a bit expensive most of the time, since sellers demand
compensation for tail risk). Pricing off realized vol therefore
systematically UNDERPRICES these synthetic options relative to what they
would really have cost. That makes every result from this module generous
to the options side, not neutral. A negative result here is real evidence
against options; a positive result is weak evidence, since real premiums
would likely have been higher and eaten into it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Real options markets don't quote sub-tick prices -- a deep-OTM, low-vol
# contract that Black-Scholes prices at a fraction of a cent (e.g. $0.003 on
# a $400 spot at 10% realized vol, 45 DTE, 10% OTM -- an entirely realistic
# input combination for a calm-market SPY put) would never actually trade
# there in practice; exchanges enforce minimum tick/quote increments, and
# market makers don't quote genuinely sub-nickel deep-OTM options tightly.
# Computing a % return off a premium that's an artifact of the model having
# no floor (not a numerical bug in the BS formula itself -- it's correctly
# computing a vanishingly small but nonzero value) produces nonsense returns
# like "355,743x" when the underlying later moves and the option is repriced
# at a normal cent-scale premium. Found this the hard way (2026-07-23) when a
# tail-hedge backtest reported an average winning trade of +35,574,300%.
# Below this floor, a trade is treated as untradeable (net_return = NaN),
# not floored-and-kept -- flooring the price would still misstate the % move.
MIN_TRADABLE_PREMIUM = 0.05


def _below_floor(*premiums: float) -> bool:
    return any(not np.isfinite(p) or p < MIN_TRADABLE_PREMIUM for p in premiums)


def realized_vol(close: "np.ndarray | list[float]", window: int) -> "np.ndarray":
    """Annualized trailing realized volatility from daily closes, as a numpy
    array aligned to the input (NaN for the first `window` points)."""
    import pandas as pd
    prices = pd.Series(close)
    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(window).std() * np.sqrt(252)
    return vol.to_numpy()


def bs_call_price(spot: float, strike: float, years_to_expiry: float, vol: float, risk_free_rate: float = 0.04) -> float:
    if years_to_expiry <= 0:
        return max(spot - strike, 0.0)
    if vol <= 0 or not np.isfinite(vol):
        return max(spot - strike, 0.0)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * vol**2) * years_to_expiry) / (vol * np.sqrt(years_to_expiry))
    d2 = d1 - vol * np.sqrt(years_to_expiry)
    return spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * years_to_expiry) * norm.cdf(d2)


def bs_put_price(spot: float, strike: float, years_to_expiry: float, vol: float, risk_free_rate: float = 0.04) -> float:
    if years_to_expiry <= 0:
        return max(strike - spot, 0.0)
    if vol <= 0 or not np.isfinite(vol):
        return max(strike - spot, 0.0)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * vol**2) * years_to_expiry) / (vol * np.sqrt(years_to_expiry))
    d2 = d1 - vol * np.sqrt(years_to_expiry)
    return strike * np.exp(-risk_free_rate * years_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def synthetic_call_trade(
    entry_spot: float, entry_vol: float, exit_spot: float, exit_vol: float,
    dte_at_entry: int, days_held: int, strike_moneyness: float = 1.0,
    spread_frac_of_premium: float = 0.05, risk_free_rate: float = 0.04,
) -> dict:
    """Prices a call at entry (dte_at_entry days to expiry) and marks it to
    exit `days_held` calendar days later (sold, not exercised -- matching
    the original 30-day-thesis-for-a-10-day-signal framing). A bid/ask
    spread (as a fraction of premium, applied against the trader both ways)
    is charged on entry and exit, on top of the underpricing bias baked
    into using realized vol as the IV proxy."""
    strike = entry_spot * strike_moneyness
    entry_premium = bs_call_price(entry_spot, strike, dte_at_entry / 365, entry_vol, risk_free_rate)
    entry_fill = entry_premium * (1 + spread_frac_of_premium)  # pay more than mid to buy

    remaining_dte = max(dte_at_entry - days_held, 0)
    exit_premium = bs_call_price(exit_spot, strike, remaining_dte / 365, exit_vol, risk_free_rate)
    exit_fill = exit_premium * (1 - spread_frac_of_premium)  # receive less than mid to sell

    if _below_floor(entry_fill):
        return {"entry_premium": entry_fill, "exit_premium": exit_fill, "net_return": np.nan, "strike": strike}
    net_return = exit_fill / entry_fill - 1
    return {"entry_premium": entry_fill, "exit_premium": exit_fill, "net_return": net_return, "strike": strike}


def synthetic_call_debit_spread_trade(
    entry_spot: float, entry_vol: float, exit_spot: float, exit_vol: float,
    dte_at_entry: int, days_held: int, long_moneyness: float = 1.0, short_moneyness: float = 1.10,
    spread_frac_of_premium: float = 0.05, risk_free_rate: float = 0.04,
) -> dict:
    """Bull call debit spread: long a call at `long_moneyness`, short a call
    at `short_moneyness` (> long_moneyness), same expiry. Net debit = long
    premium - short premium; net_return is on that net debit. Spread cost is
    charged on BOTH legs, both directions (4 crossings total, matching a real
    2-leg spread's execution cost) -- if anything this understates real cost
    since it doesn't add a separate per-leg commission (see the sleeve-ladder
    doc's cost-realism section for where that's charged instead)."""
    long_strike = entry_spot * long_moneyness
    short_strike = entry_spot * short_moneyness

    long_entry = bs_call_price(entry_spot, long_strike, dte_at_entry / 365, entry_vol, risk_free_rate) * (1 + spread_frac_of_premium)
    short_entry = bs_call_price(entry_spot, short_strike, dte_at_entry / 365, entry_vol, risk_free_rate) * (1 - spread_frac_of_premium)
    net_debit = long_entry - short_entry

    remaining_dte = max(dte_at_entry - days_held, 0)
    long_exit = bs_call_price(exit_spot, long_strike, remaining_dte / 365, exit_vol, risk_free_rate) * (1 - spread_frac_of_premium)
    short_exit = bs_call_price(exit_spot, short_strike, remaining_dte / 365, exit_vol, risk_free_rate) * (1 + spread_frac_of_premium)
    net_credit = long_exit - short_exit

    if _below_floor(long_entry, net_debit):
        return {"net_debit": net_debit, "net_credit": net_credit, "net_return": np.nan}
    # A debit spread's max loss is capped at the net debit paid, by
    # construction (long the near strike, short the far strike -- you can
    # never owe more than you paid). When both exit legs are tiny (both
    # expiring far OTM), the entry/exit spread markup is applied with
    # OPPOSITE sign to the long vs. short leg, which can flip their
    # ordering (short_exit > long_exit) and produce net_credit < 0 --
    # a modeling artifact of the markup, not a real payoff a defined-risk
    # spread can produce. Clamp to the true economic floor rather than let
    # a few near-worthless-leg trades report losses of -1000%+.
    net_return = max(net_credit / net_debit - 1, -1.0)
    return {"net_debit": net_debit, "net_credit": net_credit, "net_return": net_return}


def synthetic_straddle_trade(
    entry_spot: float, entry_vol: float, exit_spot: float, exit_vol: float,
    dte_at_entry: int, days_held: int, moneyness: float = 1.0,
    spread_frac_of_premium: float = 0.05, risk_free_rate: float = 0.04,
) -> dict:
    """Long ATM (by default) straddle: call + put, same strike/expiry.
    Direction-agnostic -- profits from realized move exceeding what was
    priced in (entry_vol), loses to time decay if the underlying sits
    still. Spread charged on all 4 crossings (2 legs x entry/exit)."""
    strike = entry_spot * moneyness
    call_entry = bs_call_price(entry_spot, strike, dte_at_entry / 365, entry_vol, risk_free_rate) * (1 + spread_frac_of_premium)
    put_entry = bs_put_price(entry_spot, strike, dte_at_entry / 365, entry_vol, risk_free_rate) * (1 + spread_frac_of_premium)
    entry_premium = call_entry + put_entry

    remaining_dte = max(dte_at_entry - days_held, 0)
    call_exit = bs_call_price(exit_spot, strike, remaining_dte / 365, exit_vol, risk_free_rate) * (1 - spread_frac_of_premium)
    put_exit = bs_put_price(exit_spot, strike, remaining_dte / 365, exit_vol, risk_free_rate) * (1 - spread_frac_of_premium)
    exit_premium = call_exit + put_exit

    if _below_floor(entry_premium):
        return {"entry_premium": entry_premium, "exit_premium": exit_premium, "net_return": np.nan}
    return {"entry_premium": entry_premium, "exit_premium": exit_premium, "net_return": exit_premium / entry_premium - 1}


def synthetic_put_trade(
    entry_spot: float, entry_vol: float, exit_spot: float, exit_vol: float,
    dte_at_entry: int, days_held: int, strike_moneyness: float = 1.0,
    spread_frac_of_premium: float = 0.05, risk_free_rate: float = 0.04,
) -> dict:
    strike = entry_spot * strike_moneyness
    entry_premium = bs_put_price(entry_spot, strike, dte_at_entry / 365, entry_vol, risk_free_rate)
    entry_fill = entry_premium * (1 + spread_frac_of_premium)

    remaining_dte = max(dte_at_entry - days_held, 0)
    exit_premium = bs_put_price(exit_spot, strike, remaining_dte / 365, exit_vol, risk_free_rate)
    exit_fill = exit_premium * (1 - spread_frac_of_premium)

    if _below_floor(entry_fill):
        return {"entry_premium": entry_fill, "exit_premium": exit_fill, "net_return": np.nan, "strike": strike}
    net_return = exit_fill / entry_fill - 1
    return {"entry_premium": entry_fill, "exit_premium": exit_fill, "net_return": net_return, "strike": strike}
