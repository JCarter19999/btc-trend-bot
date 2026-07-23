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

    if entry_fill <= 0:
        return {"entry_premium": entry_fill, "exit_premium": exit_fill, "net_return": np.nan, "strike": strike}
    net_return = exit_fill / entry_fill - 1
    return {"entry_premium": entry_fill, "exit_premium": exit_fill, "net_return": net_return, "strike": strike}


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

    if entry_fill <= 0:
        return {"entry_premium": entry_fill, "exit_premium": exit_fill, "net_return": np.nan, "strike": strike}
    net_return = exit_fill / entry_fill - 1
    return {"entry_premium": entry_fill, "exit_premium": exit_fill, "net_return": net_return, "strike": strike}
