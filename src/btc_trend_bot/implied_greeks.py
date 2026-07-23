"""Implied volatility and Greeks computed FROM REAL MARKET PRICES (ThetaData
bid/ask), not assumed from realized vol -- the inverse of options_pricing.py's
approach. Given a real quoted price, solve for the volatility Black-Scholes
would need to produce that price (this is literally the industry-standard
definition of "implied volatility" -- inverting the pricing formula against
an observed market price), then derive delta/theta/vega analytically from
that solved vol. This stays within the $40/mo Value tier (no Greeks
endpoint needed) rather than requiring the $80/mo Standard tier's direct
Greeks feed -- a deliberate choice Joey made when this tier gap was found,
not a shortcut: solving for IV from a real traded price is standard
practice, not an approximation like the realized-vol proxy this project
used everywhere before real data was available.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .options_pricing import bs_call_price, bs_put_price


def implied_volatility(market_price: float, spot: float, strike: float, years_to_expiry: float,
                        right: str, risk_free_rate: float = 0.04) -> float | None:
    """Solves for the volatility that makes Black-Scholes match a real
    quoted price. Returns None if no solution exists in a reasonable range
    (e.g. the price is below intrinsic value or otherwise not solvable --
    happens near expiry / for illiquid deep ITM contracts; treat as
    unavailable, not zero)."""
    if years_to_expiry <= 0 or market_price <= 0:
        return None
    price_fn = bs_call_price if right.lower() == "call" else bs_put_price
    intrinsic = max(spot - strike, 0.0) if right.lower() == "call" else max(strike - spot, 0.0)
    if market_price < intrinsic:
        return None  # below intrinsic value -- not solvable, likely a stale/bad quote

    def diff(vol: float) -> float:
        return price_fn(spot, strike, years_to_expiry, vol, risk_free_rate) - market_price

    try:
        # 0.1% to 500% annualized vol brackets every realistic real-market case
        return brentq(diff, 0.001, 5.0, xtol=1e-6)
    except ValueError:
        return None  # diff() doesn't change sign in this range -- no solution


def bs_delta(spot: float, strike: float, years_to_expiry: float, vol: float, right: str,
             risk_free_rate: float = 0.04) -> float | None:
    if years_to_expiry <= 0 or vol <= 0:
        return None
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * vol**2) * years_to_expiry) / (vol * np.sqrt(years_to_expiry))
    return float(norm.cdf(d1)) if right.lower() == "call" else float(norm.cdf(d1) - 1)


def bs_vega(spot: float, strike: float, years_to_expiry: float, vol: float,
            risk_free_rate: float = 0.04) -> float | None:
    """Change in option price per 1.00 (100 percentage points) change in
    vol -- caller multiplies by the actual vol change (e.g. 0.05 for a
    5-point IV move) to get a dollar contribution."""
    if years_to_expiry <= 0 or vol <= 0:
        return None
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * vol**2) * years_to_expiry) / (vol * np.sqrt(years_to_expiry))
    return float(spot * norm.pdf(d1) * np.sqrt(years_to_expiry))


def bs_theta_per_day(spot: float, strike: float, years_to_expiry: float, vol: float, right: str,
                      risk_free_rate: float = 0.04) -> float | None:
    """Change in option price per calendar day of time decay (already
    divided by 365 -- caller multiplies by days_held directly)."""
    if years_to_expiry <= 0 or vol <= 0:
        return None
    sqrt_t = np.sqrt(years_to_expiry)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * vol**2) * years_to_expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    term1 = -(spot * norm.pdf(d1) * vol) / (2 * sqrt_t)
    if right.lower() == "call":
        term2 = -risk_free_rate * strike * np.exp(-risk_free_rate * years_to_expiry) * norm.cdf(d2)
    else:
        term2 = risk_free_rate * strike * np.exp(-risk_free_rate * years_to_expiry) * norm.cdf(-d2)
    return float((term1 + term2) / 365.0)


def decompose_option_pnl(entry_price: float, exit_price: float, entry_spot: float, exit_spot: float,
                          strike: float, entry_dte_years: float, exit_dte_years: float, right: str,
                          days_held: int, risk_free_rate: float = 0.04) -> dict | None:
    """First-order (delta/vega/theta) P&L attribution for a single option
    leg, anchored at entry Greeks (the standard convention -- decompose
    the move using the sensitivities you actually had ON, not
    hindsight-adjusted ones). residual/gamma bundles everything a
    first-order decomposition can't explain (gamma convexity, higher-order
    cross terms, and any real-market deviation from Black-Scholes itself)
    -- reported explicitly rather than silently absorbed into one of the
    other buckets, so it's visible how much of the P&L this simple model
    actually explains."""
    entry_iv = implied_volatility(entry_price, entry_spot, strike, entry_dte_years, right, risk_free_rate)
    exit_iv = implied_volatility(exit_price, exit_spot, strike, max(exit_dte_years, 1e-6), right, risk_free_rate)
    if entry_iv is None or exit_iv is None:
        return None

    delta = bs_delta(entry_spot, strike, entry_dte_years, entry_iv, right, risk_free_rate)
    vega = bs_vega(entry_spot, strike, entry_dte_years, entry_iv, risk_free_rate)
    theta = bs_theta_per_day(entry_spot, strike, entry_dte_years, entry_iv, right, risk_free_rate)
    if delta is None or vega is None or theta is None:
        return None

    underlying_move_pnl = delta * (exit_spot - entry_spot)
    iv_change_pnl = vega * (exit_iv - entry_iv)
    theta_pnl = theta * days_held
    total_pnl = exit_price - entry_price
    residual_pnl = total_pnl - underlying_move_pnl - iv_change_pnl - theta_pnl

    return {
        "entry_iv": entry_iv, "exit_iv": exit_iv, "iv_change": exit_iv - entry_iv,
        "entry_delta": delta, "entry_vega": vega, "entry_theta_per_day": theta,
        "underlying_move_pnl": underlying_move_pnl, "iv_change_pnl": iv_change_pnl,
        "theta_pnl": theta_pnl, "residual_gamma_pnl": residual_pnl, "total_pnl": total_pnl,
    }
