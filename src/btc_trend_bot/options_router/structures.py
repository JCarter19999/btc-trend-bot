"""Option structure definitions, strike selection, and payoff math (spec
section 5), priced from real SPXW 0DTE quotes cached in data/opra_spx/
(10:30 entry) and data/opra_spx_exit/ (noon exit).

Sign convention: `debit` is the net premium PAID to enter a structure
(sum of long-leg asks minus sum of short-leg bids). A negative `debit` is a
net credit received. `max_loss` and `max_profit` are always positive.
Return is reported as P&L / max_loss (risk-based, so debit and credit
structures are comparable on one scale) -- documented here because it's a
convention choice, not a law.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL_RE = re.compile(r"^(\S+)\s+(\d{6})([CP])(\d{8})$")

FILL_MODES = ("natural", "midpoint", "worse_25pct")


def parse_spxw_symbol(sym: str):
    """Parses any OSI-style OPRA symbol (root padded to 6 chars, e.g.
    'SPXW  260731C05000000' or 'XSP   260507P00684000') -- despite the name
    (kept for backward compat), not SPXW-specific."""
    m = SYMBOL_RE.match(sym.strip() if isinstance(sym, str) else "")
    if not m:
        return None
    _root, yymmdd, cp, strike8 = m.groups()
    expiry = pd.to_datetime(yymmdd, format="%y%m%d")
    strike = int(strike8) / 1000.0
    return expiry.normalize(), cp, strike


@lru_cache(maxsize=512)
def _parsed_file(parquet_path_str: str) -> pd.DataFrame:
    """Parse a raw OPRA snapshot's symbol strings into expiry/cp/strike once
    per file and cache it -- the regex parse over ~90k rows is the dominant
    cost, and load_chain is called many times per file (per C/P, per
    structure, per baseline) across a backtest."""
    df = pd.read_parquet(parquet_path_str)
    parsed = df["symbol"].apply(parse_spxw_symbol)
    df = df[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    df["expiry"] = [p[0] for p in parsed]
    df["cp"] = [p[1] for p in parsed]
    df["strike"] = [p[2] for p in parsed]
    return df[(df["ask_px_00"] > 0) & (df["bid_px_00"] > 0)]


def load_chain(parquet_path: Path, trade_date: pd.Timestamp, want_cp: str) -> pd.DataFrame:
    """Last real quote per strike in the snapshot window, 0DTE + one side only."""
    df = _parsed_file(str(parquet_path))
    df = df[(df["expiry"] == trade_date) & (df["cp"] == want_cp)]
    if df.empty:
        return df
    return df.sort_values("ts_event").groupby("strike", as_index=False).last()


def nearest_strike_row(chain: pd.DataFrame, target: float) -> dict | None:
    if chain is None or chain.empty:
        return None
    row = chain.iloc[(chain["strike"] - target).abs().argsort().iloc[0]]
    return {"strike": float(row["strike"]), "bid": float(row["bid_px_00"]), "ask": float(row["ask_px_00"])}


@dataclass
class Leg:
    cp: str          # 'C' or 'P'
    strike: float
    side: str         # 'long' or 'short'
    entry_price: float


@dataclass
class OptionStructure:
    name: str
    legs: list[Leg] = field(default_factory=list)
    debit: float = 0.0          # net premium paid; negative = net credit received
    max_profit: float | None = None   # None = uncapped (straddle/strangle upside)
    max_loss: float = 0.0
    width: float | None = None

    @property
    def credit_received(self) -> float:
        return max(0.0, -self.debit)


def _intrinsic(cp: str, strike: float, close: float) -> float:
    return max(close - strike, 0.0) if cp == "C" else max(strike - close, 0.0)


def _exit_value_from_legs(legs: list[Leg], per_leg_exit_price: dict) -> float | None:
    """per_leg_exit_price maps id(leg) -> price the leg could be closed at
    (bid for long, ask for short). Returns None if any leg's exit price is
    missing (can't value the full structure)."""
    total = 0.0
    for leg in legs:
        price = per_leg_exit_price.get(id(leg))
        if price is None:
            return None
        total += price if leg.side == "long" else -price
    return total


def payoff_at_close(structure: OptionStructure, close: float) -> float:
    total = 0.0
    for leg in structure.legs:
        intrinsic = _intrinsic(leg.cp, leg.strike, close)
        total += intrinsic if leg.side == "long" else -intrinsic
    return total


def pnl_and_return(structure: OptionStructure, exit_value: float) -> tuple[float, float]:
    pnl = exit_value - structure.debit
    ret = pnl / structure.max_loss if structure.max_loss else float("nan")
    return pnl, ret


def price_exit_from_chain(structure: OptionStructure, exit_chain_by_cp: dict[str, pd.DataFrame],
                           fill_mode: str = "natural") -> float | None:
    """Look up each leg's strike in the exit-snapshot chain(s) and value the
    structure at that snapshot (real bid/ask, not intrinsic)."""
    per_leg_price = {}
    for leg in structure.legs:
        chain = exit_chain_by_cp.get(leg.cp)
        if chain is None or chain.empty:
            return None
        match = chain[chain["strike"] == leg.strike]
        if match.empty:
            return None
        bid = float(match.iloc[0]["bid_px_00"])
        ask = float(match.iloc[0]["ask_px_00"])
        per_leg_price[id(leg)] = _fill_price(bid, ask, leg.side, fill_mode, closing=True)
    return _exit_value_from_legs(structure.legs, per_leg_price)


def _fill_price(bid: float, ask: float, side: str, fill_mode: str, closing: bool) -> float:
    mid = (bid + ask) / 2.0
    if fill_mode == "midpoint":
        return mid
    # "natural": buy at ask, sell at bid. Opening a long or closing a short = buy (ask).
    # Opening a short or closing a long = sell (bid).
    buying = (side == "long") != closing
    natural = ask if buying else bid
    if fill_mode == "natural":
        return natural
    if fill_mode == "worse_25pct":
        return mid + 0.25 * (natural - mid)
    raise ValueError(f"unknown fill_mode {fill_mode!r}")


def _leg(chain: pd.DataFrame, target_strike: float, cp: str, side: str, fill_mode: str) -> Leg | None:
    row = nearest_strike_row(chain, target_strike)
    if row is None:
        return None
    price = _fill_price(row["bid"], row["ask"], side, fill_mode, closing=False)
    return Leg(cp=cp, strike=row["strike"], side=side, entry_price=price)


def _finalize(name: str, legs: list[Leg], width: float | None, uncapped_upside: bool) -> OptionStructure | None:
    if any(leg is None for leg in legs):
        return None
    debit = sum(l.entry_price if l.side == "long" else -l.entry_price for l in legs)
    if uncapped_upside:
        max_loss = debit
        max_profit = None
    elif debit >= 0:
        max_loss = debit
        max_profit = (width - debit) if width is not None else None
    else:
        credit = -debit
        max_loss = (width - credit) if width is not None else float("nan")
        max_profit = credit
    return OptionStructure(name=name, legs=legs, debit=debit, max_profit=max_profit,
                            max_loss=max_loss, width=width)


def call_debit_spread(chain_c: pd.DataFrame, spot: float, width_points: float,
                       fill_mode: str = "natural") -> OptionStructure | None:
    long_leg = _leg(chain_c, spot, "C", "long", fill_mode)
    short_leg = _leg(chain_c, spot + width_points, "C", "short", fill_mode)
    if long_leg and short_leg and long_leg.strike == short_leg.strike:
        return None
    return _finalize("call_debit_spread", [long_leg, short_leg], abs((short_leg.strike - long_leg.strike)) if long_leg and short_leg else None, uncapped_upside=False)


def put_debit_spread(chain_p: pd.DataFrame, spot: float, width_points: float,
                      fill_mode: str = "natural") -> OptionStructure | None:
    long_leg = _leg(chain_p, spot, "P", "long", fill_mode)
    short_leg = _leg(chain_p, spot - width_points, "P", "short", fill_mode)
    if long_leg and short_leg and long_leg.strike == short_leg.strike:
        return None
    return _finalize("put_debit_spread", [long_leg, short_leg], abs((short_leg.strike - long_leg.strike)) if long_leg and short_leg else None, uncapped_upside=False)


def put_credit_spread(chain_p: pd.DataFrame, spot: float, short_offset_points: float,
                       width_points: float, fill_mode: str = "natural") -> OptionStructure | None:
    short_leg = _leg(chain_p, spot - short_offset_points, "P", "short", fill_mode)
    long_leg = _leg(chain_p, spot - short_offset_points - width_points, "P", "long", fill_mode)
    if long_leg and short_leg and long_leg.strike == short_leg.strike:
        return None
    return _finalize("put_credit_spread", [long_leg, short_leg], abs((short_leg.strike - long_leg.strike)) if long_leg and short_leg else None, uncapped_upside=False)


def call_credit_spread(chain_c: pd.DataFrame, spot: float, short_offset_points: float,
                        width_points: float, fill_mode: str = "natural") -> OptionStructure | None:
    short_leg = _leg(chain_c, spot + short_offset_points, "C", "short", fill_mode)
    long_leg = _leg(chain_c, spot + short_offset_points + width_points, "C", "long", fill_mode)
    if long_leg and short_leg and long_leg.strike == short_leg.strike:
        return None
    return _finalize("call_credit_spread", [long_leg, short_leg], abs((short_leg.strike - long_leg.strike)) if long_leg and short_leg else None, uncapped_upside=False)


def long_straddle(chain_c: pd.DataFrame, chain_p: pd.DataFrame, spot: float,
                   fill_mode: str = "natural") -> OptionStructure | None:
    call_leg = _leg(chain_c, spot, "C", "long", fill_mode)
    put_leg = _leg(chain_p, spot, "P", "long", fill_mode)
    return _finalize("long_straddle", [call_leg, put_leg], None, uncapped_upside=True)


def long_strangle(chain_c: pd.DataFrame, chain_p: pd.DataFrame, spot: float, width_points: float,
                   fill_mode: str = "natural") -> OptionStructure | None:
    call_leg = _leg(chain_c, spot + width_points, "C", "long", fill_mode)
    put_leg = _leg(chain_p, spot - width_points, "P", "long", fill_mode)
    return _finalize("long_strangle", [call_leg, put_leg], None, uncapped_upside=True)


BUILDERS = {
    "call_debit_spread": lambda chains, spot, cfg, fill_mode: call_debit_spread(
        chains["C"], spot, cfg["structures"]["strike_distance_points"], fill_mode),
    "put_debit_spread": lambda chains, spot, cfg, fill_mode: put_debit_spread(
        chains["P"], spot, cfg["structures"]["strike_distance_points"], fill_mode),
    "put_credit_spread": lambda chains, spot, cfg, fill_mode: put_credit_spread(
        chains["P"], spot, cfg["structures"]["strike_distance_points"],
        cfg["structures"]["credit_spread_width_points"], fill_mode),
    "call_credit_spread": lambda chains, spot, cfg, fill_mode: call_credit_spread(
        chains["C"], spot, cfg["structures"]["strike_distance_points"],
        cfg["structures"]["credit_spread_width_points"], fill_mode),
    "long_straddle": lambda chains, spot, cfg, fill_mode: long_straddle(
        chains["C"], chains["P"], spot, fill_mode),
    "long_strangle": lambda chains, spot, cfg, fill_mode: long_strangle(
        chains["C"], chains["P"], spot, cfg["structures"]["strike_distance_points"], fill_mode),
}


def build_structure(kind: str, chains: dict[str, pd.DataFrame], spot: float, cfg: dict,
                     fill_mode: str = "natural") -> OptionStructure | None:
    builder = BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"unknown structure kind {kind!r}")
    return builder(chains, spot, cfg, fill_mode)
