"""Capability probe for ThetaData's STOCK data endpoints (separate from the
options subscription this project already has). Run this after any
subscription change to check whether tick-level equity order-flow work
(the equity analogue of BTC_ORDERFLOW_STUDY.md) has become possible.

Background: the $40/mo "Options Value" tier already purchased for this
project's options re-tests (EQUITY_OPTIONS_REAL_DATA_RETEST.md etc.) does
NOT include stock-side tick data. ThetaData gates stocks on their own
separate Free/Value/Standard/Pro ladder, independent of the options tier.
Confirmed directly (2026-07-24), not assumed:

  stock_history_eod        -> OK (works today)
  stock_list_symbols       -> OK (works today)
  stock_history_quote      -> PERMISSION_DENIED: "requiring a value subscription"
  stock_history_trade      -> PERMISSION_DENIED: "requiring a standard subscription"
  stock_history_trade_quote -> not individually tested; requires at least
                                what stock_history_trade needs (standard),
                                likely more given it combines both feeds

See EQUITY_ORDERFLOW_TICK_STUDY.md for the full writeup of why this blocks
an equity order-flow diagnostic and what's needed to unblock it.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.thetadata_pricing import get_client  # noqa: E402

PROBE_DATE = date(2024, 3, 4)
PROBE_START = date(2024, 3, 1)


def probe(symbol: str = "SPY") -> dict:
    client = get_client()
    results: dict[str, str] = {}

    checks = [
        ("stock_history_eod", lambda: client.stock_history_eod(symbol=symbol, start_date=PROBE_START, end_date=PROBE_DATE)),
        ("stock_list_symbols", lambda: client.stock_list_symbols()),
        ("stock_history_quote", lambda: client.stock_history_quote(symbol=symbol, date=PROBE_DATE)),
        ("stock_history_trade", lambda: client.stock_history_trade(symbol=symbol, date=PROBE_DATE)),
        ("stock_history_trade_quote", lambda: client.stock_history_trade_quote(symbol=symbol, date=PROBE_DATE)),
    ]
    for name, fn in checks:
        try:
            out = fn()
            n = len(out) if hasattr(out, "__len__") else "?"
            results[name] = f"OK ({n} rows)"
        except Exception as e:
            msg = str(e)
            detail_line = next((l for l in msg.splitlines() if "details" in l), msg.splitlines()[0] if msg else str(e))
            results[name] = f"FAIL: {detail_line.strip()}"
    return results


if __name__ == "__main__":
    for name, status in probe().items():
        print(f"{name:28s} {status}")
