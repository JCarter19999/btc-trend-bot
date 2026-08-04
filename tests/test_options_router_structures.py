"""Structure payoff math tests (spec section 5) -- hand-computed P&L against
a synthetic chain, no real files needed."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.options_router.structures import (
    call_debit_spread, payoff_at_close, pnl_and_return, price_exit_from_chain, put_credit_spread,
)


def chain(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["strike", "bid_px_00", "ask_px_00"])


def test_call_debit_spread_max_profit_at_or_above_short_strike():
    entry_c = chain([(5000, 10.0, 10.5), (5010, 4.6, 5.0)])
    s = call_debit_spread(entry_c, spot=5000, width_points=10)
    assert s is not None
    assert s.debit == pytest.approx(10.5 - 4.6)
    assert s.max_loss == pytest.approx(s.debit)
    assert s.max_profit == pytest.approx(10 - s.debit)

    payoff = payoff_at_close(s, close=5010)
    pnl, ret = pnl_and_return(s, payoff)
    assert pnl == pytest.approx(s.max_profit)
    assert ret == pytest.approx(s.max_profit / s.max_loss)


def test_call_debit_spread_max_loss_at_or_below_long_strike():
    entry_c = chain([(5000, 10.0, 10.5), (5010, 4.6, 5.0)])
    s = call_debit_spread(entry_c, spot=5000, width_points=10)
    payoff = payoff_at_close(s, close=4990)
    pnl, ret = pnl_and_return(s, payoff)
    assert pnl == pytest.approx(-s.debit)
    assert ret == pytest.approx(-1.0)


def test_call_debit_spread_exit_at_noon_uses_real_bid_ask():
    entry_c = chain([(5000, 10.0, 10.5), (5010, 4.6, 5.0)])
    s = call_debit_spread(entry_c, spot=5000, width_points=10)
    exit_c = chain([(5000, 9.0, 9.5), (5010, 4.0, 4.5)])
    exit_value = price_exit_from_chain(s, {"C": exit_c, "P": pd.DataFrame()})
    # sell long (bid=9.0) - buy back short (ask=4.5)
    assert exit_value == pytest.approx(9.0 - 4.5)
    pnl, ret = pnl_and_return(s, exit_value)
    assert pnl == pytest.approx((9.0 - 4.5) - s.debit)


def test_put_credit_spread_is_a_net_credit_with_capped_loss():
    entry_p = chain([(4950, 3.0, 3.3), (4925, 1.0, 1.2)])
    s = put_credit_spread(entry_p, spot=5000, short_offset_points=50, width_points=25)
    assert s is not None
    assert s.debit < 0  # net credit received
    assert s.credit_received == pytest.approx(-s.debit)
    assert s.max_profit == pytest.approx(s.credit_received)
    assert s.max_loss == pytest.approx(25 - s.credit_received)

    # close above the short strike -> both legs expire worthless -> max profit
    payoff = payoff_at_close(s, close=5000)
    pnl, ret = pnl_and_return(s, payoff)
    assert pnl == pytest.approx(s.max_profit)

    # close below the long (protective) strike -> max loss
    payoff_loss = payoff_at_close(s, close=4900)
    pnl_loss, ret_loss = pnl_and_return(s, payoff_loss)
    assert pnl_loss == pytest.approx(-s.max_loss)
    assert ret_loss == pytest.approx(-1.0)


def test_missing_strike_returns_none():
    entry_c = chain([(5000, 10.0, 10.5)])  # only one strike available
    s = call_debit_spread(entry_c, spot=5000, width_points=1000)
    assert s is None
