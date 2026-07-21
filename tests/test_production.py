from decimal import Decimal

from btc_trend_bot.exchange import floor_decimal
from btc_trend_bot.production import decide


def test_floor_decimal():
    assert floor_decimal(Decimal("12.349"), Decimal("0.01")) == Decimal("12.34")


def test_buy_decision_respects_reserve_and_cap():
    d = decide(
        bar_timestamp="2026-01-01T00:00:00+00:00", close=Decimal("100000"),
        target=Decimal("1"), quote_available=Decimal("500"), base_available=Decimal("0"),
        allocation_cap=Decimal("0.98"), cash_reserve=Decimal("10"),
        base_increment=Decimal("0.00000001"), quote_increment=Decimal("0.01"),
        min_notional=Decimal("10"), rebalance_tolerance=Decimal("0.01"),
    )
    assert d.side == "buy"
    assert d.order_size == Decimal("490.00")


def test_flat_signal_sells_existing_btc():
    d = decide(
        bar_timestamp="2026-01-01T00:00:00+00:00", close=Decimal("100000"),
        target=Decimal("0"), quote_available=Decimal("0"), base_available=Decimal("0.005"),
        allocation_cap=Decimal("0.98"), cash_reserve=Decimal("10"),
        base_increment=Decimal("0.00000001"), quote_increment=Decimal("0.01"),
        min_notional=Decimal("10"), rebalance_tolerance=Decimal("0.01"),
    )
    assert d.side == "sell"
    assert d.order_size == Decimal("0.005")


def test_tolerance_avoids_churn():
    d = decide(
        bar_timestamp="2026-01-01T00:00:00+00:00", close=Decimal("100000"),
        target=Decimal("1"), quote_available=Decimal("10"), base_available=Decimal("0.0049"),
        allocation_cap=Decimal("1"), cash_reserve=Decimal("0"),
        base_increment=Decimal("0.00000001"), quote_increment=Decimal("0.01"),
        min_notional=Decimal("10"), rebalance_tolerance=Decimal("0.03"),
    )
    assert d.side == "none"
