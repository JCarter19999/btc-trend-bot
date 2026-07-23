from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_equity_capital_constrained_overlay import (
    CapitalConfig,
    affordable_contracts,
    option_round_trip_pnl,
    simulate_capital_path,
)


def test_affordability_uses_whole_contracts_and_open_fee():
    assert affordable_contracts(1000, 3.0, 0.30) == 0
    assert affordable_contracts(1005, 3.0, 0.30) == 1


def test_option_pnl_includes_schwab_round_trip_contract_fees():
    # $200 premium with a 10% gross gain = $20, less $1.30 round trip.
    assert option_round_trip_pnl(2.0, 0.10, 1) == 18.70


def test_cash_safe_route_never_exceeds_equity_at_entry():
    trades = pd.DataFrame([{
        "signal_time": "2026-01-05T15:00:00Z",
        "symbol": "AAPL",
        "net_return": 0.10,
        "option_overlay_eligible": True,
        "entry_fill": 1.0,
        "realized_option_return": 0.20,
    }])
    path, _ = simulate_capital_path(trades, CapitalConfig(500, 0, 0.30, "cash_safe"))
    assert path.iloc[0].stock_notional + path.iloc[0].option_open_cost <= 500
    assert path.iloc[0].option_contracts == 1


def test_small_account_runs_stock_but_skips_unaffordable_option():
    trades = pd.DataFrame([{
        "signal_time": "2026-01-05T15:00:00Z",
        "symbol": "AAPL",
        "net_return": 0.10,
        "option_overlay_eligible": True,
        "entry_fill": 2.0,
        "realized_option_return": 0.20,
    }])
    path, summary = simulate_capital_path(trades, CapitalConfig(50, 0, 0.30, "cash_safe"))
    assert path.iloc[0].stock_notional == 50
    assert path.iloc[0].option_contracts == 0
    assert summary["option_affordable_signals"] == 0
