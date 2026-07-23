from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "run_equity_gap_walkforward_experiment.py"
spec = importlib.util.spec_from_file_location("equity_gap", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def _frame(next_open: float) -> pd.DataFrame:
    times = pd.date_range("2026-01-02 15:00", periods=4, freq="30min", tz="UTC")
    return pd.DataFrame({
        "open_time": times,
        "open": [100.0, 100.0, next_open, next_open],
        "high": [101.0, 101.0, max(next_open, 101.0), max(next_open, 101.0)],
        "low": [99.0, 99.0, min(next_open, 99.0), min(next_open, 99.0)],
        "close": [100.0, 100.0, next_open, next_open],
        "atr": [1.0, 1.0, 1.0, 1.0],
        "bar_in_session": [7, 8, 0, 1],
    })


def test_gap_through_stop_fills_at_open() -> None:
    frame = _frame(96.0)
    result = module.simulate_trade(frame, 0, max_bars=4)
    assert result["exit_reason"] == "gap_through_stop"
    assert result["exit_price"] == 96.0
    assert result["gap_through_stop"] is True


def test_gap_through_target_fills_at_open() -> None:
    frame = _frame(104.0)
    result = module.simulate_trade(frame, 0, max_bars=4)
    assert result["exit_reason"] == "gap_through_target"
    assert result["exit_price"] == 104.0
    assert result["gap_through_target"] is True
