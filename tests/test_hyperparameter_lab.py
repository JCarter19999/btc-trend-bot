from __future__ import annotations

import random

from btc_trend_bot import hyperparameter_lab as h


def _cfg():
    return {
        "search_space": {
            "exit_mode": {"type": "choice", "values": ["breakeven_lock"]},
            "target_bps": {"type": "int", "low": 20, "high": 30, "step": 5},
            "stop_bps": [60, 65],
            "maximum_hold_bars_5m": [96, 144],
            "minimum_hold_bars_5m": [0],
            "lock_activation_bps": [20],
            "lock_bps": [0, 5],
            "activation_bps": [25],
            "trail_bps": [10],
        }
    }


def test_sample_trial_is_deterministic_and_prunes_irrelevant_fields():
    a = h.sample_trial(1, _cfg(), random.Random(7))
    b = h.sample_trial(1, _cfg(), random.Random(7))
    assert a == b
    assert a["exit_mode"] == "breakeven_lock"
    assert "activation_bps" not in a
    assert "trail_bps" not in a
    assert "lock_activation_bps" in a


def test_score_penalizes_too_few_trades():
    cfg = {"research": {"objective": {"minimum_trades": 8, "missing_trade_penalty": 2.0}}}
    base = {
        "gross_break_even_all_in_bps_per_side": 5,
        "average_realized_gross_bps": 10,
        "net_return_pct": 0,
        "max_drawdown": -0.02,
        "best_five_trade_gross_share": 0.5,
    }
    few = h._metric_score({**base, "round_trips_closed": 2}, cfg)
    enough = h._metric_score({**base, "round_trips_closed": 8}, cfg)
    assert enough > few
