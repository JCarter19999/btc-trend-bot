from datetime import datetime, timezone
import pandas as pd
from btc_trend_bot.v1.config import load_v1_config
from btc_trend_bot.v1.labeling import mature_candidate
from btc_trend_bot.v1.retraining import should_retrain
from btc_trend_bot.v1.risk import RiskEngine, RiskState
from btc_trend_bot.v1.types import Candidate

def test_frozen_config():
    cfg = load_v1_config("config/v1.yaml")
    assert cfg["system"]["strategy_version"] == "selective_long_5m_v1"
    assert cfg["exit"]["collision_policy"] == "stop_first"

def test_same_bar_collision_is_stop_first():
    candidate = Candidate("c", datetime(2026, 1, 1, tzinfo=timezone.utc), "selective_long_5m_v1", "BTCUSDT", 100, 2, 98, 104, 144, {})
    future = pd.DataFrame([{"open_time": pd.Timestamp("2026-01-01 00:05Z"), "open": 100, "high": 105, "low": 97, "close": 103}])
    result = mature_candidate(candidate, future, {"fee_bps_per_side": 0, "spread_bps_per_side": 0, "slippage_bps_per_side": 0, "latency_bps_round_trip": 0})
    assert result["exit_reason"] == "stop_loss"
    assert result["ambiguous_same_bar"] is True
    assert result["exit_price"] == 98

def test_retraining_hybrid_trigger():
    cfg = {"weekday_utc": 6, "minimum_new_matured_candidates": 50}
    sunday = datetime(2026, 8, 16, tzinfo=timezone.utc)
    assert should_retrain(sunday, 50, cfg)[0]
    assert not should_retrain(sunday, 49, cfg)[0]

def test_risk_engine_rejects_spread():
    cfg = load_v1_config("config/v1.yaml")
    decision = RiskEngine(cfg["risk"]).evaluate(RiskState(500, 500, 500, 500), 50, 11, 0, 0)
    assert not decision.approved
    assert decision.reason_code == "SPREAD_OK"
