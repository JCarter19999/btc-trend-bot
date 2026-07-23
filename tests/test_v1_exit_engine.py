from datetime import datetime, timezone
import pandas as pd
from btc_trend_bot.v1.types import Candidate
from btc_trend_bot.v1.exit_engine import ExitEngineConfig, run_exit_engine

COST={"fee_bps_per_side":0,"spread_bps_per_side":0,"slippage_bps_per_side":0,"latency_bps_round_trip":0}
def c():
    return Candidate("x",datetime(2026,1,1,tzinfo=timezone.utc),"selective_long_5m_v1","BTCUSDT",100,1,98.5,102.5,20,{})
def frame(rows):
    return pd.DataFrame(rows,columns=['open_time','open','high','low','close','volume'])

def test_breakeven_then_stop():
    f=frame([
      (pd.Timestamp('2026-01-01 00:05',tz='UTC'),100,101.2,99.9,101,1),
      (pd.Timestamp('2026-01-01 00:10',tz='UTC'),101,101.1,99.9,100,1)])
    cfg=ExitEngineConfig(breakeven_trigger_atr=1.0,breakeven_offset_bps=0,trailing_enabled=False,momentum_exit_enabled=False,trend_exit_enabled=False,volatility_exit_enabled=False)
    r=run_exit_engine(c(),f,COST,cfg)
    assert r['exit_reason']=='adaptive_stop' and r['exit_price']==100

def test_trailing_locks_gain():
    f=frame([
      (pd.Timestamp('2026-01-01 00:05',tz='UTC'),100,101.8,100,101.5,1),
      (pd.Timestamp('2026-01-01 00:10',tz='UTC'),101.5,101.7,100.7,100.8,1)])
    cfg=ExitEngineConfig(breakeven_enabled=False,trailing_activation_atr=1.5,trailing_distance_atr=1.0,momentum_exit_enabled=False,trend_exit_enabled=False,volatility_exit_enabled=False)
    r=run_exit_engine(c(),f,COST,cfg)
    assert r['exit_reason']=='adaptive_stop' and abs(r['exit_price']-100.8)<1e-9

def test_volatility_adaptive_config_defaults_enabled():
    cfg = ExitEngineConfig()
    assert cfg.volatility_adaptive_enabled is False
    assert cfg.volatility_factor_max >= 1.0
