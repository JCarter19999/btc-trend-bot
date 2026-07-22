from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from run_equity_regime_stress_final import SafetyConfig, simulate_with_safety
from run_equity_capital_constrained_overlay import CapitalConfig

def _trades(returns):
    return pd.DataFrame([{'signal_time':f'2026-01-{i+1:02d}T15:00:00Z','symbol':'AAPL','net_return':r,
        'option_overlay_eligible':False,'entry_fill':2.0,'realized_option_return':0.0,
        'return_26':0.02,'ema_slope_atr':0.5} for i,r in enumerate(returns)])

def test_loss_streak_triggers_cooldown():
    path, summary=simulate_with_safety(_trades([-0.1]*8), CapitalConfig(1000,0,0.3,'cash_safe'),
        SafetyConfig(consecutive_loss_limit=2,cooldown_trades=3,drawdown_pause=0.99,hard_shutdown_drawdown=0.99))
    assert summary['trades_skipped_by_safety'] >= 3

def test_hard_drawdown_shutdown_prevents_further_trades():
    path, summary=simulate_with_safety(_trades([-0.4,-0.4,0.5]), CapitalConfig(1000,0,0.3,'cash_safe'),
        SafetyConfig(consecutive_loss_limit=99,drawdown_pause=0.99,hard_shutdown_drawdown=0.30))
    assert summary['hard_stopped']
    assert (path.skip_reason == 'hard_shutdown').any()

def test_bear_regime_gate_can_skip_trade():
    trades=_trades([0.1]); trades['return_26']=-0.05; trades['ema_slope_atr']=-1.0
    path, summary=simulate_with_safety(trades, CapitalConfig(1000,0,0.3,'cash_safe'), SafetyConfig())
    assert summary['trades_taken']==0
    assert path.iloc[0].skip_reason=='regime_gate'
