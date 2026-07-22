from __future__ import annotations
import dataclasses
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import run_equity_real_data_walkforward as real

def frame(n=1200,seed=1):
    rng=np.random.default_rng(seed); idx=pd.bdate_range('2018-01-01',periods=n,tz='UTC')
    ret=rng.normal(.0003,.012,n); close=100*np.exp(np.cumsum(ret)); op=np.r_[close[0],close[:-1]]
    high=np.maximum(op,close)*(1+rng.uniform(0,.01,n)); low=np.minimum(op,close)*(1-rng.uniform(0,.01,n))
    return pd.DataFrame({'open':op,'high':high,'low':low,'close':close,'volume':rng.integers(1_000_000,5_000_000,n)},index=idx)

def config(**overrides):
    base=real.BacktestConfig(('AAPL','MSFT'),'SPY','2018-01-01',None,'1d',2500,0,10,300,100,100,10,10,1.35,2.15,0,True,.15,.35,4,8,25)
    return dataclasses.replace(base,**overrides) if overrides else base

def test_features_and_no_same_bar_entry():
    f=real.add_features(frame(),frame(seed=2)); cfg=config(); t=real.simulate_trade(f,100,cfg)
    assert t['entry_time']==f.index[101]
    assert set(real.FEATURES).issubset(f.columns)

def test_walk_forward_and_stoppage():
    cfg=config(); frames={'AAPL':frame(seed=1),'MSFT':frame(seed=2),'SPY':frame(seed=3)}
    c=real.build_candidates(frames,'SPY',cfg); c.signal_time=pd.to_datetime(c.signal_time,utc=True)
    folds,trades=real.walk_forward(c,cfg)
    assert len(folds)>=1
    assert (trades.predicted_return>=0).all()
    path,summary=real.simulate_capital(trades,cfg)
    assert summary['ending_equity']>=0
    assert len(path)==len(trades)

def test_shuffle_labels_is_reproducible_and_alters_fit():
    cfg=config(); frames={'AAPL':frame(seed=1),'MSFT':frame(seed=2),'SPY':frame(seed=3)}
    c=real.build_candidates(frames,'SPY',cfg); c.signal_time=pd.to_datetime(c.signal_time,utc=True)
    _,trades_normal=real.walk_forward(c,cfg)
    _,trades_shuffled_a=real.walk_forward(c,cfg,shuffle_labels=True,seed=7)
    _,trades_shuffled_b=real.walk_forward(c,cfg,shuffle_labels=True,seed=7)
    pd.testing.assert_frame_equal(trades_shuffled_a.reset_index(drop=True),trades_shuffled_b.reset_index(drop=True))
    assert trades_normal.predicted_return.mean()!=trades_shuffled_a.predicted_return.mean()

def test_random_and_simple_trend_selection_pick_one_per_signal_time():
    cfg=config(); frames={'AAPL':frame(seed=1),'MSFT':frame(seed=2),'SPY':frame(seed=3)}
    c=real.build_candidates(frames,'SPY',cfg); c.signal_time=pd.to_datetime(c.signal_time,utc=True)
    for mode in ('random','simple_trend'):
        folds,trades=real.walk_forward(c,cfg,selection=mode)
        assert len(folds)>=1
        assert trades.groupby('signal_time').size().max()==1
        assert trades.predicted_return.isna().all()

def test_random_selection_is_reproducible_per_seed():
    cfg=config(); frames={'AAPL':frame(seed=1),'MSFT':frame(seed=2),'SPY':frame(seed=3)}
    c=real.build_candidates(frames,'SPY',cfg); c.signal_time=pd.to_datetime(c.signal_time,utc=True)
    _,trades_a=real.walk_forward(c,cfg,selection='random',seed=3)
    _,trades_b=real.walk_forward(c,cfg,selection='random',seed=3)
    pd.testing.assert_frame_equal(trades_a.reset_index(drop=True),trades_b.reset_index(drop=True))

def test_single_split_forward_trains_once_tests_once():
    cfg=config(); frames={'AAPL':frame(seed=1),'MSFT':frame(seed=2),'SPY':frame(seed=3)}
    c=real.build_candidates(frames,'SPY',cfg); c.signal_time=pd.to_datetime(c.signal_time,utc=True)
    folds,trades=real.single_split_forward(c,cfg,'2018-01-01','2019-12-31','2020-06-01','2021-12-31')
    assert len(folds)==1
    assert (trades.signal_time>=pd.Timestamp('2020-06-01',tz='UTC')).all()
    assert (trades.signal_time<=pd.Timestamp('2021-12-31',tz='UTC')).all()
    assert trades.groupby('signal_time').size().max()==1

def test_single_split_forward_supports_selection_controls():
    cfg=config(); frames={'AAPL':frame(seed=1),'MSFT':frame(seed=2),'SPY':frame(seed=3)}
    c=real.build_candidates(frames,'SPY',cfg); c.signal_time=pd.to_datetime(c.signal_time,utc=True)
    _,trades=real.single_split_forward(c,cfg,'2018-01-01','2019-12-31','2020-06-01','2021-12-31',selection='random')
    assert trades.predicted_return.isna().all()

def test_position_fraction_limits_notional_and_dampens_compounding():
    trades=pd.DataFrame({'signal_time':pd.bdate_range('2018-01-01',periods=3,tz='UTC'),
                          'symbol':['AAPL']*3,'predicted_return':[1.0]*3,'net_return':[0.5,0.5,0.5]})
    full,_=real.simulate_capital(trades,config(position_fraction=1.0))
    quarter,_=real.simulate_capital(trades,config(position_fraction=0.25))
    assert quarter.iloc[0].notional==pytest.approx(2500*0.25)
    assert quarter.iloc[-1].ending_equity < full.iloc[-1].ending_equity

def test_drawdown_pause_releases_after_cooldown():
    # A drawdown_pause must be temporary: it should release after cooldown_trades
    # skipped candidates rather than freezing equity/peak (and therefore the
    # drawdown check) forever. position_fraction=1.0 isolates the pause/release
    # mechanism from position sizing.
    cfg=config(position_fraction=1.0)
    idx=pd.bdate_range('2018-01-01',periods=20,tz='UTC')
    returns=[-0.20]+[0.0]*19  # one big loss trips drawdown_pause, then flat candidates
    trades=pd.DataFrame({
        'signal_time':idx,'symbol':['AAPL']*20,'predicted_return':[1.0]*20,'net_return':returns,
    })
    path,summary=real.simulate_capital(trades,cfg)
    assert (path.loc[1:cfg.cooldown_trades,'skip_reason']=='drawdown_pause').all()
    assert path.loc[cfg.cooldown_trades+1,'trade_taken']
    assert summary['trades_taken']>1
