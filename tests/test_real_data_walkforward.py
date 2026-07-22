from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import run_equity_real_data_walkforward as real

def frame(n=1200,seed=1):
    rng=np.random.default_rng(seed); idx=pd.bdate_range('2018-01-01',periods=n,tz='UTC')
    ret=rng.normal(.0003,.012,n); close=100*np.exp(np.cumsum(ret)); op=np.r_[close[0],close[:-1]]
    high=np.maximum(op,close)*(1+rng.uniform(0,.01,n)); low=np.minimum(op,close)*(1-rng.uniform(0,.01,n))
    return pd.DataFrame({'open':op,'high':high,'low':low,'close':close,'volume':rng.integers(1_000_000,5_000_000,n)},index=idx)

def config():
    return real.BacktestConfig(('AAPL','MSFT'),'SPY','2018-01-01',None,'1d',2500,0,10,300,100,100,10,10,1.35,2.15,0,True,.15,.35,4,8,25)

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

def test_drawdown_pause_releases_after_cooldown():
    # A drawdown_pause must be temporary: it should release after cooldown_trades
    # skipped candidates rather than freezing equity/peak (and therefore the
    # drawdown check) forever.
    cfg=config()
    idx=pd.bdate_range('2018-01-01',periods=20,tz='UTC')
    returns=[-0.20]+[0.0]*19  # one big loss trips drawdown_pause, then flat candidates
    trades=pd.DataFrame({
        'signal_time':idx,'symbol':['AAPL']*20,'predicted_return':[1.0]*20,'net_return':returns,
    })
    path,summary=real.simulate_capital(trades,cfg)
    assert (path.loc[1:cfg.cooldown_trades,'skip_reason']=='drawdown_pause').all()
    assert path.loc[cfg.cooldown_trades+1,'trade_taken']
    assert summary['trades_taken']>1
