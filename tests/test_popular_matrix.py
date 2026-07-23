from pathlib import Path
import pandas as pd
import yaml
from btc_trend_bot.popular_matrix import build_feature_frame, load_matrix_config, simulate_matrix

CFG=Path('config/settings_popular_matrix.yaml')

def synthetic(n=7000):
    ts=pd.date_range('2025-01-01',periods=n,freq='5min',tz='UTC')
    x=pd.Series(range(n),dtype=float)
    close=50000 + 0.6*x + 300*((x/150).map(__import__('math').sin))
    open_=close.shift(1).fillna(close.iloc[0])
    high=pd.concat([open_,close],axis=1).max(axis=1)+20
    low=pd.concat([open_,close],axis=1).min(axis=1)-20
    return pd.DataFrame({'timestamp':ts,'open':open_,'high':high,'low':low,'close':close,'volume':100+x*0})

def test_config_loads_unique_strategies():
    cfg=load_matrix_config(CFG)
    ids=[s['id'] for s in cfg['strategies']]
    assert len(ids)==len(set(ids))
    assert 'donchian_15m_atr' in ids

def test_features_are_causal_and_valid():
    cfg=load_matrix_config(CFG)
    f=build_feature_frame(synthetic(),cfg)
    assert f['feature_valid'].any()
    assert {'m15_donchian_high','m15_squeeze_release','h4_trend_bps','rsi2_5m'} <= set(f.columns)

def test_matrix_contains_every_strategy():
    cfg=load_matrix_config(CFG)
    f=build_feature_frame(synthetic(),cfg)
    result=simulate_matrix(f,cfg)
    assert set(result.summary)=={s['id'] for s in cfg['strategies']}

def test_next_open_execution_is_used():
    cfg=load_matrix_config(CFG)
    f=build_feature_frame(synthetic(),cfg)
    result=simulate_matrix(f,cfg)
    if not result.transactions.empty:
        tx=result.transactions.iloc[0]
        assert pd.Timestamp(tx['execution_timestamp']) > pd.Timestamp(tx['signal_timestamp'])

def test_live_order_modules_not_imported():
    text=Path('src/btc_trend_bot/popular_matrix.py').read_text()
    assert 'create_order' not in text
    assert 'SUPABASE_SECRET' not in text
