from __future__ import annotations
import sys, math, json, copy, argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from btc_trend_bot.v1.features import build_features, FEATURE_COLUMNS
from btc_trend_bot.v1.strategy import generate_candidates, candidate_mask
from btc_trend_bot.v1.labeling import mature_candidate, round_trip_cost_return
from btc_trend_bot.v1.exit_engine import ExitEngineConfig, run_exit_engine

CFG=yaml.safe_load((ROOT/'config/v1.yaml').read_text())
parser=argparse.ArgumentParser(description='Compare fixed and adaptive exit engines on synthetic candles.')
parser.add_argument('--allocation', type=float, default=None, help='Research allocation fraction (0 < x <= 1). Defaults to config risk allocation.')
parser.add_argument('--output', default='outputs/exit_engine_experiments', help='Output directory, relative to repo root unless absolute.')
ARGS=parser.parse_args()
if ARGS.allocation is not None and not (0 < ARGS.allocation <= 1):
    parser.error('--allocation must be in (0, 1].')
OUT=Path(ARGS.output)
if not OUT.is_absolute(): OUT=ROOT/OUT
OUT.mkdir(parents=True,exist_ok=True)


def base_candles(n=160_000, seed=401):
    rng=np.random.default_rng(seed)
    ts=pd.date_range('2022-01-01',periods=n,freq='5min',tz='UTC')
    prices=np.empty(n+1); prices[0]=30000.0; rows=[]
    for i in range(n):
        cyc=0.00010+0.000035*math.sin(i/7000)
        sign=1 if i%2==0 else -1
        r=cyc+sign*0.000035+rng.normal(0,0.000018)
        o=prices[i]; c=o*(1+r); wick=abs(r)*0.35+0.00120
        h=max(o,c)*(1+wick); l=min(o,c)*(1-wick)
        v=100*(1+0.08*math.sin(i/1000))+rng.lognormal(0,0.08)
        rows.append((ts[i],o,h,l,c,v)); prices[i+1]=c
    return pd.DataFrame(rows,columns=['open_time','open','high','low','close','volume'])


def overwrite_bar(df,i,ret,vol_mult,body_frac=0.75):
    o=float(df.at[i,'open']); c=o*(1+ret); body=abs(c-o)
    total=body/max(body_frac,1e-6); extra=max(total-body,o*0.00005)
    df.at[i,'close']=c; df.at[i,'high']=max(o,c)+extra/2; df.at[i,'low']=min(o,c)-extra/2
    local=float(df.loc[max(0,i-48):i-1,'volume'].mean()) if i>0 else 100
    df.at[i,'volume']=local*vol_mult
    if i+1<len(df):
        ratio=c/float(df.at[i+1,'open'])
        for col in ['open','high','low','close']: df.at[i+1,col]=float(df.at[i+1,col])*ratio


def inject_events(df,seed=501,spacing=330,start=500):
    rng=np.random.default_rng(seed); events=[]
    delays=np.array([2,8,36,180]) # 10m,40m,3h,15h resolution
    probs=np.array([0.25,0.25,0.25,0.25])
    for j,end in enumerate(range(start,len(df)-310,spacing)):
        quality=bool(rng.random()<0.5)
        p_target=0.80 if quality else 0.20
        target=bool(rng.random()<p_target)
        delay=int(rng.choice(delays,p=probs))
        overwrite_bar(df,end-3,-0.00015,0.85,0.60)
        if quality:
            rets=[0.00100,0.00110,0.00120]; vols=[1.65,1.85,2.15]; bodies=[0.72,0.80,0.88]
        else:
            rets=[0.00076,0.00080,0.00086]; vols=[1.18,1.24,1.30]; bodies=[0.56,0.60,0.64]
        for k,(r,v,b) in enumerate(zip(rets,vols,bodies)): overwrite_bar(df,end-2+k,r,v,b)
        events.append({'event_index':j,'candidate_bar':end,'quality':int(quality),'target_planted':int(target),'resolution_delay_bars':delay})
    return pd.DataFrame(events)


def patch_delayed_paths(df,events):
    feat=build_features(df,CFG); mask=candidate_mask(feat,CFG); kept=[]
    rng=np.random.default_rng(777)
    for rec in events.to_dict('records'):
        idx=int(rec['candidate_bar'])
        if idx>=len(feat) or not bool(mask.iloc[idx]): continue
        row=feat.iloc[idx]; entry=float(row.close); atr=float(row.atr)
        stop=entry-CFG['exit']['stop_atr_multiple']*atr; target=entry+CFG['exit']['target_atr_multiple']*atr
        delay=int(rec['resolution_delay_bars']); is_target=bool(rec['target_planted'])
        end_price=target if is_target else stop
        # Gradual path remains inside barriers until designated resolution.
        prev=entry
        for k in range(1,delay+1):
            i=idx+k
            if i>=len(df): break
            frac=k/delay
            if k<delay:
                desired=entry+(end_price-entry)*(0.72*frac)
                noise=rng.normal(0,atr*0.025)
                c=float(np.clip(desired+noise, stop+0.10*atr, target-0.10*atr))
                o=prev
                h=min(max(o,c)+0.04*atr,target-0.04*atr)
                l=max(min(o,c)-0.04*atr,stop+0.04*atr)
            else:
                o=prev
                if is_target:
                    c=target-0.03*atr; h=target+0.01*atr; l=max(min(o,c)-0.03*atr,stop+0.04*atr)
                else:
                    c=stop+0.03*atr; l=stop-0.01*atr; h=min(max(o,c)+0.03*atr,target-0.04*atr)
            df.at[i,'open']=o; df.at[i,'close']=c; df.at[i,'high']=max(h,o,c); df.at[i,'low']=min(l,o,c)
            df.at[i,'volume']=float(df.loc[max(0,i-48):i-1,'volume'].mean())*(1.05+0.15*quality if (quality:=bool(rec['quality'])) else 1.03)
            prev=c
        # Restore continuity after episode.
        j=idx+delay+1
        if j<len(df):
            ratio=prev/float(df.at[j,'open'])
            for col in ['open','high','low','close']: df.at[j,col]=float(df.at[j,col])*ratio
        kept.append(rec)
    return pd.DataFrame(kept)


def make_dataset(df,events,profile):
    feat=build_features(df,CFG); candidates=generate_candidates(feat,CFG)
    event_map={pd.Timestamp(df.at[int(r.candidate_bar),'open_time']):r for _,r in events.iterrows()}
    rows=[]
    for c in candidates:
        ev=event_map.get(pd.Timestamp(c.timestamp))
        if ev is None: continue
        c2=replace(c,max_hold_bars=int(profile['max_hold_bars']))
        if profile['engine']=='fixed':
            m=mature_candidate(c2,df,CFG['costs'])
            if m.get('label_status')=='MATURED':
                m['bars_held']=(pd.Timestamp(m['maturity_timestamp'])-pd.Timestamp(m['timestamp'])).total_seconds()/300
                m['mfe_return']=np.nan; m['mae_return']=np.nan
        else:
            ec=ExitEngineConfig(max_hold_bars=int(profile['max_hold_bars']), **profile['params'])
            m=run_exit_engine(c2,df,CFG['costs'],ec)
        if m.get('label_status')!='MATURED': continue
        flat={**c2.features,**{k:v for k,v in m.items() if k not in ('features','metadata')}}
        flat['planted_quality']=int(ev.quality); flat['planted_target']=int(ev.target_planted); flat['resolution_delay_bars']=int(ev.resolution_delay_bars)
        rows.append(flat)
    return pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)


def evaluate(data,name):
    n=len(data); a=int(n*.6); b=int(n*.8); train=data.iloc[:a]; cal=data.iloc[a:b]; test=data.iloc[b:]
    model=Pipeline([('scale',StandardScaler()),('reg',Ridge(alpha=10.0))])
    model.fit(train[FEATURE_COLUMNS],train.net_return.astype(float))
    raw_cal=model.predict(cal[FEATURE_COLUMNS]); raw_test=model.predict(test[FEATURE_COLUMNS])
    A=np.c_[np.ones(len(raw_cal)),raw_cal]; beta=np.linalg.lstsq(A,cal.net_return.to_numpy(float),rcond=None)[0]
    pred_net=beta[0]+beta[1]*raw_test; accept=pred_net>(CFG['model']['safety_margin_bps']/10000)
    net=test.net_return.to_numpy(float); alloc=float(ARGS.allocation if ARGS.allocation is not None else CFG['risk']['max_allocation_fraction'])
    def pf(r):
        if len(r)==0:return 0.0,0.0
        eq=np.cumprod(1+alloc*r); peak=np.maximum.accumulate(np.r_[1.0,eq]); dd=(peak[1:]-eq)/peak[1:]
        return float(eq[-1]-1),float(dd.max())
    ur,ud=pf(net); fr,fd=pf(net[accept])
    out={'profile':name,'allocation_fraction':alloc,'candidates':n,'test_candidates':len(test),'accept_count':int(accept.sum()),'accept_rate':float(accept.mean()),
         'unfiltered_mean_net_bps':float(net.mean()*10000),'filtered_mean_net_bps':float(net[accept].mean()*10000) if accept.any() else None,
         'unfiltered_portfolio_return':ur,'filtered_portfolio_return':fr,'unfiltered_max_drawdown':ud,'filtered_max_drawdown':fd,
         'positive_rate':float(test.positive_net.mean()),'target_rate':float(test.target_hit_first.mean()),
         'time_exit_rate':float((test.exit_reason=='time_exit').mean()),'median_hold_bars':float(test.bars_held.median()),
         'accepted_good_fraction':float(test.loc[accept,'planted_quality'].mean()) if accept.any() else None,
         'mean_mfe_bps':float(test.mfe_return.mean()*10000) if 'mfe_return' in test and test.mfe_return.notna().any() else None,
         'mean_mae_bps':float(test.mae_return.mean()*10000) if 'mae_return' in test and test.mae_return.notna().any() else None,
         'exit_reasons':test.exit_reason.value_counts().to_dict()}
    pred=test[['timestamp','net_return','positive_net','target_hit_first','exit_reason','planted_quality','resolution_delay_bars','bars_held']].copy()
    pred['predicted_net_return']=pred_net; pred['accepted']=accept
    return out,pred


def evaluate_buy_and_hold(df):
    """Buy BTC at the start of the untouched final 20% candle block and hold to the end."""
    split_idx=int(len(df)*0.8)
    test=df.iloc[split_idx:].copy().reset_index(drop=True)
    entry=float(test.iloc[0].close)
    final=float(test.iloc[-1].close)
    alloc=float(ARGS.allocation if ARGS.allocation is not None else CFG['risk']['max_allocation_fraction'])
    gross_asset_return=final/entry-1.0
    cost_return=round_trip_cost_return(CFG['costs'])
    net_asset_return=gross_asset_return-cost_return
    equity=1.0+alloc*(test.close.astype(float)/entry-1.0)
    peak=equity.cummax()
    drawdown=(peak-equity)/peak
    gross_portfolio_return=alloc*gross_asset_return
    net_portfolio_return=alloc*net_asset_return
    return {
        'profile':'buy_and_hold',
        'allocation_fraction':alloc,
        'candidates':1,
        'test_candidates':1,
        'accept_count':1,
        'accept_rate':1.0,
        'unfiltered_mean_net_bps':net_asset_return*10000,
        'filtered_mean_net_bps':net_asset_return*10000,
        'unfiltered_portfolio_return':net_portfolio_return,
        'filtered_portfolio_return':net_portfolio_return,
        'unfiltered_max_drawdown':float(drawdown.max()),
        'filtered_max_drawdown':float(drawdown.max()),
        'positive_rate':float(net_asset_return>0),
        'target_rate':None,
        'time_exit_rate':0.0,
        'median_hold_bars':float(len(test)-1),
        'accepted_good_fraction':None,
        'mean_mfe_bps':float((test.close.max()/entry-1.0)*10000),
        'mean_mae_bps':float((test.close.min()/entry-1.0)*10000),
        'gross_asset_return':gross_asset_return,
        'gross_portfolio_return':gross_portfolio_return,
        'round_trip_cost_bps':cost_return*10000,
        'entry_timestamp':str(test.iloc[0].open_time),
        'exit_timestamp':str(test.iloc[-1].open_time),
        'entry_price':entry,
        'exit_price':final,
        'exit_reasons':{'end_of_test_hold':1},
    }, test.assign(buy_hold_equity=equity, buy_hold_drawdown=drawdown)


PROFILES={
 'fixed_15m':{'engine':'fixed','max_hold_bars':3},
 'fixed_1h':{'engine':'fixed','max_hold_bars':12},
 'fixed_4h':{'engine':'fixed','max_hold_bars':48},
 'fixed_1d':{'engine':'fixed','max_hold_bars':288},
 'adaptive_fast_4h':{'engine':'adaptive','max_hold_bars':48,'params':dict(breakeven_trigger_atr=.8,breakeven_offset_bps=1,trailing_activation_atr=1.2,trailing_distance_atr=.8,chandelier_lookback_bars=12,momentum_lookback_bars=4,momentum_exit_bps=-8,trend_fast_bars=4,trend_slow_bars=12,volatility_range_atr=3.0,min_hold_before_soft_exit_bars=3)},
 'adaptive_balanced_1d':{'engine':'adaptive','max_hold_bars':288,'params':dict(breakeven_trigger_atr=1.0,breakeven_offset_bps=2,trailing_activation_atr=1.5,trailing_distance_atr=1.0,chandelier_lookback_bars=24,momentum_lookback_bars=6,momentum_exit_bps=-12,trend_fast_bars=6,trend_slow_bars=18,volatility_range_atr=3.5,min_hold_before_soft_exit_bars=4)},
 'adaptive_patient_1d':{'engine':'adaptive','max_hold_bars':288,'params':dict(breakeven_trigger_atr=1.25,breakeven_offset_bps=0,trailing_activation_atr=2.0,trailing_distance_atr=1.4,chandelier_lookback_bars=36,momentum_lookback_bars=12,momentum_exit_bps=-20,trend_fast_bars=12,trend_slow_bars=36,volatility_range_atr=4.5,min_hold_before_soft_exit_bars=8)},
}

df=base_candles(n=100_000); events=inject_events(df,spacing=280); events=patch_delayed_paths(df,events)
all_results=[]
for name,profile in PROFILES.items():
    data=make_dataset(df,events,profile); out,pred=evaluate(data,name); all_results.append(out)
    data.to_csv(OUT/f'candidates_{name}.csv',index=False); pred.to_csv(OUT/f'test_predictions_{name}.csv',index=False)
bh_out,bh_curve=evaluate_buy_and_hold(df); all_results.append(bh_out); bh_curve.to_csv(OUT/'buy_and_hold_equity_curve.csv',index=False)
summary=pd.DataFrame([{k:v for k,v in r.items() if k!='exit_reasons'} for r in all_results]); summary.to_csv(OUT/'summary.csv',index=False)
events.to_csv(OUT/'events.csv',index=False); df.iloc[:10000].to_csv(OUT/'candles_sample.csv',index=False)
(OUT/'results.json').write_text(json.dumps(all_results,indent=2))
print(summary.to_string(index=False))
