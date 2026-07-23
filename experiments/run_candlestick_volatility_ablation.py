from __future__ import annotations
import sys, math, json, argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from btc_trend_bot.v1.features import build_features, FEATURE_COLUMNS, BASE_FEATURE_COLUMNS
from btc_trend_bot.v1.strategy import generate_candidates, candidate_mask
from btc_trend_bot.v1.labeling import mature_candidate, round_trip_cost_return
from btc_trend_bot.v1.exit_engine import ExitEngineConfig, run_exit_engine

CFG=yaml.safe_load((ROOT/'config/v1.yaml').read_text())
parser=argparse.ArgumentParser()
parser.add_argument('--allocation',type=float,default=1.0)
parser.add_argument('--output',default='outputs/candlestick_volatility_ablation')
parser.add_argument('--bars',type=int,default=60000)
args=parser.parse_args()
OUT=Path(args.output)
if not OUT.is_absolute(): OUT=ROOT/OUT
OUT.mkdir(parents=True,exist_ok=True)


def market_params(kind,i,n):
    # Per-five-minute log-return drift and volatility. These create moderate,
    # bounded test-period moves rather than the prior 8x synthetic bull path.
    if kind=='bull':
        return 1.2e-5 + 5e-6*math.sin(i/3000), 2.0e-4
    if kind=='bear':
        return -9e-6 + 5e-6*math.sin(i/3000), 2.2e-4
    if kind=='sideways':
        return 6e-6*math.sin(i/1800), 2.0e-4
    if kind=='high_volatility':
        return 2e-6*math.sin(i/2200), 6.5e-4
    # Regime switching: bull -> chop -> bear -> high vol -> recovery.
    frac=i/n
    if frac<0.20: return 1.4e-5,2.0e-4
    if frac<0.40: return 0.0,2.2e-4
    if frac<0.60: return -1.3e-5,2.8e-4
    if frac<0.80: return 0.0,7.0e-4
    return 1.0e-5,2.5e-4


def base_candles(kind,n=80000,seed=401):
    seeds={'bull':11,'bear':22,'sideways':33,'high_volatility':44,'regime_switching':55}
    rng=np.random.default_rng(seed+seeds[kind])
    ts=pd.date_range('2022-01-01',periods=n,freq='5min',tz='UTC')
    prices=np.empty(n+1); prices[0]=30000.; rows=[]
    for i in range(n):
        mu,sigma=market_params(kind,i,n)
        # Mild negative lag-one component avoids unrealistic monotonic trends.
        shock=rng.normal(0,sigma)
        r=mu+shock
        o=prices[i]; c=o*math.exp(r)
        wick=max(abs(r)*0.45,0.00065)+(0.00055 if kind=='high_volatility' else 0.00035)
        h=max(o,c)*(1+wick); l=min(o,c)*(1-wick)
        v=100*(1+0.10*math.sin(i/900))+rng.lognormal(0,0.18)*(1+sigma/0.0003)
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


def inject_events(df,seed=501,spacing=220,start=500):
    rng=np.random.default_rng(seed); events=[]; delays=np.array([2,8,36,180])
    for j,end in enumerate(range(start,len(df)-310,spacing)):
        quality=bool(rng.random()<.5); p_target=.80 if quality else .20
        target=bool(rng.random()<p_target); delay=int(rng.choice(delays))
        overwrite_bar(df,end-3,-.00015,.85,.60)
        if quality: rets=[.00100,.00110,.00120]; vols=[1.65,1.85,2.15]; bodies=[.72,.80,.88]
        else: rets=[.00076,.00080,.00086]; vols=[1.18,1.24,1.30]; bodies=[.56,.60,.64]
        for k,(r,v,b) in enumerate(zip(rets,vols,bodies)): overwrite_bar(df,end-2+k,r,v,b)
        events.append({'event_index':j,'candidate_bar':end,'quality':int(quality),'target_planted':int(target),'resolution_delay_bars':delay})
    return pd.DataFrame(events)


def patch_paths(df,events):
    feat=build_features(df,CFG); mask=candidate_mask(feat,CFG); kept=[]; rng=np.random.default_rng(777)
    for rec in events.to_dict('records'):
        idx=int(rec['candidate_bar'])
        if idx>=len(feat) or not bool(mask.iloc[idx]): continue
        row=feat.iloc[idx]; entry=float(row.close); atr=float(row.atr)
        stop=entry-CFG['exit']['stop_atr_multiple']*atr; target=entry+CFG['exit']['target_atr_multiple']*atr
        delay=int(rec['resolution_delay_bars']); is_target=bool(rec['target_planted']); end_price=target if is_target else stop; prev=entry
        for k in range(1,delay+1):
            i=idx+k
            if i>=len(df): break
            frac=k/delay
            if k<delay:
                desired=entry+(end_price-entry)*(.72*frac); noise=rng.normal(0,atr*.025)
                c=float(np.clip(desired+noise,stop+.10*atr,target-.10*atr)); o=prev
                h=min(max(o,c)+.04*atr,target-.04*atr); l=max(min(o,c)-.04*atr,stop+.04*atr)
            else:
                o=prev
                if is_target: c=target-.03*atr; h=target+.01*atr; l=max(min(o,c)-.03*atr,stop+.04*atr)
                else: c=stop+.03*atr; l=stop-.01*atr; h=min(max(o,c)+.03*atr,target-.04*atr)
            df.at[i,'open']=o; df.at[i,'close']=c; df.at[i,'high']=max(h,o,c); df.at[i,'low']=min(l,o,c)
            df.at[i,'volume']=float(df.loc[max(0,i-48):i-1,'volume'].mean())*(1.20 if rec['quality'] else 1.03); prev=c
        j=idx+delay+1
        if j<len(df):
            ratio=prev/float(df.at[j,'open'])
            for col in ['open','high','low','close']: df.at[j,col]=float(df.at[j,col])*ratio
        kept.append(rec)
    return pd.DataFrame(kept)

PROFILES={
 'balanced_legacy':{'engine':'adaptive','max_hold_bars':288,'params':dict(breakeven_trigger_atr=1.,breakeven_offset_bps=2,trailing_activation_atr=1.5,trailing_distance_atr=1.,chandelier_lookback_bars=24,momentum_lookback_bars=6,momentum_exit_bps=-12,trend_fast_bars=6,trend_slow_bars=18,volatility_range_atr=3.5,min_hold_before_soft_exit_bars=4,volatility_adaptive_enabled=False)},
 'balanced_volatility_aware':{'engine':'adaptive','max_hold_bars':288,'params':dict(breakeven_trigger_atr=1.,breakeven_offset_bps=2,trailing_activation_atr=1.5,trailing_distance_atr=1.,chandelier_lookback_bars=24,momentum_lookback_bars=6,momentum_exit_bps=-12,trend_fast_bars=6,trend_slow_bars=18,volatility_range_atr=3.5,min_hold_before_soft_exit_bars=4,volatility_adaptive_enabled=True,volatility_lookback_bars=12,volatility_factor_min=1.,volatility_factor_max=2.5,trailing_volatility_exponent=.8,soft_exit_delay_volatility_exponent=1.,momentum_volatility_exponent=.8)},
}

def make_dataset(df,events,profile,feat=None,candidates=None):
    if feat is None: feat=build_features(df,CFG)
    if candidates is None: candidates=generate_candidates(feat,CFG)
    event_map={pd.Timestamp(df.at[int(r.candidate_bar),'open_time']):r for _,r in events.iterrows()}; rows=[]
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
            m=run_exit_engine(c2,df,CFG['costs'],ExitEngineConfig(max_hold_bars=int(profile['max_hold_bars']),**profile['params']))
        if m.get('label_status')!='MATURED': continue
        flat={**c2.features,**{k:v for k,v in m.items() if k not in ('features','metadata')}}
        flat['planted_quality']=int(ev.quality); flat['planted_target']=int(ev.target_planted); flat['resolution_delay_bars']=int(ev.resolution_delay_bars); rows.append(flat)
    return pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)


def portfolio(r,alloc=1.):
    if len(r)==0:return 0.,0.
    eq=np.cumprod(1+alloc*np.asarray(r)); peak=np.maximum.accumulate(np.r_[1.,eq]); dd=(peak[1:]-eq)/peak[1:]
    return float(eq[-1]-1),float(dd.max())


def evaluate(data,name,market,feature_cols,feature_set):
    n=len(data); a=int(n*.6); b=int(n*.8); train=data.iloc[:a]; cal=data.iloc[a:b]; test=data.iloc[b:]
    model=Pipeline([('scale',StandardScaler()),('reg',Ridge(alpha=10.))]); model.fit(train[feature_cols],train.net_return.astype(float))
    rc=model.predict(cal[feature_cols]); rt=model.predict(test[feature_cols]); beta=np.linalg.lstsq(np.c_[np.ones(len(rc)),rc],cal.net_return.to_numpy(float),rcond=None)[0]
    pred=beta[0]+beta[1]*rt; accept=pred>(CFG['model']['safety_margin_bps']/10000); net=test.net_return.to_numpy(float)
    ur,ud=portfolio(net,args.allocation); fr,fd=portfolio(net[accept],args.allocation)
    return {'market':market,'profile':name,'feature_set':feature_set,'test_candidates':len(test),'accept_count':int(accept.sum()),'accept_rate':float(accept.mean()),'unfiltered_mean_net_bps':float(net.mean()*1e4),'filtered_mean_net_bps':float(net[accept].mean()*1e4) if accept.any() else None,'unfiltered_return':ur,'filtered_return':fr,'unfiltered_max_drawdown':ud,'filtered_max_drawdown':fd,'median_hold_bars':float(test.bars_held.median()),'accepted_good_fraction':float(test.loc[accept,'planted_quality'].mean()) if accept.any() else None}


def buy_hold(df,market):
    test=df.iloc[int(len(df)*.8):].reset_index(drop=True); entry=float(test.iloc[0].close); final=float(test.iloc[-1].close)
    gross=final/entry-1; net=gross-round_trip_cost_return(CFG['costs']); equity=1+args.allocation*(test.close.astype(float)/entry-1); dd=((equity.cummax()-equity)/equity.cummax()).max()
    return {'market':market,'profile':'buy_and_hold','test_candidates':1,'accept_count':1,'accept_rate':1.,'unfiltered_mean_net_bps':net*1e4,'filtered_mean_net_bps':net*1e4,'unfiltered_return':args.allocation*net,'filtered_return':args.allocation*net,'unfiltered_max_drawdown':float(dd),'filtered_max_drawdown':float(dd),'median_hold_bars':float(len(test)-1),'accepted_good_fraction':None,'entry_price':entry,'exit_price':final}

results=[]
for market in ['bull','bear','sideways','high_volatility','regime_switching']:
    print('RUN',market,flush=True)
    df=base_candles(market,args.bars); events=patch_paths(df,inject_events(df,seed=501+len(results),spacing=220))
    feat=build_features(df,CFG); candidates=generate_candidates(feat,CFG)
    mdir=OUT/market; mdir.mkdir(exist_ok=True)
    for name,profile in PROFILES.items():
        data=make_dataset(df,events,profile,feat,candidates)
        for feature_set,feature_cols in [('baseline',BASE_FEATURE_COLUMNS),('candlestick_geometry',FEATURE_COLUMNS)]:
            r=evaluate(data,name,market,feature_cols,feature_set); results.append(r)
        data.to_csv(mdir/f'candidates_{name}.csv',index=False)
    bh=buy_hold(df,market); bh['feature_set']='benchmark'; results.append(bh); events.to_csv(mdir/'events.csv',index=False); df.iloc[:5000].to_csv(mdir/'candles_sample.csv',index=False)
summary=pd.DataFrame(results); summary.to_csv(OUT/'summary.csv',index=False); (OUT/'results.json').write_text(json.dumps(results,indent=2))
# convenient pivots
summary[summary.profile!='buy_and_hold'].pivot_table(index='market',columns=['profile','feature_set'],values='filtered_return').to_csv(OUT/'returns_pivot.csv')
summary[summary.profile!='buy_and_hold'].pivot_table(index='market',columns=['profile','feature_set'],values='filtered_max_drawdown').to_csv(OUT/'drawdowns_pivot.csv')
print(summary[['market','profile','feature_set','accept_count','filtered_mean_net_bps','filtered_return','filtered_max_drawdown']].to_string(index=False))
