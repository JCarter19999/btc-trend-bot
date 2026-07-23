"""Research-only selective RSI/Connors-style mean-reversion matrix.

This module narrows the v0.6 finding: short-horizon RSI(2) had positive gross
behavior but excessive turnover.  The variants below gate entries with broader
trend, volatility, recovery confirmation, cooldowns, and economic move hurdles.
No authenticated exchange or live-order path exists here.
"""
from __future__ import annotations

import argparse, copy, json, math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from btc_trend_bot.data import download_ohlcv, load_ohlcv_csv, timeframe_to_timedelta
from btc_trend_bot import popular_matrix as base


def load_config(path: str | Path) -> dict[str, Any]:
    p=Path(path)
    raw=yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw,dict): raise ValueError("Selective reversion config must be a mapping")
    for section in ("market","features","costs","research","strategies"):
        if section not in raw: raise ValueError(f"Missing config section: {section}")
    ids=[str(x["id"]) for x in raw["strategies"]]
    if len(ids)!=len(set(ids)): raise ValueError("Strategy IDs must be unique")
    return raw


def _completed(frame: pd.DataFrame, step: pd.Timedelta, rule: str, count: int) -> pd.DataFrame:
    return base._completed_resample(frame,step,rule,count)


def build_feature_frame(frame: pd.DataFrame,cfg:dict[str,Any])->pd.DataFrame:
    out=base.build_feature_frame(frame,cfg)
    step=timeframe_to_timedelta(str(cfg["market"]["timeframe"]))
    f=cfg["features"]
    # 15-minute RSI and recovery, aligned causally to completed 15m candles.
    raw,_=base.normalize_ohlcv(frame,timeframe=str(cfg["market"]["timeframe"]))
    raw=raw.reset_index(drop=True); raw["bar_end"]=pd.to_datetime(raw["timestamp"],utc=True)+step
    m15=_completed(raw,step,"15min",3)
    m15["m15_rsi2"]=base._rsi(m15["close"],2)
    m15["m15_rsi2_prev"]=m15["m15_rsi2"].shift(1)
    m15["m15_rsi2_prev2"]=m15["m15_rsi2"].shift(2)
    m15["m15_rsi_recovery_1"]=(m15["m15_rsi2"]>m15["m15_rsi2_prev"])
    m15["m15_rsi_recovery_2"]=(m15["m15_rsi2"]>m15["m15_rsi2_prev"])&(m15["m15_rsi2_prev"]>m15["m15_rsi2_prev2"])
    m15["m15_atr_bps"]=base._atr(m15,int(f.get("m15_atr_period",20)))/m15["close"]*10000
    med=m15["m15_atr_bps"].rolling(int(f.get("volatility_median_bars_15m",96)),min_periods=24).median()
    m15["m15_atr_ratio_to_median"]=m15["m15_atr_bps"]/med.replace(0,np.nan)
    keep=["bar_end","m15_rsi2","m15_rsi2_prev","m15_rsi_recovery_1","m15_rsi_recovery_2","m15_atr_bps","m15_atr_ratio_to_median"]
    out=pd.merge_asof(out.sort_values("bar_end"),m15[keep].sort_values("bar_end"),on="bar_end",direction="backward",allow_exact_matches=True)

    # Daily investable regime, completed daily candles only.
    day=_completed(raw,step,"1D",288)
    day["d1_ema_fast"]=base._ema(day["close"],int(f.get("d1_ema_fast",20)))
    day["d1_ema_slow"]=base._ema(day["close"],int(f.get("d1_ema_slow",50)))
    day["d1_trend_bps"]=(day["d1_ema_fast"]/day["d1_ema_slow"]-1)*10000
    day["d1_momentum_bps"]=(day["close"]/day["close"].shift(int(f.get("d1_momentum_days",5)))-1)*10000
    out=pd.merge_asof(out.sort_values("bar_end"),day[["bar_end","d1_trend_bps","d1_momentum_bps"]].sort_values("bar_end"),on="bar_end",direction="backward",allow_exact_matches=True)

    out["rsi2_5m_prev2"]=out["rsi2_5m"].shift(2)
    out["rsi5_recovery_1"]=(out["rsi2_5m"]>out["rsi2_5m_prev"])
    out["rsi5_recovery_2"]=(out["rsi2_5m"]>out["rsi2_5m_prev"])&(out["rsi2_5m_prev"]>out["rsi2_5m_prev2"])
    out["m15_atr_bps"]=out["m15_atr"]/out["close"]*10000
    required=["d1_trend_bps","d1_momentum_bps","m15_rsi2","m15_atr_bps","m15_atr_ratio_to_median"]
    out["feature_valid"]=out["feature_valid"]&out[required].notna().all(axis=1)
    return out.reset_index(drop=True)


def _finite(v:Any,default:float=0.0)->float: return base._finite(v,default)


@dataclass
class SelectiveSetupState:
    armed: bool = False
    armed_index: int | None = None
    minimum_rsi: float = 100.0

    def clear(self) -> None:
        self.armed = False
        self.armed_index = None
        self.minimum_rsi = 100.0


@dataclass(frozen=True)
class SelectiveSimulationResult:
    snapshots: pd.DataFrame
    transactions: pd.DataFrame
    episodes: pd.DataFrame
    summary: dict[str, dict[str, Any]]
    gate_counts: pd.DataFrame


def _count(diag: dict[tuple[str, str], int] | None, sid: str, gate: str) -> None:
    if diag is not None:
        key = (sid, gate)
        diag[key] = diag.get(key, 0) + 1


def decide(
    strategy:dict[str,Any],
    row:pd.Series,
    state:base.StrategyState,
    index:int,
    costs:dict[str,Any],
    setup:SelectiveSetupState|None=None,
    diagnostics:dict[tuple[str,str],int]|None=None,
)->base.Decision:
    sid=str(strategy["id"]); typ=str(strategy["type"]); prev=state.target_position
    if typ=="cash": return base.Decision(sid,0.0,"cash","cash benchmark")
    if typ=="buy_hold": return base.Decision(sid,1.0,"long","buy-and-hold benchmark")

    # Route the control through the exact v0.6 implementation.  The copied
    # strategy is changed only from rsi2_v06_control -> rsi2 so it does not
    # inherit any selective setup or regime gates.
    if typ=="rsi2_v06_control":
        control=copy.deepcopy(strategy); control["type"]="rsi2"
        return base.decide_strategy(control,row,state,index,costs)
    if typ!="selective_rsi2": raise ValueError(f"Unknown selective strategy type: {typ}")
    if setup is None: raise ValueError("selective_rsi2 requires a SelectiveSetupState")

    held=base._bars_held(state,index)
    cooldown=int(strategy.get("cooldown_bars_5m",72))
    minhold=int(strategy.get("minimum_hold_bars_5m",12))
    maxhold=int(strategy.get("maximum_hold_bars_5m",288))

    source=str(strategy.get("signal_source","5m"))
    if source=="5m":
        rsi=_finite(row["rsi2_5m"],100)
        recovery=bool(row["rsi5_recovery_2"] if int(strategy.get("recovery_ticks",2))>=2 else row["rsi5_recovery_1"])
    elif source=="15m":
        rsi=_finite(row["m15_rsi2"],100)
        recovery=bool(row["m15_rsi_recovery_2"] if int(strategy.get("recovery_ticks",2))>=2 else row["m15_rsi_recovery_1"])
    else: raise ValueError("signal_source must be 5m or 15m")

    if prev<=0:
        if base._bars_since_trade(state,index)<cooldown:
            _count(diagnostics,sid,"blocked_cooldown")
            return base.Decision(sid,0.0,"hold_cash","cooldown active")

        entry_rsi=float(strategy.get("entry_rsi",5))
        expiry=int(strategy.get("setup_expiry_bars_5m",36 if source=="5m" else 96))

        # Phase 1: remember the oversold event.  Entry is intentionally not
        # permitted on the arming candle.
        if not setup.armed:
            if rsi<=entry_rsi:
                setup.armed=True; setup.armed_index=index; setup.minimum_rsi=rsi
                _count(diagnostics,sid,"oversold_setup_armed")
                return base.Decision(sid,0.0,"arm_oversold",f"{source} RSI2 oversold setup armed")
            _count(diagnostics,sid,"no_oversold_setup")
            return base.Decision(sid,0.0,"hold_cash","no oversold setup")

        setup.minimum_rsi=min(setup.minimum_rsi,rsi)
        age=index-int(setup.armed_index if setup.armed_index is not None else index)
        if age>expiry:
            setup.clear(); _count(diagnostics,sid,"setup_expired")
            return base.Decision(sid,0.0,"expire_setup","oversold setup expired")

        if not recovery:
            _count(diagnostics,sid,"waiting_for_recovery")
            return base.Decision(sid,0.0,"wait_recovery","oversold setup armed; awaiting recovery")

        _count(diagnostics,sid,"recovery_confirmed")
        h4_ok=_finite(row["h4_trend_bps"])>=float(strategy.get("min_h4_trend_bps",-1e9))
        d1_ok=_finite(row["d1_trend_bps"])>=float(strategy.get("min_d1_trend_bps",-1e9)) and _finite(row["d1_momentum_bps"])>=float(strategy.get("min_d1_momentum_bps",-1e9))
        vol_ratio=_finite(row["m15_atr_ratio_to_median"],999)
        vol_ok=float(strategy.get("min_atr_ratio",0))<=vol_ratio<=float(strategy.get("max_atr_ratio",999))
        expected_move=_finite(row["m15_atr_bps"])*float(strategy.get("expected_move_atr_multiple",1.0))
        all_in=2*(_finite(costs.get("fee_bps_per_side"))+_finite(costs.get("slippage_bps_per_side"))+_finite(costs.get("assumed_spread_bps_per_side")))
        hurdle_ok=expected_move>=all_in*float(strategy.get("cost_hurdle_multiple",1.25))
        _count(diagnostics,sid,"economic_hurdle_pass" if hurdle_ok else "economic_hurdle_fail")

        if not h4_ok:
            _count(diagnostics,sid,"blocked_h4")
            return base.Decision(sid,0.0,"blocked_h4","recovery confirmed; 4h regime veto")
        if not d1_ok:
            _count(diagnostics,sid,"blocked_daily")
            return base.Decision(sid,0.0,"blocked_daily","recovery confirmed; daily regime veto")
        if not vol_ok:
            _count(diagnostics,sid,"blocked_volatility")
            return base.Decision(sid,0.0,"blocked_volatility","recovery confirmed; volatility veto")

        # The economic hurdle is diagnostic in v0.8, not a hard entry gate.
        setup.clear(); _count(diagnostics,sid,"entry")
        reason=f"{source} RSI2 recovered from armed oversold setup; economic_hurdle={'pass' if hurdle_ok else 'fail'}"
        return base.Decision(sid,1.0,"enter_selective_rsi",reason)

    # A live position invalidates any stale setup state.
    setup.clear()
    atr=_finite(row["m15_atr"]); high=max(_finite(state.highest_high),float(row["high"])); close=float(row["close"])
    if held>=maxhold: return base.Decision(sid,0.0,"exit_time","maximum hold")
    if held>=minhold and atr>0 and close<=high-float(strategy.get("trailing_stop_atr",3.0))*atr:
        return base.Decision(sid,0.0,"exit_trailing","ATR trailing stop")
    regime_fail=_finite(row["h4_trend_bps"])<=float(strategy.get("exit_h4_trend_bps",-25)) or _finite(row["d1_momentum_bps"])<=float(strategy.get("exit_d1_momentum_bps",-250))
    if held>=minhold and regime_fail: return base.Decision(sid,0.0,"exit_regime","broader regime failed")
    if held>=minhold and rsi>=float(strategy.get("exit_rsi",75)):
        return base.Decision(sid,0.0,"exit_recovery","RSI recovery target")
    return base.Decision(sid,1.0,"hold_btc","selective RSI position")


def simulate(feature_frame:pd.DataFrame,cfg:dict[str,Any],costs_override:dict[str,Any]|None=None)->SelectiveSimulationResult:
    costs=copy.deepcopy(costs_override if costs_override is not None else cfg["costs"])
    initial=float(cfg["research"].get("initial_cash",500.0)); strategies=cfg["strategies"]
    states={str(s["id"]):base.StrategyState(str(s["id"]),initial,initial,0.0,initial,0.0,peak_equity=initial) for s in strategies}
    setups={str(s["id"]):SelectiveSetupState() for s in strategies if str(s["type"])=="selective_rsi2"}
    diagnostics:dict[tuple[str,str],int]={}
    valid=feature_frame.index[feature_frame["feature_valid"]].tolist()
    if not valid: raise RuntimeError("No rows contain a complete feature history")
    snaps=[]; trades=[]
    for i in range(valid[0],len(feature_frame)-1):
        row=feature_frame.iloc[i]; nxt=feature_frame.iloc[i+1]
        if not bool(row["feature_valid"]): continue
        for s in strategies:
            sid=str(s["id"]); st=states[sid]
            if st.target_position>0: st.highest_high=max(_finite(st.highest_high),float(row["high"]))
            d=decide(s,row,st,i,costs,setups.get(sid),diagnostics)
            snap,tr=base.execute_decision(st,d,row,nxt,i,costs); snaps.append(snap)
            if tr is not None: trades.append(asdict(tr))
    sf=pd.DataFrame(snaps); tf=pd.DataFrame(trades)
    episodes=base.build_trade_episodes(sf,initial,5)
    days=max((pd.Timestamp(sf["execution_timestamp"].max())-pd.Timestamp(sf["execution_timestamp"].min())).total_seconds()/86400,1e-9)
    gates=pd.DataFrame([{"strategy_id":sid,"gate":gate,"count":count} for (sid,gate),count in sorted(diagnostics.items())])
    return SelectiveSimulationResult(sf,tf,episodes,base._summarize(sf,tf,episodes,initial,days,costs),gates)

def cost_grid(feature_frame:pd.DataFrame,cfg:dict[str,Any])->pd.DataFrame:
    rows=[]
    for bps in cfg["research"].get("cost_sensitivity_all_in_bps_per_side",[3,5,8,10,12]):
        override={"fee_bps_per_side":float(bps),"slippage_bps_per_side":0.0,"assumed_spread_bps_per_side":0.0,"min_notional":cfg["costs"].get("min_notional",10),"rebalance_tolerance_bps":cfg["costs"].get("rebalance_tolerance_bps",1)}
        r=simulate(feature_frame,cfg,override)
        for sid,m in r.summary.items(): rows.append({"strategy_id":sid,"all_in_bps_per_side":bps,"ending_equity":m["ending_equity"],"net_return_pct":m["net_return_pct"],"max_drawdown":m["max_drawdown"],"transaction_count":m["transaction_count"],"round_trips_closed":m["round_trips_closed"],"turnover_multiple":m["turnover_multiple"],"average_holding_bars":m["average_holding_bars"]})
    return pd.DataFrame(rows)


def run_research(config_path:str,bars_requested:int,output_dir:str,data_path:str|None=None,skip_cost_grid:bool=False,compact:bool=False)->dict[str,Any]:
    cfg=load_config(config_path); market=cfg["market"]; tf=str(market["timeframe"])
    if data_path:
        raw,_=load_ohlcv_csv(data_path,timeframe=tf); raw=raw.tail(bars_requested).reset_index(drop=True)
    else:
        step=timeframe_to_timedelta(tf); start=(pd.Timestamp.now(tz="UTC")-step*(bars_requested+400)).isoformat()
        raw=download_ohlcv(exchange_id=str(market["exchange"]),symbol=str(market["symbol"]),timeframe=tf,start=start,max_bars=bars_requested)
    ff=build_feature_frame(raw,cfg); result=simulate(ff,cfg); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    # Compact mode prevents the 2GB VM OOM / huge Git artifacts.
    if not compact:
        ff.to_csv(out/"feature_frame.csv",index=False); result.snapshots.to_csv(out/"strategy_equity.csv",index=False)
    result.transactions.to_csv(out/"transactions.csv",index=False); result.episodes.to_csv(out/"trade_episodes.csv",index=False)
    pd.DataFrame.from_dict(result.summary,orient="index").rename_axis("strategy_id").reset_index().to_csv(out/"strategy_comparison.csv",index=False)
    result.snapshots.groupby(["strategy_id","signal"]).size().rename("count").reset_index().to_csv(out/"signal_counts.csv",index=False)
    result.gate_counts.to_csv(out/"gate_counts.csv",index=False)
    if not skip_cost_grid: cost_grid(ff,cfg).to_csv(out/"cost_sensitivity.csv",index=False)
    if not compact: base._save_charts(result.snapshots,out)
    valid=ff[ff["feature_valid"]]; first=pd.Timestamp(valid["timestamp"].iloc[0]); last=pd.Timestamp(valid["timestamp"].iloc[-1])
    summary={"market":market,"downloaded_bars":len(raw),"scored_bars":max(len(valid)-1,0),"first_scored_bar":first.isoformat(),"last_scored_bar":last.isoformat(),"sample_days":max((last-first).total_seconds()/86400,1e-9),"compact_outputs":compact,"cost_assumptions":cfg["costs"],"strategies":result.summary,"outputs":{"strategy_comparison":"strategy_comparison.csv","cost_sensitivity":None if skip_cost_grid else "cost_sensitivity.csv","trade_episodes":"trade_episodes.csv","transactions":"transactions.csv","gate_counts":"gate_counts.csv","feature_frame":None if compact else "feature_frame.csv","strategy_equity":None if compact else "strategy_equity.csv"}}
    (out/"research_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8"); print(json.dumps(summary,indent=2,allow_nan=False)); return summary


def main(argv:Iterable[str]|None=None)->None:
    p=argparse.ArgumentParser(description="Selective RSI mean-reversion research matrix")
    p.add_argument("--config",default="config/settings_selective_reversion.yaml"); p.add_argument("--bars",type=int,default=10000); p.add_argument("--output",default="outputs/selective_reversion_10000"); p.add_argument("--data",default=None); p.add_argument("--skip-cost-grid",action="store_true"); p.add_argument("--compact",action="store_true")
    a=p.parse_args(list(argv) if argv is not None else None); run_research(a.config,a.bars,a.output,a.data,a.skip_cost_grid,a.compact)
if __name__=="__main__": main()
