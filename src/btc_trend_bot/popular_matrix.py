"""Research-only popular-strategy BTC matrix.

This module compares several five-minute strategy families under one common,
causal, cost-aware simulation. It intentionally contains no live-order or
exchange-authentication path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from btc_trend_bot.data import (
    download_ohlcv,
    load_ohlcv_csv,
    normalize_ohlcv,
    timeframe_to_timedelta,
)


@dataclass
class StrategyState:
    strategy_id: str
    initial_cash: float
    cash: float
    btc: float
    gross_cash: float
    gross_btc: float
    target_position: float = 0.0
    peak_equity: float | None = None
    total_fees: float = 0.0
    total_spread: float = 0.0
    total_slippage: float = 0.0
    total_turnover: float = 0.0
    trade_count: int = 0
    last_trade_index: int | None = None
    entry_index: int | None = None
    entry_mark: float | None = None
    highest_high: float | None = None
    entry_reference: float | None = None
    pending_entry_index: int | None = None


@dataclass(frozen=True)
class Decision:
    strategy_id: str
    target_position: float
    signal: str
    reason: str
    entry_reference: float | None = None


@dataclass(frozen=True)
class SimulatedTrade:
    strategy_id: str
    signal_timestamp: str
    execution_timestamp: str
    side: str
    mark_price: float
    reference_price: float
    fill_price: float
    btc_delta: float
    gross_notional: float
    fee: float
    spread_cost: float
    slippage_cost: float
    cash_after: float
    btc_after: float
    equity_after: float
    signal: str
    reason: str


@dataclass(frozen=True)
class SimulationResult:
    snapshots: pd.DataFrame
    transactions: pd.DataFrame
    episodes: pd.DataFrame
    summary: dict[str, dict[str, Any]]


def load_matrix_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Popular matrix config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Popular matrix config must contain a YAML mapping.")
    for section in ("market", "features", "costs", "research", "strategies"):
        if section not in raw:
            raise ValueError(f"Missing popular matrix config section: {section}")
    strategies = raw["strategies"]
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("strategies must be a non-empty list.")
    ids = [str(item["id"]) for item in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError("Strategy IDs must be unique.")
    return raw


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _ema(series: pd.Series, bars: int) -> pd.Series:
    if bars <= 0:
        raise ValueError("EMA bars must be positive.")
    return series.ewm(span=bars, adjust=False, min_periods=bars).mean()


def _completed_resample(
    frame: pd.DataFrame,
    base_step: pd.Timedelta,
    rule: str,
    expected_source_bars: int,
) -> pd.DataFrame:
    """Build only complete right-labeled higher-timeframe candles.

    The source timestamps are candle opens. Converting them to candle-end times
    makes the right-labeled aggregate available exactly when its final five-minute
    source candle has completed, preventing higher-timeframe lookahead.
    """
    work = frame.copy()
    work["bar_end"] = pd.to_datetime(work["timestamp"], utc=True) + base_step
    indexed = work.set_index("bar_end")
    grouped = indexed.resample(
        rule,
        origin="epoch",
        label="right",
        closed="right",
    )
    aggregated = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped["close"].count()
    aggregated = aggregated.loc[counts == expected_source_bars]
    aggregated = aggregated.dropna().reset_index()
    return aggregated


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

def _adx(frame: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = frame['high'], frame['low'], frame['close']
    up = high.diff(); down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean() / atr.replace(0,np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean() / atr.replace(0,np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0,np.nan)
    return dx.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    pc = frame['close'].shift(1)
    tr = pd.concat([(frame['high']-frame['low']), (frame['high']-pc).abs(), (frame['low']-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

def build_feature_frame(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    market = cfg["market"]; f = cfg["features"]
    timeframe = str(market["timeframe"]); step = timeframe_to_timedelta(timeframe)
    if step != pd.Timedelta(minutes=5): raise ValueError("Popular matrix requires 5m data.")
    out,_ = normalize_ohlcv(frame,timeframe=timeframe); out=out.reset_index(drop=True)
    out["bar_end"] = pd.to_datetime(out["timestamp"],utc=True)+step
    out["ema_fast_5m"]=_ema(out["close"],int(f.get("ema_fast_5m",9)))
    out["ema_slow_5m"]=_ema(out["close"],int(f.get("ema_slow_5m",21)))
    out["rsi2_5m"]=_rsi(out["close"],2)
    out["rsi2_5m_prev"]=out["rsi2_5m"].shift(1)
    out["recovery_5m"]=(out["close"]>out["ema_fast_5m"])&(out["close"].shift(1)<=out["ema_fast_5m"].shift(1))
    out["pullback_5m_bps"]=(out["close"]/out["ema_fast_5m"]-1)*10000

    def enrich(rule,bars,prefix):
        x=_completed_resample(out,step,rule,bars)
        x[f"{prefix}_atr"]=_atr(x,int(f.get(f"{prefix}_atr_period",20)))
        x[f"{prefix}_adx"]=_adx(x,int(f.get(f"{prefix}_adx_period",14)))
        x[f"{prefix}_ema_fast"]=_ema(x["close"],int(f.get(f"{prefix}_ema_fast",20)))
        x[f"{prefix}_ema_slow"]=_ema(x["close"],int(f.get(f"{prefix}_ema_slow",50)))
        x[f"{prefix}_trend_bps"]=(x[f"{prefix}_ema_fast"]/x[f"{prefix}_ema_slow"]-1)*10000
        if prefix in ("m15","m30"):
            n1=int(f.get(f"{prefix}_donchian_entry",24)); n2=int(f.get(f"{prefix}_donchian_exit",12))
            x[f"{prefix}_donchian_high"]=x["high"].shift(1).rolling(n1,min_periods=n1).max()
            x[f"{prefix}_donchian_low"]=x["low"].shift(1).rolling(n2,min_periods=n2).min()
            bb=int(f.get("squeeze_period",20)); sd=x["close"].rolling(bb,min_periods=bb).std(ddof=0); mid=x["close"].rolling(bb,min_periods=bb).mean()
            x[f"{prefix}_bb_upper"]=mid+2*sd; x[f"{prefix}_bb_lower"]=mid-2*sd
            x[f"{prefix}_kc_upper"]=_ema(x["close"],bb)+1.5*x[f"{prefix}_atr"]
            x[f"{prefix}_kc_lower"]=_ema(x["close"],bb)-1.5*x[f"{prefix}_atr"]
            x[f"{prefix}_squeeze"]=(x[f"{prefix}_bb_upper"]<x[f"{prefix}_kc_upper"])&(x[f"{prefix}_bb_lower"]>x[f"{prefix}_kc_lower"])
            x[f"{prefix}_squeeze_release"]=(~x[f"{prefix}_squeeze"])&x[f"{prefix}_squeeze"].shift(1, fill_value=False)
            fast=_ema(x["close"],12); slow=_ema(x["close"],26); x[f"{prefix}_macd"]=fast-slow; x[f"{prefix}_macd_signal"]=_ema(x[f"{prefix}_macd"],9)
        cols=[c for c in x.columns if c=="bar_end" or c.startswith(prefix+"_")]
        return x[cols]
    for rule,bars,prefix in [("15min",3,"m15"),("30min",6,"m30"),("1h",12,"h1"),("4h",48,"h4")]:
        x=enrich(rule,bars,prefix)
        out=pd.merge_asof(out.sort_values("bar_end"),x.sort_values("bar_end"),on="bar_end",direction="backward",allow_exact_matches=True)
    required=["ema_fast_5m","ema_slow_5m","rsi2_5m","m15_atr","m15_adx","m15_donchian_high","m15_squeeze_release","m30_donchian_high","h1_trend_bps","h4_trend_bps"]
    # Compatibility fields consumed by the shared simulator/report schema.
    out["streak_direction"] = 0
    out["streak_length"] = 0
    out["run_return_bps"] = 0.0
    out["range_atr_ratio"] = (out["high"] - out["low"]) / out["m15_atr"].replace(0, np.nan)
    out["relative_volume"] = 1.0
    out["vwap_zscore"] = 0.0
    out["one_hour_ema_spread_bps"] = out["h1_trend_bps"]
    out["one_hour_momentum_bps"] = out["h1_trend_bps"]
    out["four_hour_momentum_bps"] = out["h4_trend_bps"]
    out["feature_valid"]=out[required].notna().all(axis=1)
    return out.reset_index(drop=True)


def _bars_since_trade(state: StrategyState, index: int) -> int:
    if state.last_trade_index is None:
        return 1_000_000_000
    return max(0, index - state.last_trade_index)


def _bars_held(state: StrategyState, index: int) -> int:
    if state.entry_index is None:
        return 0
    return max(0, index - state.entry_index + 1)


def _one_hour_bullish(row: pd.Series, strategy: dict[str, Any]) -> bool:
    return (
        _finite(row["one_hour_ema_spread_bps"]) >= float(strategy.get("entry_1h_ema_spread_bps", 0.0))
        and _finite(row["one_hour_momentum_bps"]) >= float(strategy.get("entry_1h_momentum_bps", 0.0))
    )


def _one_hour_bearish(row: pd.Series, strategy: dict[str, Any]) -> bool:
    spread_weak = _finite(row["one_hour_ema_spread_bps"]) <= float(
        strategy.get("exit_1h_ema_spread_bps", -10.0)
    )
    momentum_weak = _finite(row["one_hour_momentum_bps"]) <= float(
        strategy.get("exit_1h_momentum_bps", -10.0)
    )
    required = int(strategy.get("exit_confirmations_required", 2))
    if required not in (1, 2):
        raise ValueError("exit_confirmations_required must be 1 or 2")
    return int(spread_weak) + int(momentum_weak) >= required


def decide_strategy(strategy: dict[str, Any], row: pd.Series, state: StrategyState, index: int, costs: dict[str, Any]) -> Decision:
    sid=str(strategy["id"]); typ=str(strategy["type"]); prev=state.target_position
    if typ=="cash": return Decision(sid,0.0,"cash","cash benchmark")
    if typ=="buy_hold": return Decision(sid,1.0,"long","buy-and-hold benchmark")
    held=_bars_held(state,index); cooldown=int(strategy.get("cooldown_bars_5m",12)); minhold=int(strategy.get("minimum_hold_bars_5m",6)); maxhold=int(strategy.get("maximum_hold_bars_5m",576))
    if prev<=0 and _bars_since_trade(state,index)<cooldown: return Decision(sid,0.0,"hold_cash","cooldown active")
    def exit_common(prefix="m15"):
        atr=_finite(row.get(f"{prefix}_atr")); high=max(_finite(state.highest_high),float(row["high"])); mult=float(strategy.get("trailing_stop_atr",2.5))
        if held>=maxhold: return Decision(sid,0.0,"exit_time","maximum hold")
        if held>=minhold and atr>0 and float(row["close"])<=high-mult*atr: return Decision(sid,0.0,"exit_trailing","ATR trailing stop")
        return None
    if typ=="donchian":
        p=str(strategy.get("prefix","m15")); trend=_finite(row[f"{p}_trend_bps"]); adx=_finite(row[f"{p}_adx"])
        if prev<=0:
            if float(row["close"])>_finite(row[f"{p}_donchian_high"],float("inf")) and trend>=float(strategy.get("min_trend_bps",0)) and adx>=float(strategy.get("min_adx",18)):
                return Decision(sid,1.0,"enter_donchian",f"{p} channel breakout",_finite(row[f"{p}_donchian_high"]))
            return Decision(sid,0.0,"hold_cash","no qualified Donchian breakout")
        ex=exit_common(p)
        if ex:return ex
        if held>=minhold and float(row["close"])<_finite(row[f"{p}_donchian_low"],-float("inf")): return Decision(sid,0.0,"exit_channel","lower channel break")
        return Decision(sid,1.0,"hold_btc","Donchian position")
    if typ=="ema_pullback":
        higher=strategy.get("higher","h1"); trigger=strategy.get("trigger","5m")
        trend=_finite(row[f"{higher}_trend_bps"]); adx=_finite(row[f"{higher}_adx"])
        if prev<=0:
            if trend<float(strategy.get("min_trend_bps",0)) or adx<float(strategy.get("min_adx",18)): return Decision(sid,0.0,"hold_cash","higher trend filter")
            if trigger=="5m": ok=_finite(row["pullback_5m_bps"],999)<=float(strategy.get("pullback_bps",-5)) and bool(row["recovery_5m"])
            else: ok=float(row["close"])>_finite(row["m15_ema_fast"]) and float(row["close"])<=_finite(row["m15_ema_fast"])*(1+float(strategy.get("max_recovery_bps",20))/10000)
            return Decision(sid,1.0,"enter_pullback","trend pullback recovery") if ok else Decision(sid,0.0,"hold_cash","waiting pullback recovery")
        ex=exit_common("m15")
        if ex:return ex
        if held>=minhold and trend<float(strategy.get("exit_trend_bps",-10)): return Decision(sid,0.0,"exit_regime","higher trend reversed")
        return Decision(sid,1.0,"hold_btc","trend pullback position")
    if typ=="squeeze":
        if prev<=0:
            ok=bool(row["m15_squeeze_release"]) and float(row["close"])>_finite(row["m15_bb_upper"],float("inf")) and _finite(row["h1_trend_bps"])>=float(strategy.get("min_h1_trend_bps",0))
            return Decision(sid,1.0,"enter_squeeze","15m squeeze release") if ok else Decision(sid,0.0,"hold_cash","no bullish squeeze release")
        ex=exit_common("m15")
        if ex:return ex
        if held>=minhold and float(row["close"])<_finite(row["m15_ema_fast"]): return Decision(sid,0.0,"exit_squeeze","lost 15m EMA")
        return Decision(sid,1.0,"hold_btc","squeeze position")
    if typ=="rsi2":
        if prev<=0:
            oversold=_finite(row["rsi2_5m"],100)<=float(strategy.get("entry_rsi",8)); trend=_finite(row["h4_trend_bps"])>=float(strategy.get("min_h4_trend_bps",0)); recover=_finite(row["rsi2_5m"])>_finite(row.get("rsi2_5m_prev",row["rsi2_5m"]-1))
            return Decision(sid,1.0,"enter_rsi2","RSI2 oversold in bullish 4h regime") if oversold and trend else Decision(sid,0.0,"hold_cash","RSI2/filter not ready")
        if held>=minhold and _finite(row["rsi2_5m"])>=float(strategy.get("exit_rsi",70)): return Decision(sid,0.0,"exit_rsi2","RSI2 recovered")
        ex=exit_common("m15")
        return ex or Decision(sid,1.0,"hold_btc","RSI2 reversion position")
    if typ=="macd_adx":
        bullish=_finite(row["m15_macd"])>_finite(row["m15_macd_signal"]) and _finite(row["m15_adx"])>=float(strategy.get("min_adx",20)) and _finite(row["h1_trend_bps"])>=0
        if prev<=0:return Decision(sid,1.0,"enter_macd","MACD/ADX bullish") if bullish else Decision(sid,0.0,"hold_cash","MACD/ADX filter")
        ex=exit_common("m15")
        if ex:return ex
        if held>=minhold and _finite(row["m15_macd"])<_finite(row["m15_macd_signal"]): return Decision(sid,0.0,"exit_macd","MACD bearish cross")
        return Decision(sid,1.0,"hold_btc","MACD position")
    raise ValueError(f"Unknown popular strategy type: {typ}")


def _equity(state: StrategyState, mark: float) -> float:
    return state.cash + state.btc * mark


def _gross_equity(state: StrategyState, mark: float) -> float:
    return state.gross_cash + state.gross_btc * mark


def execute_decision(
    state: StrategyState,
    decision: Decision,
    signal_row: pd.Series,
    next_row: pd.Series,
    signal_index: int,
    costs: dict[str, Any],
) -> tuple[dict[str, Any], SimulatedTrade | None]:
    mark = float(next_row["open"])
    fee_rate = float(costs.get("fee_bps_per_side", 0.0)) / 10_000.0
    spread_rate = float(costs.get("assumed_spread_bps_per_side", 0.0)) / 10_000.0
    slippage_rate = float(costs.get("slippage_bps_per_side", 0.0)) / 10_000.0
    target = min(1.0, max(0.0, float(decision.target_position)))

    equity_before = _equity(state, mark)
    gross_equity_before = _gross_equity(state, mark)
    desired_btc = equity_before * target / mark
    delta_btc = desired_btc - state.btc
    min_notional = float(costs.get("min_notional", 10.0))
    tolerance = equity_before * float(costs.get("rebalance_tolerance_bps", 1.0)) / 10_000.0

    transaction: SimulatedTrade | None = None
    execution_index = signal_index + 1
    if abs(delta_btc) * mark >= max(min_notional, tolerance):
        if delta_btc > 0.0:
            side = "buy"
            reference = mark * (1.0 + spread_rate)
            fill = reference * (1.0 + slippage_rate)
            affordable = state.cash / (fill * (1.0 + fee_rate))
            executed_btc = min(delta_btc, affordable)
        else:
            side = "sell"
            reference = mark * (1.0 - spread_rate)
            fill = reference * (1.0 - slippage_rate)
            executed_btc = -min(abs(delta_btc), state.btc)

        notional = abs(executed_btc) * fill
        if notional >= min_notional and abs(executed_btc) > 1e-15:
            fee = notional * fee_rate
            spread_cost = abs(executed_btc) * abs(reference - mark)
            slippage_cost = abs(executed_btc) * abs(fill - reference)
            if executed_btc > 0.0:
                state.cash -= notional + fee
                state.btc += executed_btc
                state.entry_index = execution_index
                state.entry_mark = mark
                state.highest_high = mark
                state.entry_reference = decision.entry_reference or mark
            else:
                state.cash += notional - fee
                state.btc += executed_btc
                if abs(state.btc) < 1e-12:
                    state.btc = 0.0
                state.entry_index = None
                state.entry_mark = None
                state.highest_high = None
                state.entry_reference = None
                state.pending_entry_index = None
            state.total_fees += fee
            state.total_spread += spread_cost
            state.total_slippage += slippage_cost
            state.total_turnover += notional
            state.trade_count += 1
            state.last_trade_index = execution_index
            transaction = SimulatedTrade(
                strategy_id=state.strategy_id,
                signal_timestamp=pd.Timestamp(signal_row["timestamp"]).isoformat(),
                execution_timestamp=pd.Timestamp(next_row["timestamp"]).isoformat(),
                side=side,
                mark_price=mark,
                reference_price=reference,
                fill_price=fill,
                btc_delta=executed_btc,
                gross_notional=notional,
                fee=fee,
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
                cash_after=state.cash,
                btc_after=state.btc,
                equity_after=_equity(state, mark),
                signal=decision.signal,
                reason=decision.reason,
            )

    desired_gross_btc = gross_equity_before * target / mark
    gross_delta = desired_gross_btc - state.gross_btc
    state.gross_cash -= gross_delta * mark
    state.gross_btc += gross_delta
    state.target_position = target

    equity = _equity(state, mark)
    gross_equity = _gross_equity(state, mark)
    state.peak_equity = max(float(state.peak_equity or state.initial_cash), equity)
    drawdown = equity / state.peak_equity - 1.0

    snapshot = {
        "strategy_id": state.strategy_id,
        "bar_timestamp": pd.Timestamp(signal_row["timestamp"]).isoformat(),
        "execution_timestamp": pd.Timestamp(next_row["timestamp"]).isoformat(),
        "mark_price": mark,
        "signal_close": float(signal_row["close"]),
        "signal": decision.signal,
        "reason": decision.reason,
        "target_position": target,
        "cash": state.cash,
        "btc": state.btc,
        "equity": equity,
        "gross_equity": gross_equity,
        "return_pct": equity / state.initial_cash - 1.0,
        "gross_return_pct": gross_equity / state.initial_cash - 1.0,
        "drawdown": drawdown,
        "cost_drag": gross_equity - equity,
        "trade_count": state.trade_count,
        "total_fees": state.total_fees,
        "total_spread": state.total_spread,
        "total_slippage": state.total_slippage,
        "total_turnover": state.total_turnover,
        "bars_held": _bars_held(state, execution_index),
        "streak_direction": int(signal_row["streak_direction"]),
        "streak_length": int(signal_row["streak_length"]),
        "run_return_bps": _finite(signal_row["run_return_bps"]),
        "range_atr_ratio": _finite(signal_row["range_atr_ratio"]),
        "relative_volume": _finite(signal_row["relative_volume"]),
        "vwap_zscore": _finite(signal_row["vwap_zscore"]),
        "one_hour_ema_spread_bps": _finite(signal_row["one_hour_ema_spread_bps"]),
        "one_hour_momentum_bps": _finite(signal_row["one_hour_momentum_bps"]),
        "four_hour_momentum_bps": _finite(signal_row["four_hour_momentum_bps"]),
    }
    return snapshot, transaction


def build_trade_episodes(
    snapshots: pd.DataFrame,
    initial_cash: float,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    columns = [
        "strategy_id",
        "entry_timestamp",
        "exit_timestamp",
        "is_open",
        "holding_bars",
        "holding_minutes",
        "entry_mark",
        "exit_mark",
        "gross_price_return_pct",
        "net_portfolio_return_pct",
        "maximum_adverse_excursion_pct",
        "maximum_favorable_excursion_pct",
    ]
    if snapshots.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for strategy_id, group in snapshots.groupby("strategy_id", sort=False):
        group = group.sort_values("execution_timestamp").reset_index(drop=True)
        in_position = False
        entry_timestamp: str | None = None
        entry_mark = 0.0
        equity_before_entry = initial_cash
        min_mark = 0.0
        max_mark = 0.0
        holding_bars = 0
        previous_equity = initial_cash

        for record in group.to_dict("records"):
            target = float(record["target_position"])
            mark = float(record["mark_price"])
            if not in_position and target > 0.0:
                in_position = True
                entry_timestamp = str(record["execution_timestamp"])
                entry_mark = mark
                equity_before_entry = previous_equity
                min_mark = mark
                max_mark = mark
                holding_bars = 1
            elif in_position and target > 0.0:
                holding_bars += 1
                min_mark = min(min_mark, mark)
                max_mark = max(max_mark, mark)
            elif in_position and target <= 0.0:
                min_mark = min(min_mark, mark)
                max_mark = max(max_mark, mark)
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "entry_timestamp": entry_timestamp,
                        "exit_timestamp": str(record["execution_timestamp"]),
                        "is_open": False,
                        "holding_bars": holding_bars,
                        "holding_minutes": holding_bars * timeframe_minutes,
                        "entry_mark": entry_mark,
                        "exit_mark": mark,
                        "gross_price_return_pct": mark / entry_mark - 1.0,
                        "net_portfolio_return_pct": float(record["equity"]) / equity_before_entry - 1.0,
                        "maximum_adverse_excursion_pct": min_mark / entry_mark - 1.0,
                        "maximum_favorable_excursion_pct": max_mark / entry_mark - 1.0,
                    }
                )
                in_position = False
                entry_timestamp = None
                holding_bars = 0
            previous_equity = float(record["equity"])

        if in_position:
            final = group.iloc[-1]
            final_mark = float(final["mark_price"])
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "entry_timestamp": entry_timestamp,
                    "exit_timestamp": None,
                    "is_open": True,
                    "holding_bars": holding_bars,
                    "holding_minutes": holding_bars * timeframe_minutes,
                    "entry_mark": entry_mark,
                    "exit_mark": final_mark,
                    "gross_price_return_pct": final_mark / entry_mark - 1.0,
                    "net_portfolio_return_pct": float(final["equity"]) / equity_before_entry - 1.0,
                    "maximum_adverse_excursion_pct": min_mark / entry_mark - 1.0,
                    "maximum_favorable_excursion_pct": max_mark / entry_mark - 1.0,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _summarize(
    snapshots: pd.DataFrame,
    transactions: pd.DataFrame,
    episodes: pd.DataFrame,
    initial_cash: float,
    sample_days: float,
    costs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    if snapshots.empty:
        return summary

    all_in = (
        float(costs.get("fee_bps_per_side", 0.0))
        + float(costs.get("slippage_bps_per_side", 0.0))
        + float(costs.get("assumed_spread_bps_per_side", 0.0))
    )
    for strategy_id, group in snapshots.groupby("strategy_id"):
        group = group.sort_values("execution_timestamp")
        last = group.iloc[-1]
        turnover = float(last["total_turnover"])
        gross_profit = float(last["gross_equity"]) - initial_cash
        break_even = max(0.0, gross_profit) / turnover * 10_000.0 if turnover > 0.0 else 0.0
        strategy_episodes = episodes[episodes["strategy_id"] == strategy_id]
        closed = strategy_episodes[~strategy_episodes["is_open"]]
        strategy_transactions = transactions[transactions["strategy_id"] == strategy_id] if not transactions.empty else pd.DataFrame()
        summary[strategy_id] = {
            "ending_equity": float(last["equity"]),
            "ending_gross_equity": float(last["gross_equity"]),
            "net_return_pct": float(last["return_pct"]),
            "gross_return_pct": float(last["gross_return_pct"]),
            "max_drawdown": float(group["drawdown"].min()),
            "transaction_count": int(last["trade_count"]),
            "round_trips_closed": int(len(closed)),
            "transactions_per_day": float(last["trade_count"]) / sample_days,
            "total_fees": float(last["total_fees"]),
            "total_spread": float(last["total_spread"]),
            "total_slippage": float(last["total_slippage"]),
            "total_turnover": turnover,
            "turnover_multiple": turnover / initial_cash,
            "cost_drag": float(last["cost_drag"]),
            "gross_break_even_all_in_bps_per_side": break_even,
            "assumed_all_in_bps_per_side": all_in,
            "cost_to_break_even_multiple": all_in / break_even if break_even > 0.0 else None,
            "average_holding_bars": float(closed["holding_bars"].mean()) if not closed.empty else None,
            "median_holding_bars": float(closed["holding_bars"].median()) if not closed.empty else None,
            "average_gross_return_per_round_trip": float(closed["gross_price_return_pct"].mean()) if not closed.empty else None,
            "average_net_return_per_round_trip": float(closed["net_portfolio_return_pct"].mean()) if not closed.empty else None,
            "round_trip_win_rate": float((closed["net_portfolio_return_pct"] > 0.0).mean()) if not closed.empty else None,
            "average_mae_pct": float(closed["maximum_adverse_excursion_pct"].mean()) if not closed.empty else None,
            "average_mfe_pct": float(closed["maximum_favorable_excursion_pct"].mean()) if not closed.empty else None,
            "buy_transactions": int((strategy_transactions.get("side", pd.Series(dtype=str)) == "buy").sum()) if not strategy_transactions.empty else 0,
            "sell_transactions": int((strategy_transactions.get("side", pd.Series(dtype=str)) == "sell").sum()) if not strategy_transactions.empty else 0,
        }

    buy_hold_return = summary.get("buy_hold_5m", {}).get("net_return_pct")
    if buy_hold_return is not None:
        for metrics in summary.values():
            metrics["net_excess_return_vs_buy_hold"] = metrics["net_return_pct"] - buy_hold_return
    return summary


def simulate_matrix(
    feature_frame: pd.DataFrame,
    cfg: dict[str, Any],
    costs_override: dict[str, Any] | None = None,
) -> SimulationResult:
    costs = copy.deepcopy(costs_override if costs_override is not None else cfg["costs"])
    strategies = cfg["strategies"]
    initial_cash = float(cfg["research"].get("initial_cash", 500.0))
    states = {
        str(strategy["id"]): StrategyState(
            strategy_id=str(strategy["id"]),
            initial_cash=initial_cash,
            cash=initial_cash,
            btc=0.0,
            gross_cash=initial_cash,
            gross_btc=0.0,
            peak_equity=initial_cash,
        )
        for strategy in strategies
    }

    valid_indices = feature_frame.index[feature_frame["feature_valid"]].tolist()
    if not valid_indices:
        raise RuntimeError("No rows contain a complete feature history.")
    start_index = valid_indices[0]
    snapshots: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []

    for index in range(start_index, len(feature_frame) - 1):
        row = feature_frame.iloc[index]
        next_row = feature_frame.iloc[index + 1]
        if not bool(row["feature_valid"]):
            continue
        for strategy in strategies:
            strategy_id = str(strategy["id"])
            state = states[strategy_id]
            if state.target_position > 0.0:
                state.highest_high = max(_finite(state.highest_high), float(row["high"]))
            decision = decide_strategy(strategy, row, state, index, costs)
            snapshot, transaction = execute_decision(
                state,
                decision,
                row,
                next_row,
                index,
                costs,
            )
            snapshots.append(snapshot)
            if transaction is not None:
                transactions.append(asdict(transaction))

    snapshot_frame = pd.DataFrame(snapshots)
    transaction_frame = pd.DataFrame(transactions)
    timeframe_minutes = int(timeframe_to_timedelta(str(cfg["market"]["timeframe"])).total_seconds() // 60)
    episodes = build_trade_episodes(snapshot_frame, initial_cash, timeframe_minutes)

    first_execution = pd.Timestamp(snapshot_frame["execution_timestamp"].min())
    last_execution = pd.Timestamp(snapshot_frame["execution_timestamp"].max())
    sample_days = max((last_execution - first_execution).total_seconds() / 86_400.0, 1e-9)
    summary = _summarize(
        snapshot_frame,
        transaction_frame,
        episodes,
        initial_cash,
        sample_days,
        costs,
    )
    return SimulationResult(snapshot_frame, transaction_frame, episodes, summary)


def _cost_sensitivity(
    feature_frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    grid = cfg["research"].get("cost_sensitivity_all_in_bps_per_side", [3, 5, 8, 10, 12])
    rows: list[dict[str, Any]] = []
    for all_in in grid:
        override = {
            "fee_bps_per_side": float(all_in),
            "slippage_bps_per_side": 0.0,
            "assumed_spread_bps_per_side": 0.0,
            "min_notional": float(cfg["costs"].get("min_notional", 10.0)),
            "rebalance_tolerance_bps": float(cfg["costs"].get("rebalance_tolerance_bps", 1.0)),
        }
        result = simulate_matrix(feature_frame, cfg, costs_override=override)
        for strategy_id, metrics in result.summary.items():
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "all_in_bps_per_side": float(all_in),
                    "ending_equity": metrics["ending_equity"],
                    "net_return_pct": metrics["net_return_pct"],
                    "max_drawdown": metrics["max_drawdown"],
                    "transaction_count": metrics["transaction_count"],
                    "round_trips_closed": metrics["round_trips_closed"],
                    "turnover_multiple": metrics["turnover_multiple"],
                    "average_holding_bars": metrics["average_holding_bars"],
                }
            )
    return pd.DataFrame(rows)


def _period_slices(
    feature_frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    fold_count = int(cfg["research"].get("chronological_folds", 5))
    if fold_count <= 1:
        return pd.DataFrame()
    valid = feature_frame[feature_frame["feature_valid"]].copy().reset_index(drop=True)
    if len(valid) < fold_count * 100:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    boundaries = np.linspace(0, len(valid), fold_count + 1, dtype=int)
    history_rows = int(cfg["research"].get("fold_history_rows", 1000))
    for fold in range(fold_count):
        score_start = boundaries[fold]
        score_end = boundaries[fold + 1]
        history_start = max(0, score_start - history_rows)
        fold_frame = valid.iloc[history_start:score_end].copy().reset_index(drop=True)
        # Only score the fold itself. Prior rows remain as feature history but are
        # marked invalid so the simulator starts with fresh capital at the fold.
        fold_frame.loc[: score_start - history_start - 1, "feature_valid"] = False
        if int(fold_frame["feature_valid"].sum()) < 2:
            continue
        result = simulate_matrix(fold_frame, cfg)
        score_rows = fold_frame[fold_frame["feature_valid"]]
        for strategy_id, metrics in result.summary.items():
            rows.append(
                {
                    "fold": fold + 1,
                    "strategy_id": strategy_id,
                    "first_signal_bar": pd.Timestamp(score_rows["timestamp"].iloc[0]).isoformat(),
                    "last_signal_bar": pd.Timestamp(score_rows["timestamp"].iloc[-1]).isoformat(),
                    "bars": int(len(score_rows)),
                    "net_return_pct": metrics["net_return_pct"],
                    "gross_return_pct": metrics["gross_return_pct"],
                    "max_drawdown": metrics["max_drawdown"],
                    "transaction_count": metrics["transaction_count"],
                    "turnover_multiple": metrics["turnover_multiple"],
                }
            )
    return pd.DataFrame(rows)


def _save_charts(snapshots: pd.DataFrame, output: Path) -> None:
    equity = snapshots.pivot_table(
        index="execution_timestamp",
        columns="strategy_id",
        values="equity",
        aggfunc="last",
    )
    equity.index = pd.to_datetime(equity.index, utc=True)
    figure, axis = plt.subplots(figsize=(12, 7))
    equity.plot(ax=axis)
    axis.set_title("Intraday strategy matrix: net equity")
    axis.set_xlabel("Execution time (UTC)")
    axis.set_ylabel("Portfolio value")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "equity_curve.png", dpi=150)
    plt.close(figure)

    drawdown = snapshots.pivot_table(
        index="execution_timestamp",
        columns="strategy_id",
        values="drawdown",
        aggfunc="last",
    )
    drawdown.index = pd.to_datetime(drawdown.index, utc=True)
    figure, axis = plt.subplots(figsize=(12, 7))
    drawdown.plot(ax=axis)
    axis.set_title("Intraday strategy matrix: drawdown")
    axis.set_xlabel("Execution time (UTC)")
    axis.set_ylabel("Drawdown")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "drawdown_curve.png", dpi=150)
    plt.close(figure)


def run_research(
    config_path: str,
    bars_requested: int,
    output_dir: str,
    data_path: str | None = None,
    skip_cost_grid: bool = False,
) -> dict[str, Any]:
    cfg = load_matrix_config(config_path)
    market = cfg["market"]
    timeframe = str(market["timeframe"])
    if data_path:
        raw, _ = load_ohlcv_csv(data_path, timeframe=timeframe)
        raw = raw.tail(bars_requested).reset_index(drop=True)
    else:
        step = timeframe_to_timedelta(timeframe)
        start = (pd.Timestamp.now(tz="UTC") - step * (bars_requested + 50)).isoformat()
        raw = download_ohlcv(
            exchange_id=str(market["exchange"]),
            symbol=str(market["symbol"]),
            timeframe=timeframe,
            start=start,
            max_bars=bars_requested,
        )

    feature_frame = build_feature_frame(raw, cfg)
    result = simulate_matrix(feature_frame, cfg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feature_frame.to_csv(output / "feature_frame.csv", index=False)
    result.snapshots.to_csv(output / "strategy_equity.csv", index=False)
    result.transactions.to_csv(output / "transactions.csv", index=False)
    result.episodes.to_csv(output / "trade_episodes.csv", index=False)

    signal_counts = (
        result.snapshots.groupby(["strategy_id", "signal"])
        .size()
        .rename("count")
        .reset_index()
    )
    signal_counts.to_csv(output / "signal_counts.csv", index=False)

    comparison = pd.DataFrame.from_dict(result.summary, orient="index")
    comparison.index.name = "strategy_id"
    comparison.reset_index().to_csv(output / "strategy_comparison.csv", index=False)

    if not skip_cost_grid:
        _cost_sensitivity(feature_frame, cfg).to_csv(output / "cost_sensitivity.csv", index=False)
    period_slices = _period_slices(feature_frame, cfg)
    if not period_slices.empty:
        period_slices.to_csv(output / "chronological_folds.csv", index=False)

    _save_charts(result.snapshots, output)

    first_bar = pd.Timestamp(raw["timestamp"].iloc[0])
    last_bar = pd.Timestamp(raw["timestamp"].iloc[-1])
    valid = feature_frame[feature_frame["feature_valid"]]
    first_scored = pd.Timestamp(valid["timestamp"].iloc[0])
    last_scored = pd.Timestamp(valid["timestamp"].iloc[-1])
    sample_days = max((last_scored - first_scored).total_seconds() / 86_400.0, 1e-9)
    summary = {
        "market": {
            "exchange": str(market["exchange"]),
            "symbol": str(market["symbol"]),
            "timeframe": timeframe,
        },
        "downloaded_bars": int(len(raw)),
        "first_downloaded_bar": first_bar.isoformat(),
        "last_downloaded_bar": last_bar.isoformat(),
        "scored_bars": int(len(valid) - 1),
        "first_scored_bar": first_scored.isoformat(),
        "last_scored_bar": last_scored.isoformat(),
        "sample_days": sample_days,
        "cost_assumptions": {
            "fee_bps_per_side": float(cfg["costs"].get("fee_bps_per_side", 0.0)),
            "slippage_bps_per_side": float(cfg["costs"].get("slippage_bps_per_side", 0.0)),
            "assumed_spread_bps_per_side": float(cfg["costs"].get("assumed_spread_bps_per_side", 0.0)),
            "all_in_bps_per_side": (
                float(cfg["costs"].get("fee_bps_per_side", 0.0))
                + float(cfg["costs"].get("slippage_bps_per_side", 0.0))
                + float(cfg["costs"].get("assumed_spread_bps_per_side", 0.0))
            ),
        },
        "strategies": result.summary,
        "outputs": {
            "strategy_comparison": "strategy_comparison.csv",
            "cost_sensitivity": None if skip_cost_grid else "cost_sensitivity.csv",
            "chronological_folds": "chronological_folds.csv" if not period_slices.empty else None,
            "trade_episodes": "trade_episodes.csv",
            "transactions": "transactions.csv",
            "equity_curve": "equity_curve.png",
            "drawdown_curve": "drawdown_curve.png",
        },
    }
    (output / "research_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Research-only BTC intraday strategy matrix"
    )
    parser.add_argument(
        "--config",
        default="config/settings_popular_matrix.yaml",
    )
    parser.add_argument("--bars", type=int, default=10_000)
    parser.add_argument("--output", default="outputs/popular_matrix_10000")
    parser.add_argument("--data", default=None, help="Optional normalized OHLCV CSV")
    parser.add_argument("--skip-cost-grid", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_research(
        config_path=args.config,
        bars_requested=args.bars,
        output_dir=args.output,
        data_path=args.data,
        skip_cost_grid=args.skip_cost_grid,
    )


if __name__ == "__main__":
    main()
