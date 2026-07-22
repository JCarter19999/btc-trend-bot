from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.v1.candlestick_geometry import CONTINUOUS_GEOMETRY_COLUMNS, add_candlestick_geometry

BASE_FEATURES = [
    "return_1", "return_2", "return_5", "return_10", "return_20",
    "ema_spread_atr", "ema_slope_atr", "atr_pct", "realized_vol_20",
    "relative_volume", "volume_zscore", "distance_20d_high_atr",
    "distance_20d_low_atr", "benchmark_return_5", "benchmark_return_20",
    "relative_strength_20", "market_above_ema50",
]
FEATURES = BASE_FEATURES + list(CONTINUOUS_GEOMETRY_COLUMNS)


@dataclass(frozen=True)
class BacktestConfig:
    symbols: tuple[str, ...]
    benchmark: str
    start: str
    end: str | None
    interval: str
    initial_capital: float
    return_threshold_bps: float
    max_hold_bars: int
    train_bars: int
    test_bars: int
    step_bars: int
    purge_bars: int
    ridge_alpha: float
    stop_atr: float
    target_atr: float
    slippage_bps_each_side: float
    safety_enabled: bool
    drawdown_pause: float
    hard_shutdown_drawdown: float
    consecutive_loss_limit: int
    cooldown_trades: int
    minimum_equity: float
    position_fraction: float = 0.25  # fraction of equity committed to each trade; rest sits in cash


def load_config(path: Path) -> BacktestConfig:
    raw = yaml.safe_load(path.read_text())
    safety = raw.get("safety", {})
    return BacktestConfig(
        symbols=tuple(raw["symbols"]), benchmark=str(raw.get("benchmark", "SPY")),
        start=str(raw["start"]), end=raw.get("end"), interval=str(raw.get("interval", "1d")),
        initial_capital=float(raw.get("initial_capital", 2500)),
        return_threshold_bps=float(raw.get("return_threshold_bps", 10)),
        max_hold_bars=int(raw.get("max_hold_bars", 10)), train_bars=int(raw.get("train_bars", 756)),
        test_bars=int(raw.get("test_bars", 126)), step_bars=int(raw.get("step_bars", 126)),
        purge_bars=int(raw.get("purge_bars", 10)), ridge_alpha=float(raw.get("ridge_alpha", 10)),
        stop_atr=float(raw.get("stop_atr", 1.35)), target_atr=float(raw.get("target_atr", 2.15)),
        slippage_bps_each_side=float(raw.get("slippage_bps_each_side", 2.0)),
        safety_enabled=bool(safety.get("enabled", True)), drawdown_pause=float(safety.get("drawdown_pause", .15)),
        hard_shutdown_drawdown=float(safety.get("hard_shutdown_drawdown", .35)),
        consecutive_loss_limit=int(safety.get("consecutive_loss_limit", 4)),
        cooldown_trades=int(safety.get("cooldown_trades", 8)), minimum_equity=float(safety.get("minimum_equity", 25)),
        position_fraction=float(raw.get("position_fraction", 0.25)),
    )


def _normalize(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    aliases = {"adj_close": "close"}
    out = out.rename(columns=aliases)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"{symbol}: missing columns {sorted(missing)}")
    if not isinstance(out.index, pd.DatetimeIndex):
        date_col = next((c for c in ("date", "datetime", "timestamp", "open_time") if c in out.columns), None)
        if not date_col:
            raise ValueError(f"{symbol}: CSV needs a datetime index or date column")
        out.index = pd.to_datetime(out.pop(date_col), utc=True)
    else:
        out.index = pd.to_datetime(out.index, utc=True)
    out = out[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    out = out.dropna().sort_index()
    out["symbol"] = symbol
    return out


def download_yfinance(symbols: tuple[str, ...], start: str, end: str | None, interval: str) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("Install yfinance: pip install yfinance>=0.2.54,<1") from exc
    raw = yf.download(list(symbols), start=start, end=end, interval=interval, auto_adjust=True,
                      actions=False, group_by="ticker", threads=True, progress=False, repair=True)
    if raw.empty:
        raise RuntimeError("No data returned by yfinance")
    result = {}
    for symbol in symbols:
        frame = raw[symbol] if isinstance(raw.columns, pd.MultiIndex) else raw
        result[symbol] = _normalize(frame, symbol)
    return result


def load_csv_dir(symbols: tuple[str, ...], directory: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
        path = directory / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames[symbol] = _normalize(pd.read_csv(path), symbol)
    return frames


def add_features(frame: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out.close
    prev = close.shift(1)
    tr = pd.concat([(out.high-out.low), (out.high-prev).abs(), (out.low-prev).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(span=20, adjust=False, min_periods=20).mean()
    for n in (1, 2, 5, 10, 20): out[f"return_{n}"] = close.pct_change(n)
    fast = close.ewm(span=12, adjust=False).mean(); slow = close.ewm(span=36, adjust=False).mean()
    out["ema_spread_atr"] = (fast-slow)/out.atr.replace(0, np.nan)
    out["ema_slope_atr"] = fast.diff(3)/out.atr.replace(0, np.nan)
    out["atr_pct"] = out.atr/close
    out["realized_vol_20"] = out.return_1.rolling(20).std(ddof=0)
    vm = out.volume.rolling(20).mean(); vs = out.volume.rolling(20).std(ddof=0)
    out["relative_volume"] = out.volume/vm.replace(0, np.nan)
    out["volume_zscore"] = (out.volume-vm)/vs.replace(0, np.nan)
    out["distance_20d_high_atr"] = (out.high.rolling(20).max()-close)/out.atr
    out["distance_20d_low_atr"] = (close-out.low.rolling(20).min())/out.atr
    bench = benchmark.close.reindex(out.index).ffill()
    out["benchmark_return_5"] = bench.pct_change(5)
    out["benchmark_return_20"] = bench.pct_change(20)
    out["relative_strength_20"] = out.return_20 - out.benchmark_return_20
    out["market_above_ema50"] = (bench > bench.ewm(span=50, adjust=False).mean()).astype(float)
    return add_candlestick_geometry(out)


def simulate_trade(frame: pd.DataFrame, i: int, cfg: BacktestConfig) -> dict | None:
    entry_i = i + 1
    if entry_i >= len(frame): return None
    entry = float(frame.iloc[entry_i].open) * (1 + cfg.slippage_bps_each_side/10000)
    atr = float(frame.iloc[i].atr)
    if not np.isfinite(atr) or atr <= 0: return None
    stop = entry - cfg.stop_atr*atr; target = entry + cfg.target_atr*atr
    exit_i = min(entry_i + cfg.max_hold_bars - 1, len(frame)-1)
    exit_price = float(frame.iloc[exit_i].close); reason = "time_exit"
    for j in range(entry_i, exit_i+1):
        row = frame.iloc[j]
        if float(row.low) <= stop and float(row.high) >= target:
            exit_price, exit_i, reason = stop, j, "ambiguous_stop_first"; break
        if float(row.low) <= stop:
            exit_price, exit_i, reason = stop, j, "stop"; break
        if float(row.high) >= target:
            exit_price, exit_i, reason = target, j, "target"; break
    exit_fill = exit_price * (1 - cfg.slippage_bps_each_side/10000)
    return {"entry_time": frame.index[entry_i], "exit_time": frame.index[exit_i], "entry_price": entry,
            "exit_price": exit_fill, "net_return": exit_fill/entry-1, "exit_reason": reason,
            "bars_held": exit_i-entry_i+1}


def build_candidates(frames: dict[str, pd.DataFrame], benchmark_symbol: str, cfg: BacktestConfig) -> pd.DataFrame:
    bench = frames[benchmark_symbol]
    rows = []
    for symbol in cfg.symbols:
        f = add_features(frames[symbol], bench)
        for i in range(60, len(f)-cfg.max_hold_bars-1):
            r = f.iloc[i]
            # broad, deterministic candidate generation; Ridge decides whether edge is sufficient.
            mask = (r.relative_volume > 0.6 and r.atr_pct > 0 and abs(r.ema_spread_atr) < 8 and r.return_20 > -0.25)
            if not mask: continue
            trade = simulate_trade(f, i, cfg)
            if trade is None: continue
            record = {"signal_time": f.index[i], "symbol": symbol, **trade}
            record.update({c: float(r[c]) for c in FEATURES if c in r and pd.notna(r[c])})
            if all(c in record and np.isfinite(record[c]) for c in FEATURES): rows.append(record)
    if not rows:
        return pd.DataFrame(columns=["signal_time", "symbol", *FEATURES, "net_return"])
    return pd.DataFrame(rows).sort_values(["signal_time", "symbol"]).reset_index(drop=True)


SELECTION_MODES = ("ridge", "random", "simple_trend")


def walk_forward(candidates: pd.DataFrame, cfg: BacktestConfig, shuffle_labels: bool = False, seed: int = 0,
                  selection: str = "ridge") -> tuple[pd.DataFrame, pd.DataFrame]:
    if selection not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode: {selection}")
    dates = np.array(sorted(candidates.signal_time.dt.normalize().unique()))
    selected=[]; folds=[]; fold=0
    start=cfg.train_bars
    rng=np.random.default_rng(seed)
    while start + cfg.test_bars <= len(dates):
        train_start=max(0,start-cfg.train_bars); train_end=max(train_start,start-cfg.purge_bars)
        train_dates=dates[train_start:train_end]; test_dates=dates[start:start+cfg.test_bars]
        train=candidates[candidates.signal_time.dt.normalize().isin(train_dates)].dropna(subset=FEATURES+["net_return"])
        test=candidates[candidates.signal_time.dt.normalize().isin(test_dates)].dropna(subset=FEATURES)
        needs_train = selection == "ridge"
        if (needs_train and len(train)<200) or test.empty: start += cfg.step_bars; continue
        test=test.copy(); test["fold"]=fold
        if selection == "ridge":
            # Sanity check: fitting on permuted labels severs any feature/outcome
            # relationship. If the strategy still looks profitable, the real (unshuffled)
            # result is not trustworthy evidence of learned edge.
            train_labels=rng.permutation(train.net_return.to_numpy()) if shuffle_labels else train.net_return
            model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=cfg.ridge_alpha))])
            model.fit(train[FEATURES],train_labels)
            test["predicted_return"]=model.predict(test[FEATURES])
            qualified=test[test.predicted_return >= cfg.return_threshold_bps/10000]
            winners=(qualified.sort_values(["signal_time","predicted_return"],ascending=[True,False])
                     .groupby("signal_time",as_index=False).head(1).sort_values("signal_time"))
        elif selection == "random":
            # Control: pick a uniformly random candidate per signal_time from the same
            # deterministic candidate pool Ridge would have chosen from, with no model
            # in the loop at all. If this performs comparably, the "edge" isn't coming
            # from candidate selection at all -- it's coming from the exit mechanics
            # and/or the symbol universe/period.
            test["predicted_return"]=np.nan
            winners=(test.assign(_r=rng.random(len(test)))
                          .sort_values(["signal_time","_r"])
                          .groupby("signal_time",as_index=False).head(1)
                          .drop(columns="_r").sort_values("signal_time"))
        else:  # simple_trend
            # Control: no ML, just pick the strongest-momentum candidate that's above
            # its 50-day benchmark trend, mirroring a simple trend-following rule.
            test["predicted_return"]=np.nan
            pool=test[test.market_above_ema50 >= 1] if "market_above_ema50" in test else test
            if pool.empty: pool=test
            winners=(pool.sort_values(["signal_time","relative_strength_20"],ascending=[True,False])
                          .groupby("signal_time",as_index=False).head(1).sort_values("signal_time"))
        selected.append(winners)
        folds.append({"fold":fold,"train_start":str(train_dates[0]) if len(train_dates) else "","train_end":str(train_dates[-1]) if len(train_dates) else "",
                      "test_start":str(test_dates[0]),"test_end":str(test_dates[-1]),"train_rows":len(train),
                      "test_rows":len(test),"selected_trades":len(winners),"mean_return":winners.net_return.mean() if len(winners) else np.nan})
        fold+=1; start += cfg.step_bars
    return pd.DataFrame(folds), pd.concat(selected,ignore_index=True) if selected else pd.DataFrame()


def simulate_capital(trades: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    equity=cfg.initial_capital; peak=equity; cooldown=0; dd_cooldown=0; losses=0; hard=False; rows=[]
    for _,t in trades.sort_values("signal_time").iterrows():
        dd=1-equity/peak
        reason=""
        if hard or equity<cfg.minimum_equity or dd>=cfg.hard_shutdown_drawdown:
            hard=True; reason="hard_shutdown"
        elif cfg.safety_enabled and cooldown>0:
            cooldown-=1; reason="loss_cooldown"
        elif cfg.safety_enabled and (dd_cooldown>0 or dd>=cfg.drawdown_pause):
            # Trading is halted, so equity/peak can't move on their own to clear the
            # drawdown condition. Count the pause down like loss_cooldown, then
            # re-anchor peak to current equity so it doesn't instantly re-trip.
            if dd_cooldown==0: dd_cooldown=cfg.cooldown_trades
            dd_cooldown-=1; reason="drawdown_pause"
            if dd_cooldown==0: peak=equity
        if reason:
            rows.append({"signal_time":t.signal_time,"symbol":t.symbol,"trade_taken":False,"skip_reason":reason,"ending_equity":equity,"drawdown":dd}); continue
        # Only cfg.position_fraction of equity is committed as trade notional; the
        # remainder sits in cash. Betting the full account on one sequential trade
        # after another compounds any structural edge (or noise) into path-dependent,
        # unrealistic terminal equity - see label-shuffle/random-selection controls.
        start=equity; notional=start*cfg.position_fraction; pnl=notional*float(t.net_return)
        equity=max(0,start+pnl); peak=max(peak,equity); dd=1-equity/peak
        if pnl<0:
            losses+=1
            if cfg.safety_enabled and losses>=cfg.consecutive_loss_limit: cooldown=cfg.cooldown_trades; losses=0
        else: losses=0
        rows.append({"signal_time":t.signal_time,"symbol":t.symbol,"trade_taken":True,"skip_reason":"","starting_equity":start,
                     "notional":notional,"trade_return":float(t.net_return),"trade_pnl":pnl,"ending_equity":equity,"drawdown":dd})
    path=pd.DataFrame(rows)
    taken=path[path.trade_taken] if len(path) else path
    summary={"initial_capital":cfg.initial_capital,"ending_equity":equity,"net_profit":equity-cfg.initial_capital,
             "total_return":equity/cfg.initial_capital-1,"max_drawdown":float(path.drawdown.max()) if len(path) else 0,
             "candidate_selected_trades":len(trades),"trades_taken":int(path.trade_taken.sum()) if len(path) else 0,
             "trades_skipped":int((~path.trade_taken).sum()) if len(path) else 0,"win_rate":float((taken.trade_return>0).mean()) if len(taken) else np.nan,
             "hard_stopped":hard}
    return path,summary


def benchmark_summary(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, capital: float) -> dict:
    x=frame.loc[(frame.index>=start)&(frame.index<=end)]
    if len(x)<2:return {}
    ret=float(x.close.iloc[-1]/x.close.iloc[0]-1)
    curve=x.close/x.close.iloc[0]; dd=float((1-curve/curve.cummax()).max())
    return {"return":ret,"ending_equity":capital*(1+ret),"max_drawdown":dd}


def main():
    p=argparse.ArgumentParser(description="Real-equity chronological Ridge walk-forward backtest")
    p.add_argument("--config",default="configs/real_data.yaml")
    p.add_argument("--provider",choices=["yfinance","csv"],default="yfinance")
    p.add_argument("--csv-dir",default="data/real")
    p.add_argument("--output",default="outputs/real_equity_walkforward")
    p.add_argument("--download-only",action="store_true")
    p.add_argument("--shuffle-labels",action="store_true",help="Sanity check: permute training labels per fold; a still-profitable result means the model isn't learning real signal.")
    p.add_argument("--seed",type=int,default=0,help="Seed for --shuffle-labels / --selection random.")
    p.add_argument("--selection",choices=SELECTION_MODES,default="ridge",help="Trade-selection control: 'random' or 'simple_trend' bypass the Ridge model entirely for comparison.")
    args=p.parse_args()
    cfg=load_config(ROOT/args.config if not Path(args.config).is_absolute() else Path(args.config))
    symbols=tuple(dict.fromkeys((*cfg.symbols,cfg.benchmark)))
    frames=download_yfinance(symbols,cfg.start,cfg.end,cfg.interval) if args.provider=="yfinance" else load_csv_dir(symbols,ROOT/args.csv_dir)
    rawdir=ROOT/"data"/"real"; rawdir.mkdir(parents=True,exist_ok=True)
    for s,f in frames.items(): f.drop(columns="symbol").to_csv(rawdir/f"{s}.csv",index_label="date")
    if args.download_only:
        print(f"Saved {len(frames)} symbols to {rawdir}"); return
    candidates=build_candidates(frames,cfg.benchmark,cfg)
    if candidates.empty: raise RuntimeError("No candidates generated")
    candidates.signal_time=pd.to_datetime(candidates.signal_time,utc=True)
    folds,trades=walk_forward(candidates,cfg,shuffle_labels=args.shuffle_labels,seed=args.seed,selection=args.selection)
    if trades.empty: raise RuntimeError("No trades selected; reduce threshold or increase history")
    path,summary=simulate_capital(trades,cfg)
    start=pd.Timestamp(trades.signal_time.min()); end=pd.Timestamp(trades.exit_time.max())
    benchmark=benchmark_summary(frames[cfg.benchmark],start,end,cfg.initial_capital)
    out=ROOT/args.output if not Path(args.output).is_absolute() else Path(args.output); out.mkdir(parents=True,exist_ok=True)
    candidates.to_parquet(out/"candidates.parquet",index=False); folds.to_csv(out/"folds.csv",index=False)
    trades.to_csv(out/"selected_trades.csv",index=False); path.to_csv(out/"capital_path.csv",index=False)
    report={"configuration":{**cfg.__dict__,"shuffle_labels":args.shuffle_labels,"seed":args.seed,"selection":args.selection},"strategy":summary,"benchmark":benchmark,"options_backtest":"disabled: requires historical option-chain data"}
    (out/"results.json").write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,indent=2,default=str))

if __name__=="__main__": main()
