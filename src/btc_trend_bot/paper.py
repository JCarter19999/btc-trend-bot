from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from btc_trend_bot.config import load_config
from btc_trend_bot.data import download_ohlcv, normalize_ohlcv, timeframe_to_timedelta
from btc_trend_bot.pipeline import prepare_strategy_frame


@dataclass
class PaperState:
    cash: float
    btc: float
    last_bar_timestamp: str | None = None
    halted: bool = False
    peak_equity: float | None = None


def load_state(path: Path, initial_cash: float) -> PaperState:
    if not path.exists():
        return PaperState(cash=initial_cash, btc=0.0, peak_equity=initial_cash)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return PaperState(**raw)


def save_state(path: Path, state: PaperState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(state), handle, indent=2)


def append_trade(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_paper_step(config_path: str = "config/settings.yaml") -> PaperState:
    cfg = load_config(config_path)
    market = cfg["market"]
    paper = cfg["paper"]
    backtest = cfg["backtest"]

    lookback_bars = int(paper["lookback_bars"])
    step = timeframe_to_timedelta(str(market["timeframe"]))
    start = (pd.Timestamp.now(tz="UTC") - step * (lookback_bars + 10)).isoformat()
    frame = download_ohlcv(
        exchange_id=str(market["exchange"]),
        symbol=str(market["symbol"]),
        timeframe=str(market["timeframe"]),
        start=start,
        max_bars=lookback_bars,
    )
    normalized, _ = normalize_ohlcv(frame, timeframe=str(market["timeframe"]))
    strategy_frame = prepare_strategy_frame(normalized, cfg)
    latest = strategy_frame.iloc[-1]
    timestamp = pd.Timestamp(latest["timestamp"]).isoformat()

    state_path = Path(str(paper["state_path"]))
    trades_path = Path(str(paper["trades_path"]))
    state = load_state(state_path, initial_cash=float(paper["initial_cash"]))
    if state.last_bar_timestamp == timestamp:
        print(f"Paper account already processed {timestamp}; no action taken.")
        return state

    price = float(latest["close"])
    equity_before = state.cash + state.btc * price
    state.peak_equity = max(float(state.peak_equity or equity_before), equity_before)
    current_drawdown = equity_before / state.peak_equity - 1.0
    if current_drawdown <= -float(backtest["max_drawdown_breaker"]):
        state.halted = True

    target = 0.0 if state.halted else max(0.0, float(latest["target_position"]))
    desired_btc = equity_before * target / price
    delta_btc = desired_btc - state.btc

    fee_rate = float(backtest["fee_bps_per_turnover"]) / 10_000.0
    slippage_rate = float(backtest["slippage_bps_per_turnover"]) / 10_000.0
    side = "buy" if delta_btc > 0 else "sell"
    fill_price = price * (1.0 + slippage_rate if delta_btc > 0 else 1.0 - slippage_rate)
    gross_notional = abs(delta_btc) * fill_price
    fee = gross_notional * fee_rate

    if abs(delta_btc) > 1e-10:
        if delta_btc > 0:
            affordable_btc = max(0.0, state.cash / (fill_price * (1.0 + fee_rate)))
            delta_btc = min(delta_btc, affordable_btc)
            gross_notional = delta_btc * fill_price
            fee = gross_notional * fee_rate
            state.cash -= gross_notional + fee
            state.btc += delta_btc
        else:
            sell_btc = min(abs(delta_btc), state.btc)
            gross_notional = sell_btc * fill_price
            fee = gross_notional * fee_rate
            state.cash += gross_notional - fee
            state.btc -= sell_btc
            delta_btc = -sell_btc

        append_trade(
            trades_path,
            {
                "timestamp": timestamp,
                "side": side,
                "btc_delta": delta_btc,
                "fill_price": fill_price,
                "fee": fee,
                "target_position": target,
                "cash_after": state.cash,
                "btc_after": state.btc,
            },
        )

    state.last_bar_timestamp = timestamp
    save_state(state_path, state)
    equity_after = state.cash + state.btc * price
    print(
        f"Processed {timestamp}: target={target:.3f}, side={side if abs(delta_btc) > 1e-10 else 'none'}, "
        f"cash=${state.cash:,.2f}, btc={state.btc:.8f}, equity=${equity_after:,.2f}, halted={state.halted}"
    )
    return state
