"""Live paper trading for the two-tier safety-layer candidate (see
TWO_TIER_SAFETY_LAYER.md and TWO_TIER_SAFETY_LAYER_RESULTS.md). Runs the SAME
frozen strategy signal as production (btc_trend_bot.pipeline.prepare_strategy_frame,
completely unchanged) against live public market data, but manages the locally
simulated paper account through the two-tier safety state machine validated in
run_two_tier_safety_oos_validation.py, instead of paper.py's single
always-on-or-off breaker (which is disabled in production anyway --
max_drawdown_breaker: 0.0).

This is a pure local simulation, exactly like btc_trend_bot.paper.run_paper_step:
it downloads public OHLCV (no exchange keys, no account access) and tracks a
simulated cash/BTC balance. It is a SEPARATE, independent paper account from
the frozen paper.py path (its own state/trades/run-log files) -- running this
never touches production.py, paper.py, cli.py, or config/settings_production.yaml.
There is no live-order code path here at all.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from btc_trend_bot.config import load_config  # noqa: E402
from btc_trend_bot.data import download_ohlcv, normalize_ohlcv, timeframe_to_timedelta  # noqa: E402
from btc_trend_bot.notifications import notify  # noqa: E402
from btc_trend_bot.pipeline import prepare_strategy_frame  # noqa: E402
from run_two_tier_safety_backtest import SafetyConfig  # noqa: E402

# Validated in run_two_tier_safety_oos_validation.py: chosen from a train-only
# (2018-2021) sweep for a stable neighborhood, then confirmed on held-out
# 2022-2026 data before being frozen here. See TWO_TIER_SAFETY_LAYER_RESULTS.md.
CHOSEN_SAFETY = SafetyConfig(
    drawdown_pause=0.15,
    hard_shutdown_drawdown=0.35,
    minimum_equity_fraction=0.01,
    consecutive_loss_limit_bars=8,
    cooldown_bars=30,
)

STATE_PATH = ROOT / "paper" / "two_tier_safety_state.json"
TRADES_PATH = ROOT / "paper" / "two_tier_safety_trades.csv"
RUNS_LOG_PATH = ROOT / "paper" / "two_tier_safety_runs.jsonl"
LOOKBACK_BARS = 900


@dataclass
class TwoTierPaperState:
    cash: float
    btc: float
    last_bar_timestamp: str | None = None
    peak_equity: float | None = None
    last_equity: float | None = None
    hard_halted: bool = False
    hard_halt_reason: str | None = None
    loss_cooldown_remaining: int = 0
    dd_cooldown_remaining: int = 0
    consecutive_losses: int = 0


def load_state(path: Path, initial_cash: float) -> TwoTierPaperState:
    if not path.exists():
        return TwoTierPaperState(cash=initial_cash, btc=0.0, peak_equity=initial_cash)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return TwoTierPaperState(**raw)


def save_state(path: Path, state: TwoTierPaperState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(state), handle, indent=2)


def append_trade(path: Path, row: dict) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_run_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def apply_bar(state: TwoTierPaperState, *, timestamp: str, price: float, raw_target: float,
              safety: SafetyConfig, fee_rate: float, slippage_rate: float,
              minimum_equity: float) -> tuple[TwoTierPaperState, dict | None, dict]:
    """Pure per-bar state transition (mutates and returns `state`): decides
    THIS bar's position from state carried over from the PRIOR bar only (no
    lookahead), simulates the fill exactly like paper.py's fee/slippage math,
    then updates the safety state for the NEXT call from this bar's realized
    outcome -- same causal ordering and priority (hard shutdown > loss
    cooldown > drawdown pause) as
    run_two_tier_safety_backtest.run_backtest_with_two_tier_safety's per-bar
    loop, just one bar per call instead of looped over a whole frame.
    Returns (state, trade_row_or_None, run_row)."""
    equity_before = state.cash + state.btc * price
    peak = max(float(state.peak_equity or equity_before), equity_before)

    if state.hard_halted:
        target, reason = 0.0, "hard_shutdown"
    elif state.loss_cooldown_remaining > 0:
        target, reason = 0.0, "loss_cooldown"
        state.loss_cooldown_remaining -= 1
    elif state.dd_cooldown_remaining > 0:
        target, reason = 0.0, "drawdown_pause"
        state.dd_cooldown_remaining -= 1
        if state.dd_cooldown_remaining == 0:
            peak = equity_before  # re-anchor so the pause doesn't instantly re-trip
    else:
        target, reason = raw_target, ""

    desired_btc = equity_before * target / price
    delta_btc = desired_btc - state.btc
    side = "buy" if delta_btc > 0 else "sell"
    fill_price = price * (1.0 + slippage_rate if delta_btc > 0 else 1.0 - slippage_rate)
    fee = 0.0

    trade_row = None
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

        trade_row = {
            "timestamp": timestamp, "side": side, "btc_delta": delta_btc,
            "fill_price": fill_price, "fee": fee, "target_position": target,
            "safety_reason": reason, "cash_after": state.cash, "btc_after": state.btc,
        }

    equity_after = state.cash + state.btc * price
    peak = max(peak, equity_after)
    drawdown = equity_after / peak - 1.0
    # This bar's realized P&L must be measured against the PREVIOUS call's
    # ending equity, not this call's own pre-trade equity_before -- the
    # latter is already marked-to-market at today's price, so it would only
    # ever capture this bar's transaction cost and completely miss the
    # market move on a position held unchanged since the last call.
    previous_equity = state.last_equity if state.last_equity is not None else equity_before
    net_return = equity_after / previous_equity - 1.0 if previous_equity > 0 else 0.0
    state.last_equity = equity_after

    if net_return < 0:
        state.consecutive_losses += 1
    else:
        state.consecutive_losses = 0

    if not state.hard_halted:
        if equity_after < minimum_equity or drawdown <= -safety.hard_shutdown_drawdown:
            state.hard_halted = True
            state.hard_halt_reason = (
                f"equity ${equity_after:,.2f} below floor ${minimum_equity:,.2f}"
                if equity_after < minimum_equity else f"drawdown {drawdown:.1%} breached hard shutdown"
            )
            notify(f"URGENT: two-tier safety BTC paper account HARD SHUTDOWN: {state.hard_halt_reason}")
        elif state.loss_cooldown_remaining == 0 and state.consecutive_losses >= safety.consecutive_loss_limit_bars:
            state.loss_cooldown_remaining = safety.cooldown_bars
            state.consecutive_losses = 0
        elif state.dd_cooldown_remaining == 0 and drawdown <= -safety.drawdown_pause:
            state.dd_cooldown_remaining = safety.cooldown_bars

    state.last_bar_timestamp = timestamp
    state.peak_equity = peak

    run_row = {
        "timestamp": timestamp, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "close": price, "raw_target": raw_target, "applied_target": target, "safety_reason": reason,
        "side": side if abs(delta_btc) > 1e-10 else "none",
        "cash": state.cash, "btc": state.btc, "equity": equity_after, "drawdown": drawdown,
        "hard_halted": state.hard_halted, "loss_cooldown_remaining": state.loss_cooldown_remaining,
        "dd_cooldown_remaining": state.dd_cooldown_remaining,
    }
    return state, trade_row, run_row


def run_two_tier_paper_step(config_path: str = "config/settings_production.yaml",
                             safety: SafetyConfig = CHOSEN_SAFETY) -> TwoTierPaperState:
    """One idempotent 4h-bar cycle: fetch the latest live bar, then delegate
    the actual decision/fill/safety-update to apply_bar (see its docstring)."""
    cfg = load_config(config_path)
    market = cfg["market"]
    paper = cfg["paper"]
    backtest = cfg["backtest"]

    step = timeframe_to_timedelta(str(market["timeframe"]))
    start = (pd.Timestamp.now(tz="UTC") - step * (LOOKBACK_BARS + 10)).isoformat()
    # max_bars must include the same +10 buffer as the start offset above, or
    # download_ohlcv's row-count cap is reached before catching up to "now"
    # -- see TWO_TIER_SAFETY_LAYER.md: this exact off-by-10 mismatch is a real,
    # pre-existing bug in the frozen btc_trend_bot.paper.run_paper_step
    # (max_bars=lookback_bars, no +10), left uncorrected there since paper.py
    # is not part of this candidate. production.py's _latest_signal has the
    # correct version (max_bars=lookback_bars + 10), matched here.
    frame = download_ohlcv(
        exchange_id=str(market["exchange"]), symbol=str(market["symbol"]),
        timeframe=str(market["timeframe"]), start=start, max_bars=LOOKBACK_BARS + 10,
    )
    normalized, _ = normalize_ohlcv(frame, timeframe=str(market["timeframe"]))
    strategy_frame = prepare_strategy_frame(normalized, cfg)
    latest = strategy_frame.iloc[-1]
    timestamp = pd.Timestamp(latest["timestamp"]).isoformat()

    state = load_state(STATE_PATH, initial_cash=float(paper["initial_cash"]))
    if state.last_bar_timestamp == timestamp:
        print(f"Two-tier safety paper account already processed {timestamp}; no action taken.")
        return state

    price = float(latest["close"])
    raw_target = max(0.0, min(1.0, float(latest["target_position"])))
    fee_rate = float(backtest["fee_bps_per_turnover"]) / 10_000.0
    slippage_rate = float(backtest["slippage_bps_per_turnover"]) / 10_000.0
    minimum_equity = float(paper["initial_cash"]) * safety.minimum_equity_fraction

    state, trade_row, run_row = apply_bar(
        state, timestamp=timestamp, price=price, raw_target=raw_target, safety=safety,
        fee_rate=fee_rate, slippage_rate=slippage_rate, minimum_equity=minimum_equity,
    )

    save_state(STATE_PATH, state)
    if trade_row is not None:
        append_trade(TRADES_PATH, trade_row)
    append_run_log(RUNS_LOG_PATH, run_row)

    print(f"Processed {timestamp}: raw_target={raw_target:.3f} applied_target={run_row['applied_target']:.3f} "
          f"({run_row['safety_reason'] or 'live'}), side={run_row['side']}, cash=${state.cash:,.2f}, "
          f"btc={state.btc:.8f}, equity=${run_row['equity']:,.2f}, drawdown={run_row['drawdown']:.2%}, "
          f"hard_halted={state.hard_halted}")
    return state


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Two-tier safety-layer candidate: live paper trading step")
    parser.add_argument("--config", default=str(ROOT / "config" / "settings_production.yaml"))
    args = parser.parse_args()
    run_two_tier_paper_step(args.config)


if __name__ == "__main__":
    main()
