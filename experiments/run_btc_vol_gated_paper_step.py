"""Live paper-trading step: ema_pullback_15m_4h gated to a 4-HOUR volatility
regime (not the 5-minute-bar-based ATR percentile the original research used).

Joey's instruction: "try out high volatility btc bot on paper data, and the
volatility measurement should be 4 hour signals, not daily." This reuses the
exact, already-validated ema_pullback strategy logic from popular_matrix.py
(decide_strategy, typ="ema_pullback") and its feature pipeline
(build_feature_frame) rather than reimplementing it -- only the volatility
gate is new.

4-hour regime construction: resample completed 5m bars into 4h candles (same
_completed_resample mechanism build_feature_frame already uses for h4_atr/
h4_trend_bps/h4_adx), compute ATR on that 4h series, normalize by close, and
take a trailing rolling percentile over a 42-bar window (42 x 4h = 7 days --
the same calendar lookback as the original 2,016-bar/7-day 5m construction,
translated to this timeframe), shifted one bar forward so the gate value used
for a decision was fully known before that bar closed (no lookahead). Top
quartile (>=0.75) = high-vol, matching the threshold Joey decided to KEEP
(a looser 0.65 threshold added trades in the original research but flipped
the absolute dollar result negative).

The gate only filters NEW entries. An already-open position still manages
its own exit (ATR trailing stop / max hold / trend reversal via
exit_common/decide_strategy) regardless of the current regime reading --
consistent with how every other regime-gated strategy in this project's
research has been designed (a gate is an entry filter, not a forced exit).

Paper only. No live orders, no exchange authentication used anywhere in this
file (see popular_matrix.py's own module docstring for the same invariant).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.data import download_ohlcv, normalize_ohlcv, timeframe_to_timedelta  # noqa: E402
from btc_trend_bot.paper_lab import fetch_public_quote  # noqa: E402
from btc_trend_bot.popular_matrix import (  # noqa: E402
    Decision,
    StrategyState,
    _atr,
    _completed_resample,
    _equity,
    _finite,
    build_feature_frame,
    decide_strategy,
    load_matrix_config,
)

CONFIG_PATH = ROOT / "config" / "settings_popular_matrix.yaml"
STRATEGY_ID = "ema_pullback_15m_4h"
VOL_REGIME_LOOKBACK_4H_BARS = 42  # 42 x 4h = 7 days, matching the original 7-day/2016-bar-5m lookback
VOL_REGIME_THRESHOLD = 0.75       # kept per Joey's decision not to loosen to 0.65
# NOTE: runtime/ in this worktree is root-owned (drwxr-xr-x root:root), not
# writable by this process -- using outputs/ instead. If this goes live on a
# schedule, this needs resolving (fix runtime/ ownership or keep outputs/).
LEDGER_PATH = ROOT / "outputs" / "btc_vol_gated_paper" / "btc_vol_gated_paper.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vol_gate_paper_snapshots (
    bar_timestamp       TEXT PRIMARY KEY,
    quote_timestamp     TEXT NOT NULL,
    close               REAL NOT NULL,
    h4_atr_norm         REAL,
    vol_regime_pctile   REAL,
    high_vol            INTEGER,
    prior_target        REAL NOT NULL,
    decision_signal     TEXT NOT NULL,
    decision_reason     TEXT NOT NULL,
    new_target          REAL NOT NULL,
    bid                 REAL,
    ask                 REAL,
    mark                REAL,
    created_at          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vol_gate_paper_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def build_vol_regime_gate(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Adds high_vol / vol_regime_pctile / h4_atr_norm columns to the 5m
    feature frame, computed from an independent 4h-resampled ATR series
    (not the already-merged h4_atr column, so the rolling percentile is
    computed once per real 4h bar, not once per 5m row with 48x duplication)."""
    step = timeframe_to_timedelta(str(cfg["market"]["timeframe"]))
    h4 = _completed_resample(frame, step, "4h", 48)
    h4_atr_period = int(cfg["features"].get("h4_atr_period", 20))
    h4["h4_atr_raw"] = _atr(h4, h4_atr_period)
    h4["atr_norm"] = h4["h4_atr_raw"] / h4["close"]
    h4["vol_pctile"] = h4["atr_norm"].rolling(
        VOL_REGIME_LOOKBACK_4H_BARS, min_periods=VOL_REGIME_LOOKBACK_4H_BARS
    ).apply(lambda s: (s < s.iloc[-1]).mean(), raw=False)
    h4["vol_pctile"] = h4["vol_pctile"].shift(1)  # no-lookahead: gate known before this bar's own reading is used
    h4 = h4.rename(columns={"atr_norm": "h4_atr_norm", "vol_pctile": "vol_regime_pctile"})

    out = frame.copy()
    out = pd.merge_asof(
        out.sort_values("bar_end"), h4[["bar_end", "h4_atr_norm", "vol_regime_pctile"]].sort_values("bar_end"),
        on="bar_end", direction="backward", allow_exact_matches=True,
    )
    out["high_vol"] = out["vol_regime_pctile"] >= VOL_REGIME_THRESHOLD
    return out


def execute_live_decision(state: StrategyState, decision: Decision, quote, index: int, costs: dict) -> dict | None:
    """Live-context equivalent of popular_matrix.execute_decision -- same
    state-mutation logic (entry_index/entry_mark/highest_high/entry_reference/
    cash/btc/trade_count/last_trade_index), but fills against the REAL live
    bid/ask from `quote` instead of a synthetic next-bar-open + assumed
    spread/slippage rate. This is more realistic for live paper trading (a
    genuine market spread, not an assumed one) -- matches this project's
    established "real quote over synthetic assumption" convention used
    everywhere else (see the equity-side options work's entry-at-real-ask/
    exit-at-real-bid discipline). Mutates `state` in place; returns a dict
    describing the fill, or None if no rebalance was needed."""
    if quote is None:
        return None  # can't execute without a real quote -- state stays as-is, decision not acted on
    fee_rate = float(costs.get("fee_bps_per_side", 0.0)) / 10_000.0
    target = min(1.0, max(0.0, float(decision.target_position)))
    mark = quote.mark
    equity_before = _equity(state, mark)
    desired_btc = equity_before * target / mark
    delta_btc = desired_btc - state.btc
    min_notional = float(costs.get("min_notional", 10.0))
    tolerance = equity_before * float(costs.get("rebalance_tolerance_bps", 1.0)) / 10_000.0
    if abs(delta_btc) * mark < max(min_notional, tolerance):
        state.target_position = target
        return None

    if delta_btc > 0.0:
        side, fill = "buy", quote.ask
        executed_btc = min(delta_btc, state.cash / (fill * (1.0 + fee_rate)))
    else:
        side, fill = "sell", quote.bid
        executed_btc = -min(abs(delta_btc), state.btc)

    notional = abs(executed_btc) * fill
    if notional < min_notional or abs(executed_btc) <= 1e-15:
        state.target_position = target
        return None

    fee = notional * fee_rate
    if executed_btc > 0.0:
        state.cash -= notional + fee
        state.btc += executed_btc
        state.entry_index = index
        state.entry_mark = fill
        state.highest_high = fill
        state.entry_reference = decision.entry_reference or fill
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
    state.trade_count += 1
    state.last_trade_index = index
    state.target_position = target
    return {"side": side, "fill_price": fill, "btc_delta": executed_btc, "notional": notional, "fee": fee,
            "cash_after": state.cash, "btc_after": state.btc, "equity_after": _equity(state, mark)}


def _load_prior_state(conn: sqlite3.Connection) -> StrategyState:
    row = conn.execute("SELECT value FROM vol_gate_paper_state WHERE key='strategy_state'").fetchone()
    if row is None:
        return StrategyState(strategy_id=STRATEGY_ID, initial_cash=2500.0, cash=2500.0, btc=0.0,
                              gross_cash=2500.0, gross_btc=0.0)
    d = json.loads(row["value"])
    return StrategyState(**d)


def _save_state(conn: sqlite3.Connection, state: StrategyState) -> None:
    from dataclasses import asdict
    conn.execute(
        "INSERT INTO vol_gate_paper_state(key,value) VALUES('strategy_state',?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (json.dumps(asdict(state)),))
    conn.commit()


def fetch_live_frame(cfg: dict, lookback_5m_bars: int = 30 * 288) -> pd.DataFrame:
    """30 days of 5m bars by default -- enough for 30*288/48 = 180 4h bars,
    comfortably more than the 42-bar rolling window needs to produce a
    stable percentile reading."""
    market = cfg["market"]
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5) * (lookback_5m_bars + 50)).isoformat()
    raw = download_ohlcv(exchange_id=str(market["exchange"]), symbol=str(market["symbol"]),
                          timeframe="5m", start=start, max_bars=lookback_5m_bars + 50)
    normalized, _ = normalize_ohlcv(raw, timeframe="5m")
    return normalized.tail(lookback_5m_bars).reset_index(drop=True)


def run_step(dry_run_only: bool = False) -> dict:
    cfg = load_matrix_config(CONFIG_PATH)
    strategy_cfg = next(s for s in cfg["strategies"] if s["id"] == STRATEGY_ID)
    costs_cfg = cfg["costs"]

    raw = fetch_live_frame(cfg)
    features = build_feature_frame(raw, cfg)
    gated = build_vol_regime_gate(features, cfg)
    gated = gated[gated["feature_valid"]].reset_index(drop=True)
    if gated.empty:
        return {"status": "insufficient_data"}

    last = gated.iloc[-1]
    conn = _conn()
    try:
        state = _load_prior_state(conn)
        index = len(gated) - 1

        # Match popular_matrix.py's main simulation loop exactly (line ~658-659):
        # the running highest-high must be updated from the latest bar BEFORE
        # decide_strategy's exit_common check runs, or the ATR trailing stop
        # would only ever compare against the entry price on every subsequent
        # invocation instead of the true running peak since entry.
        if state.target_position > 0.0:
            state.highest_high = max(_finite(state.highest_high), float(last["high"]))

        # Entry gate: only allow a NEW position (prev<=0) when high-vol.
        # An already-open position (prev>0) is NOT touched by the gate --
        # it exits on its own normal rule via decide_strategy regardless.
        if state.target_position <= 0 and not bool(last.get("high_vol", False)):
            decision = Decision(STRATEGY_ID, 0.0, "hold_cash", "vol_regime_gate: not high-vol, no new entry")
        else:
            decision = decide_strategy(strategy_cfg, last, state, index, costs_cfg)

        quote = None
        try:
            quote = fetch_public_quote(str(cfg["market"]["exchange"]), str(cfg["market"]["symbol"]))
        except Exception:
            pass  # feature/decision logic already ran; a quote failure shouldn't crash the step

        row = {
            "bar_timestamp": str(last["bar_end"]),
            "quote_timestamp": quote.timestamp if quote else datetime.now(timezone.utc).isoformat(),
            "close": float(last["close"]),
            "h4_atr_norm": float(last["h4_atr_norm"]) if pd.notna(last.get("h4_atr_norm")) else None,
            "vol_regime_pctile": float(last["vol_regime_pctile"]) if pd.notna(last.get("vol_regime_pctile")) else None,
            "high_vol": bool(last.get("high_vol", False)),
            "prior_target": state.target_position,
            "decision_signal": decision.signal,
            "decision_reason": decision.reason,
            "new_target": decision.target_position,
            "bid": quote.bid if quote else None,
            "ask": quote.ask if quote else None,
            "mark": quote.mark if quote else None,
        }

        fill = None
        if not dry_run_only:
            conn.execute(
                "INSERT OR REPLACE INTO vol_gate_paper_snapshots"
                "(bar_timestamp,quote_timestamp,close,h4_atr_norm,vol_regime_pctile,high_vol,"
                "prior_target,decision_signal,decision_reason,new_target,bid,ask,mark,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (row["bar_timestamp"], row["quote_timestamp"], row["close"], row["h4_atr_norm"],
                 row["vol_regime_pctile"], int(row["high_vol"]), row["prior_target"], row["decision_signal"],
                 row["decision_reason"], row["new_target"], row["bid"], row["ask"], row["mark"],
                 datetime.now(timezone.utc).isoformat()))
            # Real state mutation (entry_index/entry_mark/highest_high/cash/btc),
            # filled against the live quote's real bid/ask -- not a placeholder
            # target_position assignment. See execute_live_decision's docstring
            # for why this is the correct live-context equivalent of
            # popular_matrix.execute_decision.
            fill = execute_live_decision(state, decision, quote, index, costs_cfg)
            _save_state(conn, state)

        return {"status": "ok", "fill": fill, **row}
    finally:
        conn.close()


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    result = run_step(dry_run_only=dry)
    print(json.dumps(result, indent=2, default=str))
