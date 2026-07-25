"""Live paper-trading step: NQ=F overnight move -> QQQ 0DTE/1DTE options.

Validated in EQUITY_FUTURES_LEAD_OPTIONS_BACKTEST.md: 63.6%/63.6% win rate,
PF 3.70/3.66, Sharpe 7.84/7.77, +469.6%/+269.7% total return (0DTE/1DTE),
107 trades, 0 skipped -- the strongest result of the whole lead-lag
research thread, decisively beating the already-live DAX->SPY signal on
every metric. QQQ options liquidity verified comparable to SPY's before
trusting this leg (9/9 real quotes succeeded across spread-out dates).

Scope decision, made explicitly by Joey: deploy NQ->QQQ only, NOT also
ES->SPY. ES=F and NQ=F are both US equity-index futures, highly
correlated with each other -- running both books would concentrate one
directional bet (US equities), not diversify across two independent
mechanisms, despite testing as statistically distinct signals on paper.
NQ->QQQ was the stronger of the two results, and running it alongside the
already-live DAX->SPY signal (a genuinely different mechanism -- cross-
timezone information transmission, not futures-lead-cash) gives two real,
independently-validated mechanisms rather than one concentrated bet
described twice. Also worth remembering: shorter validated track record
than DAX (~2.4yr vs ~2.9yr).

Signal construction (reused EXACTLY from run_lead_lag_expansion_round2.py /
run_futures_lead_options_backtest.py on the research branch, not redefined
here): NQ=F's cumulative return from its own session's first 60m bar
through the close of the last 60m bar strictly before 13:30 UTC (US open)
sets direction; gate is a STATIC top-quartile-by-|move| threshold computed
once from the same 730-day historical window the backtest used
(2024-02/03 through 2026-07-23/24): |overnight move| >= 0.0074225587272400695
(q75, n=615).

Options execution: real live ThetaData 0DTE/1DTE ATM QQQ quotes, entry
always at the real ask (~9:30 ET), exit always at the real bid (~10:30 ET),
$2,500 base / $250 fixed premium per trade -- same convention as the live
DAX options books in this repo's run_european_signal_shadow_step.py, whose
ThetaData helper pattern this file adapts (parametrized to QQQ instead of
SPY).

Two invocations per trading day, same pattern as the DAX script:
`step-entry` (run ~13:32 UTC, just after US open) computes the signal and
logs entry quotes; `step-exit` (run ~14:32 UTC, one hour later) logs exit
quotes and realized returns. Never submits a live order.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = ROOT / "runtime" / "futures_lead_options_paper.sqlite3"
US_OPEN_UTC = pd.Timestamp("13:30:00").time()

LEADER_SYMBOL = "NQ=F"
UNDERLYING = "QQQ"
GATE_THRESHOLD = 0.0074225587272400695  # q75, n=615, static -- matches what was backtested
OPTIONS_INITIAL_CAPITAL = 2500.0
OPTIONS_PREMIUM_ALLOCATION = 250.0
_THETADATA_ENV_PATH = Path("/home/joey/.config/btc-trend-bot/thetadata.env")
_thetadata_client = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_log (
    date            TEXT PRIMARY KEY,
    leader_move     REAL,
    direction       INTEGER,
    eligible        INTEGER,
    underlying_spot REAL,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS options_trade_log (
    date               TEXT NOT NULL,
    variant            TEXT NOT NULL,
    right              TEXT,
    expiration         TEXT,
    strike             REAL,
    direction          INTEGER,
    entry_time         TEXT,
    entry_ask          REAL,
    exit_time          TEXT,
    exit_bid           REAL,
    net_return         REAL,
    data_quality_flags TEXT,
    PRIMARY KEY (date, variant)
);
CREATE TABLE IF NOT EXISTS kv_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _get_thetadata_client():
    global _thetadata_client
    if _thetadata_client is None:
        if "THETADATA_API_KEY" not in os.environ and _THETADATA_ENV_PATH.exists():
            for line in _THETADATA_ENV_PATH.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
        from thetadata import ThetaClient
        _thetadata_client = ThetaClient(dataframe_type="pandas")
    return _thetadata_client


def _options_expirations_0dte_1dte(today: "date") -> tuple["date", "date"] | tuple[None, None]:
    client = _get_thetadata_client()
    try:
        exps = client.option_list_expirations(symbol=UNDERLYING)
        exps["expiration"] = pd.to_datetime(exps["expiration"]).dt.date
        future = sorted(set(e for e in exps["expiration"] if e >= today))
        if len(future) < 2:
            return None, None
        return future[0], future[1]
    except Exception:
        return None, None


def _options_nearest_strike(expiration: "date", spot: float) -> float | None:
    client = _get_thetadata_client()
    try:
        strikes = client.option_list_strikes(symbol=UNDERLYING, expiration=expiration)
        strike_list = strikes["strike"].tolist()
        if not strike_list:
            return None
        return float(min(strike_list, key=lambda s: abs(s - spot)))
    except Exception:
        return None


def _options_live_quote(expiration: "date", strike: float, right: str) -> tuple[float, float] | tuple[None, None]:
    client = _get_thetadata_client()
    try:
        res = client.option_snapshot_quote(
            symbol=UNDERLYING, expiration=expiration, strike=str(strike),
            right="CALL" if right == "C" else "PUT")
        if res is None or res.empty:
            return None, None
        row = res.iloc[0]
        bid, ask = float(row["bid"]), float(row["ask"])
        if bid <= 0 or ask <= 0:
            return None, None
        return bid, ask
    except Exception:
        return None, None


def _current_quote_via_ohlcv(symbol: str) -> float | None:
    df = yf.download(symbol, interval="1m", period="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


def _leader_overnight_move_live(today: pd.Timestamp) -> tuple[float | None, dict]:
    """Same construction as the backtest: leader's cumulative return from
    its own session's first 60m bar through the close of the last 60m bar
    strictly before 13:30 UTC (US open), computed fresh from live data."""
    flags: dict = {}
    df = yf.download(LEADER_SYMBOL, interval="60m", period="5d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    if df.empty:
        return None, {"leader_missing_data": True}
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df["date"] = df.index.date
    df["time"] = df.index.time
    pre_open = df[(df["date"] == today.date()) & (df["time"] < US_OPEN_UTC)]
    if len(pre_open) < 2:
        return None, {"leader_bars_before_cutoff": len(pre_open)}
    move = float(pre_open["close"].iloc[-1] / pre_open["open"].iloc[0] - 1.0)
    return move, flags


def step_entry() -> dict:
    today = pd.Timestamp.now(tz="UTC")
    conn = _conn()
    try:
        halted = conn.execute("SELECT value FROM kv_state WHERE key='halted'").fetchone()
        if halted and halted["value"] == "true":
            return {"status": "halted"}

        existing = conn.execute("SELECT date FROM signal_log WHERE date=?", (str(today.date()),)).fetchone()
        if existing:
            return {"status": "already_processed", "date": str(today.date())}

        move, flags = _leader_overnight_move_live(today)
        if move is None:
            conn.execute(
                "INSERT INTO signal_log(date,status,created_at) VALUES(?,?,?)",
                (str(today.date()), "rejected_missing_leader", _utc_now()))
            conn.commit()
            return {"status": "rejected", "reason": "missing_leader_data", "flags": flags}

        direction = int(np.sign(move))
        eligible = bool(abs(move) >= GATE_THRESHOLD)
        spot = _current_quote_via_ohlcv(UNDERLYING)

        conn.execute(
            "INSERT INTO signal_log(date,leader_move,direction,eligible,underlying_spot,status,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (str(today.date()), move, direction, int(eligible), spot, "logged", _utc_now()))
        conn.commit()

        entries = {}
        if eligible and spot is not None:
            right = "C" if direction > 0 else "P"
            exp_0dte, exp_1dte = _options_expirations_0dte_1dte(today.date())
            if exp_0dte is None:
                flags["options_expirations_missing"] = True
            else:
                for variant, expiration in (("0dte", exp_0dte), ("1dte", exp_1dte)):
                    strike = _options_nearest_strike(expiration, spot)
                    if strike is None:
                        flags[f"options_{variant}_strike_missing"] = True
                        continue
                    bid, ask = _options_live_quote(expiration, strike, right)
                    if ask is None:
                        flags[f"options_{variant}_quote_missing"] = True
                        continue
                    conn.execute(
                        "INSERT INTO options_trade_log(date,variant,right,expiration,strike,direction,"
                        "entry_time,entry_ask,data_quality_flags) VALUES(?,?,?,?,?,?,?,?,?)",
                        (str(today.date()), variant, right, str(expiration), strike, direction,
                         _utc_now(), ask, json.dumps(flags)))
                    entries[variant] = {"right": right, "expiration": str(expiration), "strike": strike, "entry_ask": ask}
                conn.commit()

        return {"status": "ok", "date": str(today.date()), "leader_move": move, "direction": direction,
                "eligible": eligible, "threshold": GATE_THRESHOLD, "underlying_spot": spot,
                "entries": entries, "flags": flags}
    finally:
        conn.close()


def step_exit() -> dict:
    today = pd.Timestamp.now(tz="UTC")
    conn = _conn()
    try:
        results = {}
        for variant in ("0dte", "1dte"):
            row = conn.execute(
                "SELECT * FROM options_trade_log WHERE date=? AND variant=?", (str(today.date()), variant)).fetchone()
            if row is None or row["entry_ask"] is None or row["exit_bid"] is not None:
                continue
            expiration = datetime.strptime(row["expiration"], "%Y-%m-%d").date()
            bid, ask = _options_live_quote(expiration, row["strike"], row["right"])
            if bid is None:
                continue
            net_return = bid / row["entry_ask"] - 1.0
            conn.execute(
                "UPDATE options_trade_log SET exit_time=?, exit_bid=?, net_return=? WHERE date=? AND variant=?",
                (_utc_now(), bid, net_return, str(today.date()), variant))
            results[variant] = {"exit_bid": bid, "entry_ask": row["entry_ask"], "net_return": net_return}
        conn.execute("UPDATE signal_log SET status='completed' WHERE date=?", (str(today.date()),))
        conn.commit()
        return {"status": "ok", "date": str(today.date()), "results": results}
    finally:
        conn.close()


def get_equity_curve(conn: sqlite3.Connection) -> pd.DataFrame:
    trades = pd.read_sql_query(
        "SELECT date, net_return FROM options_trade_log WHERE variant='0dte' AND net_return IS NOT NULL ORDER BY date",
        conn)
    if trades.empty:
        return trades
    trades["pnl"] = trades["net_return"] * OPTIONS_PREMIUM_ALLOCATION
    trades["equity"] = OPTIONS_INITIAL_CAPITAL + trades["pnl"].cumsum()
    trades["drawdown"] = 1 - trades["equity"] / trades["equity"].cummax()
    return trades


def set_halt(halted: bool, reason: str = "manual operator halt") -> None:
    conn = _conn()
    try:
        conn.execute("INSERT INTO kv_state(key,value) VALUES('halted',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     ("true" if halted else "false",))
        conn.execute("INSERT INTO kv_state(key,value) VALUES('halt_reason',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (reason if halted else "",))
        conn.commit()
    finally:
        conn.close()


def status() -> dict:
    conn = _conn()
    try:
        halted_row = conn.execute("SELECT value FROM kv_state WHERE key='halted'").fetchone()
        halt_reason_row = conn.execute("SELECT value FROM kv_state WHERE key='halt_reason'").fetchone()
        recent_signal = conn.execute("SELECT * FROM signal_log ORDER BY date DESC LIMIT 1").fetchone()
        curve = get_equity_curve(conn)
        book = {"initial_capital": OPTIONS_INITIAL_CAPITAL, "trades_taken": int(len(curve))}
        if len(curve):
            book.update({
                "current_equity": float(curve["equity"].iloc[-1]),
                "total_return_pct": float((curve["equity"].iloc[-1] / OPTIONS_INITIAL_CAPITAL - 1) * 100),
                "max_drawdown_pct": float(curve["drawdown"].max() * 100),
                "win_rate": float((curve["net_return"] > 0).mean()),
            })
        return {
            "halted": bool(halted_row and halted_row["value"] == "true"),
            "halt_reason": halt_reason_row["value"] if halt_reason_row else "",
            "total_days_logged": int(pd.read_sql_query("SELECT COUNT(*) as c FROM signal_log", conn).iloc[0]["c"]),
            "most_recent_signal": dict(recent_signal) if recent_signal else None,
            "book": book,
            "note": "NQ=F->QQQ futures-lead options, deployed alone (not alongside ES->SPY) per Joey's explicit "
                    "correlation-avoidance decision -- see module docstring.",
        }
    finally:
        conn.close()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="NQ=F->QQQ futures-lead options paper deployment (no live orders)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("step-entry", help="Compute today's signal, log entry quotes (~13:32 UTC)")
    sub.add_parser("step-exit", help="Log exit quotes, compute realized returns (~14:32 UTC)")
    sub.add_parser("status", help="Show recent signal + book performance")
    halt_p = sub.add_parser("halt", help="Set the persistent halt flag; blocks new entries")
    halt_p.add_argument("--reason", default="manual operator halt")
    sub.add_parser("resume", help="Clear the halt flag")
    args = parser.parse_args()

    if args.command == "step-entry":
        print(json.dumps(step_entry(), indent=2, default=str))
    elif args.command == "step-exit":
        print(json.dumps(step_exit(), indent=2, default=str))
    elif args.command == "status":
        print(json.dumps(status(), indent=2, default=str))
    elif args.command == "halt":
        set_halt(True, args.reason)
        print(json.dumps(status(), indent=2, default=str))
    elif args.command == "resume":
        set_halt(False)
        print(json.dumps(status(), indent=2, default=str))


if __name__ == "__main__":
    main()
