"""European lead-signal shadow paper deployment -- Phase 2 shadow deployment
(per CLAUDE.md's roadmap, same category as the three live paper bots): log
hypothetical same-day SPY/QQQ trades against REAL live quotes, no live
orders, nothing promoted until it earns it.

Signal: DAX's cumulative return from its own session open through the last
bar closed before US open (13:30 UTC) sets direction (+1 long / -1 short /
0 flat). Two additional eligibility filters, computed as EXPANDING
percentiles using only prior days (never the full sample -- that would be
lookahead): |DAX move| in its own top quartile, and a 3-index Asian
magnitude composite (mean of |Nikkei|, |HSI|, |Shanghai| full-session
returns) in ITS top quartile. All three arms (daily / DAX-top-quartile /
DAX-top-quartile-plus-Asia-magnitude) trade the SAME direction on the same
day when eligible -- they differ only in whether a given day is INCLUDED,
so this logs one row per instrument per day with eligibility flags, and
arm-level performance is computed by filtering at query time, not by
running separate concurrent positions.

Percentile-threshold bootstrap: seeded from data/european_signal_percentile_seed.csv,
629 days of REAL historical DAX/Asian market data (2023-09 to 2026-07,
same series validated in EUROPEAN_LEAD_US_FIRST_HOUR_STUDY.txt on the
research branch) -- this calibrates what counts as "a big move" from day
one instead of needing months of live history to bootstrap. This seed is
used ONLY to define the percentile threshold; every trade's entry/exit
price and realized P&L below is 100% live, real-time, forward-only -- no
historical trade is fabricated.

Two invocations per trading day: `step-entry` (run ~13:32 UTC, just after
US open) computes the signal and logs entry quotes; `step-exit` (run
~14:32 UTC, one hour later) logs exit quotes and realized returns at
1/2/5 bps cost assumptions. Never submits a live order.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = ROOT / "runtime" / "european_signal_shadow.sqlite3"
SEED_PATH = ROOT / "data" / "european_signal_percentile_seed.csv"
INSTRUMENTS = ("SPY", "QQQ")
COST_TIERS_BPS = (1.0, 2.0, 5.0)
MIN_HISTORY_FOR_PERCENTILE = 20
US_OPEN_UTC = pd.Timestamp("13:30:00").time()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_log (
    date                 TEXT PRIMARY KEY,
    dax_pre_open_return  REAL,
    dax_abs_move         REAL,
    dax_direction        INTEGER,
    dax_percentile       REAL,
    nikkei_return        REAL,
    hsi_return           REAL,
    shanghai_return      REAL,
    asia_magnitude       REAL,
    asia_percentile      REAL,
    eligible_daily              INTEGER,
    eligible_top_quartile       INTEGER,
    eligible_top_quartile_asia  INTEGER,
    data_quality_flags   TEXT,
    status               TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trade_log (
    date              TEXT NOT NULL,
    instrument        TEXT NOT NULL,
    direction         INTEGER,
    entry_time        TEXT,
    entry_price       REAL,
    exit_time         TEXT,
    exit_price        REAL,
    gross_return      REAL,
    net_return_1bp    REAL,
    net_return_2bp    REAL,
    net_return_5bp    REAL,
    PRIMARY KEY (date, instrument)
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


def _load_seed() -> pd.DataFrame:
    seed = pd.read_csv(SEED_PATH, parse_dates=["date"])
    seed["date"] = seed["date"].dt.date
    return seed


def _percentile_history(conn: sqlite3.Connection, column: str, today: "pd.Timestamp") -> np.ndarray:
    """Seed history (real, prior, historical) + any live shadow days logged
    so far that are strictly before today -- expanding, never full-sample."""
    seed = _load_seed()
    seed_vals = seed[column].to_numpy()
    live = pd.read_sql_query(
        f"SELECT {column} FROM signal_log WHERE date < ? AND {column} IS NOT NULL", conn, params=(str(today),))
    return np.concatenate([seed_vals, live[column].to_numpy()])


def _fetch_dax_pre_open_return(today: "pd.Timestamp") -> tuple[float, dict]:
    """DAX's cumulative return from today's session open through the last
    5-minute bar closed strictly before 13:30 UTC. Fresh intraday pull,
    separate from the historical 60m backtest data."""
    flags = {}
    df = yf.download("^GDAXI", interval="5m", period="2d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    if df.empty:
        return float("nan"), {"dax_missing_data": True}
    df = df[df.index.date == today.date()]
    df = df[df.index.time < US_OPEN_UTC]
    if len(df) < 2:
        return float("nan"), {"dax_missing_data": True, "dax_bars_before_cutoff": len(df)}
    ret = float(df["close"].iloc[-1] / df["open"].iloc[0] - 1)
    age_seconds = (datetime.now(timezone.utc) - df.index[-1].to_pydatetime()).total_seconds()
    if age_seconds > 3600:
        flags["dax_stale_data"] = True
    return ret, flags


def _fetch_asia_returns(today: "pd.Timestamp") -> tuple[dict, dict]:
    """Asian sessions finish well before US open -- today's full daily bar
    should already be settled by the time this runs at ~13:32 UTC."""
    out, flags = {}, {}
    for symbol, key in (("^N225", "nikkei_return"), ("^HSI", "hsi_return"), ("000001.SS", "shanghai_return")):
        df = yf.download(symbol, interval="1d", period="5d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        todays = df[df.index.date == today.date()]
        if todays.empty:
            out[key] = None
            flags[f"{key}_missing"] = True
            continue
        row = todays.iloc[0]
        out[key] = float(row["close"] / row["open"] - 1)
    return out, flags


def _current_quote(symbol: str) -> float | None:
    df = yf.download(symbol, interval="1m", period="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


def step_entry() -> dict:
    today = pd.Timestamp.now(tz="UTC")
    conn = _conn()
    try:
        existing = conn.execute("SELECT date FROM signal_log WHERE date=?", (str(today.date()),)).fetchone()
        if existing:
            return {"status": "already_processed", "date": str(today.date())}

        dax_ret, dax_flags = _fetch_dax_pre_open_return(today)
        asia, asia_flags = _fetch_asia_returns(today)
        flags = {**dax_flags, **asia_flags}

        if not np.isfinite(dax_ret):
            conn.execute(
                "INSERT INTO signal_log(date,status,data_quality_flags,created_at,updated_at) VALUES(?,?,?,?,?)",
                (str(today.date()), "rejected_missing_dax", json.dumps(flags), _utc_now(), _utc_now()))
            conn.commit()
            return {"status": "rejected", "reason": "missing_dax_data", "flags": flags}

        dax_abs_move = abs(dax_ret)
        direction = int(np.sign(dax_ret))
        asia_vals = [v for v in asia.values() if v is not None]
        asia_magnitude = float(np.mean([abs(v) for v in asia_vals])) if len(asia_vals) == 3 else None
        if asia_magnitude is None:
            flags["asia_incomplete"] = True

        dax_hist = _percentile_history(conn, "dax_abs_move", today)
        dax_pctile = float((dax_hist < dax_abs_move).mean()) if len(dax_hist) >= MIN_HISTORY_FOR_PERCENTILE else None
        eligible_top_quartile = bool(dax_pctile is not None and dax_pctile >= 0.75)

        eligible_top_quartile_asia = False
        asia_pctile = None
        if asia_magnitude is not None:
            asia_hist = _percentile_history(conn, "asia_magnitude", today)
            asia_pctile = float((asia_hist < asia_magnitude).mean()) if len(asia_hist) >= MIN_HISTORY_FOR_PERCENTILE else None
            eligible_top_quartile_asia = bool(eligible_top_quartile and asia_pctile is not None and asia_pctile >= 0.75)

        conn.execute(
            """INSERT INTO signal_log(date,dax_pre_open_return,dax_abs_move,dax_direction,dax_percentile,
               nikkei_return,hsi_return,shanghai_return,asia_magnitude,asia_percentile,
               eligible_daily,eligible_top_quartile,eligible_top_quartile_asia,
               data_quality_flags,status,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (str(today.date()), dax_ret, dax_abs_move, direction, dax_pctile,
             asia.get("nikkei_return"), asia.get("hsi_return"), asia.get("shanghai_return"), asia_magnitude, asia_pctile,
             1, int(eligible_top_quartile), int(eligible_top_quartile_asia),
             json.dumps(flags), "pending_exit", _utc_now(), _utc_now()))

        entries = {}
        for symbol in INSTRUMENTS:
            price = _current_quote(symbol)
            if price is None:
                flags[f"{symbol}_entry_missing"] = True
                continue
            conn.execute(
                "INSERT INTO trade_log(date,instrument,direction,entry_time,entry_price) VALUES(?,?,?,?,?)",
                (str(today.date()), symbol, direction, _utc_now(), price))
            entries[symbol] = price
        conn.commit()

        return {"status": "ok", "date": str(today.date()), "dax_direction": direction,
                "dax_abs_move": dax_abs_move, "dax_percentile": dax_pctile,
                "asia_magnitude": asia_magnitude, "asia_percentile": asia_pctile,
                "eligible_top_quartile": eligible_top_quartile,
                "eligible_top_quartile_asia": eligible_top_quartile_asia,
                "entries": entries, "flags": flags}
    finally:
        conn.close()


def step_exit() -> dict:
    today = pd.Timestamp.now(tz="UTC")
    conn = _conn()
    try:
        sig = conn.execute("SELECT * FROM signal_log WHERE date=? AND status='pending_exit'", (str(today.date()),)).fetchone()
        if sig is None:
            return {"status": "no_pending_entry", "date": str(today.date())}

        results = {}
        for symbol in INSTRUMENTS:
            row = conn.execute("SELECT * FROM trade_log WHERE date=? AND instrument=?", (str(today.date()), symbol)).fetchone()
            if row is None or row["entry_price"] is None:
                continue
            exit_price = _current_quote(symbol)
            if exit_price is None:
                continue
            direction = row["direction"]
            gross = direction * (exit_price / row["entry_price"] - 1)
            nets = {f"net_return_{int(c)}bp": gross - c / 10000.0 for c in COST_TIERS_BPS}
            conn.execute(
                "UPDATE trade_log SET exit_time=?, exit_price=?, gross_return=?, "
                "net_return_1bp=?, net_return_2bp=?, net_return_5bp=? WHERE date=? AND instrument=?",
                (_utc_now(), exit_price, gross, nets["net_return_1bp"], nets["net_return_2bp"], nets["net_return_5bp"],
                 str(today.date()), symbol))
            results[symbol] = {"exit_price": exit_price, "gross_return": gross, **nets}

        conn.execute("UPDATE signal_log SET status='completed', updated_at=? WHERE date=?", (_utc_now(), str(today.date())))
        conn.commit()
        return {"status": "ok", "date": str(today.date()), "results": results}
    finally:
        conn.close()


def status() -> dict:
    conn = _conn()
    try:
        recent_signal = conn.execute("SELECT * FROM signal_log ORDER BY date DESC LIMIT 1").fetchone()
        trades = pd.read_sql_query(
            "SELECT t.*, s.eligible_daily, s.eligible_top_quartile, s.eligible_top_quartile_asia "
            "FROM trade_log t JOIN signal_log s ON t.date = s.date WHERE t.gross_return IS NOT NULL", conn)

        arm_stats = {}
        for symbol in INSTRUMENTS:
            sub = trades[trades.instrument == symbol]
            for arm_col, arm_name in [("eligible_daily", "daily"), ("eligible_top_quartile", "top_quartile"),
                                       ("eligible_top_quartile_asia", "top_quartile_plus_asia")]:
                arm_trades = sub[sub[arm_col] == 1]
                if arm_trades.empty:
                    continue
                key = f"{symbol}_{arm_name}"
                arm_stats[key] = {
                    "trades": int(len(arm_trades)),
                    "win_rate_2bp": float((arm_trades["net_return_2bp"] > 0).mean()),
                    "mean_return_2bp_bps": float(arm_trades["net_return_2bp"].mean() * 10000),
                }

        return {
            "total_days_logged": int(pd.read_sql_query("SELECT COUNT(*) as c FROM signal_log", conn).iloc[0]["c"]),
            "completed_trades": int(len(trades)),
            "most_recent_signal": dict(recent_signal) if recent_signal else None,
            "arm_performance": arm_stats,
            "note": "Shadow paper only -- not promoted to dashboard/rankings/email until arms show real positive results over enough trades (see CLAUDE.md promotion gates).",
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="European lead-signal shadow paper deployment (no live orders)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("step-entry", help="Compute today's signal, log entry quotes (~13:32 UTC)")
    sub.add_parser("step-exit", help="Log exit quotes, compute realized returns (~14:32 UTC)")
    sub.add_parser("status", help="Show recent signal + arm performance")
    args = parser.parse_args()

    if args.command == "step-entry":
        print(json.dumps(step_entry(), indent=2, default=str))
    elif args.command == "step-exit":
        print(json.dumps(step_exit(), indent=2, default=str))
    elif args.command == "status":
        print(json.dumps(status(), indent=2, default=str))


if __name__ == "__main__":
    main()
