"""Daily equity paper-trading step against real market data (Phase 2 shadow
deployment, per CLAUDE.md's roadmap): log hypothetical trades, verify fills, no live
orders ever. Neither Schwab nor yfinance offer real paper-trading support, so all
fills here are simulated locally in a SQLite ledger against real OHLCV data.

Two interchangeable live-data sources are supported via `data_source` in the
deployment YAML: "schwab" (config/settings_schwab_equity_paper.yaml, requires
SCHWAB_API_KEY/SCHWAB_APP_SECRET) and "yfinance" (config/settings_equity_paper_yfinance.yaml,
no credentials needed -- the interim source while Schwab access is pending). Both
paths share the same state machine, ledger schema, and safety rules; only the OHLCV
fetch differs.

Every run_step() also mirrors a summary row (status, open position, completed-trade
count, simulated equity/return/drawdown from the same simulate_capital math the
walk-forward backtest uses) to a Supabase table (default "equity_paper_runs", see
sql/003_equity_paper_runs.sql) through btc_trend_bot.paper_lab.SupabaseOutbox's
durable local outbox -- reusing the exact upload/retry mechanism the BTC 5-minute
paper lab already uses. This is best-effort remote monitoring only: the SQLite
ledger stays the sole source of truth for trading decisions, and a Supabase outage
never blocks or fails the step (requires SUPABASE_URL/SUPABASE_SECRET_KEY in the
environment; silently skipped otherwise).

There is no live-order code path in this module. It never submits an order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_equity_real_data_walkforward as walkforward  # noqa: E402
from btc_trend_bot.notifications import notify  # noqa: E402
from btc_trend_bot.paper_lab import SupabaseOutbox  # noqa: E402
from btc_trend_bot.production import process_lock  # noqa: E402
from btc_trend_bot.schwab_client import SchwabClientProtocol, fetch_schwab_ohlcv, load_schwab_client  # noqa: E402

PROBE_SYMBOL = "__PROBE__"


# --------------------------------------------------------------------------- #
# Deployment config (separate from the frozen strategy BacktestConfig)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PaperDeploymentConfig:
    strategy_config_path: Path
    data_source: str
    schwab_token_path: Path
    lookback_days: int
    ledger_db_path: Path
    lock_path: Path
    selection: str
    supabase_table: str = "equity_paper_runs"
    supabase_outbox_path: Path = Path("runtime/equity_supabase_outbox.jsonl")


def load_deployment_config(path: Path) -> tuple[PaperDeploymentConfig, "walkforward.BacktestConfig"]:
    raw = yaml.safe_load(path.read_text())
    strategy_path = ROOT / raw["strategy_config"]
    strategy_cfg = walkforward.load_config(strategy_path)
    data_source = str(raw.get("data_source", "schwab"))
    if data_source not in ("schwab", "yfinance"):
        raise ValueError(f"Unknown data_source: {data_source!r} (expected 'schwab' or 'yfinance')")
    schwab = raw.get("schwab", {})
    paper = raw.get("paper", {})
    ledger_db_path = ROOT / str(paper.get("ledger_db_path", "runtime/equity_schwab_paper.sqlite3"))
    # Each deployment (yfinance vs schwab) gets its own outbox file, named off its
    # own ledger, so retried telemetry never crosses between the two data sources.
    default_outbox = ledger_db_path.parent / f"{ledger_db_path.stem}_supabase_outbox.jsonl"
    supabase_outbox_path = ROOT / str(paper.get("supabase_outbox_path", default_outbox))
    deploy_cfg = PaperDeploymentConfig(
        strategy_config_path=strategy_path,
        data_source=data_source,
        schwab_token_path=ROOT / str(schwab.get("token_path", "runtime/schwab_token.json")),
        lookback_days=int(schwab.get("lookback_days", 1400)),
        ledger_db_path=ledger_db_path,
        lock_path=ROOT / str(paper.get("lock_path", "runtime/equity_schwab_paper.lock")),
        selection=str(paper.get("selection", "ridge")),
        supabase_table=str(paper.get("supabase_table", "equity_paper_runs")),
        supabase_outbox_path=supabase_outbox_path,
    )
    return deploy_cfg, strategy_cfg


def _config_hash(cfg: "walkforward.BacktestConfig") -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


# --------------------------------------------------------------------------- #
# Live single-day decision
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DailyDecision:
    signal_time: pd.Timestamp
    winner_symbol: str | None
    predicted_return: float | None
    atr: float | None
    train_rows: int
    candidate_symbols: tuple[str, ...]


def _train_date_window(prior_dates: list, train_bars: int, purge_bars: int) -> list:
    """Mirrors walk_forward's positional slicing (train_start=start-train_bars,
    train_end=start-purge_bars) with len(prior_dates) playing the role of `start`.
    Positive-index arithmetic (not prior_dates[-train_bars:-purge_bars]) avoids
    both an off-by-purge_bars error and the purge_bars=0 empty-slice trap
    negative indexing has."""
    n = len(prior_dates)
    train_start = max(0, n - train_bars)
    train_end = max(train_start, n - purge_bars)
    return prior_dates[train_start:train_end]


def decide_today(frames: dict[str, pd.DataFrame], cfg: "walkforward.BacktestConfig",
                  selection: str = "ridge", seed_date: pd.Timestamp | None = None) -> DailyDecision:
    """Today's trade decision: fit on a trailing, purged window of already-labeled
    history and predict on today's single row per symbol. Reuses
    run_equity_real_data_walkforward's add_features/build_candidates/
    _select_fold_winners unmodified -- see that module for the batch equivalent
    this must stay consistent with."""
    bench = frames[cfg.benchmark]
    as_of = bench.index[-1]
    featured = {s: walkforward.add_features(frames[s], bench) for s in cfg.symbols if s in frames}

    rows = []
    for symbol, f in featured.items():
        if as_of not in f.index:
            continue
        r = f.loc[as_of]
        if all(c in r and np.isfinite(r[c]) for c in walkforward.FEATURES):
            row = {"signal_time": as_of, "symbol": symbol}
            row.update({c: float(r[c]) for c in walkforward.FEATURES})
            rows.append(row)
    candidate_symbols = tuple(row["symbol"] for row in rows)
    if not rows:
        return DailyDecision(as_of, None, None, None, 0, candidate_symbols)
    test = pd.DataFrame(rows)

    candidates = walkforward.build_candidates(frames, cfg.benchmark, cfg)
    candidates = candidates.dropna(subset=walkforward.FEATURES + ["net_return"])
    dates = sorted(candidates.signal_time.dt.normalize().unique())
    prior_dates = [d for d in dates if d < as_of.normalize()]
    train_dates = _train_date_window(prior_dates, cfg.train_bars, cfg.purge_bars)
    train = candidates[candidates.signal_time.dt.normalize().isin(train_dates)]

    needs_train = selection == "ridge"
    if needs_train and len(train) < 200:
        return DailyDecision(as_of, None, None, None, len(train), candidate_symbols)

    rng = np.random.default_rng(int((seed_date or as_of).strftime("%Y%m%d")))
    winners = walkforward._select_fold_winners(train, test, cfg, 0, rng, False, selection)
    if winners.empty:
        return DailyDecision(as_of, None, None, None, len(train), candidate_symbols)

    winner = winners.iloc[0]
    winner_symbol = str(winner.symbol)
    atr = float(featured[winner_symbol].loc[as_of, "atr"])
    predicted_return = float(winner.predicted_return) if pd.notna(winner.predicted_return) else None
    return DailyDecision(as_of, winner_symbol, predicted_return, atr, len(train), candidate_symbols)


# --------------------------------------------------------------------------- #
# Open-position state machine
# --------------------------------------------------------------------------- #

@dataclass
class PendingEntry:
    symbol: str
    signal_time: pd.Timestamp
    atr: float


@dataclass
class OpenPosition:
    symbol: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    bars_held: int


def _fill_pending_entry(pending: PendingEntry, bar: pd.Series, bar_time: pd.Timestamp,
                         cfg: "walkforward.BacktestConfig") -> OpenPosition:
    entry_price = float(bar.open) * (1 + cfg.slippage_bps_each_side / 10000)
    return OpenPosition(
        symbol=pending.symbol, signal_time=pending.signal_time, entry_time=bar_time,
        entry_price=entry_price,
        stop_price=entry_price - cfg.stop_atr * pending.atr,
        target_price=entry_price + cfg.target_atr * pending.atr,
        bars_held=0,
    )


def _evaluate_open_position_bar(position: OpenPosition, bar: pd.Series, bar_time: pd.Timestamp,
                                 cfg: "walkforward.BacktestConfig") -> tuple[OpenPosition | None, dict | None]:
    """Mirrors simulate_trade's exact stop/target tie-break and time-exit rules,
    one new bar at a time against a persisted position instead of a single
    vectorized scan over a known future."""
    position.bars_held += 1
    low, high, close = float(bar.low), float(bar.high), float(bar.close)
    if low <= position.stop_price and high >= position.target_price:
        exit_price, reason = position.stop_price, "ambiguous_stop_first"
    elif low <= position.stop_price:
        exit_price, reason = position.stop_price, "stop"
    elif high >= position.target_price:
        exit_price, reason = position.target_price, "target"
    elif position.bars_held >= cfg.max_hold_bars:
        exit_price, reason = close, "time_exit"
    else:
        return position, None
    exit_fill = exit_price * (1 - cfg.slippage_bps_each_side / 10000)
    trade = {
        "signal_time": position.signal_time, "symbol": position.symbol,
        "entry_time": position.entry_time, "entry_price": position.entry_price,
        "exit_time": bar_time, "exit_price": exit_fill,
        "net_return": exit_fill / position.entry_price - 1,
        "exit_reason": reason, "bars_held": position.bars_held,
    }
    return None, trade


def _safety_allows_new_entry(ledger_trades: pd.DataFrame, today: pd.Timestamp,
                              cfg: "walkforward.BacktestConfig") -> tuple[bool, str]:
    """simulate_capital is a pure function that replays the whole ledger from
    scratch; append a harmless probe row (net_return=0, discarded after reading
    its skip_reason) instead of duplicating its cooldown/drawdown/hard-shutdown
    logic here."""
    probe = pd.DataFrame([{"signal_time": today, "symbol": PROBE_SYMBOL, "net_return": 0.0}])
    if ledger_trades.empty:
        combined = probe
    else:
        combined = pd.concat([ledger_trades[["signal_time", "symbol", "net_return"]], probe], ignore_index=True)
    combined["signal_time"] = pd.to_datetime(combined["signal_time"], utc=True)
    path, _ = walkforward.simulate_capital(combined, cfg)
    last = path.iloc[-1]
    return bool(last.trade_taken), str(last.skip_reason)


# --------------------------------------------------------------------------- #
# SQLite ledger
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS completed_trades (
    signal_time      TEXT PRIMARY KEY,
    symbol           TEXT NOT NULL,
    entry_time       TEXT NOT NULL,
    entry_price      REAL NOT NULL,
    exit_time        TEXT NOT NULL,
    exit_price       REAL NOT NULL,
    net_return       REAL NOT NULL,
    exit_reason      TEXT NOT NULL,
    bars_held        INTEGER NOT NULL,
    predicted_return REAL,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS open_position (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    status           TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    signal_time      TEXT NOT NULL,
    atr              REAL NOT NULL,
    predicted_return REAL,
    entry_time       TEXT,
    entry_price      REAL,
    stop_price       REAL,
    target_price     REAL,
    bars_held        INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    as_of_date   TEXT,
    status       TEXT NOT NULL,
    message      TEXT NOT NULL,
    config_hash  TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class PaperLedger:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get_kv(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM kv_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_kv(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO kv_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))
        self.conn.commit()

    def is_halted(self) -> bool:
        return self.get_kv("halted") == "true"

    def halt_reason(self) -> str:
        return self.get_kv("halt_reason") or ""

    def set_halt(self, halted: bool, reason: str = "") -> None:
        self.set_kv("halted", "true" if halted else "false")
        self.set_kv("halt_reason", reason if halted else "")

    def last_processed_bar_date(self) -> str | None:
        return self.get_kv("last_processed_bar_date")

    def set_last_processed_bar_date(self, value: str) -> None:
        self.set_kv("last_processed_bar_date", value)

    def get_open_position(self) -> dict | None:
        row = self.conn.execute("SELECT * FROM open_position WHERE id=1").fetchone()
        return dict(row) if row else None

    def upsert_open_position(self, **fields) -> None:
        fields = {**fields, "id": 1, "updated_at": _utc_now()}
        cols = ",".join(fields)
        placeholders = ",".join("?" for _ in fields)
        updates = ",".join(f"{k}=excluded.{k}" for k in fields if k != "id")
        self.conn.execute(
            f"INSERT INTO open_position({cols}) VALUES({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            list(fields.values()))
        self.conn.commit()

    def clear_open_position(self) -> None:
        self.conn.execute("DELETE FROM open_position WHERE id=1")
        self.conn.commit()

    def append_completed_trade(self, trade: dict, predicted_return: float | None) -> None:
        self.conn.execute(
            "INSERT INTO completed_trades(signal_time,symbol,entry_time,entry_price,exit_time,exit_price,"
            "net_return,exit_reason,bars_held,predicted_return,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (str(trade["signal_time"]), trade["symbol"], str(trade["entry_time"]), trade["entry_price"],
             str(trade["exit_time"]), trade["exit_price"], trade["net_return"], trade["exit_reason"],
             trade["bars_held"], predicted_return, _utc_now()))
        self.conn.commit()

    def completed_trades_frame(self) -> pd.DataFrame:
        rows = self.conn.execute(
            "SELECT signal_time, symbol, net_return FROM completed_trades ORDER BY signal_time").fetchall()
        frame = pd.DataFrame([dict(r) for r in rows], columns=["signal_time", "symbol", "net_return"])
        if len(frame):
            frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
        return frame

    def log_run(self, as_of_date, status: str, message: str, config_hash: str | None, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO runs(created_at,as_of_date,status,message,config_hash,payload_json) VALUES(?,?,?,?,?,?)",
            (_utc_now(), str(as_of_date) if as_of_date is not None else None, status, message, config_hash,
             json.dumps(payload, default=str)))
        self.conn.commit()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _process_one_day(ledger: PaperLedger, frames: dict[str, pd.DataFrame], cfg: "walkforward.BacktestConfig",
                      deploy_cfg: PaperDeploymentConfig, date: pd.Timestamp, allow_new_entry: bool) -> dict:
    position = ledger.get_open_position()

    if position is not None:
        # An existing position is always managed regardless of safety-pause status
        # or how it got here (auto pause never blocks position management; only
        # a manual halt, which short-circuits run_step entirely, does).
        symbol = position["symbol"]
        if symbol not in frames or date not in frames[symbol].index:
            return {"date": str(date), "action": "no_bar_for_open_position", "symbol": symbol}
        bar = frames[symbol].loc[date]
        if position["status"] == "pending_entry":
            pending = PendingEntry(symbol=symbol, signal_time=pd.Timestamp(position["signal_time"]),
                                    atr=position["atr"])
            filled = _fill_pending_entry(pending, bar, date, cfg)
            still_open, trade = _evaluate_open_position_bar(filled, bar, date, cfg)
        else:
            open_position = OpenPosition(
                symbol=symbol, signal_time=pd.Timestamp(position["signal_time"]),
                entry_time=pd.Timestamp(position["entry_time"]), entry_price=position["entry_price"],
                stop_price=position["stop_price"], target_price=position["target_price"],
                bars_held=position["bars_held"])
            still_open, trade = _evaluate_open_position_bar(open_position, bar, date, cfg)

        if trade is not None:
            ledger.append_completed_trade(trade, position.get("predicted_return"))
            ledger.clear_open_position()
            return {"date": str(date), "action": "exit", "trade": {k: str(v) for k, v in trade.items()}}

        ledger.upsert_open_position(
            status="open", symbol=still_open.symbol, signal_time=str(still_open.signal_time),
            atr=position["atr"], predicted_return=position.get("predicted_return"),
            entry_time=str(still_open.entry_time), entry_price=still_open.entry_price,
            stop_price=still_open.stop_price, target_price=still_open.target_price,
            bars_held=still_open.bars_held)
        return {"date": str(date), "action": "hold", "symbol": symbol, "bars_held": still_open.bars_held}

    if not allow_new_entry:
        return {"date": str(date), "action": "skip_not_latest"}

    allowed, reason = _safety_allows_new_entry(ledger.completed_trades_frame(), date, cfg)
    if not allowed:
        return {"date": str(date), "action": "no_entry_safety_paused", "reason": reason}

    decision = decide_today(frames, cfg, selection=deploy_cfg.selection, seed_date=date)
    if decision.winner_symbol is None:
        return {"date": str(date), "action": "no_signal", "candidate_symbols": list(decision.candidate_symbols)}

    ledger.upsert_open_position(
        status="pending_entry", symbol=decision.winner_symbol, signal_time=str(decision.signal_time),
        atr=decision.atr, predicted_return=decision.predicted_return,
        entry_time=None, entry_price=None, stop_price=None, target_price=None, bars_held=0)
    return {"date": str(date), "action": "signal", "symbol": decision.winner_symbol,
            "predicted_return": decision.predicted_return}


# --------------------------------------------------------------------------- #
# Supabase telemetry (best-effort; the SQLite ledger above is always the
# source of truth for trading decisions -- this is a remote-monitoring mirror)
# --------------------------------------------------------------------------- #

def _build_run_report(deploy_cfg: PaperDeploymentConfig, strategy_cfg: "walkforward.BacktestConfig",
                       ledger: "PaperLedger", status: str, message: str, as_of: object,
                       started_at: str, finished_at: str) -> dict:
    trades_frame = ledger.completed_trades_frame()
    _, summary = walkforward.simulate_capital(trades_frame, strategy_cfg)
    open_position = ledger.get_open_position() or {}
    win_rate = summary.get("win_rate")
    duration = (pd.Timestamp(finished_at) - pd.Timestamp(started_at)).total_seconds()
    return {
        "run_id": str(uuid4()),
        "server_name": socket.gethostname(),
        "data_source": deploy_cfg.data_source,
        "config_hash": _config_hash(strategy_cfg),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration, 3),
        "status": status,
        "message": message,
        "as_of_date": str(as_of) if as_of is not None else None,
        "halted": ledger.is_halted(),
        "halt_reason": ledger.halt_reason() or None,
        "last_processed_bar_date": ledger.last_processed_bar_date(),
        "completed_trades": int(len(trades_frame)),
        "ending_equity": float(summary["ending_equity"]),
        "total_return": float(summary["total_return"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "win_rate": float(win_rate) if pd.notna(win_rate) else None,
        "open_position_symbol": open_position.get("symbol"),
        "open_position_status": open_position.get("status"),
        "open_position_entry_price": open_position.get("entry_price"),
        "open_position_stop_price": open_position.get("stop_price"),
        "open_position_target_price": open_position.get("target_price"),
        "open_position_predicted_return": open_position.get("predicted_return"),
    }


def _report_run(deploy_cfg: PaperDeploymentConfig, report: dict) -> None:
    """Enqueue-then-flush so a Supabase outage never loses a run's telemetry and
    never raises into the trading step itself (see notify() for the same
    never-fail-the-run philosophy applied to webhook alerts)."""
    try:
        outbox = SupabaseOutbox(deploy_cfg.supabase_outbox_path)
        outbox.enqueue(deploy_cfg.supabase_table, "run_id", report)
        uploaded, pending = outbox.flush()
        print(f"Supabase sync ({deploy_cfg.supabase_table}): uploaded={uploaded}, pending={pending}")
    except Exception as exc:
        print(f"Supabase sync failed: {type(exc).__name__}: {exc}")


def run_step(deployment_config_path: Path, client: "SchwabClientProtocol | None" = None) -> dict:
    deploy_cfg, strategy_cfg = load_deployment_config(deployment_config_path)
    config_hash = _config_hash(strategy_cfg)
    run_started_at = _utc_now()
    with process_lock(deploy_cfg.lock_path):
        ledger = PaperLedger(deploy_cfg.ledger_db_path)
        try:
            if ledger.is_halted():
                result = {"status": "halted", "reason": ledger.halt_reason()}
                ledger.log_run(None, "halted", ledger.halt_reason(), config_hash, result)
                _report_run(deploy_cfg, _build_run_report(
                    deploy_cfg, strategy_cfg, ledger, "halted", ledger.halt_reason(), None,
                    run_started_at, _utc_now()))
                return result

            symbols = tuple(dict.fromkeys((*strategy_cfg.symbols, strategy_cfg.benchmark)))
            if deploy_cfg.data_source == "yfinance":
                # Stand-in data source for shadow-deployment dry runs while Schwab
                # API credentials are pending approval -- same walk-forward
                # normalization, no keys/token required. Swap data_source back to
                # "schwab" in the deployment YAML once approved.
                start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=deploy_cfg.lookback_days)).date().isoformat()
                frames = walkforward.download_yfinance(symbols, start, None, strategy_cfg.interval)
            else:
                if client is None:
                    api_key = _require_env("SCHWAB_API_KEY")
                    app_secret = _require_env("SCHWAB_APP_SECRET")
                    client = load_schwab_client(deploy_cfg.schwab_token_path, api_key, app_secret)
                frames = fetch_schwab_ohlcv(client, symbols, deploy_cfg.lookback_days, walkforward._normalize)

            bench = frames[strategy_cfg.benchmark]
            trading_dates = sorted(pd.Index(bench.index).normalize().unique())
            last_processed = ledger.last_processed_bar_date()
            if last_processed is None:
                # First-ever run: bootstrap at the latest completed bar rather than
                # replaying years of history through the state machine.
                new_dates = trading_dates[-1:] if trading_dates else []
            else:
                last_ts = pd.Timestamp(last_processed)
                new_dates = [d for d in trading_dates if d > last_ts]

            if not new_dates:
                result = {"status": "already_processed", "last_processed_bar_date": last_processed}
                ledger.log_run(last_processed, "already_processed", "no new bar", config_hash, result)
                _report_run(deploy_cfg, _build_run_report(
                    deploy_cfg, strategy_cfg, ledger, "already_processed", "no new bar", last_processed,
                    run_started_at, _utc_now()))
                return result

            actions = []
            for i, date in enumerate(new_dates):
                is_last = i == len(new_dates) - 1
                action = _process_one_day(ledger, frames, strategy_cfg, deploy_cfg, date, allow_new_entry=is_last)
                actions.append(action)
                ledger.set_last_processed_bar_date(str(date))

            result = {"status": "ok", "processed_dates": [str(d) for d in new_dates], "actions": actions}
            ledger.log_run(new_dates[-1], "ok", f"processed {len(new_dates)} day(s)", config_hash, result)
            _report_run(deploy_cfg, _build_run_report(
                deploy_cfg, strategy_cfg, ledger, "ok", f"processed {len(new_dates)} day(s)", new_dates[-1],
                run_started_at, _utc_now()))
            return result
        except Exception as exc:
            ledger.log_run(None, "error", str(exc), config_hash, {"error": str(exc)})
            try:
                _report_run(deploy_cfg, _build_run_report(
                    deploy_cfg, strategy_cfg, ledger, "error", str(exc), None, run_started_at, _utc_now()))
            except Exception:
                pass  # the ledger/lock state is what matters most here; never mask the original exc
            notify(f"URGENT: Schwab equity paper step error: {exc}")
            raise
        finally:
            ledger.close()


def get_status(deployment_config_path: Path) -> dict:
    deploy_cfg, _ = load_deployment_config(deployment_config_path)
    ledger = PaperLedger(deploy_cfg.ledger_db_path)
    try:
        return {
            "halted": ledger.is_halted(),
            "halt_reason": ledger.halt_reason(),
            "last_processed_bar_date": ledger.last_processed_bar_date(),
            "open_position": ledger.get_open_position(),
            "completed_trades": len(ledger.completed_trades_frame()),
        }
    finally:
        ledger.close()


def set_halt(deployment_config_path: Path, halted: bool, reason: str = "manual operator halt") -> None:
    deploy_cfg, _ = load_deployment_config(deployment_config_path)
    ledger = PaperLedger(deploy_cfg.ledger_db_path)
    try:
        ledger.set_halt(halted, reason if halted else "")
    finally:
        ledger.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Schwab daily equity paper-trading step (shadow deployment, no live orders)")
    parser.add_argument("--config", default="config/settings_schwab_equity_paper.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("step", help="Run one idempotent daily paper-trading decision/position-management cycle")
    sub.add_parser("status", help="Show halt state, open position, and ledger size")
    halt_p = sub.add_parser("halt", help="Set the persistent halt flag; blocks step entirely, including position management")
    halt_p.add_argument("--reason", default="manual operator halt")
    sub.add_parser("resume", help="Clear the halt flag")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    if args.command == "step":
        print(json.dumps(run_step(config_path), indent=2, default=str))
    elif args.command == "status":
        print(json.dumps(get_status(config_path), indent=2, default=str))
    elif args.command == "halt":
        set_halt(config_path, True, args.reason)
        print(json.dumps(get_status(config_path), indent=2, default=str))
    elif args.command == "resume":
        set_halt(config_path, False)
        print(json.dumps(get_status(config_path), indent=2, default=str))


if __name__ == "__main__":
    main()
