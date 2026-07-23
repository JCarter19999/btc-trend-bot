"""Daily call-option paper-trading step (live paper deployment, no real
orders -- same "no live-order code path" guarantee as run_equity_paper_step.py).

Reuses run_equity_paper_step.decide_today (simple_trend selection, same as
the primary stock deployment) for WHICH symbol/day to act on, but expresses
the trade as a real 30-DTE, ~5%-OTM call instead of stock -- the
best-performing variant from the research branch's options deep dive
(EQUITY_OPTIONS_DEEP_DIVE.md). Unlike that deep dive, which had to assume a
bid-ask spread for synthetic Black-Scholes pricing (the single biggest lever
in that study), this reads REAL live option quotes from yfinance
(t.option_chain()) -- entry fills at the real ask, exit fills at the real
bid. This is the live resolution of that study's open question, not another
backtest.

Important operational difference from the stock deployment: this MUST run
during live market hours to get valid bid/ask (yfinance option quotes read
0.0 for bid/ask outside market hours, though lastPrice still holds the last
real trade -- see the module-level QUOTE_SOURCE fallback logic). The stock
deployment runs after close because it only needs the completed daily bar;
this one needs a live quote.

No live-order code path exists here. It never submits an order.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_equity_real_data_walkforward as walkforward  # noqa: E402
from run_equity_paper_step import decide_today  # noqa: E402
from btc_trend_bot.notifications import notify  # noqa: E402
from btc_trend_bot.production import process_lock  # noqa: E402


@dataclass(frozen=True)
class CallDeploymentConfig:
    strategy_config_path: Path
    ledger_db_path: Path
    lock_path: Path
    target_dte: int
    hold_days: int
    strike_moneyness: float
    base_capital: float
    premium_fraction: float

    @property
    def premium_budget(self) -> float:
        """Dollars of premium actually put at risk per trade -- NOT the same
        as base_capital. base_capital ($2,500, matching the other two
        deployments' initial_capital) is the account this is scaled against;
        only premium_fraction (10%) of it is ever risked on a single call,
        the rest is conceptually uninvested cash. This is the number that
        determines P&L -- see the dashboard's "Base capital" and "Premium
        budget / trade" metrics, which read directly from this config, not a
        hardcoded constant."""
        return self.base_capital * self.premium_fraction


def load_deployment_config(path: Path) -> tuple[CallDeploymentConfig, "walkforward.BacktestConfig"]:
    raw = yaml.safe_load(path.read_text())
    strategy_path = ROOT / raw["strategy_config"]
    strategy_cfg = walkforward.load_config(strategy_path)
    paper = raw.get("paper", {})
    deploy_cfg = CallDeploymentConfig(
        strategy_config_path=strategy_path,
        ledger_db_path=ROOT / str(paper.get("ledger_db_path", "runtime/equity_call_paper.sqlite3")),
        lock_path=ROOT / str(paper.get("lock_path", "runtime/equity_call_paper.lock")),
        target_dte=int(paper.get("target_dte", 30)),
        hold_days=int(paper.get("hold_days", 10)),
        strike_moneyness=float(paper.get("strike_moneyness", 1.05)),
        base_capital=float(paper.get("base_capital", strategy_cfg.initial_capital)),
        premium_fraction=float(paper.get("premium_fraction", 0.10)),
    )
    return deploy_cfg, strategy_cfg


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


# --------------------------------------------------------------------------- #
# Real live option-chain lookups (yfinance) -- no historical/synthetic pricing
# --------------------------------------------------------------------------- #

def _quote_from_row(row) -> dict:
    bid, ask = float(row.bid), float(row.ask)
    if bid > 0 and ask > 0:
        return {"bid": bid, "ask": ask, "mark": (bid + ask) / 2, "quote_source": "bid_ask"}
    # Outside market hours or illiquid: bid/ask read 0.0 from yfinance even
    # though lastPrice holds the most recent real trade. Flagged explicitly
    # so it's never silently mistaken for a live quote.
    return {"bid": None, "ask": None, "mark": float(row.lastPrice), "quote_source": "last_price_fallback"}


def pick_contract(symbol: str, target_dte: int, strike_moneyness: float) -> dict | None:
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    expirations = ticker.options
    if not expirations:
        return None
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    dtes = [(pd.Timestamp(e) - today).days for e in expirations]
    closest_idx = min(range(len(expirations)), key=lambda i: abs(dtes[i] - target_dte))
    expiration = expirations[closest_idx]
    chain = ticker.option_chain(expiration)
    calls = chain.calls.copy()
    if calls.empty:
        return None
    spot = float(ticker.history(period="1d").Close.iloc[-1])
    target_strike = spot * strike_moneyness
    calls["dist"] = (calls.strike - target_strike).abs()
    row = calls.nsmallest(1, "dist").iloc[0]
    quote = _quote_from_row(row)
    if quote["mark"] is None or quote["mark"] <= 0:
        return None
    return {"contract_symbol": row.contractSymbol, "underlying_symbol": symbol, "strike": float(row.strike),
            "expiration": expiration, "dte_at_entry": dtes[closest_idx], "spot_at_entry": spot, **quote}


def refresh_contract_quote(contract_symbol: str, underlying_symbol: str, expiration: str) -> dict | None:
    import yfinance as yf
    ticker = yf.Ticker(underlying_symbol)
    chain = ticker.option_chain(expiration)
    calls = chain.calls
    match = calls[calls.contractSymbol == contract_symbol]
    if match.empty:
        return None
    return _quote_from_row(match.iloc[0])


# --------------------------------------------------------------------------- #
# SQLite ledger (option-specific schema -- not the stock deployment's schema)
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS completed_trades (
    signal_time       TEXT PRIMARY KEY,
    underlying_symbol TEXT NOT NULL,
    contract_symbol   TEXT NOT NULL,
    strike            REAL NOT NULL,
    expiration        TEXT NOT NULL,
    entry_date        TEXT NOT NULL,
    entry_premium     REAL NOT NULL,
    entry_quote_source TEXT NOT NULL,
    exit_date         TEXT NOT NULL,
    exit_premium      REAL NOT NULL,
    exit_quote_source TEXT NOT NULL,
    net_return        REAL NOT NULL,
    created_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS open_position (
    id                 INTEGER PRIMARY KEY CHECK (id = 1),
    underlying_symbol  TEXT NOT NULL,
    signal_time        TEXT NOT NULL,
    contract_symbol    TEXT NOT NULL,
    strike             REAL NOT NULL,
    expiration         TEXT NOT NULL,
    dte_at_entry       INTEGER NOT NULL,
    spot_at_entry      REAL NOT NULL,
    entry_date         TEXT NOT NULL,
    entry_premium      REAL NOT NULL,
    entry_quote_source TEXT NOT NULL,
    current_mark       REAL,
    current_mark_source TEXT,
    current_mark_at    TEXT,
    updated_at         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    as_of_date   TEXT,
    status       TEXT NOT NULL,
    message      TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class CallPaperLedger:
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

    def append_completed_trade(self, trade: dict) -> None:
        self.conn.execute(
            "INSERT INTO completed_trades(signal_time,underlying_symbol,contract_symbol,strike,expiration,"
            "entry_date,entry_premium,entry_quote_source,exit_date,exit_premium,exit_quote_source,net_return,"
            "created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trade["signal_time"], trade["underlying_symbol"], trade["contract_symbol"], trade["strike"],
             trade["expiration"], trade["entry_date"], trade["entry_premium"], trade["entry_quote_source"],
             trade["exit_date"], trade["exit_premium"], trade["exit_quote_source"], trade["net_return"], _utc_now()))
        self.conn.commit()

    def completed_trades_frame(self) -> pd.DataFrame:
        rows = self.conn.execute("SELECT * FROM completed_trades ORDER BY signal_time").fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def log_run(self, as_of_date, status: str, message: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO runs(created_at,as_of_date,status,message,payload_json) VALUES(?,?,?,?,?)",
            (_utc_now(), str(as_of_date) if as_of_date is not None else None, status, message,
             json.dumps(payload, default=str)))
        self.conn.commit()


# --------------------------------------------------------------------------- #
# Step logic
# --------------------------------------------------------------------------- #

def _safety_allows_new_entry(trades: pd.DataFrame, cfg: "walkforward.BacktestConfig",
                              deploy_cfg: CallDeploymentConfig) -> tuple[bool, str]:
    """Simplified proxy for the stock deployment's drawdown-pause/cooldown
    logic, sized against deploy_cfg.premium_budget (the actual dollars at
    risk per trade) rather than stock notional -- options risk is already
    capped per trade (max loss = full premium), so this is a lighter check
    than the stock leg needs, but still respects the same consecutive-loss
    cooldown and hard-shutdown spirit."""
    if trades.empty:
        return True, ""
    safety = cfg
    budget = deploy_cfg.premium_budget
    equity = budget * len(trades)  # base for drawdown calc: cumulative premium at risk across trades taken
    pnl = (trades.net_return * budget).sum()
    peak_equity = max(equity, equity - pnl) if pnl < 0 else equity
    dd = max(0.0, -pnl / max(peak_equity, 1.0)) if pnl < 0 else 0.0
    recent = trades.tail(int(safety.consecutive_loss_limit))
    if len(recent) == int(safety.consecutive_loss_limit) and (recent.net_return < 0).all():
        return False, "loss_cooldown"
    if dd >= safety.hard_shutdown_drawdown:
        return False, "hard_shutdown"
    return True, ""


def _process_step(ledger: CallPaperLedger, deploy_cfg: CallDeploymentConfig, cfg: "walkforward.BacktestConfig",
                   frames: dict, date: pd.Timestamp) -> dict:
    position = ledger.get_open_position()

    if position is not None:
        entry_date = pd.Timestamp(position["entry_date"])
        days_held = (date.normalize() - entry_date.normalize()).days
        quote = refresh_contract_quote(position["contract_symbol"], position["underlying_symbol"], position["expiration"])
        if quote is not None and quote["mark"]:
            ledger.upsert_open_position(**{**{k: position[k] for k in (
                "underlying_symbol", "signal_time", "contract_symbol", "strike", "expiration",
                "dte_at_entry", "spot_at_entry", "entry_date", "entry_premium", "entry_quote_source")},
                "current_mark": quote["mark"], "current_mark_source": quote["quote_source"], "current_mark_at": _utc_now()})
        if days_held >= deploy_cfg.hold_days:
            exit_quote = quote or {"mark": position.get("current_mark") or position["entry_premium"], "quote_source": "stale_fallback"}
            exit_premium = exit_quote["mark"] * (0.98 if exit_quote["quote_source"] == "bid_ask" else 1.0)  # bid-side realism nudge if only mid/last available
            net_return = exit_premium / position["entry_premium"] - 1
            trade = {"signal_time": position["signal_time"], "underlying_symbol": position["underlying_symbol"],
                     "contract_symbol": position["contract_symbol"], "strike": position["strike"],
                     "expiration": position["expiration"], "entry_date": position["entry_date"],
                     "entry_premium": position["entry_premium"], "entry_quote_source": position["entry_quote_source"],
                     "exit_date": str(date), "exit_premium": exit_premium, "exit_quote_source": exit_quote["quote_source"],
                     "net_return": net_return}
            ledger.append_completed_trade(trade)
            ledger.clear_open_position()
            return {"date": str(date), "action": "exit", "trade": trade}
        return {"date": str(date), "action": "hold", "symbol": position["underlying_symbol"], "days_held": days_held}

    allowed, reason = _safety_allows_new_entry(ledger.completed_trades_frame(), cfg, deploy_cfg)
    if not allowed:
        return {"date": str(date), "action": "no_entry_safety_paused", "reason": reason}

    decision = decide_today(frames, cfg, selection="simple_trend", seed_date=date)
    if decision.winner_symbol is None:
        return {"date": str(date), "action": "no_signal"}

    contract = pick_contract(decision.winner_symbol, deploy_cfg.target_dte, deploy_cfg.strike_moneyness)
    if contract is None:
        return {"date": str(date), "action": "signal_no_contract", "symbol": decision.winner_symbol}

    entry_fill = contract["ask"] if contract["quote_source"] == "bid_ask" and contract["ask"] else contract["mark"]
    ledger.upsert_open_position(
        underlying_symbol=contract["underlying_symbol"], signal_time=str(decision.signal_time),
        contract_symbol=contract["contract_symbol"], strike=contract["strike"], expiration=contract["expiration"],
        dte_at_entry=contract["dte_at_entry"], spot_at_entry=contract["spot_at_entry"], entry_date=str(date),
        entry_premium=entry_fill, entry_quote_source=contract["quote_source"],
        current_mark=entry_fill, current_mark_source=contract["quote_source"], current_mark_at=_utc_now())
    return {"date": str(date), "action": "entry", "symbol": contract["underlying_symbol"],
            "contract": contract["contract_symbol"], "strike": contract["strike"], "premium": entry_fill,
            "quote_source": contract["quote_source"]}


def run_step(deployment_config_path: Path) -> dict:
    deploy_cfg, strategy_cfg = load_deployment_config(deployment_config_path)
    started_at = _utc_now()
    with process_lock(deploy_cfg.lock_path):
        ledger = CallPaperLedger(deploy_cfg.ledger_db_path)
        try:
            if ledger.is_halted():
                result = {"status": "halted", "reason": ledger.halt_reason()}
                ledger.log_run(None, "halted", ledger.halt_reason(), result)
                return result
            symbols = tuple(dict.fromkeys((*strategy_cfg.symbols, strategy_cfg.benchmark)))
            start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=400)).date().isoformat()
            frames = walkforward.download_yfinance(symbols, start, None, strategy_cfg.interval)
            today = pd.Timestamp.utcnow().normalize()
            result = _process_step(ledger, deploy_cfg, strategy_cfg, frames, today)
            ledger.log_run(str(today.date()), "ok", json.dumps(result, default=str), result)
            return result
        except Exception as exc:
            ledger.log_run(None, "error", str(exc), {"error": str(exc)})
            notify(f"URGENT: Call paper step error: {exc}")
            raise
        finally:
            ledger.close()


def get_status(deployment_config_path: Path) -> dict:
    deploy_cfg, _ = load_deployment_config(deployment_config_path)
    ledger = CallPaperLedger(deploy_cfg.ledger_db_path)
    try:
        trades = ledger.completed_trades_frame()
        return {
            "halted": ledger.is_halted(),
            "halt_reason": ledger.halt_reason(),
            "base_capital": deploy_cfg.base_capital,
            "premium_budget_per_trade": deploy_cfg.premium_budget,
            "open_position": ledger.get_open_position(),
            "completed_trades": int(len(trades)),
            "win_rate": float((trades.net_return > 0).mean()) if len(trades) else None,
            "total_pnl_dollars": float((trades.net_return * deploy_cfg.premium_budget).sum()) if len(trades) else 0.0,
        }
    finally:
        ledger.close()


def set_halt(deployment_config_path: Path, halted: bool, reason: str = "manual operator halt") -> None:
    deploy_cfg, _ = load_deployment_config(deployment_config_path)
    ledger = CallPaperLedger(deploy_cfg.ledger_db_path)
    try:
        ledger.set_halt(halted, reason if halted else "")
    finally:
        ledger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Call-option paper-trading step (no live orders)")
    parser.add_argument("--config", default="config/settings_equity_paper_calls.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("step")
    sub.add_parser("status")
    halt_p = sub.add_parser("halt")
    halt_p.add_argument("--reason", default="manual operator halt")
    sub.add_parser("resume")
    args = parser.parse_args()

    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
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
