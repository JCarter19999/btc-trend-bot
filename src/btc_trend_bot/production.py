from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from btc_trend_bot.audit import AuditStore
from btc_trend_bot.config import load_config
from btc_trend_bot.data import download_ohlcv, normalize_ohlcv, timeframe_to_timedelta
from btc_trend_bot.exchange import CoinbaseAdvancedBroker, SpotBroker, floor_decimal
from btc_trend_bot.notifications import notify
from btc_trend_bot.pipeline import prepare_strategy_frame


LIVE_ACK = "I_UNDERSTAND_THIS_PLACES_REAL_ORDERS"


@dataclass(frozen=True)
class DeploymentDecision:
    bar_timestamp: str
    close: Decimal
    target_position: Decimal
    side: str
    order_size: Decimal
    reason: str


@contextmanager
def process_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
        with path.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
    except BlockingIOError as exc:
        raise RuntimeError("Another deployment step is already running") from exc


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _client_order_id(product: str, bar_timestamp: str, side: str) -> str:
    digest = hashlib.sha256(f"btc-trend-v1|{product}|{bar_timestamp}|{side}".encode()).hexdigest()[:24]
    return f"btcv1-{digest}"


def _latest_signal(cfg: dict[str, Any]) -> tuple[str, Decimal, Decimal]:
    market = cfg["market"]
    deployment = cfg["deployment"]
    lookback_bars = int(deployment.get("lookback_bars", 900))
    step = timeframe_to_timedelta(str(market["timeframe"]))
    start = (pd.Timestamp.now(tz="UTC") - step * (lookback_bars + 10)).isoformat()
    frame = download_ohlcv(
        exchange_id=str(market["exchange"]), symbol=str(market["symbol"]),
        timeframe=str(market["timeframe"]), start=start, max_bars=lookback_bars,
    )
    normalized, _ = normalize_ohlcv(frame, timeframe=str(market["timeframe"]))
    strategy_frame = prepare_strategy_frame(normalized, cfg)
    latest = strategy_frame.iloc[-1]
    return (
        pd.Timestamp(latest["timestamp"]).isoformat(),
        Decimal(str(latest["close"])),
        Decimal(str(max(0.0, min(1.0, float(latest["target_position"]))))),
    )


def decide(*, bar_timestamp: str, close: Decimal, target: Decimal,
           quote_available: Decimal, base_available: Decimal,
           allocation_cap: Decimal, cash_reserve: Decimal,
           base_increment: Decimal, quote_increment: Decimal,
           min_notional: Decimal, rebalance_tolerance: Decimal) -> DeploymentDecision:
    equity = quote_available + base_available * close
    desired_base = (equity * allocation_cap * target) / close
    delta_base = desired_base - base_available
    tolerance_base = (equity * rebalance_tolerance) / close
    if abs(delta_base) <= tolerance_base:
        return DeploymentDecision(bar_timestamp, close, target, "none", Decimal("0"), "within tolerance")
    if delta_base > 0:
        spendable = max(Decimal("0"), quote_available - cash_reserve)
        quote_size = floor_decimal(min(delta_base * close, spendable), quote_increment)
        if quote_size < min_notional:
            return DeploymentDecision(bar_timestamp, close, target, "none", Decimal("0"), "buy below minimum notional")
        return DeploymentDecision(bar_timestamp, close, target, "buy", quote_size, "increase BTC exposure")
    base_size = floor_decimal(min(abs(delta_base), base_available), base_increment)
    if base_size * close < min_notional:
        return DeploymentDecision(bar_timestamp, close, target, "none", Decimal("0"), "sell below minimum notional")
    return DeploymentDecision(bar_timestamp, close, target, "sell", base_size, "reduce BTC exposure")


def deployment_status(config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    d = cfg["deployment"]
    path = Path(str(d["audit_db_path"]))
    store = AuditStore(path)
    try:
        return {
            "mode": d["mode"],
            "halted": store.get_state("halted") == "true",
            "halt_reason": store.get_state("halt_reason"),
            "last_processed_bar": store.get_state("last_processed_bar"),
            "database": str(path),
        }
    finally:
        store.close()


def set_halt(config_path: str, halted: bool, reason: str = "manual") -> None:
    cfg = load_config(config_path)
    store = AuditStore(Path(str(cfg["deployment"]["audit_db_path"])))
    try:
        store.set_state("halted", "true" if halted else "false")
        store.set_state("halt_reason", reason if halted else "")
    finally:
        store.close()


def run_deployment_step(config_path: str, broker: SpotBroker | None = None) -> DeploymentDecision:
    cfg = load_config(config_path)
    d = cfg["deployment"]
    mode = str(d["mode"]).lower()
    if mode not in {"dry_run", "live"}:
        raise ValueError("deployment.mode must be dry_run or live")
    if mode == "live" and os.getenv("LIVE_TRADING_ACK") != LIVE_ACK:
        raise RuntimeError("Live mode blocked: LIVE_TRADING_ACK is missing or incorrect")

    lock_path = Path(str(d.get("lock_path", "runtime/btc-trend-bot.lock")))
    with process_lock(lock_path):
        store = AuditStore(Path(str(d["audit_db_path"])))
        try:
            if store.get_state("halted") == "true":
                reason = store.get_state("halt_reason") or "unknown"
                raise RuntimeError(f"Trading is halted: {reason}")

            bar_ts, close, target = _latest_signal(cfg)
            last_bar = store.get_state("last_processed_bar")
            if last_bar == bar_ts:
                decision = DeploymentDecision(bar_ts, close, target, "none", Decimal("0"), "bar already processed")
                store.log_run(_utc_now(), bar_ts, mode, "noop", decision.reason)
                return decision

            if broker is None:
                broker = CoinbaseAdvancedBroker(timeout_seconds=float(d.get("api_timeout_seconds", 10)))
            product_id = str(d["product_id"])
            base_currency, quote_currency = product_id.split("-", 1)
            balances = broker.balances(base_currency, quote_currency)
            decision = decide(
                bar_timestamp=bar_ts, close=close, target=target,
                quote_available=balances.quote_available, base_available=balances.base_available,
                allocation_cap=Decimal(str(d.get("allocation_cap", 1.0))),
                cash_reserve=Decimal(str(d.get("cash_reserve", 10.0))),
                base_increment=Decimal(str(d.get("base_increment", "0.00000001"))),
                quote_increment=Decimal(str(d.get("quote_increment", "0.01"))),
                min_notional=Decimal(str(d.get("min_notional", "10"))),
                rebalance_tolerance=Decimal(str(d.get("rebalance_tolerance", "0.01"))),
            )
            payload = {"decision": decision.__dict__, "balances": balances.__dict__}
            payload = json.loads(json.dumps(payload, default=str))

            if decision.side == "none":
                store.set_state("last_processed_bar", bar_ts)
                store.log_run(_utc_now(), bar_ts, mode, "noop", decision.reason, payload)
                return decision

            client_id = _client_order_id(product_id, bar_ts, decision.side)
            if store.order_exists(client_id):
                store.set_state("halted", "true")
                store.set_state("halt_reason", f"duplicate order identity detected: {client_id}")
                raise RuntimeError("Duplicate order identity detected; trading halted")

            if mode == "dry_run":
                store.log_order(client_order_id=client_id, created_at=_utc_now(), bar_timestamp=bar_ts,
                                side=decision.side, requested_size=str(decision.order_size), order_id=None,
                                status="dry_run", payload=payload)
                store.set_state("last_processed_bar", bar_ts)
                store.log_run(_utc_now(), bar_ts, mode, "success", f"dry-run {decision.side}", payload)
                notify(f"BTC trend bot dry-run: {decision.side} {decision.order_size} at bar {bar_ts}")
                return decision

            result = (broker.market_buy(product_id, decision.order_size, client_id)
                      if decision.side == "buy"
                      else broker.market_sell(product_id, decision.order_size, client_id))
            status = "submitted" if result.success else "failed"
            store.log_order(client_order_id=client_id, created_at=_utc_now(), bar_timestamp=bar_ts,
                            side=decision.side, requested_size=str(decision.order_size), order_id=result.order_id,
                            status=status, payload=result.raw)
            if not result.success:
                store.set_state("halted", "true")
                store.set_state("halt_reason", "exchange rejected order")
                notify(f"URGENT: BTC bot order failed and halted: {result.raw}")
                raise RuntimeError(f"Exchange rejected order: {result.raw}")

            time.sleep(float(d.get("post_order_wait_seconds", 2)))
            after = broker.balances(base_currency, quote_currency)
            changed = (after.base_available != balances.base_available or
                       after.quote_available != balances.quote_available)
            if not changed:
                store.set_state("halted", "true")
                store.set_state("halt_reason", "post-order balances did not change")
                notify("URGENT: BTC bot submitted an order but reconciliation failed; trading halted")
                raise RuntimeError("Post-order reconciliation failed; trading halted")

            store.set_state("last_processed_bar", bar_ts)
            store.log_run(_utc_now(), bar_ts, mode, "success", f"submitted {decision.side}",
                          {**payload, "order_id": result.order_id, "balances_after": after.__dict__})
            notify(f"BTC trend bot LIVE: {decision.side} {decision.order_size}; order {result.order_id}")
            return decision
        except Exception as exc:
            try:
                store.log_run(_utc_now(), None, mode, "error", str(exc))
            finally:
                notify(f"URGENT: BTC trend bot error: {exc}")
            raise
        finally:
            store.close()
