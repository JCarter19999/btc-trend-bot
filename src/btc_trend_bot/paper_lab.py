"""Independent five-minute BTC paper-trading laboratory.

The module intentionally contains no live-order path. It uses public market data,
maintains independent simulated portfolios in SQLite, and can mirror sampled
snapshots and simulated trades to Supabase through a durable JSONL outbox.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from btc_trend_bot.data import download_ohlcv, normalize_ohlcv, timeframe_to_timedelta


@dataclass(frozen=True)
class MarketQuote:
    timestamp: str
    mark: float
    bid: float
    ask: float


@dataclass(frozen=True)
class CandleFeatures:
    bar_timestamp: str
    signal_close: float
    streak_direction: int
    streak_length: int
    run_return_bps: float
    relative_volume: float
    body_fraction: float
    broader_trend_up: bool
    regime_spread_bps: float
    momentum_bps: float


@dataclass
class AccountState:
    strategy_id: str
    initial_cash: float
    cash: float
    btc: float
    gross_cash: float
    gross_btc: float
    target_position: float = 0.0
    last_bar_timestamp: str | None = None
    peak_equity: float | None = None
    total_fees: float = 0.0
    total_spread: float = 0.0
    total_slippage: float = 0.0
    total_turnover: float = 0.0
    trade_count: int = 0
    bars_processed: int = 0


@dataclass(frozen=True)
class StrategyDecision:
    strategy_id: str
    target_position: float
    signal: str
    reason: str


@dataclass(frozen=True)
class SimulatedTrade:
    strategy_id: str
    bar_timestamp: str
    quote_timestamp: str
    side: str
    btc_delta: float
    mark_price: float
    reference_price: float
    fill_price: float
    gross_notional: float
    fee: float
    spread_cost: float
    slippage_cost: float
    cash_after: float
    btc_after: float
    equity_after: float


@dataclass(frozen=True)
class StepResult:
    state: AccountState
    decision: StrategyDecision
    snapshot: dict[str, Any]
    trade: SimulatedTrade | None




def load_paper_lab_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Paper-lab config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Paper-lab config must contain a YAML mapping.")
    for section in ("market", "paper_lab", "paper_strategies"):
        if section not in raw:
            raise ValueError(f"Missing paper-lab config section: {section}")
    return raw

def utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


@contextmanager
def process_lock(path: Path):
    """Prevent overlapping five-minute jobs on Linux."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        yield
        return

    with path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another five-minute paper step is already running") from exc
        yield


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def fetch_completed_bars(cfg: dict[str, Any], max_bars: int | None = None) -> pd.DataFrame:
    market = cfg["market"]
    lab = cfg["paper_lab"]
    timeframe = str(market["timeframe"])
    lookback = int(max_bars or lab.get("lookback_bars", 500))
    step = timeframe_to_timedelta(timeframe)
    start = (pd.Timestamp.now(tz="UTC") - step * (lookback + 20)).isoformat()

    frame = download_ohlcv(
        exchange_id=str(market["exchange"]),
        symbol=str(market["symbol"]),
        timeframe=timeframe,
        start=start,
        max_bars=lookback + 20,
    )
    normalized, _ = normalize_ohlcv(frame, timeframe=timeframe)
    if len(normalized) < 10:
        raise RuntimeError(f"Only {len(normalized)} completed candles were returned.")
    return normalized.tail(lookback).reset_index(drop=True)


def fetch_public_quote(exchange_id: str, symbol: str) -> MarketQuote:
    """Fetch a public ticker without using exchange credentials."""
    try:
        import ccxt
    except ImportError as exc:
        raise RuntimeError("CCXT is not installed.") from exc

    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown CCXT exchange: {exchange_id}")

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()
    ticker = exchange.fetch_ticker(symbol)

    mark = _finite(ticker.get("last") or ticker.get("close"))
    bid = _finite(ticker.get("bid"), mark)
    ask = _finite(ticker.get("ask"), mark)
    if mark <= 0:
        mark = next((value for value in (bid, ask) if value > 0), 0.0)
    if mark <= 0:
        raise RuntimeError("Ticker did not contain a usable positive market price.")
    if bid <= 0:
        bid = mark
    if ask <= 0:
        ask = mark
    if bid > ask:
        bid, ask = ask, bid
    return MarketQuote(timestamp=utc_now_iso(), mark=mark, bid=bid, ask=ask)


def add_candle_features(frame: pd.DataFrame, feature_cfg: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    doji_bps = float(feature_cfg.get("doji_threshold_bps", 1.0))
    body_return_bps = (out["close"] / out["open"] - 1.0) * 10_000.0

    direction = pd.Series(0, index=out.index, dtype="int64")
    direction.loc[body_return_bps > doji_bps] = 1
    direction.loc[body_return_bps < -doji_bps] = -1

    # A doji starts a fresh group and receives a zero-length run.
    group_change = direction.ne(direction.shift(1)) | direction.eq(0)
    run_group = group_change.cumsum()
    streak_length = direction.groupby(run_group).cumcount() + 1
    streak_length = streak_length.where(direction.ne(0), 0)
    run_open = out["open"].groupby(run_group).transform("first")
    run_return_bps = (out["close"] / run_open - 1.0) * 10_000.0
    run_return_bps = run_return_bps.where(direction.ne(0), 0.0)

    volume_window = int(feature_cfg.get("volume_window_bars", 48))
    previous_volume_median = out["volume"].shift(1).rolling(
        volume_window,
        min_periods=max(1, volume_window // 2),
    ).median()
    relative_volume = out["volume"] / previous_volume_median.replace(0, float("nan"))

    candle_range = (out["high"] - out["low"]).abs()
    body_fraction = (
        (out["close"] - out["open"]).abs()
        / candle_range.replace(0, float("nan"))
    )

    trend_bars = int(feature_cfg.get("broader_trend_ema_bars", 48))
    trend_ema = out["close"].ewm(
        span=trend_bars,
        adjust=False,
        min_periods=trend_bars,
    ).mean()

    regime_fast_bars = int(feature_cfg.get("regime_fast_ema_bars", 12))
    regime_slow_bars = int(feature_cfg.get("regime_slow_ema_bars", 72))
    if regime_fast_bars >= regime_slow_bars:
        raise ValueError("regime_fast_ema_bars must be smaller than regime_slow_ema_bars")
    regime_fast = out["close"].ewm(
        span=regime_fast_bars,
        adjust=False,
        min_periods=regime_fast_bars,
    ).mean()
    regime_slow = out["close"].ewm(
        span=regime_slow_bars,
        adjust=False,
        min_periods=regime_slow_bars,
    ).mean()
    regime_spread_bps = (regime_fast / regime_slow - 1.0) * 10_000.0

    momentum_bars = int(feature_cfg.get("momentum_bars", 12))
    if momentum_bars <= 0:
        raise ValueError("momentum_bars must be positive")
    momentum_bps = (out["close"] / out["close"].shift(momentum_bars) - 1.0) * 10_000.0

    out["streak_direction"] = direction
    out["streak_length"] = streak_length.astype(int)
    out["run_return_bps"] = run_return_bps.fillna(0.0)
    out["relative_volume"] = relative_volume.replace(
        [float("inf"), -float("inf")],
        float("nan"),
    )
    out["body_fraction"] = body_fraction.fillna(0.0)
    out["broader_trend_up"] = (out["close"] > trend_ema).fillna(False)
    out["regime_spread_bps"] = regime_spread_bps
    out["momentum_bps"] = momentum_bps

    # The hypothesis concerns the color/body of the next candle, so this is
    # next-open to next-close rather than close-to-close return.
    out["next_candle_return_bps"] = (
        (out["close"].shift(-1) / out["open"].shift(-1) - 1.0) * 10_000.0
    )
    return out


def row_to_features(row: pd.Series) -> CandleFeatures:
    return CandleFeatures(
        bar_timestamp=pd.Timestamp(row["timestamp"]).isoformat(),
        signal_close=float(row["close"]),
        streak_direction=int(row["streak_direction"]),
        streak_length=int(row["streak_length"]),
        run_return_bps=float(row["run_return_bps"]),
        relative_volume=_finite(row.get("relative_volume"), 0.0),
        body_fraction=float(row["body_fraction"]),
        broader_trend_up=bool(row["broader_trend_up"]),
        regime_spread_bps=_finite(row.get("regime_spread_bps"), 0.0),
        momentum_bps=_finite(row.get("momentum_bps"), 0.0),
    )


def round_trip_cost_bps(lab_cfg: dict[str, Any]) -> float:
    fee = float(lab_cfg.get("fee_bps_per_side", 60.0))
    slippage = float(lab_cfg.get("slippage_bps_per_side", 5.0))
    assumed_spread = float(lab_cfg.get("assumed_spread_bps_per_side", 0.0))
    return 2.0 * (fee + slippage + assumed_spread)


def decide_target(
    strategy_cfg: dict[str, Any],
    features: CandleFeatures,
    previous_target: float,
    lab_cfg: dict[str, Any],
) -> StrategyDecision:
    strategy_id = str(strategy_cfg["id"])
    strategy_type = str(strategy_cfg["type"])

    if strategy_type == "cash":
        return StrategyDecision(strategy_id, 0.0, "cash", "cash benchmark")

    if strategy_type == "buy_hold":
        return StrategyDecision(strategy_id, 1.0, "long", "buy-and-hold benchmark")

    if strategy_type == "candle_swing":
        entry_bars = int(strategy_cfg.get("entry_run_bars", 2))
        exit_bars = int(strategy_cfg.get("exit_run_bars", 2))

        if previous_target <= 0.0:
            failures: list[str] = []
            if not (
                features.streak_direction > 0
                and features.streak_length >= entry_bars
            ):
                failures.append(f"need {entry_bars}-bar upward run")

            min_run = float(strategy_cfg.get("min_run_return_bps", 0.0))
            if features.run_return_bps < min_run:
                failures.append(
                    f"run {features.run_return_bps:.1f}bps below {min_run:.1f}bps"
                )

            min_volume = float(strategy_cfg.get("min_relative_volume", 0.0))
            if features.relative_volume < min_volume:
                failures.append(
                    f"relative volume {features.relative_volume:.2f} below {min_volume:.2f}"
                )

            min_body = float(strategy_cfg.get("min_body_fraction", 0.0))
            if features.body_fraction < min_body:
                failures.append(
                    f"body fraction {features.body_fraction:.2f} below {min_body:.2f}"
                )

            entry_spread = float(strategy_cfg.get("entry_regime_spread_bps", 0.0))
            if features.regime_spread_bps < entry_spread:
                failures.append(
                    f"EMA spread {features.regime_spread_bps:.1f}bps below "
                    f"{entry_spread:.1f}bps"
                )

            entry_momentum = float(strategy_cfg.get("entry_momentum_bps", 0.0))
            if features.momentum_bps < entry_momentum:
                failures.append(
                    f"momentum {features.momentum_bps:.1f}bps below "
                    f"{entry_momentum:.1f}bps"
                )

            if (
                bool(strategy_cfg.get("require_broader_uptrend", True))
                and not features.broader_trend_up
            ):
                failures.append("price below broader EMA")

            if failures:
                return StrategyDecision(strategy_id, 0.0, "hold_cash", "; ".join(failures))

            return StrategyDecision(
                strategy_id,
                1.0,
                "enter",
                f"{features.streak_length}-bar upward run inside confirmed regime",
            )

        # Exit uses hysteresis: an opposite candle run alone is not enough. A
        # configurable number of three independent deterioration signals must be
        # present, which sharply reduces five-minute whipsaw turnover.
        down_run = (
            features.streak_direction < 0
            and features.streak_length >= exit_bars
        )
        exit_spread = float(strategy_cfg.get("exit_regime_spread_bps", 0.0))
        weak_regime = features.regime_spread_bps <= exit_spread
        exit_momentum = float(strategy_cfg.get("exit_momentum_bps", 0.0))
        weak_momentum = features.momentum_bps <= exit_momentum
        confirmations = int(down_run) + int(weak_regime) + int(weak_momentum)
        required = int(strategy_cfg.get("exit_confirmations_required", 2))
        if required < 1 or required > 3:
            raise ValueError("exit_confirmations_required must be between 1 and 3")

        if confirmations >= required:
            reasons = []
            if down_run:
                reasons.append(f"{features.streak_length}-bar downward run")
            if weak_regime:
                reasons.append(
                    f"EMA spread {features.regime_spread_bps:.1f}bps <= "
                    f"{exit_spread:.1f}bps"
                )
            if weak_momentum:
                reasons.append(
                    f"momentum {features.momentum_bps:.1f}bps <= "
                    f"{exit_momentum:.1f}bps"
                )
            return StrategyDecision(strategy_id, 0.0, "exit", "; ".join(reasons))

        return StrategyDecision(
            strategy_id,
            1.0,
            "hold_btc",
            f"only {confirmations}/{required} exit confirmations",
        )

    if strategy_type != "candle_run":
        raise ValueError(f"Unknown paper strategy type: {strategy_type}")

    # Retained for explicit high-turnover stress tests. It is not included in the
    # default paper deployment after the first research run demonstrated that
    # Coinbase Intro-tier costs overwhelm the gross signal.
    entry_bars = int(strategy_cfg.get("entry_run_bars", 2))
    exit_bars = int(strategy_cfg.get("exit_run_bars", 2))

    if features.streak_direction < 0 and features.streak_length >= exit_bars:
        return StrategyDecision(
            strategy_id,
            0.0,
            "exit",
            f"{features.streak_length}-bar downward run",
        )

    if features.streak_direction > 0 and features.streak_length >= entry_bars:
        if bool(strategy_cfg.get("fee_aware", False)):
            hurdle = max(
                float(strategy_cfg.get("min_run_return_bps", 0.0)),
                round_trip_cost_bps(lab_cfg)
                * float(strategy_cfg.get("cost_hurdle_multiplier", 1.0)),
            )
            failures: list[str] = []
            if features.run_return_bps < hurdle:
                failures.append(
                    f"run {features.run_return_bps:.1f}bps below {hurdle:.1f}bps hurdle"
                )
            min_volume = float(strategy_cfg.get("min_relative_volume", 0.0))
            if features.relative_volume < min_volume:
                failures.append(
                    f"relative volume {features.relative_volume:.2f} below {min_volume:.2f}"
                )
            min_body = float(strategy_cfg.get("min_body_fraction", 0.0))
            if features.body_fraction < min_body:
                failures.append(
                    f"body fraction {features.body_fraction:.2f} below {min_body:.2f}"
                )
            if (
                bool(strategy_cfg.get("require_broader_uptrend", True))
                and not features.broader_trend_up
            ):
                failures.append("price below broader EMA")

            if failures:
                return StrategyDecision(
                    strategy_id,
                    previous_target,
                    "hold",
                    "; ".join(failures),
                )

        return StrategyDecision(
            strategy_id,
            1.0,
            "enter",
            f"{features.streak_length}-bar upward run",
        )

    return StrategyDecision(
        strategy_id,
        previous_target,
        "hold",
        "no confirmed opposite run",
    )


def _mark_equity(state: AccountState, price: float) -> float:
    return state.cash + state.btc * price


def _gross_equity(state: AccountState, price: float) -> float:
    return state.gross_cash + state.gross_btc * price


def execute_decision(
    state: AccountState,
    decision: StrategyDecision,
    features: CandleFeatures,
    quote: MarketQuote,
    lab_cfg: dict[str, Any],
) -> StepResult:
    if state.last_bar_timestamp == features.bar_timestamp:
        raise ValueError(f"{state.strategy_id} already processed {features.bar_timestamp}")

    fee_rate = float(lab_cfg.get("fee_bps_per_side", 60.0)) / 10_000.0
    slippage_rate = float(lab_cfg.get("slippage_bps_per_side", 5.0)) / 10_000.0
    allocation_cap = float(lab_cfg.get("allocation_cap", 1.0))
    min_notional = float(lab_cfg.get("min_notional", 10.0))
    tolerance_bps = float(lab_cfg.get("rebalance_tolerance_bps", 2.0))

    target = min(1.0, max(0.0, decision.target_position)) * allocation_cap
    equity_before = _mark_equity(state, quote.mark)
    gross_equity_before = _gross_equity(state, quote.mark)

    desired_btc = equity_before * target / quote.mark
    delta_btc = desired_btc - state.btc
    tolerance_notional = equity_before * tolerance_bps / 10_000.0

    trade: SimulatedTrade | None = None
    if abs(delta_btc) * quote.mark >= max(min_notional, tolerance_notional):
        if delta_btc > 0:
            side = "buy"
            reference_price = quote.ask
            fill_price = reference_price * (1.0 + slippage_rate)
            affordable = max(0.0, state.cash / (fill_price * (1.0 + fee_rate)))
            executed_btc = min(delta_btc, affordable)
        else:
            side = "sell"
            reference_price = quote.bid
            fill_price = reference_price * (1.0 - slippage_rate)
            executed_btc = -min(abs(delta_btc), state.btc)

        gross_notional = abs(executed_btc) * fill_price
        fee = gross_notional * fee_rate
        spread_cost = abs(executed_btc) * abs(reference_price - quote.mark)
        slippage_cost = abs(executed_btc) * abs(fill_price - reference_price)

        if abs(executed_btc) > 1e-12 and gross_notional >= min_notional:
            if executed_btc > 0:
                state.cash -= gross_notional + fee
                state.btc += executed_btc
            else:
                state.cash += gross_notional - fee
                state.btc += executed_btc

            # Avoid microscopic floating-point residue.
            if abs(state.cash) < 1e-10:
                state.cash = 0.0
            if abs(state.btc) < 1e-12:
                state.btc = 0.0

            state.total_fees += fee
            state.total_spread += spread_cost
            state.total_slippage += slippage_cost
            state.total_turnover += gross_notional
            state.trade_count += 1
            trade = SimulatedTrade(
                strategy_id=state.strategy_id,
                bar_timestamp=features.bar_timestamp,
                quote_timestamp=quote.timestamp,
                side=side,
                btc_delta=executed_btc,
                mark_price=quote.mark,
                reference_price=reference_price,
                fill_price=fill_price,
                gross_notional=gross_notional,
                fee=fee,
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
                cash_after=state.cash,
                btc_after=state.btc,
                equity_after=_mark_equity(state, quote.mark),
            )

    # Frictionless shadow account follows the identical target. Its divergence
    # from the net account measures the cumulative drag of fees/spread/slippage.
    desired_gross_btc = gross_equity_before * target / quote.mark
    gross_delta = desired_gross_btc - state.gross_btc
    state.gross_cash -= gross_delta * quote.mark
    state.gross_btc += gross_delta

    state.target_position = decision.target_position
    state.last_bar_timestamp = features.bar_timestamp
    state.bars_processed += 1

    equity = _mark_equity(state, quote.mark)
    gross_equity = _gross_equity(state, quote.mark)
    state.peak_equity = max(float(state.peak_equity or state.initial_cash), equity)
    drawdown = equity / state.peak_equity - 1.0
    return_pct = equity / state.initial_cash - 1.0
    gross_return_pct = gross_equity / state.initial_cash - 1.0

    snapshot = {
        "strategy_id": state.strategy_id,
        "bar_timestamp": features.bar_timestamp,
        "quote_timestamp": quote.timestamp,
        "signal_close": features.signal_close,
        "mark_price": quote.mark,
        "bid": quote.bid,
        "ask": quote.ask,
        "signal": decision.signal,
        "reason": decision.reason,
        "target_position": decision.target_position,
        "cash": state.cash,
        "btc": state.btc,
        "equity": equity,
        "gross_equity": gross_equity,
        "cost_drag": gross_equity - equity,
        "return_pct": return_pct,
        "gross_return_pct": gross_return_pct,
        "drawdown": drawdown,
        "peak_equity": state.peak_equity,
        "total_fees": state.total_fees,
        "total_spread": state.total_spread,
        "total_slippage": state.total_slippage,
        "total_turnover": state.total_turnover,
        "trade_count": state.trade_count,
        "bars_processed": state.bars_processed,
        "streak_direction": features.streak_direction,
        "streak_length": features.streak_length,
        "run_return_bps": features.run_return_bps,
        "relative_volume": features.relative_volume,
        "body_fraction": features.body_fraction,
        "broader_trend_up": features.broader_trend_up,
        "regime_spread_bps": features.regime_spread_bps,
        "momentum_bps": features.momentum_bps,
        "created_at": utc_now_iso(),
    }
    return StepResult(state=state, decision=decision, snapshot=snapshot, trade=trade)


class PaperStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self.conn.close()

    def _initialize(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;

            CREATE TABLE IF NOT EXISTS paper_accounts (
                strategy_id TEXT PRIMARY KEY,
                initial_cash REAL NOT NULL,
                cash REAL NOT NULL,
                btc REAL NOT NULL,
                gross_cash REAL NOT NULL,
                gross_btc REAL NOT NULL,
                target_position REAL NOT NULL,
                last_bar_timestamp TEXT,
                peak_equity REAL,
                total_fees REAL NOT NULL,
                total_spread REAL NOT NULL,
                total_slippage REAL NOT NULL,
                total_turnover REAL NOT NULL,
                trade_count INTEGER NOT NULL,
                bars_processed INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_snapshots (
                strategy_id TEXT NOT NULL,
                bar_timestamp TEXT NOT NULL,
                quote_timestamp TEXT NOT NULL,
                signal_close REAL NOT NULL,
                mark_price REAL NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                signal TEXT NOT NULL,
                reason TEXT NOT NULL,
                target_position REAL NOT NULL,
                cash REAL NOT NULL,
                btc REAL NOT NULL,
                equity REAL NOT NULL,
                gross_equity REAL NOT NULL,
                cost_drag REAL NOT NULL,
                return_pct REAL NOT NULL,
                gross_return_pct REAL NOT NULL,
                drawdown REAL NOT NULL,
                peak_equity REAL NOT NULL,
                total_fees REAL NOT NULL,
                total_spread REAL NOT NULL,
                total_slippage REAL NOT NULL,
                total_turnover REAL NOT NULL,
                trade_count INTEGER NOT NULL,
                bars_processed INTEGER NOT NULL,
                streak_direction INTEGER NOT NULL,
                streak_length INTEGER NOT NULL,
                run_return_bps REAL NOT NULL,
                relative_volume REAL NOT NULL,
                body_fraction REAL NOT NULL,
                broader_trend_up INTEGER NOT NULL,
                regime_spread_bps REAL NOT NULL DEFAULT 0.0,
                momentum_bps REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (strategy_id, bar_timestamp)
            );

            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                bar_timestamp TEXT NOT NULL,
                quote_timestamp TEXT NOT NULL,
                side TEXT NOT NULL,
                btc_delta REAL NOT NULL,
                mark_price REAL NOT NULL,
                reference_price REAL NOT NULL,
                fill_price REAL NOT NULL,
                gross_notional REAL NOT NULL,
                fee REAL NOT NULL,
                spread_cost REAL NOT NULL,
                slippage_cost REAL NOT NULL,
                cash_after REAL NOT NULL,
                btc_after REAL NOT NULL,
                equity_after REAL NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (strategy_id, bar_timestamp, side)
            );

            CREATE INDEX IF NOT EXISTS idx_paper_snapshots_timestamp
                ON paper_snapshots(bar_timestamp);
            CREATE INDEX IF NOT EXISTS idx_paper_trades_timestamp
                ON paper_trades(bar_timestamp);
            """
        )
        snapshot_columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(paper_snapshots)")
        }
        if "regime_spread_bps" not in snapshot_columns:
            self.conn.execute(
                "ALTER TABLE paper_snapshots ADD COLUMN regime_spread_bps REAL NOT NULL DEFAULT 0.0"
            )
        if "momentum_bps" not in snapshot_columns:
            self.conn.execute(
                "ALTER TABLE paper_snapshots ADD COLUMN momentum_bps REAL NOT NULL DEFAULT 0.0"
            )
        self.conn.commit()

    def get_or_create_account(self, strategy_id: str, initial_cash: float) -> tuple[AccountState, bool]:
        row = self.conn.execute(
            "SELECT * FROM paper_accounts WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        if row is not None:
            return (
                AccountState(
                    strategy_id=row["strategy_id"],
                    initial_cash=row["initial_cash"],
                    cash=row["cash"],
                    btc=row["btc"],
                    gross_cash=row["gross_cash"],
                    gross_btc=row["gross_btc"],
                    target_position=row["target_position"],
                    last_bar_timestamp=row["last_bar_timestamp"],
                    peak_equity=row["peak_equity"],
                    total_fees=row["total_fees"],
                    total_spread=row["total_spread"],
                    total_slippage=row["total_slippage"],
                    total_turnover=row["total_turnover"],
                    trade_count=row["trade_count"],
                    bars_processed=row["bars_processed"],
                ),
                False,
            )

        now = utc_now_iso()
        state = AccountState(
            strategy_id=strategy_id,
            initial_cash=initial_cash,
            cash=initial_cash,
            btc=0.0,
            gross_cash=initial_cash,
            gross_btc=0.0,
            peak_equity=initial_cash,
        )
        self.conn.execute(
            """
            INSERT INTO paper_accounts (
                strategy_id, initial_cash, cash, btc, gross_cash, gross_btc,
                target_position, last_bar_timestamp, peak_equity, total_fees,
                total_spread, total_slippage, total_turnover, trade_count,
                bars_processed, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.strategy_id,
                state.initial_cash,
                state.cash,
                state.btc,
                state.gross_cash,
                state.gross_btc,
                state.target_position,
                state.last_bar_timestamp,
                state.peak_equity,
                state.total_fees,
                state.total_spread,
                state.total_slippage,
                state.total_turnover,
                state.trade_count,
                state.bars_processed,
                now,
                now,
            ),
        )
        self.conn.commit()
        return state, True

    def persist_result(self, result: StepResult) -> bool:
        state = result.state
        snapshot = result.snapshot
        with self.conn:
            existing = self.conn.execute(
                """
                SELECT 1 FROM paper_snapshots
                WHERE strategy_id = ? AND bar_timestamp = ?
                """,
                (state.strategy_id, snapshot["bar_timestamp"]),
            ).fetchone()
            if existing is not None:
                return False

            self.conn.execute(
                """
                UPDATE paper_accounts SET
                    cash = ?, btc = ?, gross_cash = ?, gross_btc = ?,
                    target_position = ?, last_bar_timestamp = ?, peak_equity = ?,
                    total_fees = ?, total_spread = ?, total_slippage = ?,
                    total_turnover = ?, trade_count = ?, bars_processed = ?,
                    updated_at = ?
                WHERE strategy_id = ?
                """,
                (
                    state.cash,
                    state.btc,
                    state.gross_cash,
                    state.gross_btc,
                    state.target_position,
                    state.last_bar_timestamp,
                    state.peak_equity,
                    state.total_fees,
                    state.total_spread,
                    state.total_slippage,
                    state.total_turnover,
                    state.trade_count,
                    state.bars_processed,
                    utc_now_iso(),
                    state.strategy_id,
                ),
            )

            columns = list(snapshot)
            values = [
                int(value) if isinstance(value, bool) else value
                for value in snapshot.values()
            ]
            self.conn.execute(
                f"INSERT INTO paper_snapshots ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                values,
            )

            if result.trade is not None:
                trade_row = asdict(result.trade)
                trade_row["created_at"] = utc_now_iso()
                trade_columns = list(trade_row)
                self.conn.execute(
                    f"INSERT OR IGNORE INTO paper_trades ({', '.join(trade_columns)}) "
                    f"VALUES ({', '.join('?' for _ in trade_columns)})",
                    list(trade_row.values()),
                )
        return True

    def status_rows(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                a.strategy_id,
                a.last_bar_timestamp,
                a.cash,
                a.btc,
                s.equity,
                s.gross_equity,
                s.return_pct,
                s.drawdown,
                a.trade_count,
                a.total_fees,
                a.total_spread,
                a.total_slippage,
                a.total_turnover,
                a.target_position
            FROM paper_accounts a
            LEFT JOIN paper_snapshots s
              ON s.strategy_id = a.strategy_id
             AND s.bar_timestamp = a.last_bar_timestamp
            ORDER BY a.strategy_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def reset(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM paper_trades")
            self.conn.execute("DELETE FROM paper_snapshots")
            self.conn.execute("DELETE FROM paper_accounts")


class SupabaseOutbox:
    """Append-before-upload outbox so telemetry failure cannot lose local state."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.key = os.environ.get("SUPABASE_SECRET_KEY", "")
        self.snapshot_table = os.environ.get(
            "SUPABASE_PAPER_SNAPSHOT_TABLE",
            "paper_portfolio_snapshots",
        )
        self.trade_table = os.environ.get(
            "SUPABASE_PAPER_TRADES_TABLE",
            "paper_trades",
        )

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key)

    def enqueue(self, table: str, on_conflict: str, payload: dict[str, Any]) -> None:
        entry = {
            "table": table,
            "on_conflict": on_conflict,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, allow_nan=False) + "\n")

    def enqueue_snapshot(self, snapshot: dict[str, Any]) -> None:
        payload = {
            key: (int(value) if isinstance(value, bool) else value)
            for key, value in snapshot.items()
        }
        self.enqueue(
            self.snapshot_table,
            "strategy_id,bar_timestamp",
            payload,
        )

    def enqueue_trade(self, trade: SimulatedTrade) -> None:
        payload = asdict(trade)
        payload["created_at"] = utc_now_iso()
        self.enqueue(
            self.trade_table,
            "strategy_id,bar_timestamp,side",
            payload,
        )

    def _post(self, entry: dict[str, Any]) -> None:
        query = urllib.parse.urlencode({"on_conflict": entry["on_conflict"]})
        endpoint = f"{self.url}/rest/v1/{entry['table']}?{query}"
        body = json.dumps(entry["payload"], allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Supabase returned HTTP {response.status}")

    def flush(self) -> tuple[int, int]:
        if not self.path.exists():
            return 0, 0
        entries = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not entries:
            self.path.write_text("", encoding="utf-8")
            return 0, 0
        if not self.configured:
            return 0, len(entries)

        uploaded = 0
        pending: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            try:
                self._post(entry)
                uploaded += 1
            except Exception as exc:
                print(f"Supabase upload deferred: {type(exc).__name__}: {exc}")
                pending.extend(entries[index:])
                break

        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            "".join(
                json.dumps(entry, allow_nan=False) + "\n"
                for entry in pending
            ),
            encoding="utf-8",
        )
        temp.replace(self.path)
        return uploaded, len(pending)


def _strategy_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    strategies = list(cfg.get("paper_strategies", []))
    if not strategies:
        raise ValueError("paper_strategies must contain at least one strategy.")
    ids = [str(item["id"]) for item in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError("paper_strategies contains duplicate strategy IDs.")
    return strategies


def _quote_for_historical_bar(
    feature_frame: pd.DataFrame,
    index: int,
    lab_cfg: dict[str, Any],
) -> MarketQuote:
    """Use the next candle open with an explicit historical spread assumption."""
    next_row = feature_frame.iloc[index + 1]
    price = float(next_row["open"])
    spread_rate = float(lab_cfg.get("assumed_spread_bps_per_side", 0.0)) / 10_000.0
    return MarketQuote(
        timestamp=pd.Timestamp(next_row["timestamp"]).isoformat(),
        mark=price,
        bid=price * (1.0 - spread_rate),
        ask=price * (1.0 + spread_rate),
    )


def run_step(config_path: str) -> dict[str, Any]:
    cfg = load_paper_lab_config(config_path)
    lab_cfg = cfg["paper_lab"]
    feature_cfg = cfg.get("features_5m", {})
    strategies = _strategy_configs(cfg)

    db_path = Path(str(lab_cfg["state_db_path"]))
    lock_path = Path(str(lab_cfg["lock_path"]))
    outbox = SupabaseOutbox(Path(str(lab_cfg["outbox_path"])))

    with process_lock(lock_path):
        uploaded_before, _ = outbox.flush()
        bars = fetch_completed_bars(cfg)
        feature_frame = add_candle_features(bars, feature_cfg)
        live_quote = fetch_public_quote(
            exchange_id=str(cfg["market"]["exchange"]),
            symbol=str(cfg["market"]["symbol"]),
        )

        store = PaperStore(db_path)
        accounts: dict[str, AccountState] = {}
        newly_created: dict[str, bool] = {}
        try:
            for strategy_cfg in strategies:
                strategy_id = str(strategy_cfg["id"])
                account, created = store.get_or_create_account(
                    strategy_id,
                    initial_cash=float(lab_cfg["initial_cash"]),
                )
                accounts[strategy_id] = account
                newly_created[strategy_id] = created

            # A brand-new experiment begins at the same latest completed candle for
            # every portfolio. Existing experiments catch up every missed candle.
            latest_index = len(feature_frame) - 1
            earliest_available = pd.Timestamp(feature_frame.iloc[0]["timestamp"])
            candle_step = timeframe_to_timedelta(str(cfg["market"]["timeframe"]))
            indices_by_strategy: dict[str, list[int]] = {}
            for strategy_id, account in accounts.items():
                if newly_created[strategy_id] or account.last_bar_timestamp is None:
                    indices_by_strategy[strategy_id] = [latest_index]
                    continue
                last_ts = pd.Timestamp(account.last_bar_timestamp)
                if last_ts < earliest_available - candle_step:
                    raise RuntimeError(
                        f"{strategy_id} last processed {last_ts.isoformat()}, but the "
                        f"current lookback begins at {earliest_available.isoformat()}. "
                        "Increase paper_lab.lookback_bars before restarting so no "
                        "paper candles are silently skipped."
                    )
                indices_by_strategy[strategy_id] = [
                    index
                    for index, value in enumerate(feature_frame["timestamp"])
                    if pd.Timestamp(value) > last_ts
                ]

            all_indices = sorted(
                {index for indices in indices_by_strategy.values() for index in indices}
            )
            if not all_indices:
                summary = {
                    "status": "already_processed",
                    "latest_bar_timestamp": pd.Timestamp(
                        feature_frame.iloc[-1]["timestamp"]
                    ).isoformat(),
                    "supabase_uploaded": uploaded_before,
                    "supabase_pending": outbox.flush()[1],
                    "strategies": [],
                }
                print(json.dumps(summary, indent=2, allow_nan=False))
                return summary

            processed: list[dict[str, Any]] = []
            new_trades = 0
            snapshot_cadence = max(
                1,
                int(lab_cfg.get("supabase_snapshot_every_bars", 3)),
            )

            for index in all_indices:
                row = feature_frame.iloc[index]
                features = row_to_features(row)
                quote = (
                    _quote_for_historical_bar(feature_frame, index, lab_cfg)
                    if index < latest_index
                    else live_quote
                )

                for strategy_cfg in strategies:
                    strategy_id = str(strategy_cfg["id"])
                    if index not in indices_by_strategy[strategy_id]:
                        continue
                    state = accounts[strategy_id]
                    decision = decide_target(
                        strategy_cfg,
                        features,
                        previous_target=state.target_position,
                        lab_cfg=lab_cfg,
                    )
                    result = execute_decision(
                        state=state,
                        decision=decision,
                        features=features,
                        quote=quote,
                        lab_cfg=lab_cfg,
                    )
                    if not store.persist_result(result):
                        continue
                    accounts[strategy_id] = result.state

                    should_upload_snapshot = (
                        result.state.bars_processed % snapshot_cadence == 0
                        or result.trade is not None
                        or newly_created[strategy_id]
                    )
                    if should_upload_snapshot:
                        outbox.enqueue_snapshot(result.snapshot)
                    if result.trade is not None:
                        new_trades += 1
                        outbox.enqueue_trade(result.trade)

                    processed.append(
                        {
                            "strategy_id": strategy_id,
                            "bar_timestamp": features.bar_timestamp,
                            "signal": decision.signal,
                            "reason": decision.reason,
                            "target": decision.target_position,
                            "equity": result.snapshot["equity"],
                            "gross_equity": result.snapshot["gross_equity"],
                            "return_pct": result.snapshot["return_pct"],
                            "trade_count": result.state.trade_count,
                        }
                    )
        finally:
            store.close()

        uploaded_after, pending_after = outbox.flush()
        summary = {
            "status": "ok",
            "latest_bar_timestamp": pd.Timestamp(
                feature_frame.iloc[-1]["timestamp"]
            ).isoformat(),
            "quote_timestamp": live_quote.timestamp,
            "mark_price": live_quote.mark,
            "new_trades": new_trades,
            "supabase_uploaded": uploaded_before + uploaded_after,
            "supabase_pending": pending_after,
            "processed_strategy_bars": len(processed),
            "strategies": processed,
        }
        print(json.dumps(summary, indent=2, allow_nan=False))
        return summary


def print_status(config_path: str) -> list[dict[str, Any]]:
    cfg = load_paper_lab_config(config_path)
    store = PaperStore(Path(str(cfg["paper_lab"]["state_db_path"])))
    try:
        rows = store.status_rows()
    finally:
        store.close()

    if not rows:
        print("No paper accounts exist yet. Run one step first.")
        return []

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    return rows


def continuation_statistics(
    feature_frame: pd.DataFrame,
    max_streak: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unconditional = feature_frame["next_candle_return_bps"].dropna()
    rows.append(
        {
            "direction": "unconditional",
            "minimum_streak_length": 0,
            "observations": int(len(unconditional)),
            "next_candle_up_probability": float((unconditional > 0).mean()),
            "continuation_probability": float("nan"),
            "mean_next_return_bps": float(unconditional.mean()),
            "median_next_return_bps": float(unconditional.median()),
            "next_return_std_bps": float(unconditional.std(ddof=1)),
        }
    )
    for direction, name in ((1, "up"), (-1, "down")):
        for streak in range(1, max_streak + 1):
            subset = feature_frame[
                (feature_frame["streak_direction"] == direction)
                & (feature_frame["streak_length"] >= streak)
            ]["next_candle_return_bps"].dropna()
            if subset.empty:
                continue
            continuation = (subset > 0).mean() if direction > 0 else (subset < 0).mean()
            rows.append(
                {
                    "direction": name,
                    "minimum_streak_length": streak,
                    "observations": int(len(subset)),
                    "next_candle_up_probability": float((subset > 0).mean()),
                    "continuation_probability": float(continuation),
                    "mean_next_return_bps": float(subset.mean()),
                    "median_next_return_bps": float(subset.median()),
                    "next_return_std_bps": float(subset.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def horizon_statistics(
    feature_frame: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 6, 12, 36, 72),
    max_streak: int = 4,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    """Measure whether candle runs predict returns over longer holding horizons."""
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("Research horizons must be positive")
        future_return = (
            feature_frame["close"].shift(-horizon) / feature_frame["close"] - 1.0
        ) * 10_000.0
        for direction, name in ((1, "up"), (-1, "down")):
            for streak in range(1, max_streak + 1):
                mask = (
                    (feature_frame["streak_direction"] == direction)
                    & (feature_frame["streak_length"] >= streak)
                )
                subset = future_return[mask].dropna()
                if subset.empty:
                    continue
                continuation = (
                    (subset > 0).mean() if direction > 0 else (subset < 0).mean()
                )
                rows.append(
                    {
                        "direction": name,
                        "minimum_streak_length": streak,
                        "horizon_bars": horizon,
                        "horizon_minutes": horizon * timeframe_minutes,
                        "observations": int(len(subset)),
                        "continuation_probability": float(continuation),
                        "mean_future_return_bps": float(subset.mean()),
                        "median_future_return_bps": float(subset.median()),
                        "future_return_std_bps": float(subset.std(ddof=1)),
                    }
                )
    return pd.DataFrame(rows)


def run_research(
    config_path: str,
    bars_requested: int,
    output_dir: str,
) -> dict[str, Any]:
    cfg = load_paper_lab_config(config_path)
    lab_cfg = cfg["paper_lab"]
    feature_cfg = cfg.get("features_5m", {})
    market = cfg["market"]
    timeframe = str(market["timeframe"])

    normalized = fetch_completed_bars(cfg, max_bars=bars_requested)
    feature_frame = add_candle_features(normalized, feature_cfg)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    continuation_statistics(feature_frame).to_csv(
        output / "conditional_continuation.csv",
        index=False,
    )
    research_horizons = feature_cfg.get(
        "research_horizon_bars",
        [1, 3, 6, 12, 36, 72],
    )
    timeframe_minutes = int(timeframe_to_timedelta(timeframe).total_seconds() // 60)
    horizon_statistics(
        feature_frame,
        horizons=research_horizons,
        timeframe_minutes=timeframe_minutes,
    ).to_csv(
        output / "conditional_horizon_returns.csv",
        index=False,
    )

    strategies = _strategy_configs(cfg)
    initial_cash = float(lab_cfg["initial_cash"])
    states = {
        str(item["id"]): AccountState(
            strategy_id=str(item["id"]),
            initial_cash=initial_cash,
            cash=initial_cash,
            btc=0.0,
            gross_cash=initial_cash,
            gross_btc=0.0,
            peak_equity=initial_cash,
        )
        for item in strategies
    }

    min_history = max(
        int(feature_cfg.get("volume_window_bars", 48)),
        int(feature_cfg.get("broader_trend_ema_bars", 48)),
        int(feature_cfg.get("regime_slow_ema_bars", 72)),
        int(feature_cfg.get("momentum_bars", 12)) + 1,
    )
    snapshots: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    # Signal on completed candle i and execute at candle i+1 open. This preserves
    # causality and avoids same-candle lookahead.
    for index in range(min_history, len(feature_frame) - 1):
        row = feature_frame.iloc[index]
        if pd.isna(row["relative_volume"]):
            continue
        features = row_to_features(row)
        quote = _quote_for_historical_bar(feature_frame, index, lab_cfg)

        for strategy_cfg in strategies:
            strategy_id = str(strategy_cfg["id"])
            state = states[strategy_id]
            decision = decide_target(
                strategy_cfg,
                features,
                previous_target=state.target_position,
                lab_cfg=lab_cfg,
            )
            result = execute_decision(
                state=state,
                decision=decision,
                features=features,
                quote=quote,
                lab_cfg=lab_cfg,
            )
            states[strategy_id] = result.state
            snapshots.append(result.snapshot)
            if result.trade is not None:
                trades.append(asdict(result.trade))

    snapshot_frame = pd.DataFrame(snapshots)
    trade_frame = pd.DataFrame(trades)
    snapshot_frame.to_csv(output / "paper_research_equity.csv", index=False)
    trade_frame.to_csv(output / "paper_research_trades.csv", index=False)

    first_bar = pd.Timestamp(normalized["timestamp"].iloc[0])
    last_bar = pd.Timestamp(normalized["timestamp"].iloc[-1])
    sample_days = max((last_bar - first_bar).total_seconds() / 86_400.0, 1e-9)
    summary: dict[str, Any] = {
        "timeframe": timeframe,
        "bars": int(len(normalized)),
        "first_bar": first_bar.isoformat(),
        "last_bar": last_bar.isoformat(),
        "sample_days": sample_days,
        "cost_assumptions": {
            "fee_bps_per_side": float(lab_cfg.get("fee_bps_per_side", 60.0)),
            "slippage_bps_per_side": float(
                lab_cfg.get("slippage_bps_per_side", 5.0)
            ),
            "assumed_spread_bps_per_side": float(
                lab_cfg.get("assumed_spread_bps_per_side", 0.0)
            ),
            "round_trip_cost_bps": round_trip_cost_bps(lab_cfg),
        },
        "strategies": {},
    }
    if not snapshot_frame.empty:
        for strategy_id, group in snapshot_frame.groupby("strategy_id"):
            last = group.iloc[-1]
            gross_profit = float(last["gross_equity"]) - initial_cash
            total_turnover = float(last["total_turnover"])
            break_even_bps = (
                max(0.0, gross_profit) / total_turnover * 10_000.0
                if total_turnover > 0.0
                else 0.0
            )
            actual_all_in_bps = (
                float(lab_cfg.get("fee_bps_per_side", 60.0))
                + float(lab_cfg.get("slippage_bps_per_side", 5.0))
                + float(lab_cfg.get("assumed_spread_bps_per_side", 0.0))
            )
            summary["strategies"][strategy_id] = {
                "ending_equity": float(last["equity"]),
                "ending_gross_equity": float(last["gross_equity"]),
                "net_return_pct": float(last["return_pct"]),
                "gross_return_pct": float(last["gross_return_pct"]),
                "max_drawdown": float(group["drawdown"].min()),
                "trade_count": int(last["trade_count"]),
                "trades_per_day": float(last["trade_count"]) / sample_days,
                "total_fees": float(last["total_fees"]),
                "total_spread": float(last["total_spread"]),
                "total_slippage": float(last["total_slippage"]),
                "total_turnover": total_turnover,
                "turnover_multiple": total_turnover / initial_cash,
                "cost_drag": float(last["cost_drag"]),
                "gross_break_even_all_in_bps_per_side": break_even_bps,
                "assumed_all_in_bps_per_side": actual_all_in_bps,
                "cost_to_break_even_multiple": (
                    actual_all_in_bps / break_even_bps
                    if break_even_bps > 0.0
                    else None
                ),
            }

    (output / "paper_research_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def reset_lab(config_path: str, confirmed: bool) -> None:
    if not confirmed:
        raise SystemExit("Refusing to reset without --yes.")
    cfg = load_paper_lab_config(config_path)
    store = PaperStore(Path(str(cfg["paper_lab"]["state_db_path"])))
    try:
        store.reset()
    finally:
        store.close()
    outbox = Path(str(cfg["paper_lab"]["outbox_path"]))
    if outbox.exists():
        outbox.unlink()
    print("Five-minute paper lab reset.")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Independent five-minute BTC paper-trading laboratory"
    )
    parser.add_argument(
        "--config",
        default="config/settings_paper_5m.yaml",
        help="Paper-lab YAML configuration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("step", help="Process all new completed five-minute candles")
    subparsers.add_parser("status", help="Show local simulated portfolio status")

    research = subparsers.add_parser(
        "research",
        help="Download recent candles and run a cost-aware historical simulation",
    )
    research.add_argument("--bars", type=int, default=10_000)
    research.add_argument("--output", default="outputs/paper_5m_research")

    reset = subparsers.add_parser("reset", help="Delete all local five-minute paper state")
    reset.add_argument("--yes", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "step":
        run_step(args.config)
    elif args.command == "status":
        print_status(args.config)
    elif args.command == "research":
        run_research(
            args.config,
            bars_requested=args.bars,
            output_dir=args.output,
        )
    elif args.command == "reset":
        reset_lab(args.config, confirmed=args.yes)


if __name__ == "__main__":
    main()
