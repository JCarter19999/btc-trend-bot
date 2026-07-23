from __future__ import annotations
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import run_equity_paper_step as paper_step
import run_equity_real_data_walkforward as walkforward


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _strategy_config(**overrides):
    base = walkforward.BacktestConfig(
        symbols=("WIN", "LOSE"), benchmark="SPY", start="2024-01-01", end=None, interval="1d",
        initial_capital=2500.0, return_threshold_bps=0.0, max_hold_bars=5, train_bars=50, test_bars=20,
        step_bars=20, purge_bars=5, ridge_alpha=10.0, stop_atr=1.35, target_atr=2.15,
        slippage_bps_each_side=0.0, safety_enabled=True, drawdown_pause=0.15, hard_shutdown_drawdown=0.35,
        consecutive_loss_limit=4, cooldown_trades=8, minimum_equity=25.0, position_fraction=0.25,
    )
    return dataclasses.replace(base, **overrides)


def _series(n, seed, drift):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    ret = rng.normal(drift, 0.004, n)
    close = 100 * np.exp(np.cumsum(ret))
    op = np.r_[close[0], close[:-1]]
    high = np.maximum(op, close) * 1.002
    low = np.minimum(op, close) * 0.998
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close,
                          "volume": rng.integers(2_000_000, 5_000_000, n)}, index=idx)


def _trend_frames(n=90, benchmark_drift=0.0015, winner_drift=0.006, loser_drift=-0.004, seed=7):
    """SPY trends up (so market_above_ema50=1 by the end); WIN clearly outpaces
    it (highest relative_strength_20); LOSE clearly lags -- a deterministic
    simple_trend winner."""
    return {
        "SPY": _series(n, seed, benchmark_drift),
        "WIN": _series(n, seed + 1, winner_drift),
        "LOSE": _series(n, seed + 2, loser_drift),
    }


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.text = "" if status_code == 200 else "error"
        self._payload = payload

    def json(self):
        return self._payload


class _StepFakeClient:
    """Test double whose .frames can be swapped/trimmed between run_step() calls
    to simulate new trading days becoming available over time."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def get_price_history_every_day(self, symbol, *, start_datetime=None, end_datetime=None,
                                     need_extended_hours_data=None, need_previous_close=None):
        frame = self.frames[symbol]
        candles = [
            {"datetime": int(idx.timestamp() * 1000), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close), "volume": float(r.volume)}
            for idx, r in frame.iterrows()
        ]
        return _FakeResponse(200, {"candles": candles})


def _write_deployment_configs(tmp_path, cfg, selection="simple_trend", lookback_days=1000, data_source="schwab"):
    strategy_path = tmp_path / "strategy.yaml"
    strategy_yaml = {
        "symbols": list(cfg.symbols), "benchmark": cfg.benchmark, "start": cfg.start, "end": cfg.end,
        "interval": cfg.interval, "initial_capital": cfg.initial_capital, "position_fraction": cfg.position_fraction,
        "return_threshold_bps": cfg.return_threshold_bps, "max_hold_bars": cfg.max_hold_bars,
        "train_bars": cfg.train_bars, "test_bars": cfg.test_bars, "step_bars": cfg.step_bars,
        "purge_bars": cfg.purge_bars, "ridge_alpha": cfg.ridge_alpha, "stop_atr": cfg.stop_atr,
        "target_atr": cfg.target_atr, "slippage_bps_each_side": cfg.slippage_bps_each_side,
        "safety": {"enabled": cfg.safety_enabled, "drawdown_pause": cfg.drawdown_pause,
                   "hard_shutdown_drawdown": cfg.hard_shutdown_drawdown,
                   "consecutive_loss_limit": cfg.consecutive_loss_limit, "cooldown_trades": cfg.cooldown_trades,
                   "minimum_equity": cfg.minimum_equity},
    }
    strategy_path.write_text(yaml.safe_dump(strategy_yaml))
    deploy_path = tmp_path / "deploy.yaml"
    deploy_yaml = {
        "strategy_config": str(strategy_path),
        "data_source": data_source,
        "schwab": {"token_path": str(tmp_path / "token.json"), "lookback_days": lookback_days},
        "paper": {"ledger_db_path": str(tmp_path / "ledger.sqlite3"), "lock_path": str(tmp_path / "lock"),
                  "selection": selection},
    }
    deploy_path.write_text(yaml.safe_dump(deploy_yaml))
    return deploy_path


def _dummy_deploy_cfg(tmp_path, selection="simple_trend", data_source="schwab"):
    return paper_step.PaperDeploymentConfig(
        strategy_config_path=tmp_path / "unused.yaml", data_source=data_source,
        schwab_token_path=tmp_path / "unused_token.json",
        lookback_days=1000, ledger_db_path=tmp_path / "ledger.sqlite3", lock_path=tmp_path / "lock",
        selection=selection)


# --------------------------------------------------------------------------- #
# _train_date_window regression guard
# --------------------------------------------------------------------------- #

def test_train_date_window_matches_walk_forward_purge_and_cap():
    prior_dates = list(range(1000))
    window = paper_step._train_date_window(prior_dates, train_bars=756, purge_bars=10)
    assert window == prior_dates[1000 - 756:1000 - 10]
    assert len(window) == 756 - 10

def test_train_date_window_handles_short_history():
    prior_dates = list(range(5))
    assert paper_step._train_date_window(prior_dates, train_bars=756, purge_bars=10) == []

def test_train_date_window_handles_purge_bars_zero():
    prior_dates = list(range(100))
    window = paper_step._train_date_window(prior_dates, train_bars=50, purge_bars=0)
    assert window == prior_dates[50:100]


# --------------------------------------------------------------------------- #
# Position mechanics (direct unit tests, no ledger/Schwab involved)
# --------------------------------------------------------------------------- #

def test_stop_hit_exit():
    cfg = _strategy_config()
    pending = paper_step.PendingEntry(symbol="AAPL", signal_time=pd.Timestamp("2024-01-01", tz="UTC"), atr=2.0)
    entry_bar = pd.Series({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0})
    position = paper_step._fill_pending_entry(pending, entry_bar, pd.Timestamp("2024-01-02", tz="UTC"), cfg)
    day2 = pd.Series({"open": 100.0, "high": 100.1, "low": position.stop_price - 0.5, "close": 99.0})
    still_open, trade = paper_step._evaluate_open_position_bar(position, day2, pd.Timestamp("2024-01-03", tz="UTC"), cfg)
    assert still_open is None
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == pytest.approx(position.stop_price)

def test_target_hit_exit():
    cfg = _strategy_config()
    pending = paper_step.PendingEntry(symbol="AAPL", signal_time=pd.Timestamp("2024-01-01", tz="UTC"), atr=2.0)
    entry_bar = pd.Series({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0})
    position = paper_step._fill_pending_entry(pending, entry_bar, pd.Timestamp("2024-01-02", tz="UTC"), cfg)
    day2 = pd.Series({"open": 100.0, "high": position.target_price + 0.5, "low": 99.9, "close": 105.0})
    still_open, trade = paper_step._evaluate_open_position_bar(position, day2, pd.Timestamp("2024-01-03", tz="UTC"), cfg)
    assert still_open is None
    assert trade["exit_reason"] == "target"
    assert trade["exit_price"] == pytest.approx(position.target_price)

def test_ambiguous_same_day_tie_break_assumes_stop():
    cfg = _strategy_config()
    pending = paper_step.PendingEntry(symbol="AAPL", signal_time=pd.Timestamp("2024-01-01", tz="UTC"), atr=2.0)
    entry_bar = pd.Series({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0})
    position = paper_step._fill_pending_entry(pending, entry_bar, pd.Timestamp("2024-01-02", tz="UTC"), cfg)
    day2 = pd.Series({"open": 100.0, "high": position.target_price + 1.0, "low": position.stop_price - 1.0, "close": 100.0})
    still_open, trade = paper_step._evaluate_open_position_bar(position, day2, pd.Timestamp("2024-01-03", tz="UTC"), cfg)
    assert still_open is None
    assert trade["exit_reason"] == "ambiguous_stop_first"
    assert trade["exit_price"] == pytest.approx(position.stop_price)

def test_time_exit_at_max_hold_bars():
    cfg = _strategy_config(max_hold_bars=2)
    pending = paper_step.PendingEntry(symbol="AAPL", signal_time=pd.Timestamp("2024-01-01", tz="UTC"), atr=2.0)
    entry_bar = pd.Series({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0})
    position = paper_step._fill_pending_entry(pending, entry_bar, pd.Timestamp("2024-01-02", tz="UTC"), cfg)
    # bars_held becomes 1 on this call (below max_hold_bars=2), stays open
    still_open, trade = paper_step._evaluate_open_position_bar(
        position, pd.Series({"open": 100.0, "high": 100.3, "low": 99.7, "close": 100.1}),
        pd.Timestamp("2024-01-03", tz="UTC"), cfg)
    assert trade is None and still_open is not None
    # bars_held becomes 2 == max_hold_bars -> forced close at this bar's close
    day3 = pd.Series({"open": 100.1, "high": 100.4, "low": 99.9, "close": 101.2})
    still_open, trade = paper_step._evaluate_open_position_bar(still_open, day3, pd.Timestamp("2024-01-04", tz="UTC"), cfg)
    assert still_open is None
    assert trade["exit_reason"] == "time_exit"
    assert trade["exit_price"] == pytest.approx(101.2)


# --------------------------------------------------------------------------- #
# _process_one_day: pending-entry same-day fill+evaluate, catch-up ordering guard
# --------------------------------------------------------------------------- #

def test_pending_entry_fills_and_checks_same_day_range(tmp_path):
    cfg = _strategy_config()
    ledger = paper_step.PaperLedger(tmp_path / "ledger.sqlite3")
    deploy_cfg = _dummy_deploy_cfg(tmp_path)
    signal_time = pd.Timestamp("2024-01-01", tz="UTC")
    today = pd.Timestamp("2024-01-02", tz="UTC")
    ledger.upsert_open_position(status="pending_entry", symbol="AAPL", signal_time=str(signal_time),
                                 atr=2.0, predicted_return=0.01, entry_time=None, entry_price=None,
                                 stop_price=None, target_price=None, bars_held=0)
    frames = {"AAPL": pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1_000_000]},
        index=pd.DatetimeIndex([today]))}
    action = paper_step._process_one_day(ledger, frames, cfg, deploy_cfg, today, allow_new_entry=False)
    assert action["action"] == "hold"
    position = ledger.get_open_position()
    assert position["status"] == "open"
    assert position["bars_held"] == 1
    ledger.close()

def test_process_one_day_skips_new_entry_when_not_latest(tmp_path):
    cfg = _strategy_config()
    ledger = paper_step.PaperLedger(tmp_path / "ledger.sqlite3")
    deploy_cfg = _dummy_deploy_cfg(tmp_path)
    frames = _trend_frames()
    today = frames["SPY"].index[-1]
    action = paper_step._process_one_day(ledger, frames, cfg, deploy_cfg, today, allow_new_entry=False)
    assert action["action"] == "skip_not_latest"
    assert ledger.get_open_position() is None
    ledger.close()


# --------------------------------------------------------------------------- #
# Safety-pause vs manual-halt asymmetry
# --------------------------------------------------------------------------- #

def test_safety_pause_blocks_new_entry_but_still_manages_open_position(tmp_path):
    cfg = _strategy_config(drawdown_pause=0.10, position_fraction=1.0, consecutive_loss_limit=99)
    ledger = paper_step.PaperLedger(tmp_path / "ledger.sqlite3")
    deploy_cfg = _dummy_deploy_cfg(tmp_path)
    ledger.append_completed_trade({
        "signal_time": pd.Timestamp("2024-01-01", tz="UTC"), "symbol": "WIN",
        "entry_time": pd.Timestamp("2024-01-02", tz="UTC"), "entry_price": 100.0,
        "exit_time": pd.Timestamp("2024-01-03", tz="UTC"), "exit_price": 80.0,
        "net_return": -0.20, "exit_reason": "stop", "bars_held": 1,
    }, predicted_return=0.02)  # equity 2500->2000, dd=0.20 >= drawdown_pause 0.10

    frames = _trend_frames()
    today = frames["SPY"].index[-1]

    # Phase 1: a pre-existing open position must still be managed despite the pause.
    frames_p1 = dict(frames)
    frames_p1["LOSE"] = frames["LOSE"].copy()
    frames_p1["LOSE"].loc[today, ["open", "high", "low", "close"]] = [99.0, 99.5, 95.0, 96.0]
    ledger.upsert_open_position(status="open", symbol="LOSE", signal_time=str(today), atr=2.0,
                                 predicted_return=0.01, entry_time=str(today), entry_price=100.0,
                                 stop_price=90.0, target_price=110.0, bars_held=1)
    action1 = paper_step._process_one_day(ledger, frames_p1, cfg, deploy_cfg, today, allow_new_entry=True)
    assert action1["action"] == "hold"
    assert ledger.get_open_position() is not None

    # Phase 2: now flat -- the pause must block a brand-new entry.
    ledger.clear_open_position()
    allowed, reason = paper_step._safety_allows_new_entry(ledger.completed_trades_frame(), today, cfg)
    assert not allowed
    assert reason == "drawdown_pause"
    action2 = paper_step._process_one_day(ledger, frames, cfg, deploy_cfg, today, allow_new_entry=True)
    assert action2["action"] == "no_entry_safety_paused"
    assert ledger.get_open_position() is None
    ledger.close()


# --------------------------------------------------------------------------- #
# run_step end-to-end: idempotency, catch-up, signal, halt/resume
# --------------------------------------------------------------------------- #

def test_duplicate_day_is_noop(tmp_path):
    cfg = _strategy_config()
    deploy_path = _write_deployment_configs(tmp_path, cfg)
    client = _StepFakeClient(_trend_frames())
    first = paper_step.run_step(deploy_path, client=client)
    assert first["status"] == "ok"
    second = paper_step.run_step(deploy_path, client=client)
    assert second["status"] == "already_processed"

def test_flat_safety_ok_signal_fires_opens_pending_entry(tmp_path):
    cfg = _strategy_config()
    deploy_path = _write_deployment_configs(tmp_path, cfg)
    client = _StepFakeClient(_trend_frames())
    result = paper_step.run_step(deploy_path, client=client)
    assert result["status"] == "ok"
    action = result["actions"][-1]
    assert action["action"] == "signal"
    assert action["symbol"] == "WIN"
    status = paper_step.get_status(deploy_path)
    assert status["open_position"]["status"] == "pending_entry"
    assert status["open_position"]["symbol"] == "WIN"

def test_catch_up_multiple_missed_days(tmp_path):
    cfg = _strategy_config()
    full = _trend_frames(n=95)
    deploy_path = _write_deployment_configs(tmp_path, cfg)
    client = _StepFakeClient({s: f.iloc[:-3] for s, f in full.items()})
    first = paper_step.run_step(deploy_path, client=client)
    assert first["status"] == "ok"
    client.frames = full
    second = paper_step.run_step(deploy_path, client=client)
    assert second["status"] == "ok"
    assert len(second["processed_dates"]) == 3
    status = paper_step.get_status(deploy_path)
    assert status["last_processed_bar_date"] == str(full["SPY"].index[-1])

def test_manual_halt_blocks_entire_step(tmp_path):
    cfg = _strategy_config()
    deploy_path = _write_deployment_configs(tmp_path, cfg)
    client = _StepFakeClient(_trend_frames())
    first = paper_step.run_step(deploy_path, client=client)
    assert first["status"] == "ok"
    paper_step.set_halt(deploy_path, True, "testing")
    before = paper_step.get_status(deploy_path)
    result = paper_step.run_step(deploy_path, client=client)
    assert result["status"] == "halted"
    after = paper_step.get_status(deploy_path)
    assert after["open_position"] == before["open_position"]
    paper_step.set_halt(deploy_path, False)
    resumed = paper_step.run_step(deploy_path, client=client)
    assert resumed["status"] == "already_processed"


# --------------------------------------------------------------------------- #
# Ledger basics
# --------------------------------------------------------------------------- #

def test_ledger_open_position_upsert_and_clear(tmp_path):
    ledger = paper_step.PaperLedger(tmp_path / "ledger.sqlite3")
    assert ledger.get_open_position() is None
    ledger.upsert_open_position(status="pending_entry", symbol="AAPL", signal_time="2024-01-01",
                                 atr=1.0, predicted_return=None, entry_time=None, entry_price=None,
                                 stop_price=None, target_price=None, bars_held=0)
    assert ledger.get_open_position()["symbol"] == "AAPL"
    ledger.upsert_open_position(status="open", symbol="AAPL", signal_time="2024-01-01", atr=1.0,
                                 predicted_return=None, entry_time="2024-01-02", entry_price=100.0,
                                 stop_price=95.0, target_price=110.0, bars_held=1)
    assert ledger.get_open_position()["status"] == "open"
    ledger.clear_open_position()
    assert ledger.get_open_position() is None
    ledger.close()

def test_ledger_halt_state_roundtrip(tmp_path):
    ledger = paper_step.PaperLedger(tmp_path / "ledger.sqlite3")
    assert not ledger.is_halted()
    ledger.set_halt(True, "test reason")
    assert ledger.is_halted()
    assert ledger.halt_reason() == "test reason"
    ledger.set_halt(False)
    assert not ledger.is_halted()
    ledger.close()


# --------------------------------------------------------------------------- #
# Supabase run-report building and upload (best-effort telemetry)
# --------------------------------------------------------------------------- #

def test_build_run_report_flat_no_trades(tmp_path):
    cfg = _strategy_config()
    ledger = paper_step.PaperLedger(tmp_path / "ledger.sqlite3")
    deploy_cfg = _dummy_deploy_cfg(tmp_path)
    report = paper_step._build_run_report(
        deploy_cfg, cfg, ledger, "ok", "processed 1 day(s)", pd.Timestamp("2024-01-05", tz="UTC"),
        "2024-01-05T00:00:00+00:00", "2024-01-05T00:00:05+00:00")
    assert report["status"] == "ok"
    assert report["data_source"] == deploy_cfg.data_source
    assert report["completed_trades"] == 0
    assert report["ending_equity"] == pytest.approx(cfg.initial_capital)
    assert report["total_return"] == pytest.approx(0.0)
    assert report["open_position_symbol"] is None
    assert report["duration_seconds"] == pytest.approx(5.0)
    ledger.close()

def test_build_run_report_reflects_completed_trade_and_open_position(tmp_path):
    cfg = _strategy_config()
    ledger = paper_step.PaperLedger(tmp_path / "ledger.sqlite3")
    deploy_cfg = _dummy_deploy_cfg(tmp_path)
    ledger.append_completed_trade({
        "signal_time": pd.Timestamp("2024-01-01", tz="UTC"), "symbol": "WIN",
        "entry_time": pd.Timestamp("2024-01-02", tz="UTC"), "entry_price": 100.0,
        "exit_time": pd.Timestamp("2024-01-03", tz="UTC"), "exit_price": 110.0,
        "net_return": 0.10, "exit_reason": "target", "bars_held": 1,
    }, predicted_return=0.02)
    ledger.upsert_open_position(status="pending_entry", symbol="LOSE", signal_time="2024-01-04",
                                 atr=2.0, predicted_return=-0.01, entry_time=None, entry_price=None,
                                 stop_price=None, target_price=None, bars_held=0)

    report = paper_step._build_run_report(
        deploy_cfg, cfg, ledger, "ok", "processed 1 day(s)", pd.Timestamp("2024-01-04", tz="UTC"),
        "2024-01-04T00:00:00+00:00", "2024-01-04T00:00:01+00:00")

    expected_equity = cfg.initial_capital * (1 + cfg.position_fraction * 0.10)
    assert report["completed_trades"] == 1
    assert report["ending_equity"] == pytest.approx(expected_equity)
    assert report["total_return"] == pytest.approx(expected_equity / cfg.initial_capital - 1)
    assert report["open_position_symbol"] == "LOSE"
    assert report["open_position_status"] == "pending_entry"
    assert report["open_position_predicted_return"] == pytest.approx(-0.01)
    ledger.close()

def test_report_run_queues_locally_when_supabase_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    cfg = _strategy_config()
    ledger = paper_step.PaperLedger(tmp_path / "ledger.sqlite3")
    deploy_cfg = paper_step.PaperDeploymentConfig(
        strategy_config_path=tmp_path / "unused.yaml", data_source="yfinance",
        schwab_token_path=tmp_path / "unused_token.json", lookback_days=1000,
        ledger_db_path=tmp_path / "ledger.sqlite3", lock_path=tmp_path / "lock",
        selection="simple_trend", supabase_table="equity_paper_runs",
        supabase_outbox_path=tmp_path / "outbox.jsonl")
    report = paper_step._build_run_report(
        deploy_cfg, cfg, ledger, "ok", "processed 1 day(s)", pd.Timestamp("2024-01-04", tz="UTC"),
        "2024-01-04T00:00:00+00:00", "2024-01-04T00:00:01+00:00")

    paper_step._report_run(deploy_cfg, report)  # must not raise despite missing Supabase creds

    queued = [line for line in (tmp_path / "outbox.jsonl").read_text().strip().splitlines() if line]
    assert len(queued) == 1
    entry = json.loads(queued[0])
    assert entry["table"] == "equity_paper_runs"
    assert entry["on_conflict"] == "run_id"
    assert entry["payload"]["status"] == "ok"
    ledger.close()

def test_run_step_end_to_end_writes_supabase_outbox(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    cfg = _strategy_config()
    deploy_path = _write_deployment_configs(tmp_path, cfg)
    client = _StepFakeClient(_trend_frames())
    result = paper_step.run_step(deploy_path, client=client)
    assert result["status"] == "ok"

    outbox_path = tmp_path / "ledger_supabase_outbox.jsonl"
    assert outbox_path.exists()
    entries = [json.loads(line) for line in outbox_path.read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["payload"]["data_source"] == "schwab"
    assert entries[0]["payload"]["status"] == "ok"
