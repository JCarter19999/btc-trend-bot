"""Tests for the two-tier safety-layer candidate's LIVE paper-trading state
machine (experiments/run_two_tier_safety_paper_step.py). apply_bar() is the
same per-bar decision/fill/trigger logic as
run_two_tier_safety_backtest.run_backtest_with_two_tier_safety, just called
once per process invocation with state persisted across calls instead of
looped in memory over a whole frame -- see that module's docstring.

These tests call apply_bar() directly with synthetic (price, raw_target)
sequences, bypassing the live download_ohlcv/prepare_strategy_frame pipeline
entirely (no network, no strategy-signal warmup needed). A separate test
covers run_two_tier_paper_step()'s idempotency by monkeypatching the data
fetch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import run_two_tier_safety_paper_step as live_step  # noqa: E402
from run_two_tier_safety_backtest import SafetyConfig  # noqa: E402


def inert_safety(**overrides) -> SafetyConfig:
    base = dict(drawdown_pause=0.99, hard_shutdown_drawdown=0.999,
                minimum_equity_fraction=0.0, consecutive_loss_limit_bars=10_000,
                cooldown_bars=1)
    base.update(overrides)
    return SafetyConfig(**base)


def fresh_state(cash: float = 10_000.0) -> live_step.TwoTierPaperState:
    return live_step.TwoTierPaperState(cash=cash, btc=0.0, peak_equity=cash)


def apply_sequence(state, prices, targets, safety, minimum_equity=0.0):
    """Feed a sequence of (price, raw_target) bars through apply_bar in
    order, returning the list of run_row dicts (one per bar)."""
    rows = []
    for price, target in zip(prices, targets):
        state, trade_row, run_row = live_step.apply_bar(
            state, timestamp=f"t{len(rows)}", price=price, raw_target=target,
            safety=safety, fee_rate=0.0, slippage_rate=0.0, minimum_equity=minimum_equity,
        )
        rows.append(run_row)
    return state, rows


# --------------------------------------------------------------------------- #
# Basic fill mechanics
# --------------------------------------------------------------------------- #

def test_first_bar_buys_to_the_target_fraction():
    state = fresh_state(cash=10_000.0)
    safety = inert_safety()
    state, trade_row, run_row = live_step.apply_bar(
        state, timestamp="t0", price=100.0, raw_target=1.0, safety=safety,
        fee_rate=0.0, slippage_rate=0.0, minimum_equity=0.0,
    )
    assert trade_row is not None
    assert trade_row["side"] == "buy"
    assert state.btc == pytest.approx(100.0, rel=1e-6)  # $10,000 / $100
    assert state.cash == pytest.approx(0.0, abs=1e-6)
    assert run_row["safety_reason"] == ""


def test_flat_target_produces_no_trade():
    state = fresh_state()
    safety = inert_safety()
    state, trade_row, run_row = live_step.apply_bar(
        state, timestamp="t0", price=100.0, raw_target=0.0, safety=safety,
        fee_rate=0.0, slippage_rate=0.0, minimum_equity=0.0,
    )
    assert trade_row is None
    assert state.btc == 0.0
    assert run_row["side"] == "none"


# --------------------------------------------------------------------------- #
# Hard shutdown: permanent, forces flat regardless of raw_target.
# --------------------------------------------------------------------------- #

def test_hard_shutdown_on_drawdown_forces_flat_and_persists_across_calls():
    safety = inert_safety(hard_shutdown_drawdown=0.10)
    state = fresh_state()
    state, rows = apply_sequence(state, [100.0, 80.0, 90.0], [1.0, 1.0, 1.0], safety)
    assert state.hard_halted is True
    assert rows[1]["safety_reason"] == ""  # the crash bar itself still traded pre-existing state
    assert rows[2]["safety_reason"] == "hard_shutdown"
    assert rows[2]["applied_target"] == 0.0
    # Even a strongly bullish raw_target on a later call stays locked out.
    state, trade_row, run_row = live_step.apply_bar(
        state, timestamp="t3", price=200.0, raw_target=1.0, safety=safety,
        fee_rate=0.0, slippage_rate=0.0, minimum_equity=0.0,
    )
    assert run_row["safety_reason"] == "hard_shutdown"
    assert run_row["applied_target"] == 0.0
    assert state.hard_halted is True


def test_hard_shutdown_on_minimum_equity_floor():
    safety = inert_safety(hard_shutdown_drawdown=0.999)  # unreachable via drawdown
    state = fresh_state(cash=1000.0)
    minimum_equity = 500.0
    # call1 (price 30) is the bar that CROSSES the floor -- it still traded
    # on pre-existing (not-yet-halted) state, per no-lookahead. call2 is the
    # first bar to actually see the halt take effect.
    state, rows = apply_sequence(
        state, [100.0, 30.0, 30.0], [1.0, 1.0, 1.0], safety, minimum_equity=minimum_equity)
    assert state.hard_halted is True
    assert "floor" in state.hard_halt_reason
    assert rows[1]["safety_reason"] == ""
    assert rows[2]["safety_reason"] == "hard_shutdown"
    assert rows[2]["applied_target"] == 0.0


# --------------------------------------------------------------------------- #
# Loss-streak cooldown
# --------------------------------------------------------------------------- #

def test_loss_streak_cooldown_forces_flat_then_recovers():
    safety = inert_safety(consecutive_loss_limit_bars=2, cooldown_bars=1)
    state = fresh_state()
    # call0 establishes the position at 100. calls 1,2 (99, then 98) are two
    # consecutive losing bars relative to the PREVIOUS call's ending equity
    # -- the streak hits its limit of 2 as of call2, which itself still
    # trades normally (no lookahead); call3 is the first forced-flat bar.
    state, rows = apply_sequence(
        state, [100.0, 99.0, 98.0, 120.0, 130.0], [1.0, 1.0, 1.0, 1.0, 1.0], safety)
    assert rows[2]["safety_reason"] == ""
    assert rows[3]["safety_reason"] == "loss_cooldown"
    assert rows[3]["applied_target"] == 0.0
    assert rows[4]["safety_reason"] == ""
    assert rows[4]["applied_target"] == 1.0
    assert state.hard_halted is False


def test_consecutive_losses_reset_on_a_winning_bar():
    safety = inert_safety(consecutive_loss_limit_bars=2, cooldown_bars=5)
    state = fresh_state()
    state, rows = apply_sequence(
        state, [100.0, 99.0, 105.0, 104.0], [1.0, 1.0, 1.0, 1.0], safety)
    # bar2 loses (streak=1), bar3 wins (streak resets to 0), bar4 loses (streak=1 again)
    # -- never reaches the limit of 2 in a row, so no cooldown should ever fire.
    assert all(r["safety_reason"] == "" for r in rows)
    assert state.loss_cooldown_remaining == 0


# --------------------------------------------------------------------------- #
# Drawdown pause: soft, recovers, re-anchors peak.
# --------------------------------------------------------------------------- #

def test_drawdown_pause_forces_flat_then_recovers_and_reanchors():
    safety = inert_safety(drawdown_pause=0.10, cooldown_bars=1)
    state = fresh_state()
    # call0 establishes the position at 100. call1 (price 85, -15% vs call0's
    # ending equity) still trades on pre-existing state -- no lookahead --
    # and the pause triggers for use starting call2.
    state, rows = apply_sequence(
        state, [100.0, 85.0, 85.0, 90.0], [1.0, 1.0, 1.0, 1.0], safety)
    assert rows[1]["safety_reason"] == ""
    # call2: forced flat, cooldown ends this call, peak re-anchors.
    assert rows[2]["safety_reason"] == "drawdown_pause"
    assert rows[2]["applied_target"] == 0.0
    assert rows[2]["drawdown"] == pytest.approx(0.0, abs=1e-9)
    # call3: trading resumed against the fresh peak.
    assert rows[3]["safety_reason"] == ""
    assert rows[3]["applied_target"] == 1.0


def test_idempotent_persistence_round_trip(tmp_path):
    # save_state/load_state must round-trip every field apply_bar depends on.
    state = fresh_state()
    safety = inert_safety(consecutive_loss_limit_bars=2, cooldown_bars=3)
    state, _ = apply_sequence(state, [100.0, 99.0], [1.0, 1.0], safety)
    path = tmp_path / "state.json"
    live_step.save_state(path, state)
    reloaded = live_step.load_state(path, initial_cash=10_000.0)
    assert reloaded == state


def test_run_two_tier_paper_step_is_idempotent_on_the_same_bar(tmp_path, monkeypatch):
    # Monkeypatch the network fetch + strategy-signal computation so this
    # exercises run_two_tier_paper_step's own idempotency check without any
    # network access or 252-bar feature warmup.
    import pandas as pd

    state_path = tmp_path / "state.json"
    trades_path = tmp_path / "trades.csv"
    runs_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(live_step, "STATE_PATH", state_path)
    monkeypatch.setattr(live_step, "TRADES_PATH", trades_path)
    monkeypatch.setattr(live_step, "RUNS_LOG_PATH", runs_path)

    fixed_frame = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "close": [100.0],
        "target_position": [1.0],
    })
    monkeypatch.setattr(live_step, "download_ohlcv", lambda **kwargs: fixed_frame)
    monkeypatch.setattr(live_step, "normalize_ohlcv", lambda frame, timeframe: (frame, None))
    monkeypatch.setattr(live_step, "prepare_strategy_frame", lambda normalized, cfg: normalized)
    monkeypatch.setattr(live_step, "load_config", lambda path: {
        "market": {"exchange": "coinbase", "symbol": "BTC/USD", "timeframe": "4h"},
        "paper": {"initial_cash": 500.0},
        "backtest": {"fee_bps_per_turnover": 0, "slippage_bps_per_turnover": 0},
    })

    first = live_step.run_two_tier_paper_step("unused.yaml")
    assert first.last_bar_timestamp is not None
    assert runs_path.exists()
    assert sum(1 for _ in runs_path.read_text().splitlines() if _.strip()) == 1

    second = live_step.run_two_tier_paper_step("unused.yaml")
    assert second == first
    # No new row appended for the already-processed bar.
    assert sum(1 for _ in runs_path.read_text().splitlines() if _.strip()) == 1
