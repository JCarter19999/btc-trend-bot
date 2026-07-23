"""Tests for the two-tier safety-layer candidate's per-bar state machine
(experiments/run_two_tier_safety_backtest.py). See TWO_TIER_SAFETY_LAYER.md.

Mirrors tests/test_backtest.py's make_frame helper -- a minimal synthetic
frame with just the columns run_backtest_with_two_tier_safety needs
(timestamp/close/simple_return/target_position), no strategy-signal warmup
required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_two_tier_safety_backtest import SafetyConfig, run_backtest_with_two_tier_safety  # noqa: E402
from btc_trend_bot.backtest import run_backtest  # noqa: E402


BASE_CFG = {
    "initial_cash": 10000,
    "fee_bps_per_turnover": 0,
    "slippage_bps_per_turnover": 0,
    "max_drawdown_breaker": 0.0,  # only used by the plain run_backtest() comparison below
}


def make_frame(returns, targets):
    close = 100 * np.cumprod(1 + np.array(returns))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(returns), freq="4h", tz="UTC"),
            "close": close,
            "simple_return": returns,
            "target_position": targets,
        }
    )


def inert_safety(**overrides) -> SafetyConfig:
    """A safety config with every threshold effectively unreachable, for
    isolating one trigger at a time by overriding just that field."""
    base = dict(drawdown_pause=0.99, hard_shutdown_drawdown=0.999,
                minimum_equity_fraction=0.0, consecutive_loss_limit_bars=10_000,
                cooldown_bars=1)
    base.update(overrides)
    return SafetyConfig(**base)


# --------------------------------------------------------------------------- #
# Baseline equivalence: with every trigger unreachable, results should match
# the plain (frozen) run_backtest exactly -- the two-tier loop must not
# silently change strategy behavior when no safety condition ever fires.
# --------------------------------------------------------------------------- #

def test_matches_plain_backtest_when_no_trigger_fires():
    frame = make_frame([0.0, 0.05, -0.03, 0.04, -0.02, 0.03], [1.0] * 6)
    plain = run_backtest(frame, BASE_CFG).bars
    safe = run_backtest_with_two_tier_safety(frame, BASE_CFG, inert_safety()).bars
    assert np.allclose(plain["equity"], safe["equity"])
    assert np.allclose(plain["held_position"], safe["held_position"])
    assert (safe["safety_reason"] == "").all()


# --------------------------------------------------------------------------- #
# Hard shutdown: permanent, forces flat, triggered by EITHER drawdown or the
# minimum-equity floor.
# --------------------------------------------------------------------------- #

def test_hard_shutdown_on_drawdown_is_permanent():
    safety = inert_safety(hard_shutdown_drawdown=0.10)
    # bar1: -20% crash trips the hard shutdown; bar2/3 would otherwise profit
    # heavily (+50% each) if the signal were still followed.
    frame = make_frame([0.0, -0.20, 0.50, 0.50], [1.0, 1.0, 1.0, 1.0])
    result = run_backtest_with_two_tier_safety(frame, BASE_CFG, safety)
    assert result.breaker_timestamp is not None
    bars = result.bars
    assert bars.loc[2, "held_position"] == 0.0
    assert bars.loc[3, "held_position"] == 0.0
    assert bars.loc[2, "safety_reason"] == "hard_shutdown"
    assert bars.loc[3, "safety_reason"] == "hard_shutdown"
    # equity is frozen from the bar after the crash onward
    assert bars.loc[2, "equity"] == bars.loc[3, "equity"]


def test_hard_shutdown_on_minimum_equity_floor_independent_of_drawdown_threshold():
    # hard_shutdown_drawdown is unreachable (0.999); only the equity floor can fire.
    safety = inert_safety(hard_shutdown_drawdown=0.999, minimum_equity_fraction=0.5)
    frame = make_frame([0.0, -0.60, 0.50], [1.0, 1.0, 1.0])
    result = run_backtest_with_two_tier_safety(frame, BASE_CFG, safety)
    assert result.breaker_timestamp is not None
    assert result.bars.loc[2, "held_position"] == 0.0
    assert result.bars.loc[2, "safety_reason"] == "hard_shutdown"


def test_hard_shutdown_wins_priority_when_loss_streak_trips_the_same_bar():
    # Three consecutive -15% bars cross BOTH the loss-streak limit (3) and the
    # hard-shutdown drawdown (~38.6% cumulative) on the same bar's post-return
    # check. The trigger check's if/elif order must resolve this as a
    # permanent hard shutdown, not a merely-recoverable loss cooldown -- even
    # though cooldown_bars is set huge, which would make the wrong answer
    # look identical for many bars if the priority were reversed.
    safety = inert_safety(hard_shutdown_drawdown=0.30, consecutive_loss_limit_bars=3, cooldown_bars=50)
    frame = make_frame([0.0, -0.15, -0.15, -0.15, 0.50, 0.50], [1.0] * 6)
    result = run_backtest_with_two_tier_safety(frame, BASE_CFG, safety)
    bars = result.bars
    assert bars.loc[4, "safety_reason"] == "hard_shutdown"
    assert bars.loc[5, "safety_reason"] == "hard_shutdown"
    assert result.breaker_timestamp is not None
    # Permanent: the +50% bars that follow never resume trading.
    assert bars.loc[5, "equity"] == bars.loc[4, "equity"]


# --------------------------------------------------------------------------- #
# Loss-streak cooldown: independent of aggregate drawdown, recovers after
# cooldown_bars.
# --------------------------------------------------------------------------- #

def test_loss_streak_triggers_cooldown_independent_of_drawdown():
    # Small losing bars (drawdown stays well under the (unreachable) pause
    # threshold) but enough consecutive losses to trip the streak counter.
    safety = inert_safety(consecutive_loss_limit_bars=3, cooldown_bars=2)
    frame = make_frame([0.0, -0.01, -0.01, -0.01, 0.20, 0.20], [1.0] * 6)
    bars = run_backtest_with_two_tier_safety(frame, BASE_CFG, safety).bars
    # bars 1-3 are the three consecutive losers that trip the streak on bar 3.
    assert bars.loc[3, "held_position"] != 0.0  # the triggering bar itself still traded
    # bars 4-5: forced flat for the 2-bar cooldown.
    assert bars.loc[4, "safety_reason"] == "loss_cooldown"
    assert bars.loc[4, "held_position"] == 0.0
    assert bars.loc[5, "safety_reason"] == "loss_cooldown"
    assert bars.loc[5, "held_position"] == 0.0


def test_loss_streak_resumes_trading_after_cooldown_expires():
    safety = inert_safety(consecutive_loss_limit_bars=2, cooldown_bars=1)
    frame = make_frame([0.0, -0.01, -0.01, 0.20, 0.30], [1.0] * 5)
    bars = run_backtest_with_two_tier_safety(frame, BASE_CFG, safety).bars
    assert bars.loc[3, "safety_reason"] == "loss_cooldown"
    assert bars.loc[3, "held_position"] == 0.0
    assert bars.loc[4, "safety_reason"] == ""
    assert bars.loc[4, "held_position"] == 1.0


def test_positive_bars_reset_the_loss_streak():
    # A win between two losers must reset the streak counter to zero, so two
    # isolated single-bar losses never accidentally add up to a trip.
    safety = inert_safety(consecutive_loss_limit_bars=2, cooldown_bars=5)
    frame = make_frame([0.0, -0.01, 0.02, -0.01, -0.01], [1.0] * 5)
    bars = run_backtest_with_two_tier_safety(frame, BASE_CFG, safety).bars
    # Only the final two consecutive losers (bars 3,4) trip it -- there is no
    # bar left afterward to observe the cooldown, but bar 4 itself must still
    # have traded normally (the trigger fires only after the bar completes).
    assert bars.loc[4, "safety_reason"] == ""


# --------------------------------------------------------------------------- #
# Drawdown pause: soft, recovers, re-anchors peak on expiry.
# --------------------------------------------------------------------------- #

def test_drawdown_pause_forces_flat_then_recovers():
    safety = inert_safety(drawdown_pause=0.10, cooldown_bars=2)
    frame = make_frame([0.0, -0.15, 0.30, 0.30, 0.05], [1.0] * 5)
    bars = run_backtest_with_two_tier_safety(frame, BASE_CFG, safety).bars
    # bar1 (-15%) trips the pause for the check made AFTER it completes --
    # the crash bar itself still traded on the pre-existing signal (no lookahead).
    assert bars.loc[1, "held_position"] == 1.0
    assert bars.loc[1, "safety_reason"] == ""
    # bars 2-3: forced flat for the 2-bar cooldown.
    assert bars.loc[2, "safety_reason"] == "drawdown_pause"
    assert bars.loc[2, "held_position"] == 0.0
    assert bars.loc[3, "safety_reason"] == "drawdown_pause"
    assert bars.loc[3, "held_position"] == 0.0
    # bar4: cooldown expired, trading resumes.
    assert bars.loc[4, "safety_reason"] == ""
    assert bars.loc[4, "held_position"] == 1.0


def test_drawdown_pause_reanchors_peak_on_expiry():
    # cooldown_bars=1: bar2 is the single forced-flat cooldown bar, and the
    # re-anchor (peak = equity at the start of that bar) happens as it ends --
    # so bar2 itself already reads drawdown 0.0 against the fresh peak, even
    # though it's still labeled "drawdown_pause" (the label marks the bar that
    # was forced flat; the re-anchor is the state carried into bar3).
    safety = inert_safety(drawdown_pause=0.10, cooldown_bars=1)
    frame = make_frame([0.0, -0.15, 0.0, 0.0], [1.0] * 4)
    bars = run_backtest_with_two_tier_safety(frame, BASE_CFG, safety).bars
    assert bars.loc[2, "safety_reason"] == "drawdown_pause"
    assert bars.loc[2, "drawdown"] == 0.0
    assert bars.loc[3, "safety_reason"] == ""
    assert bars.loc[3, "held_position"] == 1.0


def test_drawdown_pause_does_not_instantly_retrigger_after_reanchor():
    # After re-anchoring and resuming, a further small dip (well under the
    # 10% threshold measured against the FRESH peak) must not re-trip the
    # pause for the bar after it.
    safety = inert_safety(drawdown_pause=0.10, cooldown_bars=1)
    frame = make_frame([0.0, -0.15, 0.0, -0.05, 0.10], [1.0] * 5)
    bars = run_backtest_with_two_tier_safety(frame, BASE_CFG, safety).bars
    assert bars.loc[3, "safety_reason"] == ""  # trading resumed, -5% applied
    assert bars.loc[4, "safety_reason"] == ""  # -5% off the re-anchored peak didn't re-trip


# --------------------------------------------------------------------------- #
# No-lookahead: a bar's OWN position decision cannot be affected by whatever
# that same bar's return turns out to be.
# --------------------------------------------------------------------------- #

def test_trigger_effect_is_visible_starting_next_bar_only():
    safety = inert_safety(drawdown_pause=0.05, cooldown_bars=3)
    frame = make_frame([0.0, -0.20, 0.10, 0.10, 0.10], [1.0] * 5)
    bars = run_backtest_with_two_tier_safety(frame, BASE_CFG, safety).bars
    # The crash bar (index 1) still reflects the pre-crash signal -- the
    # safety layer only reacts to a return after it has already happened.
    assert bars.loc[1, "held_position"] == 1.0
    assert bars.loc[1, "strategy_return"] < 0
    # Every bar after it is paused until the cooldown lapses.
    assert (bars.loc[2:4, "held_position"] == 0.0).all()
