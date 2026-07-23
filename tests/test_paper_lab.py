from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from btc_trend_bot.paper_lab import (
    AccountState,
    CandleFeatures,
    MarketQuote,
    PaperStore,
    add_candle_features,
    continuation_statistics,
    decide_target,
    execute_decision,
    round_trip_cost_bps,
)


LAB = {
    "fee_bps_per_side": 60.0,
    "slippage_bps_per_side": 5.0,
    "assumed_spread_bps_per_side": 1.0,
    "allocation_cap": 1.0,
    "min_notional": 1.0,
    "rebalance_tolerance_bps": 0.0,
}


def features(
    direction: int = 1,
    length: int = 2,
    run_bps: float = 150.0,
    rel_volume: float = 1.0,
    body: float = 0.8,
    trend: bool = True,
    regime_spread_bps: float = 25.0,
    momentum_bps: float = 40.0,
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> CandleFeatures:
    return CandleFeatures(
        bar_timestamp=timestamp,
        signal_close=100.0,
        streak_direction=direction,
        streak_length=length,
        run_return_bps=run_bps,
        relative_volume=rel_volume,
        body_fraction=body,
        broader_trend_up=trend,
        regime_spread_bps=regime_spread_bps,
        momentum_bps=momentum_bps,
    )


def state(strategy_id: str = "test") -> AccountState:
    return AccountState(
        strategy_id=strategy_id,
        initial_cash=500.0,
        cash=500.0,
        btc=0.0,
        gross_cash=500.0,
        gross_btc=0.0,
        peak_equity=500.0,
    )


class PaperLabTests(unittest.TestCase):
    def test_round_trip_cost(self):
        self.assertEqual(round_trip_cost_bps(LAB), 132.0)

    def test_one_bar_follow_enters_after_one_up_candle(self):
        decision = decide_target(
            {
                "id": "one_bar",
                "type": "candle_run",
                "entry_run_bars": 1,
                "exit_run_bars": 1,
                "fee_aware": False,
            },
            features(length=1),
            previous_target=0.0,
            lab_cfg=LAB,
        )
        self.assertEqual(decision.target_position, 1.0)
        self.assertEqual(decision.signal, "enter")

    def test_two_bar_run_waits_after_one_up_candle(self):
        decision = decide_target(
            {
                "id": "two_bar",
                "type": "candle_run",
                "entry_run_bars": 2,
                "exit_run_bars": 2,
                "fee_aware": False,
            },
            features(length=1),
            previous_target=0.0,
            lab_cfg=LAB,
        )
        self.assertEqual(decision.target_position, 0.0)
        self.assertEqual(decision.signal, "hold")

    def test_run_exits_after_configured_down_streak(self):
        decision = decide_target(
            {
                "id": "raw",
                "type": "candle_run",
                "entry_run_bars": 2,
                "exit_run_bars": 2,
                "fee_aware": False,
            },
            features(direction=-1, length=2, run_bps=-50.0),
            previous_target=1.0,
            lab_cfg=LAB,
        )
        self.assertEqual(decision.target_position, 0.0)
        self.assertEqual(decision.signal, "exit")

    def test_fee_aware_rejects_run_below_round_trip_cost(self):
        decision = decide_target(
            {
                "id": "aware",
                "type": "candle_run",
                "entry_run_bars": 2,
                "exit_run_bars": 2,
                "fee_aware": True,
                "cost_hurdle_multiplier": 1.0,
                "min_relative_volume": 0.8,
                "min_body_fraction": 0.4,
                "require_broader_uptrend": True,
            },
            features(run_bps=100.0),
            previous_target=0.0,
            lab_cfg=LAB,
        )
        self.assertEqual(decision.target_position, 0.0)
        self.assertEqual(decision.signal, "hold")
        self.assertIn("hurdle", decision.reason)

    def test_fee_aware_enters_when_all_filters_pass(self):
        decision = decide_target(
            {
                "id": "aware",
                "type": "candle_run",
                "entry_run_bars": 2,
                "exit_run_bars": 2,
                "fee_aware": True,
                "cost_hurdle_multiplier": 1.0,
                "min_relative_volume": 0.8,
                "min_body_fraction": 0.4,
                "require_broader_uptrend": True,
            },
            features(run_bps=160.0),
            previous_target=0.0,
            lab_cfg=LAB,
        )
        self.assertEqual(decision.target_position, 1.0)
        self.assertEqual(decision.signal, "enter")


    def test_swing_entry_requires_regime_confirmation(self):
        decision = decide_target(
            {
                "id": "swing",
                "type": "candle_swing",
                "entry_run_bars": 2,
                "min_run_return_bps": 8.0,
                "min_relative_volume": 0.8,
                "min_body_fraction": 0.35,
                "entry_regime_spread_bps": 10.0,
                "entry_momentum_bps": 15.0,
                "require_broader_uptrend": True,
                "exit_run_bars": 2,
                "exit_regime_spread_bps": -5.0,
                "exit_momentum_bps": -20.0,
                "exit_confirmations_required": 2,
            },
            features(regime_spread_bps=5.0),
            previous_target=0.0,
            lab_cfg=LAB,
        )
        self.assertEqual(decision.target_position, 0.0)
        self.assertEqual(decision.signal, "hold_cash")
        self.assertIn("EMA spread", decision.reason)

    def test_swing_enters_when_all_entry_filters_pass(self):
        decision = decide_target(
            {
                "id": "swing",
                "type": "candle_swing",
                "entry_run_bars": 2,
                "min_run_return_bps": 8.0,
                "min_relative_volume": 0.8,
                "min_body_fraction": 0.35,
                "entry_regime_spread_bps": 10.0,
                "entry_momentum_bps": 15.0,
                "require_broader_uptrend": True,
                "exit_run_bars": 2,
                "exit_regime_spread_bps": -5.0,
                "exit_momentum_bps": -20.0,
                "exit_confirmations_required": 2,
            },
            features(),
            previous_target=0.0,
            lab_cfg=LAB,
        )
        self.assertEqual(decision.target_position, 1.0)
        self.assertEqual(decision.signal, "enter")

    def test_swing_does_not_exit_on_down_run_alone(self):
        decision = decide_target(
            {
                "id": "swing",
                "type": "candle_swing",
                "entry_run_bars": 2,
                "exit_run_bars": 2,
                "exit_regime_spread_bps": -5.0,
                "exit_momentum_bps": -20.0,
                "exit_confirmations_required": 2,
            },
            features(
                direction=-1,
                length=2,
                run_bps=-20.0,
                regime_spread_bps=15.0,
                momentum_bps=10.0,
            ),
            previous_target=1.0,
            lab_cfg=LAB,
        )
        self.assertEqual(decision.target_position, 1.0)
        self.assertEqual(decision.signal, "hold_btc")

    def test_swing_exits_with_two_deterioration_confirmations(self):
        decision = decide_target(
            {
                "id": "swing",
                "type": "candle_swing",
                "entry_run_bars": 2,
                "exit_run_bars": 2,
                "exit_regime_spread_bps": -5.0,
                "exit_momentum_bps": -20.0,
                "exit_confirmations_required": 2,
            },
            features(
                direction=-1,
                length=2,
                run_bps=-20.0,
                regime_spread_bps=-10.0,
                momentum_bps=5.0,
            ),
            previous_target=1.0,
            lab_cfg=LAB,
        )
        self.assertEqual(decision.target_position, 0.0)
        self.assertEqual(decision.signal, "exit")

    def test_buy_has_fee_spread_and_slippage_drag(self):
        decision = decide_target(
            {"id": "buy_hold", "type": "buy_hold"},
            features(),
            previous_target=0.0,
            lab_cfg=LAB,
        )
        result = execute_decision(
            state("buy_hold"),
            decision,
            features(),
            MarketQuote(
                timestamp="2026-01-01T00:01:00+00:00",
                mark=100.0,
                bid=99.9,
                ask=100.1,
            ),
            LAB,
        )
        self.assertIsNotNone(result.trade)
        assert result.trade is not None
        self.assertGreater(result.snapshot["gross_equity"], result.snapshot["equity"])
        self.assertGreater(result.snapshot["total_fees"], 0.0)
        self.assertGreater(result.snapshot["total_spread"], 0.0)
        self.assertGreater(result.snapshot["total_slippage"], 0.0)

    def test_sell_moves_portfolio_back_to_cash(self):
        account = state("run")
        enter = decide_target(
            {"id": "run", "type": "buy_hold"},
            features(),
            previous_target=0.0,
            lab_cfg=LAB,
        )
        first = execute_decision(
            account,
            enter,
            features(),
            MarketQuote("2026-01-01T00:01:00+00:00", 100.0, 100.0, 100.0),
            LAB,
        )
        exit_decision = decide_target(
            {
                "id": "run",
                "type": "candle_run",
                "entry_run_bars": 1,
                "exit_run_bars": 1,
                "fee_aware": False,
            },
            features(
                direction=-1,
                length=1,
                timestamp="2026-01-01T00:05:00+00:00",
            ),
            previous_target=1.0,
            lab_cfg=LAB,
        )
        second = execute_decision(
            first.state,
            exit_decision,
            features(
                direction=-1,
                length=1,
                timestamp="2026-01-01T00:05:00+00:00",
            ),
            MarketQuote("2026-01-01T00:06:00+00:00", 100.0, 100.0, 100.0),
            LAB,
        )
        self.assertEqual(second.state.btc, 0.0)
        self.assertEqual(second.state.trade_count, 2)

    def test_candle_feature_streak_and_doji_break(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
                "open": [100.0, 101.0, 102.0, 103.0, 103.0],
                "high": [102.0, 103.0, 104.0, 103.01, 105.0],
                "low": [99.0, 100.0, 101.0, 102.99, 102.0],
                "close": [101.0, 102.0, 103.0, 103.0, 104.0],
                "volume": [10.0, 11.0, 12.0, 13.0, 14.0],
            }
        )
        out = add_candle_features(
            frame,
            {
                "doji_threshold_bps": 1.0,
                "volume_window_bars": 2,
                "broader_trend_ema_bars": 2,
            },
        )
        self.assertEqual(int(out.iloc[2]["streak_length"]), 3)
        self.assertEqual(int(out.iloc[3]["streak_length"]), 0)
        self.assertEqual(int(out.iloc[4]["streak_length"]), 1)

    def test_continuation_statistics_include_unconditional_row(self):
        frame = pd.DataFrame(
            {
                "streak_direction": [1, 1, -1, -1],
                "streak_length": [1, 2, 1, 2],
                "next_candle_return_bps": [2.0, -1.0, -2.0, 1.0],
            }
        )
        stats = continuation_statistics(frame, max_streak=2)
        self.assertEqual(stats.iloc[0]["direction"], "unconditional")
        self.assertEqual(int(stats.iloc[0]["observations"]), 4)

    def test_store_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PaperStore(Path(directory) / "paper.sqlite3")
            try:
                account, created = store.get_or_create_account("cash", 500.0)
                self.assertTrue(created)
                decision = decide_target(
                    {"id": "cash", "type": "cash"},
                    features(),
                    previous_target=0.0,
                    lab_cfg=LAB,
                )
                result = execute_decision(
                    account,
                    decision,
                    features(),
                    MarketQuote(
                        timestamp="2026-01-01T00:01:00+00:00",
                        mark=100.0,
                        bid=100.0,
                        ask=100.0,
                    ),
                    LAB,
                )
                self.assertTrue(store.persist_result(result))
                self.assertFalse(store.persist_result(result))
                loaded, created_again = store.get_or_create_account("cash", 500.0)
                self.assertFalse(created_again)
                self.assertEqual(loaded.last_bar_timestamp, features().bar_timestamp)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
