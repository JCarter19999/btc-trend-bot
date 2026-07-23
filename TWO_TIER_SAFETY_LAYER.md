# Two-Tier Capital Safety Layer (candidate, not promoted)

**Status: research candidate only.** This does not touch `backtest.py`,
`production.py`, `cli.py`, or `config/settings_production.yaml`. The frozen
production strategy remains fixed-size, long/cash, no drawdown breaker,
exactly as documented in `CHANGES.md` and `README.md`. Nothing here is wired
into the deploy path.

## Motivation

The README lists "a portfolio drawdown circuit breaker" as a headline design
feature, and `backtest.py` has the plumbing for one (`max_drawdown_breaker`,
checked in `run_backtest`) -- but `config/settings_production.yaml` sets it to
`0.0`, disabling it. There is no cooldown, no loss-streak protection, and the
one breaker that exists is all-or-nothing: once tripped, the position is
forced to zero for the rest of the run, permanently.

This mirrors a design already built, backtested, and stress-tested in the
`equity_v2_4` project's `simulate_capital` (see that project's
`run_equity_real_data_walkforward.py`): a two-tier scheme with a *soft*,
recoverable drawdown pause plus a *separate* consecutive-loss cooldown, in
front of the same *hard*, permanent shutdown as a backstop. That layer
survived the equity project's own regime-stress and promotion-gate testing.
This candidate asks whether the same principle transfers to this project's
completely different trading mechanics.

## Why the mechanics don't transfer literally

`equity_v2_4` trades discrete round-trip positions (ATR stop/target/time
exit, one active trade at a time). This project rebalances a **continuous**
long/flat exposure every 4h bar (EMA trend + Donchian breakout + realized-vol
sizing, `target_position` in `[0, 1]`). There is no discrete "trade" unit here
to count losses against. The risk-control *principle* is exposure-model
agnostic (it only needs equity, peak, and drawdown, which this project's loop
already tracks per bar) -- but the specific numbers had to be re-derived, not
copied:

| Parameter | Equity project | This candidate | Note |
|---|---|---|---|
| Soft drawdown pause | 15% | 15% | Percentage of equity -- portable unchanged |
| Hard shutdown drawdown | 35% | 35% | Percentage of equity -- portable unchanged |
| Minimum equity floor | $25 / $2,500 (1%) | 1% of initial cash | Portable as a percentage |
| Loss-streak trigger | 4 consecutive losing **trades** | 8 consecutive losing **4h bars** | No trade unit exists here; re-denominated in bars (see sensitivity results) |
| Cooldown length | 8 trades | 30 bars (~5 days) | Same reasoning |

## Design

Same priority order as `equity_v2_4`'s `simulate_capital`, adapted to a
per-bar loop instead of a per-trade loop (implemented in
`experiments/run_two_tier_safety_backtest.py::run_backtest_with_two_tier_safety`,
which otherwise reuses `backtest.py`'s exact cost/turnover/return mechanics):

1. **Hard shutdown** (permanent): equity below the minimum-equity floor, or
   drawdown at or beyond the hard-shutdown threshold. Forces position to zero
   for every remaining bar. Never recovers.
2. **Loss-streak cooldown**: N consecutive losing bars trips a fixed-length
   cooldown (position forced to zero) independent of aggregate drawdown --
   catches a fast, sharp losing run before it becomes a deep drawdown.
3. **Soft drawdown pause**: drawdown crosses the (shallower) soft threshold,
   trips the same fixed-length cooldown. When the cooldown ends, the equity
   peak re-anchors to current equity so the pause doesn't instantly re-trip.
4. Otherwise, trade the strategy's own signal unmodified.

All decisions for a given bar use only state carried over from the *prior*
bar's outcome -- no lookahead into the bar being decided.

## Validation performed

1. **Full-history backtest** (2018-2026, `experiments/run_two_tier_safety_backtest.py`)
   against the frozen no-breaker baseline and buy-and-hold BTC, plus a paired
   moving-block bootstrap on the per-bar return difference (this project's own
   validation method, per `README.md`'s Variant B description).
2. **Parameter sensitivity sweep** (`experiments/run_two_tier_safety_sensitivity.py`),
   one-at-a-time from a first-pass default, to find landmines and inert zones
   rather than trust a single point estimate.
3. **Out-of-sample validation** (`experiments/run_two_tier_safety_oos_validation.py`):
   parameters chosen from a train-only sweep on 2018-2021 (never sees the 2022
   crash), then evaluated unchanged on held-out 2022-2026 data.

See `TWO_TIER_SAFETY_LAYER_RESULTS.md` for numbers and interpretation.

## What would still be needed before promotion

- Cost-stress sensitivity (2x/3x/5x fees+slippage), matching this project's
  existing Variant B validation habit.
- A second data source / exchange replication check (the project already has
  this pattern for the short-overlay validation).
- A synthetic crash scenario to actually exercise the hard-shutdown code path
  (it never fired once in 8+ years of real data at any tested threshold --
  see results doc -- so it is currently unvalidated by real data and would
  need a stress test, not a historical backtest, to check).
- Sign-off that `deployment.mode` stays `dry_run` through a live shadow period
  before any live-capital consideration, per this project's own deployment
  gates in `DEPLOYMENT.md`.
