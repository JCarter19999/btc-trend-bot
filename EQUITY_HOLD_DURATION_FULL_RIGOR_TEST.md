# Hold-duration full-rigor test: 10/15/20/30 days

Extends `EQUITY_EXIT_REGIME_SIMPLE_TREND.md`'s single-run hold-duration
sweep (5/10/15/20d, no drawdown/OOS/random-control ever reported for
15/20d) with real rigor, and adds 30 days (never tested before).

Methodology: `_select_fold_winners` against the full candidate set
(bypassing `walk_forward()`'s 756-bar warmup, which `simple_trend` doesn't
need) + `portfolio_sim.simulate_single_position` (genuinely single-position),
fixed-notional $2,500. Candidates rebuilt fresh per hold duration (max_hold_bars
is baked into `simulate_trade`'s exit logic at candidate-generation time).
10-day baseline reconciliation **confirmed exact**: 188 trades, 58.0% win
rate, PF-equivalent total return +457.4%, 22.1% max drawdown, 243.3bps/trade
— matches the documented characterization exactly before trusting anything
built on top of it.

## Full comparison table

| Hold | Trades | Win rate | Total return | Max DD | Mean/trade | Random-ctrl percentile | OOS first half | OOS second half |
|---|---|---|---|---|---|---|---|---|
| 10d | 188 | 58.0% | **+457.4%** | 22.1% | 243.3bps | 90.0 | +223.9% | +204.1% |
| 15d | 122 | 54.9% | +332.2% | 25.1% | 272.3bps | **50.0** | +192.5% | +75.1% |
| 20d | 94 | 66.0% | +425.5% | **20.4%** | 452.7bps | 88.0 | +281.3% | +219.0% |
| 30d | 62 | 56.5% | +456.2% | 30.0% | **735.8bps** | 90.0 | +319.2% | **+22.6%** |

**Buy-and-hold, equal-weight, all 4 stocks, no rotation, same window: +1487.0%**
— dominates every single tested variant by a wide margin.

## What the rigor actually reveals, not just the headline numbers

1. **Mean return/trade climbs monotonically (243→272→453→736bps), but total
   return does not** — it's roughly flat/oscillating (+457% → +332% → +426%
   → +456%), because trade count shrinks (188→122→94→62) fast enough to
   offset the per-trade quality gain. The original "monotonic expectancy"
   framing is real but doesn't translate into "longer is better" on the
   metric that actually matters for an account.
2. **15-day hold shows essentially no separation from random selection**
   (50th percentile) — the weakest point in this whole sweep, worse than
   both the shorter (10d) and longer (20d, 30d) durations on this specific
   check. Its total return still looks respectable, but the selection
   criterion isn't demonstrably doing anything at this duration.
3. **30-day hold fails out-of-sample decisively**: +319.2% first half vs.
   +22.6% second half — virtually the entire headline return comes from
   one period. This is the single clearest piece of evidence in this test
   that the monotonic-expectancy trend is not a free lunch further out;
   30 days looks like it's riding one concentrated historical move, not a
   stable, repeatable edge.
4. **20-day hold is the most internally consistent longer option** (66%
   win rate, 88th-percentile random control, OOS +281%/+219% — same sign,
   proportional, not the instability 30d shows) but still doesn't beat
   10-day on total return (+425.5% vs +457.4%), only on drawdown (20.4%
   vs 22.1%).
5. **The buy-and-hold gap does not shrink with duration the way the
   "eventually converges to buy-and-hold" framing predicted** — it's
   already enormous at every duration tested (10d captures ~31% of
   buy-and-hold's return; 30d captures ~31% too). This rotation strategy
   family isn't gradually approaching buy-and-hold as holds lengthen; it
   underperforms it by roughly the same wide margin throughout the tested
   range.

## Verdict

No duration beyond 10 is worth adopting. 20d is the closest competitor on
risk (marginally better drawdown) but not on return, and doesn't clear a
high enough bar to justify changing the live deployment. 30d's apparent
strength is substantially an out-of-sample artifact, not a real edge —
its whole headline return lives in one half of the sample. 15d is
weak on its own terms (no better than random selection). The bigger,
more important finding than any single duration comparison: **this
entire strategy family, at every hold length tested, captures only
roughly a third of what simply buying and holding the same 4 stocks
equal-weighted would have returned over the same window** — a sobering
number worth keeping in view whenever this strategy's absolute returns
get discussed in isolation.

Script: `experiments/run_hold_duration_full_rigor_test.py`. Outputs:
`outputs/hold_duration_full_rigor_test/`.
