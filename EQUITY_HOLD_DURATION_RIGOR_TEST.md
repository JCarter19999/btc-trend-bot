# Hold-duration rigor test: does longer holding keep helping, or does the
# original sweep's monotonic pattern not survive real scrutiny?

The original `EQUITY_EXIT_REGIME_SIMPLE_TREND.md` hold-duration sweep (5/10/
15/20 days, single run each) found expectancy/trade increasing monotonically
with no peak, and framed longer holds as "capturing more of the underlying
drift, converging toward this basket's buy-and-hold performance." That framing
was never tested with drawdown, an out-of-sample split, a random-selection
control, or an actual buy-and-hold comparison. This does all four, plus adds
30 days (never tested before).

## Method

Same reconstruction as every prior fork this session: `_select_fold_winners`
called directly against the full candidate set (bypassing `walk_forward()`'s
756-bar warmup), fed through `portfolio_sim.simulate_single_position` (the
correct single-position simulator), `stop_atr`/`target_atr` = 100 (disabled,
matching the live deployment), $2,500 fixed-notional sizing, AAPL/MSFT/NVDA/
TSLA, 2018-2026. **Baseline (10-day) reproduces the documented characterization
exactly**: 188 trades, 58.0% win rate, 243.3bps mean/trade, +457.4% total
return, 22.1% max drawdown — confirmed before trusting anything built on top
of it.

## Results

| Hold (days) | Trades | Win rate | Mean/trade | Total return | Max DD | Buy-and-hold (same window) | Strategy vs. B&H | OOS (1st / 2nd half) | Random-control percentile |
|---|---|---|---|---|---|---|---|---|---|
| 10 (deployed) | 188 | 58.0% | 243.3bps | **+457.4%** | 22.1% | +1487.0% | **-1029.5pp** | +223.9% / +204.1% | **95.5th** |
| 15 | 122 | 54.9% | 272.3bps | +332.2% | 25.1% | +1539.9% | -1207.7pp | +192.5% / +75.1% | **55.5th** |
| 20 | 94 | 66.0% | 452.7bps | +425.5% | 20.4% | +1713.5% | -1288.0pp | +281.3% / +219.0% | 90.5th |
| 30 | 62 | 56.5% | 735.8bps | +456.2% | 30.0% | +1661.8% | -1205.6pp | **+319.2% / +22.6%** | 90.5th |

(200-seed random-selection control per hold length, same candidate pool,
selection replaced by uniform-random choice per signal date.)

## Three findings that correct the original sweep's framing

**1. Total return does NOT monotonically increase — only per-trade expectancy
does, and that's partly a mechanical artifact of fewer, bigger trades.**
Mean return/trade climbs cleanly (243→272→453→736bps) but total portfolio
return wobbles (457%→332%→426%→456%) because longer holds mean fewer trades
compounding over the same window. 15 days is actually the *worst* total
return of the four despite the second-highest expectancy/trade -- extending
the hold is not a free lunch on the metric that actually matters (ending
capital).

**2. Buy-and-hold dominates the strategy by roughly 1,000+ percentage points
at every single hold length tested.** This needed to be measured directly
rather than assumed from the original doc's "converging toward buy-and-hold"
language, which turns out to describe the *mechanism* (medium-term drift
capture), not the *magnitude*. Equal-weight buy-and-hold of the same 4
stocks over the identical windows returns +1487% to +1714% depending on the
exact window -- the strategy, at its best (10d or 30d, ~457%), captures
roughly 30% of that. This is expected given the strategy holds one $2,500
notional at a time (idle cash between trades, no compounding) versus
buy-and-hold's continuous full exposure across all 4 names -- but it's a
number worth having explicitly rather than inferring from a qualitative
"converges toward" phrase.

**3. Longer holds do NOT strengthen the case that this is a real selection
edge over random choice -- if anything, 10 days remains the best-validated
length.** Random-selection-control percentile: 10d=95.5th, 15d=**55.5th**
(essentially indistinguishable from random luck), 20d=90.5th, 30d=90.5th.
15 days in particular shows almost no evidence of real selection skill by
this measure despite looking fine in isolation (54.9% win rate, positive
expectancy) -- a good example of why a raw number without a random-control
comparison can mislead.

## The 30-day result's OOS split is the single biggest red flag here

30 days has the best full-sample expectancy (736bps/trade) and ties for
best total return (456.2%, essentially matching the deployed 10-day
number) -- but its out-of-sample split shows severe decay: **+319.2% first
half, only +22.6% second half.** Not a sign reversal, but close to a
collapse -- nearly all of the headline 30-day number comes from 2018-2022,
with the 2022-2026 half barely beating a savings account. Compare the
10-day baseline's OOS stability: +223.9% / +204.1%, almost no decay at all.
**The 30-day variant's attractive full-sample headline is substantially a
first-half artifact, not a durable property.**

## Verdict: extending the hold period is not a validated improvement

None of 15/20/30 days beats the deployed 10-day hold on the dimensions that
matter once real rigor is applied:
- 15 days: worse total return, worse OOS stability in the second half,
  and by far the weakest random-control separation (55.5th percentile) --
  the weakest candidate of the four, not an improvement.
- 20 days: roughly comparable total return and drawdown to 10 days, but
  weaker random-control separation (90.5th vs. 95.5th) and no clear
  advantage to justify the added complexity/reduced trade frequency.
- 30 days: ties 10 days on the aggregate headline number, but the OOS
  split reveals that headline is substantially a 2018-2022-driven
  artifact, not a stable edge -- the single biggest caution flag in this
  test.

**10 days remains the best-supported hold length of the five tested (5/10/
15/20/30)** -- not because longer holds were proven wrong in principle, but
because none of them demonstrated a real, stable improvement once checked
against random selection and an out-of-sample split, and one of them (30
days) actively risks mistaking a front-loaded, non-durable result for a
real edge. This doesn't contradict the original doc's "10 was kept because
it's the pre-existing default, not because a sweep picked it" caveat -- it
adds evidence that the default happens to also be the best-validated choice
among the lengths checked, which wasn't previously established.

Script: `experiments/run_hold_duration_rigor_test.py`. Outputs:
`outputs/hold_duration_rigor_test/summary.json`.
