# Does the DAX signal sharpen simple_trend's stock rotation?

Combination hypothesis: two already-validated mechanisms (the DAX
cross-market lead-lag signal, and `simple_trend`'s momentum rotation) --
does knowing the DAX reading help pick which tech stock will do well, or
whether that day's rotation trade will do well?

No-lookahead check: DAX's pre-open reading is known by 13:30 UTC (US open)
on date T. `simple_trend`'s own signal for date T is generated from date
T's close (~21:00 UTC), entered at T+1's open. DAX's same-day reading is
legitimately known hours before `simple_trend`'s own signal-generation
moment that same date -- joining on the same calendar date T is not
lookahead.

Baseline reconstruction confirmed exact before testing anything: 188
trades, 58.0% win rate, +457.4% total return (fixed-notional sum, not
compounded -- verified this matches `simulate_single_position`'s own
`total_return` exactly, 4.5742 either way).

## Hypothesis 1: joint-signal-day effect -- too thin to trust

DAX intraday data only covers 2023-09-06 onward (yfinance limit, same
constraint as every other DAX-signal study this session), so only 42 of
188 `simple_trend` trades (22.3%) fall within the DAX-covered window at
all. Of those 42, only **9** land on a DAX top-quartile-move day.

**n=9 is too thin to trust either way -- stated plainly, not forced into a
conclusion.** Same thinness pattern as the regime classifier's earlier
~25-27-day joint buckets, just worse here because `simple_trend`'s trades
are sparse (10-day holds) relative to DAX's daily signal.

## Hypothesis 2: does DAX direction correlate with the leader's return?

Larger sample: joined DAX's direction against the RAW daily candidate pool
(every day's `simple_trend` pick and its `net_return`, before
single-position filtering -- 345 of 1,539 total candidates fall in the
DAX-covered window, a real, usable sample unlike Hypothesis 1).

| | n | Mean leader return |
|---|---|---|
| DAX-bullish mornings | 173 | +0.408% |
| DAX-bearish mornings | 172 | **+1.550%** |

Real diff (bullish − bearish): **−1.14pp** -- the OPPOSITE of the naive
expectation that a DAX-bullish morning would predict stronger tech
momentum. Shuffled-null (1,000 seeds): null mean +0.03pp, std 0.87pp,
**8.7th percentile** -- the real diff sits far enough into the null's left
tail to be a genuine, non-noise-level separation, just in an unhelpful
direction (bearish mornings had the better leader returns, not bullish
ones).

**But it does not hold up out of sample -- it reverses sign:**

| Half | n | Bullish mean | Bearish mean | Diff |
|---|---|---|---|---|
| First (2023-09 to ~2025-02) | 173 | −1.32% | +0.71% | **−2.03pp** |
| Second (~2025-02 to 2026-07) | 172 | +2.66% | +2.20% | **+0.46pp** |

Same sign-reversal failure mode that's killed multiple candidates this
session (the BTC breakout strategy, the equity-side realized-vol strangle
filter, the momentum-revision check). A full-sample statistic that flips
sign between halves is not a stable, tradeable effect.

## Actionable check: does DAX-conditioning improve the real 188 trades?

Directly tested: drop the 23 of 188 real trades (12.2%) that fell on a
DAX-bearish morning within the covered window, keep the rest.

| | Trades | Total return (fixed-notional) |
|---|---|---|
| Original | 188 | **+457.4%** |
| DAX-bearish-mornings removed | 165 | **+393.9%** |

**Filtering makes it worse, not better** -- directly consistent with
Hypothesis 2's finding: bearish mornings actually had the better leader
returns in this sample, so removing them removes some of the good trades.

## Verdict

**No actionable combination found.** Hypothesis 1 is genuinely
underpowered (n=9), not rejected -- an honest "can't tell" result, not a
negative one. Hypothesis 2 shows a real, non-noise full-sample separation
(8.7th percentile) but it is in the wrong direction to be useful AND
reverses sign out-of-sample, the same failure signature that has killed
several other candidates tonight. The direct actionable test (removing
DAX-bearish-morning trades) confirms this by making performance worse,
not better. The two mechanisms (DAX cross-market lead-lag, tech-basket
momentum rotation) remain independently real and validated -- there is
just no evidence, at the sample sizes available, that combining them adds
anything actionable over running each independently.

Script: `experiments/run_dax_informed_rotation_test.py`. Outputs:
`outputs/dax_informed_rotation_test/`.
