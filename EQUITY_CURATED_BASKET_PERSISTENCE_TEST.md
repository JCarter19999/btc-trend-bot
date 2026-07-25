# Curated 8-symbol basket, 5 concurrent slots, persistence-gated swaps

Joey's idea: fix two prior findings' weaknesses at once -- hold 5 positions
from a curated basket (not the full S&P 100, which diluted signal quality
past the top 2 picks) with a persistence/dwell-time filter before swapping
any slot (fixing the whipsaw that hurt continuous single-position
switching). Tested directly, not by analogy.

## Setup

Universe: AAPL, MSFT, NVDA, TSLA (existing 4) + AMZN, GOOGL, META, AVGO (4
more mega-cap secular-growth names) -- all 8 had real cached daily data
already available (`data/real/` for the first 5, `data/real_sp100/` for
GOOGL/META/AVGO), no substitutions needed. Benchmark: SPY. Same
`relative_strength_20` / `market_above_ema50` features and pool-fallback
logic as `simple_trend`'s own selector, applied across all 8 symbols daily.

5 concurrent slots, $500 fixed notional each (not rebased on gains).
**No fixed 10-day hold** -- positions ride until displaced by the
persistence rule, a deliberately different exit mechanic from
`simple_trend`'s live time_exit, since testing whether rank-driven exits
beat a blind timer is the point.

**Persistence rule** (primary interpretation of Joey's phrasing): a
non-held symbol becomes a "confirmed challenger" once it has been
continuously ranked top-5 for MORE than X trading days. A challenger fills
an empty slot immediately if one exists; otherwise it displaces whichever
held name has fallen furthest out of the top-5 ranking (if any held name
has fallen out at all). A held name that's merely dropped out of top-5 but
has no qualified challenger yet is NOT force-exited -- it keeps riding.
Initial portfolio fill (day 1) bypasses the persistence requirement since
there's no dwell history yet. Tested X = 0 (no filter, immediate swap --
serves as the internal control), 3, 5, 10 trading days.

Date range: 2018-01-31 to 2026-07-22 (2,129 trading days).

## Results

| | 4-stock single-position (baseline) | Full S&P100 5-way (prior study) | X0 (curated, no filter) | X3 | X5 | X10 |
|---|---|---|---|---|---|---|
| Trades | 188 | 350 | 988 | 367 | 297 | **177** |
| Win rate | 58.0% | 56.6% | 47.5% | 51.8% | 50.2% | **57.1%** |
| Mean return/trade | 243.3bps | 169.1bps | 1.40% | 3.93% | 4.62% | **9.39%** |
| Total return | **+457.4%** | +193% | +275.9% | +288.5% | +274.4% | +332.4% |
| Max drawdown | **22.1%** | 29.1% | 24.9% | 26.2% | 23.4% | 37.3% |

Cost stress (5bps vs. 2bps baseline) barely moves any curated-basket
variant (X0: 275.9%->263.9%, X3: 288.5%->283.9%, X5: 274.4%->270.6%, X10:
332.4%->330.1%) -- these trades carry much larger average returns per
trade than the single-position or full-S&P100 studies, so a few bps of
cost is comparatively immaterial here, unlike every low-return-per-trade
strategy tested earlier this session.

**Random-selection control** (5 random symbols/day, no persistence, 20
seeds): mean total return 250.2%, std 30.5%. Z-scores of the real
variants: X0 = 0.84, X3 = 1.26, X5 = 0.79, **X10 = 2.7**. Only X10
clearly separates from random noise (roughly 99.6th percentile assuming
normality); X0/X3/X5 sit within about 1-1.3 standard deviations of the
random-selection mean -- weaker separation from noise than this project's
usual ~95th-99th-percentile bar for a trusted finding. Stated plainly:
most of these variants have not clearly proven they're better than random
symbol selection at this trade frequency; only X10 has.

**Out-of-sample split** (first half 2018-2022 / second half 2022-2026),
all variants: no sign reversal in any of them, but a large, consistent
asymmetry -- second half returns roughly 2-2.5x the first half's in every
variant (e.g. X10: +169.7% first half, +345.0% second half). This is
directionally stable but the magnitude gap is large enough to be a real
caveat: this window's second half coincides with a strong mega-cap tech
bull run, so some of the apparent edge may be regime-specific to this
exact period rather than a stable, repeatable property of the mechanism.

## Verdict: persistence filtering works as intended, but doesn't beat the
## concentrated single-position baseline, and only X10 clears the
## noise bar

The persistence filter does exactly what it was designed to do: X0 (988
trades, 1.40% mean/trade) -> X10 (177 trades, 9.39% mean/trade) shows
dwell-time filtering sharply improves per-trade quality and win rate as X
increases, confirming the mechanism -- requiring sustained leadership
before swapping does filter out short-term rank noise, exactly the fix
this was designed to be for the continuous-switching test's whipsaw
problem.

But **none of the four curated-basket variants beat the original 4-stock
single-position baseline** on total return (+457.4%) or, except for X5,
on drawdown (22.1%). X10 gets closest on return (+332.4%) but at the cost
of the worst drawdown of any variant tested here (37.3%, worse than even
the full-S&P100 5-way study's 29.1%) -- a real, not cherry-picked,
risk/reward tradeoff, not a clean win. And only X10 separates clearly
from the random-selection control; X3/X5 are statistically weaker than
this project's usual bar for a trusted result.

**Net recommendation**: this specific implementation (5 slots, curated
8-symbol basket, persistence-gated swaps) does not beat the deployed
4-stock single-position strategy on the two metrics that matter most
(return, drawdown), and the trade-off it offers (higher per-trade quality,
fewer trades, at the cost of materially worse drawdown at the
best-performing threshold) isn't a clean upgrade. The core finding from
the exit-regime and hold-duration work continues to hold up under yet
another test: this basket's edge is medium-term drift capture best
harvested with patience and concentration, not breadth or persistence-
gated rotation.

Script: `experiments/run_curated_basket_persistence_test.py`. Outputs:
`outputs/curated_basket_persistence_test/` (per-variant trade CSVs,
summary.json).
