# Short strangle for sideways/choppy regimes: real ThetaData, decisive negative

Joey's ask (2026-07-24): `simple_trend` is trend-following and known to
degrade in chop (see `EQUITY_EXIT_REGIME_SIMPLE_TREND.md` -- its whole edge
is "don't cut positions early, ride the drift," which has nothing to offer
when there's no drift). What's the best way to trade sideways/choppy
conditions? The recommendation was to SELL premium instead of buying more
of it -- opposite of every long-premium structure tested so far in this
project -- gated to a flat-trend regime. This is that test, full real-data
rigor.

**Verdict up front: does not survive, in any of the three regime-filter
variants tested.** Trend-slope: weak baseline, fails minimal cost stress,
inconsistent out-of-sample, no advantage over an unconditional short
strangle. Realized-volatility (follow-up 1): decisively worse, not better
-- strongly negative baseline, worse than random dates by a wide margin,
revealed via P&L decomposition to be actively selecting for vol-
*expansion* points rather than genuinely calm ones. Implied-vol-vs-
realized-vol / "volatility risk premium" (follow-up 2, the forward-looking
one): worse still in absolute terms (-75.2% total return) and the widest
gap yet below a random-date control (up to 126pp) -- also selects for
vol-expansion, most likely because an elevated IV/RV ratio is the options
market correctly anticipating a real event, not mispricing calm as risk.
None of "flat trend," "low recent realized vol," or "IV rich relative to
trailing realized vol" identifies good short-strangle entry conditions in
this SPY window.

## Motivation (stated plainly, then checked)

Two pieces of existing evidence pointed at this structure specifically:

1. `EQUITY_OPTIONS_REAL_DATA_RETEST.md`'s real-priced long volatility
   breakout straddle was a decisive loser (PF 0.36, -83.5%) -- the natural
   next question that doc never asked is whether SELLING the same
   structure recovers value.
2. That doc's P&L decomposition on real SPY 0DTE/1DTE trades found IV fell
   from entry to exit on average -- a real volatility-crush tailwind that
   should favor sellers, not buyers.

**Finding #2 does not transfer to this DTE regime** (see P&L decomposition
section below) -- flagged here rather than glossed over, because it's a
direct reversal of part of the original motivation.

## Method

- **Instrument**: SPY only. Real ThetaData bid/ask throughout -- sell at
  the real bid (both legs) to open, buy back at the real ask (both legs)
  to close. Never synthetic/Black-Scholes.
- **Structure**: short strangle, 5% OTM call + 5% OTM put (independent
  strikes, same expiration), 30 DTE at entry, closed at a fixed 10-day
  hold (same DTE/hold convention as every other options structure in this
  project) -- so every trade closes ~20 days before expiration, no
  settlement/early-assignment modeling needed.
- **Regime filter**: `|ema_slope_atr| < 0.20` on SPY itself, using the
  *exact* feature definition from `run_equity_real_data_walkforward.
  add_features` (EMA12-EMA36 3-bar slope, ATR-normalized) -- the inverse of
  the concept this project already uses elsewhere to detect "trending
  enough to trade." Primary threshold pre-registered at 0.20; swept
  0.10/0.15/0.20/0.30/0.40 as a sensitivity check, not a search.
- **Window**: 2021-06-01 onward, the same empirically-confirmed real-quote
  coverage window as every prior real-data retest in this project (2150
  daily SPY bars total; 1291 in-window after warmup).
- **Sizing**: net_return is on the CREDIT collected (+1.0 = both legs
  closed worthless, full credit kept). **Not floored at -1.0** -- a
  strangle that gets run over on either leg can lose several multiples of
  the credit, and that shows up here as net_return well below -1.0, not
  silently capped. P&L dollars scale off a fixed $250-credit-equivalent
  budget, the same "$250-of-$2,500" convention used in the straddle/calls
  retests -- **stated explicitly as a scaling convention, not a real
  margin/collateral model.** Real-world Reg-T/portfolio-margin requirements
  for a short strangle are not modeled at all.
- **Dedup methodology note**: candidate non-overlap is resolved on
  calendar dates alone, before any option is priced -- avoids the exact
  trap `EQUITY_OPTIONS_REAL_DATA_RETEST.md`'s tail-hedge section had to
  root-cause after the fact (a candidate silently dropped for missing data
  can't "block" a neighbor, mechanically inflating trades-taken).

## Threshold sensitivity (before any pricing)

| `\|ema_slope_atr\|` < | Candidate days | % of window | Trades after dedup |
|---|---|---|---|
| 0.10 | 124 | 9.7% | 61 |
| 0.15 | 199 | 15.5% | 67 |
| **0.20 (primary)** | **278** | **21.7%** | **76** |
| 0.30 | 407 | 31.8% | 92 |
| 0.40 | 555 | 43.4% | 102 |

A real, well-defined regime filter -- not degenerate at either extreme.

## Data-coverage gap, root-caused (not glossed over)

Of the 76 chop-gated candidates, only **30 priced (39.5%)** -- a real,
quantified gap, traced to its exact source rather than reported as a bare
number:

| Failure point | Count | % |
|---|---|---|
| Priced OK | 30 | 39.5% |
| Call entry quote missing | 36 | 47.4% |
| Put entry quote missing (call entry was OK) | 10 | 13.2% |
| Call/put exit quote missing | 0 | 0.0% |

This is a **different** failure mode than the tail-hedge section's
"strike not listed yet" -- confirmed directly: `option_history_eod` raises
`NoDataFoundError` for specific entry-day/strike/right combinations where
the *same contract* has a real quote 10 days later at exit (0% of exits
were missing). The pattern is asymmetric by construction time, not by
strike existence: a ~30-DTE, 5%-OTM SPY monthly contract appears to have
patchier day-by-day EOD quote capture at this subscription tier than a
0-1 DTE ATM contract does (which hit ~100% coverage in the European-lead
retest) -- liquidity/quote-capture for OTM monthly strikes seems to
improve as the contract ages toward the money or toward expiration, not
stay constant.

**Caveat worth stating plainly**: this coverage gap is not necessarily
independent of the regime filter. Quiet/flat-trend days may also be days
with thinner OTM options activity, meaning the 39.5% that *did* get priced
could be a non-random (busier) subset of "chop" days, not a clean sample
of the regime. Not correctable with this subscription tier; flagged as an
open question, not swept under the rug.

## Baseline result

| Metric | Value |
|---|---|
| Trades taken | 30 (of 76 candidates, rest skipped for missing quotes) |
| Win rate | 56.7% |
| Profit factor | 1.04 |
| Expectancy | 112.5 bps/trade |
| Total return (fixed $250-credit-equiv budget) | +3.4% |
| Max drawdown | 45.4% |

Barely above breakeven -- not the "chop is free money for premium
sellers" story sometimes assumed.

## Cost stress -- fragile, not robust

| Stress | Total return | Expectancy | Win rate |
|---|---|---|---|
| None (baseline) | +3.4% | +112.5 bps | 56.7% |
| +1 tick round-trip | +1.5% | +50.7 bps | 56.7% |
| +2 tick round-trip | **-0.3%** | -11.6 bps | 56.7% |
| +5 bp round-trip | **-27.8%** | -928.3 bps | 56.7% |
| +10 bp round-trip | **-62.5%** | -2084.8 bps | 53.3% |

Contrast with the SPY 0DTE calls result in `EQUITY_OPTIONS_REAL_DATA_
RETEST.md`, which "survived comfortably" through +10bp (146.6% -> 145.4%).
This strangle result flips negative by +2 ticks and is wiped out by +5bp --
because the baseline edge itself is thin (112.5bps), a strongly positive
edge tolerates cost stress that a barely-positive one cannot.

## Out-of-sample split -- does not hold up

| Half | n | Dates | Total return |
|---|---|---|---|
| First | 15 | 2021-06-17 to 2022-10-05 | **+18.3%** |
| Second | 15 | 2023-01-19 to 2026-06-18 | **-14.9%** |

Flips sign, not just decays in magnitude -- a materially worse pattern
than every long-premium structure's out-of-sample check in this project
(which stayed positive in both halves, just smaller).

## Worst 5 trades -- the uncapped-downside tail, shown plainly

| Signal date | Call/put strikes | Entry spot | Credit | Debit | net_return |
|---|---|---|---|---|---|
| 2023-11-01 | 433 / 392 | 412.42 | $6.74 | $17.91 | **-166%** |
| 2024-05-30 | 535 / 484 | 509.84 | $3.59 | $9.21 | **-157%** |
| 2023-01-19 | 392 / 354 | 373.01 | $11.63 | $24.79 | **-113%** |
| 2023-05-24 | 418 / 378 | 398.08 | $6.66 | $12.46 | -87% |
| 2021-06-17 | 409 / 370 | 389.83 | $12.38 | $22.06 | -78% |

Three of the five worst trades lost **more than 100% of the credit
collected** -- exactly the uncapped-risk profile that distinguishes a
short strangle from every long-premium structure tested in this project
(a long call/put/straddle's max loss is capped at -100% of premium paid;
this is not).

## Random-date control -- no demonstrated advantage from regime-gating

Same structure, same window, dates drawn WITHOUT the chop filter (single-
position dedup respecting draw order, not resorted -- see methodology
note below), 3 seeds, each drawing up to 76 non-overlapping dates:

| | Trades priced | Win rate | Expectancy | Total return |
|---|---|---|---|---|
| **Chop-gated (real)** | 30 | 56.7% | +112.5 bps | **+3.4%** |
| Random seed 0 | 34 | 67.6% | +1813.2 bps | **+61.7%** |
| Random seed 1 | 37 | 56.8% | +157.8 bps | **+5.8%** |
| Random seed 2 | 26 | 53.8% | +27.1 bps | **+0.7%** |

The chop-gated result sits **within or below** the spread of just 3
random draws -- two of three unconditional seeds outperform it, one
substantially (+61.7% vs. +3.4%). No evidence the regime filter adds
value; if anything it points the other way. Plausible mechanism, stated
as a hypothesis not a proven fact: this window (2021-2026) is a
persistent SPY bull market. A `\|ema_slope_atr\| < 0.20` filter selects
days where the trend has gone *flat*, which may often mean a
consolidation right before or after a volatility event -- while it
actively EXCLUDES the many calm, steadily-trending-up days that would
have been ideal for a strangle seller (small realized moves, comfortably
inside both strikes). "Flat slope" and "calm" are not the same thing in
this dataset, and conflating them was the premise this test intended to
check.

**Caveat on the control itself**: 26-37 trades is a small, noisy sample
for a structure whose P&L is dominated by tail moves (see worst-trades
table) -- the wide spread across just 3 seeds (0.7% to 61.7%) reflects
that noise as much as it reflects any real advantage of unconditional
dates. This comparison is suggestive that regime-gating doesn't help, not
proof that random dates are better.

**Methodology fix applied mid-run, disclosed**: the first version of this
control produced identical results across all 3 seeds -- traced to a real
bug, not a data issue: the dedup helper re-sorted candidates
chronologically internally regardless of input order, so shuffling before
calling it had zero effect. Fixed with a general non-overlap dedup that
respects the given (shuffled) order; verified the fix produces genuinely
different date samples per seed before trusting any downstream number.

## P&L decomposition -- reverses one of the two motivating findings

Applied `implied_greeks.decompose_option_pnl` to both legs of all 29
successfully-decomposed trades (anchored at entry Greeks, same convention
as the original P&L decomposition doc; components negated from the
function's native long-holder convention to reflect the short seller's
actual P&L):

| | Mean $ per share-pair (all trades) | Losers only | Winners only |
|---|---|---|---|
| Underlying-move P&L | -$0.87 | -$2.71 | +$0.43 |
| IV-change P&L | **-$3.88** | **-$8.47** | -$0.64 |
| Theta P&L | +$3.19 | +$3.39 | +$3.04 |
| Residual/gamma P&L | +$1.42 | +$2.04 | +$0.98 |
| **Total P&L** | **-$0.14** | -$5.75 | +$3.81 |
| Median share of \|total P&L\| | underlying 36.2% / **IV 94.2%** / theta 76.0% / residual 34.9% | | |

**Theta worked exactly as expected** (+$3.19 average tailwind, the
"reason to sell premium" argument holding up). **But IV changes were a
net DRAG, not a tailwind** (-$3.88 average, and the single largest median
share of |P&L| at 94.2% -- bigger than theta's 76.0% or the underlying
move's 36.2%). On losing trades specifically, IV-change P&L (-$8.47) was
a *bigger* drag than the underlying move itself (-$2.71) -- consistent
with the well-known dynamic where IV rises precisely when the underlying
is moving against a short-vol position, compounding losses rather than
acting as an independent, offsetting tailwind.

This directly reverses part of this study's own motivation: the SPY
0DTE/1DTE vol-crush finding (IV falling 31.1%->28.0% / 21.0%->19.9%
intraday) **does not transfer** to a ~30-DTE-at-entry, 10-day-hold
strangle in this chop-gated sample. Vol-crush appears to be a short-dated
(same-day-to-next-day) phenomenon in this data, not a general property of
selling premium at any horizon.

## Follow-up: realized-volatility filter instead of trend-slope

The trend-slope result's own "what's next" note flagged the likely reason
it failed: `|ema_slope_atr| < 0.20` measures DIRECTION (is the trend
flat), not MAGNITUDE (is the underlying actually moving little) -- and a
strangle seller only cares about magnitude. `realized_vol_20` (already
defined in this project's standard feature set) measures magnitude
directly, so it's the natural filter to re-test. Same instrument, same
30-DTE/10-day-hold structure, same real-ThetaData bid/ask discipline,
same window (2021-06-01+) -- only the regime filter changes, via a new
`--regime-filter {trend_slope,realized_vol}` flag on the same script
(`experiments/run_short_strangle_chop_backtest.py`); the trend-slope path
was re-run candidate-generation-only (no new pricing) to confirm the
refactor reproduces the original study's exact numbers (76 trades at
0.20) before trusting the new path.

**Threshold, pre-registered before pricing**: `realized_vol_20` below its
own sample median in-window (0.00824) -- mirroring the median-split
convention `EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md` already uses
for VIX/gap/range, not a threshold search. Sensitivity swept at the
25th/37.5th/50th/62.5th/75th percentiles for context only:

| Percentile | Threshold value | Candidate days | Trades after dedup |
|---|---|---|---|
| 25th | 0.00638 | 320 (25.0%) | 46 |
| 37.5th | 0.00725 | 480 (37.5%) | 64 |
| **50th (primary)** | **0.00824** | **640 (50.0%)** | **77** |
| 62.5th | 0.00933 | 800 (62.5%) | 96 |
| 75th | 0.01119 | 960 (75.0%) | 113 |

**Verdict: also does not survive -- and more decisively negative than the
trend-slope version, not an improvement.**

| Metric | Trend-slope (original) | Realized-vol (this follow-up) |
|---|---|---|
| Candidates -> priced | 76 -> 30 (39.5%) | 77 -> 24 (31.2%) |
| Win rate | 56.7% | 41.7% |
| Profit factor | 1.04 | **0.42** |
| Expectancy | +112.5 bps | **-1,709.0 bps** |
| Total return | +3.4% | **-41.0%** |
| Max drawdown | 45.4% | 46.0% |

**Cost stress**: already negative at baseline (unlike trend-slope, which
started barely positive) -- -43.5% at +1 tick, -46.0% at +2 tick, -70.6%
at +5bp, fully wiped to -100% at +10bp.

**Out-of-sample split**: both halves negative (-21.2% first half n=12,
2021-06-08 to 2023-12-06; -19.8% second half n=12, 2024-01-17 to
2026-04-29) -- more internally consistent than the trend-slope version's
sign flip, but consistently bad rather than consistently good.

**Random-date control -- the acid test, failed more clearly than before**:
3 seeds, same fixed order-respecting dedup used in the original control
(no risk of repeating the earlier dedup bug):

| | Trades priced | Win rate | Total return |
|---|---|---|---|
| **Realized-vol-gated (real)** | 24 | 41.7% | **-41.0%** |
| Random seed 0 | 35 | 65.7% | **+41.9%** |
| Random seed 1 | 38 | 57.9% | **+11.7%** |
| Random seed 2 | 26 | 53.8% | **+0.7%** |

All 3 random seeds beat the gated result, one by 83 percentage points.
This isn't "no advantage" (the trend-slope verdict) -- it's actively
**worse than random** at selecting good strangle-seller entry points.

**P&L decomposition -- reveals the mechanism, and it's adverse
selection**: mean call IV moved **24.9% -> 32.7%** from entry to exit
(IV *expanded*, not crushed) on this filtered sample. Plausible
mechanism, stated as a hypothesis: bottom-half realized-vol days are
disproportionately "vol compression" periods -- and volatility mean-
reverts, so a low-realized-vol entry point is, on average, closer to a
vol *expansion* than a vol *crush*. That's the worst possible timing for
a short-vega position: you're systematically entering right before the
thing you're short (implied vol) tends to rise. (The `iv_change_share_of
_abs_pnl`/`underlying_move_share_of_abs_pnl` ratios in the raw JSON output
inflate to values over 100% here because several trades have very small
total P&L in the denominator -- a known artifact of this per-trade-ratio
metric when total P&L clusters near zero, not a real >100%-of-P&L claim;
the entry-IV-to-exit-IV direction itself is the reliable number here, not
the share percentages.)

**Conclusion on this follow-up**: realized volatility is a more
theoretically sound proxy for "will the underlying stay in a tight
range" than trend slope, but empirically it selects for volatility
*compression* points that precede expansion -- exactly backwards for a
premium seller -- rather than genuinely calm stretches. Both regime
filters tested in this project now point the same direction: neither
"flat trend" nor "low recent realized vol" identifies good short-strangle
entry conditions in this SPY window. A real chop-specific edge, if one
exists, needs a different signal entirely (e.g. an explicit forward-vol
or options-market-implied signal, not a backward-looking price-action
proxy) -- not a threshold tweak on either of the two proxies tried so
far.

Scripts: `experiments/run_short_strangle_chop_backtest.py` (now supports
`--regime-filter {trend_slope,realized_vol}`). Outputs:
`outputs/short_strangle_chop_backtest_realized_vol/` (trend-slope's
original results remain at `outputs/short_strangle_chop_backtest/` under
its pre-refactor path).

## Follow-up 2: implied-vol-vs-realized-vol filter (the forward-looking one)

Both filters above are backward-looking price-action proxies that never
looked at what the options market itself is pricing. The natural next
idea, per each follow-up's own "what's next": a strangle seller's real
edge, if one exists, is being paid a volatility premium that exceeds what
actually gets realized -- IV rich relative to RV, not "the past was quiet"
under either definition. Filter: real ATM implied vol (solved from a real
ThetaData quote, 30-DTE convention matching the strangle itself) divided
by `realized_vol_20` (annualized, `*sqrt(252)`, since that feature is a raw
daily-return std, not already annualized). Gate: `iv_rv_ratio >= median`
(the OPPOSITE direction from the first two filters -- "rich" is a
HIGH-ratio condition) -- median pre-registered across the full 1,291-day
real-data-window population before any strangle pricing, same convention
as the realized-vol pass. Everything else (SPY, 30-DTE/10-day-hold
strangle, 2021-06-01+ window, real bid/ask discipline) held identical.

**Two real bugs caught and fixed before trusting any of this** -- both
worth recording since they'd silently corrupt any future real-options work
that reuses these primitives:

1. **Adjusted-vs-unadjusted spot mismatch.** The first implementation used
   this project's standard daily OHLCV (`data/real/SPY.csv`, dividend/
   split-adjusted per this project's `auto_adjust=True` convention) as the
   "spot" input to the Black-Scholes IV solve. Caught directly: on
   2021-06-01 the adjusted close was $390.96, but real quoted option
   prices at nearby strikes were only consistent with a true spot near
   $420 -- a ~7% error that turned a nominally "ATM" strike selection into
   something meaningfully in-the-money relative to true spot, producing a
   badly biased (in one checked case, 35.9% vs. a corrected 13.4%) IV
   solve. **Fixed via put-call parity**, entirely from real quotes with no
   external spot series at all: for the expiration nearest 30 DTE, the
   strike where the real call mid and put mid are closest together is (by
   parity, short maturity) the true ATM strike, and true spot ~= that
   strike + (call_mid - put_mid). `real_atm_iv_on_date` in
   `thetadata_pricing.py` now takes no spot argument at all -- self-
   contained and immune to this class of bug by construction.
2. **ThetaData client degrades over long sequential runs.** The first full
   attempt at fetching real ATM IV for all 1,291 candidate dates resolved
   only ~35% through its first ~550 calls, then **zero** of the next ~740
   -- a hard stall, not sparse missing data. Confirmed directly: every one
   of those "failing" dates succeeded immediately when re-tried in a fresh
   process/client (e.g. 100/100 resolved on the exact stretch, 2023-08-08
   through 2023-12-29, that had been 100% failing in the long-running
   process). Root cause not further isolated (client/terminal connection or
   resource exhaustion under sustained sequential load, not a per-date data
   issue) -- **mitigated, not fixed at the library level**, via
   `reset_client()` (forces a fresh `ThetaClient`) called proactively every
   100 dates and on any run of 10+ consecutive failures, plus a retry-once
   policy in the fetch loop. After this fix: 100% of dates resolved on the
   full rerun (up from 13.9% on the poisoned first attempt).

**Results:**

| | Baseline |
|---|---|
| Candidates gated / priced | 94 / 39 (58.5% skipped) |
| Win rate | 38.5% |
| Profit factor | 0.478 |
| Total return | **-75.2%** |
| Max drawdown | 88.0% |
| Expectancy | -1927.5 bps/trade |

**Worse than both prior filters in absolute terms** (trend-slope +3.4%,
realized-vol -41.0%, this -75.2%), though its profit factor (0.478) sits
marginally above realized-vol's (0.42) -- still a clear rejection either
way.

**Cost stress**: fragile in the same direction as the other two, faster --
+1 tick already worse (-78.3%), and +5bp/+10bp both **hard-stop the
account to -100%** (13 and 22 of 39 trades respectively get skipped by the
capital-safety floor before even completing the window).

**Out-of-sample split**: unlike realized-vol's sign-flip, this one is
consistently bad in both halves -- first half -42.9% (n=19), second half
-32.3% (n=20). Not a whipsaw; a persistent loser throughout.

**Random-date control -- the decisive comparison**: all three random seeds
were solidly POSITIVE (+51.2% n=42, +40.8% n=48, +12.0% n=32) while the
IV/RV-gated result was -75.2%. The gate sits as much as **126 percentage
points below** the best random-date control -- the widest gap of any
filter tested so far, and the clearest evidence yet that this project's
"calm" proxies (of any kind tried) are actively anti-selective for a
strangle seller in this window, not merely uninformative.

**Mechanism, via P&L decomposition (39 trades)**: mean call IV moved
**29.2% -> 38.6%** entry-to-exit; mean put IV moved 17.7% -> 20.3% --
IV EXPANDED on both legs, the same wrong-direction result as the
realized-vol filter, not the vol-crush a seller needs. Underlying-move
share of |P&L| is 91.1% (consistent with every other pass in this study --
losses are dominated by the underlying actually moving, not vol
mechanics); on losing trades, mean theta P&L is a genuine positive
contributor (+$3.44, decay collected as expected) but swamped by mean
underlying-move P&L of -$2.22.

**Why "IV rich vs. trailing RV" doesn't work as a free-money signal, most
likely**: selecting for an already-elevated IV/RV ratio plausibly selects
*for* days where the options market is forward-looking-ly pricing in a
real anticipated event (earnings, macro data, a recent vol spike not yet
fully round-tripped) rather than mispricing calm as risk. Efficient-
markets read: the market being willing to bid IV up relative to trailing
realized vol is itself informative about more volatility being likely,
not an exploitable premium -- consistent with theta being a genuine
tailwind here (decay is real) while the underlying-move risk it's priced
against also actually shows up (real, not phantom).

**Skip rate**: 58.5% (55/94), same order of magnitude as the other two
passes (60.5% trend-slope, 68.8% realized-vol) -- did not re-derive the
granular call/put-leg breakdown for this filter, flagged as not re-checked
rather than assumed identical.

**All three regime-filter ideas tried in this study are now rejected, for
three different, well-understood reasons**: trend-slope measures direction
not magnitude; realized-vol adversely selects into vol-compression points
that precede expansion; IV/RV-spread adversely selects into days the
options market is already correctly anticipating more volatility. None of
the three price-action-or-options-market proxies tested identifies
genuinely tradeable "calm" in this SPY window. If a real chop-specific
short-vol edge exists, it likely needs a fundamentally different
information source than any of these three (e.g. cross-asset confirmation,
an explicit event calendar to avoid rather than select for, or acceptance
that "calm" simply isn't identifiable this way and the strategy needs a
defined-risk structure -- e.g. an iron condor -- to make the tail
survivable regardless of regime-timing accuracy).

Scripts: `experiments/run_short_strangle_chop_backtest.py` now supports
`--regime-filter {trend_slope,realized_vol,iv_rv_spread}`. New primitives
in `src/btc_trend_bot/thetadata_pricing.py`: `real_atm_iv_on_date`
(put-call-parity ATM IV solve) and `reset_client` (client-degradation
mitigation). Outputs: `outputs/short_strangle_chop_backtest_iv_rv_spread/`
(includes `atm_iv_cache.csv`, the checkpointed real-IV fetch used to build
the gate).

## Overall verdict (trend-slope filter)

**Does not survive.** Every rigor check pointed the same direction:

- Baseline edge is thin (112.5 bps/trade) and wiped out by +2 ticks of
  extra cost, let alone +5-10bp.
- Out-of-sample split flips sign, not just shrinks.
- No demonstrated advantage over an unconditional (non-gated) short
  strangle in the same window -- the core premise (flat trend = calm =
  good for sellers) isn't supported here.
- Uncapped downside is real and demonstrated, not theoretical: 3 of the 5
  worst trades lost more than 100% of premium collected.
- One of the two motivating findings (vol-crush tailwind) reverses at
  this DTE horizon -- IV changes were a net cost, the single largest
  driver of variance across trades.
- A genuine, root-caused data-coverage gap (60.5% skip rate, concentrated
  in call-leg entry quotes specifically, not exit quotes) leaves the
  priced sample smaller and possibly non-randomly selected within the
  chop-gated pool.

**What's next, if this is revisited rather than shelved** (updated after
the realized-vol follow-up above): ~~(1) test whether a
*realized-volatility* filter is a better "calm" proxy than
trend-slope~~ -- **done, see "Follow-up" section above: also fails, more
decisively negative, and via a clean mechanism (adverse selection into
vol-expansion points, not vol-crush points)**. Both proxies tried so far
are rejected. Remaining open items: (2) a larger random-control sample
(more seeds, or price the full candidate universe once to remove the
small-sample noise in the control itself) before treating "no advantage
over random" as fully settled rather than merely unproven -- now doubly
relevant since both the realized-vol and IV/RV-spread controls showed an
even wider spread than trend-slope's; (3) if a shorter DTE structure is
tried instead, re-run the IV-change decomposition, since the vol-crush
tailwind that partly motivated the original test appears to be horizon-
specific (present in the SPY 0DTE/1DTE data), not general across DTE;
(4) ~~a genuinely different signal class entirely -- e.g. an explicit
forward-vol or options-market-implied-vol-rank signal~~ -- **done, see
"Follow-up 2" above: also fails, and more decisively than either
backward-looking proxy.** All three regime-filter ideas tried in this
project (direction, magnitude, options-market-implied richness) are now
rejected, each for a different, well-understood reason. Remaining open
avenue, if this is revisited rather than shelved: a fundamentally
different information source (cross-asset confirmation, an explicit
event calendar to avoid rather than select for) or a defined-risk
structure (iron condor) that makes the tail survivable regardless of
regime-timing accuracy, rather than another variation on "identify calm,
then sell premium."

Scripts: `experiments/run_short_strangle_chop_backtest.py` (supports
`--regime-filter {trend_slope,realized_vol}`). New pricing primitive
added to `src/btc_trend_bot/thetadata_pricing.py`:
`real_short_strangle_trade` (sell-at-bid/buy-back-at-ask, independent
call/put strikes, uncapped net_return). Outputs:
`outputs/short_strangle_chop_backtest/` (trend-slope) and
`outputs/short_strangle_chop_backtest_realized_vol/` (realized-vol).
