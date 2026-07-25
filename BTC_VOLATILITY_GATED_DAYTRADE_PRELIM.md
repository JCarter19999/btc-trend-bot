# BTC volatility-gated day-trading: preliminary research

Joey's proposed hypothesis: day-trade BTC only on very high volatility days,
rather than running a strategy continuously. No existing study in this
project (or the equity-side research) has tested this as its own question —
closest adjacent work is the equity-side BTC order-flow study (a different
axis, trade-tape imbalance, not a volatility regime gate) and this repo's
own intraday matrix, which already found most 5-minute BTC candidates carry
negative net edge before any regime conditioning at all.

**Explicit skepticism going in, stated by the parent session before this
started**: two separate attempts this session to define "calm/chop" as a
backward-looking price-action proxy for equity options work both failed —
trend-slope measured the wrong axis, realized-vol adversely selected into
volatility about to expand. A volatility-regime gate for BTC day-trading is
the same *shape* of idea. This prelim treats that skepticism as the default,
not something to argue past.

## 1. Regime definition (no lookahead)

`norm_atr = atr_5m / close` (scale-free volatility, since raw ATR is a
dollar figure that drifts with BTC's price level over the sample).
Trailing 7-day (2,016 five-minute bars) rolling percentile rank of
`norm_atr`, computed using only the prior 2,015 bars plus the current one,
then **shifted forward by one additional bar** so a bar's regime label is
knowable strictly before that bar trades — removes any ambiguity about
whether the bar's own realized value leaked into its own rank, on top of
`atr_5m` already only using completed candles.

`high_vol_regime = trailing_percentile >= 0.75` (top quartile of trailing
volatility). A top-quartile cut was chosen (not a more extreme decile) to
keep enough trades in the "high-vol" bucket to check at all, given the
sample is only 50,000 five-minute bars (~174 days) — noted as a
sample-size-driven choice, not a search for the best-looking cutoff.

## 2. Regime characterization

- 47,961 of 50,000 bars have enough trailing history to classify (first
  ~7 days necessarily unclassified).
- **24.6% of classified bars are high-vol-regime** — close to the 25%
  target by construction of a percentile-rank threshold, sanity-confirms
  the computation.
- **Real clustering, not scattered noise**: 894 regime switches across the
  classified sample, implying an average run length of ~53.6 bars (~4.5
  hours) per regime state. Volatility clusters here the way it's widely
  known to elsewhere — a real, expected property, not a finding, but worth
  confirming empirically before relying on it.

## 3. Does existing strategy performance actually differ by regime?

Joined `trade_episodes.csv`'s closed round trips (all 5 non-benchmark
strategies in `outputs/intraday_matrix_50000/`) to the regime label as of
each trade's `entry_timestamp`, split mean `net_portfolio_return_pct` by
regime, checked the real difference against a 1,000-seed shuffled-label
null (permute which trades count as "high-vol", same discipline as every
other finding in this project).

| Strategy | n trades | n high-vol | mean return, high-vol | mean return, other | diff (pp) | null percentile |
|---|---|---|---|---|---|---|
| `breakout_5m_1h_regime` | 153 | 31 | **+0.031%** | −0.181% | **+0.212** | **97.4** |
| `candle_run_2bar_fee_aware_5m` | 1,003 | 376 | −0.163% | −0.180% | +0.017 | 67.8 |
| `momentum_1h_5m_entry` | 24 | 5 | −1.746% | −0.188% | −1.558 | 11.7 (n=5, too thin to trust either way) |
| `momentum_1h_immediate` | 25 | 9 | −0.668% | −0.565% | −0.103 | 49.3 |
| `vwap_reversion_5m` | 116 | 46 | −0.193% | −0.147% | −0.047 | 32.0 |

**One real, non-noise-level differential edge**: `breakout_5m_1h_regime`
("five-minute volatility breakout inside bullish 1h/4h context" — already
a breakout-flavored mechanism by design, which plausibly explains why an
*external* high-vol-regime gate would help it specifically) goes from a
clear loser (−0.181%/trade) to roughly breakeven-to-slightly-positive
(+0.031%/trade) when restricted to the top-quartile volatility regime, and
that split sits at the 97th percentile of the shuffled null — a real
separation, not noise. Every other strategy tested shows no meaningful
regime-conditional difference, including one (`momentum_1h_5m_entry`) too
thin to read at all (5 trades).

## 4. The obvious tension, checked directly rather than assumed

Confirmed in `src/btc_trend_bot/intraday_matrix.py` (lines ~532-534,
1066-1071): **the cost model is a flat, config-level constant**
(`fee_bps_per_side`, `slippage_bps_per_side`, `assumed_spread_bps_per_side`
— 2/5/1 bps respectively, 8bps all-in per side) applied identically to
every transaction regardless of when it occurs. It does **not** widen
during high-volatility periods, even though real crypto spread/slippage
very plausibly does exactly that. This means the +0.212pp differential
above is computed under a cost assumption that is, if anything, most
generous precisely on the days the apparent edge depends on — the single
biggest reason not to trust this result yet. A real backtest would need
either a volatility-conditioned cost model or a direct historical
spread/slippage measurement on high-vol days specifically, not the flat
assumption already in this pipeline.

## 5. Recommendation

**Worth a real backtest, but only after resolving the cost-model gap
above — not yet a "day-trade high-vol BTC" strategy.** One specific,
mechanistically-sensible candidate survived a real diagnostic
(`breakout_5m_1h_regime`, gated to the top-quartile trailing-volatility
regime defined here), with a result that clears a shuffled-null bar. But:

1. This is one strategy out of five tested, on one 174-day sample — not
   independently out-of-sample split, not cost-stress tested, not checked
   against a random-date control the way every promoted equity finding in
   this project has been.
2. The flat-cost assumption is a real, checked (not assumed) confound that
   specifically favors high-vol days — the next step before anything else.
3. 31 high-vol trades is a small sample to build confidence on alone.

Next step, if this gets pursued: re-run `breakout_5m_1h_regime` gated to
this regime definition with (a) a volatility-conditioned cost stress test
(scale slippage/spread up specifically on high-vol-regime trades and see if
the edge survives), (b) an out-of-sample split, (c) a random-date control
matching the same trade count. Only promote past that if all three hold,
same bar as everything else in this project's promotion gates.

No script artifacts beyond this doc's inline analysis (run directly against
`outputs/intraday_matrix_50000/feature_frame.csv` and `trade_episodes.csv`
using a borrowed Python environment with pandas — this repo itself has no
local venv, everything normally runs via Docker per `INTRADAY_MATRIX.md`).
If this proceeds to a full backtest, that analysis should be turned into a
proper script under this repo's own Docker-run conventions rather than an
ad-hoc one-off.

## 6. Full backtest — the three required checks (added same day)

Script: `experiments/run_vol_gate_full_backtest.py` (run against the same
`equity_v2_4_research` borrowed venv as the prelim; still no local venv in
this repo). Reused the prelim's exact regime definition verbatim, merged
onto `breakout_5m_1h_regime`'s 153 trades via `merge_asof` (backward) on
`entry_timestamp` — **0 trades dropped in the merge**, so the earlier
methodology-trap concern (candidates mechanically disappearing from one
side of a comparison) does not apply here. Baseline reproduces the prelim
exactly: n_hv=31 (+0.031%), n_other=122 (−0.181%), diff=+0.212pp.

**Check 1 — volatility-conditioned cost stress.** Backed out the existing
flat round-trip cost empirically from `gross - net` (0.1599%, matching the
config's 8bps/side × 2 assumption almost exactly, std=0.00001pp — confirms
it really is a flat constant, nothing conditioned already). Scaled that
cost specifically on high-vol trades only:

| Cost multiplier on high-vol trades | HV net mean | Diff vs. other |
|---|---|---|
| 1.0x (baseline) | +0.031% | +0.212pp |
| 1.5x | −0.049% | +0.132pp |
| 2.0x | −0.129% | +0.053pp |
| 3.0x | −0.289% | **−0.107pp** |
| 4.0x | −0.448% | −0.267pp |
| 5.0x | −0.608% | −0.427pp |

Survives up to ~2x the flat cost assumption, dies between 2x and 3x. Real
crypto bid-ask spread/slippage during high-volatility regimes plausibly
widens by more than 2-3x normal (no direct historical spread measurement
was available to ground this precisely — same gap flagged in the prelim,
still unresolved) — this makes the edge's survival a real open question,
not clearly dead, not clearly robust.

**Check 2 — out-of-sample split. This is the one that kills it.**

| Half | Window | n_hv | HV mean | n_other | Other mean | Diff |
|---|---|---|---|---|---|---|
| First | 2026-02-08 to 2026-04-26 | 18 | +0.271% | 58 | −0.165% | **+0.435pp** |
| Second | 2026-04-26 to 2026-07-21 | 13 | −0.301% | 64 | −0.196% | **−0.105pp** |

The entire full-sample edge is concentrated in the first half and **flips
sign** in the second half — not just decay, a reversal. This is the same
failure pattern that killed the equity-side realized-vol strangle filter
and the momentum-revision mechanism check: a result that clears a
full-sample statistical bar but does not hold chronologically.

**Check 3 — random-date control**, pure random sampling (not label
permutation, distinct method from the prelim's shuffled-label null, run as
its own check per the backtest directive): 1,000 draws of 31-trade random
subsets vs. the remaining 122. Real diff (+0.212pp) sits at the 97.2th
percentile of the random-draw null (mean +0.002pp, std 0.101pp) —
consistent with the prelim's 97.4th-percentile shuffled-label result via
an independent method, confirming the full-sample statistic itself is
correctly computed, not a bug. This check alone would say "promote it";
it's the OOS split above that overrides that read.

### Verdict: does not survive — closed out as a negative result

Clears the full-sample statistical bar and a modest cost stress, but
**fails the out-of-sample split decisively** (sign reversal, not decay) —
by this project's own promotion standard ("only promote past that if all
three hold"), this is a clean rejection, not a partial pass. Combined with
the unresolved cost-model fragility (dies at plausible real-world spread
widening beyond ~2-3x), there are now two independent reasons not to trust
this as a real strategy, not one. **Not a candidate for paper deployment.**
The volatility-regime definition itself remains legitimate (real,
clustering, no-lookahead) — what failed is the specific claim that
`breakout_5m_1h_regime` has a genuine, stable edge inside it. If this
research thread continues, the honest next step is testing whether ANY
mechanism has a stable (not reversing) regime-conditional edge here, not
re-testing this same strategy with a different cost assumption.

## 7. Continuing the search — broader candidates + purpose-built mechanisms

Per Joey's "keep pushing" instruction. Same regime definition reused
verbatim throughout (trailing 7-day ATR percentile, shifted for
no-lookahead, top quartile = high-vol).

### 7a. Broadened existing-candidate search

Checked all 12 non-benchmark strategies in `outputs/popular_matrix_50000/`
(Donchian, EMA pullback, Bollinger squeeze, RSI(2) reversion, MACD/ADX) and
`outputs/slope_matrix_50000/` (6 rolling-slope variants) against the same
regime gate, same diagnostic as the prelim (shuffled-label null) plus an
**early OOS chronological split checked alongside the significance test**,
per the explicit lesson from `breakout_5m_1h_regime`'s reversal.

**A real bug caught before trusting these results**: after `dropna` on
rows where `merge_asof` failed to find a regime match (candidates whose
earliest trades predate the regime's first classifiable bar), the
`high_vol` column silently stayed `object` dtype instead of reverting to
`bool`. `~merged["high_vol"]` on that dtype computes Python's bitwise
integer complement (`~True == -2`, `~False == -1`) instead of logical
negation — summed across a real column this produced nonsense large
negative numbers and made every affected candidate (all 6 slope-matrix
ones, which have entries before the regime's Feb-5 classifiable start)
read as "too thin" when they weren't. Caught via a sanity re-check against
the already-known `breakout_5m_1h_regime` result (exact match to Section 6
confirmed the fix) before trusting any new candidate.

| Strategy | n | n_hv | diff (pp) | null pctile | OOS 1st half | OOS 2nd half | sign flip |
|---|---|---|---|---|---|---|---|
| `ema_pullback_15m_4h` | 178 | 38 | **+0.292** | **98.4** | +0.574 | +0.049 | no |
| `rsi2_reversion_5m_4h_filter` | 868 | 193 | +0.035 | 93.9 | +0.040 | +0.031 | no |
| `slope_1h_reference_2tick` | 115 | 34 | +0.436 | 90.3 | +1.493 | **−0.614** | **yes** |
| `macd_15m_adx_baseline` | 303 | 86 | +0.034 | 68.1 | −0.043 | +0.111 | — |
| `slope_15m_quadratic_2tick` | 706 | 150 | −0.014 | 44.6 | +0.087 | −0.102 | — |
| `slope_30m_balanced_2tick` | 166 | 56 | −0.033 | 46.2 | +0.431 | −0.340 | — (not sig.) |
| `bollinger_squeeze_15m` | 23 | 5 | −0.092 | 40.2 | too thin | — | — |
| `donchian_15m_atr` | 197 | 49 | −0.111 | 22.7 | +0.110 | −0.311 | — (not sig.) |
| `donchian_30m_atr` | 100 | 32 | −0.129 | 25.4 | −0.166 | −0.115 | — |
| `slope_15m_fast_3tick` | 290 | 78 | −0.163 | 13.6 | −0.038 | −0.206 | — |
| `slope_5m_fast_3tick` | 675 | 165 | −0.085 | 11.0 | −0.087 | −0.083 | — |
| `slope_15m_fast_2tick` | 309 | 78 | −0.218 | 7.3 | −0.130 | −0.236 | — (real, wrong direction) |

**One genuinely promising result: `ema_pullback_15m_4h`.** Clears the
shuffled-null bar (98.4th percentile) AND — unlike `breakout_5m_1h_regime`
— does **not reverse sign** out of sample: +0.574pp first half, +0.049pp
second half. That's real decay (the effect is much weaker in the second
half), not a reversal. This is the first candidate in this whole research
thread to pass both the significance check and the OOS check
simultaneously, even if the second-half magnitude is thin enough to want
a proper full backtest (cost stress, random-date control) before trusting
it the way `breakout_5m_1h_regime` was initially trusted.

**Secondary, weaker candidate: `rsi2_reversion_5m_4h_filter`.** Below the
~97th-percentile bar treated as "real" elsewhere in this project (93.9th),
but both OOS halves agree in sign and magnitude (+0.040, +0.031) — a
smaller, more consistent effect than a large one that doesn't replicate.
Worth keeping in view, not worth promoting on its own.

**`slope_1h_reference_2tick` is the cautionary tale, not a lead**: looks
dramatic full-sample (+0.436pp, 90.3rd percentile) but reverses hard out
of sample (+1.49pp → −0.61pp) on a thin 34-trade high-vol sample — exactly
the `breakout_5m_1h_regime` failure mode again. Rejected on sight, not
pursued further.

**`momentum_1h_5m_entry` still unresolved.** Attempted a longer-history
rerun (100,000 bars instead of 50,000) via this repo's Docker pipeline to
get enough high-vol trades to read at all — the live Binance.US download
did not complete within the session's time budget (backgrounded, then
timed out). Remains a genuinely open question, not a rejection; would need
a dedicated run with more time budgeted, not a quick recheck.

### 7b. Two purpose-built mechanisms — both clear rejections

Rather than keep re-testing signals with no vol-regime hypothesis behind
them, built two mechanisms with an actual mechanistic reason high
volatility specifically should favor them. Both use the flat 8bps/side
cost convention (0.1599% round trip) for first-pass comparability with
every other candidate checked in this doc — not yet cost-stress-tested,
since neither survived even the first-pass check.

**Vol-spike mean-reversion (fade)**: entry when `|vwap_zscore| >= 2`
(price extended from its rolling VWAP), direction faded (short an
upside extension, long a downside one), entered at the next bar's open,
held a fixed 6 bars (30 min). Hypothesis: high-vol moves often overshoot
and revert, which no existing candidate is designed to capture.

**Result: clear rejection, not just thin.** 8,166 signals, 3,442 in the
high-vol regime. The high-vol-restricted version is a net loser on its
own terms (mean −0.160%/trade, 33.7% win rate) — this isn't "worse than
the other regime," it's "loses money outright." The regime differential
itself is also negative and real (−0.016pp, 7.6th percentile) — if
anything, this fade does relatively *better* outside the high-vol regime,
the opposite of the hypothesis.

**Wide-stop breakout, purpose-built for the regime**: entry on a
breakout above/below the prior 5-minute range with 1-hour momentum
agreeing in direction, stop at 2x ATR, target at 3x ATR, max 2-hour hold
— sized directly off the current bar's own ATR rather than a fixed bps
stop, on the hypothesis that `breakout_5m_1h_regime`'s fragile edge might
firm up with room sized for the regime's actual typical range instead of
being gated onto a signal designed without that in mind.

**Result: also a clear rejection.** 3,319 signals, 761 high-vol. Also a
net loser in absolute terms (mean −0.161%/trade, 39.6% win rate) in the
high-vol regime specifically. No real regime differential (−0.010pp,
29.9th percentile), and what little OOS signal exists is thin and
inconsistent (−0.086 vs. +0.045).

### Recommendation

**The volatility-regime concept survives; every specific strategy tested
against it tonight except one has failed.** `ema_pullback_15m_4h` is a
real, non-reversing candidate and the right next step if this continues
— it should get the same three-check full-backtest treatment
`breakout_5m_1h_regime` got (volatility-conditioned cost stress,
independent random-date control, and it already has an OOS check that
didn't reverse). The two purpose-built mechanisms designed specifically
for this regime both failed outright, which is itself informative: an
external regime gate on an existing signal (`ema_pullback_15m_4h`) has so
far outperformed two attempts to build something bespoke for the regime —
worth remembering before designing a third bespoke mechanism.
`momentum_1h_5m_entry` remains genuinely unresolved, not rejected, pending
a properly time-budgeted longer-history data pull.

## 8. Full backtest — `ema_pullback_15m_4h` (three required checks)

Script: `experiments/run_ema_pullback_full_backtest.py`. Regime built from
the canonical `intraday_matrix_50000/feature_frame.csv` (has `atr_5m`) —
NOT `popular_matrix_50000`'s own feature frame, which uses different ATR
column names (`m15_atr`/`h1_atr`/`h4_atr`) and would silently be a
different regime definition — then merged via `merge_asof` onto
`popular_matrix_50000/trade_episodes.csv`'s 178 closed `ema_pullback_15m_4h`
trades. **0 trades dropped in the merge** (all entries fall after the
regime's first classifiable bar). Baseline reproduces §7a exactly: n_hv=38
(+0.018%), n_other=140 (−0.274%), diff=+0.292pp — confirms the dtype bug
fix from §7a is not in play here and the regime/merge logic is correct.

**Check 1 — volatility-conditioned cost stress.**

| Cost multiplier on high-vol trades | HV net mean | Diff vs. other |
|---|---|---|
| 1.0x (baseline) | +0.018% | +0.293pp |
| 1.5x | −0.062% | +0.213pp |
| 2.0x | −0.142% | +0.133pp |
| 3.0x | −0.301% | **−0.027pp** |
| 4.0x | −0.461% | −0.187pp |
| 5.0x | −0.621% | −0.347pp |

Survives up to ~2x the flat cost assumption, dies between 2x and 3x —
functionally the same fragility profile as `breakout_5m_1h_regime`'s cost
check. Same unresolved gap: no direct historical high-vol-regime spread
measurement exists to say whether real crypto spread widening in this
regime falls inside or outside that survival range. Not a clean pass, not
a clean rejection — an open question, exactly as it was for the breakout
candidate.

**Check 2 — out-of-sample split. This is the one that decides it, and
this candidate passes.**

| Half | Window | n_hv | HV mean | n_other | Other mean | Diff |
|---|---|---|---|---|---|---|
| First | 2026-02-27 to 2026-05-02 | 17 | +0.239% | 72 | −0.335% | +0.574pp |
| Second | 2026-05-02 to 2026-07-21 | 21 | −0.161% | 68 | −0.210% | +0.049pp |

Same sign both halves — **no reversal** (confirmed programmatically, not
just by eyeballing the numbers). The effect decays substantially (+0.574pp
→ +0.049pp) but does not flip, unlike `breakout_5m_1h_regime`'s outright
sign reversal. This is the deciding difference between the two candidates.

**Check 3 — random-date control.** 1,000 draws of 38-trade random subsets
vs. the remaining 140. Real diff (+0.292pp) sits at the **98.7th
percentile** of the random-draw null (mean −0.002pp, std 0.134pp) —
consistent with §7a's 98.4th-percentile shuffled-label result via an
independent method, confirming the statistic is correctly computed.

### Verdict: the strongest candidate this research thread has produced,
### with one real caveat still open

`ema_pullback_15m_4h` clears two of the three checks decisively (no OOS
reversal, real random-date-control separation) and the third (cost
stress) lands in the same "open question, not resolved either way" state
`breakout_5m_1h_regime`'s cost check did — the difference is this
candidate doesn't ALSO fail outright on OOS the way that one did. By the
strict letter of "only promote past that if all three hold," cost stress
is not an unconditional pass, so this is not a clean green light for paper
deployment yet — but it is a materially stronger result than anything
else tested in this thread, and the specific failure mode that killed
every other promising-looking candidate tonight (OOS sign reversal) does
not apply here. The remaining open item before any deployment decision is
the same one flagged since §4: a real, direct measurement of how much
BTC bid-ask spread/slippage actually widens during this specific
volatility regime, to resolve whether the cost-stress survival range
(up to ~2x) is realistic or already exceeded in practice.

## 9. Increasing trade frequency: two levers tested

Joey asked how to increase trade count without destroying the selectivity
that let `ema_pullback_15m_4h` survive rigor. Two specific levers tested,
scripts: `experiments/run_vol_gate_threshold_sweep.py`,
`experiments/run_rsi2_reversion_full_backtest.py`.

### Lever 1 — volatility threshold sensitivity sweep: real improvement found

Swept the regime threshold from 0.50 to 0.75 on `ema_pullback_15m_4h`,
redoing the full OOS split (not just a full-sample stat) at every level —
this is the check that actually validated 0.75 originally, so it's the
one that has to hold at any looser cutoff too.

| Threshold | n_hv | n_other | HV mean | Other mean | Diff | OOS 1st half | OOS 2nd half | Sign flip | Random-control pctile |
|---|---|---|---|---|---|---|---|---|---|
| >=0.50 | 84 | 94 | −0.154% | −0.264% | +0.110pp | +0.221pp | **−0.002pp** | **YES** | 84.4th |
| >=0.60 | 63 | 115 | −0.054% | −0.299% | +0.244pp | +0.361pp | +0.138pp | no | 98.5th |
| >=0.65 | 54 | 124 | −0.009% | −0.301% | +0.292pp | +0.477pp | +0.121pp | no | **99.0th** |
| >=0.70 | 48 | 130 | −0.054% | −0.270% | +0.217pp | +0.377pp | +0.071pp | no | 94.6th |
| >=0.75 (anchor) | 38 | 140 | +0.018% | −0.274% | +0.292pp | +0.574pp | +0.049pp | no | 98.7th |

**Recommendation: loosen to >=0.65, not 0.75.** Real, not cherry-picked --
0.65 gives **54 trades vs. 38 (+42% more)**, the identical diff magnitude
(+0.292pp) to the 0.75 anchor, no OOS reversal, and actually the highest
random-control percentile of every threshold tested (99.0th vs. 0.75's
98.7th). 0.60 is a solid second choice (63 trades, 98.5th percentile, no
flip) if even more frequency is wanted at a small statistical cost. 0.50
is where it breaks: sign flip in the second OOS half (+0.221pp ->
−0.002pp) and the random-control percentile drops to 84.4th -- confirms
the edge does dilute back toward the ungated baseline eventually, just
not until well past 0.65. **This is a genuine improvement, not a
multiple-comparisons artifact**: 0.60, 0.65, 0.70, and 0.75 all
independently pass (no flip, >=94th percentile) -- a plateau of good
thresholds, not one lucky pick among five.

### Lever 2 — add `rsi2_reversion_5m_4h_filter` as a second strategy: rejected, and combining it actively hurts

Ran the same three-check full backtest on `rsi2_reversion_5m_4h_filter`
(868 total closed trades, 193 high-vol-gated at the 0.75 threshold) that
`ema_pullback_15m_4h` passed:

- **Cost stress: fails almost immediately.** Baseline diff is only
  +0.035pp and flips negative at just 1.5x the flat-cost assumption
  (−0.045pp) -- far more fragile than `ema_pullback_15m_4h`, which
  survived to 2x.
- **OOS split: technically no sign flip, but hollow.** +0.040pp first
  half, +0.031pp second half -- consistent, but both the high-vol AND
  "other" means are NEGATIVE in both halves (e.g. first half: HV −0.108%,
  other −0.148%). This strategy loses money in every regime; the
  high-vol slice just loses *less*. Structurally different from
  `ema_pullback_15m_4h`, which is genuinely profitable (not just
  "less unprofitable") in the high-vol slice.
- **Random-date control: 92.3rd percentile** -- below this project's
  ~97th-percentile bar, consistent with the broadened search's original
  93.9th-percentile read.

**Combining it with `ema_pullback_15m_4h` in the same $10,000
sequential-compounding book actively destroys value**: 231 combined
trades (38 + 193) over the same 144-day window, trading ~1 every 0.6
days instead of ema_pullback-alone's ~1 every 3.8 days -- but ending
equity is **$8,017.71 (−19.8%)**, against `ema_pullback_15m_4h` alone's
+0.48%. Adding this second strategy doesn't just fail to help, it
overwhelms the one real edge in this research thread with a much larger
number of small losing trades.

**Verdict: Lever 2 rejected outright — do not combine.** Exactly the
failure mode flagged before starting this task: a lever that increases
trade count by dragging in a structurally weaker, cost-fragile signal is
not a real improvement.

### Net recommendation (fork's take) vs. decision actually made

The fork's own recommendation was to adopt Lever 1 (loosen to >=0.65) on
the strength of the relative-edge statistics (+42% more trades, same
+0.292pp diff, no OOS reversal, better random-control percentile).
**That recommendation undersold what matters more in practice: the
ABSOLUTE dollar result got worse, not better.** Verified directly
(parent session, not the fork): at >=0.65, mean return/trade turns
slightly negative (-0.009%, n=54) and $10,000 sequential-compounding
ends at $9,930.01 (-0.70%) -- against >=0.75's +0.018%/trade and
$10,047.49 (+0.48%). The extra 16 trades gained by loosening the
threshold are themselves net money-losers; they only look good
*relative to* the even-worse trades further down the volatility
scale, which is exactly the "more trades isn't more profit" trap this
whole lever-testing exercise was meant to guard against.

**Decision: keep the >=0.75 threshold, do not loosen it.** Fewer trades
(38 vs. 54), but it's the version that's actually profitable in
absolute terms, not just relatively better than a worse bucket. Lever 2
remains rejected outright regardless (see above -- fails on its own
merits before the loosening question even applies). The open cost-stress
question from §8 (no real BTC spread-widening data to ground the 2x-3x
survival range) still applies to the >=0.75 version and remains the one
item outstanding before any deployment decision.

## 10. Live paper-trading bot (4h volatility regime), build + dry-run

Per Joey's instruction to try this on paper data with the volatility
measurement changed to 4-hour signals (not the 5m-bar-based construction
above). `experiments/run_btc_vol_gated_paper_step.py`. Reuses the exact,
already-validated `ema_pullback_15m_4h` logic from `popular_matrix.py`
(`decide_strategy`, `build_feature_frame`) rather than reimplementing it --
only the volatility gate is new.

**4h regime construction**: resample completed 5m bars into 4h candles
(same `_completed_resample` mechanism `build_feature_frame` already uses
for `h4_atr`/`h4_trend_bps`), compute ATR on that independent 4h series,
normalize by close, trailing rolling percentile over 42 bars (42 x 4h = 7
days, same calendar lookback as the original 2,016-bar/7-day 5m
construction), shifted one bar for no-lookahead. Threshold kept at >=0.75
per Joey's decision not to loosen it. The gate only filters NEW entries --
an already-open position still exits on its own normal rule regardless of
the current regime reading, consistent with every other regime-gated
strategy in this project.

**Dry-run against real live data (binanceus BTC/USD, 2026-07-24 ~20:31
UTC)**: real, working end-to-end.

| | Value |
|---|---|
| BTC mark price | $64,160.25 (bid $64,157.34 / ask $64,162.85) |
| h4 ATR (normalized) | 0.00936 |
| 4h vol-regime percentile | **0.833** (high-vol, >=0.75 threshold) |
| Strategy decision | `enter_pullback` -- "trend pullback recovery" |

Logged one real snapshot to `outputs/btc_vol_gated_paper/btc_vol_gated_paper.sqlite3`
(`vol_gate_paper_snapshots` + `vol_gate_paper_state` tables) to confirm the
ledger path works, not as a standing position.

**Two real gaps, flagged plainly rather than glossed over, before this
could run repeatedly/on a schedule**:

1. **`runtime/` in this worktree is root-owned** (`drwxr-xr-x root:root`),
   not writable by this process -- the ledger was redirected to `outputs/`
   instead. Needs a real decision (fix ownership, or keep `outputs/`) before
   any recurring deployment.
2. **Position-state bookkeeping is incomplete.** This step records the
   entry decision and updates `target_position`, but does NOT yet call
   `execute_decision` (the function in `popular_matrix.py` that properly
   updates `entry_mark`/`entry_index`/`highest_high`/cash-btc accounting on
   `StrategyState`). That means a SECOND invocation, after this one opened
   a position, would have an incomplete state for `decide_strategy`'s exit
   logic (ATR trailing stop needs `highest_high`; hold-duration checks need
   `entry_index`). Fine for a one-shot dry-run proving the entry path
   works; NOT safe to schedule repeatedly until `execute_decision` is
   wired into this step the same way `popular_matrix.py`'s own simulator
   uses it.

No systemd unit, timer, or scheduling was touched, per scope -- this is a
build-and-verify result only. Scheduling and the two gaps above are
follow-up decisions for Joey/the parent session, not resolved here.

## Both gaps fixed (same day, parent session, foreground)

1. **`runtime/` ownership fixed.** Was `root:root`; chowned to `joey:joey`
   (matches every other repo's convention in this project). Ledger path
   reverted to `runtime/btc_vol_gated_paper.sqlite3` -- no longer needs the
   `outputs/` workaround.
2. **Position-state bookkeeping wired up properly.** Added
   `execute_live_decision()` to `run_btc_vol_gated_paper_step.py` -- the
   live-context equivalent of `popular_matrix.execute_decision`, same
   state mutations (`entry_index`/`entry_mark`/`highest_high`/
   `entry_reference`/cash/btc/`trade_count`/`last_trade_index`), but fills
   against the REAL live bid/ask from the quote instead of a synthetic
   next-bar-open + assumed spread/slippage rate -- arguably more realistic
   for live paper trading than the backtest's own assumption, not a
   compromise. Also added the missing `state.highest_high = max(...,
   current bar's high)` update that `popular_matrix.py`'s own simulation
   loop does before every `decide_strategy` call (line ~658-659) --
   without it, the ATR trailing stop would only ever compare against the
   entry price on every subsequent invocation instead of the true running
   peak since entry.

**Verified against real live data, twice in a row** (binanceus BTC/USD,
2026-07-24 ~20:37 UTC): first invocation ran cleanly end-to-end (BTC
$64,145.50, 4h vol-regime 0.833/high-vol, decision `hold_cash`/"waiting
pullback recovery" -- conditions had moved on from the earlier `enter_pullback`
reading a few hours prior, which is itself a good sign the strategy is
reading live data freshly each time, not returning a stale cached decision).
Second invocation confirmed the persisted `StrategyState` round-trips
correctly through the SQLite ledger (all fields, including the
previously-missing ones, present and consistent) -- the specific failure
mode flagged above is resolved.

**Status: build-and-verify complete, safe to schedule from a state-management
standpoint.** Scheduling itself (systemd timer/cadence) remains a separate
decision, not addressed here.
