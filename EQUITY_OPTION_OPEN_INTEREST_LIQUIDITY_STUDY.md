# Does real open interest give advance warning of illiquid/not-yet-listed contracts?

`option_history_open_interest` exists on the ThetaData client and had never
been called anywhere in this project before this diagnostic. Two prior
real-data studies hit a real liquidity ceiling that this might help with:
the tail hedge study (69% of 45-DTE, 10%-OTM SPY puts skipped, no real
quote at the intended entry date) and the short-strangle chop studies
(60-68% of 30-DTE, 5%-OTM SPY strangle legs skipped). This is a read-only
diagnostic, not a new strategy — no backtest logic changed.

## Method note: a real confound caught before trusting results

First attempt interleaved OI fetches with pricing calls inside the same
per-cycle loop and got 0/61 tail-hedge cycles priced -- vs. the original
study's 19/61. Checked directly: cycle 1 (signal 2021-06-01) prices fine in
isolation (a real quote exists, `real_put_trade` returns a valid trade), but
failed when OI calls ran in the same loop. Root cause not fully pinned down
(most likely request-rate throttling silently swallowed by this module's
broad `except Exception: return None` convention -- consistent with the
project's existing pattern of never fabricating a value on failure, but it
means a transient throttle reads identically to "no data"). Fix: separated
into two passes -- price all 61 cycles first with no OI calls, then fetch OI
as an independent second pass. Pass 1 alone now reproduces 18/61 priced
(vs. the original study's 19/61 -- close enough to be the same result,
small residual difference not chased further since it doesn't change any
conclusion below).

## Check 1: liquid baseline

OI pulled for 5 of the SPY 0DTE call contracts from the fully-successful
European-lead real-data retest (116/116 priced originally). All 5 show
large, sane OI (9,128-15,298) as of the entry date -- confirms the API call
itself is being used correctly before trusting the harder cases below.

## Check 2: tail hedge (61 reconstructed cycles, 18 priced / 43 skipped)

| | OI@entry available | Mean OI (when available) |
|---|---|---|
| Priced trades (18) | 15/18 (83%) | 18,175 |
| Skipped trades (43) | 1/43 (2.3%) | 326 (single case) |

**OI absence tracks "no quote" almost perfectly on the skip side**: 42 of
43 skipped cycles show zero OI history as of the entry date -- not low OI,
*no* OI record at all. For several of these, `earliest_oi_date` (the first
date OI data exists at all for that contract) falls well *after* the
intended entry date -- e.g. cycle 3 (signal 2021-07-30, entry would be
2021-07-31): earliest OI record is 2021-08-11, an 11-day gap. This
independently reproduces, via a different data field, the exact mechanism
the original tail-hedge section root-caused through the contract's
`created` timestamp (a 9-day gap in that specific example) -- the strike
genuinely doesn't exist in the market yet on the signal date, confirmed
twice now through two different ThetaData fields.

**Would a pre-trade "OI > 0 as of entry date" gate have worked as a
liquidity filter?** Applied retroactively across all 61 cycles:

| | Count |
|---|---|
| Gate passes (OI>0 at entry) | 15/61 |
| Gate passes AND actually priced | 14/15 (93% precision) |
| Gate passes but trade did NOT price (false positive) | 1/61 |
| Gate FAILS but trade WAS priced anyway (false negative) | 4/61 |

**Answer: yes, real advance warning, but not a bigger sample.** 93%
precision means an OI check *before* attempting a trade would correctly
flag illiquid/not-yet-listed contracts almost every time, letting a live
system skip the wasted attempt (and the corresponding `NoDataFoundError`
handling) in advance rather than discovering failure after the fact. But it
does **not** recover more tradeable cycles than were already found --
gate-passing cycles are essentially the same 14-15 that already priced, not
a superset. The 4 false negatives (real quote existed despite OI reading 0
or missing) show OI is a stricter, imperfect signal, not a magic key to a
bigger sample. **This does not change the tail hedge study's original
"inconclusive" verdict** -- it explains *why* it's inconclusive with an
independent data field, and shows a production system could detect the
same limitation cheaply in advance, but the underlying data-tier ceiling
(deep-OTM, far-dated strikes not listed yet) is real and unchanged.

## Check 3: strangle skip spot-check (read-only, realized-vol pass)

Spot-checked 7 of 53 skipped candidates from
`outputs/short_strangle_chop_backtest_realized_vol/chop_gated_trades.csv`.
Mixed pattern, smaller sample: some legs show real OI despite the pair
failing to price (e.g. 2021-08-24: call OI=824, put OI=None -- the *put*
leg is what killed this candidate, not the call), others show no OI on
either leg. Consistent with the strangle study's own finding that skips are
concentrated on the **entry date specifically** (a timing/liquidity-capture
gap, not "strike doesn't exist yet" the way the tail hedge's far-OTM,
far-dated puts show) -- OI alone doesn't cleanly separate these cases in
this small a sample. Flagged as not fully resolved, not asserted either way
-- would need a larger spot-check to trust a pattern here.

## Caveats

- Tail-hedge reconstruction is 18/61 priced vs. the original study's
  19/61 -- a 1-trade discrepancy not chased down, doesn't affect any
  conclusion above.
- The interleaved-calls confound (see Method note) means any future
  ThetaData diagnostic in this project should default to separating pricing
  and auxiliary-data passes rather than assuming sequential calls are safe.
- Strangle skip-rate mechanism (Check 3) remains only partially explained;
  the tail-hedge mechanism (Check 2) is now confirmed via two independent
  fields.

## Bottom line

Real OI data works as advertised and gives genuine advance warning of
illiquid/not-yet-listed contracts (93% precision on the tail hedge's known
failure mode) -- worth adding as a pre-trade liquidity gate in any future
options study on far-dated/deep-OTM structures, purely to fail fast and
cheap rather than discover the same limitation after attempting real
quotes. It does not expand what's actually tradeable; the tail hedge's data-
tier ceiling is real, independently confirmed, and unchanged.

Script: `experiments/run_option_open_interest_liquidity_check.py`. Outputs:
`outputs/option_oi_liquidity_check/`.
