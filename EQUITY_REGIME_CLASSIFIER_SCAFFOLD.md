# Regime classifier scaffold — first piece of a strategy-rotation system

Joey's idea: several strategies exist in this project (`simple_trend`
momentum rotation, the DAX/European-lead SPY options overlay, a short
strangle for chop, backtested separately) that each work under different
market conditions. Eventually: a backtest that rotates between them as the
market's classified regime changes. This doc is step one only — building and
validating the classifier itself. **No rotation/dispatch backtest, no
capital allocation, no strangle logic here.** Those come after the parallel
short-strangle backtest reports its own numbers and becomes the third arm.

Module: `src/btc_trend_bot/regime_classifier.py`. Diagnostic:
`experiments/run_regime_classifier_diagnostic.py`. Outputs:
`outputs/regime_classifier_scaffold/`.

## Schema

Two independent boolean flags, not one mutually-exclusive label:

- **`trending`** — the live safety layer's own trend gate (`return_26 >
  -1%` and `ema_slope_atr > -0.20`, verbatim from
  `experiments/run_equity_regime_stress_final.py`'s `trend_ok` predicate),
  evaluated on whichever symbol `simple_trend`'s selector (top
  `relative_strength_20` among symbols above their 50-day benchmark trend)
  would actually pick that day. Full 2018–2026 coverage (cached daily data).
- **`dax_signal_day`** — the validated DAX pre-US-open top-quartile |move|
  gate, AND-conditioned on the three calm-regime state variables from
  `EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md` sitting at/below their
  established sample medians (prior-day VIX close ≤17.22, overnight gap
  ≤0.44%, DAX session range ≤1.40%). Coverage limited to ~2023-09 onward
  (yfinance intraday-history limit).

They're independent flags, not exclusive states, because they describe
different things at different timeframes: `trending` is a multi-day regime
gating a 10-day-hold stock rotation across AAPL/MSFT/NVDA/TSLA;
`dax_signal_day` is a same-day intraday event trigger for a SPY 0DTE/1DTE
options trade. Both can and do fire on the same calendar date. A convenience
`primary_bucket` column collapses this to one label per day for reporting,
with an explicit `TRENDING+DAX_SIGNAL_DAY` bucket for the overlap rather
than silently picking a precedence order.

**`CHOPPY_SIDELINED`** is a placeholder: neither flag fires. This is where
the short strangle is intended to plug in once its own backtest lands.

**Caveat stated up front, not buried**: the `dax_signal_day` AND-combination
(quartile gate + all three calm conditions simultaneously) is itself a *new*
hypothesis. The real-ThetaData options results in
`EQUITY_OPTIONS_REAL_DATA_RETEST.md` (+146.6%/+88.6%) were measured on the
raw 116-day quartile gate, not this narrower 27-day subset. What's validated
below is that `dax_signal_day` is constructed with no lookahead and is a
proper subset of those 116 days — not that its real-money options
profitability has been independently re-tested.

## No-lookahead check

Every input is knowable at or before the classification date: `return_26`/
`ema_slope_atr` are `pct_change`/`ewm().diff()` — backward-looking by
construction. VIX uses the **prior** trading day's close (not same-day).
Gap uses today's open vs. yesterday's close (known at open). DAX session
range uses only the pre-13:30-UTC morning session. Classifying date `t`
labels the signal date only — it doesn't change or bypass this project's
existing next-bar-entry convention for any strategy that trades on that
label.

## Coverage / base rates

Full 2018–2026 calendar (2,150 days):

| Bucket | Days | % |
|---|---|---|
| TRENDING | 1,672 | 77.8% |
| CHOPPY_SIDELINED | 451 | 21.0% |
| TRENDING+DAX_SIGNAL_DAY | 24 | 1.1% |
| DAX_SIGNAL_DAY (alone) | 3 | 0.1% |

Restricted to the window where `dax_signal_day` actually has data
(2023-09-06 → 2026-07-23, 462 days): TRENDING 333, CHOPPY_SIDELINED 102,
overlap 24, DAX-alone 3 — same shape, confirming the full-history base
rates aren't distorted by the shorter DAX window. CHOPPY_SIDELINED at ~21%
of the calendar is a meaningful chunk, not a sliver — there's a real amount
of trading days for a chop strategy to fill.

**Reasonable, not degenerate**: no bucket swallows everything, and the
placeholder bucket is big enough to matter for the eventual rotation
backtest.

## Check 1 — does `trending` actually track `simple_trend`'s real performance?

Ran the real walk-forward `simple_trend` selection over cached 2018–2026
daily data, joined each selected trade's signal date to the classifier's
`trending` flag, compared mean per-trade `net_return`:

| | Trades | Mean net_return |
|---|---|---|
| TRENDING=True | 843 | **+0.58%** |
| TRENDING=False | 62 | **−0.62%** |

Real difference +1.20pp, checked against a 1,000-seed shuffled-label null
(permute the flag across trades): null mean −0.01pp, std 0.75pp, real
difference sits at the **95th percentile** — a real, not noise-level,
separation in the expected direction.

**Methodology note, stated plainly**: this used the standard `walk_forward()`
harness for convenience, which requires a 756-bar training warm-up before
its first fold even though `simple_trend` needs no training — so this
905-trade sample starts later and is smaller than
`EQUITY_EXIT_REGIME_SIMPLE_TREND.md`'s full-history 1,539-signal
characterization (which used a dedicated full-history driver bypassing the
fold structure). The direction and separation are clear regardless, but
this isn't a byte-for-byte reproduction of that doc's sample.

**Design fork, resolved**: the first pass above used a per-candidate trend
gate — only 62 of 905 actual `simple_trend` candidate-days (6.9%) failed
it, far below the ~21% `CHOPPY_SIDELINED` rate at the whole-calendar level.
That's because `simple_trend`'s own selector already restricts to symbols
above their 50-day benchmark trend before ranking by momentum, and taking
the max of 4 correlated-but-partially-independent names over a lenient
gate (`return_26 > -1%`, `ema_slope_atr > -0.20`) passes far more often
than the gate would on any single name — an availability check ("does
simple_trend have a valid pick today"), not a read on the shared market
environment two other strategies would also be operating in.

Measured directly (`compute_trend_flag_market_wide`, same gate, evaluated
on SPY itself instead of the daily winner):

| | Per-candidate (original) | Market-wide (SPY, resolved) |
|---|---|---|
| Trending share of `simple_trend` trade-days | 93.1% (843/905) | 87.5% (792/905) |
| Trending share of whole calendar | 78.9% | 65.5% |
| Mean net_return, TRENDING=True | +0.58% | +0.70% |
| Mean net_return, TRENDING=False | −0.62% | −0.94% |
| Real diff | +1.20pp | **+1.64pp** |
| Shuffled-null percentile (1,000 seeds) | 95th | **100th** |
| Non-trending complement (trade-days) | 62 (6.9%) | **113 (12.5%)** |
| `CHOPPY_SIDELINED` share, whole calendar | ~21% | **34.5%** |

The market-wide version wins on both counts that matter for a rotation
system: it separates `simple_trend`'s real performance at least as well
(slightly better, in fact), and it hands the chop/strangle arm a
meaningfully larger, more usable window to route capital into — 34.5% of
the calendar instead of 21%, and roughly double the non-trending
trade-day complement. **Resolution: `classify_regimes`'s canonical
`trending` column now uses the market-wide (SPY) definition
(`compute_trend_flag_market_wide`).** The per-candidate version is kept in
the module as `compute_trend_flag_percandidate` and exposed as the
secondary `trending_percandidate` diagnostic column, but no longer feeds
`primary_bucket`. Updated base rates (full 2018-2026 calendar, 2,150
days): TRENDING 1,382 (64.3%), CHOPPY_SIDELINED 741 (34.5%),
TRENDING+DAX_SIGNAL_DAY 25, DAX_SIGNAL_DAY-alone 2 — regenerated in
`outputs/regime_classifier_scaffold/`.

## Check 2 — does `dax_signal_day` behave as the market-state doc predicted?

`dax_signal_day` (27 days) is confirmed a **proper subset** of the full
validated 116-day quartile gate (checked directly, not assumed).

| | Trades | Win rate | Sharpe | Total return |
|---|---|---|---|---|
| Full quartile gate | 116 | 61.2% | 3.76 | +11.6% |
| `dax_signal_day` (quartile+calm) | 27 | 63.0% | **6.62** | +2.9% |

Sharpe nearly doubles on the narrower, calm-conditioned subset — directly
consistent with `EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md`'s finding
that this edge is a calm-regime edge (more consistent, not necessarily
bigger). Total compounded return is naturally smaller with 89 fewer trades
compounding. 27 trades is a thin sample — this is a consistency check on
construction, not a re-validation of profitability at that N.

**Reminder**: this table is the underlying SPY-first-hour **stock** signal
(what the original backtest and market-state doc both measured), not the
real-ThetaData **options** trade. The narrow subset's options profitability
has not been independently re-tested with real quotes.

## Open questions for the eventual rotation backtest (explicitly not solved here)

- Capital allocation across strategies/buckets — not attempted.
- What happens when a regime flips mid-trade (e.g. `simple_trend` mid-way
  through a 10-day hold when tomorrow reclassifies as `CHOPPY_SIDELINED`) —
  flagged, not resolved.
- ~~Whether `trending` should gate on the specific rotation candidate or a
  broader market-wide measure~~ — **resolved**, see Check 1: market-wide
  (SPY) is the canonical definition.
- `CHOPPY_SIDELINED` has no strategy behind it yet — waiting on the
  short-strangle backtest's result.

## Status

Classifier built and validated against real historical data on both flags.
Base rates are sane, no-lookahead is confirmed by construction, and both
flags demonstrably track what they're supposed to track. Not yet wired into
any capital-allocating rotation backtest — that's explicitly the next step,
pending the strangle result.
