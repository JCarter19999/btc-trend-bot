# Regime-rotation backtest: does classifying the market and dispatching strategies by it actually help?

First real dispatch backtest on top of `EQUITY_REGIME_CLASSIFIER_SCAFFOLD.md`,
combining only the two arms that survived real-data validation. Built and
run immediately after the short-strangle chop backtest reported back
negative (`EQUITY_SHORT_STRANGLE_CHOP_STUDY.md`), per Joey's standing
instruction.

**Headline verdict: rotation provides no measurable benefit over just
running `simple_trend` continuously.** Gating it to TRENDING days is
slightly *worse* than the ungated benchmark and statistically
indistinguishable from gating to a random same-size subset of days. The
only real value in this backtest is stacking the already-independently-
validated DAX_SIGNAL_DAY options overlay on top — which doesn't actually
need a regime classifier at all, since it fires on its own independent
trigger regardless of the trend state.

## Design choices, stated up front

**DAX_SIGNAL_DAY trigger**: the FULL validated 116-day top-quartile-|DAX
move| gate (recomputed directly from `build_dataset()`), **not** the
classifier's narrower 27-day `dax_signal_day` column. That column
additionally ANDs in three calm-regime conditions shown to correlate with
the underlying *stock* signal's Sharpe but never independently re-tested
against real option P&L (see the classifier module's explicit caveat).
Using the narrower, unvalidated subset here would have silently smuggled an
untested hypothesis into a backtest meant to combine only what's proven.

**CHOPPY_SIDELINED = cash, not the strangle.** The short-strangle chop
strategy was built and real-data-tested separately and rejected: barely-
above-breakeven baseline (+3.4%, PF 1.04), flips negative at +2 ticks of
cost stress, out-of-sample split flips sign entirely, and — critically —
showed no demonstrated advantage over an unconditional random-date short
strangle. This is a deliberate, evidence-based choice, not an unfilled
placeholder. A real chop-specific edge remains an open problem (the
strangle study's own suggested next step: try realized-vol instead of
trend-slope as the "calm" proxy).

**Capital-sleeve design**: TRENDING and DAX_SIGNAL_DAY do not compete for
capital or time. TRENDING is a multi-day stock hold at the full $2,500
fixed notional (matching `simple_trend`'s own validated sizing);
DAX_SIGNAL_DAY is a same-day ~1-hour options round-trip at a separate $250
(10%) fixed-premium sleeve, matching the DAX study's own sizing and this
project's existing live-deployment precedent of running each arm as an
independently-ledgered account rather than one pool of undivided cash. The
"combined account" below is the **sum** of both sleeves' dollar P&L against
one $2,500 base — it does not model shared-capital competition (e.g. the
DAX sleeve being blocked because $2,500 is "tied up" in a stock position).
That would be a materially different, harder assumption, left as an open
question rather than silently decided.

**Overlap handling**: TRENDING and DAX_SIGNAL_DAY co-occur on ~1.2% of
calendar days (25 of 2,150). Because they don't share capital or a same-day
time window (the DAX trade closes by ~10:30 ET; the next TRENDING entry
decision isn't until the next session), both simply fire independently on
overlap days — no precedence order was needed or imposed.

**No lookahead**: TRENDING is computed from `return_26`/`ema_slope_atr` as
of signal date d's close, and `simple_trend` already enters on d+1 — gating
d+1's entry on information known at d's close adds no new lookahead.
DAX_SIGNAL_DAY uses only pre-13:30-UTC DAX data for a same-day ~9:30 ET
entry, unchanged from the original validated study.

## Results

| Arm | Trades | Total return | Max DD | Win rate | PF |
|---|---|---|---|---|---|
| TRENDING, **gated** (rotation) | 207 | +146.6% | 27.7% | 48.8% | 1.30 |
| TRENDING, **ungated** (plain `simple_trend`, benchmark) | 225 | **+162.9%** | 24.9% | 48.0% | 1.31 |
| DAX_SIGNAL_DAY, SPY 0DTE (primary) | 116 | +146.6% | 17.5% | 45.7% | 1.69 |
| DAX_SIGNAL_DAY, SPY 1DTE (reference) | 116 | +88.6% | 10.6% | 47.4% | 1.70 |

**Gating hurts, not helps**: +146.6% gated vs. +162.9% ungated, on 18 fewer
trades (207 vs. 225 — the 792 TRENDING-flagged signal-days out of 905 total
candidate-days, after single-position overlap filtering). Win rate and
profit factor are nearly identical between gated and ungated (48.8%/1.30 vs.
48.0%/1.31) — the regime filter isn't discriminating trade *quality*, it's
just removing trades.

**Shuffled-regime-label null control (200 seeds)**: is the TRENDING-gated
result even distinguishable from gating to a random same-size subset of
trade-days? Real total return (+146.6%) sits at the **49.5th percentile**
of the null distribution (null mean +148.5%, std 30.0pp) — indistinguishable
from chance. The market-wide trend filter carries **no information** about
which of `simple_trend`'s own trade-days will be good or bad. This directly
echoes `EQUITY_EXIT_REGIME_SIMPLE_TREND.md`'s original finding that
expectancy increases monotonically with hold duration and that cutting
positions early on short-term noise costs more than it saves — pausing
*entries* on a "choppy day" turns out to be the same kind of noise-reactive
cut, just applied to entries instead of exits, and it doesn't help here
either.

**Combined account (TRENDING gated + DAX 0DTE sleeves, additive)**: 323
total trades, **+293.2%** combined total return, 27.7% max drawdown,
$9,830 final equity on a $2,500 base. This number is real but shouldn't be
over-read as evidence of a "rotation" edge — it's the arithmetic sum of two
*already independently validated* positive-expectancy strategies that
happen not to compete for capital or time. You get the same +293.2% by
just running `simple_trend` unconditionally (+162.9%, even better than the
gated version) and separately running the DAX overlay whenever its own
trigger fires (+146.6%), with no regime classifier involved in either
decision.

**Day-count accounting** (whole calendar, 2,150 days, 2018–2026): TRENDING
1,382 (64.3%), CHOPPY_SIDELINED 741 (34.5%), TRENDING+DAX_SIGNAL_DAY 25,
DAX_SIGNAL_DAY-alone 2. Note DAX_SIGNAL_DAY's trigger is only computable
from ~2023-09 onward (yfinance intraday + real ThetaData coverage window),
while TRENDING covers the full 2018–2026 daily history — the two arms' data
windows genuinely don't match, stated plainly rather than blended.

## Honest read

The regime classifier's TRENDING flag is a legitimate, no-lookahead,
real-data-validated signal for *characterizing* the market (Check 1 in the
scaffold doc: it does correlate with `simple_trend`'s per-trade returns,
+1.64pp real diff, 100th-percentile null). But **characterizing the market
and knowing what to do about it are different problems**, and this backtest
shows the second one isn't solved: knowing today is "choppy" doesn't tell
you to skip `simple_trend` (skipping doesn't help — the null control proves
it), and there's currently no validated alternative strategy to redirect
that capital into during those 741 CHOPPY_SIDELINED days (34.5% of the
calendar) — the strangle was the candidate and it failed.

**What actually works right now, with no rotation logic required**: run
`simple_trend` continuously (don't gate it), and separately run the DAX
0DTE SPY overlay whenever its own independent trigger fires. Both are
real, validated, and stack without competing for capital. The "rotate
between regimes" framing this whole thread of work set out to test doesn't
currently add anything beyond that — not because the classifier is wrong,
but because there's no validated third strategy for it to route into yet,
and gating the one strategy that IS validated turns out to just be
noise-driven trade-count reduction.

## What's next, if this is worth continuing

- The open problem is still finding a real edge for the CHOPPY_SIDELINED
  34.5% of the calendar — the strangle study's own suggestions (realized-
  vol filter instead of trend-slope, larger random-control sample) are the
  most concrete leads.
- If a chop-specific edge is ever found, re-run this exact rotation
  backtest with it substituted in — the harness (`run_regime_rotation_backtest.py`)
  is built to take a third arm without restructuring the other two.
- The capital-sleeve assumption (DAX overlay doesn't compete with TRENDING
  for capital) was stated but not stress-tested against a shared-capital
  alternative — worth a sensitivity check if this ever approaches a real
  deployment decision.

Scripts: `experiments/run_regime_rotation_backtest.py`. Outputs:
`outputs/regime_rotation_backtest/` (`summary.json`,
`combined_equity_curve.csv`, per-arm trade CSVs).
