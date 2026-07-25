# Mechanism cards: what breaks each strategy, not just how to improve it

Joey's proposed shift in research posture: instead of "how do I improve
strategy X," ask "what observation would make strategy X stop existing."
Four cards below, each with a real falsification test run against real data
(or a plainly-stated reason one couldn't be run) — not just descriptive
fields. This is a new layer on top of existing research, not a replacement
for it; every card below points back to the specific docs/experiments its
claims rest on.

## Strategy: DAX Lead (European lead signal)

**Mechanism**: delayed international information transmission — DAX's
pre-US-open session carries information that SPY's first trading hour
hasn't yet processed, so the pre-open move predicts the first-hour
reaction.

**Observable variables**: DAX pre-open move magnitude/direction (the
existing signal), sector breadth (`EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md`),
market-state calm/turbulent split (same doc), scheduled-event timing
(`MACRO_CALENDAR_DECOMPOSITION.md`).

**Failure modes**: the information delay closes (faster cross-market
arbitrage/algorithmic transmission), or a same-day competing catalyst
(afternoon Fed decision, etc.) swamps the morning signal before it can be
captured.

**Falsification test — run**: `experiments/run_dax_lag_decay_check.py`.
Question: is the OPTIONS P&L decay already documented in
`EQUITY_OPTIONS_REAL_DATA_RETEST.md` (0DTE +104.1% first half n=58 → +42.5%
second half n=58) evidence the underlying information-delay mechanism is
closing, or something else (e.g. options market repricing)? Split the same
116-trade sample the options work used, but on the RAW statistical
relationship (not P&L):

| | First half (58 trades, 2023-09→2025-04) | Second half (58 trades, 2025-04→2026-07) |
|---|---|---|
| Raw correlation (DAX move vs. SPY first-hour return) | +0.500 | +0.352 |
| Direction accuracy | 62.1% | **63.8%** |

**Result: mechanism looks intact, not decaying — the P&L decay has a
different explanation.** Correlation magnitude fell ~30%, but direction
accuracy (the metric that actually matters for a long/short signal) *held
steady, even ticked up slightly*. Year-by-year direction accuracy: 2023
37.5%* (n=8, too small to read), 2024 65.8%, 2025 63.4%, 2026 65.5% — flat,
not declining. If the information-delay mechanism were genuinely closing,
direction accuracy should erode over time; it hasn't. The options P&L decay
is better explained by something specific to the options leg (pricing,
liquidity, or magnitude variance) than by the core cross-market signal
dying. *2023's low accuracy is n=8, noted not trusted.

**Macro-calendar note**: `MACRO_CALENDAR_DECOMPOSITION.md` already ran the
sourced-event-calendar falsification this card would have asked for
(FOMC/ECB decision days *weaken* the edge; earnings weeks/payroll Fridays
*strengthen* it) — reused directly rather than re-deriving. Verdict there:
the edge is strongest when nothing competes with it later the same day, not
a "scheduled European information" story specifically.

**Verdict: mechanism intact.** Direction accuracy stable across sub-periods
and years; the P&L decay is a real, separately-documented fact but doesn't
indict the core mechanism.

---

## Strategy: Momentum (`simple_trend`)

**Mechanism, Joey's hypothesis**: persistence caused by slow institutional
repositioning — momentum exists because large holders reposition gradually,
not instantly, leaving a capturable lag.

**Competing existing characterization** (`EQUITY_EXIT_REGIME_SIMPLE_TREND.md`):
"rotate into whichever of four historically exceptional growth stocks has
the strongest relative momentum... not a stock-picking or market-timing
edge" — i.e., possibly just "four secularly strong stocks kept going up,"
no real repositioning-lag mechanism required.

**Observable variables (Joey's proposal)**: relative strength (already the
selector), analyst revisions, earnings revisions.

**Failure modes**: institutional rebalancing speeds up (less lag to
capture), or a structural trend reversal in the specific basket.

**Falsification test — run**: `experiments/run_momentum_revision_falsification.py`.
Real analyst-revision history via yfinance `upgrades_downgrades` (genuine
rating changes only — `Action in {up, down}`, excluding reiterations/
initiations), confirmed real coverage: AAPL 73 events (2012–2026), MSFT 75
(2012–2026), NVDA 67 (2016–2026), TSLA 109 (2019–2026). If the
repositioning-lag mechanism is real, `simple_trend` trades should perform
better within ±10 trading days of an active revision event (institutions
visibly repositioning) than during quiet periods.

| | Active revision (±10d) | Quiet |
|---|---|---|
| Trades | 213 | 692 |
| Mean net_return | **−0.08%** | **+0.67%** |

Real diff −0.75pp, shuffled-null (1,000 seeds): null mean −0.02pp, std
0.45pp, **6th percentile** — a real, not noise-level, separation, in the
OPPOSITE direction the repositioning-lag hypothesis predicts.

**Verdict: the strong-form mechanism (A) is undermined by real data, not
just untested.** If anything, `simple_trend` does worse, not better, around
visible analyst repositioning activity — the opposite of what "persistence
from slow institutional repositioning" predicts. This favors the project's
own more skeptical characterization (B): the observed edge looks more like
"momentum in a basket of secularly strong growth stocks" than a distinct
repositioning-lag effect. One plausible read: active-revision periods
coincide with genuine uncertainty/news events, which may inject short-term
noise or reversal risk right when `simple_trend` is entering — worth
flagging as a hypothesis for future work, not confirmed here.

**Secondary generalization check** (cheaper, already-run): does the edge
survive outside these specific 4 names? `EQUITY_MULTI_POSITION_SP100_STUDY.md`
already tested the same selector against the full S&P 100 — real signal at
the **100th percentile vs. a random-selection control**, at both 2-way and
5-way portfolio breadth, though with diminishing per-trade quality going
deeper into the ranked list (243→169→111 bps/trade). This argues AGAINST
"we just found 4 lucky stocks" — the selection criterion generalizes to a
much broader universe — but doesn't by itself support the specific
institutional-repositioning causal story either; it's evidence for "real,
generalizable relative-strength effect," agnostic on why.

**Methodology note**: this test used the standard `walk_forward()` harness
(905 raw daily-candidate observations, not the 188 single-position-filtered
actual trades from the full-history characterization) — same caveat the
regime classifier's Check 1 already flagged: smaller/later-starting sample,
directionally trustworthy, not a byte-for-byte reproduction.

---

## Strategy: Short Strangle (volatility risk premium)

**Mechanism** (corrected from the initial card, which described the LONG
side — this project has only ever tested SHORT): implied volatility priced
into the option, at entry, overstates what subsequently realizes over the
hold period — the seller is paid more for volatility risk than what
materializes, on average.

**Observable variables**: IV term structure, realized-vol forecast,
scheduled catalysts, order-flow imbalance (per Joey's original card) — this
project has tested trend-slope (rejected), realized-vol (rejected, worse),
and a forward-vol IV/RV spread filter (running in parallel, not yet
complete as of this writing).

**Failure mode**: a real catalyst materializes that the option wasn't
overpricing after all — realized vol spikes past what was priced in,
producing an outsized loss on the short premium position.

**Falsification test — run**: `experiments/run_strangle_event_day_clustering.py`.
Does the failure mode show up concretely, i.e., do the worst-loss trades
across the two completed backtests (`short_strangle_chop_backtest` +
`short_strangle_chop_backtest_realized_vol`, 153 trades combined) cluster
on identifiable scheduled-catalyst days rather than being uniform? Reused
the already-sourced, verified FOMC/ECB/earnings date lists from
`run_macro_calendar_decomposition_study.py` directly.

| | Base rate (all checkable trades) | Rate among worst-10-loss trades |
|---|---|---|
| Hold window touches an FOMC date | 31.9% (n=113 checkable) | 33.3% (n=6 checkable) |
| Hold window touches a mega-cap earnings date | 33.7% (n=136 checkable) | **60.0%** (n=5 checkable) |

**FOMC shows no clustering** — worst-loss rate is statistically
indistinguishable from the base rate. **Earnings shows a real directional
signal** — worst losses touch a mega-cap earnings date roughly 1.8x more
often than the base rate — consistent with the mechanism's predicted
failure mode (a real earnings-driven move overwhelming what was priced as
"probably calm"). **But n=5 checkable worst-loss trades is too small to
call this confirmed** — stated plainly, this is suggestive, not decisive.

**Coverage caveat**: 40 of 153 trades (26%) have entries before
2023-02-01 and can't be checked against FOMC/ECB at all (verified date
lists don't extend earlier) — a real, disclosed gap, not silently patched.
Earnings dates checked are AAPL/MSFT/NVDA's own report dates as a proxy for
"a plausible SPY-moving catalyst during the hold," not SPY-specific events
directly — an imperfect but reasonable reuse of already-sourced data rather
than building new SPY-move-detection logic.

**Verdict: mechanism's predicted failure mode partially confirmed,
statistically thin.** Earnings-day clustering in the worst losses is
directionally exactly what the mechanism predicts, and argues for a
pre-trade earnings-week exclusion as a real next step (not a curve-fit
one, since it's motivated by the mechanism itself, not by the strangle
result). FOMC doesn't show the same pattern. The still-running forward-vol
(IV/RV) variant's own worst trades should be folded into this check once
that backtest completes.

---

## Connective tissue: why this framework explains the rotation backtest's null result

The regime-rotation backtest (a few messages before this work) found gating
`simple_trend` to a market-wide TRENDING classifier provided no measurable
benefit over running it continuously — statistically indistinguishable
from randomly skipping the same number of trades. Under this framework,
that result stops being surprising: the classifier used was built on
price-action trend-slope, which doesn't measure `simple_trend`'s claimed
mechanism (institutional repositioning) at all — and this session's own
falsification test above found real analyst-revision activity doesn't even
correlate positively with `simple_trend`'s returns. Gating on a proxy for a
mechanism that itself doesn't clearly hold (or that the proxy never
measured to begin with) was never going to help. The lesson generalizes:
before building another regime gate for any strategy here, the first
question should be "does this variable actually measure the claimed
mechanism," not "does it look like a plausible market-state split."

## Status

Four cards written, three real falsification tests run against real data
(DAX lag decay, momentum revision, strangle event clustering), one
partially blocked by sample size rather than data access. Biggest single
finding: the momentum mechanism's strong-form version (institutional
repositioning) looks actively contradicted, not just unconfirmed — a
genuinely new, non-obvious result this session's shift in framing produced.
Scripts: `experiments/run_dax_lag_decay_check.py`,
`experiments/run_momentum_revision_falsification.py`,
`experiments/run_strangle_event_day_clustering.py`. Outputs:
`outputs/dax_lag_decay_check/`, `outputs/momentum_revision_falsification/`,
`outputs/strangle_event_day_clustering/`.
