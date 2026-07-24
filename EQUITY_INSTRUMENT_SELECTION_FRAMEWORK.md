# Instrument selection framework: given an edge, what's the best way to express it?

Per Joey's reframing (2026-07-24): stop asking "should I trade options,"
start asking "given this validated edge, which instrument produces the
best risk-adjusted outcome." Built as a reusable comparison
(`experiments/run_instrument_selection_experiment.py`,
`src/btc_trend_bot/thetadata_intraday_pricing.py`) applied to the
European lead signal (DAX-top-quartile direction, 116 real trading days,
2023-09 to 2026-07) as the concrete demonstration. All five instruments
tested on the exact same 116 signal days, real ThetaData quotes (ask-in/
bid-out) for every options leg -- only shares involve no options pricing
at all.

**Mini futures: explicitly out of scope**, not silently skipped -- no
data access for that asset class at this subscription tier or in this
project's existing data pipeline.

## Result, ranked by Sharpe (the risk-adjusted metric, not total return)

| Rank | Instrument | Win rate | Sharpe | Max drawdown | Total return |
|---|---|---|---|---|---|
| 1 | **SPY shares** | 61.2% | **3.76** | **2.3%** | +11.6% |
| 2 | Vertical spread (1DTE) | 49.6% | 3.24 | 7.2% | +80.0% |
| 3 | ATM 0DTE | 45.7% | 3.09 | 15.1% | +146.6% |
| 3 | ATM 1DTE | 47.4% | 3.09 | 8.7% | +88.6% |
| 5 | 0.40-delta 1DTE | 44.8% | 2.75 | 10.4% | +88.0% |

## Reading this honestly

**Shares win on risk-adjusted return, despite the lowest total return of
the five.** Every options expression trades some Sharpe for more
absolute return -- none beat shares risk-adjusted. If "best outcome"
means Sharpe specifically, the honest answer for this signal is: don't
use options at all.

**If leverage/amplification is the actual goal** (not pure Sharpe), the
vertical spread is the clear best options structure -- closest Sharpe to
shares of any options variant, roughly half the drawdown of the outright
0DTE call/put, while still capturing 80% total return. This is the
"sweet spot" the framework exists to find: most of the amplification,
much less of the risk that comes with an outright option's unlimited
theta/gamma exposure.

**0DTE has the biggest headline number (+146.6%) and the worst risk
profile** -- ties 1DTE on Sharpe but carries nearly double the drawdown.
Consistent with the concentration finding in
`EQUITY_OPTIONS_REAL_DATA_RETEST.md` (66.8% of 0DTE's profit came from 5
of 116 trades) -- the big total return is real, but it's convexity-driven,
not smoothly earned, and that shows up here as extra drawdown risk, not
just extra reward.

**0.40-delta ranks last on Sharpe.** Going further OTM than ATM didn't
earn its keep on this signal -- more theoretical convexity, but not
enough extra realized return to justify the added return volatility.

## Why this framework matters more than any single result

The point isn't "vertical spreads are best" -- that's specific to this
signal's risk/return shape and could easily differ for a different edge
(a higher-conviction, lower-frequency signal might favor more leverage;
a noisier one might favor shares even more strongly). The point is the
platform can now answer the *general* question systematically for any
validated signal, rather than defaulting to whichever instrument
produced the most exciting-looking backtest. Total return alone would
have pointed at 0DTE; Sharpe points at shares or the spread depending on
what's actually being optimized for -- and having both numbers, not just
the flashy one, is what keeps this honest.

## Addendum: futures (ES=F), added 2026-07-24 per Joey's "what other profit
vehicles" question -- explicitly flagged futures as the likely-overlooked
one. No new data purchase needed; ThetaData has no futures API at all
(checked directly -- not in the client's method list), but yfinance
already serves free continuous ES=F 60m bars.

| Instrument | Win rate | Sharpe | Max drawdown | Total return |
|---|---|---|---|---|
| **ES=F futures (2hr proxy, unlevered)** | 64.8% | **4.55** | 2.4% | +16.9% |
| SPY shares | 61.2% | 3.76 | 2.3% | +11.6% |

At 1bp round-trip cost, on the RAW index-point return (no margin
leverage applied at all -- see caveat below), futures already beats
shares on every axis: higher win rate, higher Sharpe, comparable
drawdown, higher total return. A real margin account would then multiply
both the return AND the drawdown by whatever leverage ratio it actually
uses -- not modeled here since this project has no verified broker margin
schedule to apply, but the unlevered baseline being this strong is itself
the finding worth having.

**Two caveats that keep this honest, not a clean apples-to-apples row:**

1. **Window is a proxy, not the exact signal target.** ES=F's free 60m
   bars from yfinance align to the top of the hour (13:00, 14:00 UTC...),
   not the :30 grid SPY/DAX get -- there is no 13:30-14:30 UTC bar to
   look up at all. Fixed (after a first run produced only 1 usable trade)
   by using 13:00-15:00 UTC instead -- a 2-hour window that fully
   contains the true first hour, on the full ~2.4-year 60m history. 30m
   bars would align correctly but yfinance caps that granularity at ~60
   days lookback, too short to be useful here. So this measures "how did
   ES do over a slightly wider window that contains the SPY signal
   window," not literally the same 60-minute target -- a real, disclosed
   difference, not a redefinition to make the number look better.
2. **Coverage is shorter than the SPY backtest.** 105 of 116 top-quartile
   signal days have ES=F 60m coverage (yfinance's 60m history only goes
   back to 2024-02-29, vs SPY/DAX's full 2023-09 start) -- 11 of the
   earliest signal days are dropped, not mispriced.

## A real bug caught mid-run, fixed before trusting these numbers

The vertical spread pass crashed on its first run with an unhandled
`NoDataFoundError` -- `thetadata_intraday_pricing.py`'s functions called
the ThetaData client directly without the try/except wrapping that
`thetadata_pricing.py` (built earlier, more carefully) already had. The
0DTE/1DTE passes "succeeding" with 0 skipped earlier was luck (never hit
a missing date/strike/right combination in 116 calls), not correctness.
Fixed by wrapping every direct client call in the module through shared
`_safe_quote`/`_safe_expirations` helpers, verified against the exact
call that crashed before relaunching the full batch.
