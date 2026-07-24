# Real ThetaData re-test: calls and the volatility breakout straddle

Follow-up to every "not promotion-grade without real spread data" verdict
in this project's options work. Joey purchased ThetaData's Options Value
tier ($40/mo, 4yr history, 1-min granularity) specifically to resolve
this. Two studies re-tested so far with real quotes: the original calls
deep dive, and the volatility breakout straddle (previously the strongest
synthetic finding of the night, and Joey's stated top priority to
re-check).

**Data window, confirmed empirically not assumed:** real quotes return
correctly from 2021 onward; 2020 and earlier return "no data" under this
subscription tier (2019 returns an explicit `PERMISSION_DENIED` requiring
a higher tier). All re-tests below are restricted to signals from
2021-06-01 onward — roughly half the original 2018+ sample, stated
plainly rather than hidden.

## Method discipline: same signals, same window, only pricing differs

Every comparison here holds the signal, universe, dates, and hold period
identical between the old (Black-Scholes-off-realized-vol) and new (real
ThetaData bid/ask) pricing — the only thing that changes is how the
option leg is priced. This is deliberate: it's the only way a difference
in outcome can be attributed to data quality rather than a confound.

**A methodology trap caught before trusting the first comparison**: raw
trade counts between synthetic and real runs are NOT directly comparable
in this project's `simulate_single_position` — a candidate with no real
quote available gets dropped from the sequence entirely before the
single-position overlap filter runs, and a removed candidate can't
"block" a neighboring one anymore. This mechanically inflates the real
run's trades-taken count relative to synthetic in a way that has nothing
to do with pricing quality. Verified this directly (checked that 0 of 924
signals were floor-rejected by the synthetic pricer, ruling out the
"synthetic silently drops most signals" theory before it got repeated as
a finding) before trusting anything downstream. Per-trade expectancy
doesn't have this artifact and is the fairer metric throughout.

## 1) Calls deep dive

| | Synthetic (old, same window) | Real (ThetaData) |
|---|---|---|
| ATM: expectancy/trade | -27.3% | **-4.4%** |
| ATM: profit factor | 0.43 | **0.88** |
| ATM: total return | -100% | -48.5% |
| 5% OTM: expectancy/trade | -26.6% | -22.4% |
| 5% OTM: profit factor | 0.47 | **0.24** |

Mixed, not a clean "real data vindicates calls" story. ATM real pricing
is dramatically less bad than synthetic implied (nearly breakeven PF vs.
a clean loser) — a real chunk of the original "not a clean nail"
uncertainty was genuinely a data artifact. But real ATM calls still lost
money overall in this window, and 5% OTM real data is actually *worse*
by profit factor than synthetic suggested despite a higher win rate.
5% OTM real sample is thin (29 trades) — held loosely.

**Reading it honestly**: real data resolves the uncertainty in the
direction of "still not a winner, but less badly wrong than the
synthetic pricing implied." Not promotion-grade either way.

## 1b) Long-call threshold analysis — does signal strength predict when calls beat shares?

Hypothesis going in (Joey's framing): calls only earn their premium on the
strongest signals — say the top 5-10% by conviction — and lose it on
weaker ones. Tested directly: bucketed the same 715/694 real-priced
call trades (moneyness 1.0/1.05) by `relative_strength_20` percentile
within the signal pool, real calls vs. real shares in each bucket,
independently compounded per bucket (small-N buckets, so treat total
return as directional, profit factor as the more robust number).

| Bucket (by relative_strength_20 percentile) | ATM call PF | ATM stock PF | 5%OTM call PF | 5%OTM stock PF |
|---|---|---|---|---|
| Bottom 50% (weakest) | **1.43** | 1.04 | **1.42** | 1.02 |
| 50-75th pct | 0.83 | 1.07 | 0.63 | 0.78 |
| 75-90th pct | 2.57 | **3.03** | 2.66 | **3.72** |
| Top 10% (strongest) | 0.94 | 1.02 | 1.37 | **1.53** |

**The hypothesis is rejected — and inverted.** Calls only beat shares on
the *weakest* half of signals; shares dominate calls by a wide margin on
the strongest ones (75th percentile and up), most dramatically in the
75-90th bucket where stock's total return outran the call's by ~140
percentage points on both moneyness levels. Mechanistic read: on the
strongest-momentum signals the stock itself already captures a large,
clean move — the option adds leverage to a bet that's already working,
but also adds time decay and spread cost that eat into a move that
didn't need the extra convexity. On the weakest signals, where the stock
mostly just chops, the (still net-losing-but-less-so on a PF basis)
convexity of the option loses less on the numerous small losers.

**Caveat, stated plainly**: this doesn't mean "buy calls on weak
signals" — every bucket besides bottom-50%-calls is still marginal or
negative on an absolute basis, and small per-bucket sample sizes (70-179
trades) mean this ranking could reshuffle with more data. The finding
that survives scrutiny is negative, not positive: **there's no evidence
that signal strength is the missing ingredient that makes calls worth
their premium** — if anything, conviction is exactly when you should
reach for the instrument that doesn't decay, which lines up with the
instrument-selection framework's separate finding that shares win
risk-adjusted (`EQUITY_INSTRUMENT_SELECTION_FRAMEWORK.md`).

## 2) Volatility breakout straddle — the decisive result

This was flagged as the strongest synthetic finding of the whole project
(+342.5% total return, 64.9% win rate, full 2018+ window) and Joey's
explicit top priority to re-check with real data.

| | Real (2021-06-01+) | Synthetic, SAME window | Synthetic, ORIGINAL full window (2018+) |
|---|---|---|---|
| Trades | 66 | 72 | (full sample) |
| Win rate | 27.3% | 30.6% | 64.9% |
| Profit factor | **0.36** | 0.62 | — |
| Total return | **-83.5%** | -40.0% | **+342.5%** |

**Two separate effects, both pointing the same direction:**

1. **Period effect**: even under the OLD synthetic pricing, the
   2021-06-01+ window alone is already a loser (-40% total return, PF
   0.62) — nowhere near the +342.5% headline number. That number's
   strength was concentrated in 2018–mid-2021, exactly the period this
   subscription tier can't verify with real quotes.
2. **Pricing effect**: holding the window constant, real quotes make it
   meaningfully worse than synthetic (-83.5% vs -40.0%, PF 0.36 vs
   0.62). Consistent with the volatility risk premium bias flagged in
   every options doc all along — a straddle is a pure long-vega bet with
   no directional payoff to fall back on, making it the single
   structure most exposed to synthetic pricing understating what a real
   premium costs.

## Verdict

**Calls**: still not a winner, but real data narrows the uncertainty
in a mildly encouraging direction (ATM specifically).

**Volatility breakout straddle**: does not survive. The good years are
unverifiable at this subscription tier, and the verifiable years are a
real loser under both pricing methods, with real data making it *worse*
than synthetic already suggested. This reverses the night's most
promising options finding — a clean, decisive result, and the strongest
validation yet of why every prior options doc withheld promotion pending
real data.

## 3) European lead signal: SPY options vs. shares — real leverage amplification, cleanly positive

Same DAX-top-quartile signal dates as the validated backtest
(`EUROPEAN_LEAD_US_FIRST_HOUR_BACKTEST.txt`), same ~9:30/10:30 ET
entry/exit, priced with real intraday 1-minute SPY option quotes
(`option_history_quote`, not EOD data — the 1-hour hold needs intraday
granularity). 116 top-quartile days, **0 skipped for either DTE variant**
— unlike the volatile-universe equities, SPY 0DTE/1DTE liquidity is deep
enough that every single signal day had a real, tradeable quote.

| | Trades | Win rate | Expectancy/trade | Profit factor | Total return |
|---|---|---|---|---|---|
| SPY shares (reference) | 116 | 61.2% | 95.7 bps | — | +12.9% |
| **SPY ATM 0DTE (real)** | 116 | 45.7% | **1263.8 bps** | 1.69 | **+146.6%** |
| **SPY ATM 1DTE (real)** | 116 | 47.4% | **764.2 bps** | 1.70 | **+88.6%** |

This is the cleanest positive real-data options result of the night, and
the opposite outcome from the volatility breakout straddle. Lower win
rate than shares (46-47% vs 61%, exactly the "direction alone isn't
enough, the move has to be big enough" pattern seen in the calls deep
dive too) but the payoff asymmetry more than compensates — real leverage
amplification on a signal that was already independently validated
(out-of-sample split, random-shuffle control, doesn't concentrate on
FOMC days) before any option pricing was involved. 0DTE outperforms
1DTE, consistent with more gamma/leverage per correct call within the
same 1-hour window.

**One real caveat before treating this as decided**: a bug was caught and
fixed mid-run here — the first attempt returned 0/116 priced on both
variants due to a timezone mismatch (`build_dataset()`'s index is
tz-naive, the spot-price lookup index was tz-aware UTC, so every lookup
silently returned `None` instead of raising). Verified the fix on a
single manual trade before re-running the full 232-call batch, and the
corrected run shows 100% success rate, which is itself a good sign the
fix was complete rather than partial.

### Full rigor suite (Joey's due-diligence list, before touching the strategy logic further)

Confirmed directly from the code: entry is always the real quoted ask,
exit always the real quoted bid (never midpoint, never theoretical).
"+146.6%" is return on total account equity ($2,500 base, $250/10%
allocated per trade) — the underlying per-contract return is the
1263.8bps/trade expectancy figure.

| | 0DTE | 1DTE |
|---|---|---|
| Win rate | 45.7% | 47.4% |
| Profit factor | 1.69 | 1.70 |
| Payoff ratio (avg win / avg loss) | 2.01 | 1.89 |
| Avg winner / avg loser | $169 / -$84 | $98 / -$52 |
| Longest losing streak | 5 | 5 |
| Top 3 trades' share of total profit | **43.3%** | **45.1%** |
| Top 5 trades' share of total profit | **66.8%** | **65.5%** |
| Sharpe (annualized) | 3.09 | 3.09 |
| Max drawdown | 15.1% | 8.7% |
| Total return | +146.6% | +88.6% |

**Real concern, not glossed over**: return concentration. ~2/3 of total
profit comes from 5 of 116 trades (4.3% of trades). This is a convexity-
driven edge, not a smoothly distributed one -- expected for a positive-
payoff-ratio leveraged strategy, but it means the headline Sharpe/total-
return numbers overstate how a typical month would actually feel.

**Cost stress** (0DTE): no-extra-cost 146.6% -> +1 tick 130.5% -> +2 tick
114.6% -> +5bp 145.9% -> +10bp 145.4%. Survives comfortably at every
level -- tick-based costs bite harder than bps here since ticks are a
larger fraction of these smaller option premiums, but even +2 ticks
round-trip the edge stays strongly positive.

**Out-of-sample split**: both halves stay positive (0DTE: +104.1%
first half n=58, +42.5% second half n=58; 1DTE: +63.0% vs +25.6%) --
doesn't flip to a loser, but there's real decay in magnitude, not flat.
Stated plainly rather than only reporting the flattering full-sample
number.

**$1,000/trade instead of $250**: total return scales to +586.4% (0DTE)
but max drawdown scales from 15.1% to **70.1%**, not proportionally --
fixed-notional sizing means a losing streak eats a much bigger bite of
the same $2,500 base at 40% allocation vs 10%. Also a real practical
gap the simulator doesn't model: it keeps applying $1,000/trade even
once equity is depleted, which isn't achievable in practice. More
profit, meaningfully more real risk -- not a free scaling.

**Long vs. short breakdown** -- direct answer to "does shorting/puts
work better with real data": within this signal, **puts already
slightly outperform calls** (0DTE: shorts +83.8% n=60 vs longs +62.8%
n=56; 1DTE: shorts +49.3% vs longs +39.4%). Different mechanism than
the older weakest-momentum put thesis below though -- this short side
comes from the independently-validated DAX-down-move signal, not a
"short the weakest stock" selector.

## 4) Tail hedge (45-DTE, 10%-OTM SPY puts) — inconclusive, and the reason
why matters

First pass: 61 re-entry cycles (2021-06+, 21-trading-day cadence), only
19 priced (69% skipped as "no real data"), and the 19 that survived
looked disastrous (win 5.3%, PF 0.05, -26.8% total return) with one
outright suspicious number -- the sole 2022-bear-window trade showed an
exact 0.0% return during an 18.2% SPY drawdown, which a 10%-OTM put
should have profited handsomely from.

Didn't take that at face value. Traced it to source with raw client
calls (bypassing this module's swallowed exceptions) on a specific
skipped cycle (signal 2021-07-30, target 370 strike, spot ~$412 -- ~10%
OTM): `option_history_eod` for the intended entry date (2021-08-02)
raised `NoDataFoundError`, but the *same contract* had good data by the
exit date (2021-08-30). Pulling the raw quote history directly showed
why -- the contract's earliest `created` timestamp was **2021-08-11**,
nine days *after* the intended entry date. The strike hadn't been
listed yet.

Root cause, confirmed by inspecting the actual client method
signatures: `option_list_strikes(symbol, expiration)` and
`option_list_expirations(symbol)` **take no as-of-date parameter at
all** -- they return the full lifetime set of strikes/expirations a
series has ever had, not what existed on the signal date. Exchanges add
new deep-OTM strikes incrementally as the underlying drifts, so a
45-DTE, 10%-OTM put's target strike routinely doesn't exist yet on the
date this strategy would need to buy it. This is a genuine data/API
ceiling at this subscription tier, not a bug in `real_put_trade` --
and it's specific to *this* structure: every other real-data retest in
this project (ATM/5%-OTM 30-day calls, 0-1 DTE ATM SPY options) uses
strikes that are listed well in advance and stay liquid the whole time,
which is exactly why those hit ~100% coverage and this one hit 31%.

**Verdict: inconclusive, not "hedge costs 27% a year."** The 19
surviving trades are not a random subset of the 61 cycles -- they're
systematically the cycles where a thin, newly-listed, deep-OTM contract
happened to already have a market-maker quote on day one, which is not
independent of market conditions (more likely exactly when volatility
was already elevated and premium already expensive). Reporting the raw
PF/win-rate number here would be reporting a selection artifact as if it
were a cost measurement. This project's data tier cannot currently
answer "what does this hedge really cost" for structures this far OTM
and this far dated -- a real, disclosed dead end, not a swept-under-the-
rug one.

## 5) P&L decomposition — did trades lose because the signal was wrong,
or because the option overpaid for volatility?

Applied `implied_greeks.decompose_option_pnl` (implied-vol-from-real-
price, then first-order delta/vega/theta attribution) to all 232
completed real-quote European-signal trades (116 0DTE + 116 1DTE) --
100% coverage, no skips (entry/exit SPY spot reconstructed from the
already-cached 60m bars, no new API calls needed). Anchored at entry
Greeks per standard convention; residual bucket absorbs gamma convexity
and anything a first-order model can't explain, reported explicitly
rather than hidden in another bucket.

| | 0DTE | 1DTE |
|---|---|---|
| Mean total P&L/trade | +$0.194 | +$0.225 |
| Underlying-move share of \|P&L\| | **90.1%** | **89.4%** |
| IV-change share of \|P&L\| | 30.2% | 35.0% |
| Theta share of \|P&L\| | 15.7% | 4.6% |
| Residual/gamma share of \|P&L\| | 22.5% | 12.6% |
| Mean entry IV -> exit IV | 31.1% -> 28.0% | 21.0% -> 19.9% |
| **Losers: mean underlying-move P&L** | **-$0.402** | **-$0.426** |
| Losers: mean theta P&L | -$0.147 | -$0.045 |

**Direct answer: mostly the signal, not the premium.** On losing trades,
the dominant term is the underlying moving the wrong way (-$0.40 to
-$0.43 mean), roughly 3x the size of the theta drag (-$0.045 to -$0.15).
IV genuinely fell from entry to exit on average in both structures (a
real, quantifiable volatility-crush cost during the first trading hour,
not assumed) but it's a secondary drag, not the main story. 0DTE pays
meaningfully more theta than 1DTE (15.7% vs 4.6% of |P&L|) as expected
-- a contract expiring same-day burns much faster per hour held than one
with 24 more hours of extrinsic value -- but even for 0DTE, direction
still dominates over decay. This is the honest version of "is this
options edge real or is it just theta-negative noise that happens to
work sometimes": the underlying-move share being ~90% in both cases
means the European lead signal's directional accuracy is what's earning
the money: the option is a leveraged vehicle for a real signal, not
disguised vol-selling.

## Status: full rigor pass complete

Every item from the original post-ThetaData-purchase agenda is done:
calls re-test, volatility breakout straddle re-test, European signal
stock-vs-options re-test with full rigor suite, long-call threshold
analysis, tail hedge attempt (root-caused as a genuine data-tier
limitation, not silently accepted), and this P&L decomposition. Nothing
left queued in this doc.

Scripts: `experiments/run_equity_options_real_data_retest.py`,
`experiments/run_equity_vol_breakout_real_data_retest.py`,
`experiments/run_equity_options_real_vs_synthetic_comparison.py`,
`experiments/run_call_threshold_analysis.py`,
`experiments/run_tail_hedge_real_data_retest.py`,
`experiments/run_pnl_decomposition.py`
