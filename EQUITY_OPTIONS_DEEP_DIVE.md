# Options Deep Dive — Real Underlying Data, Synthetic Pricing, More Volatile Universe

Research-branch experiment. Follow-up to the earlier synthetic-bar
`EQUITY_HYBRID_OPTION_OPTIMIZER_RESULTS.md` finding (options added risk
without return on synthetic data). This version uses **real** daily
underlying price data (not synthetic bars) and a genuinely more volatile
universe, per Joey's explicit request ("im ok with more volatile stocks for
this").

## What this is and isn't

Checked first (see this doc's origin in conversation): no free historical
options-chain data exists with the depth needed (bid/ask, OI, IV, expired
contracts). yfinance only exposes *current* live chains. Comprehensive
vendors (ORATS, IVolatility, EODHD, Market Data API) are paid. So this is
**not a real options backtest** — it prices synthetic contracts via
Black-Scholes on real underlying price paths, using trailing realized
volatility (computed from real daily returns) as the implied-vol proxy.

**Stated bias, load-bearing for reading every number below**: real implied
volatility is almost always higher than trailing realized volatility (the
volatility risk premium). Pricing off realized vol therefore **systematically
underprices** these synthetic options relative to what they would really
have cost. Every result here is generous to the options side. A negative
result is real evidence against options; a positive one is weak evidence,
since real premiums (and real bid-ask spreads, which turned out to matter
enormously — see below) would likely have been worse.

## Method

Universe: TSLA, COIN, MSTR, PLTR, GME — annualized realized vol 62.6% /
85.0% / 79.4% / 69.5% / 126.6% (vs. the original AAPL/MSFT/NVDA/TSLA
basket's much lower vol). Selection: `simple_trend` (same validated
selector as the live deployment) for the long/call side; weakest-momentum
(mirror image, same as the earlier short-selling study) for the short/put
side. Options: 30 DTE at entry, sold (not exercised) at the same 10-day
exit the stock leg uses, at a few strike/spread combinations. Calls sized
at a fixed $250 premium budget (10% of the stock leg's $2,500 notional) —
a defined-risk allocation, not notional-matched, matching how a retail
trader actually sizes options.

**A real bug surfaced and got fixed along the way**: `hard_shutdown_drawdown`
in the portfolio simulator is checked unconditionally, not gated by
`safety_enabled` (inherited from the original `simulate_capital`'s design).
On this much-more-volatile universe, the $2,500-fixed-notional stock leg
blew through the 35%-drawdown hard-shutdown threshold almost immediately (a
single bad move on a 100%+-vol name approaches $875 in loss by itself),
permanently halting the simulation after ~15 of 1,533 signals. Neutralized
both `safety_enabled` and `hard_shutdown_drawdown` for this comparison,
since it's testing expression economics, not safety-layer robustness on a
mismatched-volatility universe — this is a labeling/config issue specific to
this study, not a finding about the live deployment (which runs on the
original, correctly-calibrated low-vol universe).

## Results (192 long signals, 83 short signals, single-position-correct, fixed budgets)

| Expression | Trades | Win rate | Profit factor | Max drawdown | Total return |
|---|---|---|---|---|---|
| Stock only (long) | 192 | 49.0% | 1.93 | 79.7% | **+980.0%** |
| Call, ATM, 5% spread | 192 | 39.6% | 1.92 | 72.4% | +623.8% |
| Call, ATM, 10% spread | 192 | 35.9% | 1.52 | 89.4% | +380.4% |
| **Call, 5% OTM, 5% spread** | 192 | 39.1% | **2.32** | **61.7%** | **+975.3%** |
| Put, ATM, 5% spread | 83 | 28.9% | 1.01 | 47.8% | +1.7% |
| Put, ATM, 10% spread | 83 | 27.7% | 0.75 | 85.3% | −77.9% |

(Total return is on a common $2,500 basis for every row — the call/put legs
only ever risk the $250 premium budget per trade, so this is a fair
apples-to-apples comparison of "how would a $2,500 account have grown,"
not a comparison of per-trade percentage swings.)

## Reading this honestly

**Calls: not the clean negative the earlier synthetic-bar test found.**
Under the most generous assumptions tested (5% OTM, tight 5% spread), the
call expression essentially **ties** stock-only on total return (975.3% vs
980.0%), while beating it on profit factor (2.32 vs 1.93) and max drawdown
(61.7% vs 79.7%) — and only ever risking 10% of capital per trade instead
of 100%. That's a genuinely more competitive picture than the original
hybrid-optimizer result.

**But the result is extremely sensitive to the spread assumption**, which
is exactly the input this synthetic study is weakest on (no real bid-ask
data). Moving from 5% to 10% spread (still a plausible real-world number,
maybe even optimistic for a further-OTM, longer-dated contract on a
lower-liquidity name like GME or MSTR) drops total return from 975% to
380% and drags profit factor from 2.32 to 1.52. Real spreads on these
specific names, at 30 DTE, are very plausibly wider than 10% for anything
but the most liquid at-the-money strikes — meaning the honest expectation,
combining (a) the underpriced-premium bias and (b) realistic spread
uncertainty, is that a true backtest would likely land at or below the
10%-spread row, not the favorable 5%-OTM row.

**Puts: a clean negative, consistent with the earlier finding.** Best
case (5% spread) is breakeven (+1.7%, PF 1.01); the more realistic 10%
spread case is clearly negative (−77.9%, PF 0.75). This reinforces
`EQUITY_SHORT_STRATEGY_STUDY`'s conclusion — the weakest-momentum short
thesis doesn't work, whether expressed as a short stock position (borrow
costs) or a capped-risk put (premium + spread costs). Different instrument,
same structural conclusion: no exploitable short-side edge was found
anywhere tonight, on either the original or this more volatile universe.

## Verdict

**Not a nail in the coffin for calls — an honest "inconclusive without real
spread data."** The single biggest lever in this entire study wasn't
volatility, strike, or even the underpricing bias — it was the bid-ask
spread assumption. If this is worth resolving definitively, the
highest-leverage next real-data purchase isn't full historical IV surfaces
or Greeks, it's **realistic historical bid-ask spread data** for 30-DTE
contracts on this specific volatile-name universe — a narrower, likely
cheaper ask than a full options-data vendor subscription.

**Puts are a clean negative** — no further work warranted there absent a
fundamentally different short-side signal (this test used the same
weakest-momentum selector already shown not to work).
