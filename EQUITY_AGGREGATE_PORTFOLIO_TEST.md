# Aggregate 3-sleeve portfolio test: Joey's proposed "finalized" configuration

Allocation (locked in per Joey's instruction): **tech rotation with the
crash/drawdown-protection overlay 60%, DAX-informed SPY 0DTE/1DTE options
35%, BTC high-volatility bot 5%**, $10,000 total base. All figures below use
only real, already-validated trade-level data from prior work in this
project -- no new backtesting, no synthetic returns.

## Sizing convention -- a real bug caught and fixed before trusting anything

All three sleeves use fixed-notional-per-trade sizing (no compounding),
consistent with this project's convention throughout. But two different
per-trade allocation fractions are in play, and conflating them produces a
mathematically impossible result:

- **Tech rotation and BTC sleeves**: single-position, 100% of sleeve capital
  per trade (matches `simple_trend`'s own $2,500-fixed-notional convention).
- **DAX options sleeve**: `fixed_premium_250` on a $2,500 base = **10%
  allocation per trade**, not 100% -- options already carry embedded
  leverage, so the original backtest deliberately risked only a tenth of
  capital per contract, not the full account.

First pass treated all three sleeves as 100%-per-trade when rescaling to
$10,000, and the DAX-alone result came out with a **154.3% max drawdown**
-- mathematically impossible (drawdown cannot exceed 100% without going
negative) and an immediate signal something was wrong. Root cause: scaling
DAX's per-trade dollar size as if the whole sleeve were staked on every
options trade, rather than preserving the original 10% allocation ratio.
Fixed by scaling `pnl = net_return * (sleeve_capital * 0.10)` for the DAX
sleeve specifically, `* 1.0` for tech rotation and BTC. All numbers below
reflect the corrected calculation.

## Part 1: each sleeve alone at the full $10,000

| Sleeve | Window | Trades | Total return | Max drawdown |
|---|---|---|---|---|
| Tech rotation (crash overlay) | 2018-05-22 to 2026-07-09 | 166 | **+350.0%** | 12.0% |
| DAX 0DTE/1DTE options (50/50 split) | 2023-09-14 to 2026-07-22 | 232 | +117.6% | 11.9% |
| BTC vol-gated bot | 2026-03-10 to 2026-07-20 | 38 | +0.7% | 5.6% |

Same-window SPY buy-and-hold, for direct comparison:

| Sleeve's window | SPY return | SPY max DD |
|---|---|---|
| 2018-05-22 to 2026-07-09 | +212.8% | 33.7% |
| 2023-09-14 to 2026-07-22 | +72.3% | 18.8% |
| 2026-03-10 to 2026-07-20 | +10.2% | 6.4% |

**Tech rotation alone beats SPY on both return and drawdown over its full
8-year window.** DAX options alone also clearly beats SPY over its shorter
window, with a comparable drawdown. BTC alone, over its ~4-month window,
actually **underperforms** SPY on both counts (+0.7% vs +10.2%) -- the
smallest, newest, least-tested sleeve is also currently the weakest
standalone performer, which matters directly for the diversification
question below.

## Part 2: the blended 60/35/5 portfolio

Modeled as a **realistic phased deployment**, not a fictional
apples-to-apples comparison: from 2018-05 the tech sleeve's $6,000 is
invested and the other $4,000 sits in 0%-return cash (the DAX and BTC
strategies had no real data/deployment yet); the DAX sleeve's $3,500
activates 2023-09; the BTC sleeve's $500 activates ~2026-03.

| | Total return | Max drawdown |
|---|---|---|
| **Blended 60/35/5** | **+251.2%** | **9.4%** |
| SPY buy-and-hold, same full window | +211.1% | 33.7% |

The blend beats SPY on both return and drawdown, and its drawdown is even
*lower* than any single sleeve alone (tech: 12.0%, DAX: 11.9%). **Important
honesty check on that last point**: this is partly a mechanical consequence
of the phased-deployment cash drag, not purely a diversification benefit --
for the first 5+ years of the window, 40% of the portfolio sits in
zero-volatility cash while only the tech sleeve's 60% fluctuates, which
naturally dampens the blended portfolio's overall drawdown relative to any
single fully-invested sleeve during that period. It is a real, honestly
computed number reflecting what actually deploying this configuration
progressively would have looked like -- but it should not be read as "the
three strategies smooth each other out" without that caveat.

### Secondary check: short fully-overlapping window (all 3 sleeves active)

2026-03-10 to 2026-07-09 (120 days, the only period with real data behind
all three sleeves simultaneously): **+2.57% return, 2.45% max drawdown**.
Directional only -- 63 portfolio-affecting trade-days is a small sample,
and it's dominated by the tech sleeve given the other two are barely
contributing gains in this short window (DAX and BTC are both flat-to-weak
over this specific stretch).

## Does diversifying across the three actually help?

**Answer: mostly no, on a pure risk-adjusted-since-inception basis --
concentrating in tech rotation alone wins on total return.** Tech rotation
alone (+350.0%, 12.0% DD) beats the blend (+251.2%, 9.4% DD) on return, and
the blend's slightly better drawdown is substantially explained by the cash
drag above, not a genuine three-way netting-out of independent losses. The
blend's *only* real advantage over concentrating in tech alone is a modest
further drawdown reduction -- worth something if smoother returns are a
real priority (as Joey indicated caring about with the crash-overlay
discussion), but it comes at the cost of ~100 percentage points of total
return, mirroring the same tradeoff the crash-overlay test itself already
surfaced one level up.

**The DAX and BTC sleeves' actual contribution is currently small in
dollar terms relative to tech**, simply because they've had far less real
history to compound over (DAX: ~3 years; BTC: ~4 months) even though their
own per-window returns look reasonable in isolation. This is not evidence
they're weak strategies -- it's an artifact of how recently they came
online -- but it means today's blended number is still substantially a
"tech rotation with two small, young side bets" result, not yet a mature
three-way diversified portfolio with years of overlapping history to judge.

## Confidence level, stated plainly

The three sleeves rest on very different amounts of real evidence: tech
rotation has ~8 years and 166 trades behind it (the most-tested strategy in
this entire project); DAX options has ~3 years and 232 trades, all on real
ThetaData quotes; BTC vol-gated has ~4 months and 38 trades. **The
aggregate portfolio's overall statistical confidence is bottlenecked by the
BTC sleeve** -- 38 trades over 4 months is not enough to trust its
standalone numbers with anywhere near the confidence the tech and DAX
sleeves have earned. Treat the blended result as directionally informative,
not as a mature, fully-validated three-way portfolio track record yet.

Script: `experiments/run_aggregate_portfolio_test.py`. Outputs:
`outputs/aggregate_portfolio_test/` (equity curve CSVs for each sleeve and
the blend).
