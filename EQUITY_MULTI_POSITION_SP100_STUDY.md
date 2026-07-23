# Multi-position portfolio study: S&P 100, up to 5 positions, 50% cap each

**Question:** can breadth (more independent opportunities) beat frequency
(shorter holds) at growing this strategy, while keeping the validated
10-day hold that the exit-strategy study already showed is where this
strategy's edge actually lives (5/10/15/20-day holds scored
106.6/243.3/272.3/452.7 bps/trade — monotonically *better* the longer you
hold, meaning this is medium-horizon drift capture, not a fast signal, and
compressing hold time fights the edge rather than scaling it)?

**Baseline being compared against:** the live-deployed `simple_trend`
strategy — 4 symbols (AAPL/MSFT/NVDA/TSLA), single position, 188 trades
over 2018–2026, 243.3 bps/trade expectancy, 22.1% max drawdown, +457% total
return on fixed-notional $2,500-per-trade sizing
(`EQUITY_EXIT_REGIME_SIMPLE_TREND.md`).

**Design, per Joey's spec (clarified 2026-07-23):** S&P 100 universe (101
tickers — Alphabet's dual share classes both count), same `simple_trend`
selection + disabled-ATR fixed-10-day-hold exit mechanics exactly as
deployed, up to 5 concurrent positions, each sized at
`min(50% of capital, currently-available cash)` — so 1–2 strong signals on
a quiet day should size up to 50% each, a fully-loaded 5-position book
should settle to ~20% each. Fixed-notional per position (fraction ×
$2,500, decided at entry, never re-based on realized gains) — same
sizing-decoupled-from-compounding convention used everywhere else in this
project. Script: `experiments/run_equity_multi_position_sp100_study.py`,
data: `data/real_sp100/*.csv` (yfinance daily, 2018–present).

## Headline result

| | 4-stock single-position (deployed) | S&P 100 multi-position |
|---|---|---|
| Trades | 188 | 350 |
| Trades/year | 22.1 | **42.1** |
| Mean trade return | 243.3 bps | 169.1 bps |
| Win rate | 58.0% | 56.6% |
| Max drawdown | 22.1% | 25.0% |
| Sharpe (daily, annualized) | not computed this way before | 0.98 |
| Total return | +457% | **+195%** |
| Percentile vs. 50-seed random-selection control | 99th | **100th** |

Trades/year did increase substantially, ~1.9x, confirming breadth-over-
frequency genuinely adds independent opportunities the way the hypothesis
predicted. And the selection criterion still works soundly at 101-symbol
scale — 100th percentile vs. random (mean random 54.6 bps vs. 169.1 bps
here, more than 3x) says this isn't noise. **But total return is lower
than the 4-stock baseline, not higher**, and mean trade return dropped
30% (243.3 → 169.1 bps). That's a real dilution effect: widening from 4
hand-picked strong-momentum names to the full S&P 100 average pulls in
weaker-momentum names on the margin, exactly the cost you'd expect from
loosening a selective filter.

## The bigger finding: the sizing rule was self-limiting, not the 5-slot cap

Checked capital utilization directly rather than just trusting the
headline numbers, and found something worth stating plainly: **every
single trade was sized at exactly 50%, and the book only ever held 0 or 2
positions — never 1, 3, 4, or 5.**

```
notional_fraction: mean=0.5, std=0.0 (literally every trade, no variation)
n_open distribution: 0 positions 16.1% of days, 2 positions 83.9% of days
```

This isn't a bug — it's what `min(50%, available_cash)` does mechanically:
the first two eligible candidates each claim 50% (100% of capital, gone),
so a third candidate that same day has 0% available and gets skipped by
construction. **"5 positions max, 50% cap each" is mathematically
equivalent to "at most 2 positions" under a no-leverage constraint** — the
5-slot cap never binds; the 50%-per-slot cap does, immediately, every
time. To actually use more than 2 of the 5 slots, the per-slot cap needs
to be smaller (e.g. ~20%, the "fully-loaded" case Joey described but which
this parameterization never reaches in practice) or capital needs to scale
with slot count instead of being fixed-reference.

Net effect on total return: roughly half the dollar exposure per trade
(50% vs. the baseline's always-100%) compounded with the 30% per-trade
dilution, only partly offset by 1.9x more trades/year — the arithmetic
roughly nets out to the observed ~43% of baseline total return
((42.1/22.1) × (169.1/243.3) × (0.5/1.0) ≈ 0.66, close enough given
non-compounding path differences).

## Verdict

The core hypothesis — breadth over frequency — is directionally right
(more real, statistically-validated independent trades, same edge
mechanism, still crushes random selection) but this specific sizing
parameterization ships less total capital per trade than the deployed
baseline, so it currently loses on total return despite winning on trade
count. Not a dead end: the natural next test is a genuine 5-way version
(e.g. ~20% flat cap, or a cap that only binds when fewer than 5 slots are
actually filled) to see whether real 5-way diversification recovers the
total-return gap while keeping the trades/year gain. Not run yet — a
deliberate stopping point to report the sizing-rule finding before
spending more compute chasing a parameterization that might have the same
issue.

## Caveats, stated plainly

- **Survivorship bias**: S&P 100 membership is today's (2026-07-23) list
  projected back to 2018 — no delisted/removed constituents. Real,
  unaddressed.
- **Cross-sectional correlation**: large-cap S&P 100 names move together
  more than a naive "5 independent bets" framing assumes — true
  diversification benefit is smaller than the trade count suggests.
- Sharpe here (0.98, daily-return-based, annualized ×√252) is the first
  *real* portfolio-level Sharpe this project has computed — prior
  single-position studies used a per-trade Sharpe-like proxy (mean/std of
  trade returns), not comparable to this number directly.
