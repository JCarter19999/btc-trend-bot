# Continuous momentum-rank switching vs. the fixed 10-day hold

Joey's question: "if day 0 Apple is the strongest but day 5 Tesla is, is it
worth adding logic that ranks and switches stocks continuously?" Two prior
findings in this project (ATR-stop removal, hold-duration monotonicity)
suggested reacting faster would hurt for this exact strategy/universe, but
neither tested this specific mechanism directly. This does.

## What was built

**`outputs/continuous_momentum_switching_test/daily_momentum_leaderboard.csv`**
— a genuine, reusable daily log: for every trading day 2018-2026 (2,129
days), the #1-ranked symbol by `relative_strength_20` among AAPL/MSFT/NVDA/TSLA
currently above their 50-day benchmark trend (falling back to the full pool
if none qualify) — same convention as `simple_trend`'s own selector. The
leaderboard changes hands 362 times over the full history (roughly once
every 2-3 weeks on average, though not evenly spaced).

**A continuous-switching simulator**: holds the current position as long as
it stays #1; the day the #1 ranking changes to a different symbol, exits at
that day's close and opens the new leader at the next day's open (same
next-bar-entry, no-lookahead convention as the existing `simulate_trade`).
Same $2,500 fixed-notional-per-trade sizing as the baseline it's compared
against (not the live deployment config's separate `position_fraction`
convention, which is for a different purpose).

## Baseline reconstruction — confirmed exact before trusting anything else

188 trades, 58.0% win rate, PF 2.042, +457.4% total return, 22.1% max
drawdown — matches `EQUITY_EXIT_REGIME_SIMPLE_TREND.md`'s documented
full-history characterization exactly (2018-04 through 2026-06, fixed-notional
$2,500/trade). One methodology fix needed en route: the live deployment
config (`config/simple_trend_exit_regime_strategy.yaml`) specifies
`position_fraction: 0.25` (fixed-fractional, compounding) for its own
purposes — using that sizing for this reconstruction gave a materially
wrong result (189 trades, +175.9% total return, 11.7% max drawdown) before
switching to the doc's actual fixed-notional-$2,500 convention.

## Result: continuous switching underperforms, but the failure mode is
## different from what the ATR-stop analogy predicted

| | Fixed 10-day hold (validated) | Continuous rank switching (2bps) | Continuous rank switching (5bps stress) |
|---|---|---|---|
| Trades | 188 | 361 | 361 |
| Win rate | 58.0% | 47.9% | 47.4% |
| Total return | **+457.4%** | +427.7% | +405.8% |
| Max drawdown | **22.1%** | **40.7%** | 43.7% |
| Mean return/trade | 243.3 bps | 118.5 bps | 112.4 bps |

**Total return is only modestly lower (~30pp), not the dramatic multi-fold
gap the ATR-stop finding (+50.3bps vs +132.8bps expectancy, >2x) might have
predicted by analogy.** What actually degrades sharply is risk quality:
max drawdown nearly doubles (22.1% -> 40.7%), win rate drops meaningfully
(58.0% -> 47.9%), and it takes almost double the trades (361 vs 188) to get
there — nearly double the transaction-cost exposure and operational
complexity for a slightly worse absolute result. Cost stress (5bps) makes
it modestly worse still, as expected given the much higher turnover.

**Random-switching control**: does the specific rank-change signal carry
real information about *when* to switch, or is any switching-at-this-frequency
roughly as good? Compared the real rank-driven total return (427.7%)
against 1,000 seeds of switching on the SAME number of dates (362) chosen
at random instead of by rank change. Real result sits at the **95.4th
percentile** (null mean 322.1%, std 63.9%) — the rank signal genuinely adds
information over switching-at-random-times-with-the-same-frequency. This
matters for interpretation: it's not that switching itself is random noise;
the leaderboard-change signal is doing real work. It's that switching
*this often*, however well-timed, costs more in risk and turnover than the
fixed-hold approach's patience is worth.

**Out-of-sample split**: first half (2018-2022) +238.7%, second half
(2022-2026) +192.3% — same sign, no reversal, roughly proportional decay.
Stable, not the sign-flip failure mode that's killed several other ideas
tonight.

## Verdict: don't switch continuously — the fixed hold still wins, but for
## a more specific reason than "reacting faster always hurts"

Not a blowout rejection like the ATR-stop finding was. The fixed 10-day
hold wins on every practical dimension (higher return, much better
drawdown, better win rate, half the trades) but the margin on absolute
return alone is thin. The honest mechanism, updated from the prior
(analogy-based) framing: it's not that the momentum-rank signal is noise
— the random-switching control shows it carries real information. It's
that this basket's edge is a medium-term drift/rotation effect (consistent
with `EQUITY_EXIT_REGIME_SIMPLE_TREND.md`'s own "medium-term trend/beta
exposure, not a precisely-timed edge" framing and the hold-duration
sweep's monotonic-with-duration finding) — chasing every rank flip trades
away calm patience for turnover and drawdown without a compensating gain
large enough to justify it.

**Building the daily leaderboard itself was worthwhile independent of this
result** — `outputs/continuous_momentum_switching_test/daily_momentum_leaderboard.csv`
is reusable infrastructure (e.g., for monitoring/dashboarding "who's
currently strongest" without acting on it), separate from the
switching-strategy question this doc answers.

Script: `experiments/run_continuous_momentum_switching_test.py`. Outputs:
`outputs/continuous_momentum_switching_test/` (leaderboard CSV, trade CSV,
summary.json).
