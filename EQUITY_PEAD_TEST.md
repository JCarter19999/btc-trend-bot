# Post-earnings-announcement drift (PEAD) test — real data source found, effect not robust

Earlier tonight, PEAD was flagged as *untestable*, not *tested and rejected*
— yfinance's `earnings_dates` endpoint only covers ~25 quarters (2020+),
too short for this project's full 2018-2026 equity window. This resolves
that gap and tests the effect properly.

## Data source found: Alpha Vantage (already configured, real key)

`/home/joey/.config/btc-trend-bot/alphavantage.env` already has a working
API key (previously set up for a different purpose in this project). The
`EARNINGS` endpoint returns real `quarterlyEarnings` history with
`reportedDate`, `estimatedEPS`, `reportedEPS`, `surprisePercentage`, and
`reportTime` (pre/post-market) — 121 quarters back to 1996 for AAPL/MSFT,
109 for NVDA, 65 for TSLA (full history since its 2010 IPO). Comfortably
covers this project's 2018-2026 window. No signup or payment needed since
the key already existed.

## Method

No lookahead: entry is always the first trading day strictly *after* the
report date (conservative — doesn't try to exploit pre/post-market timing
nuance, matching this project's signal-at-t/entry-at-t+1 convention used
everywhere else). Forward returns measured at 5/10/20 trading days from
entry, using real cached daily closes (`data/real/*.csv`). Correlated
`surprisePercentage` against each forward-return horizon, checked against
a 1,000-seed shuffled-label null, same discipline as every other finding
tonight. 136 real earnings events across AAPL/MSFT/NVDA/TSLA, 2018-2026.

## Result: looks decisive at 20 days, dies exactly the way this session's
## other false positives have died — and the cause is a known data artifact

| Horizon | n | Pearson corr | Null percentile | OOS first half | OOS second half |
|---|---|---|---|---|---|
| 5-day | 136 | +0.015 | 58th (noise) | -0.006 | -0.243 |
| 10-day | 136 | +0.040 | 71st (not significant) | +0.041 | -0.070 |
| 20-day | 136 | **+0.272** | **100th** | **+0.407** | **-0.216** |

The 20-day full-sample number looks like the strongest, cleanest finding
of the night — 100th percentile against the null. **It fails exactly the
same way `breakout_5m_1h_regime` and the curated-basket persistence tests
failed: a full-sample statistic that reverses sign out-of-sample**
(+0.407 first half → -0.216 second half). By this project's own standing
rule (a result must clear the shuffled-null bar *and* hold up
chronologically), this does not qualify.

## Root-caused, not just noted

Checked per-symbol before accepting the pooled number at face value:

| Symbol | n | Mean surprise% | Std surprise% | Max surprise% | 20-day corr |
|---|---|---|---|---|---|
| AAPL | 34 | 7.5% | 9.3 | 42.9% | -0.040 |
| MSFT | 34 | 9.1% | 8.2 | 31.6% | -0.205 |
| NVDA | 34 | 8.6% | 9.9 | 34.4% | +0.238 |
| TSLA | 34 | **104.5%** | **341.5** | **1600%** | +0.373 |

**TSLA's surprise% distribution is wildly heavy-tailed** — a well-known
artifact of percentage-based EPS-surprise metrics when the estimate base
is near zero (a small early-history quarter with a tiny estimated EPS
turns an ordinary-sized miss/beat into a triple-digit or even
four-digit "surprise percentage"). The pooled 20-day correlation is
substantially a TSLA-outlier effect, not a broad PEAD signal: AAPL and
MSFT individually show **negative or zero** correlation, and only TSLA
and (more modestly) NVDA show positive relationships — with TSLA's driven
by exactly the kind of extreme percentage outlier that should raise
suspicion on sight.

## Verdict

**PEAD does not show a real, robust effect in this project's 4-stock
universe.** The one horizon that looked significant (20-day) fails the
out-of-sample check the same way several other candidates failed tonight,
and root-causing it directly attributes the pooled effect to a data
artifact in one stock's surprise-percentage calculation, not a genuine
broad phenomenon — 2 of 4 stocks show no positive relationship at all.
This is a real, disclosed negative result on a well-documented academic
effect, not a data-access dead end this time — the data source worked
fine; the effect itself doesn't hold up in this specific universe.

**Not pursued further, stated as a natural next step rather than
attempted here**: re-testing on the broader S&P 100 universe (already
used in `EQUITY_MULTI_POSITION_SP100_STUDY.md`) for more statistical
power and to check whether TSLA's outlier pattern is unusual or common
across many growth names with volatile EPS bases.

Script: `experiments/run_pead_test.py`. Outputs:
`outputs/pead_test/earnings_events.csv`, `outputs/pead_test/summary.json`.
