# Exit-Regime / simple_trend Strategy — Validation and Live Promotion

Research-branch experiment. **Promoted to live paper deployment on `main`
2026-07-23** (`config/settings_equity_paper_yfinance_simpletrend.yaml`),
replacing the halted Ridge deployment. This doc is the validation record
that promotion was based on.

## Origin

`EQUITY_EXPECTANCY_MATRIX_FINDINGS.md`'s candidate/exit decomposition found
two things: (1) selection method barely matters — Ridge, Kalman, random all
land in a similar range — and (2) exit mechanics matter a lot, but not the
way the live strategy assumed: a naive fixed 10-day hold with no stop/target
beat the ATR-managed exit by >2x on identical entries (+132.8 bps vs. +50.3
bps, on a mask-filtered random-selection pool). Joey's hypothesis: if the
exit regime (specifically, "get long and just hold ~10 days") is where the
real structure is, drop Ridge, disable the ATR stop/target, and see how a
transparent, non-ML selector performs.

## Method

Strategy config: symbols/universe/costs/safety unchanged from the Ridge-era
config; `stop_atr`/`target_atr` raised to 100 (effectively disabled, so
every trade runs the full `max_hold_bars=10` and exits via `time_exit`).
Selection: `simple_trend` (deterministic — picks the candidate with the
strongest `relative_strength_20` among symbols currently above their 50-day
benchmark trend; no model, no training window). Single-position-correct
timing and fixed-notional ($2,500/trade, no compounding) sizing throughout,
via `portfolio_sim.py`. Full 2018–2026 history used (not walk-forward's
`train_bars`-delayed folds, which don't apply to a selector that doesn't
train — see `_select_fold_winners`, which ignores `train` entirely for
`simple_trend`/`random`).

## Results

**Full sample:** 1,539 daily signals selected → 188 actually taken after
single-position overlap filtering (1,319 skipped as overlapping a still-open
position — 10-day holds are much "stickier" than the old ATR exits, which
often resolved early).

| Metric | Value |
|---|---|
| Trades taken | 188 |
| Win rate | 58.0% |
| Profit factor | 2.04 |
| Mean return/trade | 2.43% |
| Median return/trade | 1.15% |
| Expectancy | 243.3 bps/trade |
| Max drawdown | 22.1% |
| Total return (fixed notional) | +457% |

**Year-by-year** (mean return per trade taken):

| Year | Trades | Win rate | Mean return |
|---|---|---|---|
| 2018 | 13 | 46.2% | −1.4% |
| 2019 | 26 | 65.4% | +3.4% |
| 2020 | 25 | 72.0% | +6.1% |
| 2021 | 26 | 46.2% | +1.3% |
| 2022 | 13 | 46.2% | −0.9% |
| 2023 | 25 | 56.0% | +4.0% |
| 2024 | 27 | 66.7% | +3.0% |
| 2025 | 23 | 65.2% | +1.4% |
| 2026 (partial) | 10 | 30.0% | −0.1% |

Only two down years, both mild (−1.4%, −0.9%) — a sharp contrast with
Ridge's −25.9% in 2022 alone.

**Random-selection control** (same exit-regime config, 100 seeds, full
history): mean 132.8 bps, std 61.9, min −159.3, max 244.1. `simple_trend`'s
243.3 bps sits at the **99th percentile** — one seed out of 100 beat it.
(This matches the earlier decomposition's unrestricted-pool fixed-10-day-hold
figure of 132.8 bps almost exactly, a good consistency check across the two
experiments.)

**Bootstrap 95% CI** on mean per-trade return (moving-block, n=1,539
selected signals pre-overlap-filtering — a larger but less realistic
sample than the 188 single-position trades, included as a supplementary
check): mean 2.00%, CI [0.95%, 3.07%], probability positive = 1.0,
distinguishable from zero. Caveat: this CI is on the pre-single-position-
filter population, not the 188-trade realistic sample — directionally
consistent, not a substitute for it.

**Cost stress** (5 bps slippage each side vs. 2 bps default — 2.5x costs):
expectancy 237.2 bps, barely changed from baseline. Robust to costs.

**Hold-duration sensitivity** (5/10/15/20 days, single run each, NOT
re-tuned or used to pick a different deployment value):

| Hold (days) | Trades taken | Win rate | Expectancy |
|---|---|---|---|
| 5 | 371 | 55.0% | 106.6 bps |
| **10 (deployed)** | **188** | **58.0%** | **243.3 bps** |
| 15 | 122 | 54.9% | 272.3 bps |
| 20 | 94 | 66.0% | 452.7 bps |

**Read this carefully.** Expectancy increases *monotonically* with hold
duration, with no sign of a peak or reversal. That is not evidence that 10
days is special — it's evidence that this is capturing medium-term
directional drift, and holding longer captures more of it, converging
toward this basket's extraordinary buy-and-hold performance
(NVDA +4,208%, TSLA +1,650%, AAPL +709%, MSFT +396%, 2018–2026). **10 was
kept because it's the pre-existing project default, not because this sweep
picked it.** Deploying 20 days because it backtested better here would be
exactly the "optimize on the holdout" mistake this whole project exists to
avoid — the sweep was a sanity check on direction, not a hyperparameter
search.

## Honest characterization of what's being deployed

This is **not** a stock-picking or market-timing edge. It's best understood
as: rotate into whichever of four historically exceptional growth stocks
currently has the strongest relative momentum, and stay long for a fixed
medium-term window without actively managing risk intra-trade. The edge (to
the extent one exists going forward, which live paper trading is now
testing) is a bet that (a) trend/momentum-based symbol rotation among this
specific basket continues to identify the stronger performer better than
chance, and (b) not cutting positions early on short-term noise continues to
capture more of the underlying drift than it costs in unmanaged downside —
both empirically true in this historical window, neither guaranteed forward,
and both far more exposed to "this basket keeps outperforming" than any
version of the Ridge-based story was.

## Options overlay — deliberately deferred, not dropped

Joey's question: if a position is expected to be favorable for ~10 days, why
not express it with a 30-day call for convexity? Already tested (with a
*synthetic* option chain) in `EQUITY_HYBRID_OPTION_OPTIMIZER_RESULTS.md`:
stock-only had the best risk-adjusted result; every options expression
(overlay, dynamic stock-or-call, fractional-Kelly calls) added drawdown and
probability-of-ruin without improving mean return, and the optimizer
rejected ~78% of stock signals as lacking a positive-EV contract even under
synthetic (favorable) liquidity/spread assumptions. `CLAUDE.md`'s existing
guardrail forbids presenting a synthetic/Black-Scholes options backtest as
real historical performance, and it's a materially harder problem than the
stock version: a 30-day call bought to express a 10-day thesis has to be
*sold*, not exercised, at day 10, and that exit's P&L depends on how IV
moved in the meantime — something a fixed-vol model can't honestly capture.
**Next step, not abandoned:** source real historical option-chain data
(bid/ask, open interest, IV) for this universe — e.g. ORATS, CBOE DataShop,
Polygon.io — before attempting this again. Until then it stays out of the
live deployment.

## Status

Live on `main` as of 2026-07-23
(`config/settings_equity_paper_yfinance_simpletrend.yaml`,
`runtime/equity_yfinance_paper_simpletrend.sqlite3`, systemd timer
`equity-paper-yfinance-simpletrend.timer`, first fire 22:32 UTC same day).
Halted Ridge deployment's ledger and timer left in place, untouched, as a
documented negative result. Re-evaluate this deployment the same way Ridge
was re-evaluated — don't let "it's live and running" substitute for
periodically re-running it through the same null controls as real trade
history accumulates.
