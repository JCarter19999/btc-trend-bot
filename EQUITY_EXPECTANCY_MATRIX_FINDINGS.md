# Expectancy Matrix: Corrected Sizing + Single-Position Timing

Research-branch experiment. Supersedes the compounded-equity read of every
prior result in this repo (`REAL_DATA_BACKTEST.md`'s 689% figure, this
branch's own initial Kalman write-up, the reference-doc AMZN ablation) as the
basis for any go/no-go decision — those all shared the same two structural
distortions this experiment isolates and removes.

## What was wrong with every prior backtest number

1. **Overlapping positions treated as sequential.** `simulate_capital` (the
   base runner) walks selected trades in signal_time order and hands each
   one a fresh slice of "current equity," without checking whether the
   *previous* trade's exit_time has passed. Audited on the Kalman baseline
   run: **77.3% of trades start before the prior trade's exit_time**, a
   median 5 days early against a 10-day max hold. That means the backtest
   implicitly runs several positions at once, each drawing capital as if it
   were free — a different, much more optimistic strategy than the
   "long-only, one active position" model `CLAUDE.md` describes, and
   different from what the live paper-step engine
   (`run_equity_paper_step.py`) actually does — it tracks a real
   `open_position` singleton and only evaluates new entries when nothing is
   open. **The live paper deployment is not affected by this bug**; every
   historical backtest number quoted anywhere in this repo is.
2. **25%-of-compounding-equity sizing.** Even without the overlap bug,
   reinvesting a fraction of a compounding account into 1,000+ trades makes
   near-zero or even negative average edge look explosive purely from
   multiplicative path effects (this project's own `simple_trend` control
   — $2,500 → $11.9M — already demonstrated this; it just wasn't applied to
   the Ridge/Kalman numbers themselves).

`src/btc_trend_bot/portfolio_sim.py` fixes both: a candidate is skipped
outright if it would overlap the currently open position (matching what a
real single-position account could do), and sizing is decoupled from
compounding via three modes — fixed notional ($2,500/trade, no compounding,
the cleanest read on selector quality), fixed fractional (10% of equity),
and volatility-targeted (0.5% of equity risked, sized off the ATR stop
distance).

## Result: real-label Ridge ranks in the bottom decile of a random distribution

`experiments/run_equity_expectancy_matrix.py`. Same candidates, same folds,
same exits, same data as every other experiment in this repo — only the
portfolio simulator and sizing changed. Metric: **expectancy in basis
points per trade** (mean trade return × 10,000), the sizing-invariant
number this whole redesign exists to isolate.

| Selector | Fixed notional | Fixed fractional (10%) | Vol-targeted (0.5% risk) |
|---|---|---|---|
| **Ridge (real labels)** | **−19.6 bps** | **+8.6 bps** | **+8.6 bps** |
| Ridge (shuffled, 3 seeds) | −3.3 / +33.6 / +41.9 bps | +15.9 / +10.8 / +18.8 bps | +15.9 / +10.8 / +18.8 bps |
| simple_trend | +61.1 bps | +66.6 bps | +66.6 bps |
| Kalman-online | +65.3 bps | +40.6 bps | +40.6 bps |
| Random, 50 seeds — mean ± sd | +35.1 ± 37.6 bps | +38.8 ± 23.0 bps | +38.8 ± 23.0 bps |
| **Ridge percentile within random** | **2nd** | **10th** | **10th** |

Real-label Ridge is **negative-expectancy on a fixed-notional basis, and
sits at the 2nd percentile of a 50-seed random-selection distribution** —
worse than 98% of random seeds. Under fractional/vol-targeted sizing it's
barely positive (8.6 bps) and still only the 10th percentile. Every other
selector tested — including *shuffled-label* Ridge, which by construction
carries no real signal — outperforms real-label Ridge on this metric. Under
the previous (flawed) simulator, real-label Ridge looked like the best
performer; under the corrected one, it looks like the worst.

## Reading this

This is not "Ridge is mediocre." A model with genuinely no skill should
land near the *center* of the random distribution (matching a handful of
the shuffled-label seeds above, which do land in the 35–65 bps range,
consistent with noise around the pool's baseline expectancy). Ridge landing
in the bottom decile suggests the model isn't just failing to add
information — cross-sectionally, picking the single *highest-predicted*
candidate each day is anti-correlated with realized forward return in this
setup. A plausible mechanism: Ridge's highest-conviction picks skew toward
candidates with the most extreme recent momentum/feature values (since
those produce the largest dot products against a linear model's weights),
and the most extreme values in this feature set are disproportionately
followed by reversion rather than continuation — the model may be
systematically selecting the walk-forward equivalent of "priced-in" moves.
This is a hypothesis, not yet tested directly (see next steps).

## Verdict

Confirms and sharpens the earlier "kill" call on both Ridge and the Kalman
candidate. Once compounding and the overlap bug are removed:

- **Ridge has no demonstrated positive expectancy** — its real-label result
  is statistically indistinguishable from, or worse than, chance.
- **Candidate generation + exit mechanics carry most of the pool's
  positive expectancy** (random/simple_trend/shuffled-Ridge all cluster in
  a similar positive-bps range) — consistent with the earlier raw-pool audit
  (unselected candidates: 48.5% win rate, +56.6 bps mean return).
- The bull-trending universe (NVDA +4,208%, TSLA +1,650%, AAPL +709%,
  MSFT +396% buy-and-hold over the same window, vs. SPY +217%) means "long
  almost anything in this basket, most of the time" was already a
  reasonable trade before any model was involved.

**Research status:** the infrastructure and the null controls both work as
intended — the original predictive-model hypothesis has failed. The open
question is now whether there's a simpler, transparent structural edge
underneath (candidate-generation rules, exit mechanics) independent of any
learned model, or whether the pool's positive expectancy is itself just this
specific 2018–2026 mega-cap tech bull run.

## Update: candidate-edge / exit-edge decomposition (100 seeds each)

`experiments/run_equity_candidate_exit_decomposition.py`. Isolates where the
pool's ~35-55 bps mean expectancy actually comes from.

### Candidate edge: does `build_candidates`'s mask matter?

| Pool | Mean expectancy | Win rate |
|---|---|---|
| Mask-filtered (current) | +50.3 bps | 47.5% |
| Unrestricted (any day/symbol, no mask) | +55.8 bps | 48.0% |

**No.** The mask (`relative_volume > 0.6`, `atr_pct > 0`,
`abs(ema_spread_atr) < 8`, `return_20 > -0.25`) is a liquidity/sanity filter,
not a selector — trading literally any day on any of the four symbols does
at least as well. It contributes nothing beyond restricting to
"tradeable-looking" bars.

### Exit edge: does the ATR stop/target mechanism matter?

Same (symbol, day) selections across all four rows — only the exit rule
changes:

| Exit rule | Mean expectancy | Win rate |
|---|---|---|
| Buy next open, sell next close (1-day) | **−0.6 bps** | 50.6% |
| Fixed 5-day hold, no stop/target | +56.1 bps | 55.5% |
| Fixed 10-day hold, no stop/target | **+132.8 bps** | 57.9% |
| Current ATR stop (1.35) / target (2.15) / max 10-day hold | +50.3 bps | 47.5% |

This is the sharpest finding in the whole exercise: **the pure 1-day return
has ~zero expectancy** (confirming there's no short-horizon directional
signal being captured at all, by anyone), and expectancy climbs steadily
with hold duration. But **holding a fixed 10 days with no stop-loss or
profit-taking beats the current ATR-managed exit by more than 2x** (132.8
vs. 50.3 bps) on the identical entries. The "risk management" layer — a
1.35 ATR stop that can cut a trade short in 46% of raw-pool cases (see the
exit_reason breakdown earlier in this doc) — is trimming more upside than
it protects on this specific universe/period, because these four symbols
were in a historically exceptional, low-mean-reversion uptrend.

### What this means

There is no evidence of alpha anywhere in this pipeline — not in candidate
selection (mask does nothing), not in symbol/day timing (Ridge underperforms
random), not in exit timing (ATR management underperforms just holding).
What positive expectancy exists is **pure medium-horizon directional drift
in a small basket of exceptional stocks during an exceptional period**
(AAPL/MSFT/NVDA/TSLA buy-and-hold 2018–2026: +709% / +396% / +4,208% /
+1,650%, vs. SPY +217%). The 1-day-hold result is the control that nails
this down: if there were a real short-horizon signal (from Ridge, Kalman, or
anything else), it would show up there, and it doesn't.

## Not yet done (scoped down for time; flagged, not skipped)

- Only 50 random seeds, not the 100+ a tighter confidence interval would
  want.
- Candidate-edge test not run: candidate-restricted random selection vs.
  random entries on *any* eligible date (isolates whether `build_candidates`'s
  mask is doing real work or whether any long entry in this universe would
  do).
- Exit-edge test not run: current ATR stop/target vs. fixed-N-day hold vs.
  buy-next-open/sell-next-close, on the identical candidate pool (isolates
  whether the ATR exit mechanics specifically matter).
- Liquidity/capital caps (max $ position, max % of ADV, min cash reserve)
  not implemented — vol-targeted sizing above is unconstrained by them.
- The "Ridge over-weights extreme feature values" mechanism above is a
  hypothesis inferred from the percentile result, not directly tested
  (e.g., by inspecting whether `predicted_return` correlates negatively
  with `realized net_return` in a rank-correlation sense).

## Implication for the live deployment — action taken

`equity_v2_4` (main branch) was running Phase 2 shadow paper trading on this
same Ridge model against real daily data via `equity-paper-yfinance.timer`.
Given full authorization from Joey (2026-07-23) to update the live
deployment if the research pointed that way: **there is no validated
candidate to promote in Ridge's place.** Kalman, simple_trend, and literal
random selection all land in the same statistical neighborhood as each
other (see the expectancy table above) — none of them is a demonstrated
improvement, they're just different random draws from a pool whose
positive expectancy comes entirely from medium-horizon drift in an
exceptional four-stock bull-market basket, not from any selection
mechanism. Promoting any of them in Ridge's place would repeat the exact
mistake this whole exercise was set up to catch.

The correct action is therefore not a swap but a **halt**: the live paper
deployment was stopped via its own `halt` control
(`experiments/run_equity_paper_step.py halt`, which persists a halted flag
in the ledger and is fully reversible via `resume` — no systemd units,
scheduling, or ledger history were touched). See `LIVE_DEPLOYMENT.md` /
`CLAUDE.md` in the main branch for the halt record and rationale. Resuming
it (or promoting anything new) should require a candidate that beats the
1-day-hold / 10-day-fixed-hold / random-selection controls established here
by a margin outside their own seed-to-seed variance — not just "looks
profitable."
