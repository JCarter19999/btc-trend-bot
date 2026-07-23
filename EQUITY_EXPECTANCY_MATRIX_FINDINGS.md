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

## Implication for the live deployment

`equity_v2_4` (main branch) is currently running Phase 2 shadow paper
trading on this same Ridge model against real daily data via
`equity-paper-yfinance.timer`. It's paper-only — no live orders, no capital
at risk — so there's no urgent action required, but this result means the
model backing that deployment has not demonstrated the edge its original
promotion was based on, once the backtest that promotion relied on is
corrected. Worth a deliberate decision (not a unilateral one) on whether to
keep it running as a monitoring exercise, pause it, or treat it as itself
another data point pending a version of this expectancy-matrix check that
also covers the live ledger's realized trades.
