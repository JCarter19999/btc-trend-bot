# BTC information-driven-bar trend study

**Question:** does sampling by dollar turnover (Lopez de Prado-style
"information-driven bars") instead of wall-clock candles let a plain
trend-continuation strategy — the exact "widen the candidate net, hold to a
fixed exit, don't manage it early" recipe that won for the deployed equity
strategy — beat `simple_trend`'s live performance (243.3 bps/trade over a
10-day hold = **24.33 bps/day**) when applied single-asset to BTC?

This is a distinct question from `BTC_ORDERFLOW_STUDY.md` (Tier 1), which
tested a sub-second flow-imbalance *predictive* signal and found it real but
~76-100x too small for costs. This test drops the order-flow-imbalance
signal entirely and just asks whether the *bar construction* itself helps a
standard trend filter.

**Script:** `experiments/run_btc_infobar_trend_study.py`. Data:
`data/orderflow/btcusdt_trades_14d.parquet` (8.2M trades, **11.4 days**
span, 2026-07-09 to 2026-07-20).

## Method

- Dollar bars at 3 thresholds ($5M/$10M/$20M per bar → 2299/1149/574 bars,
  median duration 5min/11min/24min).
- Trend filter: `close > EMA(close, w)` and `EMA` rising over a short lag,
  `w` in {20, 50, 100} bars.
- Entry: next bar's open, whenever the filter is true. Exit: fixed hold of
  `h` bars in {10, 20, 40, 80} — **no early stop/target**, matching the
  exact exit mechanics validated for `simple_trend` on equities.
- Cost: 12–16 bps round-trip (spread + slippage; Binance spot commission is
  genuinely zero, established in the Tier-1 study) subtracted directly from
  each trade's return.
- Single-position enforcement via `portfolio_sim.simulate_single_position`
  (same simulator as the equity studies — no overlapping-position artifact).
- Control: **random-timing**, not label-shuffling — same trade count and
  hold duration, but entries fire at uniformly random bars instead of only
  when the trend filter is true. 50 seeds, percentile-ranked, same standard
  used throughout this project.

36 (threshold × ema_window × hold) configs swept. Full grid:
`outputs/btc_infobar_trend_study/sweep_results.csv`.

## Result: no credible win, and a clean reason why

Context: BTC drifted **+4.4% (38.6 bps/day)** over this window on its own —
a strong tailwind, same trap flagged for the equity basket's bull-market
caveat.

Two clusters of outcomes, split cleanly by hold length:

1. **Short holds (10–40 bars) → real sample size, net negative.** At
   hold=10 with enough trades to mean anything (27–139 taken), expectancy
   is **-4.6 to -18.6 bps/trade** net of costs at every dollar-bar
   threshold — the 12–16bps round-trip cost floor dominates at these bar
   scales the same way it did in the Tier-1 study. Costs don't shrink; bar
   count needs to.
2. **Long holds (80 bars, ~1.6–1.7 days) → apparent edge, but n=5–13.**
   The best-looking configs ($20M bars, hold=80) show 40–53 bps/trade net,
   26–32 bps/day, 98th percentile vs. random-timing control — genuinely
   above the `simple_trend` benchmark on paper. But the single-position
   constraint caps these at **6 trades** in an 11.4-day window (hold ≈
   15% of the whole span), and those 6 trades are concentrated in a window
   that was independently rallying 38.6 bps/day. A win rate and percentile
   computed on 6 trades is not distinguishable from "happened to be long
   BTC during a bull run" — exactly the standard this project has already
   rejected results on (Ridge, Kalman-online).

Requiring `trades_taken >= 15` (a floor, not rigorous, but rules out the
n=6 cases) alongside beating 24.33 bps/day and clearing the 95th percentile
control: **0 of 36 configs qualify**, at either the 12bps or 16bps cost
assumption (checked both; direction doesn't change).

## Verdict

**Not yet — inconclusive rather than negative.** Unlike Tier 1 (order-flow
imbalance), this isn't a clean kill: the fundamental tension is that BTC's
12–16bps cost floor requires holding ~1.5+ days to amortize, but 11.4 days
of data structurally cannot produce a statistically meaningful trade count
at that hold length under a single-position constraint. This is a data-span
problem, not (yet) evidence the hypothesis is wrong.

**Next step, not yet taken:** extend the tick-data pull well beyond 14 days
(60–90+ days) so hold≈80-bar configs can accumulate 20+ non-overlapping
trades across more than one short window's drift regime. That download is
meaningfully bigger than the 14-day pull (which took real wall-clock time
already) — flagging before running it rather than kicking it off silently.
