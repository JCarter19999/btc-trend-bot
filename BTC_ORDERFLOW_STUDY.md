# BTC Order-Flow Study — Tier 1 ("engines besides candlesticks")

Research-branch experiment. Not connected to the frozen BTC production
strategy or the equity work — genuinely new territory, real data.

## Motivation

Candlesticks are OHLC aggregates of trades — they discard *how* price moved
(who was crossing the spread, how aggressively) in favor of just where it
ended up. The equity-side research tonight found essentially no exploitable
signal at short horizons using candle-derived features (1-day-hold
expectancy ≈ 0 bps; see `EQUITY_EXIT_REGIME_SIMPLE_TREND.md`), and the
existing BTC 5-minute strategy matrix
(`/home/joey/btc-paper-5m/outputs/intraday_matrix_50000`) found the same
thing on BTC candles — most candidates had negative *gross* edge before any
costs. Order-flow data (trade tape, funding rate, open interest) is a
genuinely different information source, not just a faster clock, and was
scoped as "Tier 1" — free, real, historically available via Binance's public
API, no paid vendor or months-long collection required (unlike order-book
depth, which is live-only — see the option-space discussion this followed).

## What was built

- `src/btc_trend_bot/orderflow_data.py` — checkpointed, resumable historical
  downloaders against live Binance (via CCXT): `download_trades_range`
  (tick-level trade tape), `download_funding_rate_history`,
  `download_open_interest_history`.
- `src/btc_trend_bot/orderflow_features.py` — information-driven bars
  (`build_volume_bars`, `build_dollar_bars` — bars close on fixed traded
  volume/dollars, not fixed clock time) and `flow_imbalance` (signed
  buy-vs-sell volume within a bar) / `rolling_flow_imbalance`.
- `experiments/run_orderflow_signal_diagnostic.py` — correlation + shuffled-
  label-control diagnostic (same discipline as every equity finding
  tonight: don't trust a raw correlation, check it against a randomization
  null before believing it).

## Empirically confirmed data limits (measured live, not assumed)

- **Trade tape**: ~541,000 trades/day for BTC/USDT spot. Downloads at
  roughly 3 minutes of wall-clock time per day of trade history (measured:
  53,505 trades / 3 hours in 18.4s). Available arbitrarily far back via
  pagination — the constraint is download time/storage, not data
  availability.
- **Funding rate** (BTC/USDT:USDT perpetual): at least 2 years of history
  available, trivial data volume (3 obs/day). No practical limit hit.
- **Open interest**: Binance's free history endpoint **hard-caps at ~29
  days back** — confirmed empirically (29 days succeeds, 30+ fails with
  `startTime is invalid`, not a silent truncation). This bounds any
  OI-based analysis to a recent rolling window regardless of how much tick
  or funding history is pulled.

## First finding: real signal, not economically tradeable

Diagnostic run on a 3-hour pilot pull (53,505 trades, 1,540 dollar bars,
median bar duration well under 1 second at this granularity):

| Rolling imbalance window | Pearson corr. vs. next-bar return | Percentile vs. 10-seed shuffled-label null |
|---|---|---|
| 1 bar | **+0.268** | 100th |
| 5 bars | +0.148 | 100th |
| 20 bars | +0.083 | 100th |
| 50 bars | +0.045 | 100th |

This is the strongest, cleanest statistical signal found anywhere tonight —
not close. The 1-bar correlation sits roughly 15 standard deviations outside
the shuffled-label null distribution (null mean +0.0015, std 0.018). Recent
aggressive buy/sell pressure really does predict the immediate next bar's
direction. This is a real, well-known microstructure effect (short-horizon
order-flow autocorrelation / brief impact persistence after a sweep), not a
fluke of this dataset.

**But it fails the economic test decisively.** Directly measured: going
with the sign of 1-bar imbalance and capturing the next bar's return yields
a **gross** mean edge of **0.211 bps per bar-trade** (51.4% win rate). The
realistic round-trip execution cost floor established earlier tonight
(spread + slippage, which don't disappear even at zero commission) is
12–16 bps. The edge is **roughly 76x too small** to clear that floor. Unlike
the equity trend signal, this doesn't scale up by holding longer — imbalance
predictive power *decays* with window length (0.268 → 0.045 from 1 to 50
bars), so there's no "just capture it over a longer horizon" escape hatch;
by construction this is a fast-decaying effect, not a persistent one.

There's also a structural reason this specific number is unreachable in
practice even before the cost comparison: median bar duration in this pilot
was under one second. Capturing a sub-second signal requires infrastructure
this project doesn't have and isn't going to build for a retail-scale
account — real participants competing for this edge are colocated with
sub-millisecond latency. This is the concrete, measured version of the
"latency reality check" flagged when order-book infrastructure was scoped
out earlier — now demonstrated with real numbers instead of a general
caveat.

## Status

A 14-day full tick-data pull is running in the background
(`data/orderflow/btcusdt_trades_14d.parquet`) to check whether this holds up
on a much larger, more statistically robust sample (the pilot is only 3
hours — real risk of it being idiosyncratic to that window) and whether
the picture changes at a coarser bar size where realistic execution might
actually be possible (e.g., bars sized for multi-minute rather than
sub-second granularity, trading off signal strength against reachability).
Funding rate (1 year) and open interest (29-day max) history are already
downloaded and not yet analyzed as features — natural next step once the
larger tick dataset is validated.

## Honest read

Tier 1 delivered exactly what it was supposed to: proof that a
fundamentally different data source (order flow, not price action) *can*
contain real, statistically overwhelming signal that pure candlestick
features never showed all night. The catch is that the signal lives at a
timescale where realistic transaction costs and retail execution latency
both work against capturing it. Whether a coarser-grained version of the
same effect (still order-flow-derived, but on bars sized for seconds-to-
minutes rather than sub-second) survives is the open, concrete next
question — not yet answered.
