# Five-minute paper-lab change set

## New research capability

- Native Coinbase/CCXT five-minute candle ingestion.
- Directional candle-run features, run length, cumulative run return, relative
  volume, body-to-range ratio, and a broader EMA regime feature.
- Conditional next-candle continuation statistics for streak lengths one through six.
- Causal historical simulation: signal on completed candle, fill at next open.

## New paper strategies

- Cash benchmark.
- BTC buy-and-hold benchmark.
- One-candle direction-following stress test.
- Two-candle run confirmation.
- Fee-aware two-candle run strategy.

## Execution-cost model

- Configurable fee per side.
- Configurable slippage per side.
- Historical spread assumption.
- Live public bid/ask spread.
- Net and frictionless gross shadow portfolios.
- Fees, spread, slippage, turnover, and total cost drag tracked independently.

## Persistence and telemetry

- Independent SQLite accounts, snapshots, and simulated trades.
- Idempotency key per strategy/candle.
- Catch-up processing after missed runs.
- Fail-closed behavior when downtime exceeds available candle lookback.
- Durable Supabase JSONL outbox.
- Supabase sampling every 15 minutes plus every simulated trade.

## Deployment

- Separate Dockerfile and image.
- Separate Compose service.
- Separate systemd service and five-minute timer.
- Separate Git branch/worktree instructions.
- Separate Streamlit multipage dashboard.
- No changes to the existing four-hour production service.

## 0.2.0

- Preserved the original high-turnover strategies in `config/settings_paper_5m_stress.yaml`.
- Replaced the default deployment candidates with three slower `candle_swing` variants.
- Added 1-hour momentum and 1-hour/6-hour EMA-regime features.
- Added exit hysteresis requiring two or three independent deterioration confirmations.
- Added `conditional_horizon_returns.csv` to test whether candle runs persist over 15 minutes through 24 hours.
- Added per-strategy gross break-even cost in basis points and actual-cost-to-break-even ratio.
- Updated Supabase and Streamlit fields for regime spread and momentum.
- Added local SQLite migration support for existing v0.1 paper databases.
