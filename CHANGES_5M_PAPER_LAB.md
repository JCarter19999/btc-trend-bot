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
