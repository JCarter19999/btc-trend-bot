# Five-minute paper lab v0.3 changes

## Added

- A research-only intraday strategy matrix using Binance.US five-minute candles.
- Required cash and buy-and-hold benchmarks.
- Prior two-candle fee-aware strategy retained as a control.
- Volatility-breakout candidate with one-hour and four-hour regime filters.
- Rolling-VWAP mean-reversion candidate.
- One-hour momentum strategy with immediate entry.
- Matched one-hour momentum strategy with five-minute pullback/recovery entry.
- Causal completed-candle resampling for one-hour and four-hour features.
- Next-candle-open execution to prevent same-bar lookahead.
- Transaction-level fee, spread, slippage, and turnover accounting.
- Closed-trade holding time, gross/net return, MAE, MFE, and win-rate analysis.
- Cost-sensitivity reruns at 3, 5, 8, 10, and 12 bps all-in per side.
- Five chronological stability slices.
- Net-equity and drawdown charts.
- Ten new tests; complete suite now passes 58 tests.

## Deliberately unchanged

- Four-hour production configuration and service.
- Existing Coinbase deployment path.
- Supabase schema.
- Five-minute persistent paper database.
- Five-minute systemd timer.
- Live-order code.

The new strategy families remain research candidates until the 10,000- and
50,000-bar validations are reviewed.
