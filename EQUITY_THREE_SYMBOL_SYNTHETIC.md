# AAPL + TSLA + MSFT Synthetic Offshoot

This is a separate research branch from `selective_long_5m_v1`. It does not alter the BTC champion.

## Design

- Instruments: AAPL, TSLA, MSFT
- Synthetic 30-minute regular-session candles
- 13 bars per trading session
- Explicit overnight gaps
- Shared market factor plus symbol-specific beta and volatility
- Cross-sectional model ranks simultaneous candidates
- At most one selected position at each signal timestamp
- Maximum hold: 26 bars (approximately two regular sessions)
- Volatility-aware ATR stop, target, breakeven and trailing management
- Continuous candlestick geometry only; named pattern flags are excluded

## Run

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-v1 \
  experiments/run_equity_three_symbol_experiment.py \
  --sessions 1300 \
  --output outputs/equity_three_symbol_synthetic
```

The experiment compares baseline features against baseline plus continuous geometry, and reports AAPL, TSLA, MSFT and equal-weight buy-and-hold benchmarks over the same final chronological test period.
